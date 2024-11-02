"""Bella: Bespoke Labs Synthetic Data Generation Library."""

import asyncio
import json
import logging
import os
from typing import Optional

from api_request_parallel_processor import process_api_requests_from_file
from datasets import Dataset
from datasets.arrow_writer import ArrowWriter
from prompt import Prompter


def _create_requests_file(
    dataset, requests_file: str, prompter: Prompter, resume: bool = True
):
    if os.path.exists(requests_file):
        if resume:
            logging.info(f"Loading existing jobs from {requests_file}")
            logging.debug(
                f"Alternatively, delete the jobs file and re-run the annotator: `rm -rf {requests_file}`"
            )
            # count existing jobs in file and print first job
            with open(requests_file, "r") as f:
                num_jobs = sum(1 for _ in f)
                f.seek(0)
                first_job = json.loads(f.readline())
            logging.debug(f"Found {num_jobs} jobs in {requests_file}")
            logging.debug("Example job:")
            logging.debug(json.dumps(first_job, indent=2))
        else:
            error_message = (
                f"Existing job file {requests_file}. "
                f"Delete the jobs file and re-run the annotator: `rm -rf {requests_file}`. "
                f"Or run the annotator with the --resume flag to continue from the previous run."
            )
            raise ValueError(error_message)
    else:
        os.makedirs(os.path.dirname(requests_file), exist_ok=True)
        with open(requests_file, "w") as f:
            if len(dataset) == 0:
                request = prompter.get_request_object(dict(), 0)
                f.write(json.dumps(request) + "\n")
            else:
                for idx, sample in enumerate(dataset):
                    request = prompter.get_request_object(sample, idx)
                    f.write(json.dumps(request) + "\n")
        logging.info(f"Requests file {requests_file} written to disk.")


def _read_responses_file_and_write_to_dataset(
    prompter: Prompter, responses_file, dataset_file, output_column
):
    total_count = 0
    failed_count = 0
    with ArrowWriter(path=dataset_file) as writer:
        with open(responses_file, "r") as f_in:
            for line in f_in:
                total_count += 1
                try:
                    response = json.loads(line)
                    if isinstance(response[1], list):
                        # A failed requests contains a list of all the errors before max_retries
                        logging.info(
                            f"Request {response[2].get('request_idx')} failed due to errors: {response[1]}"
                        )
                        failed_count += 1
                        continue

                    metadata = response[2]
                    response_message = response[1]["choices"][0]["message"]["content"]

                    # Parse the response for structured output
                    if prompter.response_format:
                        response_parsed = prompter.response_format.model_validate_json(
                            response_message
                        )
                        response_parsed = response_parsed.model_dump()
                    else:
                        response_parsed = response_message

                    sample = metadata["sample"]

                    sample[output_column] = response_parsed
                    if sample == None:
                        raise ValueError("Trying to write a None sample")
                    writer.write(sample)

                except Exception as e:
                    logging.warning(f"Error: {e}")
                    logging.warning(f"Full response: {response}")
                    continue
        logging.debug(f"Read {total_count} responses, {failed_count} failed")
        if failed_count == total_count:
            raise ValueError("All requests failed")
        writer.finalize()


def completions(
    dataset,
    prompter: Prompter,
    output_column: str,
    name: Optional[str] = None,
    resume: bool = True,
) -> "Dataset":
    """
    Apply structured completions to the dataset using specified model and prompts.

    Args:
        prompter: A function that goes from row to request object
        output_column: Name of the column to store the response
        name: Name of the task
        resume: Whether to resume from the previous completions run

    Returns:
        A Dataset with the completions added in the output_column
    """
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    bella_cache_dir = os.environ.get(
        "BELLA_CACHE_DIR", os.path.expanduser("~/.cache/bella")
    )
    name = (
        f"{name.replace(' ', '-')}--{dataset._fingerprint}"
        if name
        else f"{dataset._fingerprint}"
    )
    requests_path = os.path.join(bella_cache_dir, f"{name}/requests.jsonl")
    responses_path = os.path.join(bella_cache_dir, f"{name}/responses.jsonl")
    dataset_path = os.path.join(bella_cache_dir, f"{name}/dataset.arrow")
    _create_requests_file(dataset, requests_path, prompter, resume)
    asyncio.run(
        process_api_requests_from_file(
            requests_filepath=requests_path,
            save_filepath=responses_path,
            request_url="https://api.openai.com/v1/chat/completions",
            max_attempts=5,
            resume=True,
            model=prompter.model_name,
        )
    )
    _read_responses_file_and_write_to_dataset(
        prompter, responses_path, dataset_path, output_column
    )
    return Dataset.from_file(dataset_path)
