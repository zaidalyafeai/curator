"""Bella: Bespoke Labs Synthetic Data Generation Library."""

import asyncio
import hashlib
import json
import logging
import math
import os
import time
from concurrent.futures import ProcessPoolExecutor
from typing import Iterable, Optional

from datasets import Dataset
from pydantic import BaseModel
from xxhash import xxh64

from api_request_parallel_processor import process_api_requests_from_file
from prompt import Prompter


def _create_requests_file(
    dataset: Iterable, requests_file: str, prompter: Prompter, resume: bool = True
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


def _parse_responses_file(prompter: Prompter, responses_file):
    total_count = 0
    failed_count = 0
    samples = []
    with open(responses_file, "r") as f_in:
        for line in f_in:
            total_count += 1
            try:
                # Each response is a tuple of (request, response, metadata) where:
                # - request is the original request object
                # - response is the response from the API
                # - metadata is a dictionary of metadata about the request (such as the request index)
                response = json.loads(line)
                if isinstance(response[1], list):
                    # A failed requests contains a list of all the errors before max_retries
                    logging.info(
                        f"Request {response[2].get('request_idx')} failed due to errors: {response[1]}"
                    )
                    failed_count += 1
                    continue

                response_message = response[1]["choices"][0]["message"]["content"]
                metadata = response[2]

                # Parse the response for structured output
                if prompter.response_format:
                    response_parsed = prompter.response_format.model_validate_json(
                        response_message
                    )
                else:
                    response_parsed = response_message

                samples.append((metadata["request_idx"], response_parsed))

            except Exception as e:
                logging.warning(f"Error: {e}. Full response: {response}")
                continue
    logging.debug(f"Read {total_count} responses, {failed_count} failed")
    # Sort by idx then return only the responses
    samples.sort(key=lambda x: x[0])
    samples = [sample[1] for sample in samples]
    if failed_count == total_count:
        raise ValueError("All requests failed")
    return samples


def _hash_chunk(chunks: list) -> list:
    """Hash a chunk of data."""

    def _json_dumps_row(row):
        if isinstance(row, BaseModel):
            row = row.model_dump()
        return json.dumps(row, sort_keys=True)

    chunks = [_json_dumps_row(row) for row in chunks]
    chunk_str = "|||".join(chunks)
    return xxh64(chunk_str).hexdigest()


def _hash_dataset(dataset: Iterable):
    """Hash a dataset to a consistent value using parallel processing."""
    start = time.perf_counter_ns()

    # Convert to list and determine chunking parameters
    dataset_list = list(dataset)
    if len(dataset_list) == 0:
        return xxh64("").hexdigest()

    num_cores = 4
    total_size = len(dataset_list)
    chunk_size = math.ceil(total_size / (num_cores * 4))  # 4 chunks per core

    chunks = [
        dataset_list[i : i + chunk_size] for i in range(0, total_size, chunk_size)
    ]

    # Process chunks in parallel
    with ProcessPoolExecutor(max_workers=num_cores) as executor:
        chunk_hash = list(executor.map(_hash_chunk, chunks))
        chunk_hash_str = "|||".join(chunk_hash)
        hash_value = xxh64(chunk_hash_str).hexdigest()

    logging.debug(
        f"Dataset hash time: {(time.perf_counter_ns() - start) / 1e6:.6f} milliseconds"
    )
    return hash_value


def completions(
    dataset: Iterable,
    prompter: Prompter,
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
    bella_cache_dir = os.environ.get(
        "BELLA_CACHE_DIR", os.path.expanduser("~/.cache/bella")
    )

    dataset_hash = _hash_dataset(dataset)
    # Convert all elements to strings and join them before hashing
    fingerprint_str = "_".join(
        [
            str(dataset_hash),
            str(prompter.user_prompt),
            str(prompter.system_prompt),
            str(prompter.model_name),
            str(prompter.response_format.schema_json()),
        ]
    )

    fingerprint = hashlib.md5(fingerprint_str.encode("utf-8")).hexdigest()

    name = f"{name.replace(' ', '-')}--{fingerprint}" if name else fingerprint
    requests_path = os.path.join(bella_cache_dir, f"{name}/requests.jsonl")
    responses_path = os.path.join(bella_cache_dir, f"{name}/responses.jsonl")

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
    return _parse_responses_file(prompter, responses_path)
