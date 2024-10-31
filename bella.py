"""Bella: Bespoke Labs Synthetic Data Generation Library."""

import asyncio
from typing import Generic, List, Optional, TypeVar, Tuple

from prompt import Prompter

import pandas as pd
from datasets import Dataset as HFDataset
from IPython.display import HTML, display
from pydantic import BaseModel, ConfigDict, Field
from tqdm import tqdm
import logging
import json
import os
from datasets.arrow_writer import ArrowWriter
from api_request_parallel_processor import process_api_requests_from_file
import requests
import tiktoken

T = TypeVar("T", bound=BaseModel)


class ListModel(BaseModel, Generic[T]):
    """A list of items to be used as a response format."""

    model_config = ConfigDict(title="ListResponse")  # This sets a valid schema name.
    items: List[T] = Field(description="List of items")


# TODO: Where should this code actually live?
def get_rate_limits(
    model: str, url: str = "https://api.openai.com/v1/chat/completions"
) -> Tuple[int, int]:
    """
    Function to get rate limits for a given annotator. Makes a single request to openAI API
    and gets the rate limits from the response headers. These rate limits vary per model
    and are determined by your organization's usage tier. View the following:
    https://platform.openai.com/docs/guides/rate-limits/usage-tiers
    https://platform.openai.com/settings/organization/limits

    Args:
        annotator (str): The annotator for which to get the rate limits.

    Returns:
        Tuple[int, int]: The maximum number of requests and tokens per minute.
    """
    if "openai" in url:
        # Send a dummy request to get rate limit information
        response = requests.post(
            url,
            headers={"Authorization": f"Bearer {os.getenv('OPENAI_API_KEY')}"},
            json={"model": model, "messages": []},
        )

        # Extract rate limit information from headers
        max_requests = int(response.headers.get("x-ratelimit-limit-requests", 1500))
        max_tokens = int(response.headers.get("x-ratelimit-limit-tokens", 6250000))
    else:
        raise ValueError(f"Unknown API: {url}")

    return max_requests, max_tokens


class Dataset(HFDataset):
    """A wrapper around a HuggingFace Dataset with extra functionality for data generation."""

    initialized: bool = True
    _list_columns: List[str] = []

    # TODO: static method for completions instead of using empty().

    @classmethod
    def empty(cls) -> "Dataset":
        dataset = cls.from_list([])
        dataset.initialized = False
        dataset._list_columns = []
        return dataset

    def display(self):
        display(HTML(self.to_pandas().to_html()))

    def flatten(self) -> "Dataset":
        """Flatten any list columns in the dataset"""
        if not self._list_columns:
            return self

        flattened_rows = []
        for row in self:
            row_dict = dict(row)
            list_values = {
                col: row_dict[col] for col in self._list_columns if col in row_dict
            }

            if not any(isinstance(v, list) for v in list_values.values()):
                flattened_rows.append(row_dict)
                continue

            max_length = max(
                len(v) for v in list_values.values() if isinstance(v, list)
            )

            for i in range(max_length):
                new_row = {}
                for col, value in row_dict.items():
                    if col in list_values and isinstance(value, list):
                        new_row[col] = value[i] if i < len(value) else None
                    else:
                        new_row[col] = value
                flattened_rows.append(new_row)

        dataset = Dataset.from_pandas(pd.DataFrame(flattened_rows))
        dataset.initialized = True
        dataset._list_columns = []
        return dataset

    def flatten_objects(self) -> "Dataset":
        return super(Dataset, self).flatten()

    def create_requests_file(
        self, requests_file: str, prompter: Prompter, resume: bool = True
    ):
        if os.path.exists(requests_file):
            if resume:
                logging.info(f"Loading existing jobs from {requests_file}")
                logging.info(
                    f"Alternatively, delete the jobs file and re-run the annotator: `rm -rf {requests_file}`"
                )
                # count existing jobs in file and print first job
                with open(requests_file, "r") as f:
                    num_jobs = sum(1 for _ in f)
                    f.seek(0)
                    first_job = json.loads(f.readline())
                logging.info(f"Using existing jobs from {requests_file}")
                logging.info(f"Number of jobs: {num_jobs}")
                logging.info("Example job:")
                logging.info(json.dumps(first_job, indent=2))
            else:
                error_message = (
                    f"Existing job file {requests_file}. "
                    f"Delete the jobs file and re-run the annotator: `rm -rf {requests_file}`. "
                    f"Or run the annotator with the --resume flag to continue from the previous run."
                )
                raise ValueError(error_message)
        else:
            # NOTE: this creates one and only request per row, we are limiting ourselves to this for now
            os.makedirs(os.path.dirname(requests_file), exist_ok=True)
            with open(requests_file, "w") as f:
                if len(self) == 0:
                    request = prompter.get_request_object(dict(), 0)
                    f.write(json.dumps(request) + "\n")
                else:
                    for idx, sample in tqdm(
                        enumerate(self), desc="Writing requests to file"
                    ):
                        request = prompter.get_request_object(sample, idx)
                        f.write(json.dumps(request) + "\n")
            logging.info(f"Requests file {requests_file} written to disk.")

    def read_responses_file_and_write_to_dataset(
        self, prompter: Prompter, responses_file, dataset_file, output_column
    ):
        total_count = 0
        failed_count = 0
        with ArrowWriter(path=dataset_file) as writer:
            with open(responses_file, "r") as f_in:
                for line in tqdm(f_in, desc="Reading responses and writing to dataset"):
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
                        response_message = response[1]["choices"][0]["message"][
                            "content"
                        ]
                        response_parsed = prompter.response_format.model_validate_json(
                            response_message
                        )

                        if len(self) == 0:
                            sample = {}
                        else:
                            # Get the original sample
                            sample = self[metadata["request_idx"]]

                        # Add the parsed response to the sample
                        sample[output_column] = response_parsed.model_dump()

                        writer.write(sample)

                    except Exception as e:
                        logging.warning(f"Error: {e}")
                        logging.warning(f"Full response: {response}")
                        continue
            print(f"Read {total_count} responses, {failed_count} failed")
            print("Finalizing writer")
            if failed_count == total_count:
                raise ValueError("All requests failed")
            writer.finalize()

    def run_online_generation(
        self,
        jobs_file,
        responses_file,
        log_file,
        model,
        url="https://api.openai.com/v1/chat/completions",
    ):

        rpm, tpm = get_rate_limits(model, url)
        max_requests_per_minute = rpm
        print(f"Automatically set max_requests_per_minute to {rpm}")
        max_tokens_per_minute = tpm
        print(f"Automatically set max_tokens_per_minute to {tpm}")

        print(
            f"Online generation with parallel processing starting, logging to {log_file}"
        )
        if url.startswith("https://api.openai.com/"):
            api_key = os.getenv("OPENAI_API_KEY")
            token_encoding_name = tiktoken.encoding_for_model(model).name
        else:
            raise ValueError(f"Unknown API: {url}")

        asyncio.run(
            process_api_requests_from_file(
                requests_filepath=jobs_file,
                save_filepath=responses_file,
                request_url=url,
                api_key=api_key,
                max_requests_per_minute=max_requests_per_minute,
                max_tokens_per_minute=max_tokens_per_minute,
                token_encoding_name=token_encoding_name,
                max_attempts=5,
                resume=True,  # detects existing jobs and resume from there
                log_filepath=log_file,
            )
        )
        print(f"Parallel processing complete. Check {log_file} for details.")

    def completions(
        self,
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
            keep_columns: Whether to keep original columns in the output dataset
            verbose: Whether to show a progress bar
            name: Name of the task
            resume: Whether to resume from the previous completions run

        Returns:
            A new Dataset with the completions added
        """
        bella_cache_dir = os.environ.get(
            "BELLA_CACHE_DIR", os.path.expanduser("~/.cache/bella")
        )
        name = (
            f"{self._fingerprint}--{name.replace(' ', '-')}"
            if name
            else f"{self._fingerprint}"
        )
        requests_path = os.path.join(bella_cache_dir, f"{name}/requests.jsonl")
        responses_path = os.path.join(bella_cache_dir, f"{name}/responses.jsonl")
        dataset_path = os.path.join(bella_cache_dir, f"{name}/dataset.arrow")
        log_file = os.path.join(bella_cache_dir, f"{name}/completions.log")
        self.create_requests_file(requests_path, prompter, resume)
        self.run_online_generation(
            requests_path, responses_path, log_file, prompter.model_name
        )
        self.read_responses_file_and_write_to_dataset(
            prompter, responses_path, dataset_path, output_column
        )
        return Dataset.from_file(dataset_path)
