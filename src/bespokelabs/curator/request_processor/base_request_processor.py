import json
import logging
import os
import glob

from abc import ABC, abstractmethod
from typing import Optional

from datasets import Dataset
from bespokelabs.curator.prompter.prompt_formatter import PromptFormatter
from bespokelabs.curator.request_processor.generic_request import GenericRequest
from bespokelabs.curator.request_processor.generic_response import GenericResponse
from datasets.arrow_writer import ArrowWriter, SchemaInferenceError
from pydantic import BaseModel
from math import ceil
import asyncio
import aiofiles


class BaseRequestProcessor(ABC):
    """
    Base class for all request processors.
    """

    def __init__(self, batch_size: int = 1000):
        self.batch_size = batch_size

    @abstractmethod
    def get_rate_limits(self) -> dict:
        """
        Returns the rate limits for the API.

        Returns:
            dict: A dictionary containing the rate limit information.
        """
        pass

    @abstractmethod
    def create_api_specific_request(self, generic_request: GenericRequest) -> dict:
        """
        Creates a API-specific request body from a GenericRequest.

        Must use request_body.metadata.dataset_row or request_body.metadata.dataset_row_idx

        Note that the generic request body has both the original dataset row or the index of the row in the original dataset.
        What you pass on depends on what is works with the API. For example, OpenAI and Anthropic offline batch API allows you to pass a custom_id (can be index).
        For online, the api_parallel_processor can store the original dataset row in the metadata.

        Returns:
            dict: API specific request body
        """
        pass

    @abstractmethod
    def get_generic_response(self, response: dict) -> GenericResponse:
        """
        Parses a API-specific response into a generic response body.
        Does error handling on the response.

        IMPORTANT: In the generic response body you need to provide either the original dataset row OR the index of the row in the original dataset.
        Must return request_body.metadata.dataset_row or request_body.metadata.dataset_row_idx

        Args:
            response (dict): API-specific response

        Returns:
            dict: Generic response body with an extra field "metadata" which contains the original dataset row or the index of the row in the original dataset
        """
        pass

    @abstractmethod
    def run(
        self, dataset: Dataset, working_dir: str, prompt_formatter: PromptFormatter
    ) -> Dataset:
        """
        Uses the API to completing the specific map by calling the LLM.

        Args:
            dataset (Dataset): Dataset that is being mapped over
            working_dir (str): Working directory to save files (requests.jsonl, responses.jsonl, dataset.arrow)

        Returns:
            Dataset: Completed dataset
        """
        pass

    def create_request_files(
        self,
        dataset: Optional[Dataset],
        working_dir: str,
        prompt_formatter: PromptFormatter,
        batch_size: Optional[int] = None,
    ) -> list[str]:
        """
        Creates a request file if they don't already exist or use existing.

        Args:
            dataset (Dataset): The dataset to be processed.
            working_dir (str): The directory where request files will be saved.

        Returns:
            list[str]: Paths to the request files that were created.
        """
        os.makedirs(working_dir, exist_ok=True)
        requests_files = glob.glob(f"{working_dir}/requests_*.jsonl")

        # By default use existing requests in working_dir
        if len(requests_files) > 0:
            logging.info(
                f"Using existing requests in {working_dir} by default. Found {len(requests_files)} request files."
                f"If this is not what you want, delete the directory or specify a new one and re-run."
            )
            # count existing jobs in file and print first job
            with open(requests_files[0], "r") as f:
                # Count lines and store first job
                first_job = None
                num_jobs = 0
                for i, line in enumerate(f):
                    if i == 0:
                        first_job = json.loads(line)
                    num_jobs = i + 1

                if num_jobs > 0:
                    logging.info(
                        f"There are {num_jobs} existing requests in {requests_files[0]}"
                    )
                    logging.info(
                        f"Example request in {requests_files[0]}:\n{json.dumps(first_job, indent=2)}"
                    )

                    # Some simple sanity checks for the user
                    if batch_size is not None:
                        if batch_size != num_jobs:
                            logging.warning(
                                f"Batch size is {batch_size}, but there are {num_jobs} requests in {requests_files[0]}. "
                                f"If you want to run with new batch size, you will have to delete the working directory and re-run (looses progress)"
                            )
                        if len(requests_files) == 1 and len(dataset) > batch_size:
                            logging.warning(
                                f"Only one request file was found, but batch size is specified and dataset is larger than batch size."
                                f"You might be resuming from a different dataset or weren't using batching before."
                                f"If you want to run with batching, you will have to delete working directory and re-run (looses progress)"
                            )
                    return requests_files

        request_count = 0
        request_file_idx = 0
        requests_file = f"{working_dir}/requests_{request_file_idx}.jsonl"
        requests_files = []

        # Create new requests file
        with open(requests_file, "w") as f:
            if dataset is None:
                request = prompt_formatter.get_generic_request(dict(), 0)
                api_request = self.create_api_specific_request(request)
                f.write(json.dumps(api_request) + "\n")
            else:
                if batch_size:
                    num_batches = ceil(len(dataset) / batch_size)
                    requests_files = [
                        f"{working_dir}/requests_{i}.jsonl" for i in range(num_batches)
                    ]

                    async def create_all_request_files():
                        tasks = [
                            self.acreate_request_file(
                                dataset,
                                prompt_formatter,
                                requests_files[i],
                                i * batch_size,
                                batch_size,
                            )
                            for i in range(num_batches)
                        ]
                        await asyncio.gather(*tasks)

                    asyncio.run(create_all_request_files())
                else:
                    requests_files = [f"{working_dir}/requests_0.jsonl"]
                    asyncio.run(
                        self.acreate_request_file(
                            dataset, prompt_formatter, requests_files[0]
                        )
                    )

        if request_count > 0:
            logging.info(f"Wrote {request_count:,} requests to {requests_file}.")

        return requests_files

    # NOTE(Ryan): Instead of doing this, just iterate over iterable and keep counter and change filename when hit batch_size, this will be slower but this whole thing is dominated by llm calls anyways
    async def acreate_request_file(
        self,
        dataset: Dataset,
        prompt_formatter: PromptFormatter,
        request_file: str,
        start_idx: int = 0,
        batch_size: int = None,
    ) -> str:
        if batch_size is not None:
            end_idx = min(start_idx + batch_size, len(dataset))
            dataset = dataset.select(range(start_idx, end_idx))

        # NOTE(Ryan): For loops only for IterableDataset which allows for _very_ large datasets, when start_idx and batch_size are not specified
        async with aiofiles.open(request_file, "w") as f:
            for idx, dataset_row in enumerate(dataset):
                dataset_row_idx = idx + start_idx
                # Get the generic request from the map function
                request = prompt_formatter.get_generic_request(
                    dataset_row, dataset_row_idx
                )
                # Convert the generic request to an API-specific request
                api_request = self.create_api_specific_request(request)
                # Write the API-specific request to file
                await f.write(json.dumps(api_request) + "\n")
        logging.info(f"Requests file {request_file} written to disk.")

    def create_dataset_files(
        self,
        dataset: Dataset,
        working_dir: str,
        prompt_formatter: PromptFormatter,
    ) -> None:
        """
        Creates the request files if they don't already exist or use existing.
        A single request file (requests_0.jsonl) or multiple request files
        (requests_0.jsonl, requests_1.jsonl, etc.) are created depending on
        batch_size.

        Args:
            dataset (Dataset): The dataset to be processed.
            working_dir (str): The directory where request files will be saved.
            prompt_formatter (PromptFormatter): The prompt formatter to use for parsing the responses.

        Returns:
            Dataset: Completed dataset
        """
        total_responses_count = 0
        failed_responses_count = 0

        responses_files = glob.glob(f"{working_dir}/responses_*.jsonl")
        if len(responses_files) == 0:
            raise ValueError(f"No responses files found in {working_dir}")
        dataset_file = f"{working_dir}/dataset.arrow"

        # Process all response files
        with ArrowWriter(path=dataset_file) as writer:
            for responses_file in responses_files:
                with open(responses_file, "r") as f_in:
                    for generic_response_string in f_in:
                        total_responses_count += 1
                        response = GenericResponse.model_validate_json(
                            generic_response_string
                        )
                        if prompt_formatter.response_format:
                            response.response = prompt_formatter.response_format(
                                **response.response
                            )

                        if response is None:
                            failed_responses_count += 1
                            continue

                        # Requires dataset to be Dataset object with random access
                        if response.row is None:
                            response.row = dataset[response.row_idx]

                        # parse_func can return a single row or a list of rows
                        if prompt_formatter.parse_func:
                            dataset_rows = prompt_formatter.parse_func(
                                response.row, response.response
                            )
                            if not isinstance(dataset_rows, list):
                                dataset_rows = [dataset_rows]
                        else:
                            dataset_rows = [response.response]

                        for row in dataset_rows:
                            if isinstance(row, BaseModel):
                                row = row.model_dump()
                            writer.write(row)

            logging.info(
                f"Read {total_responses_count} responses, {failed_responses_count} failed"
            )
            if failed_responses_count == total_responses_count:
                raise ValueError("All requests failed")

            logging.info("Finalizing writer")
            try:
                writer.finalize()
            except SchemaInferenceError as e:
                raise ValueError(
                    "Arrow writer is complaining about the schema: likely all of your parsed rows were None and writer.write only wrote None objects."
                ) from e

        return Dataset.from_file(dataset_file)
