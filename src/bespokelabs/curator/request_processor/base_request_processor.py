import json
import logging
import os
import asyncio
import glob
import aiofiles

from abc import ABC, abstractmethod
from typing import Dict, List, Optional
from math import ceil

from bespokelabs.curator.dataset import Dataset
from bespokelabs.curator.prompter.prompt_formatter import PromptFormatter
from bespokelabs.curator.request_processor.generic_request import GenericRequest
from bespokelabs.curator.request_processor.generic_response import GenericResponse


class BaseRequestProcessor(ABC):
    """
    Base class for all request processors.
    """

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
    ) -> str:
        """
        Creates a request file if they don't already exist or use existing.

        Args:
            dataset (Dataset): The dataset to be processed.
            working_dir (str): The directory where request files will be saved.

        Returns:
            str: Path to the request file that was created.
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
        # Create new requests file
        with open(requests_file, "w") as f:
            if dataset is None:
                request = prompt_formatter.get_generic_request(dict(), 0)
                api_request = self.create_api_specific_request(request)
                f.write(json.dumps(api_request) + "\n")
            else:
                for dataset_row_idx, dataset_row in enumerate(dataset):
                    request = prompt_formatter.get_generic_request(
                        dataset_row, dataset_row_idx
                    )
                    # Convert the generic request to an API-specific request
                    api_request = self.create_api_specific_request(request)
                    # Write the API-specific request to file
                    f.write(json.dumps(api_request) + "\n")
                    request_count += 1

                    # Batches could be created in parallel, but dataset is iterated sequentially
                    if batch_size is not None and request_count == batch_size:
                        request_count = 0
                        request_file_idx += 1
                        requests_file = (
                            f"{working_dir}/requests_{request_file_idx}.jsonl"
                        )
                        logging.info(
                            f"Wrote {request_count:,} requests to {requests_file}."
                        )
        if request_count > 0:
            logging.info(f"Wrote {request_count:,} requests to {requests_file}.")
        return requests_file
