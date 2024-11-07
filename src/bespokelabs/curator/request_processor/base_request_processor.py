import json
import logging
import os
from abc import ABC, abstractmethod
from typing import Dict, List, Optional

import tqdm
from tqdm import tqdm

from bespokelabs.curator.dataset import Dataset
from bespokelabs.curator.prompter.prompt_formatter import PromptFormatter
from bespokelabs.curator.request_processor.generic_request import GenericRequest
from bespokelabs.curator.request_processor.generic_response import GenericResponse


class BaseRequestProcessor(ABC):
    """
    Base class for all request processors.
    """

    def __init__(self, prompt_formatter: PromptFormatter):
        self.prompt_formatter = prompt_formatter

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
    def run(self, dataset: Dataset, working_dir: str) -> Dataset:
        """
        Uses the API to completing the specific map by calling the LLM.

        Args:
            dataset (Dataset): Dataset that is being mapped over
            working_dir (str): Working directory to save files (requests.jsonl, responses.jsonl, dataset.arrow)

        Returns:
            Dataset: Completed dataset
        """
        pass

    def create_request_files(self, dataset: Optional[Dataset], working_dir: str) -> str:
        """
        Creates a request file if they don't already exist or use existing.

        Args:
            dataset (Dataset): The dataset to be processed.
            working_dir (str): The directory where request files will be saved.

        Returns:
            str: Path to the request file that was created.
        """
        os.makedirs(working_dir, exist_ok=True)
        requests_file = f"{working_dir}/requests.jsonl"

        # By default use existing requests in working_dir
        if os.path.exists(requests_file):
            logging.info(
                f"Using existing requests in {working_dir} by default. "
                f"If this is not what you want, delete the directory or specify a new one and re-run."
            )
            # count existing jobs in file and print first job
            with open(requests_file, "r") as f:
                # Count lines and store first job
                first_job = None
                num_jobs = 0
                for i, line in enumerate(f):
                    if i == 0:
                        first_job = json.loads(line)
                    num_jobs = i + 1

                if num_jobs > 0:
                    logging.info(
                        f"There are {num_jobs} existing requests in {requests_file}"
                    )
                    logging.info(f"Example request:\n{json.dumps(first_job, indent=2)}")
                    return requests_file
                else:
                    logging.warning(
                        f"No requests found in {requests_file}. Will delete the file and start over."
                    )
                    os.remove(requests_file)

        # Create new requests file
        with open(requests_file, "w") as f:
            if dataset is None:
                request = self.prompt_formatter.get_generic_request(dict(), 0)
                api_request = self.create_api_specific_request(request)
                f.write(json.dumps(api_request) + "\n")
            else:
                for dataset_row_idx, dataset_row in enumerate(dataset):
                    request = self.prompt_formatter.get_generic_request(
                        dataset_row, dataset_row_idx
                    )
                    # Convert the generic request to an API-specific request
                    api_request = self.create_api_specific_request(request)
                    # Write the API-specific request to file
                    f.write(json.dumps(api_request) + "\n")
        logging.info(f"Requests file {requests_file} written to disk.")
        return requests_file
