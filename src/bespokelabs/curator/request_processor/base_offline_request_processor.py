from abc import ABC
from dataclasses import dataclass, field
import datetime
from typing import Optional
import logging
import os
import json

from bespokelabs.curator.dataset import Dataset
from bespokelabs.curator.request_processor.base_request_processor import BaseRequestProcessor
from bespokelabs.curator.llm.prompt_formatter import PromptFormatter
from bespokelabs.curator.request_processor.generic_request import GenericRequest
from bespokelabs.curator.request_processor.generic_response import GenericResponse

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


@dataclass
class StatusTracker:
    """Tracks the status of all requests."""

    total_requests: int = 0
    time_started: datetime.datetime = field(default_factory=datetime.datetime.now)
    time_finished: Optional[datetime.datetime] = None

    def __str__(self):
        return (
            f"Total requests: {self.total_requests}, "
            f"Time started: {self.time_started}, "
            f"Time finished: {self.time_finished}"
        )

    def __repr__(self):
        return str(self)


@dataclass
class APIRequest:
    """Stores an API request's inputs, outputs, and other metadata."""

    task_id: int
    generic_request: GenericRequest
    api_specific_request: dict
    attempts_left: int
    result: list = field(default_factory=list)
    prompt_formatter: PromptFormatter = field(default=None)
    created_at: datetime.datetime = field(default_factory=datetime.datetime.now)


class BaseOfflineRequestProcessor(BaseRequestProcessor, ABC):
    """
    Base class for offline request processors.

    Args:
        model (str): The model (path) to use
    """

    def __init__(self, model: str):
        super().__init__(batch_size=None, require_all_responses=True)
        self.model: str = model
        self.prompt_formatter: Optional[PromptFormatter] = None

    def load_offline_model(self):
        """Load the offline model."""
        pass

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

    def destroy(self) -> None:
        """Destroy the model."""
        pass

    def run(
        self,
        dataset: Optional[Dataset],
        working_dir: str,
        parse_func_hash: str,
        prompt_formatter: PromptFormatter,
    ) -> Dataset:
        # load from already completed dataset
        output_dataset = self.attempt_loading_cached_dataset(working_dir, parse_func_hash)
        if output_dataset is not None:
            return output_dataset
        logger.info(f"Running {self.__class__.__name__} completions with model: {self.model}")

        self.load_offline_model()

        self.prompt_formatter = prompt_formatter
        if self.prompt_formatter.response_format:
            if not self.check_structured_output_support():
                raise ValueError(
                    f"Model {self.model} does not support structured output, "
                    f"response_format: {self.prompt_formatter.response_format}"
                )
        generic_request_files = self.create_request_files(dataset, working_dir, prompt_formatter)
        generic_responses_files = [
            f"{working_dir}/responses_{i}.jsonl" for i in range(len(generic_request_files))
        ]

        for request_file, response_file in zip(generic_request_files, generic_responses_files):
            self.process_requests_from_file(
                generic_request_filepath=request_file,
                save_filepath=response_file,
                resume=True,
            )

        self.destroy()

        return self.create_dataset_files(working_dir, parse_func_hash, prompt_formatter)

    def process_requests(
        self, requests: list[APIRequest], status_tracker: StatusTracker
    ) -> list[GenericResponse]:
        pass

    def process_requests_from_file(
        self,
        generic_request_filepath: str,
        save_filepath: str,
        resume: bool,
        resume_no_retry: bool = False,
    ) -> None:

        status_tracker = StatusTracker()

        # Track completed requests for resume functionality
        completed_request_ids = set()
        if os.path.exists(save_filepath):
            if resume:
                logger.info(f"Resuming progress by reading existing file: {save_filepath}")
                logger.debug(
                    f"Removing all failed requests from {save_filepath} so they can be retried"
                )
                temp_filepath = f"{save_filepath}.temp"
                num_previously_failed_requests = 0

                with open(save_filepath, "r") as input_file, open(
                    temp_filepath, "w"
                ) as output_file:
                    for line in input_file:
                        response = GenericResponse.model_validate_json(line)
                        if response.response_errors:
                            logger.debug(
                                f"Request {response.generic_request.original_row_idx} previously failed due to errors: "
                                f"{response.response_errors}, removing from output and will retry"
                            )
                            num_previously_failed_requests += 1
                        if response.response_message is None:
                            logger.debug(
                                f"Request {response.generic_request.original_row_idx} previously failed due to no response, removing from output and will retry"
                            )
                            num_previously_failed_requests += 1
                        else:
                            completed_request_ids.add(response.generic_request.original_row_idx)
                            output_file.write(line)

                logger.info(
                    f"Found {len(completed_request_ids)} completed requests and "
                    f"{num_previously_failed_requests} previously failed requests"
                )
                logger.info("Failed requests and remaining requests will now be processed.")
                os.replace(temp_filepath, save_filepath)

            elif resume_no_retry:
                logger.warning(
                    f"Resuming progress from existing file: {save_filepath}, without retrying failed requests"
                )
                num_previously_failed_requests = 0

                with open(save_filepath, "r") as input_file:
                    for line in input_file:
                        response = GenericResponse.model_validate_json(line)
                        if response.response_errors:
                            logger.debug(
                                f"Request {response.generic_request.original_row_idx} previously failed due to errors: "
                                f"{response.response_errors}, will NOT retry"
                            )
                            num_previously_failed_requests += 1
                        completed_request_ids.add(response.generic_request.original_row_idx)

                logger.info(
                    f"Found {len(completed_request_ids)} total requests and "
                    f"{num_previously_failed_requests} previously failed requests"
                )
                logger.info("Remaining requests will now be processed.")

            else:
                user_input = input(
                    f"File {save_filepath} already exists.\n"
                    f"To resume if there are remaining requests without responses, run with --resume flag.\n"
                    f"Overwrite? (Y/n): "
                )
                if user_input.lower() not in ["y", ""]:
                    logger.info("Aborting operation.")
                    return

        # Count total requests
        total_requests = sum(1 for _ in open(generic_request_filepath))

        logger.info(f"Processing {total_requests} requests from {generic_request_filepath}")

        requests = []
        with open(generic_request_filepath, "r") as f:
            for line in f:
                request = GenericRequest.model_validate_json(line)
                if request.original_row_idx not in completed_request_ids:
                    requests.append(
                        APIRequest(
                            task_id=request.original_row_idx,
                            generic_request=request,
                            api_specific_request=self.create_api_specific_request(request),
                            attempts_left=0,
                            prompt_formatter=self.prompt_formatter,
                        )
                    )
        responses = self.process_requests(
            requests=requests,
            status_tracker=status_tracker,
        )

        # Save responses
        with open(save_filepath, "a") as f:
            for response in responses:
                json_string = json.dumps(response.model_dump(), default=str)
                f.write(json_string + "\n")

        # Log final status
        logger.info(f"Processing complete. Results saved to {save_filepath}")
        logger.info(f"Status tracker: {status_tracker}")
