from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import datetime
import time
from typing import Optional
from tqdm import tqdm
import logging
import asyncio
import aiohttp
import os
import json
import resource

from bespokelabs.curator.dataset import Dataset
from bespokelabs.curator.request_processor.base_request_processor import BaseRequestProcessor
from bespokelabs.curator.prompter.prompter import PromptFormatter
from bespokelabs.curator.request_processor.generic_request import GenericRequest
from bespokelabs.curator.request_processor.event_loop import run_in_event_loop
from bespokelabs.curator.request_processor.generic_response import GenericResponse
import aiofiles

logger = logging.getLogger(__name__)


@dataclass
class StatusTracker:
    """Tracks the status of all requests."""

    num_tasks_started: int = 0
    num_tasks_in_progress: int = 0
    num_tasks_succeeded: int = 0
    num_tasks_failed: int = 0
    num_tasks_already_completed: int = 0
    num_api_errors: int = 0
    num_other_errors: int = 0
    available_request_capacity: float = 0
    available_token_capacity: float = 0
    last_update_time: float = field(default_factory=time.time)
    max_requests_per_minute: int = 0
    max_tokens_per_minute: int = 0
    pbar: tqdm = field(default=None)
    response_cost: float = 0

    def __str__(self):
        return (
            f"Tasks - Started: {self.num_tasks_started}, "
            f"In Progress: {self.num_tasks_in_progress}, "
            f"Succeeded: {self.num_tasks_succeeded}, "
            f"Failed: {self.num_tasks_failed}, "
            f"Already Completed: {self.num_tasks_already_completed}\n"
            f"Errors: {self.num_other_errors}"
        )

    def update_capacity(self):
        """Update available capacity based on time elapsed"""
        current_time = time.time()
        seconds_since_update = current_time - self.last_update_time

        self.available_request_capacity = min(
            self.available_request_capacity
            + self.max_requests_per_minute * seconds_since_update / 60.0,
            self.max_requests_per_minute,
        )

        self.available_token_capacity = min(
            self.available_token_capacity
            + self.max_tokens_per_minute * seconds_since_update / 60.0,
            self.max_tokens_per_minute,
        )

        self.last_update_time = current_time

    def has_capacity(self, token_estimate: int) -> bool:
        """Check if there's enough capacity for a request"""
        self.update_capacity()
        return (
            self.available_request_capacity >= 1 and self.available_token_capacity >= token_estimate
        )

    def consume_capacity(self, token_estimate: int):
        """Consume capacity for a request"""
        self.available_request_capacity -= 1
        self.available_token_capacity -= token_estimate


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


class OnlineRequestProcessor(BaseRequestProcessor, ABC):
    """Abstract base class for online request processors that make real-time API calls."""

    def __init__(
        self,
        model: str,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        presence_penalty: Optional[float] = None,
        frequency_penalty: Optional[float] = None,
    ):
        super().__init__(batch_size=None)
        self.model: str = model
        self.temperature: float | None = temperature
        self.top_p: float | None = top_p
        self.presence_penalty: float | None = presence_penalty
        self.frequency_penalty: float | None = frequency_penalty
        self.prompt_formatter: Optional[PromptFormatter] = None

    @abstractmethod
    def estimate_total_tokens(self, messages: list) -> int:
        """Estimate total tokens for a request"""
        pass

    @abstractmethod
    def estimate_output_tokens(self) -> int:
        """Estimate output tokens for a request"""
        pass

    @abstractmethod
    def process_single_request(
        self,
        request: APIRequest,
        session: aiohttp.ClientSession,
        retry_queue: asyncio.Queue,
        save_filepath: str,
        status_tracker: StatusTracker,
    ) -> None:
        """Process a single API request"""
        pass

    def check_structured_output_support(self, prompt_formatter: PromptFormatter) -> bool:
        """Check if the model supports structured output"""
        return True

    def run(
        self,
        dataset: Optional[Dataset],
        working_dir: str,
        parse_func_hash: str,
        prompt_formatter: PromptFormatter,
    ) -> Dataset:
        """Run completions using the online API with async processing."""
        logger.info(f"Running {self.__class__.__name__} completions with model: {self.model}")

        self.prompt_formatter = prompt_formatter
        if self.prompt_formatter.response_format:
            assert self.check_structured_output_support(
                prompt_formatter
            ), f"Model {self.model} does not support structured output, response_format: {self.prompt_formatter.response_format}"
        generic_requests_files = self.create_request_files(dataset, working_dir, prompt_formatter)
        generic_responses_files = [
            f"{working_dir}/responses_{i}.jsonl" for i in range(len(generic_requests_files))
        ]

        for request_file, response_file in zip(generic_requests_files, generic_responses_files):
            run_in_event_loop(
                self.process_requests_from_file(
                    generic_requests_filepath=request_file,
                    save_filepath=response_file,
                    max_attempts=5,
                    resume=True,
                )
            )

        return self.create_dataset_files(working_dir, parse_func_hash, prompt_formatter)

    async def process_requests_from_file(
        self,
        generic_requests_filepath: str,
        save_filepath: str,
        max_attempts: int,
        resume: bool,
    ) -> None:
        """Processes API requests in parallel, throttling to stay under rate limits."""

        # Initialize trackers
        queue_of_requests_to_retry: asyncio.Queue[APIRequest] = asyncio.Queue()
        status_tracker = StatusTracker()

        # Get rate limits
        rate_limits = self.get_rate_limits()
        status_tracker.max_requests_per_minute = rate_limits["max_requests_per_minute"]
        status_tracker.max_tokens_per_minute = rate_limits["max_tokens_per_minute"]
        rpm = rate_limits["max_requests_per_minute"]

        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        resource.setrlimit(
            resource.RLIMIT_NOFILE,
            (min(hard, int(10 * status_tracker.max_requests_per_minute)), hard),
        )

        # Track completed requests for resume functionality
        completed_request_ids = set()
        if os.path.exists(save_filepath) and resume:
            with open(save_filepath, "r") as f:
                for line in f:
                    response = GenericResponse.model_validate_json(line)
                    if not response.response_errors:
                        completed_request_ids.add(response.generic_request.original_row_idx)

        # Count total requests
        total_requests = sum(1 for _ in open(generic_requests_filepath))

        # Create progress bar
        status_tracker.pbar = tqdm(
            initial=len(completed_request_ids),
            total=total_requests,
            desc=f"Processing {self.__class__.__name__} requests",
        )

        # Use higher connector limit for better throughput
        connector = aiohttp.TCPConnector(limit=10 * rpm)
        async with aiohttp.ClientSession(connector=connector) as session: # Initialize ClientSession here
            async with aiofiles.open(generic_requests_filepath) as file:
                pending_requests = []

                async for line in file:
                    generic_request = GenericRequest.model_validate_json(line)

                    if resume and generic_request.original_row_idx in completed_request_ids:
                        status_tracker.num_tasks_already_completed += 1
                        continue

                    request = APIRequest(
                        task_id=status_tracker.num_tasks_started,
                        generic_request=generic_request,
                        api_specific_request=self.create_api_specific_request(generic_request),
                        attempts_left=max_attempts,
                        prompt_formatter=self.prompt_formatter,
                    )

                    token_estimate = self.estimate_total_tokens(request.generic_request.messages)

                    # Wait for capacity if needed
                    while not status_tracker.has_capacity(token_estimate):
                        await asyncio.sleep(0.1)

                    # Consume capacity before making request
                    status_tracker.consume_capacity(token_estimate)

                    task = asyncio.create_task(
                        self.process_single_request(
                            request=request,
                            session=session,
                            retry_queue=queue_of_requests_to_retry,
                            save_filepath=save_filepath,
                            status_tracker=status_tracker,
                        )
                    )
                    pending_requests.append(task)

                    status_tracker.num_tasks_started += 1
                    status_tracker.num_tasks_in_progress += 1

                    # Process any retries that are in the queue
                    while not queue_of_requests_to_retry.empty():
                        retry_request = await queue_of_requests_to_retry.get()
                        token_estimate = self.estimate_total_tokens(
                            retry_request.generic_request.messages
                        )

                        # Wait for capacity if needed
                        while not status_tracker.has_capacity(token_estimate):
                            await asyncio.sleep(0.1)

                        # Consume capacity before making request
                        status_tracker.consume_capacity(token_estimate)

                        task = asyncio.create_task(
                            self.process_single_request(
                                request=retry_request,
                                session=session,
                                retry_queue=queue_of_requests_to_retry,
                                save_filepath=save_filepath,
                                status_tracker=status_tracker,
                            )
                        )
                        pending_requests.append(task)

            # Wait for all tasks to complete
            if pending_requests:
                await asyncio.gather(*pending_requests)

        status_tracker.pbar.close()

        # Log final status
        logger.info(f"Processing complete. Results saved to {save_filepath}")
        logger.info(f"Status tracker: {status_tracker}")

        if status_tracker.num_tasks_failed > 0:
            logger.warning(
                f"{status_tracker.num_tasks_failed} / {status_tracker.num_tasks_started} "
                f"requests failed. Errors logged to {save_filepath}."
            )

    async def append_generic_response(self, data: GenericResponse, filename: str) -> None:
        """Append a response to a jsonl file with async file operations."""
        json_string = json.dumps(data.model_dump(), default=str)
        async with aiofiles.open(filename, "a") as f:
            await f.write(json_string + "\n")
        logger.debug(f"Successfully appended response to {filename}")
