import json
import logging
import os
from typing import Optional
import asyncio
import time
from dataclasses import dataclass, field
import datetime
from tqdm import tqdm
import litellm

import instructor
from bespokelabs.curator.dataset import Dataset
from bespokelabs.curator.prompter.prompter import PromptFormatter
from bespokelabs.curator.request_processor.base_request_processor import (
    BaseRequestProcessor,
    GenericRequest,
    GenericResponse,
)
from bespokelabs.curator.request_processor.event_loop import run_in_event_loop
from bespokelabs.curator.request_processor.generic_response import TokenUsage

logger = logging.getLogger(__name__)

@dataclass
class StatusTracker:
    """Tracks the status of all requests."""
    num_tasks_started: int = 0
    num_tasks_in_progress: int = 0
    num_tasks_succeeded: int = 0
    num_tasks_failed: int = 0
    num_tasks_already_completed: int = 0
    num_rate_limit_errors: int = 0
    time_of_last_rate_limit_error: float = 0

    def __str__(self):
        return (
            f"Tasks - Started: {self.num_tasks_started}, "
            f"In Progress: {self.num_tasks_in_progress}, "
            f"Succeeded: {self.num_tasks_succeeded}, "
            f"Failed: {self.num_tasks_failed}, "
            f"Already Completed: {self.num_tasks_already_completed}, "
            f"Rate Limit Errors: {self.num_rate_limit_errors}"
        )

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

    async def call_api(
        self,
        client,
        retry_queue: asyncio.Queue,
        save_filepath: str,
        status_tracker: StatusTracker,
    ) -> None:
        """Calls the LiteLLM API and saves results."""
        try:
            # Make API call with instructor
            if self.generic_request.response_format:
                response, completion_obj = await client.chat.completions.create_with_completion(
                    **self.api_specific_request,
                    response_model=self.prompt_formatter.response_format,
                    timeout=60.0
                )
                response_message = response.model_dump() if hasattr(response, 'model_dump') else response
            else:
                completion_obj = await litellm.acompletion(**self.api_specific_request, timeout=60.0)
                response_message = completion_obj.choices[0].message.content

            # Extract token usage
            usage = completion_obj.usage
            token_usage = TokenUsage(
                prompt_tokens=usage.prompt_tokens,
                completion_tokens=usage.completion_tokens,
                total_tokens=usage.total_tokens
            )

            # Calculate cost
            cost = litellm.completion_cost(completion_response=completion_obj.model_dump())

            # Create and save response
            generic_response = GenericResponse(
                response_message=response_message,
                response_errors=None,
                raw_request=self.api_specific_request,
                raw_response=completion_obj.model_dump(),
                generic_request=self.generic_request,
                created_at=self.created_at,
                finished_at=datetime.datetime.now(),
                token_usage=token_usage,
                response_cost=cost
            )
            
            await append_generic_response(generic_response, save_filepath)
            status_tracker.num_tasks_in_progress -= 1
            status_tracker.num_tasks_succeeded += 1

        except Exception as e:
            logger.error(f"Error in request {self.task_id}: {str(e)}")
            self.result.append(e)
            
            if "rate limit" in str(e).lower():
                status_tracker.num_rate_limit_errors += 1
                status_tracker.time_of_last_rate_limit_error = time.time()
            
            if self.attempts_left > 0:
                self.attempts_left -= 1
                retry_queue.put_nowait(self)
            else:
                generic_response = GenericResponse(
                    response_message=None,
                    response_errors=[str(e) for e in self.result],
                    raw_request=self.api_specific_request,
                    raw_response=None,
                    generic_request=self.generic_request,
                    created_at=self.created_at,
                    finished_at=datetime.datetime.now(),
                )
                await append_generic_response(generic_response, save_filepath)
                status_tracker.num_tasks_in_progress -= 1
                status_tracker.num_tasks_failed += 1

class LiteLLMOnlineRequestProcessor(BaseRequestProcessor):
    """Request processor for LiteLLM that supports structured outputs via instructor."""

    def __init__(
        self,
        model: str,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        presence_penalty: Optional[float] = None,
        frequency_penalty: Optional[float] = None,
    ):
        super().__init__(batch_size=None)
        self.model = model
        self.temperature = temperature
        self.top_p = top_p
        self.presence_penalty = presence_penalty
        self.frequency_penalty = frequency_penalty
        self.client = instructor.from_litellm(litellm.acompletion)
        logger.info(f"Initialized LiteLLM processor with model: {self.model}")

    def get_rate_limits(self) -> dict:
        """Get rate limits for the model. Uses default conservative values."""
        try:
            # Try to get a test completion to extract rate limits
            completion = litellm.completion(
                model=self.model,
                messages=[{"role": "user", "content": "test"}],
            )
            
            # Try to get rate limits from response headers
            headers = completion._hidden_params.get('additional_headers', {})
            rpm = int(headers.get('x-ratelimit-limit-requests', 3000))
            tpm = int(headers.get('x-ratelimit-limit-tokens', 150_000))
            
        except Exception as e:
            logger.warning(f"Failed to get rate limits from API, using default values: {str(e)}")
            # Use conservative default values
            rpm = 3000  # requests per minute
            tpm = 150_000  # tokens per minute

        logger.info(f"Rate limits for {self.model} - RPM: {rpm}, TPM: {tpm}")
        
        return {
            "max_requests_per_minute": rpm,
            "max_tokens_per_minute": tpm,
        }

    def create_api_specific_request(self, generic_request: GenericRequest) -> dict:
        """Create a LiteLLM-specific request from a generic request.
        
        Args:
            generic_request (GenericRequest): The generic request to convert
            
        Returns:
            dict: LiteLLM-specific request parameters
        """
        # Get supported parameters for this model
        supported_params = litellm.get_supported_openai_params(model=self.model)
        
        request = {
            "model": generic_request.model,
            "messages": generic_request.messages,
        }

        # Only add parameters that are supported by this model
        if "temperature" in supported_params and self.temperature is not None:
            request["temperature"] = self.temperature
            
        if "top_p" in supported_params and self.top_p is not None:
            request["top_p"] = self.top_p
            
        if "presence_penalty" in supported_params and self.presence_penalty is not None:
            request["presence_penalty"] = self.presence_penalty
            
        if "frequency_penalty" in supported_params and self.frequency_penalty is not None:
            request["frequency_penalty"] = self.frequency_penalty

        return request

    async def process_requests_from_file(
        self,
        generic_requests_filepath: str,
        save_filepath: str,
        max_attempts: int,
        resume: bool,
    ) -> None:
        """Process API requests with simple rate limiting."""
        # Get rate limits
        rate_limits = self.get_rate_limits()
        requests_per_minute = rate_limits["max_requests_per_minute"]
        seconds_per_request = 60.0 / requests_per_minute

        # Initialize trackers
        status_tracker = StatusTracker()
        retry_queue = asyncio.Queue()
        next_request = None
        last_request_time = 0

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
        pbar = tqdm(total=total_requests, desc="Processing LiteLLM requests")

        with open(generic_requests_filepath) as file:
            while True:
                # Get next request
                if next_request is None:
                    if not retry_queue.empty():
                        next_request = retry_queue.get_nowait()
                    else:
                        try:
                            line = next(file)
                            generic_request = GenericRequest.model_validate_json(line)
                            
                            if resume and generic_request.original_row_idx in completed_request_ids:
                                status_tracker.num_tasks_already_completed += 1
                                continue
                            
                            next_request = APIRequest(
                                task_id=status_tracker.num_tasks_started,
                                generic_request=generic_request,
                                api_specific_request=self.create_api_specific_request(generic_request),
                                attempts_left=max_attempts,
                                prompt_formatter=self.prompt_formatter
                            )
                            status_tracker.num_tasks_started += 1
                            status_tracker.num_tasks_in_progress += 1
                        except StopIteration:
                            if status_tracker.num_tasks_in_progress == 0:
                                break

                # Process request with rate limiting
                if next_request:
                    current_time = time.time()
                    time_since_last_request = current_time - last_request_time
                    
                    if time_since_last_request < seconds_per_request:
                        await asyncio.sleep(seconds_per_request - time_since_last_request)
                    
                    asyncio.create_task(
                        next_request.call_api(
                            client=self.client,
                            retry_queue=retry_queue,
                            save_filepath=save_filepath,
                            status_tracker=status_tracker,
                        )
                    )
                    last_request_time = time.time()
                    next_request = None

                # Update progress
                total_completed = (
                    status_tracker.num_tasks_succeeded
                    + status_tracker.num_tasks_failed
                    + status_tracker.num_tasks_already_completed
                )
                if total_completed > pbar.n:
                    pbar.update(total_completed - pbar.n)

                await asyncio.sleep(0.1)

        pbar.close()
        logger.info(f"Processing complete. Final status: {status_tracker}")

    def run(
        self,
        dataset: Optional[Dataset],
        working_dir: str,
        parse_func_hash: str,
        prompt_formatter: PromptFormatter,
    ) -> Dataset:
        """Run the LiteLLM processor on the dataset."""
        self.prompt_formatter = prompt_formatter
        generic_requests_files = self.create_request_files(
            dataset, working_dir, prompt_formatter
        )
        
        generic_responses_files = [
            f"{working_dir}/responses_{i}.jsonl"
            for i in range(len(generic_requests_files))
        ]

        for request_file, response_file in zip(
            generic_requests_files, generic_responses_files
        ):
            run_in_event_loop(
                self.process_requests_from_file(
                    generic_requests_filepath=request_file,
                    save_filepath=response_file,
                    max_attempts=5,
                    resume=True,
                )
            )

        return self.create_dataset_files(
            working_dir, parse_func_hash, prompt_formatter
        )

async def append_generic_response(data: GenericResponse, filename: str) -> None:
    """Append a response to a jsonl file."""
    json_string = json.dumps(data.model_dump(), default=str)
    with open(filename, "a") as f:
        f.write(json_string + "\n")
