import json
import logging
import os
from typing import Optional, Any
import asyncio
import time
import aiohttp
import resource
from tqdm import tqdm
from dataclasses import dataclass, field
from pydantic import BaseModel
import instructor
import litellm
from litellm import completion, get_supported_openai_params
import aiofiles
from bespokelabs.curator.dataset import Dataset
from bespokelabs.curator.prompter.prompter import PromptFormatter
from bespokelabs.curator.request_processor.base_request_processor import (
    BaseRequestProcessor,
    GenericRequest,
    GenericResponse,
)
from bespokelabs.curator.request_processor.event_loop import run_in_event_loop
import datetime
from bespokelabs.curator.request_processor.generic_response import TokenUsage
from litellm import token_counter, get_max_tokens

logger = logging.getLogger(__name__)
# litellm.set_verbose=True
litellm.suppress_debug_info = True

@dataclass
class StatusTracker:
    """Tracks the status of all requests."""
    num_tasks_started: int = 0
    num_tasks_in_progress: int = 0
    num_tasks_succeeded: int = 0
    num_tasks_failed: int = 0
    num_tasks_already_completed: int = 0
    num_other_errors: int = 0
    available_request_capacity: float = 0
    available_token_capacity: float = 0
    last_update_time: float = field(default_factory=time.time)
    max_requests_per_minute: int = 0
    max_tokens_per_minute: int = 0
    pbar: tqdm = field(default=None)

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
            self.available_request_capacity + 
            self.max_requests_per_minute * seconds_since_update / 60.0,
            self.max_requests_per_minute
        )
        
        self.available_token_capacity = min(
            self.available_token_capacity + 
            self.max_tokens_per_minute * seconds_since_update / 60.0,
            self.max_tokens_per_minute
        )
        
        self.last_update_time = current_time

    def has_capacity(self, token_estimate: int) -> bool:
        """Check if there's enough capacity for a request"""
        self.update_capacity()
        return (self.available_request_capacity >= 1 and 
                self.available_token_capacity >= token_estimate)

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

    async def call_api(
        self,
        session: aiohttp.ClientSession,
        client: Any,
        retry_queue: asyncio.Queue,
        save_filepath: str,
        status_tracker: StatusTracker,
    ) -> None:
        """Calls the LiteLLM API and saves results."""
        try:
            # Get response directly without extra logging
            if self.generic_request.response_format:
                response, completion_obj = await client.chat.completions.create_with_completion(
                    **self.api_specific_request,
                    response_model=self.prompt_formatter.response_format,
                    timeout=60.0
                )
                response_message = response.model_dump() if hasattr(response, 'model_dump') else response
            else:
                completion_obj = await completion(**self.api_specific_request, timeout=60.0)
                response_message = completion_obj.content if hasattr(completion_obj, 'content') else str(completion_obj)

            # Extract token usage
            usage = completion_obj.usage if hasattr(completion_obj, 'usage') else {}
            token_usage = TokenUsage(
                prompt_tokens=usage.prompt_tokens,
                completion_tokens=usage.completion_tokens,
                total_tokens=usage.total_tokens
            )

            # Calculate cost using litellm
            cost = litellm.completion_cost(
                completion_response=completion_obj.model_dump()
            )

            # Create and save response immediately
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
            status_tracker.pbar.update(1)

        except Exception as e:
            status_tracker.num_other_errors += 1
            self.result.append(e)
            
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
        self.token_estimate_cache = {}
        
        logger.info(f"Using model: {self.model}")
        logger.debug(f"Parameters - Temperature: {self.temperature}, Top P: {self.top_p}, "
                    f"Presence Penalty: {self.presence_penalty}, Frequency Penalty: {self.frequency_penalty}")
        self.client = instructor.from_litellm(litellm.acompletion)
        logger.info("Instructor client initialized with LiteLLM backend")

    def estimate_output_tokens(self) -> int:
        """Estimate output tokens for a request"""
        try:
            return get_max_tokens(model=self.model) // 4
        except Exception:
            return 0

    def estimate_total_tokens(self, messages: list) -> int:
        """Estimate total tokens for a request"""
        input_tokens = token_counter(model=self.model, messages=messages)
        output_tokens = self.estimate_output_tokens()
        return input_tokens + output_tokens

    def get_rate_limits(self) -> dict:
        """Get rate limits from LiteLLM response headers."""
        logger.info(f"Getting rate limits for model: {self.model}")
        
        completion = litellm.completion(
            model=self.model,
            messages=[{"role": "user", "content": "hi"}], # Some models (e.g. Claude) require an non-empty message to get rate limits.
        )
        
        headers = completion._hidden_params.get('additional_headers', {})
        logger.debug(f"Rate limit headers: {headers}")
        print(headers)
        
        rpm = int(headers.get('x-ratelimit-limit-requests', 3000))
        tpm = int(headers.get('x-ratelimit-limit-tokens', 150_000))
        
        logger.info(f"Rate limits - Requests/min: {rpm}, Tokens/min: {tpm}")
        
        return {
            "max_requests_per_minute": rpm,
            "max_tokens_per_minute": tpm
        }

    def create_api_specific_request(self, generic_request: GenericRequest) -> dict:
        """Create a LiteLLM-specific request from a generic request.
        
        Args:
            generic_request (GenericRequest): The generic request to convert
            
        Returns:
            dict: LiteLLM-specific request parameters
        """
        # Get supported parameters for this model
        supported_params = get_supported_openai_params(model=self.model)
        
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
        """Processes API requests in parallel, throttling to stay under rate limits."""
        # Increase file descriptor limit for higher throughput
        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        resource.setrlimit(resource.RLIMIT_NOFILE, (min(hard, 10000), hard))

        # Initialize trackers
        queue_of_requests_to_retry = asyncio.Queue()
        status_tracker = StatusTracker()
        next_request = None
        file_not_finished = True

        # Get rate limits
        rate_limits = self.get_rate_limits()
        status_tracker.max_requests_per_minute = rate_limits["max_requests_per_minute"]
        status_tracker.max_tokens_per_minute = rate_limits["max_tokens_per_minute"]
        rpm = rate_limits["max_requests_per_minute"]

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
        status_tracker.pbar = tqdm(initial=len(completed_request_ids), total=total_requests, desc="Processing LiteLLM requests")

        # Use higher connector limit for better throughput
        connector = aiohttp.TCPConnector(limit=rpm)
        async with aiohttp.ClientSession(connector=connector) as session:
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
                        prompt_formatter=self.prompt_formatter
                    )
                    
                    token_estimate = self.estimate_total_tokens(request.generic_request.messages)
                    
                    # Wait for capacity if needed
                    while not status_tracker.has_capacity(token_estimate):
                        await asyncio.sleep(0.1)
                    
                    # Consume capacity before making request
                    status_tracker.consume_capacity(token_estimate)
                    
                    task = asyncio.create_task(
                        request.call_api(
                            session=session,
                            client=self.client,
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
                        token_estimate = self.estimate_total_tokens(retry_request.generic_request.messages)
                        
                        # Wait for capacity if needed
                        while not status_tracker.has_capacity(token_estimate):
                            await asyncio.sleep(0.1)
                        
                        # Consume capacity before making request
                        status_tracker.consume_capacity(token_estimate)
                        
                        task = asyncio.create_task(
                            retry_request.call_api(
                                session=session,
                                client=self.client,
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

    def run(
        self,
        dataset: Optional[Dataset],
        working_dir: str,
        parse_func_hash: str,
        prompt_formatter: PromptFormatter,
    ) -> Dataset:
        """Run completions using LiteLLM with async processing."""
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
    """Append a response to a jsonl file with async file operations."""
    json_string = json.dumps(data.model_dump(), default=str)
    async with aiofiles.open(filename, "a") as f:
        await f.write(json_string + "\n")
    logger.debug(f"Successfully appended response to {filename}")