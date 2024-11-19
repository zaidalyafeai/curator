import json
import logging
import os
import argparse
from typing import Optional
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

logger = logging.getLogger(__name__)

@dataclass
class StatusTracker:
    """Tracks the status of all requests."""
    num_tasks_started: int = 0
    num_tasks_in_progress: int = 0
    num_tasks_succeeded: int = 0
    num_tasks_failed: int = 0
    num_tasks_already_completed: int = 0
    num_other_errors: int = 0

    def __str__(self):
        return (
            f"Tasks - Started: {self.num_tasks_started}, "
            f"In Progress: {self.num_tasks_in_progress}, "
            f"Succeeded: {self.num_tasks_succeeded}, "
            f"Failed: {self.num_tasks_failed}, "
            f"Already Completed: {self.num_tasks_already_completed}\n"
            f"Errors: {self.num_other_errors}"
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

    async def call_api(
        self,
        session: aiohttp.ClientSession,
        client,
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
                    response_model=self.prompt_formatter.response_format
                )
                response_message = response.model_dump() if hasattr(response, 'model_dump') else response
            else:
                completion_obj = await completion(**self.api_specific_request)
                response_message = completion_obj.content if hasattr(completion_obj, 'content') else str(completion_obj)

            # Create and save response immediately
            generic_response = GenericResponse(
                response_message=response_message,
                response_errors=None,
                raw_request=self.api_specific_request,
                raw_response=completion_obj.model_dump(),
                generic_request=self.generic_request,
            )
            
            await append_generic_response(generic_response, save_filepath)
            status_tracker.num_tasks_in_progress -= 1
            status_tracker.num_tasks_succeeded += 1

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
        
        logger.info(f"Using model: {self.model}")
        logger.debug(f"Parameters - Temperature: {self.temperature}, Top P: {self.top_p}, "
                    f"Presence Penalty: {self.presence_penalty}, Frequency Penalty: {self.frequency_penalty}")
        
        self.client = instructor.from_litellm(litellm.acompletion)
        logger.info("Instructor client initialized with LiteLLM backend")

    def get_rate_limits(self) -> dict:
        """Get rate limits from LiteLLM response headers."""
        logger.info(f"Getting rate limits for model: {self.model}")
        
        completion = litellm.completion(
            model=self.model,
            messages=[{"role": "user", "content": "hi"}],
        )
        
        headers = completion._hidden_params.get('additional_headers', {})
        logger.debug(f"Rate limit headers: {headers}")
        
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
        pbar = tqdm(total=total_requests, desc="Processing LiteLLM requests")

        # Use higher connector limit for better throughput
        connector = aiohttp.TCPConnector(limit=10 * rpm)
        async with aiohttp.ClientSession(connector=connector) as session:
            async with aiofiles.open(generic_requests_filepath) as file:
                async for line in file:
                    if next_request is None:
                        if not queue_of_requests_to_retry.empty():
                            next_request = queue_of_requests_to_retry.get_nowait()
                        elif file_not_finished:
                            try:
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
                                file_not_finished = False

                    if next_request:
                        asyncio.create_task(
                            next_request.call_api(
                                session=session,
                                client=self.client,
                                retry_queue=queue_of_requests_to_retry,
                                save_filepath=save_filepath,
                                status_tracker=status_tracker,
                            )
                        )
                        next_request = None

                    # Update progress
                    total_completed = (
                        status_tracker.num_tasks_succeeded
                        + status_tracker.num_tasks_failed
                        + status_tracker.num_tasks_already_completed
                    )
                    if total_completed > pbar.n:
                        pbar.update(total_completed - pbar.n)

                    if status_tracker.num_tasks_in_progress == 0:
                        break

                    await asyncio.sleep(0.001)  # 1ms sleep to prevent CPU spinning while allowing high throughput

            pbar.close()
            
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
