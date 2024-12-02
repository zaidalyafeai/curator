import logging
from typing import Optional
import asyncio
import aiohttp
import litellm
from litellm import get_supported_openai_params
import datetime
import instructor
from bespokelabs.curator.request_processor.online_request_processor import (
    OnlineRequestProcessor,
    APIRequest,
    StatusTracker,
)
from bespokelabs.curator.request_processor.generic_request import GenericRequest
from bespokelabs.curator.request_processor.generic_response import TokenUsage, GenericResponse
from pydantic import BaseModel
from bespokelabs.curator.prompter.prompt_formatter import PromptFormatter

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

litellm.suppress_debug_info = True


class LiteLLMOnlineRequestProcessor(OnlineRequestProcessor):
    """LiteLLM implementation of the OnlineRequestProcessor for multi-provider LLM support.

    This processor uses LiteLLM to handle requests across different LLM providers (OpenAI, Anthropic, etc.)
    with unified interface and structured output support via instructor.

    Features:
        - Multi-provider support through LiteLLM
        - Structured output via instructor
        - Automatic token counting and rate limiting
        - Provider-specific parameter handling
        - Cost tracking per request

    Attributes:
        model (str): The model identifier (e.g., "gpt-4", "claude-2")
        temperature (Optional[float]): Temperature for response randomness
        top_p (Optional[float]): Top-p sampling parameter
        presence_penalty (Optional[float]): Presence penalty for response diversity
        frequency_penalty (Optional[float]): Frequency penalty for response diversity
        client: Instructor-wrapped LiteLLM client for structured outputs
    """

    def __init__(
        self,
        model: str,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        presence_penalty: Optional[float] = None,
        frequency_penalty: Optional[float] = None,
    ):
        super().__init__(
            model=model,
            temperature=temperature,
            top_p=top_p,
            presence_penalty=presence_penalty,
            frequency_penalty=frequency_penalty,
        )
        self.client = instructor.from_litellm(litellm.acompletion)

    def check_structured_output_support(self, prompt_formatter: PromptFormatter):
        """Verify if the model supports structured output via instructor.

        Tests the model's capability to handle structured output by making a test request
        with a simple schema.

        Args:
            prompt_formatter (PromptFormatter): Contains response format requirements

        Returns:
            bool: True if structured output is supported, False otherwise

        Note:
            - Uses a simple User schema as test case
            - Logs detailed information about support status
            - Required for models that will use JSON schema responses
        """
        assert prompt_formatter.response_format

        class User(BaseModel):
            name: str
            age: int

        try:
            client = instructor.from_litellm(litellm.completion)
            response = client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": "Jason is 25 years old."}],
                response_model=User,
            )
            logger.info(f"Check instructor structure output response: {response}")
            assert isinstance(response, User)
            logger.info(
                f"Model {self.model} supports structured output via instructor, response: {response}"
            )
            return True
        except Exception as e:
            logger.warning(
                f"Model {self.model} does not support structured output via instructor: {e}"
            )
            return False

    def estimate_output_tokens(self) -> int:
        """Estimate the number of tokens in the model's response.

        Uses LiteLLM's get_max_tokens and applies a conservative estimate
        by dividing by 4 to avoid hitting context limits.

        Returns:
            int: Estimated number of output tokens

        Note:
            Falls back to 0 if token estimation fails
        """
        try:
            return litellm.get_max_tokens(model=self.model) // 4
        except Exception:
            return 0

    def estimate_total_tokens(self, messages: list) -> int:
        """Calculate the total token usage for a request.

        Uses LiteLLM's token_counter for accurate input token counting
        and adds estimated output tokens.

        Args:
            messages (list): List of message dictionaries

        Returns:
            int: Total estimated tokens (input + output)
        """
        input_tokens = litellm.token_counter(model=self.model, messages=messages)
        output_tokens = self.estimate_output_tokens()
        return input_tokens + output_tokens

    def get_rate_limits(self) -> dict:
        """Retrieve rate limits from the LLM provider via LiteLLM.

        Makes a test request to get rate limit information from response headers.

        Returns:
            dict: Contains 'max_requests_per_minute' and 'max_tokens_per_minute'

        Note:
            - Falls back to default values if headers are missing
            - Some providers (e.g., Claude) require non-empty messages
        """
        logger.info(f"Getting rate limits for model: {self.model}")

        completion = litellm.completion(
            model=self.model,
            messages=[
                {"role": "user", "content": "hi"}
            ],  # Some models (e.g. Claude) require an non-empty message to get rate limits.
        )

        headers = completion._hidden_params.get("additional_headers", {})
        logger.info(f"Rate limit headers: {headers}")

        rpm = int(headers.get("x-ratelimit-limit-requests", 3000))
        tpm = int(headers.get("x-ratelimit-limit-tokens", 150_000))

        logger.info(f"Rate limits - Requests/min: {rpm}, Tokens/min: {tpm}")

        return {"max_requests_per_minute": rpm, "max_tokens_per_minute": tpm}

    def create_api_specific_request(self, generic_request: GenericRequest) -> dict:
        """Convert a generic request into a LiteLLM-compatible format.

        Checks supported parameters for the specific model and only includes
        applicable parameters.

        Args:
            generic_request (GenericRequest): The generic request to convert

        Returns:
            dict: LiteLLM-compatible request parameters

        Note:
            Uses LiteLLM's get_supported_openai_params to check parameter support
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

    async def process_single_request(
        self,
        request: APIRequest,
        session: aiohttp.ClientSession,
        retry_queue: asyncio.Queue,
        save_filepath: str,
        status_tracker: StatusTracker,
    ) -> None:
        """Process a single request through LiteLLM with error handling.

        Handles both structured and unstructured outputs, tracks token usage
        and costs, and manages retries.

        Args:
            request (APIRequest): Request to process
            session (aiohttp.ClientSession): Async HTTP session
            retry_queue (asyncio.Queue): Queue for failed requests
            save_filepath (str): Path to save responses
            status_tracker (StatusTracker): Tracks request status and limits

        Note:
            - Supports both instructor-based structured output and raw completions
            - Calculates and tracks costs using LiteLLM's completion_cost
            - Implements retry logic for failed requests
            - Saves responses immediately to prevent data loss
        """
        try:
            # Get response directly without extra logging
            if request.generic_request.response_format:
                response, completion_obj = (
                    await self.client.chat.completions.create_with_completion(
                        **request.api_specific_request,
                        response_model=request.prompt_formatter.response_format,
                        timeout=60.0,
                    )
                )
                response_message = (
                    response.model_dump() if hasattr(response, "model_dump") else response
                )
            else:
                completion_obj = await litellm.acompletion(
                    **request.api_specific_request, timeout=60.0
                )
                response_message = completion_obj["choices"][0]["message"]["content"]

            # Extract token usage
            usage = completion_obj.usage if hasattr(completion_obj, "usage") else {}
            token_usage = TokenUsage(
                prompt_tokens=usage.prompt_tokens,
                completion_tokens=usage.completion_tokens,
                total_tokens=usage.total_tokens,
            )

            # Calculate cost using litellm
            cost = litellm.completion_cost(completion_response=completion_obj.model_dump())

            # Create and save response immediately
            generic_response = GenericResponse(
                response_message=response_message,
                response_errors=None,
                raw_request=request.api_specific_request,
                raw_response=completion_obj.model_dump(),
                generic_request=request.generic_request,
                created_at=request.created_at,
                finished_at=datetime.datetime.now(),
                token_usage=token_usage,
                response_cost=cost,
            )

            await self.append_generic_response(generic_response, save_filepath)
            status_tracker.num_tasks_in_progress -= 1
            status_tracker.num_tasks_succeeded += 1
            status_tracker.pbar.update(1)

        except Exception as e:
            logger.error(f"Error in API request: {e}")
            status_tracker.num_other_errors += 1
            request.result.append(e)

            if request.attempts_left > 0:
                request.attempts_left -= 1
                retry_queue.put_nowait(request)
            else:
                generic_response = GenericResponse(
                    response_message=None,
                    response_errors=[str(e) for e in request.result],
                    raw_request=request.api_specific_request,
                    raw_response=None,
                    generic_request=request.generic_request,
                    created_at=request.created_at,
                    finished_at=datetime.datetime.now(),
                )
                await self.append_generic_response(generic_response, save_filepath)
                status_tracker.num_tasks_in_progress -= 1
                status_tracker.num_tasks_failed += 1
