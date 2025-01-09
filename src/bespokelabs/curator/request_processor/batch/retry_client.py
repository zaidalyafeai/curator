import asyncio
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    before_sleep_log,
)
from openai import APIConnectionError as OpenAIAPIConnectionError
from anthropic import APIConnectionError as AnthropicAPIConnectionError
import logging
from functools import wraps

logger = logging.getLogger(__name__)


class RetryableClient:
    """Wrapper class that adds retry logic to AsyncOpenAI client methods."""

    def __init__(self, client, max_retries):
        self.client = client
        self.max_retries = max_retries

    def __getattr__(self, name):
        attr = getattr(self.client, name)
        if callable(attr) and asyncio.iscoroutinefunction(attr):
            return self._wrap_with_retry(attr)
        return attr

    def _wrap_with_retry(self, func):
        @retry(
            retry=retry_if_exception_type(
                (OpenAIAPIConnectionError, AnthropicAPIConnectionError)
            ),
            wait=wait_exponential(multiplier=1, min=1, max=60),
            stop=stop_after_attempt(lambda: self.max_retries + 1),
            before_sleep=before_sleep_log(logger, logging.WARNING),
            after=lambda retry_state: (
                logger.error(
                    f"API Connection Error after {retry_state.attempt_number} attempts. This could be due to:\n"
                    "1. Network connectivity issues\n"
                    "2. API service being down\n"
                    "3. Firewall or proxy settings blocking the connection\n"
                    "4. DNS resolution problems\n"
                    f"Original error: {str(retry_state.outcome.exception())}"
                )
                if retry_state.attempt_number
                > retry_state.retry_object.stop.max_attempt_number
                else None
            ),
        )
        @wraps(func)
        async def wrapper(*args, **kwargs):
            return await func(*args, **kwargs)

        return wrapper
