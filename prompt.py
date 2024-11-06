import inspect
from typing import Any, Callable, Dict, Optional, Type

from jinja2 import Template
from pydantic import BaseModel


def prompter(model_name: str, response_format: Optional[Type[BaseModel]] = None):
    def decorator(func):
        return Prompter(model_name, func, response_format=response_format)

    return decorator


class Prompter:
    """Interface for prompting LLMs."""

    def __init__(
        self,
        model_name: str,
        prompting_func: Callable,
        response_format: Optional[Type[BaseModel]] = None,
    ):
        self.model_name = model_name
        self.prompting_func = prompting_func
        self.response_format = response_format

    def get_request_object(
        self, row: Dict[str, Any] | BaseModel, idx: int
    ) -> Dict[str, Any]:
        """Format the request object based off Prompter attributes."""
        if isinstance(row, BaseModel):
            row = row.model_dump()

        sig = inspect.signature(self.prompting_func)
        kwargs_row = {k: v for k, v in row.items() if k in sig.parameters}
        prompts = self.prompting_func(**kwargs_row)

        messages = []
        system_prompt = prompts.get("system_prompt", "You are a helpful AI assistant.")
        messages.append({"role": "system", "content": system_prompt})

        if "user_prompt" not in prompts:
            raise ValueError("user_prompt is required")
        messages.append({"role": "user", "content": prompts["user_prompt"]})

        if self.response_format:
            # OpenAI API
            # https://platform.openai.com/docs/api-reference/chat/create#chat-create-response_format
            request = {
                "model": self.model_name,
                "messages": messages,
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {
                        # TODO(ryan): not sure if this should be something else.
                        # TODO(ryan): also not sure if we should use strict: True
                        "name": "output_schema",
                        "schema": self.response_format.model_json_schema(),
                    },
                },
                "metadata": {"request_idx": idx, "sample": row},
            }
        else:
            request = {
                "model": self.model_name,
                "messages": messages,
                "metadata": {"request_idx": idx, "sample": row},
            }
        return request
