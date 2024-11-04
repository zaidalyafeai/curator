from re import M
from typing import Any, Dict, Optional, Type

from jinja2 import Template
from pydantic import BaseModel


class Prompter:
    """Interface for prompting LLMs."""

    def __init__(
        self,
        model_name,
        user_prompt,
        system_prompt: Optional[str] = None,
        response_format: Optional[Type[BaseModel]] = None,
    ):
        self.model_name = model_name
        self.system_prompt = system_prompt
        self.user_prompt = user_prompt
        self.response_format = response_format


    def _render_row(self, prompt: str, row: Dict[str, Any]) -> Dict[str, Any]:
        # Replace all '.' with '____' in the prompt to avoid conflicts with
        # Jinja2 template syntax. Jinja2 uses '.' to access dictionary keys.
        template = Template(prompt.replace(".", "______"))
        return template.render(**{k.replace(".", "______"): v for k, v in row.items()})

    def get_request_object(self, row: Dict[str, Any], idx: int) -> Dict[str, Any]:
        """Format the request object based off Prompter attributes."""
        messages = []
        if self.system_prompt:
            
            messages.append(
                {"role": "system", "content": self._render_row(self.system_prompt, row)}
            )

        messages.append({"role": "user", "content": self._render_row(self.user_prompt, row)})

        if self.response_format:
            # OpenAI API https://platform.openai.com/docs/api-reference/chat/create#chat-create-response_format
            request = {
                "model": self.model_name,
                "messages": messages,
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "output_schema",  # not sure if this should be something else. Also not sure if we should use strict: True
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
