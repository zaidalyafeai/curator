from jinja2 import Template
from pydantic import BaseModel
from typing import Any, Dict, Optional, Type


class Prompter:
    """Interface for prompting LLMs."""

    def __init__(
        self,
        model_name: str,
        user_prompt: str,
        system_prompt: Optional[str] = None,
        response_format: Optional[Type[BaseModel]] = None,
    ):
        self.model_name = model_name
        self.system_prompt = system_prompt
        self.user_prompt = user_prompt
        self.response_format = response_format

    def get_request_object(self, row: Dict[str, Any], idx: int) -> Dict[str, Any]:
        """Format the request object based off Prompter attributes."""
        messages = []
        if self.system_prompt:
            system_template = Template(self.system_prompt)
            messages.append(
                {"role": "system", "content": system_template.render(**row)}
            )

        user_template = Template(self.user_prompt)
        messages.append({"role": "user", "content": user_template.render(**row)})

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
