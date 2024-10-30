import instructor
from litellm import acompletion as litellm_acompletion
from jinja2 import Template
from pydantic import BaseModel
from typing import Any, Dict, Type


class PromptCaller:
    """Interface for prompting LLMs."""

    def __init__(self, model_name, system_prompt, user_prompt, response_format: Type[BaseModel]):
        self.model_name = model_name
        self.system_prompt = system_prompt
        self.user_prompt = user_prompt
        self.response_format = response_format

    def get_api_call_fn(self):
      """Get the LLM call API."""
      litellm_client = instructor.from_litellm(litellm_acompletion)
      # Create Jinja2 templates
      system_template = Template(self.system_prompt)
      user_template = Template(self.user_prompt)

      async def litellm_call_with_instructor(row: Dict[str, Any]) -> Any:
          """Call the LLM with instructor."""
          messages = [
              {"role": "system", "content": system_template.render(**row)},
              {"role": "user", "content": user_template.render(**row)},
          ]

          output = await litellm_client.chat.completions.create(
              model=self.model_name,
              messages=messages,
              response_model=self.response_format,
          )
          return output
      return litellm_call_with_instructor
