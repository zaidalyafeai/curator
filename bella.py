from typing import Optional

import dask
import instructor
from litellm import completion as litellm_completion
import openai


class Models:
  GPT_4O = "gpt-4o-2024-08-06"
  GPT_4O_MINI = "gpt-4o-mini"
  GPT_3_5_TURBO = "gpt-3.5-turbo"
  SONNET_3_5 = "anthropic/claude-3-5-sonnet-20240620"


class PromptLayer:
  def __init__(self, system_prompt, response_format, model_name):
    self.system_prompt = system_prompt
    self.response_format = response_format
    self.model_name = model_name

  def get_method(self):
    litellm_client = instructor.from_litellm(litellm_completion)
    openai_client = openai.OpenAI()
        
    def openai_call_with_json_mode(user_prompt: str):
      messages = [
        {"role": "system", "content": self.system_prompt},
        {"role": "user", "content": user_prompt},
      ]
      completion = openai_client.beta.chat.completions.parse(
        model="gpt-4o-2024-08-06",
        messages=messages,
        response_format=self.response_format,
      )
      return completion.choices[0].message.parsed

    def litellm_call_with_instructor(user_prompt: str):
      messages = [
          {"role": "system", "content": self.system_prompt},
          {"role": "user", "content": user_prompt},
      ]
      output = litellm_client.chat.completions.create(
          model=self.model_name,
          messages=messages,
          response_model=self.response_format,
      )
      return output

    if self.model_name == "gpt-4o-2024-08-06":
      return openai_call_with_json_mode
    return litellm_call_with_instructor

  def __call__(self, prompt: Optional[str] = ''):
    method = self.get_method()
    return method(prompt)


class DelayedPromptLayer(PromptLayer):
    """A wrapper around PromptLayer that returns a dask delayed object."""
    def __call__(self, prompt: Optional[str] = ''):
        return dask.delayed(super().__call__)(prompt)