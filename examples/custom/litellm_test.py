import litellm
from litellm import completion
litellm._turn_on_debug()
response = completion(
        model = "hosted_vllm/google/gemma-3-12b-it",
        messages = [
            {
                "role": "user",
                "content": "what is ai?"
            }
        ],
        api_base = "http://localhost:8787/v1",
)
