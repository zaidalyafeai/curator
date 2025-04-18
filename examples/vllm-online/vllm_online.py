"""Generate synthetic recipes for different cuisines using vLLM online API.

This script demonstrates using vLLM backend with curator to generate recipes for various
world cuisines in an efficient batched manner. It uses Meta-Llama-3.1-8B-Instruct model.

To start the vLLM server, run the following command:
vllm serve
Qwen/Qwen2.5-3B-Instruct
--host localhost
--port 8787
--api-key token-abc123
"""

import os

from datasets import Dataset, load_dataset

from bespokelabs import curator


class Prompter(curator.LLM):
    """A recipe generator that generates recipes for different cuisines."""

    def prompt(self, input: dict) -> str:
        """Generate a prompt for the recipe generator."""
        return f"Answer the following in Arabic {input['instruction']}"

    def validate(self, response: str) -> bool:
        """Validate the model response is mostly in Arabic."""
        ## calcualte the percentage of Arabic and English characters in the response
        english_chars = "abcdefghijklmnopqrstuvwxyz"    
        arabic_chars = "أبتثجحخدذرزسشصضطظعغفقكلمنهوي"
        other_chars = ".,!?'\" "
        chars = sum(1 for char in response if char.lower() in english_chars + arabic_chars + other_chars) 
        return chars / len(response) > 0.9

    def parse(self, input: dict, response: str) -> dict:
        """Parse the model response along with the input to the model into the desired output format.."""
        if not self.validate(response):
            response = "[This is not a valid response]"
        return {
            "answer": response,
        }


def main():
    """Generate recipes for different world cuisines using vLLM.

    Creates a dataset of cuisine names, sets up a recipe generation prompter using vLLM backend,
    and generates creative but realistic recipes for each cuisine. The results are printed
    as a pandas DataFrame.
    """
    # List of cuisines to generate recipes for
   
    dataset = load_dataset("arbml/CIDAR", trust_remote_code=True)["train"].select(
        range(1000)
    )

    model_path = "Qwen/Qwen2.5-7B-Instruct"
    model_path = f"hosted_vllm/{model_path}"  # Use the hosted_vllm backend

    API_KEY = "token-abc123"
    os.environ["HOSTED_VLLM_API_KEY"] = API_KEY

    # Define the vLLM server params
    PORT = 8787
    HOST = "localhost"

    prompter = Prompter(
        model_name=model_path,
        backend="litellm",
        backend_params={
            "base_url": f"http://{HOST}:{PORT}/v1",
            "max_requests_per_minute": 10000,
            "max_retries": 50,
        },
    )

    # Generate recipes for all cuisines
    recipes = prompter(dataset)

    # Print results
    print(recipes.to_pandas())


if __name__ == "__main__":
    main()
