import os

from datasets import Dataset, load_dataset

from bespokelabs import curator

from dotenv import load_dotenv

from utils import count_char_types
load_dotenv("../../.env")

class Prompter(curator.LLM):
    """A recipe generator that generates recipes for different cuisines."""

    def prompt(self, input: dict) -> str:
        """Generate a prompt for the recipe generator."""
        return f"Answer the following in Arabic {input['instruction']}"

    def validate(self, response: str) -> bool:
        """Validate the model response is mostly in Arabic."""
        ## calcualte the percentage of Arabic and English characters in the response
        counts = count_char_types(response)
        return (counts['other_language_count'][0] / len(response)) < 0.05

    def parse(self, input: dict, response: str) -> dict:
        """Parse the model response along with the input to the model into the desired output format.."""
        return {
            "answer": response,
            "valid" : self.validate(response),
        }


def main():
    dataset = load_dataset("arbml/CIDAR", trust_remote_code=True)["train"].select(
        range(100)
    )
    model_name = "openrouter/quasar-alpha"
    model_name = f"openrouter/{model_name}"
    llm = Prompter(
        model_name=model_name,
        backend_params={
            "max_requests_per_minute": 10000,
            "max_tokens_per_minute": 10000000,
            "request_timeout": 30 * 60,
        },
    )

    # Generate recipes for all cuisines
    responses = llm(dataset)

    # Print results
    print(responses.to_pandas())


if __name__ == "__main__":
    main()
