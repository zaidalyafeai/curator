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
    dataset = load_dataset("arbml/CIDAR", trust_remote_code=True)["train"].select(
        range(10)
    )

    os.environ['DEEPSEEK_API_KEY'] = os.environ['DEEPSEEK_API_KEY']
    llm = Prompter(
        model_name="deepseek-chat",
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
