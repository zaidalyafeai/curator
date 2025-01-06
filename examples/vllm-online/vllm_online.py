"""Generate synthetic recipes for different cuisines using vLLM online API."""

from bespokelabs import curator
from datasets import Dataset


def main():
    # List of cuisines to generate recipes for
    cuisines = [
        {"cuisine": cuisine}
        for cuisine in [
            "Chinese",
            "Italian",
            "Mexican",
            "French",
            "Japanese",
            "Indian",
            "Thai",
            "Korean",
            "Vietnamese",
            "Brazilian",
        ]
    ]
    cuisines = Dataset.from_list(cuisines)

    model_path = "meta-llama/Meta-Llama-3.1-8B-Instruct"

    # Define the vLLM server params
    PORT = 8787
    HOST = "localhost"
    URL = f"http://{HOST}:{PORT}/v1/chat/completions"
    API_KEY = "token-abc123"

    recipe_prompter = curator.LLM(
        model_name=model_path,
        prompt_func=lambda row: f"Generate a random {row['cuisine']} recipe. Be creative but keep it realistic.",
        parse_func=lambda row, response: {
            "recipe": response,
            "cuisine": row["cuisine"],
        },
        backend="openai",
        api_key=API_KEY,
        url=URL,
    )

    # Generate recipes for all cuisines
    recipes = recipe_prompter(cuisines)

    # Print results
    print(recipes.to_pandas())


if __name__ == "__main__":
    main()
