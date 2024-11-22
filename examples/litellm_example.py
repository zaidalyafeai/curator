from typing import List
from pydantic import BaseModel, Field
from bespokelabs import curator
from datasets import Dataset

# Define response format using Pydantic
class Recipe(BaseModel):
    title: str = Field(description="Title of the recipe")
    ingredients: List[str] = Field(description="List of ingredients needed")
    instructions: List[str] = Field(description="Step by step cooking instructions")

def prompt_func(cuisine):
    return f"Generate a random {cuisine['cuisine_type']} recipe. Be creative but keep it realistic."

def parse_func(row, response):
    return {
        "title": response.title,
        "ingredients": response.ingredients,
        "instructions": response.instructions,
        "cuisine_type": row["cuisine_type"]  # Keep track of cuisine type
    }

def main():
    # List of cuisines to generate recipes for
    cuisines = ["Chinese"] * 100
    
    # Create input dataset with cuisine types
    input_data = [{"cuisine_type": cuisine} for cuisine in cuisines]
    input_dataset = Dataset.from_list(input_data)

    # List of models for testing
    model_names = [
                    # "claude-3-5-sonnet-20240620", # https://docs.litellm.ai/docs/providers/anthropic
                    # "claude-3-haiku-20240307",
                    # "claude-3-opus-20240229",
                    # "claude-3-sonnet-20240229",
                    "gpt-4o-mini", # https://docs.litellm.ai/docs/providers/openai
                    # "gpt-4o-mini-2024-07-18	", # https://docs.litellm.ai/docs/providers/openai
                    # "gpt-4o-2024-08-06",
                    # "gpt-4-0125-preview",
                    # "gpt-3.5-turbo-1106",
                    # "o1-mini",
                    # "gemini/gemini-1.5-flash", # https://docs.litellm.ai/docs/providers/gemini; https://ai.google.dev/gemini-api/docs/models
                    # "gemini/gemini-1.5-pro",
                    # "sambanova/Meta-Llama-3.1-8B-Instruct", # https://docs.litellm.ai/docs/providers/sambanova; https://community.sambanova.ai/t/supported-models
                    # "sambanova/Meta-Llama-3.1-70B-Instruct",
                    # "together/meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo", # https://docs.together.ai/docs/serverless-models
                    # "together/meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo",
                    # "together/nvidia/Llama-3.1-Nemotron-70B-Instruct-HF",
                    ]

    for model_name in model_names:
        # Create prompter using LiteLLM backend
        recipe_prompter = curator.Prompter(
            model_name=model_name,
            prompt_func=prompt_func,
            parse_func=parse_func,
            response_format=Recipe,
            backend="litellm",
        )

        # Generate recipes for all cuisines
        recipes = recipe_prompter(input_dataset)
    
        # Print results
        df = recipes.to_pandas()
        print(df.head())

if __name__ == "__main__":
    main()