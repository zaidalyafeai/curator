from typing import List
from pydantic import BaseModel, Field
from bespokelabs import curator
from datasets import Dataset

def prompt_func(cuisine_type):
    return f"Generate a random {cuisine_type} recipe. Be creative but keep it realistic."

def parse_func(row, response):
    return {
        "recipe": response,
        "cuisine": row["cuisine"]  # Keep track of cuisine type
    }

def main():
    # List of cuisines to generate recipes for
    cuisines = [{"cuisine": cuisine} for cuisine in ["Chinese", "Italian", "Mexican", "French", "Japanese", "Indian", "Thai", "Korean", "Vietnamese", "Brazilian"]]
    cuisines = Dataset.from_list(cuisines)

    # Create prompter using LiteLLM backend
    recipe_prompter = curator.Prompter(
        model_name="gpt-4o-mini",
        prompt_func=prompt_func,
        parse_func=parse_func,
        backend="litellm",
    )

    # Generate recipes for all cuisines
    recipes = recipe_prompter(cuisines)
    
    # Print results
    df = recipes.to_pandas()
    print(df)

if __name__ == "__main__":
    main()