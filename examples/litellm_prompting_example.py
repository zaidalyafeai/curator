from typing import List
from pydantic import BaseModel, Field
from bespokelabs import curator
from datasets import Dataset

# Define response format using Pydantic
class Recipe(BaseModel):
    title: str = Field(description="Title of the recipe")
    ingredients: List[str] = Field(description="List of ingredients needed")
    instructions: List[str] = Field(description="Step by step cooking instructions")

def prompt_func(cuisine_type):
    return f"Generate a random {cuisine_type} recipe. Be creative but keep it realistic."

def parse_func(row, response):
    return {
        "response": response,
        "cuisine": row["cuisine"]  # Keep track of cuisine type
    }

def main():
    # List of cuisines to generate recipes for
    cuisines = ["Chinese"] * 100
    
    # Create input dataset with cuisine types
    input_data = [{"cuisine": cuisine} for cuisine in cuisines]
    input_dataset = Dataset.from_list(input_data)

    # Create prompter using LiteLLM backend
    recipe_prompter = curator.Prompter(
        # model_name="gpt-4o-mini",
        model_name="together_ai/mistralai/Mistral-7B-Instruct-v0.1",
        prompt_func=prompt_func,
        parse_func=parse_func,
        backend="litellm",
    )

    # Generate recipes for all cuisines
    recipes = recipe_prompter(input_dataset)
    
    # Print results
    df = recipes.to_pandas()
    print(df)

if __name__ == "__main__":
    main()