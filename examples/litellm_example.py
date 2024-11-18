from typing import List
from pydantic import BaseModel, Field
from bespokelabs import curator

# Define response format using Pydantic
class Recipe(BaseModel):
    title: str = Field(description="Title of the recipe")
    ingredients: List[str] = Field(description="List of ingredients needed")
    instructions: List[str] = Field(description="Step by step cooking instructions")

def prompt_func(row=None):
    return "Generate a random recipe. Be creative but keep it realistic."

def parse_func(row, response):
    # Response is already a Recipe object thanks to response_format
    return {
        "title": response.title,
        "ingredients": response.ingredients,
        "instructions": response.instructions
    }

# Create prompter using LiteLLM backend
recipe_prompter = curator.Prompter(
    model_name="claude-3-opus-20240229",  # Could also use "anthropic/claude-2", "bedrock/claude", etc
    prompt_func=prompt_func,
    parse_func=parse_func,
    response_format=Recipe,
    backend="litellm",  # Specify LiteLLM backend
)

# Generate recipes
recipes = recipe_prompter()
print(recipes.to_pandas())