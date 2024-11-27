from typing import List
from pydantic import BaseModel, Field
from bespokelabs import curator
from datasets import Dataset
import logging

logger = logging.getLogger(__name__)

# Define response format using Pydantic
class Recipe(BaseModel):
    title: str = Field(description="Title of the recipe")
    ingredients: List[str] = Field(description="List of ingredients needed")
    instructions: List[str] = Field(description="Step by step cooking instructions")
    prep_time: int = Field(description="Preparation time in minutes")
    cook_time: int = Field(description="Cooking time in minutes")
    servings: int = Field(description="Number of servings")    

class Cuisines(BaseModel):
    cuisines_list: List[str] = Field(description="A list of cuisines.")

def prompt_func(row):
    return f"Generate a random {row['cuisine']} recipe. Be creative but keep it realistic."

def parse_func(row, response):
    return {
        "title": response.title,
        "ingredients": response.ingredients,
        "instructions": response.instructions,
        "cuisine": row["cuisine"]  # Keep track of cuisine type
    }

def main():
    # List of cuisines to generate recipes for
    # We define a prompter that generates topics.
    cuisines_generator = curator.Prompter(
        prompt_func=lambda: f"Generate 10 diverse cuisines.",
        model_name="gpt-4o-mini",
        response_format=Cuisines,
        parse_func=lambda _, cuisines: [{"cuisine": t} for t in cuisines.cuisines_list],
        backend="litellm",
    )
    cuisines = cuisines_generator()
    print(cuisines)
    
    recipe_prompter = curator.Prompter(
            model_name="together_ai/mistralai/Mixtral-8x7B-Instruct-v0.1",
            prompt_func=prompt_func,
            parse_func=parse_func,
            response_format=Recipe,
        backend="litellm",
    )

    # Generate recipes for all cuisines
    recipes = recipe_prompter(cuisines)
    
    # Print results
    df = recipes.to_pandas()
    print(df)

if __name__ == "__main__":
    main()