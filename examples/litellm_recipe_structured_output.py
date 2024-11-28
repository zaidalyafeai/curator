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


def main():
    # We define a prompter that generates cuisines
    cuisines_generator = curator.Prompter(
        prompt_func=lambda: f"Generate 10 diverse cuisines.",
        model_name="gpt-4o-mini",
        response_format=Cuisines,
        parse_func=lambda _, cuisines: [{"cuisine": t} for t in cuisines.cuisines_list],
        backend="litellm",
    )
    cuisines = cuisines_generator()
    print(cuisines.to_pandas())

    recipe_prompter = curator.Prompter(
        model_name="claude-3-5-haiku-20241022",
        prompt_func=lambda row: f"Generate a random {row['cuisine']} recipe. Be creative but keep it realistic.",
        parse_func=lambda row, response: {
            "title": response.title,
            "ingredients": response.ingredients,
            "instructions": response.instructions,
            "prep_time": response.prep_time,
            "cook_time": response.cook_time,
            "servings": response.servings,
            "cuisine": row["cuisine"],
        },
        response_format=Recipe,
        backend="litellm",
    )

    # Generate recipes for all cuisines
    recipes = recipe_prompter(cuisines)

    # Print results
    print(recipes.to_pandas())


if __name__ == "__main__":
    main()
