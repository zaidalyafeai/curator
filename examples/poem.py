from bespokelabs import curator
from datasets import Dataset
from pydantic import BaseModel, Field
from typing import List

# Create a dataset object for the topics you want to create the poems.
topics = Dataset.from_dict({"topic": [
    "Urban loneliness in a bustling city",
    "Beauty of Bespoke Labs's Curator library"
]})

# Define a class to encapsulate a list of poems.
class Poem(BaseModel):
    poem: str = Field(description="A poem.")

class Poems(BaseModel):
    poems_list: List[Poem] = Field(description="A list of poems.")


# We define a Prompter that generates poems which gets applied to the topics dataset.
poet = curator.Prompter(
    # The prompt_func takes a row of the dataset as input.
    # The row is a dictionary with a single key 'topic' in this case.
    prompt_func=lambda row: f"Write two poems about {row['topic']}.",
    model_name="gpt-4o-mini",
    response_format=Poems,
    # row is the input row, and poems is the Poems class which 
    # is parsed from the structured output from the LLM.
    parse_func=lambda row, poems: [
        {"topic": row["topic"], "poem": p.poem} for p in poems.poems_list
    ],
)

poem = poet(topics)
print(poem.to_pandas())
#                                       topic                                               poem
# 0       Urban loneliness in a bustling city  In the city's heart, where the sirens wail,\nA...
# 1       Urban loneliness in a bustling city  City streets hum with a bittersweet song,\nHor...
# 2  Beauty of Bespoke Labs's Curator library  In whispers of design and crafted grace,\nBesp...
# 3  Beauty of Bespoke Labs's Curator library  In the hushed breath of parchment and ink,\nBe...