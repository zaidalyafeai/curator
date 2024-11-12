"""Example of using the curator library to generate diverse poems.

We generate 10 diverse topics and then generate 2 poems for each topic.
You can do this in a loop, but that is inefficient.
And you will need to add a lot of boilerplate code for calling the LLM,
and error handling etc.
When you need to do this thousands of times (if not more), you need better
abstractions.

Curator provides a prompter class that takes care of a lot of the heavy lifting.

The mental model to use for prompter is that it calls LLM 
on each row of a given HF dataset in parallel.
So the prompt_func should take a row of the dataset as input.
But if there is no dataset, then the prompt_func can just be a
lambda without any arguments.

The parse_func is a function that takes the response from
the LLM (which is of type response_format), and converts it
into a list of dictionaries. 
These then get converted into a HF dataset by the library.

For example, if the input HF dataset has two rows, A and B,
then prompt_func(A) and prompt_func(B) are called in parallel.
The outputs can be structured outputs R1 and R2.
The parse_func expects two arguments: 
the input row (A or B), and the structured output (R1 or R2).

It should return a list of dictionaries.
These then get converted into a HF dataset by the library.
For example, if the parse_func returns C and D for A after parsing R1,
and E and F for B after parsing R2,
then the output HF dataset has four rows: C, D, E, and F.

Note that you can keep iterating on the dataset, by applying other prompters.
"""
from bespokelabs import curator
from datasets import Dataset
from pydantic import BaseModel, Field
from typing import List


# We use Pydantic and structured outputs to define the format of the response.
# This defines a single topic.
class Topic(BaseModel):
    topic: str = Field(description="A topic.")

# This defines a list of topics, which is the response format for the topic generator.
class Topics(BaseModel):
    topics: List[Topic] = Field(description="A list of topics.")

# We define a prompter that generates topics.
topic_generator = curator.Prompter(
    prompt_func=lambda: f"Generate 10 diverse topics that are suitable for writing poems about.",
    model_name="gpt-4o-mini",
    response_format=Topics,
    parse_func=lambda _, topics_obj: [{"topic": t.topic} for t in topics_obj.topics],
)

# We call the prompter to generate the dataset.
topics: Dataset = topic_generator()
print(topics['topic'])

# We define a poem.
class Poem(BaseModel):
    poem: str = Field(description="A poem.")

# And a list of poems.
class Poems(BaseModel):
    poems: List[Poem] = Field(description="A list of poems.")

# We define a prompter that generates poems which gets applied to the topics dataset.
poet = curator.Prompter(
    # The prompt_func takes a row of the dataset as input.
    # The row is a dictionary with a single key 'topic' in this case.
    prompt_func=lambda row: f"Write two poems about {row['topic']}.",
    model_name="gpt-4o-mini",
    response_format=Poems,
    # `row` is the input row, and `poems_obj` is the structured output from the LLM.
    parse_func=lambda row, poems_obj: [{"topic": row["topic"], "poem": p.poem} for p in poems_obj.poems],
)

# We apply the prompter to the topics dataset.
poems = poet(topics)
print(poems.to_pandas())

# Expected output:
#                                           topic                                               poem
# 0                            Dreams vs. reality  In the realm where dreams take flight,\nWhere ...
# 1                            Dreams vs. reality  Reality stands with open eyes,\nA weighty thro...
# 2           Urban loneliness in a bustling city  In the city's heart where shadows blend,\nAmon...
# 3           Urban loneliness in a bustling city  Among the crowds, I walk alone,\nA sea of face...

