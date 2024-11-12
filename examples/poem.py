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

# Key Components of Prompter

## prompt_func

1. Takes a dataset row as input
2. Can be a simple lambda if no dataset is used
3. Generates the prompt for the LLM

## parse_func

1. Takes two arguments:
    - Input row
    - LLM response (in response_format)
2. Returns a list of dictionaries
3. Converts LLM output into structured data


# Data Flow Example
Input Dataset: 
    Row A
    Row B
Processing by Prompter:
    Row A    →    prompt_func(A)    →    Response R1    →    parse_func(A, R1)    →    [C, D]
    Row B    →    prompt_func(B)    →    Response R2    →    parse_func(B, R2)    →    [E, F]

Output Dataset: 
    Row C
    Row D
    Row E
    Row F

In this example:

- Two input rows (A and B) are processed in parallel
- Each generates a response (R1 and R2)
- The parse function converts each response into multiple rows
- The final dataset contains all generated rows

Note that you can keep iterating on the dataset, by applying other prompters.
"""
from bespokelabs import curator
from datasets import Dataset
from pydantic import BaseModel, Field
from typing import List


# We use Pydantic and structured outputs to define the format of the response.

# This defines a list of topics, which is the response format for the topic generator.
class Topics(BaseModel):
    topics: List[str] = Field(description="A list of topics.")

# We define a prompter that generates topics.
topic_generator = curator.Prompter(
    prompt_func=lambda: f"Generate 10 diverse topics that are suitable for writing poems about.",
    model_name="gpt-4o-mini",
    response_format=Topics,
    parse_func=lambda _, topics_obj: [{"topic": t} for t in topics_obj.topics],
)

# We call the prompter to generate the dataset.
topics: Dataset = topic_generator()
print(topics['topic'])


# Define a list of poems.
class Poems(BaseModel):
    poems: List[str] = Field(description="A list of poems.")

# We define a prompter that generates poems which gets applied to the topics dataset.
poet = curator.Prompter(
    # The prompt_func takes a row of the dataset as input.
    # The row is a dictionary with a single key 'topic' in this case.
    prompt_func=lambda row: f"Write two poems about {row['topic']}.",
    model_name="gpt-4o-mini",
    response_format=Poems,
    # `row` is the input row, and `poems_obj` is the structured output from the LLM.
    parse_func=lambda row, poems_obj: [{"topic": row["topic"], "poem": p} for p in poems_obj.poems],
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

