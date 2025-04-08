import os
from datasets import Dataset, load_dataset
from bespokelabs import curator
from dotenv import load_dotenv
import matplotlib.pyplot as plt
from pydantic import BaseModel

class Score(BaseModel):
    score: int
    reasoning: str

load_dotenv("../../.env")

prompt = """Below is an extract from a web page. Evaluate whether the page has a high educational
value and could be useful in an educational setting for teaching from primary school to
grade school levels using the additive 5-point scoring system described below. Points are
accumulated based on the satisfaction of each criterion:
- Add 1 point if the extract provides some basic information relevant to educational top-
ics, even if it includes some irrelevant or non-academic content like advertisements and
promotional material.
- Add another point if the extract addresses certain elements pertinent to education but
does not align closely with educational standards. It might mix educational content with
non-educational material, offering a superficial overview of potentially useful topics, or
presenting information in a disorganized manner and incoherent writing style.
- Award a third point if the extract is appropriate for educational use and introduces key
concepts relevant to school curricula. It is coherent though it may not be comprehensive
or could include some extraneous information. It may resemble an introductory section of
a textbook or a basic tutorial that is suitable for learning but has notable limitations like
treating concepts that are too complex for grade school students.
- Grant a fourth point if the extract highly relevant and beneficial for educational purposes
for a level not higher than grade school, exhibiting a clear and consistent writing style. It
could be similar to a chapter from a textbook or a tutorial, offering substantial educational
content, including exercises and solutions, with minimal irrelevant information, and the
concepts aren’t too advanced for grade school students. The content is coherent, focused,
and valuable for structured learning.
- Bestow a fifth point if the extract is outstanding in its educational value, perfectly suited for
teaching either at primary school or grade school. It follows detailed reasoning, the writing
style is easy to follow and offers profound and thorough insights into the subject matter,
devoid of any non-educational or complex content.
The extract: <EXAMPLE>.
After examining the extract:
- Briefly reason your total score, up to 100 words "Reasoning: <reasoning>".
- Conclude with the score using the format: "Educational score: <total points>"
"""
class Prompter(curator.LLM):
    """A recipe generator that generates recipes for different cuisines."""

    def prompt(self, input: dict) -> str:
        """Generate a prompt for the recipe generator."""
        return f"{prompt}\nThe extract: {input['text']}"

    def parse(self, input: dict, response: str) -> dict:
        """Parse the model response along with the input to the model into the desired output format.."""        
        return {
            "score": response.score,
            "reasoning": response.reasoning,
        }


def main():
    fw = load_dataset("HuggingFaceFW/fineweb-2", name="arb_Arab", split="train", streaming=True).take(100)

    model_name = "qwen/qwen-2.5-72b-instruct"
    model_name = f"openrouter/{model_name}"
    llm = Prompter(
        model_name=model_name,
        backend_params={
            "max_requests_per_minute": 100,
            "max_tokens_per_minute": 10000000,
            "request_timeout": 30 * 60,
        },
        response_format=Score,
    )

    responses = llm(fw)

    # convert to pandas
    df = responses.to_pandas()

    # plot by score
    plt.hist(df["score"], bins=5)
    plt.savefig("score.png")


if __name__ == "__main__":
    main()
