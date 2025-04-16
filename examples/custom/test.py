from bespokelabs import curator
from datasets import load_dataset
from pydantic import BaseModel

class Score(BaseModel):
    score: int
    reasoning: str

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
        try:
            return {
                "score": response.score,
                "reasoning": response.reasoning,
            }
        except Exception as e:
            return {
                "score": response.split("Educational score:")[1].split("Reasoning:")[0].strip(),
                "reasoning": response.split("Reasoning:")[1].strip(),
            }

fw  = load_dataset("json", data_files="fineweb-2-arb-Arab-1000.json")["train"]
fw = fw.select(range(100))
llm = Prompter(
        model_name="Qwen/Qwen2.5-7B-Instruct",
        backend="openai",
        backend_params={
            "base_url": "http://localhost:8787/v1",
            "api_key": "EMPTY",
            "max_requests_per_minute": 30,
            "max_retries": 1,
            },
        #response_format = Score
        )

llm(fw)
