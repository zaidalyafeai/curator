# generation of code. generating test cases for the code and evaluating them on those test cases. 

from bespokelabs import curator
from pydantic import BaseModel, Field
from typing import List, Dict
from datasets import load_dataset

from bespokelabs.curator.experimental.code_execution_backend.e2b_backend import E2BCodeExecutionBackend

class CodeGenerationOutput(BaseModel):
    code: str = Field(..., description="The generated code")

class PromptConstants:
    SYSTEM_MESSAGE_GENERIC = f"You are an expert Python programmer. You will be given a question (problem specification) and will generate a correct Python program that matches the specification and passes all tests."
    FORMATTING_MESSAGE_WITH_STARTER_CODE = "You will use the following starter code to write the solution to the problem and enclose your code within delimiters."
    FORMATTING_WITHOUT_STARTER_CODE = "Read the inputs from stdin solve the problem and write the answer to stdout (do not directly test on the sample inputs). Enclose your code within delimiters as follows. Ensure that when the python program runs, it reads the inputs, runs the algorithm and writes output to STDOUT."



class SumCodeVerifier(curator.CodeVerifier):
    def __init__(self):
        super().__init__(config=curator.CodeVerifierConfig(backend="e2b", backend_params="python:3.12"))

    def verify(self, code):
        return self.execute(code) == 100
        
class CodeGenerationLLM(curator.LLM):
    response_format = CodeGenerationOutput
    verifier = SumCodeVerifier()

    def prompt(self, row: str):
        prompt = f"### Question:\n{row['question_content']}\n\n"
        if row['starter_code']:
            prompt += (
                f"### Format: {PromptConstants.FORMATTING_MESSAGE_WITH_STARTER_CODE}\n"
            )
            prompt += f"```python\n{row['starter_code']}\n```\n\n"
        else:
            prompt += f"### Format: {PromptConstants.FORMATTING_WITHOUT_STARTER_CODE}\n"
            prompt += "```python\n# YOUR CODE HERE\n```\n\n"
        return prompt


    def parse(self, row, response):
        row['generated_code'] = response.code
        return row
    
code_generator = CodeGenerationLLM(
    model_name="gpt-4o",
    generation_params={"temperature": 0.2, "max_tokens": 4096},
)

# problems_dataset = load_dataset("livecodebench/code_generation_lite", version_tag="release_v2",trust_remote_code=True)

problems_dataset = curator.LLM(["Generate two numbers whose sum is 100"] * 10)

solutions = code_generator(problems_dataset)

execution_backend = E2BCodeExecutionBackend()

for solution in solutions:
    execution_backend.execute()
