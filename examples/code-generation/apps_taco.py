from datasets import load_dataset

from bespokelabs import curator
from bespokelabs.curator.experimental.code_executor.code_executor import TestCase

dataset = load_dataset("bespokelabs/sky-t1-taco-test-rejection-sampled-shreyas")

dataset["train"][0]


class APPSCodeExecutor(curator.experimental.CodeExecutor):
    def function_name(self, row):
        return row["function_name"]

    def function_code(self, row):
        import re

        code = re.search(r"```python\n(.*)\n```", row["deepseek_solution"]).group(1)
        return code

    def test_cases(self, row) -> list[TestCase]:
        import pdb

        pdb.set_trace()
        return [TestCase(input=row["input_output"][0], expected_output=row["input_output"][1])]

    def parse_results(self, row, test_cases, execution_results):
        return row["test_results"]


executor = APPSCodeExecutor(backend="e2b")

execution_output = executor(dataset["train"].select(range(10)))
