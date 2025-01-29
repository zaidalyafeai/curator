import json
import re

from datasets import load_dataset

from bespokelabs import curator
from bespokelabs.curator.experimental.code_executor.code_executor import TestCase

dataset = load_dataset("bespokelabs/sky-t1-taco-test-rejection-sampled-shreyas")


class APPSCodeExecutor(curator.experimental.CodeExecutor):
    """APPS Code Executor."""

    def code_string(self, row):
        """Extract code string from a dataset row."""
        try:
            code = re.search(r"```python\n(.*?)\n```", row["deepseek_solution"], re.DOTALL).group(1)
        except (AttributeError, IndexError):
            code = ""
        return code

    def test_cases(self, row) -> list[TestCase]:
        """Extract test cases from a dataset row."""
        test_cases = []
        inputs_outputs = row["input_output_x"]
        try:
            inputs_outputs = json.loads(inputs_outputs)
            inputs = inputs_outputs["inputs"]
            outputs = inputs_outputs["outputs"]

            for input, output in zip(inputs, outputs):
                test_cases.append(TestCase(input=input, expected_output=output))

        except Exception as e:
            print("Error parsing input output", e)

        return test_cases

    def parse_results(self, row, test_cases, execution_results):
        """Parse execution results."""
        row["correct"] = True
        for test_case, result in zip(test_cases, execution_results):
            if result.response_stdout != test_case.expected_output:
                row["correct"] = False
                break

        return row


if __name__ == "__main__":
    executor = APPSCodeExecutor(backend="multiprocessing")
    execution_output = executor(dataset["train"].select(range(1, 2)))

    print("================")
    print(execution_output)

    print(execution_output[0])
    print("================")
