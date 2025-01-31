import json
import re

from datasets import load_dataset

from bespokelabs import curator
from bespokelabs.curator.experimental.code_executor.code_executor import TestCase


class APPSCodeExecutor(curator.experimental.CodeExecutor):
    """APPS Code Executor."""

    def code(self, row):
        """Extract code string from a dataset row."""
        try:
            code = re.search(r"```python\n(.*?)\n```", row["deepseek_solution"], re.DOTALL).group(1)
        except (AttributeError, IndexError):
            code = ""
        return code

    def input(self, row):
        """Extract input from a dataset row."""
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

        return test_cases[:5]

    def output(self, row, test_cases, execution_results):
        """Parse execution results."""
        row["correct"] = True

        if len(test_cases) != len(execution_results):
            row["correct"] = False
            return row

        for test_case, result in zip(test_cases, execution_results):
            if result.response_stdout != test_case.expected_output:
                print("================")
                print("result.response_stdout[:20]", result.response_stdout[:20])
                print("test_case.expected_output[:20]", test_case.expected_output[:20])
                print("================")
                row["correct"] = False
                break

        return row


if __name__ == "__main__":
    executor = APPSCodeExecutor(backend="ray")
    dataset = load_dataset("bespokelabs/sky-t1-taco-test-rejection-sampled-shreyas")
    execution_output = executor(dataset["train"].select(range(1, 100)))

    print("================")
    print(execution_output)

    print(execution_output["correct"])
    print("================")
