import json
import re

from datasets import load_dataset

from bespokelabs import curator
from bespokelabs.curator.experimental.code_executor.code_executor import TestCase

dataset = load_dataset("bespokelabs/sky-t1-taco-test-rejection-sampled-shreyas")


class APPSCodeExecutor(curator.experimental.CodeExecutor):
    def code_string(self, row):
        import re

        try:
            code = re.search(r"```python\n(.*?)\n```", row["deepseek_solution"], re.DOTALL).group(1)
        except (AttributeError, IndexError):
            code = ""
        return code

    def function_name(self, row):
        code = self.code_string(row)
        # import re
        # match = re.search(r"def\s+(\w+)\(", code)
        # if match is None:
        #     return None
        # fn_name = match.group(1)
        # return fn_name
        # check if a function is being called
        if re.search(r"\w+\(\)", code.split("\n")[-1]):
            return None

        # and it is not in __main__
        if "__main__" in code:
            return None

        # else return the function name
        match = re.search(r"def\s+(\w+)\(", code)
        if match is None:
            return None
        fn_name = match.group(1)
        return fn_name

    def test_cases(self, row) -> list[TestCase]:
        test_cases = []
        ios = row["input_output_x"]
        try:
            ios = json.loads(ios)
            inputs = ios["inputs"]
            outputs = ios["outputs"]

            for input, output in zip(inputs, outputs):
                test_cases.append(TestCase(input=input, expected_output=output))

        except Exception as e:
            print("Error parsing input output", e)

        return test_cases

    def parse_results(self, row, test_cases, execution_results):
        
        # row['function_name'] = self.function_name(row)
        row["correct"] = True
        
        # row['execution_results'] = execution_results

        if len(test_cases) != len(execution_results):
            row['correct'] = False
            return row

        for test_case, result in zip(test_cases, execution_results):
            if type(result.response_stdout) == str:
                result.response_stdout = [result.response_stdout]

            if result.response_stdout != test_case.expected_output:
                row["correct"] = False
                break

        return row


executor = APPSCodeExecutor()


# ex_st = "To solve this problem, we need to convert a given number into a specified base, which can be a non-integer like"

data = dataset['train'].select(range(300))

# select only the row where deepseek_solution contains ex_st
# data = data.filter(lambda x: ex_st in x['deepseek_solution'])

import pdb; pdb.set_trace()
execution_output = executor(data)

import pdb

pdb.set_trace()
