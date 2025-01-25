from datasets import load_dataset
import json
from bespokelabs import curator
from bespokelabs.curator.experimental.code_executor.code_executor import TestCase
import re

dataset = load_dataset("bespokelabs/sky-t1-taco-test-rejection-sampled-shreyas")

class APPSCodeExecutor(curator.experimental.CodeExecutor):
    
    def code_string(self, row):
        import re
        try:
            code = re.search(r"```python\n(.*?)\n```", row["deepseek_solution"], re.DOTALL).group(1)
        except (AttributeError, IndexError) as e:
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
            inputs = ios['inputs']
            outputs = ios['outputs']
            
            for input, output in zip(inputs, outputs):
                test_cases.append(TestCase(input=input, expected_output=output))

        except Exception as e:
            print("Error parsing input output", e)

        return test_cases

    def parse_results(self, row, test_cases, execution_results):
        row['correct'] = True
        for test_case, result in zip(test_cases, execution_results):
            if result.response_stdout != test_case.expected_output:
                row['correct'] = False
                break

        return row

executor = APPSCodeExecutor()

execution_output = executor(dataset["train"].select(range(10)))

import pdb; pdb.set_trace()