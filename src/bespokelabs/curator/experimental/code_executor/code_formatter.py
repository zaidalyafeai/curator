from dataclasses import dataclass
from typing import Callable

from bespokelabs.curator.experimental.types import CodeExecutionRequest, CodeExecutionRequestParams, CodeExecutionResponse
from bespokelabs.curator.llm.prompt_formatter import GenericRequest


@dataclass
class CodeFormatter:
    function_name: Callable
    code_string: Callable
    test_cases: Callable
    parse_results: Callable
    execution_params: CodeExecutionRequestParams

    def create_code_execution_request(self, row: dict, idx: int) -> CodeExecutionRequest:
        """Format the request object based off of `LLM` attributes.

        Args:
            row: Input data to format into a prompt
            idx: Index of the row in the dataset

        Returns:
            GenericRequest object containing the formatted request

        Raises:
            ValueError: If prompt_func has invalid number of arguments or returns invalid format
        """
        # Convert BaseModel to dict for serialization
        code = self.code_string(row)
        fn_name = self.function_name(row)
        test_case_list = self.test_cases(row)

        # if function name is None, request_type is

        if fn_name is None:
            self.code_request_type = "standard_input"
        else:
            self.code_request_type = "call_based"

        base_code = self.base_imports()

        if self.code_request_type == "standard_input":
            add_code = self.synthesize_std_code(code)
        else:
            add_code = self.synthesize_cb_code(code)

        base_code += "\n" + add_code

        return CodeExecutionRequest(
            code=base_code,
            test_cases=test_case_list,
            function_name=fn_name,
            request_type=self.code_request_type,
            execution_params=self.execution_params,
            original_row=row,
            original_row_idx=idx,
        )

    def base_imports(self):
        return "import sys\nimport time\nimport itertools\nfrom itertools import accumulate, product, permutations, combinations\nimport collections\nfrom collections import Counter, OrderedDict, deque, defaultdict, ChainMap\nfrom functools import lru_cache\nimport math\nfrom math import sqrt, sin, cos, tan, ceil, fabs, floor, gcd, exp, log, log2\nimport fractions\nfrom typing import List, Tuple\nimport numpy as np\nimport random\nimport heapq\nfrom heapq import *\n"

    def synthesize_cb_code(self, raw_code, debug=False):
        sol = "import sys\nimport time\nimport itertools\nfrom itertools import accumulate, product, permutations, combinations\nimport collections\nfrom collections import Counter, OrderedDict, deque, defaultdict, ChainMap\nfrom functools import lru_cache\nimport math\nfrom math import sqrt, sin, cos, tan, ceil, fabs, floor, gcd, exp, log, log2\nimport fractions\nfrom typing import List, Tuple\nimport numpy as np\nimport random\nimport heapq\nfrom heapq import *\n"
        sol += raw_code

        # check if last line is a function call
        # doesn't work
        # if re.search(r"\w+\(\)", raw_code.split("\n")[-1]):
        #     sol = raw_code.split("\n")[:-1]
        #     sol = "\n".join(sol)

        return sol

    def synthesize_std_code(self, raw_code, debug=False):
        normal_import_lines = "import sys\nimport time\nimport itertools\nfrom itertools import accumulate, product, permutations, combinations\nimport collections\nfrom collections import Counter, OrderedDict, deque, defaultdict, ChainMap\nfrom functools import lru_cache\nimport math\nfrom math import sqrt, sin, cos, tan, ceil, fabs, floor, gcd, exp, log, log2\nimport fractions\nfrom typing import List, Tuple\nimport numpy as np\nimport random\nimport heapq\nfrom heapq import *\n"

        sol = ""  # code for compile
        sol2 = ""  # code for execute

        tmp_test = raw_code.split("\n")
        # define the code line type, 1 for import lines, 2 for import * lines with indent, 0 for normal codes
        code_types = []

        for x in tmp_test:
            if "import *" in x:
                code_types.append(2)
            elif x.startswith("from ") or x.startswith("import "):
                code_types.append(1)
            else:
                code_types.append(0)

        started = False

        special_import_lines = [i.lstrip("\t") for idx, i in enumerate(tmp_test) if code_types[idx] == 2]
        special_import_lines = "\n".join(special_import_lines)

        for idx, i in enumerate(tmp_test):
            code_type = code_types[idx]
            if code_type == 0 and not started:
                sol2 += normal_import_lines
                sol2 += "\nstdin = sys.stdin\nstdout = sys.stdout\n"
                sol2 += f"{i}\n"

                sol += normal_import_lines
                sol += special_import_lines
                sol += "\nstdin = sys.stdin\nstdout = sys.stdout\n"
                sol += "def code():\n"
                sol += f"\t{i}\n"
                started = True
            else:
                sol2 += f"{i}\n"
                if code_type < 2:
                    if started:
                        sol += "\t"
                    sol += f"{i}\n"

        return sol

    def response_to_response_format(self, response: CodeExecutionResponse):
        # Convert responses to proper dictionary format
        return {
            "responses": [
                {
                    "response_message": r.response_message,
                    "response_errors": r.response_errors,
                    "response_stdout": r.response_stdout,
                    "response_stderr": r.response_stderr,
                }
                for r in response.responses
            ],
            "code_api_request": response.code_api_request.model_dump(),
        }
