from typing import Callable
from datasets import Dataset
from dataclasses import dataclass
from bespokelabs.curator.llm.prompt_formatter import GenericRequest
from bespokelabs.curator.experimental.types import CODE_REQUEST_TYPE


@dataclass
class CodeFormatter:
    function_name: Callable
    preprocess: Callable
    test_cases: Callable
    parse_results: Callable


    def create_code_execution_request(self, row: dict, idx: int) -> GenericRequest:
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
        code = self.preprocess(row)
        fn_name = self.function_name(code)
        test_case_list = self.test_cases(code)

        # if function name is None, request_type is 

        if fn_name is None:
            self.code_request_type = CODE_REQUEST_TYPE.standard_input
        else:
            self.code_request_type = CODE_REQUEST_TYPE.call_based

        base_code = self.base_imports()

        if self.code_request_type == CODE_REQUEST_TYPE.standard_input:
            add_code = self.synthesize_std_code(code)
        else:
            add_code = self.synthesize_cb_code(code)

        base_code += '\n' + add_code

        return base_code


    def base_imports(self):
        return "import sys\nimport time\nimport itertools\nfrom itertools import accumulate, product, permutations, combinations\nimport collections\nfrom collections import Counter, OrderedDict, deque, defaultdict, ChainMap\nfrom functools import lru_cache\nimport math\nfrom math import sqrt, sin, cos, tan, ceil, fabs, floor, gcd, exp, log, log2\nimport fractions\nfrom typing import List, Tuple\nimport numpy as np\nimport random\nimport heapq\nfrom heapq import *\n"


    def synthesize_cb_code(self, raw_code, debug=False):
        sol = "import sys\nimport time\nimport itertools\nfrom itertools import accumulate, product, permutations, combinations\nimport collections\nfrom collections import Counter, OrderedDict, deque, defaultdict, ChainMap\nfrom functools import lru_cache\nimport math\nfrom math import sqrt, sin, cos, tan, ceil, fabs, floor, gcd, exp, log, log2\nimport fractions\nfrom typing import List, Tuple\nimport numpy as np\nimport random\nimport heapq\nfrom heapq import *\n"
        sol += raw_code
        return sol

    def synthesize_std_code(self, raw_code, debug=False):
        normal_import_lines = "import sys\nimport time\nimport itertools\nfrom itertools import accumulate, product, permutations, combinations\nimport collections\nfrom collections import Counter, OrderedDict, deque, defaultdict, ChainMap\nfrom functools import lru_cache\nimport math\nfrom math import sqrt, sin, cos, tan, ceil, fabs, floor, gcd, exp, log, log2\nimport fractions\nfrom typing import List, Tuple\nimport numpy as np\nimport random\nimport heapq\nfrom heapq import *\n"
        
        sol2 = "" # code for execute

        tmp_test = raw_code.split("\n")
        # define the code line type, 1 for import lines, 2 for import * lines with indent, 0 for normal codes
        code_types = [] 


        for x in tmp_test:
            if 'import *' in x:
                code_types.append(2)
            elif x.startswith("from ") or x.startswith("import "):
                code_types.append(1) 
            else:
                code_types.append(0)
        
        started = False

        special_import_lines = [i.lstrip('\t') for idx, i in enumerate(tmp_test) if code_types[idx]==2]
        special_import_lines = '\n'.join(special_import_lines)

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
                        sol += '\t'
                    sol += f"{i}\n"

        return sol + '\n' + sol2
