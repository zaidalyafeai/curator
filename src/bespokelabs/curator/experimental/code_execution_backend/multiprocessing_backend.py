"""Multiprocessing Code Execution Backend"""

import os
import aiohttp
import signal
import sys
import time
import subprocess
import tempfile
from pyext import RuntimeModule

from bespokelabs.curator.experimental.code_execution_backend.base_backend import BaseCodeExecutionBackend
from bespokelabs.curator.experimental.types import CodeAPIRequest, CodeExecutionResponse, CodeExecutionStatusTracker, CodeTestCaseResponse

class MultiprocessingCodeExecutionBackend(BaseCodeExecutionBackend):
    def __init__(self, config):
        super().__init__(config)
        self.config = config

    @property
    def max_requests_per_minute(self):
        return 10000

    async def compile_and_get_function(
        self,
        program: str,
        which_type: str,
        method_name: str,
        timeout: int,
    ):
        original_handler = signal.getsignal(signal.SIGALRM)
        try:
            signal.alarm(timeout)
            tmp_sol = RuntimeModule.from_string("tmp_sol", "", program)
            if which_type == "call_based" and "class Solution" in program:
                tmp = tmp_sol.Solution()
            else:
                tmp = tmp_sol
            signal.alarm(0)
        except Exception as e:
            signal.alarm(0)
            print(f"compilation error = {e}")
            return False
        finally:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, original_handler)
        
        if which_type == "call_based":
            assert isinstance(method_name, str)
        else:
            method_name = "code"
        
        try:
            signal.alarm(timeout)
            method = getattr(tmp, method_name)  # get_attr second arg must be str
            signal.alarm(0)
        except:
            signal.alarm(0)
            e = sys.exc_info()
            print(f"unable to get function error = {e}")
            return False
        return method

    async def execute_request(
        self, request: CodeAPIRequest, session: aiohttp.ClientSession, status_tracker: CodeExecutionStatusTracker
    ) -> CodeExecutionResponse:
        """Execute a single request."""

        # Disable functionalities that can make destructive changes to the test.
        # self.reliability_guard()

        generic_request = request.generic_request
        method = await self.compile_and_get_function(generic_request.code, generic_request.request_type, generic_request.function_name, generic_request.execution_params.timeout)

        if method is False:
            
            return CodeExecutionResponse(
                responses=[CodeTestCaseResponse(
                    response_message="Compilation error",
                    response_errors=["Compilation error"],
                    response_stdout="",
                    response_stderr="",
                )],
                code_api_request=request,
            )
        
        if generic_request.request_type == "call_based":
            results = await self.execute_call_based_request(method, generic_request.test_cases, generic_request.execution_params.timeout)
        else:
            results = await self.execute_standard_input_request(method, generic_request.code + "\ncode()", generic_request.test_cases, generic_request.function_name, generic_request.execution_params.timeout)

        final_results = []
        for test_case_idx, result in results.items():
            final_results.append(CodeTestCaseResponse(
                response_message=result['status'],
                response_errors=None,
                response_stdout=result['output'],
                response_stderr=result['error'],
            ))

        return CodeExecutionResponse(
            responses=final_results,
            code_api_request=request,
        )

    def create_temp_file(self, content: str) -> str:
        """Create a temporary file with the given content.
        
        Args:
            content: Content to write to temp file
            
        Returns:
            Path to the created temp file
        """
        with tempfile.NamedTemporaryFile(delete=False, mode='w', encoding='utf-8') as temp_file:
            temp_file.write(content)
            return temp_file.name

    async def execute_standard_input_request(self, method_func, code: str, test_cases: list, function_name: str, 
                                 timeout: int, early_stop: bool = True) -> dict:
        """Execute code with function calls and test cases.
        
        Args:
            method_func: Compiled method to execute
            code: Source code
            test_cases: List of test cases to run
            function_name: Name of function to call
            timeout: Execution timeout in seconds
            early_stop: Whether to stop on first failure
            
        Returns:
            Dict containing execution results
        """

        temp_program_path = None
        try:
            temp_program_path = self.create_temp_file(code)
            exec_results = {}
            
            for test_case_idx, test_case in enumerate(test_cases):
                input_data = test_case.input
                expected = test_case.expected_output
                
                if isinstance(input_data, list):
                    input_data = "\n".join(input_data)
                if isinstance(expected, list):
                    expected = "\n".join(expected)
                    
                try:
                    result = subprocess.run(['python', temp_program_path], 
                                          input=input_data,
                                          text=True,
                                          capture_output=True,
                                          timeout=timeout)

                    # print("Done")
                    exec_results[test_case_idx] = {
                        'status': 'success',
                        'output': result.stdout,
                        'error': result.stderr
                    }
                    
                except subprocess.TimeoutExpired:
                    exec_results[test_case_idx] = {
                        'status': 'timeout',
                        'output': None,
                        'error': f'Execution timed out after {timeout}s'
                    }
                    if early_stop:
                        break
                    
                except Exception as e:
                    exec_results[test_case_idx] = {
                        'status': 'error', 
                        'output': None,
                        'error': str(e)
                    }
                    if early_stop:
                        break

            return exec_results
        finally:
            if temp_program_path:
                try:
                    os.unlink(temp_program_path)
                except:
                    pass
            
    async def execute_call_based_request(self, method, inputs_list, timeout, early_stop=True):
        original_handler = signal.getsignal(signal.SIGALRM)
        try:
            self.reliability_guard()
            results = {}

            for index, inputs in enumerate(inputs_list):
                try:
                    signal.alarm(timeout)
                    exec_outputs = method(*inputs)
                    signal.alarm(0)
                    results[index] = {
                        'status': 'success',
                        'output': str(exec_outputs),
                        'error': None
                    }
                except Exception as e:
                    signal.alarm(0)
                    if early_stop:
                        break
                    results[index] = {
                        'status': 'error',
                        'error': str(e)
                    }
            return results
        finally:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, original_handler)