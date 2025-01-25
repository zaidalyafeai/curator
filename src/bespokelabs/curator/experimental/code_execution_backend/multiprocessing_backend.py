"""Multiprocessing Code Execution Backend"""

import os
import signal
import subprocess
import sys
import tempfile
import logging

import asyncio
import aiohttp
from pyext import RuntimeModule
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool

from bespokelabs.curator.experimental.code_execution_backend.base_backend import BaseCodeExecutionBackend
from bespokelabs.curator.experimental.types import CodeAPIRequest, CodeExecutionResponse, CodeExecutionStatusTracker, CodeTestCaseResponse


class MultiprocessingCodeExecutionBackend(BaseCodeExecutionBackend):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.max_retries = 3
        self.retry_delay = 1
        self._create_process_pool()

    def _create_process_pool(self):
        """Create a new process pool."""
        if hasattr(self, 'process_pool'):
            self.process_pool.shutdown(wait=False)
        self.process_pool = ProcessPoolExecutor(max_workers=100)

    @property
    def max_requests_per_minute(self):
        return 10000

    async def execute_request(
        self, request: CodeAPIRequest
    ) -> CodeExecutionResponse:
        """Execute a single request with retry logic."""
        simple_request = {
            'code': request.generic_request.code,
            'request_type': request.generic_request.request_type,
            'function_name': request.generic_request.function_name,
            'timeout': request.generic_request.execution_params.timeout,
            'test_cases': [
                {'input': tc.input, 'expected': tc.expected_output} 
                for tc in request.generic_request.test_cases
            ]
        }
        
        attempts = 0
        last_error = None
        
        while attempts < self.max_retries:
            try:
                # Execute with simplified data
                future = self.process_pool.submit(self._execute_request, simple_request)
                results = await asyncio.get_event_loop().run_in_executor(
                    None,
                    future.result
                )
                break  # Success - exit retry loop
                
            except (BrokenProcessPool, Exception) as e:
                attempts += 1
                last_error = str(e)
                logging.warning(f"Process pool failed (attempt {attempts}/{self.max_retries}): {last_error}")
                
                # Always recreate the pool on failure
                self._create_process_pool()
                
                if attempts >= self.max_retries:
                    logging.error(f"Max retries reached, last error: {last_error}")
                    # Return error response instead of raising
                    return CodeExecutionResponse(
                        responses=[
                            CodeTestCaseResponse(
                                response_message="error",
                                response_errors=[f"Process pool failed after {self.max_retries} attempts: {last_error}"],
                                response_stdout="",
                                response_stderr=""
                            )
                        ],
                        code_api_request=request
                    )
                
                await asyncio.sleep(self.retry_delay)

        # Convert back to response objects
        final_results = []
        for result in results:
            final_results.append(
                CodeTestCaseResponse(
                    response_message=result['status'],
                    response_errors=result.get('errors'),
                    response_stdout=result.get('stdout', ''),
                    response_stderr=result.get('stderr', '')
                )
            )
            
        return CodeExecutionResponse(
            responses=final_results,
            code_api_request=request
        )

    @staticmethod
    def _execute_request(simple_request: dict) -> list:
        """Execute request with simplified data structures.
        
        Args:
            simple_request: Dict containing:
                - code: str
                - request_type: str 
                - function_name: str
                - timeout: int
                - test_cases: list[dict]
                
        Returns:
            List of result dicts
        """
        method = MultiprocessingCodeExecutionBackend.compile_and_get_function(
            simple_request['code'],
            simple_request['request_type'],
            simple_request['function_name'], 
            simple_request['timeout']
        )

        if method is False:
            return [{
                'status': 'error',
                'errors': ['Compilation error'],
                'stdout': '',
                'stderr': ''
            }]

        if simple_request['request_type'] == 'call_based':
            results = MultiprocessingCodeExecutionBackend.execute_call_based_request(
                method,
                simple_request['test_cases'],
                simple_request['timeout']
            )
        else:
            results = MultiprocessingCodeExecutionBackend.execute_standard_input_request(
                method,
                simple_request['code'] + '\ncode()',
                simple_request['test_cases'],
                simple_request['function_name'],
                simple_request['timeout']
            )

        return [
            {
                'status': r['status'],
                'stdout': r['output'],
                'stderr': r['error']
            }
            for r in results.values()
        ]

    @staticmethod
    def compile_and_get_function(
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

    @staticmethod
    def execute_standard_input_request(method_func, code: str, test_cases: list, function_name: str, timeout: int, early_stop: bool = True) -> dict:
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
            temp_program_path = MultiprocessingCodeExecutionBackend.create_temp_file(code)
            exec_results = {}

            for test_case_idx, test_case in enumerate(test_cases):
                input_data = test_case['input']
                expected = test_case['expected']

                if isinstance(input_data, list):
                    input_data = "\n".join(input_data)
                if isinstance(expected, list):
                    expected = "\n".join(expected)

                try:
                    result = subprocess.run(["python", temp_program_path], input=input_data, text=True, capture_output=True, timeout=timeout)
                    # print("Done")
                    exec_results[test_case_idx] = {"status": "success", "output": result.stdout, "error": result.stderr}

                    if result.stdout != expected:
                        exec_results[test_case_idx] = {"status": "success", "output": result.stdout, "error": result.stderr}
                        if early_stop:
                            break

                except subprocess.TimeoutExpired:
                    exec_results[test_case_idx] = {"status": "timeout", "output": None, "error": f"Execution timed out after {timeout}s"}
                    if early_stop:
                        break

                except Exception as e:
                    exec_results[test_case_idx] = {"status": "error", "output": None, "error": str(e)}
                    if early_stop:
                        break


            return exec_results
        finally:
            if temp_program_path:
                try:
                    os.unlink(temp_program_path)
                except:
                    pass

    @staticmethod
    def execute_call_based_request(method, inputs_list, timeout, early_stop=True):
        original_handler = signal.getsignal(signal.SIGALRM)
        try:
            results = {}

            for index, inputs in enumerate(inputs_list):
                try:
                    signal.alarm(timeout)
                    exec_outputs = method(*inputs['input'])

                    if isinstance(exec_outputs, tuple):
                        exec_outputs = list(exec_outputs)

                    signal.alarm(0)
                    results[index] = {"status": "success", "output": str(exec_outputs), "error": None}
                except Exception as e:
                    print("Error in executing call based request: ", e)
                    signal.alarm(0)
                    if early_stop:
                        break
                    results[index] = {"status": "error", "error": str(e)}
            return results
        finally:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, original_handler)

    @staticmethod
    def create_temp_file(content: str) -> str:
        """Create a temporary file with the given content.

        Args:
            content: Content to write to temp file

        Returns:
            Path to the created temp file
        """
        with tempfile.NamedTemporaryFile(delete=False, mode="w", encoding="utf-8") as temp_file:
            temp_file.write(content)
            return temp_file.name
