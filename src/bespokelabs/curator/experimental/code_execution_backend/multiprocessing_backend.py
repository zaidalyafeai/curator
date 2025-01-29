"""Multiprocessing Code Execution Backend."""

import asyncio
import os
import subprocess
import tempfile
from concurrent.futures import ProcessPoolExecutor

from bespokelabs.curator.experimental.code_execution_backend.base_backend import BaseCodeExecutionBackend
from bespokelabs.curator.experimental.types import CodeAPIRequest, CodeExecutionResponse, CodeTestCaseResponse


class MultiprocessingCodeExecutionBackend(BaseCodeExecutionBackend):
    """Multiprocessing Code Execution Backend."""

    def __init__(self, config):
        """Initialize the backend."""
        super().__init__(config)
        self.config = config
        self.process_pool = ProcessPoolExecutor(max_workers=os.cpu_count())

    async def execute_request(self, request: CodeAPIRequest) -> CodeExecutionResponse:
        """Execute a single request."""
        loop = asyncio.get_running_loop()

        results = await loop.run_in_executor(
            self.process_pool,
            self.execute_standard_input_request,
            request.generic_request.code,
            request.generic_request.test_cases,
            request.generic_request.execution_params.timeout,
        )

        return CodeExecutionResponse(
            responses=results,
            code_api_request=request,
        )

    @classmethod
    def _create_temp_file(cls, content: str) -> str:
        """Create a temporary file with the given content.

        Args:
            content: Content to write to temp file

        Returns:
            Path to the created temp file
        """
        with tempfile.NamedTemporaryFile(delete=False, mode="w", encoding="utf-8") as temp_file:
            temp_file.write(content)
            return temp_file.name

    @classmethod
    def execute_standard_input_request(cls, code: str, test_cases: list, timeout: int, early_stop: bool = True) -> dict:
        """Execute code with function calls and test cases.

        Args:
            code: Source code
            test_cases: List of test cases to run
            timeout: Execution timeout in seconds
            early_stop: Whether to stop on first failure

        Returns:
            Dict containing execution results
        """
        temp_program_path = None
        try:
            temp_program_path = cls._create_temp_file(code)
            exec_results = []

            for test_case_idx, test_case in enumerate(test_cases):
                input_data = test_case.input
                if isinstance(input_data, list):
                    input_data = "\n".join(input_data)

                try:
                    result = subprocess.run(["python", temp_program_path], input=input_data, text=True, capture_output=True, timeout=timeout)
                    exec_results.append(
                        CodeTestCaseResponse(
                            test_case_idx=test_case_idx,
                            response_message="success",
                            response_errors=None,
                            response_stdout=result.stdout,
                            response_stderr=result.stderr,
                        )
                    )

                except subprocess.TimeoutExpired:
                    exec_results.append(
                        CodeTestCaseResponse(
                            test_case_idx=test_case_idx,
                            response_message="timeout",
                            response_errors=[f"Execution timed out after {timeout}s"],
                            response_stdout=None,
                            response_stderr=None,
                        )
                    )
                    if early_stop:
                        break

                except Exception as e:
                    exec_results.append(
                        CodeTestCaseResponse(
                            test_case_idx=test_case_idx,
                            response_message="error",
                            response_errors=[str(e)],
                            response_stdout=None,
                            response_stderr=None,
                        )
                    )
                    if early_stop:
                        break

            print(exec_results[0])
            return exec_results

        finally:
            if temp_program_path:
                os.unlink(temp_program_path)

    def __del__(self):
        """Clean up pool when object is destroyed."""
        self.process_pool.shutdown(wait=True)
