"""Ray-based Code Execution Backend."""

import asyncio
import logging
import os
import subprocess
import tempfile
from typing import List

import ray

from bespokelabs.curator.experimental.code_execution_backend.base_backend import BaseCodeExecutionBackend
from bespokelabs.curator.experimental.types import CodeAPIRequest, CodeExecutionResponse, CodeTestCaseResponse


@ray.remote
class CodeExecutorActor:
    """Ray actor for executing code in isolation."""

    def __init__(self):
        """Initialize the actor."""
        self.temp_files = []

    def __del__(self):
        """Cleanup temporary files on actor shutdown."""
        if hasattr(self, "temp_files") and self.temp_files:
            for temp_file in self.temp_files:
                try:
                    if os.path.exists(temp_file):
                        os.unlink(temp_file)
                except Exception:
                    pass

    async def execute_test_case(self, code: str, test_case, test_case_idx: int, timeout: int) -> CodeTestCaseResponse:
        """Execute a single test case in isolation."""
        temp_program_path = None

        try:
            # Create temporary file
            with tempfile.NamedTemporaryFile(delete=False, mode="w", encoding="utf-8") as temp_file:
                temp_file.write(code)
                temp_program_path = temp_file.name
                self.temp_files.append(temp_program_path)

            input_data = test_case.input
            if isinstance(input_data, list):
                input_data = "\n".join(str(item) for item in input_data)

            try:
                result = subprocess.run(["python", temp_program_path], input=input_data, text=True, capture_output=True, timeout=timeout)
                return CodeTestCaseResponse(
                    test_case_idx=test_case_idx,
                    response_message="success",
                    response_errors=None,
                    response_stdout=result.stdout,
                    response_stderr=result.stderr,
                )

            except subprocess.TimeoutExpired:
                return CodeTestCaseResponse(
                    test_case_idx=test_case_idx,
                    response_message="timeout",
                    response_errors=[f"Execution timed out after {timeout}s"],
                    response_stdout=None,
                    response_stderr=None,
                )

            except Exception as e:
                return CodeTestCaseResponse(
                    test_case_idx=test_case_idx,
                    response_message="error",
                    response_errors=[str(e)],
                    response_stdout=None,
                    response_stderr=None,
                )

        finally:
            if temp_program_path and temp_program_path in self.temp_files:
                try:
                    os.unlink(temp_program_path)
                    self.temp_files.remove(temp_program_path)
                except Exception:
                    pass

    async def execute_test_cases(self, code: str, test_cases: List, timeout: int) -> List[CodeTestCaseResponse]:
        """Execute multiple test cases in isolation."""
        results = []
        for idx, test_case in enumerate(test_cases):
            result = await self.execute_test_case(code, test_case, idx, timeout)
            results.append(result)
        return results


class RayCodeExecutionBackend(BaseCodeExecutionBackend):
    """Ray-based code execution backend."""

    def __init__(self, config):
        """Initialize the backend."""
        super().__init__(config)
        self.config = config
        self.logger = logging.getLogger(__name__)

        # Initialize Ray if needed
        if not ray.is_initialized():
            ray.init(num_cpus=os.cpu_count())

        self.executor = CodeExecutorActor.remote()

    async def execute_request(self, request: CodeAPIRequest) -> CodeExecutionResponse:
        """Execute a single request using Ray actors."""
        code = request.generic_request.code
        test_cases = request.generic_request.test_cases
        timeout = request.generic_request.execution_params.timeout

        # Execute all test cases in the remote executor
        results = await asyncio.get_event_loop().run_in_executor(None, ray.get, self.executor.execute_test_cases.remote(code, test_cases, timeout))

        return CodeExecutionResponse(
            responses=results,
            code_api_request=request,
        )

    def shutdown(self):
        """Cleanup resources when shutting down the backend."""
        if ray.is_initialized():
            ray.shutdown()
