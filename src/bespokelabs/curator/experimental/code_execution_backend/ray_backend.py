"""Ray-based Code Execution Backend."""

import asyncio
import logging
import os
import subprocess
import tempfile

import ray

from bespokelabs.curator.experimental.code_execution_backend.base_backend import BaseCodeExecutionBackend
from bespokelabs.curator.experimental.types import CodeAPIRequest, CodeExecutionResponse, CodeExecutionRequestParams


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

    async def execute_standard_input_request(self, code: str, code_input: str, execution_params: CodeExecutionRequestParams) -> CodeExecutionResponse:
        """Execute code with standard input.

        Args:
            code: Source code
            code_input: Input to the code
            execution_params: Execution parameters

        Returns:
            CodeExecutionResponse: Execution results
        """
        temp_program_path = None

        try:
            # Create temporary file
            with tempfile.NamedTemporaryFile(delete=False, mode="w", encoding="utf-8") as temp_file:
                temp_file.write(code)
                temp_program_path = temp_file.name
                self.temp_files.append(temp_program_path)

            try:
                result = subprocess.run(["python", temp_program_path], input=code_input, text=True, capture_output=True, timeout=execution_params.timeout)
                return CodeExecutionResponse(
                    response_message="success",
                    response_stdout=result.stdout,
                    response_stderr=result.stderr,
                )

            except subprocess.TimeoutExpired:
                return CodeExecutionResponse(
                    response_message="timeout",
                    response_error=f"Execution timed out after {execution_params.timeout}s"
                )

            except Exception as e:
                return CodeExecutionResponse(
                    response_message="error",
                    response_error=str(e)
                )

        finally:
            if temp_program_path and temp_program_path in self.temp_files:
                try:
                    os.unlink(temp_program_path)
                    self.temp_files.remove(temp_program_path)
                except Exception:
                    pass


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
        code = request.execution_request.code
        code_input = request.execution_request.code_input
        execution_params = request.execution_request.execution_params

        # Execute request in the remote executor
        result = await asyncio.get_event_loop().run_in_executor(
            None,
            ray.get,
            self.executor.execute_standard_input_request.remote(code, code_input, execution_params)
        )

        return result

    def shutdown(self):
        """Cleanup resources when shutting down the backend."""
        if ray.is_initialized():
            ray.shutdown()
