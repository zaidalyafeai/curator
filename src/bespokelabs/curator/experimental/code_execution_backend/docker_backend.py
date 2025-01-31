"""Docker Code Execution Backend."""

import os
import tempfile

import aiodocker
from aiodocker.exceptions import DockerError

from bespokelabs.curator.experimental.code_execution_backend.base_backend import BaseCodeExecutionBackend
from bespokelabs.curator.experimental.types import CodeAPIRequest, CodeExecutionResponse, CodeTestCaseResponse


class DockerCodeExecutionBackend(BaseCodeExecutionBackend):
    """Docker-based code execution backend."""

    def __init__(self, config):
        """Initialize the backend."""
        super().__init__(config)
        self.config = config
        self.client = None  # Will be initialized in execute_request

    async def _ensure_client(self):
        """Ensure aiodocker client is initialized."""
        if self.client is None:
            self.client = aiodocker.Docker()
            # Pull image asynchronously
            try:
                await self.client.images.pull("python:3.9-slim")
            except DockerError as e:
                raise RuntimeError(f"Failed to pull Docker image: {e}") from e

    async def execute_request(self, request: CodeAPIRequest) -> CodeExecutionResponse:
        """Execute a single request in a Docker container."""
        results = await self.execute_standard_input_request(
            request.generic_request.code, request.generic_request.input, request.generic_request.execution_params.timeout
        )

        return CodeExecutionResponse(responses=results, code_api_request=request)

    async def execute_standard_input_request(self, code: str, input: list, timeout: int) -> list:
        """Execute code with test cases in Docker container."""
        await self._ensure_client()
        temp_program_path = None
        try:
            # Create temporary file with code
            with tempfile.NamedTemporaryFile(delete=False, mode="w", encoding="utf-8") as temp_file:
                temp_file.write(code)
                temp_program_path = temp_file.name

            exec_results = []
            for test_case_idx, test_case in enumerate(input):
                input_data = test_case.input
                if isinstance(input_data, list):
                    input_data = "\n".join(input_data)

                try:
                    print(f"Running test case {test_case_idx} in the container...")
                    config = {
                        "Cmd": ["/bin/sh", "-c", "python /code/program.py"],
                        "Image": "python:3.9-slim",
                        "Volumes": {"/code": {}},
                        "OpenStdin": True,
                        "Tty": True,
                    }

                    container = await self.client.containers.create(config=config)
                    await container.put_archive("/code", self._make_tarfile(temp_program_path))

                    try:
                        await container.start()
                        if input_data:
                            exec_config = {
                                "Cmd": ["sh", "-c", f'echo "{input_data}" | python /code/program.py'],
                                "AttachStdout": True,
                                "AttachStderr": True,
                            }
                            exec_obj = await container.exec.create(exec_config)
                            output = await exec_obj.start()
                            logs = await output.read()
                        else:
                            logs = await container.log(stdout=True, stderr=True)

                        exec_results.append(
                            CodeTestCaseResponse(
                                test_case_idx=test_case_idx,
                                response_message="success",
                                response_errors=None,
                                response_stdout="".join(logs),
                                response_stderr=None,
                            )
                        )

                    except Exception as e:
                        exec_results.append(
                            CodeTestCaseResponse(
                                test_case_idx=test_case_idx, response_message="error", response_errors=[str(e)], response_stdout=None, response_stderr=None
                            )
                        )

                    finally:
                        try:
                            await container.stop()
                            await container.delete()
                        except Exception:
                            pass

                except DockerError as e:
                    exec_results.append(
                        CodeTestCaseResponse(
                            test_case_idx=test_case_idx,
                            response_message="error",
                            response_errors=[f"Docker error: {str(e)}"],
                            response_stdout=None,
                            response_stderr=None,
                        )
                    )

            return exec_results

        finally:
            if temp_program_path and os.path.exists(temp_program_path):
                os.unlink(temp_program_path)

    async def shutdown(self):
        """Cleanup Docker resources."""
        if self.client:
            await self.client.close()

    def _make_tarfile(self, source_path):
        """Create a tar archive containing the source file."""
        import io
        import tarfile

        tar_stream = io.BytesIO()
        with tarfile.open(fileobj=tar_stream, mode="w") as tar:
            tar.add(source_path, arcname="program.py")
        tar_stream.seek(0)
        return tar_stream
