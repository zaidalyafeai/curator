from dataclasses import dataclass

from e2b_code_interpreter import Sandbox

from bespokelabs.curator.experimental.code_execution_backend.base_backend import BaseCodeExecutionBackend


@dataclass
class E2BCodeExecutionBackendConfig:
    template_id: str


class E2BCodeExecutionBackend(BaseCodeExecutionBackend):
    def __init__(self, config: E2BCodeExecutionBackendConfig):
        self.sandbox = Sandbox()

    def execute(self, code: str, timeout: int = 10) -> str:
        return self.sandbox.run_code(code, timeout=timeout)

    def execute_testcase(self, code: str, testcases: str, timeout: int = 10) -> str:
        return self.sandbox.run_code(code, timeout=timeout)

    def execute_request(
        self, request: CodeExecutionRequest, session: aiohttp.ClientSession, status_tracker: CodeExecutionStatusTracker
    ) -> CodeExecutionResponse:
        """Execute a single request."""
        pass
