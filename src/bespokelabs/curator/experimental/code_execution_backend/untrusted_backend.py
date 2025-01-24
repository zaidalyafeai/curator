from bespokelabs.curator.experimental.code_execution_backend.base_backend import BaseCodeExecutionBackend
from bespokelabs.curator.experimental.types import CodeExecutionRequest, CodeExecutionResponse, CodeExecutionStatusTracker
import aiohttp

class UntrustedBackend(BaseCodeExecutionBackend):
    def __init__(self, config):
        self.config = config

    def execute_request(self, request: CodeExecutionRequest, session: aiohttp.ClientSession, status_tracker: CodeExecutionStatusTracker) -> CodeExecutionResponse:
        """Execute a single request."""
        