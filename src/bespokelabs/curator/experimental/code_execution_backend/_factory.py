from bespokelabs.curator.experimental.types import CodeExecutionBackendConfig

class _CodeExecutionBackendFactory:
    @classmethod
    def create(cls, backend: str, backend_params: dict):
        if backend == "e2b":
            from bespokelabs.curator.experimental.code_execution_backend.e2b_backend import E2BCodeExecutionBackend

            _code_executor = E2BCodeExecutionBackend(CodeExecutionBackendConfig(**backend_params))
        elif backend == "multiprocessing":
            from bespokelabs.curator.experimental.code_execution_backend.multiprocessing_backend import MultiprocessingCodeExecutionBackend
            _code_executor = MultiprocessingCodeExecutionBackend(CodeExecutionBackendConfig(**backend_params))
        elif backend == "rpy":
            from bespokelabs.curator.experimental.code_execution_backend.rpy_backend import RPyCodeExecutionBackend

            _code_executor = RPyCodeExecutionBackend(backend_params)
        elif backend == "docker":
            from bespokelabs.curator.experimental.code_execution_backend.docker_backend import DockerCodeExecutionBackend

            _code_executor = DockerCodeExecutionBackend(backend_params)
        else:
            raise ValueError(f"Unsupported backend: {backend}")

        return _code_executor
