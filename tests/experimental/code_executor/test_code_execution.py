# test with multiprocessing backend
import pytest

from bespokelabs.curator.experimental.code_execution_backend.ray_backend import RayCodeExecutionBackend
from bespokelabs.curator.experimental.types import CodeExecutionRequest, CodeExecutionRequestParams


@pytest.mark.asyncio
async def test_simple_code_execution():
    """Test simple code execution with basic input/output."""
    # Initialize backend
    backend = RayCodeExecutionBackend(config=None)

    try:
        # Simple Python code that reads input and prints output
        code = """
input_value = input()
print(f"You entered: {input_value}")
"""

        # Create execution request
        request = CodeExecutionRequest(code=code, code_input="Hello World", code_output="", execution_params=CodeExecutionRequestParams(timeout=5))

        # Execute code
        response = await backend.execute_standard_input_request(code=request.code, code_input=request.code_input, execution_params=request.execution_params)

        # Verify execution was successful
        assert response.response_message == "success"
        assert "You entered: Hello World" in response.response_stdout
        assert not response.response_error

    finally:
        # Cleanup
        backend.shutdown()
