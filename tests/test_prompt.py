import pytest
from pydantic import BaseModel
from typing import Optional

from prompt import PromptCaller

class MockResponseFormat(BaseModel):
    """Mock response format for testing."""
    message: str
    confidence: Optional[float] = None


@pytest.fixture
def prompt_caller() -> PromptCaller:
    """Create a PromptCaller instance for testing.
    
    Returns:
        PromptCaller: A configured prompt caller instance.
    """
    system_prompt = "You are a helpful assistant. Context: {{ context }}"
    user_prompt = "Answer this question: {{ question }}"
    return PromptCaller(
        model_name="gpt-3.5-turbo",
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        response_format=MockResponseFormat
    )


@pytest.mark.asyncio
async def test_prompt_caller_get_callapi(prompt_caller: PromptCaller):
    """Test that get_callapi returns a callable function that processes prompts correctly.
    
    Args:
        prompt_caller: Fixture providing a configured PromptCaller instance.
    """
    call_api = prompt_caller.get_api_call_fn()
    
    # Test input data
    row = {
        "context": "You are testing a prompt system",
        "question": 'Respond with a message "Hello, world!" and a confidence of 0.9.'
    }
    
    # Call the API
    result = await call_api(row)

    # Assertions
    assert isinstance(result, MockResponseFormat)
    assert result.message == "Hello, world!"
    assert result.confidence == 0.9
