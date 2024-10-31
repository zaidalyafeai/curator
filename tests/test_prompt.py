import pytest
from pydantic import BaseModel
from typing import Optional

from prompt import Prompter


class MockResponseFormat(BaseModel):
    """Mock response format for testing."""

    message: str
    confidence: Optional[float] = None


@pytest.fixture
def prompter() -> Prompter:
    """Create a Prompter instance for testing.

    Returns:
        PromptCaller: A configured prompt caller instance.
    """
    system_prompt = "You are a helpful assistant. Context: {{ context }}"
    user_prompt = "Answer this question: {{ question }}"
    return Prompter(
        model_name="gpt-3.5-turbo",
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        response_format=MockResponseFormat,
    )


@pytest.mark.asyncio
async def test_prompter_get_api_call_fn(prompter: Prompter):
    """Test that get_api_call_fn returns a callable function that processes prompts correctly.

    Args:
        prompter: Fixture providing a configured Prompter instance.
    """
    call_api_fn = prompter.get_api_call_fn()

    # Test input data
    row = {
        "context": "You are testing a prompt system",
        "question": 'Respond with a message "Hello, world!" and a confidence of 0.9.',
    }

    # Call the API
    result = await call_api_fn(row)

    # Assertions
    assert isinstance(result, MockResponseFormat)
    assert result.message == "Hello, world!"
    assert result.confidence == 0.9
