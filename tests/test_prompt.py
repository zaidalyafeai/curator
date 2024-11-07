import pytest
from pydantic import BaseModel
from typing import Optional
import os

from prompt import Prompter
from datasets import Dataset
import bespokelabs.curator.prompter.prompter as prompter


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
        model_name="gpt-4o-mini",
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        response_format=MockResponseFormat,
    )


@pytest.mark.test
def test_completions(prompter: Prompter, tmp_path):
    """Test that completions processes a dataset correctly.

    Args:
        prompter: Fixture providing a configured Prompter instance.
        tmp_path: Pytest fixture providing temporary directory.
    """
    # Create a simple test dataset
    test_data = {
        "context": ["Test context 1", "Test context 2"],
        "question": ["What is 1+1?", "What is 2+2?"],
    }
    dataset = Dataset.from_dict(test_data)

    # Set up temporary cache directory
    os.environ["BELLA_CACHE_DIR"] = str(tmp_path)

    # Run completions
    result_dataset = prompter.completions(
        dataset=dataset,
        prompter=prompter,
        output_column="response",
        name="test_completions",
    )

    # Assertions
    assert len(result_dataset) == len(dataset)
    assert "response" in result_dataset.column_names

    # Check first row's response format
    first_response = result_dataset[0]["response"]
    assert isinstance(first_response, dict)
    assert "message" in first_response
    assert "confidence" in first_response
