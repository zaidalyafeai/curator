import pytest

from bespokelabs import curator


@pytest.mark.parametrize("temp_working_dir", ([{"integration": "openai"}]), indirect=True)
def test_simple(temp_working_dir):
    """Test that using the same dataset multiple times uses cache."""
    _, _, vcr_config = temp_working_dir
    with vcr_config.use_cassette("simple.yaml"):
        llm = curator.SimpleLLM(model_name="gpt-4o-mini")
        poem = llm("Say '1'. Do not explain.")
        print(poem)
