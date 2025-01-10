import pytest

from bespokelabs import curator

_ONLINE_BACKEND = [{"integration": backend, "vcr_path": "tests/unittests/cassettes/"} for backend in {"openai"}]


@pytest.mark.parametrize("temp_working_dir", (_ONLINE_BACKEND), indirect=True)
def test_simple(temp_working_dir):
    """Test that using the same dataset multiple times uses cache."""
    _, _, vcr_config = temp_working_dir
    with vcr_config.use_cassette("basic_diff_cache.yaml"):
        llm = curator.SimpleLLM(model_name="gpt-4o-mini")
        poem = llm("Say '1'. Do not explain.")
        print(poem)
