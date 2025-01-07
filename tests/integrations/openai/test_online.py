import pytest
import vcr
import shutil
import tempfile
from datasets import Dataset

vcr_config = vcr.VCR(
    serializer="yaml",
    cassette_library_dir="tests/integrations/openai/fixtures",
    record_mode=None,
    match_on=["method", "uri", "body"],
)


def prompt_func(row):
    return row["conversation"][0]["content"]


def parse_func(row, response):
    instruction = row["conversation"][0]["content"]
    return {"instruction": instruction, "new_response": response}


@pytest.fixture(scope="module")
def helper_fixture():
    dataset = Dataset.from_parquet("tests/integrations/openai/fixtures/dataset.parquet")
    temp_directory = tempfile.mkdtemp()
    import os

    os.environ["OPENAI_API_KEY"] = "sk-mocked-key-**"
    try:
        yield dataset, temp_directory
    finally:
        shutil.rmtree(temp_directory)


@vcr_config.use_cassette("basic_completion.yaml")
def test_basic_openai(helper_fixture):
    from bespokelabs import curator

    prompter = curator.LLM(
        prompt_func=prompt_func, parse_func=parse_func, model_name="gpt-3.5-turbo", backend="openai"
    )
    dataset, temp_directory = helper_fixture
    distilled_dataset = prompter(dataset, working_dir=temp_directory)

    # TODO: Maybe move this to fixture
    # Fingerprint is changing
    # assert distilled_dataset._fingerprint == '6a8096fbdf540ae2'
