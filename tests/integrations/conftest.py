from datasets import Dataset
import os
import pytest
import shutil

_KEY_MAP = {"openai": "OPENAI_API_KEY"}


@pytest.fixture
def temp_working_dir(request):
    backend = request.param["integration"]
    os.environ[_KEY_MAP[backend]] = "sk-mocked-**"
    os.environ["HF_DATASETS_CACHE"] = "/dev/null"
    temp_working_dir = f"tests/integrations/{backend}/fixtures/.test_cache"
    os.makedirs(temp_working_dir, exist_ok=True)

    try:
        yield temp_working_dir
    finally:
        shutil.rmtree(temp_working_dir)


@pytest.fixture
def mock_dataset():
    dataset = Dataset.from_parquet(f"tests/integrations/common_fixtures/dataset.parquet")
    try:
        yield dataset
    finally:
        # TODO: cleanup?
        pass
