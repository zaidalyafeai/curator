from datasets import Dataset
import os
import pytest
import shutil

_KEY_MAP = {'openai': 'OPENAI_API_KEY'}

@pytest.fixture
def mocked_workspace(request):
    backend = request.param['integration']
    os.environ[_KEY_MAP[backend]] = "sk-mocked-**"
    os.environ["HF_DATASETS_CACHE"] = "/dev/null"
    mocked_request_dir = f'tests/integrations/{backend}/fixtures/.test_cache'
    os.makedirs(mocked_request_dir, exist_ok=True)

    try:
        yield mocked_request_dir
    finally:
        shutil.rmtree(mocked_request_dir)

@pytest.fixture
def mock_dataset():
    dataset = Dataset.from_parquet(f"tests/integrations/common_fixtures/dataset.parquet")
    try:
        yield dataset
    finally:
        # TODO: cleanup?
        pass
