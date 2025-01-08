import pytest
import vcr
import pandas as pd
import os

from tests.integrations import helper
mode = os.environ.get('VCR_MODE', None)

vcr_config = vcr.VCR(
    serializer="yaml",
    cassette_library_dir="tests/integrations/openai/fixtures",
    record_mode=mode,
)

##############################
# Online                     #
##############################

@pytest.mark.parametrize(
    'mocked_workspace',
    ([{'integration':'openai'}]),
    indirect=True
)
@vcr_config.use_cassette("basic_completion.yaml")
def test_basic(mocked_workspace, mock_dataset):
    distilled_dataset = helper.create_basic(mocked_workspace, mock_dataset)
    distilled_dataset.cleanup_cache_files()
    pandas_hash = pd.util.hash_pandas_object(distilled_dataset.to_pandas())
    assert str(int(pandas_hash.sum())) == '603110695445175717'


@pytest.mark.parametrize(
    'mocked_workspace',
    ([{'integration':'openai'}]),
    indirect=True
)
@vcr_config.use_cassette("camel_completion.yaml")
def test_camel(mocked_workspace):
    qa_dataset = helper.create_camel(mocked_workspace)

    qa_dataset.cleanup_cache_files()
    pandas_hash = pd.util.hash_pandas_object(qa_dataset.to_pandas())
    assert str(int(pandas_hash.sum())) == '15734523020459649279'


##############################
# Batch                      #
##############################

@pytest.mark.parametrize(
    'mocked_workspace',
    ([{'integration':'openai'}]),
    indirect=True
)
@vcr_config.use_cassette("basic_batch_completion.yaml")
def test_basic_batch(mocked_workspace, mock_dataset):
    distilled_dataset = helper.create_basic(mocked_workspace, mock_dataset, batch=True)
    distilled_dataset.cleanup_cache_files()
    pandas_hash = pd.util.hash_pandas_object(distilled_dataset.to_pandas())
    assert str(int(pandas_hash.sum())) == '18416626377197880177'

    # Fingerprint is changing
    # assert distilled_dataset._fingerprint == '6a8096fbdf540ae2'
