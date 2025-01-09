import time
import pytest
import vcr
import logging
import signal
import pandas as pd
import os

from tests.integrations import helper

mode = os.environ.get("VCR_MODE", None)

vcr_config = vcr.VCR(
    serializer="yaml",
    cassette_library_dir="tests/integrations/openai/fixtures",
    record_mode=mode,
)

##############################
# Online                     #
##############################

class TimeoutException(Exception):
    pass

class timeout:
    def __init__(self, seconds):
        self.seconds = seconds

    def __enter__(self):
        signal.signal(signal.SIGALRM, self._handle_timeout)
        signal.alarm(self.seconds)

    def __exit__(self, exc_type, exc_value, traceback):
        signal.alarm(0)

    @staticmethod
    def _handle_timeout(signum, frame):
        raise TimeoutException("Function execution exceeded time limit!")


@pytest.mark.parametrize("temp_working_dir", ([{"integration": "openai"}]), indirect=True)
@vcr_config.use_cassette("basic_completion.yaml")
def test_basic(temp_working_dir, mock_dataset):
    distilled_dataset = helper.create_basic(temp_working_dir, mock_dataset)
    distilled_dataset.cleanup_cache_files()
    pandas_hash = pd.util.hash_pandas_object(distilled_dataset.to_pandas())
    assert str(int(pandas_hash.sum())) == "603110695445175717"


@pytest.mark.parametrize("temp_working_dir", ([{"integration": "openai"}]), indirect=True)
@vcr_config.use_cassette("camel_completion.yaml")
def test_camel(temp_working_dir):
    qa_dataset = helper.create_camel(temp_working_dir)

    qa_dataset.cleanup_cache_files()
    pandas_hash = pd.util.hash_pandas_object(qa_dataset.to_pandas())
    assert str(int(pandas_hash.sum())) == "15734523020459649279"


@pytest.mark.parametrize("temp_working_dir", ([{"integration": "openai"}]), indirect=True)
@vcr_config.use_cassette("basic_completion.yaml")
def test_basic_cache(caplog, temp_working_dir, mock_dataset):
    st = time.time()
    distilled_dataset = helper.create_basic(temp_working_dir, mock_dataset)
    tt = time.time() - st

    # This should use cache
    from bespokelabs.curator.request_processor.base_request_processor import CACHE_MSG

    logger = "bespokelabs.curator.request_processor.base_request_processor"
    with caplog.at_level(logging.INFO, logger=logger):
        st = time.time()
        distilled_dataset_cached = helper.create_basic(temp_working_dir, mock_dataset)
        cached_tt = time.time() - st
        distilled_dataset.cleanup_cache_files()
        assert f"Using cached output dataset. {CACHE_MSG}" in caplog.text
        assert cached_tt < tt // 2


@pytest.mark.skip
@pytest.mark.parametrize("temp_working_dir", ([{"integration": "openai"}]), indirect=True)
@vcr_config.use_cassette("basic_completion.yaml")
def test_low_rpm_setting(temp_working_dir, mock_dataset):
    distilled_dataset = helper.create_basic(
        temp_working_dir, mock_dataset, llm_params={"max_requests_per_minute": 5}
    )


@pytest.mark.parametrize("temp_working_dir", ([{"integration": "openai"}]), indirect=True)
@vcr_config.use_cassette("basic_completion.yaml")
def test_auto_rpm(temp_working_dir):
    llm = helper.create_llm()
    assert llm._request_processor.header_based_max_requests_per_minute == 10_000
    assert llm._request_processor.header_based_max_tokens_per_minute == 200_000




@pytest.mark.parametrize("temp_working_dir", ([{"integration": "openai"}]), indirect=True)
@vcr_config.use_cassette("basic_resume.yaml")
def test_resume(caplog, temp_working_dir, mock_dataset):
    with pytest.raises(TimeoutException):
        with timeout(3):
            distilled_dataset = helper.create_basic(temp_working_dir, mock_dataset, llm_params={'max_requests_per_minute': 1})

    logger = "bespokelabs.curator.request_processor.online.base_online_request_processor"
    with caplog.at_level(logging.INFO, logger=logger):
        distilled_dataset = helper.create_basic(temp_working_dir, mock_dataset)
        resume_msg = "Resuming progress by reading existing file: tests/integrations/"
        assert resume_msg in caplog.text

    distilled_dataset.cleanup_cache_files()
    pandas_hash = pd.util.hash_pandas_object(distilled_dataset.to_pandas())
    assert str(int(pandas_hash.sum())) == "7531379036901868056"

##############################
# Batch                      #
##############################


@pytest.mark.parametrize("temp_working_dir", ([{"integration": "openai"}]), indirect=True)
@vcr_config.use_cassette("basic_batch_completion.yaml")
def test_basic_batch(temp_working_dir, mock_dataset):
    distilled_dataset = helper.create_basic(temp_working_dir, mock_dataset, batch=True)
    distilled_dataset.cleanup_cache_files()
    pandas_hash = pd.util.hash_pandas_object(distilled_dataset.to_pandas())
    assert str(int(pandas_hash.sum())) == "18416626377197880177"
