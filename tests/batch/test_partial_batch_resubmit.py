from bespokelabs.curator.request_processor.batch.base_batch_request_processor import (
    BaseBatchRequestProcessor,
)
from bespokelabs.curator.request_processor.batch.openai_batch_request_processor import (
    OpenAIBatchRequestProcessor,
)
from bespokelabs.curator.request_processor.batch.anthropic_batch_request_processor import (
    AnthropicBatchRequestProcessor,
)
from bespokelabs.curator.request_processor.config import BatchRequestProcessorConfig
from bespokelabs.curator.types.generic_batch import GenericBatch
from bespokelabs.curator.llm.prompt_formatter import PromptFormatter
import json
import shutil
import os
from bespokelabs.curator import LLM
from datasets import Dataset


def test_generic_response_file_from_responses():
    with open("tests/batch/msgbatch_01AAs5rns9HhDrWvFxCoivqg_results.jsonl", "r") as f:
        responses = [json.loads(line) for line in f]

    config = BatchRequestProcessorConfig(
        model="claude-3-5-haiku-20241022",
    )
    batch = GenericBatch(
        request_file="tests/batch/requests_20.jsonl",
        id="msgbatch_01AAs5rns9HhDrWvFxCoivqg",
        created_at="2024-12-29T15:19:35.497436Z",
        finished_at="2024-12-29T15:19:35.497436Z",
        status="submitted",
        api_key_suffix="fwAA",
        request_counts={
            "total": 1551,
            "failed": 0,
            "succeeded": 0,
            "raw_request_counts_object": {
                "canceled": 0,
                "errored": 0,
                "expired": 0,
                "processing": 1551,
                "succeeded": 0,
            },
        },
        raw_status="in_progress",
        raw_batch={
            "id": "msgbatch_01AAs5rns9HhDrWvFxCoivqg",
            "archived_at": None,
            "cancel_initiated_at": None,
            "created_at": "2024-12-29T15:19:35.497436Z",
            "ended_at": None,
            "expires_at": "2024-12-30T15:19:35.497436Z",
            "processing_status": "in_progress",
            "request_counts": {
                "canceled": 0,
                "errored": 0,
                "expired": 0,
                "processing": 1551,
                "succeeded": 0,
            },
            "results_url": None,
            "type": "message_batch",
        },
    )
    abrp = AnthropicBatchRequestProcessor(config)
    abrp.prompt_formatter = PromptFormatter(
        model_name="claude-3-5-haiku-20241022",
        prompt_func=lambda x: x["instruction"],
        parse_func=lambda x, y: x,
    )
    abrp.generic_response_file_from_responses(responses, batch)


def test_partial_batch_resubmit():
    # Clean up test directory if it exists
    test_dir = "tests/batch/test_working_dir"
    if os.path.exists(test_dir):
        shutil.rmtree(test_dir)

    # Create test directory and copy files
    os.makedirs(test_dir)
    shutil.copy("tests/batch/requests_0.jsonl", f"{test_dir}/requests_0.jsonl")
    shutil.copy("tests/batch/batch_objects.jsonl", f"{test_dir}/batch_objects.jsonl")

    llm = LLM(
        prompt_func=lambda row: row["instruction"],
        model_name="claude-3-5-haiku-20241022",
        response_format=None,
        batch=True,
        batch_size=50_000,
        batch_check_interval=60,
        base_url=None,
    )

    # this doesn't work because it creates a subdir with the hash in the test_working_dir
    llm(
        dataset=Dataset.from_dict({"instruction": ["just say 'hi'"] * 1551}),
        working_dir=test_dir,
    )


if __name__ == "__main__":
    test_partial_batch_resubmit()
