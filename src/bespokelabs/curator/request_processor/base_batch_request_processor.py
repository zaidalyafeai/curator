import datetime
import json
import logging
import os
import litellm

from bespokelabs.curator.dataset import Dataset
from bespokelabs.curator.llm.prompt_formatter import PromptFormatter
from bespokelabs.curator.request_processor.base_request_processor import (
    parse_response_message,
    BaseRequestProcessor,
)
from bespokelabs.curator.types.generic_batch import GenericBatch
from bespokelabs.curator.types.generic_batch import GenericBatchRequestCounts
from bespokelabs.curator.request_processor.event_loop import run_in_event_loop
from bespokelabs.curator.types.token_usage import TokenUsage
from bespokelabs.curator.types.generic_request import GenericRequest
from bespokelabs.curator.types.generic_response import GenericResponse
from bespokelabs.curator.batch_manager.base_batch_manager import BaseBatchManager
from bespokelabs.curator.batch_manager.anthropic_batch_manager import AnthropicBatchManager
from bespokelabs.curator.batch_manager.openai_batch_manager import OpenAIBatchManager

logger = logging.getLogger(__name__)


class BaseBatchRequestProcessor(BaseRequestProcessor):
    def __init__(
        self,
        batch_size: int,
        model: str,
        delete_successful_batch_files: bool,
        delete_failed_batch_files: bool,
        temperature: float | None = None,
        top_p: float | None = None,
        batch_check_interval: int = 60,
        presence_penalty: float | None = None,
        frequency_penalty: float | None = None,
    ):
        super().__init__(batch_size)
        self.model = model
        self.check_interval: int = batch_check_interval
        self.temperature: float | None = temperature
        self.top_p: float | None = top_p
        self.presence_penalty: float | None = presence_penalty
        self.frequency_penalty: float | None = frequency_penalty
        self.delete_successful_batch_files: bool = delete_successful_batch_files
        self.delete_failed_batch_files: bool = delete_failed_batch_files

    async def run_batch_operations(self, batch_manager, request_files):
        # For running in a single event loop (so sempahore does not change)
        await batch_manager.submit_batches_from_request_files(request_files)
        await batch_manager.poll_and_process_batches()

    def get_batch_manager(
        self, working_dir: str, prompt_formatter: PromptFormatter
    ) -> BaseBatchManager:
        # TODO(Ryan): Maybe better to do the determine backend like in llm class
        MODEL_TO_BATCH_MANAGER = {
            "claude-3-5-sonnet-20240620": AnthropicBatchManager,
            "claude-3-5-sonnet-20241022": AnthropicBatchManager,
            "gpt-4o-mini": OpenAIBatchManager,
            "gpt-4o-2024-08-06": OpenAIBatchManager,
        }

        BatchManagerClass = MODEL_TO_BATCH_MANAGER.get(self.model)
        if not BatchManagerClass:
            raise ValueError(f"Model {self.model} is not supported for batch processing")

        return BatchManagerClass(
            working_dir,
            self.check_interval,
            prompt_formatter,
            delete_successful_batch_files=self.delete_successful_batch_files,
            delete_failed_batch_files=self.delete_failed_batch_files,
        )

    def run(
        self,
        dataset: Dataset | None,
        working_dir: str,
        parse_func_hash: str,
        prompt_formatter: PromptFormatter,
    ) -> Dataset:
        """
        Processes a dataset using OpenAI's batch API.

        This function orchestrates the complete batch processing workflow:
        1. Attempts to load cached results if available
        2. Creates request files from the dataset
        3. Submits and processes batches
        4. Creates output dataset files

        Args:
            dataset (Dataset | None): Input dataset to process.
            working_dir (str): Directory for storing intermediate files and results.
            parse_func_hash (str): Hash of the parsing function for cache identification.
            prompt_formatter (PromptFormatter): Formatter for processing prompts and responses.

        Returns:
            Dataset: Processed dataset

        Raises:
            RuntimeError: If batch processing fails or no successful responses are received.
        """
        batch_manager = self.get_batch_manager(working_dir, prompt_formatter)

        if self.batch_size > batch_manager.max_requests_per_batch:
            raise ValueError(
                f"batch_size {self.batch_size} is greater than the maximum of "
                f"{batch_manager.max_requests_per_batch:,} requests per batch that {batch_manager.__class__.__name__} supports. "
                f"Please set your batch_size to be less than or equal to {batch_manager.max_requests_per_batch:,}."
            )

        # load from already completed dataset
        output_dataset = self.attempt_loading_cached_dataset(working_dir, parse_func_hash)
        if output_dataset is not None:
            return output_dataset

        request_files = set(self.create_request_files(dataset, working_dir, prompt_formatter))
        self.prompt_formatter = prompt_formatter

        run_in_event_loop(self.run_batch_operations(batch_manager, request_files))

        return self.create_dataset_files(working_dir, parse_func_hash, prompt_formatter)

    def cancel_batches(self, working_dir: str) -> Dataset:
        """
        Cancels all submitted batches and exits the program.

        Args:
            working_dir (str): The directory where submitted batch object file is stored.
        """
        batch_manager = self.get_batch_manager(working_dir)
        run_in_event_loop(batch_manager.cancel_batches())
        logger.warning("Exiting program after batch cancellation.")
        os._exit(1)
