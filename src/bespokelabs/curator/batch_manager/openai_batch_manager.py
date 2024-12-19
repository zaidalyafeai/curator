import asyncio
import datetime
import json
import logging
import os
import litellm

from typing import Optional
from openai import AsyncOpenAI, NotFoundError
from openai.types.batch import Batch
from openai.types.batch_request_counts import BatchRequestCounts

from bespokelabs.curator.llm.prompt_formatter import PromptFormatter
from bespokelabs.curator.request_processor.base_request_processor import parse_response_message
from bespokelabs.curator.types.token_usage import TokenUsage
from bespokelabs.curator.types.generic_request import GenericRequest
from bespokelabs.curator.types.generic_response import GenericResponse
from bespokelabs.curator.batch_manager.base_batch_manager import BaseBatchManager
from bespokelabs.curator.batch_manager.base_batch_manager import GenericBatch
from bespokelabs.curator.types.generic_batch import GenericBatchRequestCounts


logger = logging.getLogger(__name__)


class OpenAIBatchManager(BaseBatchManager):
    def __init__(
        self,
        working_dir: str,
        check_interval: int = 60,
        prompt_formatter: PromptFormatter | None = None,
        delete_successful_batch_files: bool = False,
        delete_failed_batch_files: bool = False,
        max_retries: Optional[int] = None,
    ) -> None:
        """Initialize BatchManager to handle OpenAI batch processing operations.

        Args:
            working_dir (str): Directory for storing batch-related files including requests, responses,
                and tracking files.
            check_interval (int): Time interval (in seconds) between batch status checks.
            prompt_formatter (PromptFormatter): Formatter used to process prompts and validate responses.
            delete_successful_batch_files (bool): Whether to delete input/output files from OpenAI
                after successful batch completion.
            delete_failed_batch_files (bool): Whether to delete input/error files from OpenAI
                after batch failure.
        """
        super().__init__(
            working_dir=working_dir,
            check_interval=check_interval,
            prompt_formatter=prompt_formatter,
            delete_successful_batch_files=delete_successful_batch_files,
            delete_failed_batch_files=delete_failed_batch_files,
            max_retries=max_retries,
            working_dir=working_dir,
        )
        self.client = AsyncOpenAI(max_retries=self.max_retries_per_operation)

    @property
    def max_requests_per_batch(self) -> int:
        return 50_000

    @property
    def max_bytes_per_batch(self) -> int:
        return 200 * 1024 * 1024  # 200 MB

    @property
    def max_concurrent_batch_operations(self) -> int:
        return 100

    def parse_api_specific_batch_object(self, batch: object) -> GenericBatch:
        if batch.status in ["validating", "finalizing", "cancelling", "in_progress"]:
            status = "submitted"
        elif batch.status in ["completed", "failed", "expired", "cancelled"]:
            status = "finished"
        else:
            raise ValueError(f"Unknown batch status: {batch.status}")

        request_counts = GenericBatchRequestCounts(
            completed=batch.request_counts.completed,
            failed=batch.request_counts.failed,
            succeeded=batch.request_counts.completed,  # OpenAI uses "completed" for succeeded
            total=batch.request_counts.total,
            raw_request_counts_object=batch.request_counts.model_dump(),
        )

        return GenericBatch(
            request_file=batch.request_file,
            id=batch.id,
            output=batch.output,
            created_at=batch.created_at,
            finished_at=batch.finished_at,
            status=status,
            api_key_suffix=self.client.api_key[-4:],
            request_counts=request_counts,
            raw_batch=batch.model_dump(),
        )

    def parse_api_specific_response(
        self,
        raw_response: dict,
        generic_request: GenericRequest,
        batch_created_at: datetime.datetime,
    ) -> GenericResponse:
        if raw_response["response"]["status_code"] != 200:
            response_message = None
            response_errors = [raw_response["response"]["status_code"]]
            token_usage = None
            cost = None
        else:
            response_body = raw_response["response"]["body"]
            response_message_raw = response_body["choices"][0]["message"]["content"]
            usage = response_body.get("usage", {})

            token_usage = TokenUsage(
                prompt_tokens=usage.get("prompt_tokens", 0),
                completion_tokens=usage.get("completion_tokens", 0),
                total_tokens=usage.get("total_tokens", 0),
            )
            response_message, response_errors = parse_response_message(
                response_message_raw, self.prompt_formatter.response_format
            )

            cost = litellm.completion_cost(
                model=self.model,
                prompt=str(self.generic_request.messages),
                completion=response_message,
            )
            cost *= 0.5  # 50% off for batch

        return GenericResponse(
            response_message=response_message,
            response_errors=response_errors,
            raw_response=raw_response,
            raw_request=None,
            generic_request=generic_request,
            created_at=batch_created_at,
            finished_at=datetime.datetime.now(),
            token_usage=token_usage,
            response_cost=cost,
        )

    def create_api_specific_request(self, generic_request: GenericRequest) -> dict:
        """
        Creates an API-specific request body from a generic request body.

        This function transforms a GenericRequest into the format expected by OpenAI's batch API.
        It handles both standard requests and those with JSON schema response formats.

        Args:
            generic_request (GenericRequest): The generic request object containing model, messages,
                and optional response format.

        Returns:
            dict: API specific request body formatted for OpenAI's batch API, including:
                - custom_id: String identifier from the original row index
                - method: Always "POST"
                - url: OpenAI chat completions endpoint
                - body: Request parameters including model, messages, and optional formatting
        """
        # NOTE(Ryan): We can have a shared place that creates the body (since it is the same for both openai online and batch).
        if generic_request.response_format:
            body = {
                "model": generic_request.model,
                "messages": generic_request.messages,
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {
                        # NOTE(Ryan): we are not using strict: True
                        "name": "output_schema",
                        "schema": generic_request.response_format,
                    },
                },
            }
        else:
            body = {
                "model": generic_request.model,
                "messages": generic_request.messages,
            }

        if self.temperature is not None:
            body["temperature"] = self.temperature

        if self.top_p is not None:
            body["top_p"] = self.top_p

        if self.presence_penalty is not None:
            body["presence_penalty"] = self.presence_penalty

        if self.frequency_penalty is not None:
            body["frequency_penalty"] = self.frequency_penalty

        request = {
            "custom_id": str(generic_request.original_row_idx),
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": body,
        }

        return request

    async def upload_batch_file(self, file_content: bytes) -> str:
        """
        Uploads a batch file to OpenAI and waits until ready.

        Args:
            file_content (bytes): The encoded file content to upload

        Returns:
            str: The uploaded file object from OpenAI
        """
        try:
            batch_file_upload = await self.client.files.create(file=file_content, purpose="batch")
        except Exception as e:
            logger.error(f"Error uploading batch file: {e}")
            raise e

        # When submitting a file, sometimes the file is not ready immediately for status checking
        # Which results in a file not found error, so we briefly pause before checking the status
        await asyncio.sleep(1)

        try:
            batch_file_upload = await self.client.files.wait_for_processing(batch_file_upload.id)
        except Exception as e:
            logger.error(f"Error waiting for batch file to be processed: {e}")
            raise e

        logger.debug(f"File uploaded with id {batch_file_upload.id}")

        return batch_file_upload

    async def create_batch(self, batch_file_id: str, metadata: dict) -> GenericBatch:
        """
        Creates a batch job with OpenAI using an uploaded file.

        Args:
            batch_file_id (str): ID of the uploaded file to use for the batch
            metadata (dict): Metadata to be included with the batch

        Returns:
            Batch: The created batch object from OpenAI

        Raises:
            Exception: If batch creation fails
        """
        try:
            batch = await self.client.batches.create(
                input_file_id=batch_file_id,
                endpoint="/v1/chat/completions",
                completion_window="24h",
                metadata=metadata,
            )
            logger.debug(f"Batch submitted with id {batch.id}")
        except Exception as e:
            logger.error(f"Error submitting batch: {e}")
            raise e
        return batch

    async def submit_batch(self, requests: list[dict], metadata: dict) -> GenericBatch:
        """
        Handles the complete batch submission process.

        Args:
            requests (list[dict]): List of API-specific requests to submit
            metadata (dict): Metadata to be included with the batch

        Returns:
            Batch: The created batch object from OpenAI

        Side Effects:
            - Updates tracker with submitted batch status
            - Appends batch object to submitted_batch_objects_file
        """
        async with self.semaphore:
            file_content = self.create_batch_file(requests)
            batch_file_upload = await self.upload_batch_file(file_content)
            batch = await self.create_batch(batch_file_upload.id, metadata)

            # Simplified file writing
            with open(self.submitted_batch_objects_file, "a") as f:
                json.dump(batch.model_dump(), f, default=str)
                f.write("\n")
                f.flush()

            return batch

    async def cancel_batches(self):
        if not os.path.exists(self.submitted_batch_objects_file):
            logger.warning("No batches to be cancelled, but cancel_batches=True.")
        else:
            logger.info(f"Batch objects file exists, cancelling all batches.")
            batch_ids = []
            with open(self.submitted_batch_objects_file, "r") as f:
                for line in f:
                    batch_obj = json.loads(line.strip())
                    batch_ids.append(batch_obj["id"])
            tasks = [self.cancel_batch(batch_id) for batch_id in batch_ids]
            results = await asyncio.gather(*tasks)
            failed = abs(sum(results))
            logger.warning(
                f"{len(results)-failed:,} out of {len(results):,} batches successfully cancelled"
            )

    async def retrieve_batch(self, batch_id: str) -> GenericBatch:
        try:
            batch = await self.client.batches.retrieve(batch_id)
        except NotFoundError:
            logger.warning(
                f"batch object {batch_id} not found. "
                f"Your API key (***{self.client.api_key[-4:]}) might not have access to this batch."
            )
        self._validate_batch_status(batch.status)
        return batch

    async def check_batch_status(self, batch_id: str) -> GenericBatch | None:
        """
        Checks the current status of a batch job.

        Args:
            batch_id (str): The ID of the batch to check

        Returns:
            Batch | None: The batch object if completed (including failures), None if in progress

        Side Effects:
            - Updates tracker with current batch status
            - Updates request completion counts
        """
        async with self.semaphore:
            batch = await self.client.batches.retrieve(batch_id)
            self.tracker.update_submitted(batch)

            n_completed_requests = batch.request_counts.completed
            n_failed_requests = batch.request_counts.failed
            n_total_requests = batch.request_counts.total

            logger.debug(
                f"Batch {batch.id} status: {batch.status} requests: "
                f"{n_completed_requests}/{n_failed_requests}/{n_total_requests} "
                "completed/failed/total"
            )

            finished_statuses = ["completed", "failed", "expired", "cancelled"]
            batch_returned = batch.status in finished_statuses
            if not self._validate_batch_status(batch.status):
                logger.warning(f"Unknown batch status: {batch.status}")

            if batch_returned:
                logger.debug(f"Batch {batch.id} returned with status: {batch.status}")
                self.tracker.mark_as_finished(batch)
                return batch

    async def delete_file(self, file_id: str, semaphore: asyncio.Semaphore):
        """
        Deletes a file from OpenAI's storage.

        Args:
            file_id (str): The ID of the file to delete
            semaphore (asyncio.Semaphore): Semaphore to limit concurrent operations
        """
        async with semaphore:
            try:
                delete_response = await self.client.files.delete(file_id)
                if delete_response.deleted:
                    logger.debug(f"Deleted file {file_id}")
                else:
                    logger.warning(f"Failed to delete file {file_id}")
            except NotFoundError:
                # This is fine, the file may have been deleted already. Deletion should be best-effort.
                logger.warning(f"Trying to delete file {file_id} but it was not found.")

    async def download_batch(self, batch: GenericBatch) -> str | None:
        file_content = None
        async with self.semaphore:
            # Completed batches have an output file
            if batch.status == "completed" and batch.output_file_id:
                file_content = await self.client.files.content(batch.output_file_id)
                logger.debug(f"Batch {batch.id} completed and downloaded")

            # Failed batches with an error file
            elif batch.status == "failed" and batch.error_file_id:
                file_content = await self.client.files.content(batch.error_file_id)
                logger.warning(f"Batch {batch.id} failed\n. Errors will be parsed below.")
                if self.delete_failed_batch_files:
                    await self.delete_file(batch.input_file_id, self.semaphore)
                    await self.delete_file(batch.error_file_id, self.semaphore)

            # Failed batches without an error file
            elif batch.status == "failed" and not batch.error_file_id:
                errors = "\n".join([str(error) for error in batch.errors.data])
                logger.error(
                    f"Batch {batch.id} failed and likely failed validation. "
                    f"Batch errors: {errors}. "
                    f"Check https://platform.openai.com/batches/{batch.id} for more details."
                )
                if self.delete_failed_batch_files:
                    await self.delete_file(batch.input_file_id, self.semaphore)

            # Cancelled or expired batches
            elif batch.status == "cancelled" or batch.status == "expired":
                logger.warning(f"Batch {batch.id} was cancelled or expired")
                if self.delete_failed_batch_files:
                    await self.delete_file(batch.input_file_id, self.semaphore)

        return file_content

    @staticmethod
    def _validate_batch_status(status: str) -> bool:
        # See https://github.com/openai/openai-python/blob/995cce048f9427bba4f7ac1e5fc60abbf1f8f0b7/src/openai/types/batch.py#L40C1-L41C1
        # for all possible batch statuses
        return status in [
            "completed",
            "failed",
            "expired",
            "cancelled",
            "validating",
            "finalizing",
            "cancelling",
            "in_progress",
        ]
