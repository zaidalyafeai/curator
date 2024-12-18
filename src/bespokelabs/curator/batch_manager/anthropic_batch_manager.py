import asyncio
import datetime
import json
import logging

import litellm
from anthropic import AsyncAnthropic
from anthropic.types import BetaMessageBatch

from bespokelabs.curator.batch_manager.base_batch_manager import BaseBatchManager
from bespokelabs.curator.types.generic_batch_object import GenericBatchObject
from bespokelabs.curator.types.generic_request import GenericRequest
from bespokelabs.curator.request_processor.base_request_processor import parse_response_message
from bespokelabs.curator.types.token_usage import TokenUsage
from bespokelabs.curator.types.generic_response import GenericResponse

logger = logging.getLogger(__name__)

# https://docs.anthropic.com/en/api/creating-message-batches
# https://docs.anthropic.com/en/docs/build-with-claude/message-batches#batch-limitations
MAX_REQUESTS_PER_BATCH = 100_000
MAX_BYTES_PER_BATCH = 256 * 1024 * 1024

MAX_CONCURRENT_BATCH_OPERATIONS = 100  # this might need to be reduced
MAX_RETRIES_PER_OPERATION = 10

BATCH_STATUSES = ["in_progress", "canceling", "ended"]
REQUEST_STATUSES = ["succeeded", "errored", "cancelled", "expired"]


class AnthropicBatchManager(BaseBatchManager):
    def __init__(
        self,
        working_dir: str,
        check_interval: int = 60,
        delete_successful_batch_files: bool = False,
        delete_failed_batch_files: bool = False,
    ) -> None:
        """Initialize BatchManager to handle OpenAI batch processing operations.

        Args:
            working_dir (str): Directory for storing batch-related files including requests, responses,
                and tracking files.
            check_interval (int): Time interval (in seconds) between batch status checks.
            delete_successful_batch_files (bool): Whether to delete input/output files from OpenAI
                after successful batch completion.
            delete_failed_batch_files (bool): Whether to delete input/error files from OpenAI
                after batch failure.
        """
        super().__init__(
            working_dir,
            max_concurrent_batch_operations=MAX_CONCURRENT_BATCH_OPERATIONS,
            check_interval=check_interval,
            delete_successful_batch_files=delete_successful_batch_files,
            delete_failed_batch_files=delete_failed_batch_files,
        )
        self.client = AsyncAnthropic(max_retries=MAX_RETRIES_PER_OPERATION)

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
        if generic_request.response_format:
            # TODO(Ryan) how can we support this the way litellm does?
            raise NotImplementedError("response_format is not yet supported for Anthropic")

        params = {
            "model": generic_request.model,
        }
        if self.generic_request.messages[0]["role"] == "system":
            params["system"] = self.generic_request.messages[0]["content"]
            params["messages"] = self.generic_request.messages[1:]
        else:
            params["messages"] = self.generic_request.messages

        if self.temperature is not None:
            params["temperature"] = self.temperature

        if self.top_p is not None:
            params["top_p"] = self.top_p

        if self.presence_penalty is not None:
            raise NotImplementedError("presence_penalty is not yet supported for Anthropic")

        if self.frequency_penalty is not None:
            raise NotImplementedError("frequency_penalty is not yet supported for Anthropic")

        request = {
            "custom_id": str(generic_request.original_row_idx),
            "params": params,
        }

        return request

    def parse_api_specific_response(
        self,
        raw_response: dict,
        generic_request: GenericRequest,
        batch_created_at: datetime.datetime,
    ) -> GenericResponse:
        if raw_response["result"]["type"] != "succeeded":
            response_message = None
            response_errors = [
                raw_response["result"]["type"]
            ]  # no examples of a failed response, we can probably include more information here
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

    def create_batch_file(self, api_specific_requests: list[dict]) -> str:
        """
        Creates a batch file from a list of API-specific requests.

        Args:
            api_specific_requests (list[dict]): List of API-specific request bodies

        Returns:
            str: The encoded file content ready for upload

        Raises:
            ValueError: If the batch file contains more requests than OpenAI supports
        """
        n_requests = len(api_specific_requests)
        if n_requests > MAX_REQUESTS_PER_BATCH:
            raise ValueError(
                f"Batch file contains {n_requests:,} requests, "
                f"which is more than the maximum of {MAX_REQUESTS_PER_BATCH:,} requests per batch that OpenAI supports. "
                f"Preventing batch submission. Please reduce `batch_size`."
            )

        # Join requests with newlines and encode to bytes for upload
        file_content = "\n".join(api_specific_requests).encode()
        file_content_size = len(file_content)
        logger.debug(
            f"Batch file content size: {file_content_size / (1024*1024):.2f} MB ({file_content_size:,} bytes)"
        )
        if file_content_size > MAX_BYTES_PER_BATCH:
            raise ValueError(
                f"Batch file content size {file_content_size:,} bytes "
                f"is greater than the maximum of {MAX_BYTES_PER_BATCH:,} bytes per batch that OpenAI supports. "
                f"Please reduce your batch size or request content size (via prompt_func and response_format)."
            )
        return file_content

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

    async def create_batch(self, batch_file_id: str, metadata: dict) -> GenericBatchObject:
        """
        Creates a batch job with OpenAI using an uploaded file.

        Args:
            batch_file_id (str): ID of the uploaded file to use for the batch
            metadata (dict): Metadata to be included with the batch

        Returns:
            GenericBatchObject: The created batch object from OpenAI

        Raises:
            Exception: If batch creation fails
        """
        try:
            batch_object = await self.client.batches.create(
                input_file_id=batch_file_id,
                endpoint="/v1/chat/completions",
                completion_window="24h",
                metadata=metadata,
            )
            logger.debug(f"Batch submitted with id {batch_object.id}")
        except Exception as e:
            logger.error(f"Error submitting batch: {e}")
            raise e
        return batch_object

    async def submit_batch(self, requests: list[dict], metadata: dict) -> GenericBatchObject:
        """
        Handles the complete batch submission process.

        Args:
            requests (list[dict]): List of API-specific requests to submit
            metadata (dict): Metadata to be included with the batch

        Returns:
            GenericBatchObject: The created batch object from OpenAI

        Side Effects:
            - Updates tracker with submitted batch status
            - Appends batch object to submitted_batch_objects_file
        """
        async with self.semaphore:
            file_content = self.create_batch_file(requests)
            batch_file_upload = await self.upload_batch_file(file_content)
            batch_object = await self.create_batch(batch_file_upload.id, metadata)

            # Simplified file writing
            with open(self.submitted_batch_objects_file, "a") as f:
                json.dump(batch_object.model_dump(), f, default=str)
                f.write("\n")
                f.flush()

            return batch_object

    async def retrieve_batch(self, batch_id: str) -> GenericBatchObject:
        try:
            batch_object = await self.client.batches.retrieve(batch_id)
        except Exception as e:
            raise e
        return batch_object

    async def cancel_batch(self, batch_id: str) -> int:
        async with self.semaphore:
            batch_object = await self.retrieve_batch(batch_id)
            if batch_object.status == "completed":
                logger.warning(f"Batch {batch_id} is already completed, cannot cancel")
                return 0
            try:
                await self.client.batches.cancel(batch_id)
                logger.info(f"Successfully cancelled batch: {batch_id}")
                return 0
            except Exception as e:
                error_msg = str(e)
                logger.error(f"Failed to cancel batch {batch_id}: {error_msg}")
                return -1

    async def check_batch_status(self, batch_id: str) -> GenericBatchObject | None:
        """
        Checks the current status of a batch job.

        Args:
            batch_id (str): The ID of the batch to check

        Returns:
            GenericBatchObject | None: The batch object if completed (including failures), None if in progress

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

    async def download_batch(self, batch: GenericBatchObject) -> str | None:
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
