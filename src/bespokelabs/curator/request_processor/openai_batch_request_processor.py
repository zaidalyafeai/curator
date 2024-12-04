import asyncio
import datetime
import json
import logging
from dataclasses import dataclass, field
from typing import Callable

import aiofiles
import glob
import os
import litellm
from openai import AsyncOpenAI
from openai.types import Batch
from tqdm import tqdm

from bespokelabs.curator.dataset import Dataset
from bespokelabs.curator.prompter.prompt_formatter import PromptFormatter
from bespokelabs.curator.request_processor.base_request_processor import (
    BaseRequestProcessor,
    GenericRequest,
    GenericResponse,
    parse_response_message,
)
from bespokelabs.curator.request_processor.event_loop import run_in_event_loop
from bespokelabs.curator.request_processor.generic_response import TokenUsage

logger = logging.getLogger(__name__)

MAX_REQUESTS_PER_BATCH = 50_000
MAX_BYTES_PER_BATCH = 200 * 1024 * 1024

# NOTE(Ryan): This allows us to stay under the rate limit when submitting ~1,000 batches at a time
# When submitting >1,000 batches the batch submission and batch download operations get rate limited
MAX_CONCURRENT_BATCH_OPERATIONS = 100
MAX_RETRIES_PER_OPERATION = 5


class OpenAIBatchRequestProcessor(BaseRequestProcessor):
    def __init__(
        self,
        batch_size: int,
        model: str,
        delete_successful_batch_files: bool,
        delete_failed_batch_files: bool,
        temperature: float | None = None,
        top_p: float | None = None,
        batch_check_interval: int = 60,
        url: str = "https://api.openai.com/v1/chat/completions",
        presence_penalty: float | None = None,
        frequency_penalty: float | None = None,
    ):
        if batch_size > MAX_REQUESTS_PER_BATCH:
            raise ValueError(
                f"batch_size {batch_size} is greater than the maximum of "
                f"{MAX_REQUESTS_PER_BATCH:,} requests per batch that OpenAI supports. "
                f"Please set your batch_size to be less than or equal to {MAX_REQUESTS_PER_BATCH:,}."
            )
        super().__init__(batch_size)
        self.url: str = url
        self.check_interval: int = batch_check_interval
        self.temperature: float | None = temperature
        self.top_p: float | None = top_p
        self.presence_penalty: float | None = presence_penalty
        self.frequency_penalty: float | None = frequency_penalty
        self._file_lock = asyncio.Lock()
        self.delete_successful_batch_files: bool = delete_successful_batch_files
        self.delete_failed_batch_files: bool = delete_failed_batch_files

    def get_rate_limits(self) -> dict:
        """
        Function to get rate limits for a given annotator. Not available via response headers, so
        the following is based on tier 5 limits on Nov 6th, 2024.

        These rate limits vary per model
        and are determined by your organization's usage tier. View the following:
        https://platform.openai.com/docs/guides/rate-limits/usage-tiers
        https://platform.openai.com/settings/organization/limits

        Args:
            model (str): The model for which to get the rate limits.
            request_url (str): The request URL for which to get the rate limits.

        Returns:
            tuple[int, int]: A tuple containing the maximum number of requests and tokens per minute.
        """
        model_tpd = {
            "gpt-3.5-turbo": 5_000_000_000,
            "gpt-3.5-turbo-0125": 5_000_000_000,
            "gpt-3.5-turbo-1106": 5_000_000_000,
            "gpt-3.5-turbo-16k": 5_000_000_000,
            "gpt-3.5-turbo-instruct": 200_000,
            "gpt-3.5-turbo-instruct-0914": 200_000,
            "gpt-4": 150_000_000,
            "gpt-4-0613": 150_000_000,
            "gpt-4-turbo": 300_000_000,
            "gpt-4o": 10_000_000_000,
            "gpt-4o-mini": 15_000_000_000,
        }

        if self.model not in model_tpd:
            tpd = 1_000_000_000
        else:
            tpd = model_tpd[self.model]

        logger.debug(f"Automatically set max_tokens_per_day to {tpd}, model: {self.model} ")

        rate_limits = {"max_tokens_per_day": tpd}

        return rate_limits

    def create_api_specific_request(self, generic_request: GenericRequest) -> dict:
        """
        Creates a API-specific request body from a generic request body.

        Using the api_parallel_processor, we can store whatever we want in the metadata. We will store both the row and the index.
        This is so we can later construct the new dataset row.

        Returns:
            dict: API specific request body
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

    async def requests_from_generic_request_file(self, request_file: str) -> list[dict]:
        api_specific_requests = []

        async with aiofiles.open(request_file, "r") as file:
            file_content = await file.read()
            for line in file_content.splitlines():
                request = GenericRequest.model_validate_json(line)
                api_specific_request = self.create_api_specific_request(request)
                api_specific_requests.append(json.dumps(api_specific_request))

        return api_specific_requests

    async def manage_batches(
        self, request_files: set[str], working_dir: str, prompt_formatter: PromptFormatter
    ):
        # submit and download remaining batches
        if len(request_files) == 0:
            return

        batch_manager = BatchManager(
            working_dir,
            self.check_interval,
            prompt_formatter,
            delete_successful_batch_files=self.delete_successful_batch_files,
            delete_failed_batch_files=self.delete_failed_batch_files,
        )
        await batch_manager.submit_batches_from_request_files(
            request_files, self.requests_from_generic_request_file
        )
        await batch_manager.poll_and_download_batches()
        await batch_manager.close_client()

    def run(
        self,
        dataset: Dataset | None,
        working_dir: str,
        parse_func_hash: str,
        prompt_formatter: PromptFormatter,
    ) -> Dataset:
        """
        Uses the API to completing the specific map by calling the LLM.

        Args:
            dataset (Dataset): Dataset that is being mapped over
            working_dir (str): Working directory to save files (requests.jsonl, responses.jsonl, dataset.arrow)
            parse_func_hash (str): Hash of the parse_func to be used as the dataset file name
            prompt_formatter (PromptFormatter): Prompt formatter to be used to format the prompt

        Returns:
            Dataset: Completed dataset
        """
        # load from already completed dataset
        output_dataset = self.attempt_loading_cached_dataset(working_dir, parse_func_hash)
        if output_dataset is not None:
            return output_dataset

        request_files = set(self.create_request_files(dataset, working_dir, prompt_formatter))
        run_in_event_loop(self.manage_batches(request_files, working_dir, prompt_formatter))

        return self.create_dataset_files(working_dir, parse_func_hash, prompt_formatter)


@dataclass
class BatchStatusTracker:
    # total number of request files
    n_total_batches: int = 0

    # Request files are in one of two states: unsubmitted, submitted
    unsubmitted_request_files: set = field(default_factory=set)
    submitted_request_files: set = field(default_factory=set)

    # Batches are in one of three states: submitted, finished, downloaded
    submitted_batch_ids: set = field(default_factory=set)
    finished_batch_ids: set = field(default_factory=set)
    downloaded_batch_ids: set = field(default_factory=set)

    # Batches can finish in 4 ways: completed, failed, cancelled, expired
    # completed and failed can be downloaded, cancelled and expired cannot
    # for simplicity not tracking these

    # Requests are in one of two states: finished and downloaded
    # requests can be completed or failed, for simplicity not tracking these
    n_total_requests: int = 0
    n_finished_requests: int = 0
    n_downloaded_requests: int = 0


def request_file_to_response_file(request_file: str, working_dir: str) -> str:
    request_file_idx = request_file.split("/")[-1].split("_", 1)[1]
    return f"{working_dir}/responses_{request_file_idx}"


def response_file_to_request_file(response_file: str, working_dir: str) -> str:
    response_file_idx = response_file.split("/")[-1].split("_", 1)[1]
    return f"{working_dir}/requests_{response_file_idx}"


class BatchManager:
    def __init__(
        self,
        working_dir: str,
        check_interval: int,
        prompt_formatter: PromptFormatter,
        delete_successful_batch_files: bool,
        delete_failed_batch_files: bool,
    ) -> None:
        """Initialize BatchWatcher with batch objects file and check interval.

        Args:
            working_dir (str): Directory containing the batch objects JSON file.
            check_interval (int): Time interval (in seconds) to check batch status.
            prompt_formatter (PromptFormatter): Prompt formatter to be used to format the prompt
            n_submitted_requests (int): Number of requests submitted to the batches (used for progress bar)
        """
        self.client = AsyncOpenAI()
        self.check_interval = check_interval
        self.working_dir = working_dir
        self.tracker = BatchStatusTracker()
        self.prompt_formatter = prompt_formatter
        self.semaphore = asyncio.Semaphore(MAX_CONCURRENT_BATCH_OPERATIONS)
        self.delete_successful_batch_files = delete_successful_batch_files
        self.delete_failed_batch_files = delete_failed_batch_files
        self._file_lock = asyncio.Lock()
        self.batch_objects_file = f"{working_dir}/batch_objects_{self.client.api_key[-4:]}.jsonl"
        self.batch_id_to_request_file_name = {}
        self.batch_submit_pbar: tqdm | None = None
        self.request_pbar: tqdm | None = None

    async def close_client(self):
        await self.client.close()

    async def create_batch_file(self, api_specific_requests: list[dict], metadata: dict) -> str:
        n_requests = len(api_specific_requests)
        if n_requests > MAX_REQUESTS_PER_BATCH:
            raise ValueError(
                f"Batch file with metadata {metadata} contains {n_requests:,} requests, "
                f"which is more than the maximum of {MAX_REQUESTS_PER_BATCH:,} requests per batch that OpenAI supports. "
                f"Preventing batch submission."
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

    async def create_batch(self, batch_file_id: str, metadata: dict) -> dict:
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

    async def submit_batch(self, requests: list[dict], metadata: dict) -> dict:
        async with self.semaphore:
            file_content = await self.create_batch_file(requests, metadata)
            batch_file_upload = await self.upload_batch_file(file_content)
            batch_object = await self.create_batch(batch_file_upload.id, metadata)

            async with self._file_lock:
                async with aiofiles.open(self.batch_objects_file, "a") as f:
                    await f.write(json.dumps(batch_object.model_dump(), default=str) + "\n")
                    await f.flush()

            self.tracker.n_total_requests += len(requests)
            self.tracker.submitted_batch_ids.add(batch_object.id)

            return batch_object

    async def requests_from_api_specific_request_file(self, request_file: str) -> list[dict]:
        return (await aiofiles.open(request_file, "r").read()).splitlines()

    async def submit_batch_from_request_file(
        self,
        request_file: str,
        requests_from_request_file_func: Callable = requests_from_api_specific_request_file,
    ):
        metadata = {"request_file_name": request_file}
        requests = await requests_from_request_file_func(request_file)
        batch_object = await self.submit_batch(requests, metadata)
        self.batch_id_to_request_file_name[batch_object.id] = request_file
        self.tracker.unsubmitted_request_files.remove(request_file)
        self.tracker.submitted_request_files.add(request_file)
        self.batch_submit_pbar.update(1)

    async def submit_batches_from_request_files(
        self,
        request_files: set[str],
        requests_from_request_file_func: Callable = requests_from_api_specific_request_file,
    ):
        self.tracker.n_total_batches = len(request_files)
        self.tracker.unsubmitted_request_files = request_files.copy()

        # resume already downloaded batches
        downloaded_response_files = set(glob.glob(f"{self.working_dir}/responses_*.jsonl"))
        for response_file in downloaded_response_files:
            request_file = response_file_to_request_file(response_file, self.working_dir)
            logger.debug(f"File {response_file} already downloaded.")
            if request_file in self.tracker.unsubmitted_request_files:
                async with aiofiles.open(response_file, "r") as f:
                    n_requests_in_file = 0
                    async for _ in f:
                        n_requests_in_file += 1
                self.tracker.unsubmitted_request_files.remove(request_file)
                self.tracker.submitted_request_files.add(request_file)
                self.tracker.downloaded_batch_ids.add(request_file)
                self.tracker.n_total_requests += n_requests_in_file
                self.tracker.n_downloaded_requests += n_requests_in_file

            # TODO(Ryan): alternatively if this is slow, read the request counts from the batch objects in all batch objects files
            # Also this would allows us to differentiate between completed and failed requests

        if len(self.tracker.downloaded_batch_ids) > 0:
            logger.info(
                f"{len(self.tracker.downloaded_batch_ids):,} out of {self.tracker.n_total_batches:,} batches already downloaded."
            )

        # resume already submitted batches
        if os.path.exists(self.batch_objects_file):
            with open(self.batch_objects_file, "r") as f:
                for line in f:
                    batch_object = Batch.model_validate(json.loads(line))
                    request_file_name = batch_object.metadata["request_file_name"]
                    logger.debug(
                        f"Batch job already submitted for request file {request_file_name}"
                    )
                    if request_file_name in self.tracker.unsubmitted_request_files:
                        self.tracker.unsubmitted_request_files.remove(request_file_name)
                        self.tracker.submitted_request_files.add(request_file_name)
                        self.tracker.submitted_batch_ids.add(batch_object.id)
                        self.tracker.n_total_requests += batch_object.request_counts.total

        if len(self.tracker.submitted_batch_ids) > 0:
            logger.info(
                f"{len(self.tracker.submitted_batch_ids):,} out of {self.tracker.n_total_batches - len(self.tracker.downloaded_batch_ids):,} remaining batches are already submitted."
            )
        if len(self.tracker.unsubmitted_request_files) == 0:
            return

        # submit remaining batches
        self.batch_submit_pbar = tqdm(
            total=len(self.tracker.unsubmitted_request_files),
            desc="Submitting batches",
            unit="batch",
        )
        tasks = [
            self.submit_batch_from_request_file(f, requests_from_request_file_func)
            for f in self.tracker.unsubmitted_request_files
        ]
        await asyncio.gather(*tasks)

        assert self.tracker.unsubmitted_request_files == set()

        logger.debug(f"All batch objects submitted and written to {self.batch_objects_file}")

    async def check_batch_status(self, batch_id: str) -> Batch | None:
        """Check the status of a batch by its ID.

        Args:
            batch_id (str): The ID of the batch to check.

        Returns:
            Batch: The batch object. None if the batch has not returned yet.
        """
        async with self.semaphore:
            batch = await self.client.batches.retrieve(batch_id)
            assert batch.id == batch_id

            n_completed_requests = batch.request_counts.completed
            n_failed_requests = batch.request_counts.failed
            n_total_requests = batch.request_counts.total

            self.tracker.n_finished_requests += n_completed_requests + n_failed_requests

            logger.debug(
                f"Batch {batch.id} status: {batch.status} requests: "
                f"{n_completed_requests}/{n_failed_requests}/{n_total_requests} "
                "completed/failed/total"
            )

            finished_statuses = ["completed", "failed", "expired", "cancelled"]
            in_progress_statuses = ["validating", "finalizing", "cancelling", "in_progress"]
            batch_returned = batch.status in finished_statuses
            if batch.status not in in_progress_statuses + finished_statuses:
                logger.warning(f"Unknown batch status: {batch.status}")

            if batch_returned:
                logger.debug(f"Batch {batch.id} returned with status: {batch.status}")
                self.tracker.submitted_batch_ids.remove(batch.id)
                self.tracker.finished_batch_ids.add(batch.id)
                return batch

    async def poll_and_download_batches(self) -> None:
        """Monitor the status of batches until all are completed (includes successfully, failed, expired or cancelled)."""
        # progress bar for finished requests
        self.request_pbar = tqdm(
            total=self.tracker.n_total_requests,
            desc="Finished requests in batches",
            unit="request",
        )

        # loop until all batches have been returned
        all_response_files = []
        while len(self.tracker.submitted_batch_ids) > 0:
            # check batch status also updates the tracker
            status_tasks = [
                self.check_batch_status(batch_id) for batch_id in self.tracker.submitted_batch_ids
            ]
            batches_to_download = await asyncio.gather(*status_tasks)
            batches_to_download = filter(None, batches_to_download)

            # update progress bar
            self.request_pbar.n = (
                self.tracker.n_finished_requests + self.tracker.n_downloaded_requests
            )
            self.request_pbar.refresh()

            download_tasks = [
                self.download_batch_to_generic_responses_file(batch)
                for batch in batches_to_download
            ]
            # Failed downloads return None and print any errors that occurred
            all_response_files.extend(await asyncio.gather(*download_tasks))

            if (
                self.tracker.n_finished_requests + self.tracker.n_downloaded_requests
                < self.tracker.n_total_requests
            ):
                logger.debug(
                    f"Batches returned: {len(self.tracker.finished_batch_ids) + len(self.tracker.downloaded_batch_ids)}/{len(self.tracker.submitted_batch_ids) + len(self.tracker.finished_batch_ids) + len(self.tracker.downloaded_batch_ids)} "
                    f"Requests completed: {self.tracker.n_finished_requests + self.tracker.n_downloaded_requests}/{self.tracker.n_total_requests}"
                )
                logger.debug(f"Sleeping for {self.check_interval} seconds...")
                await asyncio.sleep(self.check_interval)

        self.request_pbar.close()
        response_files = filter(None, all_response_files)
        if len(self.tracker.downloaded_batch_ids) == 0 or not response_files:
            raise RuntimeError(
                "None of the submitted batches completed successfully. "
                "Please check the logs above and https://platform.openai.com/batches for errors."
            )

    async def delete_file(self, file_id: str, semaphore: asyncio.Semaphore):
        """
        Delete a file by its ID.

        Args:
            file_id (str): The ID of the file to delete.
            semaphore (asyncio.Semaphore): Semaphore to limit concurrent operations.
        """
        async with semaphore:
            delete_response = await self.client.files.delete(file_id)
            if delete_response.deleted:
                logger.debug(f"Deleted file {file_id}")
            else:
                logger.warning(f"Failed to delete file {file_id}")

    async def download_batch_to_generic_responses_file(self, batch: Batch) -> str | None:
        """Download the result of a completed batch to file.

        To prevent an accumulation of files, we delete the batch input and output files
        Without this the 100GB limit for files will be reached very quickly
        The user can control this behavior with delete_successful_batch_files and delete_failed_batch_files

        Args:
            batch: The batch object to download results from.

        Returns:
            str: Path to the downloaded result file.
        """
        async with self.semaphore:
            if batch.status == "completed" and batch.output_file_id:
                file_content = await self.client.files.content(batch.output_file_id)
                logger.debug(f"Batch {batch.id} completed and downloaded")
            elif batch.status == "failed" and batch.error_file_id:
                file_content = await self.client.files.content(batch.error_file_id)
                logger.warning(f"Batch {batch.id} failed\n. Errors will be parsed below.")
                if self.delete_failed_batch_files:
                    await self.delete_file(batch.input_file_id, self.semaphore)
                    await self.delete_file(batch.error_file_id, self.semaphore)
            elif batch.status == "failed" and not batch.error_file_id:
                errors = "\n".join([str(error) for error in batch.errors.data])
                logger.error(
                    f"Batch {batch.id} failed and likely failed validation. "
                    f"Batch errors: {errors}. "
                    f"Check https://platform.openai.com/batches/{batch.id} for more details."
                )
                if self.delete_failed_batch_files:
                    await self.delete_file(batch.input_file_id, self.semaphore)
                return None
            elif batch.status == "cancelled" or batch.status == "expired":
                logger.warning(f"Batch {batch.id} was cancelled or expired")
                if self.delete_failed_batch_files:
                    await self.delete_file(batch.input_file_id, self.semaphore)
                return None

            # Naming is consistent with the request file (e.g. requests_0.jsonl -> responses_0.jsonl)
            request_file = self.batch_id_to_request_file_name[batch.id]
            response_file = request_file_to_response_file(request_file, self.working_dir)

            generic_request_map = {}
            request_creation_times = {}  # Track creation times for requests
            with open(request_file, "r") as f:
                for line in f:
                    generic_request = GenericRequest.model_validate_json(line)
                    generic_request_map[generic_request.original_row_idx] = generic_request
                    request_creation_times[generic_request.original_row_idx] = (
                        datetime.datetime.now()
                    )

            with open(response_file, "w") as f:
                for raw_response in file_content.text.splitlines():
                    raw_response = json.loads(raw_response)
                    request_idx = int(raw_response["custom_id"])
                    generic_request = generic_request_map[request_idx]

                    # TODO(Ryan): Add more specific error handling
                    if raw_response["response"]["status_code"] != 200:
                        logger.warning(
                            f"Request {generic_request} failed with status code {raw_response['response']['status_code']}"
                        )
                        generic_response = GenericResponse(
                            response_message=None,
                            response_errors=[
                                f"Request {generic_request} failed with status code {raw_response['response']['status_code']}"
                            ],
                            raw_response=raw_response,
                            raw_request=None,
                            generic_request=generic_request,
                            created_at=request_creation_times[request_idx],
                            finished_at=datetime.datetime.now(),
                            token_usage=None,
                            response_cost=None,
                        )
                    else:
                        response_body = raw_response["response"]["body"]
                        choices = response_body["choices"]
                        usage = response_body.get("usage", {})

                        token_usage = TokenUsage(
                            prompt_tokens=usage.get("prompt_tokens", 0),
                            completion_tokens=usage.get("completion_tokens", 0),
                            total_tokens=usage.get("total_tokens", 0),
                        )

                        # Calculate cost using litellm
                        cost = litellm.completion_cost(
                            model=generic_request.model,
                            prompt=str(
                                generic_request.messages
                            ),  # Convert messages to string for cost calculation
                            completion=choices[0]["message"]["content"],
                        )
                        # Batch requests are 50% off
                        cost = cost * 0.5

                        response_message = choices[0]["message"]["content"]
                        response_message, response_errors = parse_response_message(
                            response_message, self.prompt_formatter.response_format
                        )

                        generic_response = GenericResponse(
                            response_message=response_message,
                            response_errors=response_errors,
                            raw_response=raw_response,
                            raw_request=None,
                            generic_request=generic_request,
                            created_at=request_creation_times[request_idx],
                            finished_at=datetime.datetime.now(),
                            token_usage=token_usage,
                            response_cost=cost,
                        )
                    f.write(json.dumps(generic_response.model_dump(), default=str) + "\n")

                logger.debug(f"Batch {batch.id} written to {response_file}")

                if self.delete_successful_batch_files:
                    await self.delete_file(batch.input_file_id, self.semaphore)
                    await self.delete_file(batch.output_file_id, self.semaphore)

            self.tracker.n_finished_requests -= batch.request_counts.total
            self.tracker.n_downloaded_requests += batch.request_counts.total
            self.tracker.finished_batch_ids.remove(batch.id)
            self.tracker.downloaded_batch_ids.add(batch.id)

            return response_file
