"""Curator: Bespoke Labs Synthetic Data Generation Library."""

import asyncio
import inspect
import json
import logging
import math
import os
import time
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from typing import Any, Callable, Dict, Iterable, Optional, Type, TypeVar, Union
import traceback

from datasets import Dataset
from pydantic import BaseModel
from xxhash import xxh64

from bespokelabs.curator.api_request_parallel_processor import (
    process_api_requests_from_file,
)
from bespokelabs.curator.db import MetadataDB

T = TypeVar("T")


class Prompter:
    """Interface for prompting LLMs."""

    def __init__(
        self,
        model_name: str,
        prompt_func: Callable[[Union[Dict[str, Any], BaseModel]], Dict[str, str]],
        parse_func: Optional[
            Callable[
                [Union[Dict[str, Any], BaseModel], Union[Dict[str, Any], BaseModel]], T
            ]
        ] = None,
        response_format: Optional[Type[BaseModel]] = None,
    ):
        """Initialize a Prompter.

        Args:
            model_name (str): The name of the LLM to use
            prompt_func (Callable[[Dict[str, Any]], Dict[str, str]]): A function that takes a single row
                and returns a dict with "system_prompt" and "user_prompt"
            parse_func (Callable[[Dict[str, Any], Any], T]): A function that takes the input row and
                response object and returns the parsed output
            response_format (Optional[Type[BaseModel]]): A Pydantic model specifying the
                response format from the LLM.
        """
        self.model_name = model_name

        prompt_sig = inspect.signature(prompt_func)
        if len(prompt_sig.parameters) > 1:
            raise ValueError(
                f"prompt_func must take one argument or less, got {len(prompt_sig.parameters)}"
            )

        if parse_func is not None:
            parse_sig = inspect.signature(parse_func)
            if len(parse_sig.parameters) != 2:
                raise ValueError(
                    f"parse_func must take exactly 2 arguments, got {len(parse_sig.parameters)}"
                )

        self.prompt_func = prompt_func
        self.parse_func = parse_func
        self.response_format = response_format

    def get_request_object(
        self, row: Dict[str, Any] | BaseModel, idx: int
    ) -> Dict[str, Any]:
        """Format the request object based off Prompter attributes."""
        sig = inspect.signature(self.prompt_func)
        if len(sig.parameters) == 0:
            prompts = self.prompt_func()
        elif len(sig.parameters) == 1:
            prompts = self.prompt_func(row)
        else:
            raise ValueError(
                f"Prompting function {self.prompt_func} must have 0 or 1 arguments."
            )

        messages = []
        system_prompt = prompts.get("system_prompt", "You are a helpful AI assistant.")
        messages.append({"role": "system", "content": system_prompt})

        if "user_prompt" not in prompts:
            raise ValueError("user_prompt is required")
        messages.append({"role": "user", "content": prompts["user_prompt"]})

        # Convert BaseModel to dict for serialization
        if isinstance(row, BaseModel):
            row = row.model_dump()

        if self.response_format:
            # OpenAI API
            # https://platform.openai.com/docs/api-reference/chat/create#chat-create-response_format
            request = {
                "model": self.model_name,
                "messages": messages,
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {
                        # TODO(ryan): not sure if this should be something else.
                        # TODO(ryan): also not sure if we should use strict: True
                        "name": "output_schema",
                        "schema": self.response_format.model_json_schema(),
                    },
                },
                "metadata": {"request_idx": idx, "sample": row},
            }
        else:
            request = {
                "model": self.model_name,
                "messages": messages,
                "metadata": {"request_idx": idx, "sample": row},
            }
        return request

    def __call__(self, dataset: Iterable = []):
        """Run completions on a dataset."""
        return _completions(dataset, self)


def _completions(
    dataset: Iterable = (),
    prompter: Prompter = None,
    name: Optional[str] = None,
    resume: bool = True,
) -> "Dataset":
    """
    Apply structured completions in parallel to a dataset using specified model and
    prompts.

    Args:
        dataset (Iterable): A dataset consisting of a list of items to apply completions
        prompter (Prompter): A Prompter that contains the logic for formatting each
            item in the dataset
        name (str): Name of the task
        resume (bool): Whether to resume from the previous completions run. If True,
            we use a fingerprint from the input dataset and the prompter to resume
            from a previous run that matches the same fingerprint.

    Returns:
        Iterable: A list of structured outputs from the completions
    """
    if prompter is None:
        raise ValueError("Prompter must be provided")

    curator_cache_dir = os.environ.get(
        "BELLA_CACHE_DIR", os.path.expanduser("~/.cache/curator")
    )

    dataset_hash = _hash_dataset(dataset)
    prompt_func_hash = _get_function_hash(prompter.prompt_func)
    parse_func_hash = _get_function_hash(prompter.parse_func)

    fingerprint_str = "_".join(
        [
            str(dataset_hash),
            str(prompt_func_hash),
            str(parse_func_hash),
            str(prompter.model_name),
            str(prompter.response_format.schema_json() if prompter.response_format else "plain_text"),
        ]
    )

    fingerprint = xxh64(fingerprint_str.encode("utf-8")).hexdigest()

    name = f"{name.replace(' ', '-')}--{fingerprint}" if name else fingerprint
    requests_path = os.path.join(curator_cache_dir, f"{name}/requests.jsonl")
    responses_path = os.path.join(curator_cache_dir, f"{name}/responses.jsonl")
    metadata_db_path = os.path.join(curator_cache_dir, "metadata.db")
    metadata_db = MetadataDB(metadata_db_path)

    # Get the source code of the prompt function
    prompt_func_source = inspect.getsource(prompter.prompt_func)
    if prompter.parse_func is not None:
        parse_func_source = inspect.getsource(prompter.parse_func)
    else:
        parse_func_source = ""

    metadata_dict = {
        "timestamp": datetime.now().isoformat(),
        "dataset_hash": dataset_hash,
        "prompt_func": prompt_func_source,
        "parse_func": parse_func_source,
        "model_name": prompter.model_name,
        "response_format": prompter.response_format.schema_json() if prompter.response_format else "text",
        "run_hash": fingerprint,
    }
    metadata_db.store_metadata(metadata_dict)

    _create_requests_file(dataset, requests_path, prompter, resume)
    asyncio.run(
        process_api_requests_from_file(
            requests_filepath=requests_path,
            save_filepath=responses_path,
            request_url="https://api.openai.com/v1/chat/completions",
            max_attempts=5,
            resume=True,
            model=prompter.model_name,
        )
    )
    return _parse_responses_file(prompter, responses_path)


def _create_requests_file(
    dataset: Iterable, requests_file: str, prompter: Prompter, resume: bool = True
):
    if os.path.exists(requests_file):
        if resume:
            print(f"Loading existing jobs from {requests_file}")
            logging.debug(
                f"Alternatively, delete the jobs file and re-run the annotator: `rm -rf {requests_file}`"
            )
            # count existing jobs in file and print first job
            with open(requests_file, "r") as f:
                num_jobs = sum(1 for _ in f)
                f.seek(0)
                first_job = json.loads(f.readline())
            logging.debug(f"Found {num_jobs} jobs in {requests_file}")
            logging.debug("Example job:")
            logging.debug(json.dumps(first_job, indent=2))
        else:
            error_message = (
                f"Existing job file {requests_file}. "
                f"Delete the jobs file and re-run the annotator: `rm -rf {requests_file}`. "
                f"Or run the annotator with the --resume flag to continue from the previous run."
            )
            raise ValueError(error_message)
    else:
        os.makedirs(os.path.dirname(requests_file), exist_ok=True)
        with open(requests_file, "w") as f:
            if len(dataset) == 0:
                request = prompter.get_request_object(dict(), 0)
                f.write(json.dumps(request) + "\n")
            else:
                for idx, sample in enumerate(dataset):
                    request = prompter.get_request_object(sample, idx)
                    f.write(json.dumps(request) + "\n")
        logging.info(f"Requests file {requests_file} written to disk.")


def _parse_responses_file(prompter: Prompter, responses_file):
    total_count = 0
    failed_count = 0
    samples = []
    with open(responses_file, "r") as f_in:
        for line in f_in:
            total_count += 1
            try:
                # Each response is a tuple of (request, response, metadata) where:
                # - request is the original request object
                # - response is the response from the API
                # - metadata is a dictionary of metadata about the request (such as the request index)
                response = json.loads(line)
                if isinstance(response[1], list):
                    # A failed requests contains a list of all the errors before max_retries
                    logging.info(
                        f"Request {response[2].get('request_idx')} failed due to errors: {response[1]}"
                    )
                    failed_count += 1
                    continue

                response_message = response[1]["choices"][0]["message"]["content"]
                metadata = response[2]

                if prompter.response_format:
                    response = prompter.response_format.model_validate_json(
                        response_message
                    )
                else:
                    response = response_message

                if prompter.parse_func:
                    parsed_output = prompter.parse_func(metadata["sample"], response)
                else:
                    parsed_output = response

                if isinstance(parsed_output, list):
                    samples.extend(
                        [(metadata["request_idx"], output) for output in parsed_output]
                    )
                else:
                    samples.append((metadata["request_idx"], parsed_output))

            except Exception as e:
                logging.warning(f"Error: {e}. Full traceback: {traceback.format_exc()}. Full response: {response}")
                continue

    logging.debug(f"Read {total_count} responses, {failed_count} failed")
    # Sort by idx then return only the responses
    samples.sort(key=lambda x: x[0])
    samples = [sample[1] for sample in samples]
    if failed_count == total_count:
        raise ValueError("All requests failed")
    return samples


def _hash_chunk(chunks: list) -> list:
    """Hash a chunk of data."""

    def _json_dumps_row(row):
        if isinstance(row, BaseModel):
            row = row.model_dump()
        return json.dumps(row, sort_keys=True)

    chunks = [_json_dumps_row(row) for row in chunks]
    chunk_str = "|||".join(chunks)
    return xxh64(chunk_str).hexdigest()


def _hash_dataset(dataset: Iterable):
    """Hash a dataset to a consistent value using parallel processing."""
    start = time.perf_counter_ns()

    # Convert to list and determine chunking parameters
    dataset_list = list(dataset)
    if len(dataset_list) == 0:
        return xxh64("").hexdigest()

    num_cores = 4
    total_size = len(dataset_list)
    chunk_size = math.ceil(total_size / (num_cores * 4))  # 4 chunks per core

    chunks = [
        dataset_list[i : i + chunk_size] for i in range(0, total_size, chunk_size)
    ]

    # Process chunks in parallel
    with ProcessPoolExecutor(max_workers=num_cores) as executor:
        chunk_hash = list(executor.map(_hash_chunk, chunks))
        chunk_hash_str = "|||".join(chunk_hash)
        hash_value = xxh64(chunk_hash_str).hexdigest()

    logging.debug(
        f"Dataset hash time: {(time.perf_counter_ns() - start) / 1e6:.6f} milliseconds"
    )
    return hash_value


def _get_function_hash(func) -> str:
    """Get a hash of a function's source code."""
    if func is None:
        return xxh64("").hexdigest()

    return xxh64(inspect.getsource(func)).hexdigest()
