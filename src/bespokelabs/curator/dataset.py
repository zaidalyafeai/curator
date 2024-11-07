import json
import logging
import os
import traceback
from typing import Any, Callable, Dict, Iterable, List, Optional, TypeVar

from datasets import Dataset as HFDataset
from datasets.arrow_writer import ArrowWriter
from pydantic import BaseModel
from tqdm import tqdm

from bespokelabs.curator.prompter import Prompter
from bespokelabs.curator.request_processor.generic_response import GenericResponse

T = TypeVar("T")


class Dataset:
    def __init__(self, working_dir: str, prompter: Prompter):
        self.working_dir = working_dir
        self.prompter = prompter

    def __init__(
        self, iterable: Iterable[Dict[str, Any] | BaseModel], prompter: Prompter
    ):
        self.iterable = iterable
        self.prompter = prompter

    def __iter__(self):
        if self.iterable:
            yield from self.iterable
            return

        if self.working_dir:
            response_file = f"{self.working_dir}/responses.jsonl"

            for line in open(response_file, "r"):
                response = GenericResponse.model_validate_json(json.loads(line))
                yield self.prompter.parse_func(response.row, response.response)

    def to_huggingface(self) -> None:
        """
        Returns a HuggingFace Dataset

        Returns:
            Dataset: Completed dataset
        """

        total_responses_count = 0
        failed_responses_count = 0

        os.makedirs(self.working_dir, exist_ok=True)

        dataset_file = f"{self.working_dir}/dataset.arrow"
        response_file = f"{self.working_dir}/responses.jsonl"

        # Process all response files
        with ArrowWriter(path=dataset_file) as writer:
            if not os.path.exists(response_file):
                raise ValueError(f"Responses file {response_file} does not exist")

            with open(response_file, "r") as f_in:
                for line in f_in:
                    total_responses_count += 1
                    try:
                        response = GenericResponse.model_validate_json(json.loads(line))

                        # TODO(Ryan): We can make this more sophisticated by making response_generic a class
                        if response is None:
                            failed_responses_count += 1
                            continue

                        # TODO(Ryan): Error handling on the parse function
                        dataset_rows = self.prompter.parse_func(
                            response.row, response.response
                        )
                        for row in dataset_rows:
                            # NOTE(Ryan): This throws a strange error if there are null values in the row
                            writer.write(row)

                    # NOTE(Ryan): Catching naked exceptions is bad practice, but this prevents the program from crashing
                    # TODO(Ryan): Add in handling for specific exceptions as they come up
                    except Exception as e:
                        logging.warning(f"Error: {e}\nFull response: {response}")
                        continue

            logging.info(
                f"Read {total_responses_count} responses, {failed_responses_count} failed"
            )
            logging.info("Finalizing writer")

            if failed_responses_count == total_responses_count:
                raise ValueError("All requests failed")

            # NOTE(Ryan): This throws an error if all rows were None
            # TODO(Ryan): Look at what this file looks like before finalize. What happens during finalize?
            writer.finalize()

        return HFDataset.from_file(dataset_file)

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
                        parsed_output = prompter.parse_func(
                            metadata["sample"], response
                        )
                    else:
                        parsed_output = response

                    if isinstance(parsed_output, list):
                        samples.extend(
                            [
                                (metadata["request_idx"], output)
                                for output in parsed_output
                            ]
                        )
                    else:
                        samples.append((metadata["request_idx"], parsed_output))

                except Exception as e:
                    logging.warning(
                        f"Error: {e}. Full traceback: {traceback.format_exc()}. Full response: {response}"
                    )
                    continue

        logging.debug(f"Read {total_count} responses, {failed_count} failed")
        # Sort by idx then return only the responses
        samples.sort(key=lambda x: x[0])
        samples = [sample[1] for sample in samples]
        if failed_count == total_count:
            raise ValueError("All requests failed")
        return samples
