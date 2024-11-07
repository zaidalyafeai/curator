import json
import logging
import os
from typing import Any, Dict, Iterable, Iterator, List, TypeVar

import pandas as pd
from datasets import Dataset as HFDataset
from datasets.arrow_writer import ArrowWriter
from pydantic import BaseModel

from bespokelabs.curator.prompter.prompt_formatter import PromptFormatter
from bespokelabs.curator.request_processor.generic_response import GenericResponse

T = TypeVar("T")


class Dataset:
    def __init__(
        self,
        iterable: Iterable[Dict[str, Any] | BaseModel] | None = None,
        working_dir: str | None = None,
        prompt_formatter: PromptFormatter | None = None,
    ):
        self.working_dir = working_dir
        self.prompt_formatter = prompt_formatter
        self.iterable = iterable

    def from_iterable(iterable: Iterable[Dict[str, Any] | BaseModel]):
        return Dataset(iterable=iterable)

    def from_working_dir(working_dir: str, prompt_formatter: PromptFormatter):
        return Dataset(working_dir=working_dir, prompt_formatter=prompt_formatter)

    def __iter__(self) -> Iterator[Dict[str, Any] | BaseModel]:
        if self.iterable is not None:
            yield from self.iterable
            return

        if self.working_dir:
            response_file = f"{self.working_dir}/responses.jsonl"

            for line in open(response_file, "r"):
                response = GenericResponse.model_validate_json(line)
                if self.prompt_formatter.response_format:
                    response.response = self.prompt_formatter.response_format(
                        **response.response
                    )
                if self.prompt_formatter.parse_func:
                    response = self.prompt_formatter.parse_func(
                        response.row, response.response
                    )
                else:
                    response = [response.response]

                if isinstance(response, list):
                    yield from response
                else:
                    yield response

    def to_list(self) -> List[Dict[str, Any] | BaseModel]:
        return list(self)

    def to_huggingface(self, in_memory: bool = False) -> None:
        """
        Returns a HuggingFace Dataset

        Args:
            in_memory (bool): Whether to load the dataset into memory

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
                        response = GenericResponse.model_validate_json(line)
                        if self.prompt_formatter.response_format:
                            response.response = self.prompt_formatter.response_format(
                                **response.response
                            )
                            
                        # TODO(Ryan): We can make this more sophisticated by making response_generic a class
                        if response is None:
                            failed_responses_count += 1
                            continue

                        if self.prompt_formatter.parse_func:
                            dataset_rows = self.prompt_formatter.parse_func(
                                response.row, response.response
                            )
                        else:
                            dataset_rows = [response.response]

                        for row in dataset_rows:
                            if isinstance(row, BaseModel):
                                row = row.model_dump()
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

        return HFDataset.from_file(dataset_file, in_memory=in_memory)
