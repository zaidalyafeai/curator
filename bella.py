"""Bella: Bespoke Labs Synthetic Data Generation Library."""

import asyncio
from typing import Any, Dict, Generic, List, Optional, Type, TypeVar

from prompt import PromptCaller

import instructor
import pandas as pd
from datasets import Dataset as HFDataset
from IPython.display import HTML, display
from jinja2 import Template
from litellm import acompletion as litellm_acompletion
from pydantic import BaseModel, ConfigDict, Field
from tqdm import tqdm
from tqdm.asyncio import tqdm as tqdm_async

T = TypeVar("T", bound=BaseModel)


class ListModel(BaseModel, Generic[T]):
    """A list of items to be used as a response format."""
    model_config = ConfigDict(title="ListResponse")  # This sets a valid schema name.
    items: List[T] = Field(description="List of items")


class Dataset(HFDataset):
    """A wrapper around a Hugging Face Dataset with extra functionality for data generation."""
    initialized: bool = True
    _list_columns: List[str] = []

    @classmethod
    def empty(cls) -> "Dataset":
        dataset = cls.from_list([])
        dataset.initialized = False
        dataset._list_columns = []
        return dataset

    def display(self):
        display(HTML(self.to_pandas().to_html()))

    def flatten(self) -> "Dataset":
        """Flatten any list columns in the dataset"""
        if not self._list_columns:
            return self

        flattened_rows = []
        for row in self:
            row_dict = dict(row)
            list_values = {
                col: row_dict[col] for col in self._list_columns if col in row_dict
            }

            if not any(isinstance(v, list) for v in list_values.values()):
                flattened_rows.append(row_dict)
                continue

            max_length = max(
                len(v) for v in list_values.values() if isinstance(v, list)
            )

            for i in range(max_length):
                new_row = {}
                for col, value in row_dict.items():
                    if col in list_values and isinstance(value, list):
                        new_row[col] = value[i] if i < len(value) else None
                    else:
                        new_row[col] = value
                flattened_rows.append(new_row)

        dataset = Dataset.from_pandas(pd.DataFrame(flattened_rows))
        dataset.initialized = True
        dataset._list_columns = []
        return dataset

    def flatten_objects(self) -> "Dataset":
        return super(Dataset, self).flatten()

    async def completions(
        self,
        prompt_caller: PromptCaller,
        output_column: str,
        keep_columns: bool = True,
        verbose: bool = True,
        name: Optional[str] = None,
    ) -> "Dataset":
        """
        Apply structured completions to the dataset using specified model and prompts.

        Args:
            model_name: Name of the model to use
            system_prompt: System prompt template
            user_prompt: User prompt template
            response_format: Pydantic model defining the response structure
            output_column: Name of the column to store the response
            keep_columns: Whether to keep original columns in the output dataset
            verbose: Whether to show a progress bar
            name: Name of the task

        Returns:
            A new Dataset with the completions added
        """
        if not self.initialized:
            self = self.from_dict({"dummy": ["dummy row"]})
            keep_columns = False
            self.initialized = True

        call_api = prompt_caller.get_api_call_fn()

        rows = [dict(row) for row in self]

        # Gather all API calls with progress bar if verbose.
        if verbose:
            responses = await tqdm_async.gather(
                *[call_api(row) for row in rows],
                desc=(
                    f"Making API calls for {name}"
                    if name is not None
                    else "Making API calls"
                ),
                total=len(rows),
                miniters=1,
            )
        else:
            responses = await asyncio.gather(*[call_api(row) for row in rows])

        # Process responses into a flat dictionary structure.
        processed_rows = []

        # Use regular tqdm for synchronous processing.
        response_iterator = (
            tqdm(
                zip(rows, responses),
                desc=(
                    f"Processing responses for {name}"
                    if name is not None
                    else "Processing responses"
                ),
                total=len(rows),
            )
            if verbose
            else zip(rows, responses)
        )

        for original_row, response in response_iterator:
            new_row = {}
            if keep_columns:
                new_row.update(original_row)

            if isinstance(response, ListModel):
                new_row[output_column] = [
                    item.dict() if isinstance(item, BaseModel) else item
                    for item in response.items
                ]
                self._list_columns.append(output_column)
            else:
                new_row[output_column] = (
                    response.dict() if isinstance(response, BaseModel) else response
                )

            processed_rows.append(new_row)

        dataset = Dataset.from_pandas(pd.DataFrame(processed_rows))
        dataset.initialized = True
        dataset._list_columns = self._list_columns
        return dataset
