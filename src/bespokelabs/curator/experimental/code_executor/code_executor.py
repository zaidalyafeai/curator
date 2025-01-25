import logging
import os
from datetime import datetime
from typing import TYPE_CHECKING, Any, Dict, Iterable, Optional, TypeVar

from pydantic import BaseModel
from xxhash import xxh64

from bespokelabs.curator.experimental.code_executor.code_formatter import CodeFormatter

if TYPE_CHECKING:
    from datasets import Dataset

_CURATOR_DEFAULT_CACHE_DIR = "~/.cache/curator"
T = TypeVar("T")
_DictOrBaseModel = Dict[str, Any] | BaseModel
logger = logging.getLogger(__name__)


from bespokelabs.curator.experimental.code_execution_backend._factory import _CodeExecutionBackendFactory
from bespokelabs.curator.experimental.db import CodeMetadataDB
from bespokelabs.curator.experimental.types import CodeExecutionRequestParams, CodeExecutionResult, TestCase
from bespokelabs.curator.llm.llm import _convert_to_dataset, _get_function_hash, _get_function_source


class CodeExecutor:
    def function_name(self, row: dict):
        pass

    def code_string(self, row: dict):
        pass

    def test_cases(self, row: dict) -> list[TestCase]:
        pass

    def parse_results(self, row: dict, test_cases: list[TestCase], execution_results: list[CodeExecutionResult]):
        pass

    def __init__(
        self,
        backend: str = "multiprocessing",
        backend_params: dict = {
            "max_requests_per_minute": 10000,
        },
    ):
        self._code_executor = _CodeExecutionBackendFactory.create(backend=backend, backend_params=backend_params)

    def _hash_fingerprint(self, dataset_hash, disable_cache):
        if disable_cache:
            fingerprint = xxh64(os.urandom(8)).hexdigest()
        else:
            # get the source code of the function_name, code_string, test_cases, and parse methods
            function_name_hash = _get_function_hash(self.function_name)
            code_string_hash = _get_function_hash(self.code_string)
            test_cases_hash = _get_function_hash(self.test_cases)
            parse_results_hash = _get_function_hash(self.parse_results)

            fingerprint_str = "_".join(
                [
                    str(dataset_hash),
                    str(function_name_hash),
                    str(code_string_hash),
                    str(test_cases_hash),
                    str(parse_results_hash),
                ]
            )

            fingerprint = xxh64(fingerprint_str.encode("utf-8")).hexdigest()
            logger.debug(f"Curator Cache Fingerprint String: {fingerprint_str}")
            logger.debug(f"Curator Cache Fingerprint: {fingerprint}")

        return fingerprint

    def __call__(
        self,
        dataset: Optional[Iterable] = None,
        working_dir: str = None,
        execution_params: Optional[CodeExecutionRequestParams] = CodeExecutionRequestParams(),
    ) -> "Dataset":
        dataset = _convert_to_dataset(dataset)

        if working_dir is None:
            curator_cache_dir = os.environ.get(
                "CURATOR_CACHE_DIR",
                os.path.expanduser(_CURATOR_DEFAULT_CACHE_DIR),
            )
        else:
            curator_cache_dir = working_dir

        dataset_hash = dataset._fingerprint if dataset is not None else xxh64("").hexdigest()

        disable_cache = os.getenv("CURATOR_DISABLE_CACHE", "").lower() in ["true", "1"]
        fingerprint = self._hash_fingerprint(dataset_hash, disable_cache)

        metadata_db_path = os.path.join(curator_cache_dir, "metadata.db")
        metadata_db = CodeMetadataDB(metadata_db_path)

        metadata_dict = {
            "timestamp": datetime.now().isoformat(),
            "dataset_hash": dataset_hash,
            "function_name": _get_function_source(self.function_name),
            "code_string": _get_function_source(self.code_string),
            "test_cases": _get_function_source(self.test_cases),
            "parse_results": _get_function_source(self.parse_results),
            "run_hash": fingerprint,
        }

        metadata_db.store_metadata(metadata_dict)

        run_cache_dir = os.path.join(curator_cache_dir, fingerprint)

        all_func_hash = [
            _get_function_hash(self.function_name),
            _get_function_hash(self.code_string),
            _get_function_hash(self.test_cases),
            _get_function_hash(self.parse_results),
        ]

        # get a hash of all the function hashes
        all_func_hash_str = "_".join(all_func_hash)
        all_func_hash_hash = xxh64(all_func_hash_str.encode("utf-8")).hexdigest()

        self.code_formatter = CodeFormatter(
            function_name=self.function_name,
            code_string=self.code_string,
            test_cases=self.test_cases,
            parse_results=self.parse_results,
            execution_params=execution_params,
        )

        dataset = self._code_executor.run(
            dataset=dataset,
            working_dir=run_cache_dir,
            code_formatter=self.code_formatter,
            all_func_hash_hash=all_func_hash_hash,
        )

        return dataset
