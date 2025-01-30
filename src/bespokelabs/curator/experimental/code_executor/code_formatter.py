from dataclasses import dataclass
from typing import Callable

from bespokelabs.curator.experimental.types import CodeExecutionRequest, CodeExecutionRequestParams, CodeExecutionResponse


@dataclass
class CodeFormatter:
    """Formatter for the code execution backend."""

    code_string: Callable
    test_cases: Callable
    parse_results: Callable
    execution_params: CodeExecutionRequestParams

    def create_code_execution_request(self, row: dict, idx: int) -> CodeExecutionRequest:
        """Format the request object based off of `LLM` attributes.

        Args:
            row: Input data to format into a prompt
            idx: Index of the row in the dataset

        Returns:
            CodeExecutionRequest object containing the formatted request

        Raises:
            ValueError: If prompt_func has invalid number of arguments or returns invalid format
        """
        return CodeExecutionRequest(
            code=self.code_string(row),
            test_cases=self.test_cases(row),
            execution_params=self.execution_params,
            original_row=row,
            original_row_idx=idx,
        )

    def response_to_response_format(self, response: CodeExecutionResponse):
        """Convert responses to proper dictionary format."""
        return {
            "responses": [
                {
                    "test_case_idx": r.test_case_idx,
                    "response_message": r.response_message,
                    "response_errors": r.response_errors,
                    "response_stdout": r.response_stdout,
                    "response_stderr": r.response_stderr,
                }
                for r in response.responses
            ],
            "code_api_request": response.code_api_request.model_dump(),
        }
