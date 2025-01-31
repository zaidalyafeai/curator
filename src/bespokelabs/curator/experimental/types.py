import datetime
from dataclasses import field
from typing import Any, Dict, List, Optional

from pydantic import BaseModel


class CodeExecutionResult(BaseModel):
    """Result of the code execution backend."""

    stdout: str
    stderr: str
    exit_code: int

class CodeExecutionRequestParams(BaseModel):
    """Parameters for the code execution backend."""

    timeout: int = 10
    memory_limit: int = 1024 * 1024 * 1024


class CodeExecutionRequest(BaseModel):
    """Request to the code execution backend."""

    code: str
    code_input: str
    code_output: str
    execution_params: Optional[CodeExecutionRequestParams] = None
    original_row: Optional[Dict[str, Any]] = None
    original_row_idx: Optional[int] = None

class CodeAPIRequest(BaseModel):
    """Request to the code execution backend."""

    task_id: Optional[int] = None
    execution_request: CodeExecutionRequest
    attempts_left: int
    code_formatter: Any
    created_at: datetime.datetime = field(default_factory=datetime.datetime.now)
    result: list = field(default_factory=list)


class CodeExecutionResponse(BaseModel):
    """Response from the code execution backend."""

    code_api_request: Optional[CodeAPIRequest] = None
    response_message: Optional[Dict[str, Any]] | str = None
    response_error: Optional[str] = None
    response_stdout: Optional[str] = None
    response_stderr: Optional[str] = None
    created_at: datetime.datetime = field(default_factory=datetime.datetime.now)
    finished_at: datetime.datetime = field(default_factory=datetime.datetime.now)


class CodeExecutionBackendConfig(BaseModel):
    """Configuration for the code execution backend."""

    max_requests_per_minute: int = 10000
    max_retries: int = 3
    seconds_to_pause_on_rate_limit: int = 10
