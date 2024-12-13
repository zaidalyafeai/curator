"""Batch processing functionality for LLM."""

from dataclasses import dataclass
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .llm import LLM


@dataclass
class BatchConfig:
    """Configuration for batch processing in LLM.

    This class holds all configuration parameters related to batch processing,
    used by the LLM context manager for batch operations.

    Args:
        batch_size: Maximum number of requests per batch. If None, defaults to 1,000.
        batch_check_interval: How often to check batch status, in seconds.
        delete_successful_batch_files: Whether to delete batch files after successful processing.
        delete_failed_batch_files: Whether to delete batch files after failed processing.
    """

    batch_size: Optional[int] = None
    batch_check_interval: int = 60
    delete_successful_batch_files: bool = True
    delete_failed_batch_files: bool = False


class BatchContext:
    """Context manager for batch processing.

    This class provides a context manager interface for batch processing with LLM instances.
    It handles the setup and teardown of batch processing configuration, ensuring proper
    state management of the LLM instance.

    Example:
        ```python
        from bespokelabs.curator.llm import LLM, batch

        llm = LLM(...)
        with batch(llm, batch_size=100):
            results = llm(dataset)
        ```
    """

    def __init__(self, llm: "LLM", **kwargs):
        """Initialize batch context.

        Args:
            llm: The LLM instance to use for batch processing.
            **kwargs: Batch configuration parameters passed to BatchConfig.
        """
        self.llm = llm
        self.config = BatchConfig(**kwargs)
        self._original_processor = None

    def __enter__(self):
        """Enter batch context.

        Returns:
            The LLM instance configured for batch processing.

        Raises:
            RuntimeError: If already in a batch context.
        """
        if hasattr(self.llm, "_batch_config") and self.llm._batch_config is not None:
            raise RuntimeError("Already in batch context")
        self.llm._batch_config = self.config
        self.llm._setup_request_processor()
        return self.llm

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit batch context and restore original request processor."""
        self.llm._batch_config = None
        if self.llm._original_request_processor is not None:
            self.llm._request_processor = self.llm._original_request_processor


def batch(llm: "LLM", **kwargs) -> BatchContext:
    """Create a batch processing context for an LLM instance.

    This function creates a context manager that configures an LLM instance for
    batch processing. The batch processing configuration is active only within
    the context manager's scope.

    Args:
        llm: The LLM instance to configure for batch processing.
        **kwargs: Configuration parameters for batch processing.
            See BatchConfig for available parameters.

    Returns:
        A BatchContext instance that can be used as a context manager.

    Example:
        ```python
        from bespokelabs.curator.llm import LLM, batch

        llm = LLM(...)
        with batch(llm, batch_size=100):
            results = llm(dataset)
        ```
    """
    return BatchContext(llm, **kwargs)
