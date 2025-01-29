"""Ray-based Autoscaling Code Execution Backend."""

import asyncio
import logging
import os
import subprocess
import tempfile
import time
from typing import Dict

import ray

from bespokelabs.curator.experimental.code_execution_backend.base_backend import BaseCodeExecutionBackend
from bespokelabs.curator.experimental.types import CodeAPIRequest, CodeExecutionResponse, CodeTestCaseResponse


@ray.remote
class CodeExecutorActor:
    """Ray actor for executing code in isolation."""

    def __init__(self):
        """Initialize the actor."""
        self.temp_files = []
        self.busy = False
        self.last_active = time.time()

    def __del__(self):
        """Cleanup temporary files on actor shutdown."""
        for temp_file in self.temp_files:
            try:
                if os.path.exists(temp_file):
                    os.unlink(temp_file)
            except Exception:
                pass

    async def execute_test_case(self, code: str, test_case, test_case_idx: int, timeout: int) -> CodeTestCaseResponse:
        """Execute a single test case in isolation."""
        self.busy = True
        self.last_active = time.time()
        temp_program_path = None

        try:
            # Create temporary file
            with tempfile.NamedTemporaryFile(delete=False, mode="w", encoding="utf-8") as temp_file:
                temp_file.write(code)
                temp_program_path = temp_file.name
                self.temp_files.append(temp_program_path)

            input_data = test_case.input
            if isinstance(input_data, list):
                input_data = "\n".join(input_data)

            try:
                result = subprocess.run(["python", temp_program_path], input=input_data, text=True, capture_output=True, timeout=timeout)
                return CodeTestCaseResponse(
                    test_case_idx=test_case_idx,
                    response_message="success",
                    response_errors=None,
                    response_stdout=result.stdout,
                    response_stderr=result.stderr,
                )

            except subprocess.TimeoutExpired:
                return CodeTestCaseResponse(
                    test_case_idx=test_case_idx,
                    response_message="timeout",
                    response_errors=[f"Execution timed out after {timeout}s"],
                    response_stdout=None,
                    response_stderr=None,
                )

            except Exception as e:
                return CodeTestCaseResponse(
                    test_case_idx=test_case_idx,
                    response_message="error",
                    response_errors=[str(e)],
                    response_stdout=None,
                    response_stderr=None,
                )

        finally:
            if temp_program_path and temp_program_path in self.temp_files:
                try:
                    os.unlink(temp_program_path)
                    self.temp_files.remove(temp_program_path)
                except Exception:
                    pass
            self.busy = False

    def is_busy(self) -> bool:
        """Check if the actor is currently processing a task."""
        return self.busy

    def get_last_active(self) -> float:
        """Get the timestamp of last activity."""
        return self.last_active


class RayAutoscalingBackend(BaseCodeExecutionBackend):
    """Ray-based autoscaling code execution backend."""

    def __init__(self, config):
        """Initialize the backend."""
        super().__init__(config)
        self.config = config
        self.logger = logging.getLogger(__name__)

        # Initialize Ray if needed
        if not ray.is_initialized():
            ray.init()

        self.executors: Dict[str, ray.actor.ActorHandle] = {}
        self.last_scaling_time = 0

        # Create initial executor pool
        self._scale_to(self.config.initial_executors)

    def _create_executor(self) -> ray.actor.ActorHandle:
        """Create a new executor actor."""
        executor = CodeExecutorActor.remote()
        self.executors[ray.get_runtime_context().get_actor_id()] = executor
        return executor

    async def _get_available_executor(self) -> ray.actor.ActorHandle:
        """Get an available executor, waiting if necessary."""
        while True:
            for executor in self.executors.values():
                if not await ray.get(executor.is_busy.remote()):
                    return executor
            await asyncio.sleep(0.1)

    async def _check_scaling(self):
        """Check and adjust scaling if needed."""
        current_time = time.time()
        if current_time - self.last_scaling_time < self.config.scaling_cooldown:
            return

        # Get busy count
        busy_states = await asyncio.gather(*[ray.get(executor.is_busy.remote()) for executor in self.executors.values()])
        busy_count = sum(1 for state in busy_states if state)

        load_factor = busy_count / len(self.executors)

        if load_factor >= self.config.scale_up_threshold:
            # Scale up
            target = min(int(len(self.executors) * self.config.scale_up_factor), self.config.max_executors)
            if target > len(self.executors):
                self.logger.info(f"Scaling up to {target} executors")
                self._scale_to(target)
                self.last_scaling_time = current_time

        elif load_factor <= self.config.scale_down_threshold:
            # Scale down
            target = max(int(len(self.executors) * self.config.scale_down_factor), self.config.min_executors)
            if target < len(self.executors):
                self.logger.info(f"Scaling down to {target} executors")
                self._scale_to(target)
                self.last_scaling_time = current_time

    def _scale_to(self, target_count: int):
        """Scale the executor pool to the target size."""
        current_count = len(self.executors)

        if target_count > current_count:
            # Scale up
            for _ in range(target_count - current_count):
                self._create_executor()
        elif target_count < current_count:
            # Scale down - remove least recently used executors
            to_remove = current_count - target_count
            if to_remove > 0:
                executor_times = [(eid, ray.get(executor.get_last_active.remote())) for eid, executor in self.executors.items()]
                executor_times.sort(key=lambda x: x[1])

                for eid, _ in executor_times[:to_remove]:
                    del self.executors[eid]

    async def execute_request(self, request: CodeAPIRequest) -> CodeExecutionResponse:
        """Execute a single request using Ray actors."""
        await self._check_scaling()

        code = request.generic_request.code
        test_cases = request.generic_request.test_cases
        timeout = request.generic_request.execution_params.timeout

        # Distribute test cases across executors
        futures = []
        for idx, test_case in enumerate(test_cases):
            executor = await self._get_available_executor()
            future = executor.execute_test_case.remote(code, test_case, idx, timeout)
            futures.append(future)

        # Wait for all results
        results = await asyncio.get_event_loop().run_in_executor(None, ray.get, futures)

        return CodeExecutionResponse(
            responses=results,
            code_api_request=request,
        )

    def shutdown(self):
        """Cleanup resources when shutting down the backend."""
        if ray.is_initialized():
            ray.shutdown()
