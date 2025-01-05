import logging
from typing import Optional
import vllm
import datetime
from bespokelabs.curator.request_processor.base_online_request_processor import APIRequest
from bespokelabs.curator.request_processor.base_offline_request_processor import (
    BaseOfflineRequestProcessor,
    StatusTracker,
)
from bespokelabs.curator.request_processor.generic_request import GenericRequest
from bespokelabs.curator.request_processor.generic_response import GenericResponse
from vllm.distributed import destroy_model_parallel, destroy_distributed_environment
import contextlib
import torch
import gc
from vllm.sampling_params import GuidedDecodingParams
from pydantic import BaseModel
from typing import List

logger = logging.getLogger(__name__)


class VLLMOfflineRequestProcessor(BaseOfflineRequestProcessor):
    """
    Offline request processor for the VLLM model.

    Args:
        model (str): The model (path) to use
        temperature (float, optional): The temperature for sampling. Defaults to None.
        top_p (float, optional): The top_p for sampling. Defaults to None.
        presence_penalty (float, optional): The presence penalty for sampling. Defaults to None.
        frequency_penalty (float, optional): The frequency penalty for sampling. Defaults to None.
        max_model_length (int, optional): The maximum model length. Defaults to 4096.
        max_tokens (int, optional): The maximum number of tokens. Defaults to 1024.
        enforce_eager (bool, optional): Whether to enforce eager execution. Defaults to False.
        tensor_parallel_size (int, optional): The tensor parallel size. Defaults to 1.
        max_num_seqs (int, optional): The maximum number of sequences (aka batch size). Defaults to 256.
        gpu_memory_utilization (float, optional): The GPU memory utilization. Defaults to 0.95.
        response_format (Optional[BaseModel], optional): The response format. Defaults to None.
    """

    def __init__(
        self,
        model: str,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        presence_penalty: Optional[float] = None,
        frequency_penalty: Optional[float] = None,
        max_model_length: int = 4096,
        max_tokens: Optional[int] = 1024,
        min_tokens: Optional[int] = 0,
        enforce_eager: bool = False,
        tensor_parallel_size: int = 1,
        max_num_seqs: int = 256,
        gpu_memory_utilization: float = 0.95,
        response_format: Optional[BaseModel] = None,
    ):
        self.max_model_length = max_model_length
        self.max_tokens = max_tokens
        self.min_tokens = min_tokens
        self.enforce_eager = enforce_eager
        self.tensor_parallel_size = tensor_parallel_size
        self.temperature = temperature if temperature is not None else 1.0
        self.top_p = top_p if top_p is not None else 1.0
        self.presence_penalty = presence_penalty if presence_penalty is not None else 0.0
        self.frequency_penalty = frequency_penalty if frequency_penalty is not None else 0.0
        self.max_num_seqs = max_num_seqs
        self.response_format = response_format
        self.gpu_memory_utilization = gpu_memory_utilization
        self.support_structured_output = False
        super().__init__(
            model=model,
        )

    def load_offline_model(self):
        """Load the VLLM model for offline processing."""
        self.model_class = vllm.LLM(
            self.model,
            trust_remote_code=True,
            max_model_len=self.max_model_length,
            enforce_eager=self.enforce_eager,
            tensor_parallel_size=self.tensor_parallel_size,
            max_num_seqs=self.max_num_seqs,
            gpu_memory_utilization=self.gpu_memory_utilization,
            disable_custom_all_reduce=self.tensor_parallel_size > 1,
        )

    def format_prompts(self, prompts: list) -> list:
        """Format prompts for the VLLM model.

        Args:
            prompts (list): List of prompts to format
        """

        tokenizer = self.model_class.get_tokenizer()
        try:
            formatted_prompts = [
                tokenizer.apply_chat_template(
                    conversation=prompt, tokenize=False, add_generation_prompt=True
                )
                for prompt in prompts
            ]
        except Exception as e:
            logger.error(f"Error formatting prompts: {e}")
            raise e

        return formatted_prompts

    def check_structured_output_support(self) -> bool:
        """Verify if the model supports structured output via instructor.

        Tests the model's capability to handle structured output by making a test request
        with a simple schema.

        Returns:
            bool: True if structured output is supported, False otherwise

        Note:
            - Uses a simple User schema as test case
            - Logs detailed information about support status
            - Required for models that will use JSON schema responses
        """

        class User(BaseModel):
            name: str
            age: int

        try:
            json_schema = User.schema()
            guided_decoding_params = GuidedDecodingParams(json=json_schema)
            sampling_params = vllm.SamplingParams(guided_decoding=guided_decoding_params)
            messages = [[{"role": "user", "content": "Jason is 25 years old."}]]
            formatted_prompts = self.format_prompts(messages)
            response = self.model_class.generate(
                formatted_prompts,
                sampling_params=sampling_params,
            )
            response = response[0].outputs[0].text
            response = self.fix_json(response)
            logger.info(f"Check guided decoding structure output response: {response}")
            response = User.parse_raw(response)
            assert isinstance(response.age, int)
            assert isinstance(response.name, str)
            assert response.name == "Jason"
            assert response.age == 25
            logger.info(
                f"Model {self.model} supports structured output via instructor, response: {response}"
            )
            self.support_structured_output = True
            return True
        except Exception as e:
            logger.info(response)

            logger.warning(
                f"Model {self.model} does not support structured output via guided decoding: {e} {type(e)} {e}"
            )
            return False

    def create_api_specific_request(self, generic_request: GenericRequest) -> dict:
        """Convert a generic request into a vLLM-compatible format.

        Checks supported parameters for the specific model and only includes
        applicable parameters.

        Args:
            generic_request (GenericRequest): The generic request to convert

        Returns:
            dict: LiteLLM-compatible request parameters

        Note:
            Uses LiteLLM's get_supported_openai_params to check parameter support
        """

        request = {
            "model": generic_request.model,
            "messages": generic_request.messages,
        }

        # Only add parameters that are supported by this model
        if self.temperature is not None:
            request["temperature"] = self.temperature

        if self.top_p is not None:
            request["top_p"] = self.top_p

        if self.presence_penalty is not None:
            request["presence_penalty"] = self.presence_penalty

        if self.frequency_penalty is not None:
            request["frequency_penalty"] = self.frequency_penalty

        if self.tensor_parallel_size is not None:
            request["tensor_parallel_size"] = self.tensor_parallel_size

        if self.max_model_length is not None:
            request["max_model_length"] = self.max_model_length

        return request

    def destroy(self):
        """Destroy the VLLM model and cleanup the environment."""
        destroy_model_parallel()
        destroy_distributed_environment()
        del self.model_class.llm_engine.model_executor
        del self.model_class
        with contextlib.suppress(AssertionError):
            torch.distributed.destroy_process_group()
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    def fix_json(self, response_message: str) -> str:
        """
        Fix vLLM issue (https://github.com/vllm-project/vllm/issues/8350)
        During guided decoding, the JSON structure may not be closed properly.

        Args:
            response_message (str): The response message to fix

        Returns:
            str: The fixed response message
        """
        if not response_message.endswith("}"):
            response_message += "}"
        return response_message

    def process_requests(
        self, requests: list[APIRequest], status_tracker: StatusTracker
    ) -> list[GenericResponse]:
        """Process a list of generic requests using the VLLM model.

        Args:
            generic_requests (list[GenericRequest]): List of generic requests to process
            status_tracker (StatusTracker): Status tracker for the request

        Returns:
            list[GenericResponse]: List of generic responses
        """
        guided_decoding_params = None
        if self.response_format is not None and self.support_structured_output:
            guided_decoding_params = GuidedDecodingParams(json=self.response_format.schema())
        else:
            if self.response_format is not None:
                logger.warning(
                    f"Model {self.model} does not support structured output via guided decoding, "
                    f"response_format: {self.response_format}"
                )

        sampling_params = {
            "temperature": self.temperature,
            "top_p": self.top_p,
            "presence_penalty": self.presence_penalty,
            "frequency_penalty": self.frequency_penalty,
            "guided_decoding": guided_decoding_params,
            "max_tokens": self.max_tokens,
            "min_tokens": self.min_tokens,
        }

        formatted_prompts = self.format_prompts(
            [request.generic_request.messages for request in requests]
        )

        completions = self.model_class.generate(
            formatted_prompts,
            sampling_params=vllm.SamplingParams(**sampling_params),
        )
        status_tracker.time_finished = datetime.datetime.now()

        responses = []

        for completion, request in zip(completions, requests):
            response_message = completion.outputs[0].text
            response_message = self.fix_json(response_message)

            raw_response = {
                "request_id": completion.request_id,
                "finished": completion.finished,
                "encoder_prompt": completion.encoder_prompt,
                "prompt": completion.prompt,
                "metrics": completion.metrics,
            }
            response = GenericResponse(
                model=self.model,
                response_message=response_message,
                raw_response=raw_response,
                created_at=status_tracker.time_started,
                finished_at=status_tracker.time_finished,
                generic_request=request.generic_request,
            )
            responses.append(response)
        return responses
