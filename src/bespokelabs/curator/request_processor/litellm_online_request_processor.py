import json
import logging
import os
from typing import Optional

import instructor
from litellm import completion, get_supported_openai_params, model_cost

from bespokelabs.curator.dataset import Dataset
from bespokelabs.curator.prompter.prompter import PromptFormatter
from bespokelabs.curator.request_processor.base_request_processor import (
    BaseRequestProcessor,
    GenericRequest,
    GenericResponse,
)

logger = logging.getLogger(__name__)

class LiteLLMOnlineRequestProcessor(BaseRequestProcessor):
    """
    Request processor for LiteLLM that supports structured outputs via instructor.
    """

    def __init__(
        self,
        model: str,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        presence_penalty: Optional[float] = None,
        frequency_penalty: Optional[float] = None,
    ):
        """Initialize LiteLLM request processor.
        
        Args:
            model (str): Name of the model to use
            temperature (Optional[float]): Sampling temperature
            top_p (Optional[float]): Nucleus sampling parameter
            presence_penalty (Optional[float]): Presence penalty parameter  
            frequency_penalty (Optional[float]): Frequency penalty parameter
        """
        super().__init__(batch_size=None)  # LiteLLM doesn't support batching yet
        self.model = model
        self.temperature = temperature
        self.top_p = top_p
        self.presence_penalty = presence_penalty
        self.frequency_penalty = frequency_penalty
        
        # Initialize instructor client
        self.client = instructor.from_litellm(completion)

    def get_rate_limits(self) -> dict:
        """Get rate limits for the model from LiteLLM."""
        # Get model costs/limits from LiteLLM
        costs = model_cost.get(self.model, {})
        
        # Default conservative limits if not found
        return {
            "max_tokens_per_minute": costs.get("max_tokens", 150_000),
            "max_requests_per_minute": 3000
        }

    def create_api_specific_request(self, generic_request: GenericRequest) -> dict:
        """Create a LiteLLM-specific request from a generic request.
        
        Args:
            generic_request (GenericRequest): The generic request to convert
            
        Returns:
            dict: LiteLLM-specific request parameters
        """
        # Get supported parameters for this model
        supported_params = get_supported_openai_params(model=self.model)
        
        request = {
            "model": generic_request.model,
            "messages": generic_request.messages,
        }

        # Only add parameters that are supported by this model
        if "temperature" in supported_params and self.temperature is not None:
            request["temperature"] = self.temperature
            
        if "top_p" in supported_params and self.top_p is not None:
            request["top_p"] = self.top_p
            
        if "presence_penalty" in supported_params and self.presence_penalty is not None:
            request["presence_penalty"] = self.presence_penalty
            
        if "frequency_penalty" in supported_params and self.frequency_penalty is not None:
            request["frequency_penalty"] = self.frequency_penalty

        return request

    def run(
        self,
        dataset: Optional[Dataset],
        working_dir: str,
        parse_func_hash: str,
        prompt_formatter: PromptFormatter,
    ) -> Dataset:
        """Run completions using LiteLLM.
        
        Args:
            dataset (Optional[Dataset]): Input dataset
            working_dir (str): Directory for saving request/response files
            parse_func_hash (str): Hash of the parse function
            prompt_formatter (PromptFormatter): Prompt formatting logic
            
        Returns:
            Dataset: Dataset with completion results
        """
        # Create request files
        generic_requests_files = self.create_request_files(
            dataset, working_dir, prompt_formatter
        )
        
        # Create response files
        generic_responses_files = [
            f"{working_dir}/responses_{i}.jsonl"
            for i in range(len(generic_requests_files))
        ]

        # Process each request file
        for request_file, response_file in zip(
            generic_requests_files, generic_responses_files
        ):
            self._process_requests(
                request_file,
                response_file,
                prompt_formatter.response_format
            )

        # Create final dataset
        return self.create_dataset_files(
            working_dir, parse_func_hash, prompt_formatter
        )

    def _process_requests(
        self, 
        request_file: str,
        response_file: str,
        response_format: Optional[type] = None
    ):
        """Process requests from file using LiteLLM."""
        with open(request_file, "r") as f:
            for line in f:
                request = GenericRequest.model_validate_json(line)
                
                try:
                    # Make request with structured output if response_format provided
                    if response_format:
                        response = self.client.chat.completions.create(
                            **self.create_api_specific_request(request),
                            response_model=response_format
                        )
                        # Convert Pydantic model to dict for storage
                        response_message = response.model_dump()
                    else:
                        response = self.client.chat.completions.create(
                            **self.create_api_specific_request(request)
                        )
                        response_message = response.content if hasattr(response, 'content') else str(response)
                        
                    # Create generic response
                    generic_response = GenericResponse(
                        response_message=response_message,  # Now using the properly formatted response
                        response_errors=None,
                        raw_request=self.create_api_specific_request(request),
                        raw_response=None,  # LiteLLM response object isn't JSON serializable
                        generic_request=request,
                    )
                        
                except Exception as e:
                    logger.error(f"Error processing request: {e}")
                    # Create error response
                    generic_response = GenericResponse(
                        response_message=None,
                        response_errors=[str(e)],
                        raw_request=self.create_api_specific_request(request),
                        raw_response=None,
                        generic_request=request,
                    )

                # Save response to file
                with open(response_file, "a") as f_out:
                    f_out.write(json.dumps(generic_response.model_dump(), default=str) + "\n")
