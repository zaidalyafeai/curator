# from abc import ABC, abstractmethod
# from typing import Any, Dict, Optional


# class Verifier(ABC):
#     """Base class for verifying model outputs.

#     Verifiers check if model outputs meet specified criteria or constraints.
#     Subclasses must implement the verify() method with custom verification logic.
#     """

#     @abstractmethod
#     def verify(self, llm_response: str, test_cases: Optional[list[Dict[str, Any]]] = None) -> bool:
#         """Verify if the model output meets the required criteria.

#         Args:
#             llm_response: Model output to verify
#             test_cases: Test cases to verify against

#         Returns:
#             bool: True if verification passes, False otherwise
#         """
#         pass

#     def __call__(self, llm_response: str, test_cases: Optional[list[Dict[str, Any]]] = None) -> bool:
#         return self.verify(llm_response, test_cases)
