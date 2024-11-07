from typing import Dict, List, Any

from pydantic import BaseModel


class GenericResponse(BaseModel):
    response: Dict[str, Any] | BaseModel
    errors: List[str]
    row: Dict[str, Any]
    row_idx: int
