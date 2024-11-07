from typing import Any, Dict, List, Optional

from pydantic import BaseModel


class GenericResponse(BaseModel):
    response: Optional[Dict[str, Any]] | str = None
    errors: Optional[List[str]] = None
    row: Dict[str, Any]
    row_idx: int
    raw_response: Optional[Dict[str, Any]] = None
