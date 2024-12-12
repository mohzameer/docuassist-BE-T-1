from pydantic import BaseModel
from typing import Optional, Any

class ErrorResponse(BaseModel):
    detail: str
    error_code: Optional[str] = None
    metadata: Optional[dict[str, Any]] = None 