from typing import Dict, Any, Optional, List
from pydantic import BaseModel

class SearchRequest(BaseModel):
    """Search request model"""
    query: str
    filters: Optional[Dict[str, Any]] = None
    limit: Optional[int] = 10
    summarize: Optional[bool] = False

class SearchResponse(BaseModel):
    """Search response model"""
    content: str
    metadata: Dict[str, Any]
    score: Optional[float] = None
    document_id: str 