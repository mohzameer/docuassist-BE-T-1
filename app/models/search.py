from typing import Dict, Any, Optional, List
from pydantic import BaseModel

class SearchRequest(BaseModel):
    """Search request model"""
    query: str
    filters: Optional[Dict[str, Any]] = None
    limit: Optional[int] = 10
    summarize: Optional[bool] = False

class DocumentResponse(BaseModel):
    """Individual document response"""
    content: str
    metadata: Dict[str, Any]
    score: float
    vector_score: float
    reranker_score: float
    document_id: str

class SearchResponse(BaseModel):
    """Search response model"""
    answer: Optional[str]
    documents: List[DocumentResponse]
    total_results: int 