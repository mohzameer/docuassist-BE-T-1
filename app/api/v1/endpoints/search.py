from fastapi import APIRouter, HTTPException, Query, Depends
from typing import List, Optional, Dict
from app.services.search_service import SearchService
from app.models.search import SearchRequest, SearchResponse
import logging

logger = logging.getLogger(__name__)
router = APIRouter()

@router.post("/search", response_model=List[SearchResponse])
async def search(
    request: SearchRequest,
    search_service: SearchService = Depends(),
):
    """
    Perform hybrid search across documents
    """
    try:
        results = await search_service.hybrid_search(
            query=request.query,
            filters=request.filters,
            limit=request.limit
        )
        return results
    except Exception as e:
        logger.error(f"Search endpoint error: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="An error occurred while performing the search"
        ) 