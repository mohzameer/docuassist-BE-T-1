from fastapi import HTTPException, status
from typing import Optional, Any, Dict

class AppException(HTTPException):
    def __init__(
        self,
        status_code: int,
        detail: str,
        error_code: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        super().__init__(status_code=status_code, detail=detail)
        self.error_code = error_code
        self.metadata = metadata or {}

class DocumentProcessingError(AppException):
    def __init__(self, detail: str, metadata: Optional[Dict[str, Any]] = None):
        super().__init__(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=detail,
            error_code="DOCUMENT_PROCESSING_ERROR",
            metadata=metadata
        )

class SearchError(AppException):
    def __init__(self, detail: str, metadata: Optional[Dict[str, Any]] = None):
        super().__init__(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=detail,
            error_code="SEARCH_ERROR",
            metadata=metadata
        ) 