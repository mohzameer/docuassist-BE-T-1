from app.models.document import DocumentResponse
from app.models.batch import BatchProcessingStatus

examples = {
    "search": {
        "summary": "Search documents",
        "description": "Perform semantic search across processed documents",
        "request_example": {
            "query": "machine learning applications",
            "filters": {"mime_type": "application/pdf"},
            "limit": 10
        },
        "response_example": [{
            "content": "Example content...",
            "metadata": {"title": "ML Paper"},
            "score": 0.95
        }]
    },
    "batch_processing": {
        "summary": "Process documents in batch",
        "description": "Start batch processing of documents from Google Drive",
        "response_example": {
            "batch_id": "123",
            "status": "processing",
            "stats": {
                "total_documents": 100,
                "processed_documents": 45,
                "failed_documents": 2
            }
        }
    }
} 