from pydantic import BaseModel
from typing import List, Optional, Dict
from datetime import datetime
from enum import Enum

class BatchStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"

class BatchProcessingStats(BaseModel):
    total_documents: int
    processed_documents: int = 0
    failed_documents: int = 0
    start_time: datetime = datetime.utcnow()
    end_time: Optional[datetime] = None
    
class BatchProcessingError(BaseModel):
    document_id: str
    error_message: str
    timestamp: datetime = datetime.utcnow()

class BatchProcessingStatus(BaseModel):
    batch_id: str
    status: str
    stats: BatchProcessingStats
    errors: List[BatchProcessingError] = []
    metadata: Dict[str, str] = {} 