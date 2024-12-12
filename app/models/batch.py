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
    processed_documents: int
    failed_documents: int
    start_time: datetime
    end_time: Optional[datetime] = None
    
class BatchProcessingError(BaseModel):
    document_id: str
    error_message: str
    timestamp: datetime

class BatchProcessingStatus(BaseModel):
    batch_id: str
    status: BatchStatus
    stats: BatchProcessingStats
    errors: List[BatchProcessingError] = []
    metadata: Dict[str, str] = {} 