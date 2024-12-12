from pydantic import BaseModel, Field
from typing import Optional, Dict, Any
from datetime import datetime

class DocumentBase(BaseModel):
    title: str
    mime_type: str
    drive_id: str
    parent_folder_id: Optional[str] = None
    
class DocumentCreate(DocumentBase):
    content: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)

class DocumentResponse(DocumentBase):
    id: str
    created_at: datetime
    updated_at: datetime
    size: Optional[int] = None
    web_view_link: Optional[str] = None
    thumbnail_link: Optional[str] = None
    metadata: Dict[str, Any]

class DocumentUpdate(BaseModel):
    title: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None 