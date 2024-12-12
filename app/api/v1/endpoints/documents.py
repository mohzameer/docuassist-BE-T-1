from fastapi import APIRouter, HTTPException, Query, BackgroundTasks, Depends
from fastapi.responses import StreamingResponse
from typing import List, Optional
from app.core.dependencies import get_drive_service, get_document_service
from app.services.google_drive_service import GoogleDriveService
from app.services.document_service import DocumentService
from app.models.document import DocumentResponse
from app.models.batch import BatchProcessingStatus
import logging

logger = logging.getLogger(__name__)
router = APIRouter()

@router.get("/documents", response_model=List[DocumentResponse])
async def list_documents(
    folder_id: Optional[str] = Query(None, description="Specific folder ID to list files from"),
    recursive: bool = Query(False, description="Whether to list files from subfolders"),
    mime_types: Optional[List[str]] = Query(None, description="Filter by MIME types"),
    drive_service: GoogleDriveService = Depends(get_drive_service),
):
    """
    List documents from Google Drive
    """
    try:
        documents = await drive_service.list_files(
            folder_id=folder_id,
            mime_types=mime_types,
            recursive=recursive
        )
        return documents
    except Exception as e:
        logger.error(f"Error listing documents: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="Failed to list documents"
        )

@router.post("/documents/sync", response_model=List[DocumentResponse])
async def sync_documents(
    drive_service: GoogleDriveService = Depends(get_drive_service),
    document_service: DocumentService = Depends(get_document_service),
):
    """
    Synchronize documents from Google Drive and update the search index
    """
    try:
        # Get documents from Google Drive
        documents = await drive_service.list_files(recursive=True)
        
        # Process and index documents
        processed_docs = await document_service.process_documents(documents)
        
        return processed_docs
    except Exception as e:
        logger.error(f"Error syncing documents: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="Failed to sync documents"
        )

@router.post("/documents/batch", response_model=BatchProcessingStatus)
async def start_batch_processing(
    folder_id: Optional[str] = Query(None),
    recursive: bool = Query(False),
    drive_service: GoogleDriveService = Depends(get_drive_service),
    document_service: DocumentService = Depends(get_document_service),
):
    """
    Start batch processing of documents from Google Drive
    """
    try:
        # Get documents from Google Drive
        documents = await drive_service.list_files(
            folder_id=folder_id,
            recursive=recursive
        )
        
        # Start batch processing
        batch_status = await document_service.start_batch_processing(documents)
        return batch_status
        
    except Exception as e:
        logger.error(f"Error starting batch processing: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="Failed to start batch processing"
        )

@router.get("/documents/batch/{batch_id}", response_model=BatchProcessingStatus)
async def get_batch_status(
    batch_id: str,
    document_service: DocumentService = Depends(get_document_service),
):
    """
    Get the status of a batch processing job
    """
    status = await document_service.get_batch_status(batch_id)
    if not status:
        raise HTTPException(
            status_code=404,
            detail="Batch processing job not found"
        )
    return status

@router.get("/documents/batch/{batch_id}/stream")
async def stream_batch_progress(
    batch_id: str,
    document_service: DocumentService = Depends(get_document_service),
):
    """
    Stream batch processing progress updates
    """
    async def progress_generator():
        async for status in document_service.get_processing_progress(batch_id):
            yield f"data: {status.json()}\n\n"

    return StreamingResponse(
        progress_generator(),
        media_type="text/event-stream"
    ) 