from typing import List, Optional, Dict, Any, Tuple
from datetime import datetime
import logging
import io
import PyPDF2
from docx import Document
from PIL import Image
import pytesseract
from app.models.document import DocumentResponse
from app.services.google_drive_service import GoogleDriveService
from app.services.search_service import SearchService
from app.models.batch import BatchProcessingStatus
import uuid
import asyncio

logger = logging.getLogger(__name__)

SUPPORTED_MIME_TYPES = {
    'text/plain',
    'text/csv',
    'application/pdf',
    'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
    'application/msword',
    'image/jpeg',
    'image/png',
    'image/gif',
    'image/tiff'
}

class DocumentService:
    def __init__(
        self,
        drive_service: GoogleDriveService,
        search_service: SearchService
    ):
        self.drive_service = drive_service
        self.search_service = search_service
        self._batch_status = {}

    async def _process_pdf(self, file_content: bytes) -> str:
        """Extract text from PDF file"""
        pdf_file = io.BytesIO(file_content)
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"
        return text

    async def _process_docx(self, file_content: bytes) -> str:
        """Extract text from DOCX file"""
        doc = Document(io.BytesIO(file_content))
        return "\n".join([paragraph.text for paragraph in doc.paragraphs])

    async def _process_image(self, file_content: bytes) -> str:
        """Extract text from image using OCR"""
        image = Image.open(io.BytesIO(file_content))
        return pytesseract.image_to_string(image)

    async def _process_single_document(
        self,
        document: DocumentResponse
    ) -> tuple[Optional[str], Dict[str, Any]]:
        """Process a single document based on its mime type"""
        try:
            file_content, mime_type = await self.drive_service.download_file(document.drive_id)
            
            # Initialize metadata
            metadata = {
                "processed_timestamp": str(datetime.utcnow()),
                "original_mime_type": mime_type,
            }

            # Process based on mime type
            if mime_type.startswith('text/'):
                content = file_content.decode('utf-8')
                
            elif mime_type == 'application/pdf':
                content = await self._process_pdf(file_content)
                
            elif mime_type in ['application/vnd.openxmlformats-officedocument.wordprocessingml.document',
                             'application/msword']:
                content = await self._process_docx(file_content)
                
            elif mime_type.startswith('image/'):
                content = await self._process_image(file_content)
                metadata['image_processed'] = True
                
            else:
                logger.warning(f"Unsupported mime type: {mime_type} for document {document.id}")
                return None, metadata

            return content, metadata

        except Exception as e:
            logger.error(f"Error processing document {document.id}: {e}", exc_info=True)
            raise

    async def process_documents(
        self,
        documents: List[DocumentResponse]
    ) -> List[DocumentResponse]:
        """Process multiple documents and update search index"""
        processed_docs = []
        
        for document in documents:
            content, metadata = await self._process_single_document(document)
            if content:
                # Update search index
                await self.search_service.update_index([{
                    "text": content,
                    "metadata": {
                        **metadata,
                        "document_id": document.id,
                        "title": document.title,
                        "drive_id": document.drive_id
                    }
                }])
                processed_docs.append(document)
        
        return processed_docs

    async def start_batch_processing(
        self,
        documents: List[DocumentResponse]
    ) -> BatchProcessingStatus:
        """Start batch processing of documents"""
        batch_id = str(uuid.uuid4())
        total = len(documents)
        
        self._batch_status[batch_id] = BatchProcessingStatus(
            batch_id=batch_id,
            total=total,
            processed=0,
            status="processing"
        )
        
        # Start processing in background
        asyncio.create_task(self._process_batch(batch_id, documents))
        
        return self._batch_status[batch_id]

    async def _process_batch(
        self,
        batch_id: str,
        documents: List[DocumentResponse]
    ):
        """Process documents in batch"""
        try:
            for document in documents:
                content, metadata = await self._process_single_document(document)
                if content:
                    await self.search_service.update_index([{
                        "text": content,
                        "metadata": {
                            **metadata,
                            "document_id": document.id,
                            "title": document.title,
                            "drive_id": document.drive_id
                        }
                    }])
                
                self._batch_status[batch_id].processed += 1
                
            self._batch_status[batch_id].status = "completed"
            
        except Exception as e:
            logger.error(f"Batch processing error: {e}", exc_info=True)
            self._batch_status[batch_id].status = "failed"
            self._batch_status[batch_id].error = str(e)

    async def get_batch_status(self, batch_id: str) -> Optional[BatchProcessingStatus]:
        """Get status of a batch processing job"""
        return self._batch_status.get(batch_id)

    async def get_processing_progress(self, batch_id: str):
        """Stream batch processing progress"""
        while True:
            status = self._batch_status.get(batch_id)
            if not status:
                break
                
            yield status
            
            if status.status in ["completed", "failed"]:
                break
                
            await asyncio.sleep(1) 