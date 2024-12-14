from typing import List, Optional, Dict, Any, Tuple
from datetime import datetime
import logging
import io
import os
import tempfile
import PyPDF2
from docx import Document
from PIL import Image, UnidentifiedImageError
import pytesseract
from app.models.document import DocumentResponse
from app.services.google_drive_service import GoogleDriveService, GOOGLE_MIME_TYPES
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
    'image/tiff',
    # Add Google Workspace MIME types
    'application/vnd.google-apps.document',
    'application/vnd.google-apps.spreadsheet',
    'application/vnd.google-apps.presentation',
    'application/vnd.google-apps.drawing',
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

    async def _process_pdf(self, file_content: bytes, filename: str) -> str:
        """Extract text from PDF file"""
        logger.info(f"Processing PDF file: {filename}")
        pdf_file = io.BytesIO(file_content)
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        text = ""
        for page_num, page in enumerate(pdf_reader.pages, 1):
            logger.debug(f"Processing page {page_num} of {filename}")
            text += page.extract_text() + "\n"
        return text

    async def _process_docx(self, file_content: bytes, filename: str) -> str:
        """Extract text from DOCX file"""
        logger.info(f"Processing DOCX file: {filename}")
        doc = Document(io.BytesIO(file_content))
        return "\n".join([paragraph.text for paragraph in doc.paragraphs])

    async def _process_image(self, file_content: bytes, filename: str) -> str:
        """Extract text from image using OCR"""
        logger.info(f"Processing image file: {filename}")
        try:
            # Create a temporary file to save the image
            with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as temp_file:
                try:
                    # Open image from bytes
                    with Image.open(io.BytesIO(file_content)) as image:
                        logger.debug(f"Image format: {image.format}, Mode: {image.mode}, Size: {image.size}")
                        
                        # Convert image to RGB mode if necessary
                        if image.mode not in ('L', 'RGB'):
                            logger.debug(f"Converting image mode from {image.mode} to RGB")
                            image = image.convert('RGB')
                        
                        # Save the image as PNG
                        image.save(temp_file.name, 'PNG')
                        logger.debug(f"Saved temporary image: {temp_file.name}")
                    
                    # Use pytesseract to extract text
                    text = pytesseract.image_to_string(temp_file.name)
                    extracted_text = text.strip()
                    logger.debug(f"Extracted {len(extracted_text)} characters from image")
                    return extracted_text
                    
                finally:
                    # Clean up the temporary file
                    try:
                        os.unlink(temp_file.name)
                        logger.debug(f"Cleaned up temporary file: {temp_file.name}")
                    except Exception as e:
                        logger.warning(f"Error cleaning up temporary file for {filename}: {e}")
            
        except UnidentifiedImageError:
            logger.error(f"Could not identify image format for file: {filename}")
            raise
        except Exception as e:
            logger.error(f"Error processing image {filename}: {str(e)}")
            raise

    async def _process_single_document(
        self,
        document: DocumentResponse
    ) -> tuple[Optional[str], Dict[str, Any]]:
        """Process a single document based on its mime type"""
        logger.info(f"Processing document: {document.title} (ID: {document.id}, Type: {document.mime_type})")
        
        try:
            # Skip unsupported file types
            if document.mime_type not in SUPPORTED_MIME_TYPES:
                logger.warning(f"Skipping unsupported file: {document.title} (Type: {document.mime_type})")
                return None, {"error": "Unsupported file type"}

            # Initialize metadata
            metadata = {
                "processed_timestamp": str(datetime.utcnow()),
                "original_mime_type": document.mime_type,
                "is_google_workspace": document.mime_type in GOOGLE_MIME_TYPES,
                "title": document.title
            }

            # Download and process the file
            logger.info(f"Downloading file: {document.title}")
            file_content, mime_type = await self.drive_service.download_file(
                document.drive_id,
                original_mime_type=document.mime_type
            )
            
            try:
                # Process based on mime type
                if mime_type == 'application/pdf':
                    content = await self._process_pdf(file_content, document.title)
                elif mime_type in ['application/vnd.openxmlformats-officedocument.wordprocessingml.document', 'application/msword']:
                    content = await self._process_docx(file_content, document.title)
                elif mime_type.startswith('image/'):
                    content = await self._process_image(file_content, document.title)
                else:
                    logger.warning(f"Unsupported mime type for processing: {document.title} (Type: {mime_type})")
                    return None, {"error": f"Unsupported mime type: {mime_type}"}
                
                if not content:
                    logger.warning(f"No content extracted from document: {document.title}")
                    return None, {"error": "No content extracted"}
                
                logger.info(f"Successfully processed document: {document.title} ({len(content)} characters extracted)")
                return content, metadata
                
            except Exception as e:
                logger.error(f"Error processing document {document.title}: {str(e)}", exc_info=True)
                raise
                
        except Exception as e:
            logger.error(f"Error processing document {document.title}: {str(e)}", exc_info=True)
            raise

    async def process_documents(
        self,
        documents: List[DocumentResponse]
    ) -> List[DocumentResponse]:
        """Process multiple documents and update search index"""
        logger.info(f"Starting batch processing of {len(documents)} documents")
        processed_docs = []
        failed_docs = []
        
        for document in documents:
            try:
                logger.info(f"Processing document: {document.title}")
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
                    logger.info(f"Successfully indexed document: {document.title}")
                else:
                    failed_docs.append(document.title)
            except Exception as e:
                failed_docs.append(document.title)
                logger.error(f"Failed to process document {document.title}: {str(e)}")
                continue
        
        logger.info(f"Batch processing completed. Processed: {len(processed_docs)}, Failed: {len(failed_docs)}")
        if failed_docs:
            logger.info(f"Failed documents: {', '.join(failed_docs)}")
        
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