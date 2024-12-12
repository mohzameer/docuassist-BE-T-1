from fastapi import Depends
from app.services.google_drive_service import GoogleDriveService
from app.services.document_service import DocumentService
from app.services.search_service import SearchService

def get_drive_service() -> GoogleDriveService:
    return GoogleDriveService()

def get_search_service() -> SearchService:
    return SearchService()

def get_document_service(
    drive_service: GoogleDriveService = Depends(get_drive_service),
    search_service: SearchService = Depends(get_search_service)
) -> DocumentService:
    return DocumentService(drive_service=drive_service, search_service=search_service) 