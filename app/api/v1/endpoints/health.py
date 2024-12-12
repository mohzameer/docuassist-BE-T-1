from fastapi import APIRouter, Depends
from app.services.search_service import SearchService
from app.services.google_drive_service import GoogleDriveService

router = APIRouter()

@router.get("/health")
async def health_check(
    search_service: SearchService = Depends(),
    drive_service: GoogleDriveService = Depends(),
):
    """Check the health of all services"""
    return {
        "status": "healthy",
        "services": {
            "search": await check_search_service(search_service),
            "drive": await check_drive_service(drive_service)
        }
    } 