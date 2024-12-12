from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.openapi.docs import get_swagger_ui_html, get_redoc_html
from app.core.config import get_settings
from app.api.v1.router import api_router
from app.core.logging_config import setup_logging
import logging

settings = get_settings()
logger = logging.getLogger(__name__)

app = FastAPI(
    title=settings.APP_NAME,
    description="""
    LLM Search Backend API with document processing and semantic search capabilities.
    
    Features:
    * Google Drive integration
    * Document processing (PDF, DOCX, Images)
    * Semantic search with Azure AI Search
    * Batch processing with real-time progress tracking
    """,
    version="1.0.0",
    openapi_url=f"{settings.API_V1_STR}/openapi.json",
)

# CORS middleware configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Global exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"}
    )

# Include API router
app.include_router(api_router, prefix=settings.API_V1_STR)

# Startup event
@app.on_event("startup")
async def startup_event():
    setup_logging()
    logger.info("Application startup")

# Shutdown event
@app.on_event("shutdown")
async def shutdown_event():
    logger.info("Application shutdown") 