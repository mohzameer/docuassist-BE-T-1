from pydantic_settings import BaseSettings
from typing import Optional
from functools import lru_cache

class Settings(BaseSettings):
    # App Settings
    APP_NAME: str = "LLM Search Backend"
    API_V1_STR: str = "/api/v1"
    DEBUG: bool = False
    
    # Security
    SECRET_KEY: str
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    
    # Azure AI Search
    AZURE_SEARCH_SERVICE_ENDPOINT: str
    AZURE_SEARCH_ADMIN_KEY: str
    AZURE_SEARCH_INDEX_NAME: str
    
    # Google Drive
    GOOGLE_DRIVE_CREDENTIALS_FILE: str
    GOOGLE_DRIVE_TOKEN_FILE: Optional[str] = None
    GOOGLE_DRIVE_FOLDER_ID: str
    
    # LlamaIndex
    OPENAI_API_KEY: str
    
    # Search Configuration
    AZURE_SEARCH_SCORE_THRESHOLD: float = 0.7
    AZURE_SEARCH_K_MULTIPLIER: int = 2
    AZURE_SEARCH_ENABLE_RERANKING: bool = True
    AZURE_SEARCH_ENABLE_SEMANTIC: bool = True
    
    class Config:
        env_file = ".env"
        case_sensitive = True

@lru_cache()
def get_settings():
    return Settings() 