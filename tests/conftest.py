import pytest
from fastapi.testclient import TestClient
from app.main import app
from app.core.config import get_settings
from app.services.search_service import SearchService
from app.services.google_drive_service import GoogleDriveService
from unittest.mock import Mock

@pytest.fixture
def test_client():
    return TestClient(app)

@pytest.fixture
def mock_search_service():
    return Mock(spec=SearchService)

@pytest.fixture
def mock_drive_service():
    return Mock(spec=GoogleDriveService)

@pytest.fixture
def test_settings():
    return get_settings() 