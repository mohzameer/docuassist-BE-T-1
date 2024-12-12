import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch

def test_search_endpoint(test_client: TestClient, mock_search_service):
    # Mock search response
    mock_search_service.hybrid_search.return_value = [
        {
            "content": "Test content",
            "metadata": {"title": "Test Doc"},
            "score": 0.95,
            "document_id": "test123"
        }
    ]
    
    with patch("app.api.v1.endpoints.search.SearchService", return_value=mock_search_service):
        response = test_client.post(
            "/api/v1/search",
            json={"query": "test query", "filters": {}, "limit": 10}
        )
        
        assert response.status_code == 200
        assert len(response.json()) == 1
        assert response.json()[0]["document_id"] == "test123" 