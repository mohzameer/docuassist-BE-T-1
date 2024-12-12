from typing import List, Optional, Dict, Any
from llama_index.core import (
    VectorStoreIndex,
    ServiceContext,
    StorageContext,
    load_index_from_storage,
)
from llama_index.vector_stores.azureaisearch import AzureAISearchVectorStore
from llama_index.embeddings.openai import OpenAIEmbedding
from app.core.config import get_settings
import logging

logger = logging.getLogger(__name__)
settings = get_settings()

class SearchService:
    def __init__(self):
        self.index_name = settings.AZURE_SEARCH_INDEX_NAME or "llm-search-index"
        
        self.vector_store = AzureAISearchVectorStore(
            azure_search_endpoint=settings.AZURE_SEARCH_SERVICE_ENDPOINT,
            azure_search_key=settings.AZURE_SEARCH_ADMIN_KEY,
            index_name=self.index_name,
            index_creation_params={
                "vector_search_dimensions": 1536,
                "semantic_search_config": "default",
                "fields": [
                    {
                        "name": "metadata",
                        "type": "Edm.String",
                        "filterable": True,
                        "searchable": True
                    },
                    {
                        "name": "document_id",
                        "type": "Edm.String",
                        "filterable": True,
                        "searchable": True
                    },
                    {
                        "name": "mime_type",
                        "type": "Edm.String",
                        "filterable": True
                    }
                ]
            }
        )
        
        self.embed_model = OpenAIEmbedding(api_key=settings.OPENAI_API_KEY)
        self.service_context = ServiceContext.from_defaults(
            embed_model=self.embed_model,
        )
        
        self.storage_context = StorageContext.from_defaults(
            vector_store=self.vector_store
        )
        
        try:
            self.index = load_index_from_storage(
                storage_context=self.storage_context,
                service_context=self.service_context,
            )
        except Exception as e:
            logger.info(f"Creating new index '{self.index_name}': {e}")
            self.index = None

    async def hybrid_search(
        self,
        query: str,
        filters: Optional[Dict[str, Any]] = None,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Perform hybrid search using both vector and keyword search
        """
        try:
            # Create query engine
            query_engine = self.index.as_query_engine(
                similarity_top_k=limit,
                filters=filters,
            )
            
            # Execute search
            response = await query_engine.aquery(query)
            
            # Process and format results
            results = []
            for node in response.source_nodes:
                result = {
                    "content": node.text,
                    "metadata": node.metadata,
                    "score": node.score if hasattr(node, 'score') else None,
                    "document_id": node.node_id,
                }
                results.append(result)
            
            return results
            
        except Exception as e:
            logger.error(f"Search error: {e}", exc_info=True)
            raise

    async def update_index(self, documents: List[Dict[str, Any]]):
        """
        Update the search index with new documents
        """
        try:
            # Create new index if it doesn't exist
            if not self.index:
                self.index = VectorStoreIndex.from_documents(
                    documents,
                    storage_context=self.storage_context,
                    service_context=self.service_context,
                )
            else:
                # Update existing index
                for doc in documents:
                    self.index.insert(doc)
                    
            logger.info(f"Successfully updated index with {len(documents)} documents")
            
        except Exception as e:
            logger.error(f"Index update error: {e}", exc_info=True)
            raise 