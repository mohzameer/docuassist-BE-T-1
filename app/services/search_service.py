from typing import List, Optional, Dict, Any
from llama_index.core import (
    VectorStoreIndex,
    StorageContext,
    load_index_from_storage,
    Document
)
from llama_index.vector_stores.azureaisearch import AzureAISearchVectorStore
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.settings import Settings
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
from azure.core.credentials import AzureKeyCredential
from app.core.config import get_settings
import requests
import logging
import json

logger = logging.getLogger(__name__)
settings = get_settings()

class SearchService:
    def __init__(self):
        self.index_name = settings.AZURE_SEARCH_INDEX_NAME or "documents-index"
        
        # Create Azure Search clients
        self.credential = AzureKeyCredential(settings.AZURE_SEARCH_ADMIN_KEY)
        self.index_client = SearchIndexClient(
            endpoint=settings.AZURE_SEARCH_SERVICE_ENDPOINT,
            credential=self.credential
        )
        
        # Ensure index exists
        self._ensure_index_exists()
        
        # Create search client
        self.search_client = SearchClient(
            endpoint=settings.AZURE_SEARCH_SERVICE_ENDPOINT,
            index_name=self.index_name,
            credential=self.credential
        )
        
        # Initialize vector store
        self.vector_store = AzureAISearchVectorStore(
            search_or_index_client=self.search_client,
            id_field_key="id",
            chunk_field_key="chunk",
            embedding_field_key="vector",
            metadata_string_field_key="metadata",
            doc_id_field_key="doc_id",
            vector_field_key="vector",
        )
        
        # Configure llama-index settings
        self.embed_model = OpenAIEmbedding(api_key=settings.OPENAI_API_KEY)
        Settings.embed_model = self.embed_model
        Settings.chunk_size = 1024
        Settings.chunk_overlap = 20
        
        self.storage_context = StorageContext.from_defaults(
            vector_store=self.vector_store
        )
        
        try:
            self.index = load_index_from_storage(
                storage_context=self.storage_context,
            )
        except Exception as e:
            logger.info(f"Creating new index '{self.index_name}': {e}")
            self.index = None

    def _ensure_index_exists(self):
        """Create the search index if it doesn't exist"""
        try:
            # Check if index exists
            if self.index_name not in [index.name for index in self.index_client.list_indexes()]:
                logger.info(f"Creating new Azure Search index: {self.index_name}")
                
                # Create index using REST API directly
                headers = {
                    'Content-Type': 'application/json',
                    'api-key': settings.AZURE_SEARCH_ADMIN_KEY
                }
                
                index_definition = {
                    "name": self.index_name,
                    "fields": [
                        {
                            "name": "id",
                            "type": "Edm.String",
                            "key": True,
                            "sortable": True,
                            "filterable": True
                        },
                        {
                            "name": "chunk",
                            "type": "Edm.String",
                            "searchable": True
                        },
                        {
                            "name": "vector",
                            "type": "Collection(Edm.Single)",
                            "searchable": True,
                            "dimensions": 1536,
                            "vectorSearchProfile": "my-profile"
                        },
                        {
                            "name": "metadata",
                            "type": "Edm.String",
                            "filterable": True,
                            "searchable": True
                        },
                        {
                            "name": "doc_id",
                            "type": "Edm.String",
                            "filterable": True
                        },
                        {
                            "name": "mime_type",
                            "type": "Edm.String",
                            "filterable": True
                        }
                    ],
                    "vectorSearch": {
                        "algorithms": [
                            {
                                "name": "my-hnsw",
                                "kind": "hnsw",
                                "hnswParameters": {
                                    "m": 4,
                                    "efConstruction": 400,
                                    "efSearch": 500,
                                    "metric": "cosine"
                                }
                            }
                        ],
                        "profiles": [
                            {
                                "name": "my-profile",
                                "algorithm": "my-hnsw"
                            }
                        ]
                    }
                }
                
                # Make direct REST API call
                response = requests.post(
                    f"{settings.AZURE_SEARCH_SERVICE_ENDPOINT}/indexes?api-version=2023-11-01",
                    headers=headers,
                    json=index_definition
                )
                
                if response.status_code not in (200, 201):
                    logger.error(f"Failed to create index: {response.text}")
                    raise Exception(f"Failed to create index: {response.text}")
                    
                logger.info(f"Successfully created index: {self.index_name}")
                
        except Exception as e:
            logger.error(f"Error ensuring index exists: {e}", exc_info=True)
            raise

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
            # Convert dictionaries to Document objects
            llama_docs = []
            for doc in documents:
                # Convert metadata to string to ensure it's serializable
                metadata_str = json.dumps(doc["metadata"])
                
                # Create Document object
                llama_doc = Document(
                    text=doc["text"],
                    metadata={"metadata_str": metadata_str},
                    id_=doc["metadata"]["document_id"]  # Use document_id as the unique identifier
                )
                llama_docs.append(llama_doc)
            
            # Create new index if it doesn't exist
            if not self.index:
                self.index = VectorStoreIndex.from_documents(
                    llama_docs,
                    storage_context=self.storage_context,
                )
            else:
                # Update existing index
                for doc in llama_docs:
                    self.index.insert(doc)
                    
            logger.info(f"Successfully updated index with {len(documents)} documents")
            
        except Exception as e:
            logger.error(f"Index update error: {e}", exc_info=True)
            raise 