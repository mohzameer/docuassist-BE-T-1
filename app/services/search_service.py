from typing import List, Optional, Dict, Any
from llama_index.core import (
    VectorStoreIndex,
    StorageContext,
    Document,
    Settings as LlamaSettings
)
from llama_index.vector_stores.azureaisearch import AzureAISearchVectorStore
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
from azure.core.credentials import AzureKeyCredential
from app.core.config import get_settings
import openai
import requests
import logging
import json

logger = logging.getLogger(__name__)
settings = get_settings()

class SearchService:
    def __init__(self):
        self.index_name = settings.AZURE_SEARCH_INDEX_NAME or "documents-index"
        
        # Configure OpenAI settings
        openai.api_key = settings.OPENAI_API_KEY
        
        # Create Azure Search clients
        self.credential = AzureKeyCredential(settings.AZURE_SEARCH_ADMIN_KEY)
        
        # Create index client
        self.index_client = SearchIndexClient(
            endpoint=settings.AZURE_SEARCH_SERVICE_ENDPOINT,
            credential=self.credential
        )
        
        # Ensure index exists before creating search client
        self._ensure_index_exists()
        
        # Create search client
        self.search_client = SearchClient(
            endpoint=settings.AZURE_SEARCH_SERVICE_ENDPOINT,
            index_name=self.index_name,
            credential=self.credential
        )
        
        # Configure embeddings model
        self.embed_model = OpenAIEmbedding(api_key=settings.OPENAI_API_KEY)
        
        # Configure LLM
        self.llm = OpenAI(api_key=settings.OPENAI_API_KEY, model="gpt-3.5-turbo")
        
        # Configure global settings
        LlamaSettings.embed_model = self.embed_model
        LlamaSettings.llm = self.llm
        LlamaSettings.chunk_size = 1024
        LlamaSettings.chunk_overlap = 20
        
        # Initialize vector store with search client
        self.vector_store = AzureAISearchVectorStore(
            search_or_index_client=self.search_client,
            id_field_key="id",
            chunk_field_key="chunk",
            embedding_field_key="vector",
            metadata_string_field_key="metadata",
            doc_id_field_key="doc_id",
            vector_field_key="vector"
        )
        
        # Initialize storage context
        self.storage_context = StorageContext.from_defaults(
            vector_store=self.vector_store
        )
        
        # Create index object
        self.index = VectorStoreIndex.from_vector_store(
            vector_store=self.vector_store,
            embed_model=self.embed_model,
            llm=self.llm,
            show_progress=True
        )
        logger.info(f"Initialized index: {self.index_name}")

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
            else:
                logger.info(f"Using existing index: {self.index_name}")
                
        except Exception as e:
            logger.error(f"Error ensuring index exists: {e}", exc_info=True)
            raise

    async def search(
        self,
        query: str,
        filters: Optional[Dict[str, Any]] = None,
        limit: int = 10,
        summarize: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Perform search with optional LLM-powered summarization
        """
        try:
            # Check if index exists and has been initialized
            if not self.index:
                logger.warning("Search index is not initialized. No documents have been indexed yet.")
                return []
            
            # Convert filters to OData filter string for Azure Search
            filter_str = None
            if filters:
                filter_conditions = []
                for key, value in filters.items():
                    if isinstance(value, (list, tuple)):
                        or_conditions = [f"{key} eq '{v}'" for v in value]
                        filter_conditions.append(f"({' or '.join(or_conditions)})")
                    else:
                        filter_conditions.append(f"{key} eq '{value}'")
                if filter_conditions:
                    filter_str = " and ".join(filter_conditions)
                logger.debug(f"Created filter string: {filter_str}")
            
            # Create query engine
            query_engine = self.index.as_query_engine(
                similarity_top_k=limit,
                vector_store_kwargs={
                    "filter": filter_str,
                    "search_client": self.search_client
                } if filter_str else {
                    "search_client": self.search_client
                }
            )
            
            # Execute search
            logger.debug(f"Executing search with query: {query}")
            response = await query_engine.aquery(query)
            
            # Process and format results
            results = []
            for node in response.source_nodes:
                # Parse metadata string back to dictionary
                try:
                    metadata = json.loads(node.metadata.get("metadata_str", "{}"))
                except json.JSONDecodeError:
                    metadata = node.metadata
                
                content = node.text
                
                # If summarization is requested, use LLM to summarize the content
                if summarize and content:
                    try:
                        summary_prompt = f"Please summarize the following text concisely:\n\n{content}\n\nSummary:"
                        summary_response = await self.llm.acomplete(summary_prompt)
                        content = summary_response.text.strip()
                    except Exception as e:
                        logger.error(f"Error summarizing content: {e}")
                        # Fall back to original content if summarization fails
                
                result = {
                    "content": content,
                    "metadata": metadata,
                    "score": node.score if hasattr(node, 'score') else None,
                    "document_id": node.node_id,
                }
                results.append(result)
            
            logger.info(f"Search query '{query}' returned {len(results)} results")
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
                logger.info("Creating new VectorStoreIndex")
                self.index = VectorStoreIndex.from_documents(
                    llama_docs,
                    storage_context=self.storage_context,
                )
            else:
                # Update existing index
                logger.info(f"Updating existing index with {len(documents)} documents")
                for doc in llama_docs:
                    self.index.insert(doc)
                    
            logger.info(f"Successfully updated index with {len(documents)} documents")
            
        except Exception as e:
            logger.error(f"Index update error: {e}", exc_info=True)
            raise 