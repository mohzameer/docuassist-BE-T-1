from typing import List, Optional, Dict, Any
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
from azure.core.credentials import AzureKeyCredential
from app.core.config import get_settings
from openai import AsyncOpenAI
import requests
import logging
import json
import aiohttp
import ssl
import certifi

logger = logging.getLogger(__name__)
settings = get_settings()

class SearchService:
    def __init__(self):
        self.index_name = settings.AZURE_SEARCH_INDEX_NAME or "documents-index"
        
        # Configure OpenAI client
        self.openai_client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
        
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
        
        # Create SSL context for aiohttp
        self.ssl_context = ssl.create_default_context(cafile=certifi.where())

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
                            "name": "content",
                            "type": "Edm.String",
                            "searchable": True
                        },
                        {
                            "name": "embedding",
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
                            "name": "document_id",
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
        Perform vector search with optional summarization
        """
        try:
            # Generate embedding for the query
            embedding_response = await self.openai_client.embeddings.create(
                model="text-embedding-ada-002",
                input=query
            )
            vector = embedding_response.data[0].embedding

            # Prepare search request according to Azure Search REST API
            search_request = {
                "count": True,
                "top": limit,
                "select": "content,embedding,metadata,document_id",
                "vectorQueries": [
                    {
                        "kind": "vector",
                        "vector": vector,
                        "fields": "embedding",
                        "k": limit
                    }
                ]
            }

            # Add filters if provided
            if filters:
                filter_conditions = []
                for key, value in filters.items():
                    if isinstance(value, (list, tuple)):
                        or_conditions = [f"{key} eq '{v}'" for v in value]
                        filter_conditions.append(f"({' or '.join(or_conditions)})")
                    else:
                        filter_conditions.append(f"{key} eq '{value}'")
                if filter_conditions:
                    search_request["filter"] = " and ".join(filter_conditions)

            # Log the search request for debugging
            logger.debug(f"Search request: {json.dumps(search_request, indent=2)}")

            # Execute search using REST API directly
            headers = {
                'Content-Type': 'application/json',
                'api-key': settings.AZURE_SEARCH_ADMIN_KEY,
                'Accept': 'application/json'
            }
            
            # Add API version to URL
            search_url = f"{settings.AZURE_SEARCH_SERVICE_ENDPOINT}/indexes/{self.index_name}/docs/search?api-version=2023-11-01"
            
            connector = aiohttp.TCPConnector(ssl=self.ssl_context)
            async with aiohttp.ClientSession(connector=connector) as session:
                async with session.post(search_url, headers=headers, json=search_request) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        logger.error(f"Search request failed: {error_text}")
                        raise Exception(f"Search request failed: {error_text}")
                    
                    search_results = await response.json()

            # Process results
            processed_results = []
            for result in search_results.get('value', []):
                try:
                    print("result")
                    print(result)
                    print("-------")
                    metadata = json.loads(result.get("metadata", "{}"))
                    content = result.get("content", "")

                    # If summarization is requested, use LLM
                    if summarize and content:
                        try:
                            chat_completion = await self.openai_client.chat.completions.create(
                                model="gpt-3.5-turbo",
                                messages=[
                                    {"role": "system", "content": "You are a helpful assistant that summarizes text."},
                                    {"role": "user", "content": f"Please summarize this text concisely:\n\n{content}"}
                                ],
                                temperature=0.7,
                                max_tokens=150
                            )
                            content = chat_completion.choices[0].message.content.strip()
                        except Exception as e:
                            logger.error(f"Error summarizing content: {e}")
                            # Fall back to original content if summarization fails

                    processed_results.append({
                        "content": content,
                        "metadata": metadata,
                        "score": result.get("@search.score"),
                        "document_id": result.get("document_id")
                    })

                except Exception as e:
                    logger.error(f"Error processing search result: {e}")
                    continue

            logger.info(f"Search query '{query}' returned {len(processed_results)} results")
            return processed_results

        except Exception as e:
            logger.error(f"Search error: {e}", exc_info=True)
            raise

    async def update_index(self, documents: List[Dict[str, Any]]):
        """
        Update the search index with new documents
        """
        try:
            # Prepare documents for indexing
            index_documents = []
            for doc in documents:
                # Generate embedding for the document
                embedding_response = await self.openai_client.embeddings.create(
                    model="text-embedding-ada-002",
                    input=doc["text"]
                )
                vector = embedding_response.data[0].embedding

                # Create document for indexing
                index_doc = {
                    "id": doc["metadata"]["document_id"],
                    "content": doc["text"],
                    "embedding": vector,
                    "metadata": json.dumps(doc["metadata"]),
                    "document_id": doc["metadata"]["document_id"]
                }
                index_documents.append(index_doc)

            # Upload documents in batches
            batch_size = 50
            for i in range(0, len(index_documents), batch_size):
                batch = index_documents[i:i + batch_size]
                try:
                    self.search_client.upload_documents(documents=batch)
                    logger.info(f"Indexed batch of {len(batch)} documents")
                except Exception as e:
                    logger.error(f"Error indexing batch: {e}")
                    raise

            logger.info(f"Successfully indexed {len(documents)} documents")

        except Exception as e:
            logger.error(f"Index update error: {e}", exc_info=True)
            raise 