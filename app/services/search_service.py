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

    async def _rerank_results(self, query: str, results: List[dict]) -> List[dict]:
        """Rerank results using LLM to improve relevance"""
        if not settings.AZURE_SEARCH_ENABLE_RERANKING:
            return results

        try:
            rerank_prompt = f"""Given the search query: "{query}"
            Rate how relevant each result is on a scale of 0-1, where 1 is most relevant.
            Consider semantic meaning, not just keyword matches.
            Return only the numeric score.
            
            Content to rate: {{content}}
            Relevance score:"""
            
            reranked = []
            for result in results:
                chat_completion = await self.openai_client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": "You are a relevance scoring assistant. Respond only with a number between 0 and 1."},
                        {"role": "user", "content": rerank_prompt.format(content=result["content"][:1000])}
                    ],
                    temperature=0.3
                )
                
                try:
                    llm_score = float(chat_completion.choices[0].message.content.strip())
                    result["score"] = (result.get("score", 0) + llm_score) / 2  # Combine scores
                except ValueError:
                    continue
                    
                reranked.append(result)
                
            return sorted(reranked, key=lambda x: x["score"], reverse=True)
        except Exception as e:
            logger.error(f"Error in reranking: {e}")
            return results

    def _get_semantic_window(self, content: str, query: str, window_size: int = 3) -> str:
        """Get semantic window around the most relevant part of the content"""
        sentences = content.split('. ')
        if len(sentences) <= window_size:
            return content
            
        best_score = 0
        best_window = content
        
        for i in range(len(sentences) - window_size + 1):
            window = '. '.join(sentences[i:i + window_size])
            # Simple relevance scoring
            score = sum(1 for word in query.lower().split() if word in window.lower())
            if score > best_score:
                best_score = score
                best_window = window
                
        return best_window

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
            # Generate embedding with context
            embedding_response = await self.openai_client.embeddings.create(
                model="text-embedding-ada-002",
                input=f"Search query for document retrieval: {query}"  # Add context
            )
            vector = embedding_response.data[0].embedding

            # Prepare search request according to Azure Search REST API
            search_request = {
                "count": True,
                "top": limit,
                "select": "content,embedding,metadata,document_id",
                "search": query if settings.AZURE_SEARCH_ENABLE_SEMANTIC else None,
                "queryType": "semantic" if settings.AZURE_SEARCH_ENABLE_SEMANTIC else "simple",
                "searchMode": "all",
                "semanticConfiguration": "default-config" if settings.AZURE_SEARCH_ENABLE_SEMANTIC else None,
                "vectorQueries": [
                    {
                        "kind": "vector",
                        "vector": vector,
                        "fields": "embedding",
                        "k": limit * settings.AZURE_SEARCH_K_MULTIPLIER
                    }
                ],
                "vectorFilterMode": "preFilter" if settings.AZURE_SEARCH_ENABLE_SEMANTIC else None
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
            
            search_url = f"{settings.AZURE_SEARCH_SERVICE_ENDPOINT}/indexes/{self.index_name}/docs/search?api-version=2023-11-01"
            
            # Log the full request for debugging
            logger.info(f"Search URL: {search_url}")
            logger.info(f"Search headers: {headers}")
            logger.info(f"Search request body: {json.dumps(search_request, indent=2)}")
            
            connector = aiohttp.TCPConnector(ssl=self.ssl_context)
            async with aiohttp.ClientSession(connector=connector) as session:
                async with session.post(search_url, headers=headers, json=search_request) as response:
                    response_text = await response.text()
                    logger.info(f"Search response status: {response.status}")
                    logger.info(f"Search response body: {response_text}")
                    
                    if response.status != 200:
                        logger.error(f"Search request failed: {response_text}")
                        raise Exception(f"Search request failed: {response_text}")
                    
                    search_results = json.loads(response_text)

            # Log the number of results
            logger.info(f"Raw search results count: {len(search_results.get('value', []))}")

            # Process results
            processed_results = []
            for result in search_results.get('value', []):
                try:
                    logger.info(f"Processing result: {json.dumps(result, indent=2)}")
                    
                    # Get both scores
                    vector_score = float(result.get("@search.score", 0))
                    reranker_score = float(result.get("@search.rerankerScore", 0))
                    
                    logger.info(f"Vector score: {vector_score}, Reranker score: {reranker_score}")
                    
                    # Use reranker score if available, otherwise use vector score
                    if reranker_score > 0:
                        # Normalize reranker score from 0-4 to 0-1 range
                        normalized_score = reranker_score / 4.0
                    else:
                        # Vector score is already in 0-1 range
                        normalized_score = vector_score
                    
                    logger.info(f"Final normalized score: {normalized_score}, threshold: {settings.AZURE_SEARCH_SCORE_THRESHOLD}")
                    
                    # Apply score threshold
                    if normalized_score < settings.AZURE_SEARCH_SCORE_THRESHOLD:
                        logger.info(f"Skipping result due to low normalized score: {normalized_score}")
                        continue

                    metadata = json.loads(result.get("metadata", "{}"))
                    content = result.get("content", "")

                    # Get semantic window around relevant content
                    content = self._get_semantic_window(content, query)

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

                    processed_results.append({
                        "content": content,
                        "metadata": metadata,
                        "score": normalized_score,
                        "vector_score": vector_score,
                        "reranker_score": reranker_score,  # Include both scores in response
                        "document_id": result.get("document_id")
                    })

                except Exception as e:
                    logger.error(f"Error processing search result: {e}")
                    continue

            # Rerank results if enabled
            if processed_results and settings.AZURE_SEARCH_ENABLE_RERANKING:
                processed_results = await self._rerank_results(query, processed_results)

            logger.info(f"Final processed results count: {len(processed_results)}")
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