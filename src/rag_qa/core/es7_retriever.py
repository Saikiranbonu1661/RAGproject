"""
Elasticsearch 7.x Compatible Retriever

Uses script_score queries with cosineSimilarity for ES versions
that don't support native KNN indexing.
"""

import logging
from typing import List, Any
from langchain.schema import Document
from langchain.callbacks.manager import CallbackManagerForRetrieverRun
from langchain.schema import BaseRetriever

logger = logging.getLogger(__name__)


class ES7ScriptScoreRetriever(BaseRetriever):
    """Retriever using script_score for ES 7.x compatibility."""
    
    es_client: Any
    index_name: str
    embedding_function: Any
    top_k: int = 5
    
    class Config:
        """Pydantic configuration."""
        arbitrary_types_allowed = True
    
    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun = None
    ) -> List[Document]:
        """
        Retrieve documents using script_score with cosineSimilarity.
        
        Args:
            query: Search query
            
        Returns:
            List of relevant documents
        """
        # Generate query embedding
        query_vector = self.embedding_function.embed_query(query)
        
        # Build script_score query for cosine similarity
        search_query = {
            "query": {
                "script_score": {
                    "query": {"match_all": {}},
                    "script": {
                        "source": "cosineSimilarity(params.query_vector, 'vector_field') + 1.0",
                        "params": {"query_vector": query_vector}
                    }
                }
            },
            "size": self.top_k
        }
        
        try:
            # Execute search
            response = self.es_client.search(
                index=self.index_name,
                body=search_query
            )
            
            # Convert to LangChain documents
            documents = []
            for hit in response['hits']['hits']:
                source = hit['_source']
                
                doc = Document(
                    page_content=source.get('text', ''),
                    metadata=source.get('metadata', {})
                )
                documents.append(doc)
            
            logger.info(f"Retrieved {len(documents)} documents for query: {query[:50]}...")
            return documents
            
        except Exception as e:
            logger.error(f"Error retrieving documents: {e}")
            return []
    
    async def _aget_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun = None
    ) -> List[Document]:
        """Async version - not implemented."""
        raise NotImplementedError("Async retrieval not supported")

