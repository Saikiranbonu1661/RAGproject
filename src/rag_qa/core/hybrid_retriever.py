"""
Advanced Hybrid Retriever with BM25 + Semantic Search

Combines keyword-based BM25 search with vector semantic search
using Reciprocal Rank Fusion (RRF) for improved retrieval quality.
"""

import logging
from typing import List, Any, Optional, Tuple
from langchain.schema import Document
from langchain.callbacks.manager import CallbackManagerForRetrieverRun
from langchain.schema import BaseRetriever

logger = logging.getLogger(__name__)


class HybridRetriever(BaseRetriever):
    """
    Hybrid Retriever combining BM25 (keyword) and semantic (vector) search.
    
    Uses Reciprocal Rank Fusion (RRF) to merge results from both search methods.
    RRF score = sum(1 / (k + rank)) where k is a constant (default 60).
    
    Attributes:
        es_client: Elasticsearch client
        index_name: Name of the ES index
        embedding_function: Function to generate embeddings
        top_k: Number of final results to return
        top_k_initial: Number of results to fetch from each search method
        alpha: Weight for semantic search (1-alpha for BM25). Default 0.5
        rrf_k: RRF constant (default 60, higher = more weight to lower ranks)
        use_rrf: If True, use RRF fusion. If False, use weighted score fusion.
    """
    
    es_client: Any
    index_name: str
    embedding_function: Any
    top_k: int = 5
    top_k_initial: int = 20  # Fetch more initially for better fusion
    alpha: float = 0.5  # Weight for semantic search
    rrf_k: int = 60  # RRF constant
    use_rrf: bool = True  # Use RRF by default
    text_field: str = "text"  # Field name for text content
    vector_field: str = "vector_field"  # Field name for vectors
    
    class Config:
        """Pydantic configuration."""
        arbitrary_types_allowed = True
    
    def _semantic_search(self, query: str) -> List[Tuple[str, float, dict]]:
        """
        Perform semantic (vector) search.
        
        Returns:
            List of (doc_id, score, source) tuples
        """
        query_vector = self.embedding_function.embed_query(query)
        
        search_query = {
            "query": {
                "script_score": {
                    "query": {"match_all": {}},
                    "script": {
                        "source": f"cosineSimilarity(params.query_vector, '{self.vector_field}') + 1.0",
                        "params": {"query_vector": query_vector}
                    }
                }
            },
            "size": self.top_k_initial
        }
        
        try:
            response = self.es_client.search(index=self.index_name, body=search_query)
            results = []
            for hit in response['hits']['hits']:
                doc_id = hit['_id']
                score = hit['_score']
                source = hit['_source']
                results.append((doc_id, score, source))
            return results
        except Exception as e:
            logger.error(f"Semantic search error: {e}")
            return []
    
    def _bm25_search(self, query: str) -> List[Tuple[str, float, dict]]:
        """
        Perform BM25 (keyword) search.
        
        Returns:
            List of (doc_id, score, source) tuples
        """
        search_query = {
            "query": {
                "match": {
                    self.text_field: {
                        "query": query,
                        "fuzziness": "AUTO"  # Handle typos
                    }
                }
            },
            "size": self.top_k_initial
        }
        
        try:
            response = self.es_client.search(index=self.index_name, body=search_query)
            results = []
            for hit in response['hits']['hits']:
                doc_id = hit['_id']
                score = hit['_score']
                source = hit['_source']
                results.append((doc_id, score, source))
            return results
        except Exception as e:
            logger.error(f"BM25 search error: {e}")
            return []
    
    def _reciprocal_rank_fusion(
        self, 
        semantic_results: List[Tuple[str, float, dict]], 
        bm25_results: List[Tuple[str, float, dict]]
    ) -> List[Tuple[str, float, dict]]:
        """
        Combine results using Reciprocal Rank Fusion (RRF).
        
        RRF score = sum(1 / (k + rank)) for each ranking list
        
        Args:
            semantic_results: Results from semantic search
            bm25_results: Results from BM25 search
            
        Returns:
            Fused and sorted results
        """
        doc_scores = {}  # doc_id -> (rrf_score, source)
        
        # Process semantic results
        for rank, (doc_id, _, source) in enumerate(semantic_results, 1):
            rrf_score = 1.0 / (self.rrf_k + rank)
            if doc_id in doc_scores:
                doc_scores[doc_id] = (doc_scores[doc_id][0] + rrf_score, source)
            else:
                doc_scores[doc_id] = (rrf_score, source)
        
        # Process BM25 results
        for rank, (doc_id, _, source) in enumerate(bm25_results, 1):
            rrf_score = 1.0 / (self.rrf_k + rank)
            if doc_id in doc_scores:
                doc_scores[doc_id] = (doc_scores[doc_id][0] + rrf_score, source)
            else:
                doc_scores[doc_id] = (rrf_score, source)
        
        # Sort by RRF score (descending)
        sorted_results = sorted(
            [(doc_id, score, source) for doc_id, (score, source) in doc_scores.items()],
            key=lambda x: x[1],
            reverse=True
        )
        
        return sorted_results
    
    def _weighted_score_fusion(
        self, 
        semantic_results: List[Tuple[str, float, dict]], 
        bm25_results: List[Tuple[str, float, dict]]
    ) -> List[Tuple[str, float, dict]]:
        """
        Combine results using weighted score fusion.
        
        Combined score = alpha * norm_semantic + (1-alpha) * norm_bm25
        
        Args:
            semantic_results: Results from semantic search
            bm25_results: Results from BM25 search
            
        Returns:
            Fused and sorted results
        """
        doc_scores = {}  # doc_id -> {'semantic': score, 'bm25': score, 'source': source}
        
        # Normalize semantic scores (min-max normalization)
        if semantic_results:
            sem_scores = [s for _, s, _ in semantic_results]
            sem_min, sem_max = min(sem_scores), max(sem_scores)
            sem_range = sem_max - sem_min if sem_max != sem_min else 1.0
            
            for doc_id, score, source in semantic_results:
                norm_score = (score - sem_min) / sem_range
                doc_scores[doc_id] = {'semantic': norm_score, 'bm25': 0.0, 'source': source}
        
        # Normalize BM25 scores
        if bm25_results:
            bm25_scores = [s for _, s, _ in bm25_results]
            bm25_min, bm25_max = min(bm25_scores), max(bm25_scores)
            bm25_range = bm25_max - bm25_min if bm25_max != bm25_min else 1.0
            
            for doc_id, score, source in bm25_results:
                norm_score = (score - bm25_min) / bm25_range
                if doc_id in doc_scores:
                    doc_scores[doc_id]['bm25'] = norm_score
                else:
                    doc_scores[doc_id] = {'semantic': 0.0, 'bm25': norm_score, 'source': source}
        
        # Calculate weighted combined scores
        combined_results = []
        for doc_id, scores in doc_scores.items():
            combined_score = self.alpha * scores['semantic'] + (1 - self.alpha) * scores['bm25']
            combined_results.append((doc_id, combined_score, scores['source']))
        
        # Sort by combined score (descending)
        combined_results.sort(key=lambda x: x[1], reverse=True)
        
        return combined_results
    
    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun = None
    ) -> List[Document]:
        """
        Retrieve documents using hybrid search (BM25 + Semantic).
        
        Args:
            query: Search query
            
        Returns:
            List of relevant documents
        """
        # Perform both searches
        semantic_results = self._semantic_search(query)
        bm25_results = self._bm25_search(query)
        
        logger.debug(f"Semantic search returned {len(semantic_results)} results")
        logger.debug(f"BM25 search returned {len(bm25_results)} results")
        
        # Fuse results
        if self.use_rrf:
            fused_results = self._reciprocal_rank_fusion(semantic_results, bm25_results)
            logger.debug("Using Reciprocal Rank Fusion (RRF)")
        else:
            fused_results = self._weighted_score_fusion(semantic_results, bm25_results)
            logger.debug(f"Using Weighted Score Fusion (alpha={self.alpha})")
        
        # Convert to LangChain documents
        documents = []
        for doc_id, score, source in fused_results[:self.top_k_initial]:
            # Add fusion score to metadata
            metadata = source.get('metadata', {}).copy()
            metadata['fusion_score'] = score
            metadata['retrieval_method'] = 'hybrid'
            
            doc = Document(
                page_content=source.get('text', ''),
                metadata=metadata
            )
            documents.append(doc)
        
        logger.info(f"Hybrid retrieval returned {len(documents)} documents for query: {query[:50]}...")
        return documents
    
    async def _aget_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun = None
    ) -> List[Document]:
        """Async version - not implemented."""
        raise NotImplementedError("Async retrieval not supported")


class SemanticOnlyRetriever(BaseRetriever):
    """
    Fallback retriever using only semantic search.
    Compatible with the same interface as HybridRetriever.
    """
    
    es_client: Any
    index_name: str
    embedding_function: Any
    top_k: int = 5
    top_k_initial: int = 20
    vector_field: str = "vector_field"
    
    class Config:
        """Pydantic configuration."""
        arbitrary_types_allowed = True
    
    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun = None
    ) -> List[Document]:
        """Retrieve documents using semantic search only."""
        query_vector = self.embedding_function.embed_query(query)
        
        search_query = {
            "query": {
                "script_score": {
                    "query": {"match_all": {}},
                    "script": {
                        "source": f"cosineSimilarity(params.query_vector, '{self.vector_field}') + 1.0",
                        "params": {"query_vector": query_vector}
                    }
                }
            },
            "size": self.top_k_initial
        }
        
        try:
            response = self.es_client.search(index=self.index_name, body=search_query)
            documents = []
            for hit in response['hits']['hits']:
                source = hit['_source']
                metadata = source.get('metadata', {}).copy()
                metadata['semantic_score'] = hit['_score']
                metadata['retrieval_method'] = 'semantic'
                
                doc = Document(
                    page_content=source.get('text', ''),
                    metadata=metadata
                )
                documents.append(doc)
            
            logger.info(f"Semantic retrieval returned {len(documents)} documents")
            return documents
            
        except Exception as e:
            logger.error(f"Error in semantic retrieval: {e}")
            return []
    
    async def _aget_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun = None
    ) -> List[Document]:
        """Async version - not implemented."""
        raise NotImplementedError("Async retrieval not supported")
