"""
RAG System with OpenAI Generation

Uses Elasticsearch/FAISS for retrieval and OpenAI API for generation.
Supports advanced RAG features:
- Hybrid Search (BM25 + Semantic)
- Cross-Encoder Reranking

Configuration loaded from config/config.yml.
"""

import logging
from typing import List, Dict, Any, Optional
import warnings
warnings.filterwarnings("ignore")

# LangChain
from langchain.schema import Document, HumanMessage
from langchain_openai import ChatOpenAI

# Local imports
from ..utils.config_loader import get_llm_config, get_retrieval_config
from .reranker import create_reranker, CrossEncoderReranker, DummyReranker

# Setup logging
logger = logging.getLogger(__name__)


class RAGWithOpenAI:
    """
    Advanced RAG system using OpenAI for generation.
    
    Features:
    - Hybrid Search support (BM25 + Semantic via HybridRetriever)
    - Cross-Encoder Reranking for improved precision
    - Token usage tracking with reasoning token breakdown
    """
    
    def __init__(self, retriever=None, enable_reranking: Optional[bool] = None):
        """
        Initialize RAG with OpenAI.
        
        Args:
            retriever: LangChain retriever object (from FAISS, Elasticsearch, or HybridRetriever)
            enable_reranking: Override config to enable/disable reranking. None uses config value.
        """
        self.retriever = retriever
        self.llm_config = get_llm_config()
        self.retrieval_config = get_retrieval_config()
        
        # Initialize OpenAI LLM
        self._initialize_openai()
        
        # Initialize Reranker
        self._initialize_reranker(enable_reranking)
        
        logger.info("RAG with OpenAI initialized successfully")
    
    def _initialize_openai(self):
        """Initialize OpenAI LLM with proxy configuration."""
        if not self.llm_config.get('use_openai', False):
            logger.warning("use_openai is False, OpenAI will not be initialized")
            self.llm = None
            return
        
        try:
            # Get API key and model from config
            api_key = self.llm_config.get('api_key', '').strip()
            model = self.llm_config.get('model', 'gpt-4o-mini')
            base_url = self.llm_config.get('base_url', None)
            params = self.llm_config.get('params', {})
            
            # Initialize ChatOpenAI with simple configuration (supports custom proxy)
            llm_kwargs = {
                'api_key': api_key,
                'model': model,
                'temperature': params.get('temperature', 0),
                'max_tokens': params.get('max_tokens', 1000)
            }
            
            # Add base_url if provided (for custom proxy)
            if base_url:
                llm_kwargs['base_url'] = base_url
                logger.info(f"Using custom base_url: {base_url}")
            
            self.llm = ChatOpenAI(**llm_kwargs)
            
            logger.info(f"OpenAI LLM initialized with model: {model}")
            
        except Exception as e:
            logger.error(f"Error initializing OpenAI: {e}")
            self.llm = None
    
    def _initialize_reranker(self, enable_reranking: Optional[bool] = None):
        """Initialize the cross-encoder reranker based on config."""
        rerank_config = self.retrieval_config.get('reranking', {})
        
        # Use override if provided, otherwise use config
        enabled = enable_reranking if enable_reranking is not None else rerank_config.get('enabled', False)
        
        if enabled:
            try:
                self.reranker = create_reranker(
                    enabled=True,
                    model_name=rerank_config.get('model', 'cross-encoder/ms-marco-MiniLM-L-6-v2'),
                    top_k=self.retrieval_config.get('top_k', 5),
                    device=rerank_config.get('device', 'cpu')
                )
                self.reranking_enabled = True
                logger.info(f"Cross-encoder reranking enabled with model: {rerank_config.get('model')}")
            except Exception as e:
                logger.warning(f"Failed to initialize reranker: {e}. Disabling reranking.")
                self.reranker = None
                self.reranking_enabled = False
        else:
            self.reranker = None
            self.reranking_enabled = False
            logger.info("Reranking disabled")
    
    def set_retriever(self, retriever):
        """Set the retriever (FAISS, Elasticsearch, or HybridRetriever)."""
        self.retriever = retriever
        logger.info("Retriever set successfully")
    
    def answer_question(self, question: str) -> Dict[str, Any]:
        """
        Answer a question using RAG with OpenAI.
        
        Args:
            question: User's question
            
        Returns:
            Dictionary with answer and source documents
        """
        if self.retriever is None:
            raise ValueError("Retriever not set. Call set_retriever() first.")
        
        if self.llm is None:
            raise ValueError("OpenAI LLM not initialized. Check configuration.")
        
        try:
            # Step 1: Retrieve relevant documents
            # If reranking is enabled, fetch more initially (top_k_initial)
            # Otherwise, just fetch top_k
            top_k = self.retrieval_config.get('top_k', 5)
            top_k_initial = self.retrieval_config.get('top_k_initial', 20) if self.reranking_enabled else top_k
            
            docs = self.retriever.get_relevant_documents(question)[:top_k_initial]
            
            if not docs:
                logger.warning(f"No relevant documents found for query")
                return {
                    "question": question,
                    "answer": "No relevant documents found.",
                    "source_documents": [],
                    "retrieval_info": {
                        "method": "hybrid" if hasattr(self.retriever, 'use_rrf') else "semantic",
                        "reranking_applied": False,
                        "initial_retrieved": 0,
                        "final_count": 0
                    }
                }
            
            logger.debug(f"Retrieved {len(docs)} documents from vector store")
            
            # Step 1.5: Apply reranking if enabled
            initial_count = len(docs)
            if self.reranking_enabled and self.reranker:
                docs = self.reranker.rerank(question, docs, return_scores=True)
                logger.info(f"Reranked {initial_count} docs to {len(docs)} (top_k={top_k})")
            else:
                # Just truncate to top_k if no reranking
                docs = docs[:top_k]
            
            # Step 2: Build context from retrieved documents
            context_parts = []
            for i, doc in enumerate(docs, 1):
                content = doc.page_content[:500]  # Limit chunk size
                source = doc.metadata.get('file_name', 'Unknown')
                context_parts.append(f"[Source {i}: {source}]\n{content}")
            
            context = "\n\n".join(context_parts)
            
            # Step 3: Create prompt for OpenAI
            prompt = f"""You are a helpful assistant that answers questions based on the provided context.

Context:
{context}

Question: {question}

Instructions:
- Answer the question based ONLY on the information in the context
- If the context doesn't contain enough information, say so
- Be concise and accurate
- Do NOT include source citations in your answer

***IMPORTANT***
- Strictly note that the output should limit to 50 words only.

Answer:"""
            
            # Step 4: Generate answer using OpenAI (LangChain way)
            logger.debug("Calling OpenAI API for answer generation")
            message = HumanMessage(content=prompt)
            response = self.llm.invoke([message])
            
            answer = response.content
            
            # Log token usage with detailed breakdown
            if hasattr(response, 'response_metadata') and 'token_usage' in response.response_metadata:
                token_usage = response.response_metadata['token_usage']
                completion_details = token_usage.get('completion_tokens_details', {})
                
                # Build detailed log message
                log_msg = f"Token Usage - Prompt: {token_usage.get('prompt_tokens', 0)}, " \
                         f"Completion: {token_usage.get('completion_tokens', 0)}"
                
                # Show reasoning tokens if present (for models like o1/gpt-5-nano)
                if completion_details:
                    reasoning = completion_details.get('reasoning_tokens', 0)
                    output = token_usage.get('completion_tokens', 0) - reasoning
                    if reasoning > 0:
                        log_msg += f" (Output: {output}, Reasoning: {reasoning})"
                
                log_msg += f", Total: {token_usage.get('total_tokens', 0)}"
                logger.info(log_msg)
            
            # Handle empty responses
            if not answer or answer.strip() == '':
                logger.warning(f"Empty response. Metadata: {response.response_metadata}")
                answer = "I apologize, but I received an empty response. Please try again."
            
            logger.debug(f"Generated answer ({len(answer)} chars)")
            
            # Extract token usage for response
            token_usage = None
            if hasattr(response, 'response_metadata') and 'token_usage' in response.response_metadata:
                token_usage = response.response_metadata['token_usage']
            
            # Step 5: Format response
            # Determine retrieval method from document metadata
            retrieval_method = "semantic"  # default
            if docs and docs[0].metadata.get('retrieval_method'):
                retrieval_method = docs[0].metadata.get('retrieval_method')
            
            result = {
                "question": question,
                "answer": answer,
                "source_documents": [
                    {
                        "content": doc.page_content[:300] + "..." if len(doc.page_content) > 300 else doc.page_content,
                        "source": doc.metadata.get("file_name", "Unknown"),
                        "type": doc.metadata.get("content_type", "file"),
                        "rerank_score": doc.metadata.get("rerank_score"),  # Include if available
                        "fusion_score": doc.metadata.get("fusion_score")   # Include if available
                    }
                    for doc in docs
                ],
                "num_retrieved": len(docs),
                "retrieval_info": {
                    "method": retrieval_method,
                    "reranking_applied": self.reranking_enabled,
                    "initial_retrieved": initial_count,
                    "final_count": len(docs)
                }
            }
            
            # Add token usage if available
            if token_usage:
                result["token_usage"] = {
                    "prompt_tokens": token_usage.get('prompt_tokens', 0),
                    "completion_tokens": token_usage.get('completion_tokens', 0),
                    "total_tokens": token_usage.get('total_tokens', 0)
                }
                
                # Add reasoning token breakdown if available (for o1/gpt-5-nano models)
                completion_details = token_usage.get('completion_tokens_details', {})
                if completion_details:
                    reasoning = completion_details.get('reasoning_tokens', 0)
                    if reasoning > 0:
                        result["token_usage"]["reasoning_tokens"] = reasoning
                        result["token_usage"]["output_tokens"] = token_usage.get('completion_tokens', 0) - reasoning
            
            return result
            
        except Exception as e:
            logger.error(f"Error processing question: {e}")
            return {
                "question": question,
                "answer": f"Error: {str(e)}",
                "source_documents": []
            }
    
    def batch_answer(self, questions: List[str]) -> List[Dict[str, Any]]:
        """
        Answer multiple questions.
        
        Args:
            questions: List of questions
            
        Returns:
            List of answer dictionaries
        """
        results = []
        for question in questions:
            result = self.answer_question(question)
            results.append(result)
        return results

