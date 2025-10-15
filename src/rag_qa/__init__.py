"""
Data-Driven Document QA Using RAG and Vector Embeddings

A comprehensive Retrieval-Augmented Generation (RAG) system using:
- Hugging Face transformers for embeddings and LLM
- FAISS for small-scale vector storage (in-memory)
- Elasticsearch for production-scale vector storage (distributed)
- LangChain for RAG pipeline orchestration
- Support for PDF, DOCX, TXT, and Markdown documents
"""

__version__ = "1.0.0"
__author__ = "RAG QA Team"

from .core.rag_openai import RAGWithOpenAI
from .core.es7_retriever import ES7ScriptScoreRetriever
from .core.document_processor import DocumentProcessor

__all__ = [
    "RAGWithOpenAI",
    "ES7ScriptScoreRetriever",
    "DocumentProcessor"
] 