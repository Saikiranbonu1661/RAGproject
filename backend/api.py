"""
FastAPI Backend for RAG Document QA System
Supports dynamic document upload and querying
"""

from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import os
import sys
import uuid
import tempfile
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.rag_qa import RAGWithOpenAI, ES7ScriptScoreRetriever
from src.rag_qa.core.document_processor import FileIoDataLoader
from src.rag_qa.utils.config_loader import (
    get_elasticsearch_config,
    get_embeddings_config,
    get_text_splitter_config,
    get_retrieval_config
)
from langchain_community.embeddings import HuggingFaceEmbeddings
from elasticsearch import Elasticsearch

# Initialize FastAPI
app = FastAPI(
    title="RAG Document QA API",
    description="Dynamic document upload and intelligent Q&A system",
    version="1.0.0"
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # React dev server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global state (in production, use Redis or database)
sessions = {}  # session_id -> {documents, retriever, rag}


# Request/Response Models
class QueryRequest(BaseModel):
    question: str
    session_id: str


class QueryResponse(BaseModel):
    answer: str
    sources: List[dict]
    token_usage: Optional[dict] = None


class DocumentInfo(BaseModel):
    id: str
    filename: str
    size: int
    chunks: int
    uploaded_at: str


class SessionResponse(BaseModel):
    session_id: str
    documents: List[DocumentInfo]
    total_chunks: int


# Helper Functions
def get_or_create_session(session_id: str) -> dict:
    """Get or create a session."""
    if session_id not in sessions:
        sessions[session_id] = {
            "documents": [],
            "chunks": [],
            "retriever": None,
            "rag": None,
            "embeddings": None,
            "es_client": None
        }
    return sessions[session_id]


def initialize_embeddings():
    """Initialize embeddings model (shared across sessions)."""
    embeddings_config = get_embeddings_config()
    device = embeddings_config.get('device', 'cpu')
    
    if device == 'auto':
        import torch
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    return HuggingFaceEmbeddings(
        model_name=embeddings_config.get('sbert_model', 'BAAI/bge-base-en-v1.5'),
        model_kwargs={'device': device}
    )


def initialize_elasticsearch():
    """Initialize Elasticsearch client."""
    es_config = get_elasticsearch_config()
    es_client = Elasticsearch([es_config['es_url']])
    
    if not es_client.ping():
        raise ConnectionError("Cannot connect to Elasticsearch")
    
    return es_client, es_config['index_name']


# API Endpoints

@app.get("/")
async def root():
    """Health check endpoint."""
    return {
        "status": "online",
        "service": "RAG Document QA API",
        "version": "1.0.0"
    }


@app.post("/api/session/create")
async def create_session():
    """Create a new session."""
    session_id = str(uuid.uuid4())
    get_or_create_session(session_id)
    
    return {
        "session_id": session_id,
        "message": "Session created successfully"
    }


@app.post("/api/documents/upload")
async def upload_document(
    file: UploadFile = File(...),
    session_id: str = None
):
    """Upload and process a document."""
    
    # Create session if not provided
    if not session_id:
        session_id = str(uuid.uuid4())
    
    session = get_or_create_session(session_id)
    
    # Validate file type
    allowed_extensions = ['.pdf', '.txt', '.docx', '.md']
    file_ext = os.path.splitext(file.filename)[1].lower()
    
    if file_ext not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type. Allowed: {', '.join(allowed_extensions)}"
        )
    
    try:
        # Save file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp_file:
            content = await file.read()
            tmp_file.write(content)
            tmp_file_path = tmp_file.name
        
        # Process document
        text_splitter_config = get_text_splitter_config()
        file_loader = FileIoDataLoader(
            chunk_size=text_splitter_config['chunk_size'],
            chunk_overlap=text_splitter_config['chunk_overlap'],
            separators=text_splitter_config['separators']
        )
        
        with open(tmp_file_path, 'rb') as f:
            chunks = file_loader.scrap_and_create_documents_for_file_data(
                bytes_data=f,
                file_path=tmp_file_path,
                file_name=file.filename
            )
        
        # Clean up temp file
        os.unlink(tmp_file_path)
        
        # Store document info
        doc_id = str(uuid.uuid4())
        doc_info = {
            "id": doc_id,
            "filename": file.filename,
            "size": len(content),
            "chunks": len(chunks),
            "uploaded_at": datetime.now().isoformat()
        }
        
        session["documents"].append(doc_info)
        session["chunks"].extend(chunks)
        
        # Initialize or update retriever
        if session["embeddings"] is None:
            session["embeddings"] = initialize_embeddings()
        
        if session["es_client"] is None:
            session["es_client"], index_name = initialize_elasticsearch()
        
        # Create session-specific index
        session_index = f"rag_session_{session_id}"
        
        # Index documents to Elasticsearch
        from langchain_elasticsearch import ElasticsearchStore
        
        vectorstore = ElasticsearchStore.from_documents(
            session["chunks"],
            session["embeddings"],
            es_url=get_elasticsearch_config()['es_url'],
            index_name=session_index,
            vector_query_field="vector_field",
            distance_strategy="COSINE",
            es_connection=session["es_client"]
        )
        
        # Create retriever
        retrieval_config = get_retrieval_config()
        session["retriever"] = ES7ScriptScoreRetriever(
            es_client=session["es_client"],
            index_name=session_index,
            embedding_function=session["embeddings"],
            top_k=retrieval_config.get('top_k', 3)
        )
        
        # Create RAG system
        session["rag"] = RAGWithOpenAI(retriever=session["retriever"])
        
        return {
            "session_id": session_id,
            "document": doc_info,
            "message": f"Successfully processed {file.filename}",
            "total_documents": len(session["documents"]),
            "total_chunks": len(session["chunks"])
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing document: {str(e)}")


@app.post("/api/query", response_model=QueryResponse)
async def query_documents(request: QueryRequest):
    """Query the uploaded documents."""
    
    if request.session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    session = sessions[request.session_id]
    
    if not session["rag"]:
        raise HTTPException(status_code=400, detail="No documents uploaded yet")
    
    try:
        result = session["rag"].answer_question(request.question)
        
        return QueryResponse(
            answer=result["answer"],
            sources=result["source_documents"],
            token_usage=result.get("token_usage")
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")


@app.get("/api/session/{session_id}", response_model=SessionResponse)
async def get_session_info(session_id: str):
    """Get session information."""
    
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    session = sessions[session_id]
    
    return SessionResponse(
        session_id=session_id,
        documents=session["documents"],
        total_chunks=len(session["chunks"])
    )


@app.delete("/api/session/{session_id}")
async def delete_session(session_id: str):
    """Delete a session and cleanup."""
    
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    session = sessions[session_id]
    
    # Cleanup Elasticsearch index
    try:
        if session["es_client"]:
            session_index = f"rag_session_{session_id}"
            if session["es_client"].indices.exists(index=session_index):
                session["es_client"].indices.delete(index=session_index)
    except Exception as e:
        print(f"Error cleaning up ES index: {e}")
    
    # Remove session
    del sessions[session_id]
    
    return {"message": "Session deleted successfully"}


@app.delete("/api/session/{session_id}/documents/{document_id}")
async def delete_document(session_id: str, document_id: str):
    """Delete a specific document from session."""
    
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    session = sessions[session_id]
    
    # Find and remove document
    doc_to_remove = None
    for doc in session["documents"]:
        if doc["id"] == document_id:
            doc_to_remove = doc
            break
    
    if not doc_to_remove:
        raise HTTPException(status_code=404, detail="Document not found")
    
    session["documents"].remove(doc_to_remove)
    
    # Note: In production, you'd need to re-index without this document
    # For now, we'll just remove it from the list
    
    return {"message": f"Document {doc_to_remove['filename']} removed"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

