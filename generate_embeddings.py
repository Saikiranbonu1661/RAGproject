"""
Advanced PDF to Elasticsearch Ingestion with Smart Chunking

Features:
- UnstructuredFileIOLoader for robust PDF parsing
- Regex-based text cleaning (removes artifacts)
- Tiktoken-based chunking (accurate token counting)
- Metadata tracking
- 768-dim BGE embeddings
- Elasticsearch vector storage

Usage:
    python generate_embeddings.py
"""

import os
import sys

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_elasticsearch import ElasticsearchStore
from elasticsearch import Elasticsearch, exceptions

# Import advanced chunking components
from src.rag_qa.core.document_processor import FileIoDataLoader
from src.rag_qa.utils.config_loader import (
    get_text_splitter_config,
    get_embeddings_config,
    get_elasticsearch_config
)

# --- CONFIGURATION FROM config/config.yml ---
# Load configuration
es_config = get_elasticsearch_config()
embeddings_config = get_embeddings_config()
text_splitter_config = get_text_splitter_config()

ES_URL = es_config['es_url']
INDEX_NAME = es_config['index_name']
PDF_PATH = "/home/saikiranbonu/Downloads/Aboutau.pdf"  # Override if needed

# BGE Model - BAAI General Embedding (768 dimensions, state-of-the-art)
EMBEDDING_MODEL_NAME = "BAAI/bge-base-en-v1.5"
EMBEDDING_DIMENSIONS = 768

# Text Splitter Configuration (from config)
CHUNK_SIZE = text_splitter_config['chunk_size']
CHUNK_OVERLAP = text_splitter_config['chunk_overlap']
SEPARATORS = text_splitter_config['separators']

def check_elastic_connection(es_url):
    """Verifies connection to Elasticsearch and raises an error if unavailable."""
    try:
        client = Elasticsearch([es_url])
        if client.ping():
            print(f"✅ Successfully connected to Elasticsearch at {es_url}")
            return client
        else:
            raise ConnectionError("Elasticsearch ping failed. It may be running but not healthy.")
    except exceptions.ConnectionError as e:
        print(f"❌ Failed to connect to Elasticsearch at {es_url}. Connection refused.")
        print("   Ensure Elasticsearch container is running: docker-compose up -d")
        raise e

def ingest_pdf_to_elasticsearch(pdf_file_path):
    """
    Orchestrates the RAG ingestion pipeline with ADVANCED chunking:
    - UnstructuredFileIOLoader for loading
    - Regex-based text cleaning  
    - Tiktoken-based chunking
    - Metadata tracking
    - Embed (768-dim BGE)
    - Store in Elasticsearch
    """
    
    # 1. Check Connection Status
    es_client = check_elastic_connection(ES_URL)
    if not es_client:
        return

    # 2. Load and Process PDF with Advanced Chunking
    print(f"\n2. Loading document from: {pdf_file_path}")
    if not os.path.exists(pdf_file_path):
        raise FileNotFoundError(f"PDF not found at {pdf_file_path}")
    
    # Initialize FileIoDataLoader with config settings
    print(f"3. Processing with advanced chunking (Tiktoken-based)...")
    print(f"   Config: {CHUNK_SIZE} tokens, {CHUNK_OVERLAP} overlap")
    file_loader = FileIoDataLoader(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=SEPARATORS
    )
    
    # Load and chunk document with text cleaning
    with open(pdf_file_path, 'rb') as file:
        filename = os.path.basename(pdf_file_path)
        
        chunks = file_loader.scrap_and_create_documents_for_file_data(
            bytes_data=file,
            file_path=pdf_file_path,
            file_name=filename,
        )
    
    print(f"✅ Created {len(chunks)} clean chunks with metadata")

    # 4. Define Embeddings (768-dim BGE) with device detection
    device = embeddings_config.get('device', 'cpu')
    if device == 'auto':
        import torch
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"4. Initializing BGE model: {EMBEDDING_MODEL_NAME} ({EMBEDDING_DIMENSIONS} dimensions) on {device}")
    bge_embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={'device': device}
    )
    
    # 5. Store to Elasticsearch
    print(f"5. Indexing {len(chunks)} chunks into '{INDEX_NAME}'...")
    
    # ElasticsearchStore handles the embedding generation and bulk insertion in one go.
    # It automatically uses the 'vector_field' and 'text' fields defined in the index mapping.
    vectorstore = ElasticsearchStore.from_documents(
        chunks,
        bge_embeddings,
        es_url=ES_URL,
        index_name=INDEX_NAME,
        # IMPORTANT: Match the field name used in the curl command!
        vector_query_field="vector_field", 
        distance_strategy="COSINE",
        # Using the existing client avoids creating a new connection instance
        es_connection=es_client
    )

    print("-" * 60)
    print("✅ Ingestion complete with ADVANCED CHUNKING!")
    print(f"📊 Statistics:")
    print(f"   • Total chunks: {len(chunks)}")
    print(f"   • Chunk size: {CHUNK_SIZE} tokens (tiktoken-based)")
    print(f"   • Chunk overlap: {CHUNK_OVERLAP} tokens")
    print(f"   • Embedding model: {EMBEDDING_MODEL_NAME}")
    print(f"   • Vector dimensions: {EMBEDDING_DIMENSIONS}")
    print(f"   • Index name: {INDEX_NAME}")
    print(f"\n💡 Each chunk includes:")
    print(f"   ✓ Cleaned text (regex-based)")
    print(f"   ✓ Metadata (content_type, file_name, timestamp, source)")
    print(f"   ✓ Source tracking")
    print("-" * 60)
    
    # Show sample metadata from first chunk
    if chunks:
        print(f"\n📝 Sample chunk metadata:")
        sample_metadata = chunks[0].metadata
        for key, value in sample_metadata.items():
            print(f"   {key}: {value}")
        print()

if __name__ == "__main__":
    try:
        ingest_pdf_to_elasticsearch(PDF_PATH)
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
    except Exception as e:
        # Catches connection or index errors
        print(f"A fatal error occurred during the process: {e}")