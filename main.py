#!/usr/bin/env python3
"""
Main Entry Point for RAG Document QA System with OpenAI

This script provides an interactive command-line interface for the RAG system.
Uses OpenAI for generation and Elasticsearch for vector storage (ES7 compatible).
"""

import argparse
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.rag_qa.utils.helpers import setup_logging
from src.rag_qa import RAGWithOpenAI, ES7ScriptScoreRetriever
from src.rag_qa.utils.config_loader import get_elasticsearch_config, get_embeddings_config, get_llm_config, get_retrieval_config
from langchain_community.embeddings import HuggingFaceEmbeddings
from elasticsearch import Elasticsearch

def run_interactive():
    """Run interactive command-line interface."""
    
    print("🤖 RAG Document QA with OpenAI")
    print("=" * 40)
    
    # Initialize system
    print("\nInitializing system...")
    
    try:
        # Load configurations
        es_config = get_elasticsearch_config()
        embeddings_config = get_embeddings_config()
        llm_config = get_llm_config()
        retrieval_config = get_retrieval_config()
        
        # Check if OpenAI is configured
        if not llm_config.get('use_openai'):
            print("❌ OpenAI is not configured in config/config.yml")
            print("💡 Set 'use_openai: true' and configure the openai_proxy section")
            return
        
        # Connect to Elasticsearch
        print(f"Connecting to Elasticsearch at {es_config['es_url']}...")
        es_client = Elasticsearch([es_config['es_url']])
        if not es_client.ping():
            raise ConnectionError("Elasticsearch ping failed")
        print("✅ Connected to Elasticsearch")
        
        # Initialize embeddings
        embedding_model = embeddings_config.get('sbert_model', 'BAAI/bge-base-en-v1.5')
        device = embeddings_config.get('device', 'cpu')
        
        # Auto-detect GPU if device is 'auto'
        if device == 'auto':
            import torch
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        print(f"Loading embedding model: {embedding_model} on {device}...")
        embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model,
            model_kwargs={'device': device}
        )
        
        # Create ES7 retriever
        retriever = ES7ScriptScoreRetriever(
            es_client=es_client,
            index_name=es_config['index_name'],
            embedding_function=embeddings,
            top_k=retrieval_config.get('top_k', 3)
        )
        
        # Create RAG system with OpenAI
        print("Initializing OpenAI RAG system...")
        rag = RAGWithOpenAI(retriever=retriever)
        print("✅ System initialized with OpenAI + Elasticsearch (ES7 compatible)")
        
    except Exception as e:
        print(f"❌ Failed to initialize system: {e}")
        print("\n💡 Troubleshooting:")
        print("   1. Start Elasticsearch: docker-compose up -d")
        print("   2. Check config/config.yml for valid OpenAI proxy settings")
        return
    
    # Documents are already ingested in Elasticsearch
    print("\n📄 Ready to answer questions from your documents")
    
    # Interactive Q&A loop
    print("\n🔍 Ask questions about your documents (type 'quit' to exit):")
    
    while True:
        question = input("\nQuestion: ").strip()
        
        if question.lower() in ['quit', 'exit', 'q']:
            break
        
        if not question:
            continue
        
        try:
            result = rag.answer_question(question)
            print(f"\n💬 Answer: {result['answer']}")
            print(f"\n📚 Sources: {', '.join([doc['source'] for doc in result['source_documents']])}")
            
            # Display token usage if available
            if 'token_usage' in result:
                usage = result['token_usage']
                msg = f"\n🔢 Token Usage: Prompt={usage['prompt_tokens']}, Completion={usage['completion_tokens']}"
                
                # Show reasoning token breakdown if available
                if 'reasoning_tokens' in usage:
                    msg += f" (Output={usage['output_tokens']}, Reasoning={usage['reasoning_tokens']})"
                
                msg += f", Total={usage['total_tokens']}"
                print(msg)
        except Exception as e:
            print(f"❌ Error: {e}")
    
    print("\n👋 Goodbye!")

def main():
    """Main function with argument parsing."""
    parser = argparse.ArgumentParser(
        description="RAG Document QA System - Interactive Mode",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
            Example:
                python main.py              # Run interactive mode
                python main.py --help       # Show this help message
        """
    )
    
    parser.add_argument(
        '--log-level',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        default='INFO',
        help='Set logging level'
    )
    
    parser.add_argument(
        '--log-file',
        help='Optional log file path'
    )
    
    args = parser.parse_args()
    
    # Setup enhanced logging with filename and line numbers
    setup_logging(args.log_level, args.log_file)
    
    # Log the startup
    import logging
    logger = logging.getLogger(__name__)
    logger.info("Starting RAG Document QA System in interactive mode")
    
    # Run interactive mode
    run_interactive()

if __name__ == "__main__":
    main() 