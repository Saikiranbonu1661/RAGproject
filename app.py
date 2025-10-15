#!/usr/bin/env python3
"""
Streamlit Chat UI for RAG Document QA with OpenAI

A clean, modern chat interface for querying documents using OpenAI.
"""

import streamlit as st
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.rag_qa import RAGWithOpenAI, ES7ScriptScoreRetriever
from src.rag_qa.utils.config_loader import (
    get_elasticsearch_config, 
    get_embeddings_config, 
    get_llm_config, 
    get_retrieval_config
)
from langchain_community.embeddings import HuggingFaceEmbeddings
from elasticsearch import Elasticsearch


def init_system():
    """Initialize the RAG system with OpenAI."""
    if 'rag_system' in st.session_state:
        return st.session_state.rag_system
    
    try:
        # Load configurations
        es_config = get_elasticsearch_config()
        embeddings_config = get_embeddings_config()
        llm_config = get_llm_config()
        retrieval_config = get_retrieval_config()
        
        # Check OpenAI configuration
        if not llm_config.get('use_openai'):
            st.error("❌ OpenAI is not configured in config/config.yml")
            st.stop()
        
        # Connect to Elasticsearch
        with st.spinner("Connecting to Elasticsearch..."):
            es_client = Elasticsearch([es_config['es_url']])
            if not es_client.ping():
                st.error("❌ Cannot connect to Elasticsearch")
                st.info("💡 Start Elasticsearch: `docker-compose up -d`")
                st.stop()
        
        # Initialize embeddings
        with st.spinner("Loading embedding model..."):
            embedding_model = embeddings_config.get('sbert_model', 'BAAI/bge-base-en-v1.5')
            device = embeddings_config.get('device', 'cpu')
            
            if device == 'auto':
                import torch
                device = 'cuda' if torch.cuda.is_available() else 'cpu'
            
            embeddings = HuggingFaceEmbeddings(
                model_name=embedding_model,
                model_kwargs={'device': device}
            )
        
        # Create retriever
        retriever = ES7ScriptScoreRetriever(
            es_client=es_client,
            index_name=es_config['index_name'],
            embedding_function=embeddings,
            top_k=retrieval_config.get('top_k', 3)
        )
        
        # Create RAG system
        with st.spinner("Initializing OpenAI RAG system..."):
            rag = RAGWithOpenAI(retriever=retriever)
        
        st.session_state.rag_system = rag
        st.session_state.system_ready = True
        return rag
        
    except Exception as e:
        st.error(f"❌ Failed to initialize system: {e}")
        st.info("**Troubleshooting:**\n1. Start Elasticsearch: `docker-compose up -d`\n2. Check config/config.yml for valid OpenAI settings")
        st.stop()


def main():
    """Main Streamlit app."""
    st.set_page_config(
        page_title="RAG Chat with OpenAI",
        page_icon="💬",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS for better chat UI
    st.markdown("""
        <style>
        .stChatMessage {
            padding: 1rem;
            border-radius: 0.5rem;
        }
        .main > div {
            padding-top: 2rem;
        }
        </style>
    """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.title("⚙️ RAG Document QA")
        st.markdown("**Powered by OpenAI + Elasticsearch**")
        
        st.divider()
        
        # System status
        st.subheader("📊 System Status")
        
        if 'system_ready' not in st.session_state:
            st.info("System not initialized")
        else:
            st.success("✅ System Ready")
            
            # Show configuration
            try:
                retrieval_config = get_retrieval_config()
                es_config = get_elasticsearch_config()
                embeddings_config = get_embeddings_config()
                
                st.metric("Top K Results", retrieval_config.get('top_k', 3))
                st.metric("Index", es_config.get('index_name', 'N/A'))
                st.caption(f"Model: {embeddings_config.get('sbert_model', 'N/A')}")
            except:
                pass
        
        st.divider()
        
        # Actions
        st.subheader("🛠️ Actions")
        
        if st.button("🔄 Reinitialize System"):
            if 'rag_system' in st.session_state:
                del st.session_state.rag_system
            if 'system_ready' in st.session_state:
                del st.session_state.system_ready
            st.rerun()
        
        if st.button("🗑️ Clear Chat History"):
            st.session_state.messages = []
            st.rerun()
        
        st.divider()
        
        # Information
        with st.expander("ℹ️ About"):
            st.markdown("""
            **RAG Document QA System**
            
            This system uses:
            - **OpenAI** for generation
            - **Elasticsearch** for vector storage
            - **BGE Embeddings** for semantic search
            
            **Terminal Interface:**
            ```bash
            python main.py
            ```
            
            Documents are already indexed in Elasticsearch.
            """)
    
    # Main chat interface
    st.title("💬 Chat with Your Documents")
    st.caption("Ask questions about your documents and get AI-powered answers")
    
    # Initialize system
    rag = init_system()
    
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
            # Display token usage for assistant messages
            if message["role"] == "assistant" and "token_usage" in message and message["token_usage"]:
                usage = message["token_usage"]
                msg = f"🔢 Tokens: {usage['prompt_tokens']} prompt + {usage['completion_tokens']} completion"
                if 'reasoning_tokens' in usage:
                    msg += f" (output: {usage['output_tokens']}, reasoning: {usage['reasoning_tokens']})"
                msg += f" = {usage['total_tokens']} total"
                st.caption(msg)
            
            # Display sources for assistant messages
            if message["role"] == "assistant" and "sources" in message:
                with st.expander("📚 Sources"):
                    for i, source in enumerate(message["sources"], 1):
                        st.markdown(f"**Source {i}: {source['source']}**")
                        st.caption(source['content'])
                        st.divider()
    
    # Chat input
    if prompt := st.chat_input("Ask a question about your documents..."):
        # Add user message to chat
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Get assistant response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    result = rag.answer_question(prompt)
                    answer = result['answer']
                    sources = result['source_documents']
                    token_usage = result.get('token_usage', None)
                    
                    # Display answer
                    st.markdown(answer)
                    
                    # Display token usage
                    if token_usage:
                        msg = f"🔢 Tokens: {token_usage['prompt_tokens']} prompt + {token_usage['completion_tokens']} completion"
                        if 'reasoning_tokens' in token_usage:
                            msg += f" (output: {token_usage['output_tokens']}, reasoning: {token_usage['reasoning_tokens']})"
                        msg += f" = {token_usage['total_tokens']} total"
                        st.caption(msg)
                    
                    # Display sources
                    if sources:
                        with st.expander("📚 Sources"):
                            for i, source in enumerate(sources, 1):
                                st.markdown(f"**Source {i}: {source['source']}**")
                                st.caption(source['content'])
                                st.divider()
                    
                    # Add to chat history
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                        "sources": sources,
                        "token_usage": token_usage
                    })
                    
                except Exception as e:
                    error_msg = f"❌ Error: {str(e)}"
                    st.error(error_msg)
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": error_msg
                    })


if __name__ == "__main__":
    main()

