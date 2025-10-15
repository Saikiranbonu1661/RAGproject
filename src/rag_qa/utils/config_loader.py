"""
Configuration Loader

Centralized configuration management for the RAG system.
Loads settings from config/config.yml.
"""

import os
import yaml
from typing import Any, Dict, Optional
from pathlib import Path


class ConfigLoader:
    """Singleton configuration loader."""
    
    _instance = None
    _config = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ConfigLoader, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        """Initialize the configuration loader."""
        if self._config is None:
            self._load_config()
    
    def _load_config(self):
        """Load configuration from YAML file."""
        # Find config file relative to project root
        current_dir = Path(__file__).resolve()
        project_root = current_dir.parent.parent.parent.parent
        config_path = project_root / "config" / "config.yml"
        
        if not config_path.exists():
            raise FileNotFoundError(
                f"Configuration file not found at {config_path}. "
                "Please create config/config.yml"
            )
        
        with open(config_path, 'r', encoding='utf-8') as f:
            self._config = yaml.safe_load(f)
    
    def get(self, key_path: str, default: Any = None) -> Any:
        """
        Get configuration value using dot notation.
        
        Args:
            key_path: Path to config value (e.g., 'text_splitter.chunk_size')
            default: Default value if key not found
            
        Returns:
            Configuration value
            
        Example:
            config = ConfigLoader()
            chunk_size = config.get('text_splitter.chunk_size')
            es_url = config.get('vector_store.elasticsearch.es_url')
        """
        keys = key_path.split('.')
        value = self._config
        
        try:
            for key in keys:
                value = value[key]
            return value
        except (KeyError, TypeError):
            return default
    
    def get_text_splitter_config(self) -> Dict[str, Any]:
        """Get text splitter configuration."""
        return {
            'chunk_size': self.get('text_splitter.chunk_size', 256),
            'chunk_overlap': self.get('text_splitter.chunk_overlap', 50),
            'separators': self.get('text_splitter.separators', ["\n\n", "\n", " ", ""])
        }
    
    def get_embeddings_config(self) -> Dict[str, Any]:
        """Get embeddings configuration."""
        return {
            'default_model': self.get('embeddings.default_model', 
                                     'sentence-transformers/all-MiniLM-L6-v2'),
            'sbert_model': self.get('embeddings.sbert_model',
                                   'sentence-transformers/distiluse-base-multilingual-cased'),
            'device': self.get('embeddings.device', 'auto')
        }
    
    def get_llm_config(self) -> Dict[str, Any]:
        """Get LLM configuration."""
        return {
            'use_openai': self.get('llm.use_openai', True),
            'api_key': self.get('llm.api_key', ''),
            'model': self.get('llm.model', 'gpt-4o-mini'),
            'base_url': self.get('llm.base_url', None),
            'params': self.get('llm.params', {})
        }
    
    def get_retrieval_config(self) -> Dict[str, Any]:
        """Get retrieval configuration."""
        return {
            'top_k': self.get('retrieval.top_k', 5),
            'search_type': self.get('retrieval.search_type', 'similarity')
        }
    
    def get_elasticsearch_config(self) -> Dict[str, Any]:
        """Get Elasticsearch configuration."""
        return {
            'es_url': self.get('vector_store.elasticsearch.es_url', 'http://localhost:9200'),
            'index_name': self.get('vector_store.elasticsearch.index_name', 'rag_documents'),
            'batch_size': self.get('vector_store.elasticsearch.batch_size', 100)
        }
    
    def get_faiss_config(self) -> Dict[str, Any]:
        """Get FAISS configuration."""
        return {
            'index_path': self.get('vector_store.faiss.index_path', 'data/faiss_index'),
            'save_index': self.get('vector_store.faiss.save_index', True)
        }
    
    def get_metadata_config(self) -> Dict[str, Any]:
        """Get metadata configuration (simplified)."""
        return {
            'content_type': 'file',
            'timestamp_enabled': True
        }
    
    def get_all(self) -> Dict[str, Any]:
        """Get all configuration."""
        return self._config.copy()


# Global config instance
config = ConfigLoader()


# Convenience functions
def get_config(key_path: str, default: Any = None) -> Any:
    """Get configuration value."""
    return config.get(key_path, default)


def get_text_splitter_config() -> Dict[str, Any]:
    """Get text splitter configuration."""
    return config.get_text_splitter_config()


def get_embeddings_config() -> Dict[str, Any]:
    """Get embeddings configuration."""
    return config.get_embeddings_config()


def get_llm_config() -> Dict[str, Any]:
    """Get LLM configuration."""
    return config.get_llm_config()


def get_retrieval_config() -> Dict[str, Any]:
    """Get retrieval configuration."""
    return config.get_retrieval_config()


def get_elasticsearch_config() -> Dict[str, Any]:
    """Get Elasticsearch configuration."""
    return config.get_elasticsearch_config()


def get_faiss_config() -> Dict[str, Any]:
    """Get FAISS configuration."""
    return config.get_faiss_config()


def get_metadata_config() -> Dict[str, Any]:
    """Get metadata configuration."""
    return config.get_metadata_config()

