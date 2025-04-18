"""
Configuration module for the QA application.
Handles loading environment variables and providing application settings.
"""

import os
from typing import Dict, Any
from dotenv import load_dotenv


def load_config() -> Dict[str, Any]:
    """
    Load application configuration from environment variables and defaults.

    Returns:
        Dictionary containing all configuration settings required by the application
    """
    # Load environment variables from .env file if present
    load_dotenv()

    # Environment configuration for Azure
    env_config = {
        "openai_api_key": os.getenv("OPENAI_API_KEY", ""),
    }

    core_config = {
        "embedding_model": {
            "provider": "openai",
            "model_name": "text-embedding-ada-002"
        },
        # "huggingface": "sentence-transformers/all-MiniLM-l6-v2"

        "llm_config": {
            "model_name": "gpt-4o-mini",
            "temperature": 0
        },
        "vector_db": {
            "db_type": "chroma",
            "persist_path": "persist.db",
        }
    }

    # RAG system configuration
    rag_config = {
        "top_k": 10,
        "chunk_size": 1000,
        "chunk_overlap": 200
    }

    # QA system configuration
    qa_config = {
        "fix_grammar": False,
        "ambiguous": False,
        "query_translation_type": "base",  # base, multi_query, #rag_fusion
        "filter_by_topic": False,
        "ambiguous_threshold": 0.8
    }

    eval_config = {
        "data_path": "./eval_qa.json",
        "output_path": "./eval_qa.csv"
    }

    # Complete configuration dictionary
    config = {
        "env_config": env_config,
        "rag_config": rag_config,
        "core_config": core_config,
        "qa_config": qa_config,
        "data_path": os.path.join(".", "rag", "data_files"),
        "eval_config": eval_config
    }

    return config
