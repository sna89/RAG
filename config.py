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

    # RAG system configuration
    rag_config = {
        # "embedding_model": "sentence-transformers/all-MiniLM-l6-v2",
        # embedding_model model: "intfloat/e5-small-v2",
        "embedding_model": "text-embedding-ada-002",
        "top_k": 10,
        "persist_path": "persist.db",
        "chunk_size": 1000,
        "chunk_overlap": 200
    }

    # LLM configuration
    llm_config = {
        "model_name": "gpt-4o-mini",
        "temperature": 0
    }

    # QA system configuration
    qa_config = {
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
        "llm_config": llm_config,
        "qa_config": qa_config,
        "data_path": os.path.join(".", "rag", "data_files"),
        "eval_config": eval_config
    }

    return config