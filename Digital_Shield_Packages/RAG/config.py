"""
Configuration settings for the RAG module
"""

import os
from pathlib import Path

class RAGConfig:
    """Configuration class for RAG module settings"""
    
    # Base paths
    BASE_DIR = Path(__file__).parent.parent.parent
    DATA_DIR = BASE_DIR / "Digital_Shield_data" / "proccesed"
    EMBEDDINGS_DIR = BASE_DIR / "Digital_Shield_data" / "embeddings"
    
    # Data file
    CSV_FILE = DATA_DIR / "Cleaned_Digital_Shield_with_severity.csv"
    
    # Embedding settings
    EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
    EMBEDDING_DIMENSION = 384
    
    # Text chunking settings
    CHUNK_SIZE = 512
    CHUNK_OVERLAP = 50
    
    # Retrieval settings (dynamic based on query type)
    TOP_K = 500  # Default fallback (not used with dynamic retrieval)
    SIMILARITY_THRESHOLD = 0.7
    
    # Dynamic retrieval settings
    STATISTICAL_TOP_K = 100  # For statistical queries
    SPECIFIC_TOP_K = 20      # For specific queries  
    GENERAL_TOP_K = 50        # For general queries
    
    # Vector database settings
    VECTOR_DB_TYPE = "chromadb"  # Options: "chromadb", "faiss"
    COLLECTION_NAME = "digital_shield_cybersecurity"
    
    # LLM settings (for future use)
    LLM_MODEL = "gpt-3.5-turbo"
    MAX_TOKENS = 500
    TEMPERATURE = 0.7
    
    # Text columns to include in RAG chunks
    TEXT_COLUMNS = [
        "country",
        "attack type", 
        "target industry",
        "security vulnerability type",
        "defense mechanism used",
        "severity_kmeans"
    ]
    
    # Numeric columns to include
    NUMERIC_COLUMNS = [
        "year",
        "financial loss (in million $)",
        "number of affected users", 
        "incident resolution time (in hours)",
        "data breach in gb"
    ]
    
    @classmethod
    def ensure_directories(cls):
        """Ensure required directories exist"""
        cls.EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)
        return True
