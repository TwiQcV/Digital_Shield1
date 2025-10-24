"""
Embedding store for vector database operations
Handles creation, storage, and retrieval of embeddings
"""

import numpy as np
import pickle
import json
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import logging

try:
    import chromadb
    from chromadb.config import Settings
    CHROMADB_AVAILABLE = True
except ImportError:
    CHROMADB_AVAILABLE = False
    logging.warning("ChromaDB not available. Install with: pip install chromadb")

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    logging.warning("SentenceTransformers not available. Install with: pip install sentence-transformers")

from .config import RAGConfig

logger = logging.getLogger(__name__)

class EmbeddingStore:
    """Handles embedding creation, storage, and retrieval"""
    
    def __init__(self, config: RAGConfig = None):
        self.config = config or RAGConfig()
        self.config.ensure_directories()
        
        self.model = None
        self.client = None
        self.collection = None
        self.embeddings_cache = {}
        
    def initialize_model(self):
        """Initialize the embedding model"""
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            raise ImportError("SentenceTransformers not available. Install with: pip install sentence-transformers")
            
        try:
            self.model = SentenceTransformer(self.config.EMBEDDING_MODEL)
            logger.info(f"Initialized embedding model: {self.config.EMBEDDING_MODEL}")
        except Exception as e:
            logger.error(f"Error initializing embedding model: {e}")
            raise
    
    def initialize_vector_db(self):
        """Initialize the vector database"""
        if not CHROMADB_AVAILABLE:
            raise ImportError("ChromaDB not available. Install with: pip install chromadb")
            
        try:
            # Initialize ChromaDB client
            self.client = chromadb.PersistentClient(
                path=str(self.config.EMBEDDINGS_DIR / "chroma_db"),
                settings=Settings(anonymized_telemetry=False)
            )
            
            # Get or create collection
            try:
                self.collection = self.client.get_collection(self.config.COLLECTION_NAME)
                logger.info(f"Loaded existing collection: {self.config.COLLECTION_NAME}")
            except:
                self.collection = self.client.create_collection(
                    name=self.config.COLLECTION_NAME,
                    metadata={"description": "Digital Shield Cybersecurity Data"}
                )
                logger.info(f"Created new collection: {self.config.COLLECTION_NAME}")
                
        except Exception as e:
            logger.error(f"Error initializing vector database: {e}")
            raise
    
    def create_embeddings(self, texts: List[str]) -> np.ndarray:
        """Create embeddings for a list of texts"""
        if self.model is None:
            self.initialize_model()
            
        try:
            # Suppress progress bar for cleaner output
            embeddings = self.model.encode(texts, show_progress_bar=False)
            return embeddings
        except Exception as e:
            logger.error(f"Error creating embeddings: {e}")
            raise
    
    def store_embeddings(self, texts: List[str], embeddings: np.ndarray, 
                        metadata: List[Dict[str, Any]] = None) -> bool:
        """Store embeddings in the vector database"""
        if self.collection is None:
            self.initialize_vector_db()
            
        try:
            # ChromaDB batch size limit (conservative estimate)
            BATCH_SIZE = 5000
            
            # Prepare data for ChromaDB
            total_docs = len(texts)
            embeddings_list = embeddings.tolist()
            
            # Prepare metadata
            if metadata is None:
                metadata = [{"chunk_id": i} for i in range(len(texts))]
            
            # Process in batches
            for i in range(0, total_docs, BATCH_SIZE):
                end_idx = min(i + BATCH_SIZE, total_docs)
                
                # Prepare batch data
                batch_texts = texts[i:end_idx]
                batch_embeddings = embeddings_list[i:end_idx]
                batch_metadata = metadata[i:end_idx]
                batch_ids = [f"chunk_{j}" for j in range(i, end_idx)]
                
                # Add batch to collection
                self.collection.add(
                    documents=batch_texts,
                    embeddings=batch_embeddings,
                    metadatas=batch_metadata,
                    ids=batch_ids
                )
                
                logger.info(f"Stored batch {i//BATCH_SIZE + 1}/{(total_docs-1)//BATCH_SIZE + 1}: {len(batch_texts)} documents")
            
            logger.info(f"Successfully stored {total_docs} embeddings in vector database")
            return True
            
        except Exception as e:
            logger.error(f"Error storing embeddings: {e}")
            return False
    
    def search_similar(self, query: str, top_k: int = None) -> List[Dict[str, Any]]:
        """Search for similar embeddings"""
        if self.collection is None:
            self.initialize_vector_db()
            
        if self.model is None:
            self.initialize_model()
            
        top_k = top_k or self.config.TOP_K
        
        try:
            # Create query embedding
            query_embedding = self.model.encode([query])
            
            # Search in collection
            results = self.collection.query(
                query_embeddings=query_embedding.tolist(),
                n_results=top_k
            )
            
            # Format results
            formatted_results = []
            for i in range(len(results['documents'][0])):
                distance = results['distances'][0][i]
                # Convert distance to similarity (ensure it's between 0 and 1)
                similarity = max(0, 1 - distance)
                
                result = {
                    'text': results['documents'][0][i],
                    'metadata': results['metadatas'][0][i],
                    'distance': distance,
                    'similarity': similarity
                }
                formatted_results.append(result)
            
            logger.info(f"Found {len(formatted_results)} similar documents")
            return formatted_results
            
        except Exception as e:
            logger.error(f"Error searching embeddings: {e}")
            return []
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about the collection"""
        if self.collection is None:
            return {"error": "Collection not initialized"}
            
        try:
            count = self.collection.count()
            return {
                "total_documents": count,
                "collection_name": self.config.COLLECTION_NAME,
                "embedding_dimension": self.config.EMBEDDING_DIMENSION
            }
        except Exception as e:
            logger.error(f"Error getting collection stats: {e}")
            return {"error": str(e)}
    
    def clear_collection(self) -> bool:
        """Clear all data from the collection"""
        if self.collection is None:
            return False
            
        try:
            # Delete the collection and recreate it
            self.client.delete_collection(self.config.COLLECTION_NAME)
            self.collection = self.client.create_collection(
                name=self.config.COLLECTION_NAME,
                metadata={"description": "Digital Shield Cybersecurity Data"}
            )
            logger.info("Cleared collection")
            return True
        except Exception as e:
            logger.error(f"Error clearing collection: {e}")
            return False
    
    def save_embeddings_to_file(self, embeddings: np.ndarray, 
                               texts: List[str], 
                               metadata: List[Dict[str, Any]] = None) -> bool:
        """Save embeddings to file as backup"""
        try:
            save_data = {
                'embeddings': embeddings.tolist(),
                'texts': texts,
                'metadata': metadata or [],
                'model': self.config.EMBEDDING_MODEL,
                'dimension': self.config.EMBEDDING_DIMENSION
            }
            
            save_path = self.config.EMBEDDINGS_DIR / "embeddings_backup.pkl"
            with open(save_path, 'wb') as f:
                pickle.dump(save_data, f)
                
            logger.info(f"Saved embeddings backup to {save_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving embeddings to file: {e}")
            return False
    
    def load_embeddings_from_file(self) -> Optional[Dict[str, Any]]:
        """Load embeddings from backup file"""
        try:
            save_path = self.config.EMBEDDINGS_DIR / "embeddings_backup.pkl"
            if not save_path.exists():
                return None
                
            with open(save_path, 'rb') as f:
                save_data = pickle.load(f)
                
            logger.info(f"Loaded embeddings backup from {save_path}")
            return save_data
            
        except Exception as e:
            logger.error(f"Error loading embeddings from file: {e}")
            return None
