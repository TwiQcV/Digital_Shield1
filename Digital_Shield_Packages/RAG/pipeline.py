"""
Main RAG pipeline that orchestrates the entire retrieval-augmented generation process
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime

from .config import RAGConfig
from .data_loader import DataLoader
from .embed_store import EmbeddingStore
from .retriever import Retriever
from .generator import Generator

# Set logger to WARNING level to suppress INFO messages
logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)

class RAGPipeline:
    """Main RAG pipeline that orchestrates the entire process"""
    
    def __init__(self, config: RAGConfig = None):
        self.config = config or RAGConfig()
        
        # Initialize components
        self.data_loader = DataLoader(self.config)
        self.embed_store = EmbeddingStore(self.config)
        self.retriever = Retriever(self.config, self.embed_store, self.data_loader)
        self.generator = Generator(self.config)
        
        # Pipeline state
        self.is_initialized = False
        self.initialization_log = []
        
    def initialize(self, force_rebuild: bool = False) -> bool:
        """Initialize the RAG pipeline"""
        try:
            logger.info("Initializing RAG pipeline...")
            
            # Check if already initialized and not forcing rebuild
            if self.is_initialized and not force_rebuild:
                logger.info("Pipeline already initialized")
                return True
            
            # Step 1: Load data
            logger.info("Loading data...")
            self.data_loader.load_data()
            self.initialization_log.append("Data loaded successfully")
            
            # Step 2: Create text chunks
            logger.info("Creating text chunks...")
            text_chunks = self.data_loader.create_text_chunks()
            self.initialization_log.append(f"Created {len(text_chunks)} text chunks")
            
            # Step 3: Initialize embedding store
            logger.info("Initializing embedding store...")
            self.embed_store.initialize_model()
            self.embed_store.initialize_vector_db()
            self.initialization_log.append("Embedding store initialized")
            
            # Step 4: Create and store embeddings
            if force_rebuild or not self._check_existing_embeddings():
                logger.info("Creating embeddings...")
                embeddings = self.embed_store.create_embeddings(text_chunks)
                
                # Prepare metadata
                metadata = []
                for i in range(len(text_chunks)):
                    meta = self.data_loader.get_chunk_metadata(i)
                    metadata.append(meta)
                
                # Store embeddings
                logger.info("Storing embeddings...")
                self.embed_store.store_embeddings(text_chunks, embeddings, metadata)
                self.initialization_log.append("Embeddings created and stored")
            else:
                logger.info("Using existing embeddings")
                self.initialization_log.append("Using existing embeddings")
            
            self.is_initialized = True
            logger.info("RAG pipeline initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error initializing RAG pipeline: {e}")
            self.initialization_log.append(f"Error: {str(e)}")
            return False
    
    def _check_existing_embeddings(self) -> bool:
        """Check if embeddings already exist"""
        try:
            stats = self.embed_store.get_collection_stats()
            return stats.get('total_documents', 0) > 0
        except:
            return False
    
    def query(self, question: str, 
              top_k: int = None,
              similarity_threshold: float = None,
              include_metadata: bool = True) -> Dict[str, Any]:
        """Process a query through the RAG pipeline"""
        try:
            if not self.is_initialized:
                logger.warning("Pipeline not initialized, initializing now...")
                if not self.initialize():
                    return self._create_error_response("Failed to initialize pipeline")
            
            logger.info(f"Processing query: '{question[:100]}...'")
            
            # Step 1: Analyze query
            query_analysis = self.retriever.analyze_query_intent(question)
            
            # Step 2: Retrieve relevant documents
            documents = self.retriever.retrieve_documents(
                question, 
                top_k=top_k, 
                similarity_threshold=similarity_threshold
            )
            
            if not documents:
                return self._create_no_results_response(question)
            
            # Step 3: Get context
            context = self.retriever.get_context_for_query(question)
            
            # Step 4: Get statistical summary
            statistical_summary = self.retriever.get_statistical_summary(question)
            
            # Step 5: Generate response
            response_text = self.generator.generate_response(
                question, 
                context, 
                query_analysis, 
                statistical_summary
            )
            
            # Step 6: Format response
            formatted_response = self.generator.format_response(
                response_text, 
                include_metadata
            )
            
            # Add additional information
            formatted_response.update({
                'query_analysis': query_analysis,
                'retrieved_documents_count': len(documents),
                'context_length': len(context),
                'suggested_queries': self.retriever.suggest_related_queries(question)
            })
            
            logger.info(f"Query processed successfully. Retrieved {len(documents)} documents")
            return formatted_response
            
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            return self._create_error_response(str(e))
    
    def query_with_filters(self, question: str,
                          country: str = None,
                          attack_type: str = None,
                          severity: str = None,
                          year_range: Tuple[int, int] = None) -> Dict[str, Any]:
        """Process a query with additional filters"""
        try:
            if not self.is_initialized:
                if not self.initialize():
                    return self._create_error_response("Failed to initialize pipeline")
            
            logger.info(f"Processing filtered query: '{question[:100]}...'")
            
            # Analyze query
            query_analysis = self.retriever.analyze_query_intent(question)
            
            # Retrieve with filters
            documents = self.retriever.retrieve_by_criteria(
                question,
                country=country,
                attack_type=attack_type,
                severity=severity,
                year_range=year_range
            )
            
            if not documents:
                return self._create_no_results_response(question)
            
            # Get context and generate response
            context = self.retriever.get_context_for_query(question)
            statistical_summary = self.retriever.get_statistical_summary(question)
            
            response_text = self.generator.generate_response(
                question,
                context,
                query_analysis,
                statistical_summary
            )
            
            formatted_response = self.generator.format_response(response_text)
            formatted_response.update({
                'query_analysis': query_analysis,
                'filters_applied': {
                    'country': country,
                    'attack_type': attack_type,
                    'severity': severity,
                    'year_range': year_range
                },
                'retrieved_documents_count': len(documents)
            })
            
            return formatted_response
            
        except Exception as e:
            logger.error(f"Error processing filtered query: {e}")
            return self._create_error_response(str(e))
    
    def get_pipeline_status(self) -> Dict[str, Any]:
        """Get the current status of the pipeline"""
        try:
            status = {
                'is_initialized': self.is_initialized,
                'initialization_log': self.initialization_log,
                'config': {
                    'embedding_model': self.config.EMBEDDING_MODEL,
                    'vector_db_type': self.config.VECTOR_DB_TYPE,
                    'collection_name': self.config.COLLECTION_NAME
                }
            }
            
            if self.is_initialized:
                # Get collection statistics
                stats = self.embed_store.get_collection_stats()
                status['collection_stats'] = stats
                
                # Get data statistics
                data_stats = self.data_loader.get_statistics()
                status['data_stats'] = data_stats
            
            return status
            
        except Exception as e:
            logger.error(f"Error getting pipeline status: {e}")
            return {'error': str(e)}
    
    def rebuild_embeddings(self) -> bool:
        """Rebuild all embeddings"""
        try:
            logger.info("Rebuilding embeddings...")
            
            # Clear existing embeddings
            self.embed_store.clear_collection()
            
            # Reinitialize
            return self.initialize(force_rebuild=True)
            
        except Exception as e:
            logger.error(f"Error rebuilding embeddings: {e}")
            return False
    
    def _create_error_response(self, error_message: str) -> Dict[str, Any]:
        """Create an error response"""
        return {
            'response': f"I apologize, but I encountered an error: {error_message}",
            'error': True,
            'timestamp': datetime.now().isoformat(),
            'suggested_queries': [
                "What are the most common attack types?",
                "Which countries are most affected by cyber attacks?",
                "What are the trends in cybersecurity incidents?"
            ]
        }
    
    def _create_no_results_response(self, question: str) -> Dict[str, Any]:
        """Create a response when no results are found"""
        return {
            'response': f"I couldn't find relevant information for your query: '{question}'. Please try rephrasing your question or asking about cybersecurity incidents, attack types, or security trends.",
            'no_results': True,
            'timestamp': datetime.now().isoformat(),
            'suggested_queries': [
                "What are the most common cybersecurity attack types?",
                "Which countries have the highest number of cyber incidents?",
                "What are the financial impacts of cybersecurity breaches?",
                "How do different industries compare in terms of security incidents?"
            ]
        }
    
    def get_sample_queries(self) -> List[str]:
        """Get sample queries for testing"""
        return [
            "What are the most common attack types?",
            "Which countries are most affected by cyber attacks?",
            "What is the average financial loss from ransomware attacks?",
            "How many users are typically affected by phishing attacks?",
            "What are the most effective defense mechanisms?",
            "Which industries are most vulnerable to cyber attacks?",
            "What are the trends in cybersecurity incidents over time?",
            "How long does it typically take to resolve security incidents?",
            "What is the relationship between attack severity and financial loss?",
            "Which security vulnerabilities are most common?"
        ]
