#!/usr/bin/env python3
"""
RAG System Main Entry Point
Digital Shield Cybersecurity RAG System
"""

import os
import sys
import logging
import argparse
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime

# Add the project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Import RAG components
from .pipeline import RAGPipeline
from .config import RAGConfig

# Configure logging
logging.basicConfig(
    level=logging.WARNING,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class RAGSystem:
    """Main RAG System class for Digital Shield"""
    
    def __init__(self, config: Optional[RAGConfig] = None):
        self.config = config or RAGConfig()
        self.pipeline = None
        self.is_ready = False
        self.initialization_time = None
        
    def initialize(self, force_rebuild: bool = False, silent: bool = False) -> bool:
        """Initialize the RAG pipeline"""
        try:
            if not silent:
                print("🔒 Initializing Ḥimā RAG System...")
                print("With Saudi heritage of protection")
                print("=" * 60)
            
            self.pipeline = RAGPipeline(self.config)
            success = self.pipeline.initialize(force_rebuild=force_rebuild)
            
            if success:
                self.is_ready = True
                self.initialization_time = datetime.now()
                if not silent:
                    print("✅ RAG System initialized successfully!")
                return True
            else:
                if not silent:
                    print("❌ RAG System initialization failed!")
                return False
                
        except Exception as e:
            if not silent:
                print(f"❌ Error initializing RAG System: {e}")
            logger.error(f"RAG System initialization error: {e}")
            return False
    
    def query(self, question: str, **kwargs) -> Dict[str, Any]:
        """Process a query through the RAG system"""
        if not self.is_ready:
            return {
                'response': 'RAG System not initialized. Please initialize first.',
                'error': True,
                'timestamp': datetime.now().isoformat()
            }
        
        try:
            result = self.pipeline.query(question, **kwargs)
            result['system_info'] = {
                'initialization_time': self.initialization_time.isoformat() if self.initialization_time else None,
                'config': {
                    'embedding_model': self.config.EMBEDDING_MODEL,
                    'vector_db': self.config.VECTOR_DB_TYPE,
                    'collection': self.config.COLLECTION_NAME
                }
            }
            return result
            
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            return {
                'response': f'I apologize, but I encountered an error processing your question: {str(e)}',
                'error': True,
                'timestamp': datetime.now().isoformat()
            }
    
    def get_status(self) -> Dict[str, Any]:
        """Get the current status of the RAG system"""
        if not self.pipeline:
            return {
                'is_ready': False,
                'status': 'Not initialized',
                'timestamp': datetime.now().isoformat()
            }
        
        try:
            status = self.pipeline.get_pipeline_status()
            status['is_ready'] = self.is_ready
            status['initialization_time'] = self.initialization_time.isoformat() if self.initialization_time else None
            return status
            
        except Exception as e:
            logger.error(f"Error getting status: {e}")
            return {
                'is_ready': False,
                'status': f'Error: {str(e)}',
                'timestamp': datetime.now().isoformat()
            }

def check_environment() -> bool:
    """Check if the environment is properly configured"""
    # Check .env file
    env_file = project_root / ".env"
    if not env_file.exists():
        print("❌ .env file not found")
        return False
    
    # Check for GEMINI_API_KEY
    with open(env_file, 'r') as f:
        content = f.read()
        if 'GEMINI_API_KEY' not in content:
            print("❌ GEMINI_API_KEY not found in .env file")
            return False
    
    # Check data file
    config = RAGConfig()
    if not config.CSV_FILE.exists():
        print(f"❌ Data file not found: {config.CSV_FILE}")
        return False
    
    return True

def interactive_mode():
    """Run the RAG system in interactive mode for testing"""
    print("🔒 Ḥimā - Securing Your Digital World")
    print("With Saudi heritage of protection")
    print()
    print("💬 What would you like to know about cyber threats?")
    print("=" * 60)
    
    # Check environment
    if not check_environment():
        print("\n❌ Environment check failed")
        print("Please check the error messages above")
        return
    
    # Initialize RAG system
    rag_system = RAGSystem()
    if not rag_system.initialize(silent=True):
        print("\n❌ RAG System initialization failed")
        return
    
    # Interactive loop
    try:
        while True:
            query = input("\n🔍 Enter your question (or 'quit' to exit): ").strip()
            
            if query.lower() in ['quit', 'exit', 'q']:
                print("👋 Goodbye! Stay secure!")
                break
            
            if not query:
                print("❌ Please enter a question")
                continue
            
            # Process query
            result = rag_system.query(query)
            
            if result.get('error'):
                print(f"❌ Error: {result.get('response')}")
            else:
                print("\n" + "=" * 50)
                print("📝 ANSWER:")
                print("=" * 50)
                print(result.get('response', ''))
                print("=" * 50)
                
                # Show suggested queries
                suggested = result.get('suggested_queries', [])
                if suggested:
                    print(f"\n💡 You might also ask:")
                    for i, suggestion in enumerate(suggested[:3], 1):
                        print(f"   {i}. {suggestion}")
    
    except KeyboardInterrupt:
        print("\n👋 Goodbye! Stay secure!")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")

def main():
    """Main entry point for the RAG system"""
    parser = argparse.ArgumentParser(
        description="Digital Shield RAG System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                    # Run interactive mode
  python main.py --rebuild          # Rebuild embeddings
  python main.py --status           # Check system status
        """
    )
    
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Rebuild embeddings"
    )
    
    parser.add_argument(
        "--status",
        action="store_true", 
        help="Check system status"
    )
    
    parser.add_argument(
        "--interactive",
        action="store_true",
        default=True,
        help="Run in interactive mode (default)"
    )
    
    args = parser.parse_args()
    
    # Handle rebuild
    if args.rebuild:
        print("🔄 Rebuilding RAG System...")
        rag_system = RAGSystem()
        if rag_system.initialize(silent=True):
            rag_system.rebuild_embeddings()
        return
    
    # Handle status check
    if args.status:
        print("📊 RAG System Status:")
        rag_system = RAGSystem()
        if rag_system.initialize(silent=True):
            status = rag_system.get_status()
            print(f"✅ System Ready: {status.get('is_initialized', False)}")
            if 'collection_stats' in status:
                stats = status['collection_stats']
                print(f"📚 Documents: {stats.get('total_documents', 0)}")
        else:
            print("❌ System not ready")
        return
    
    # Run interactive mode
    if args.interactive:
        interactive_mode()

if __name__ == "__main__":
    main()
