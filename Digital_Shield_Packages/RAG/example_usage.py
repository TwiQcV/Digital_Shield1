"""
Simple RAG System Interface
Ask one question at a time about cybersecurity data
"""

import logging
import os
import sys
from pathlib import Path

# Add the parent directory to Python path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from Digital_Shield_Packages.RAG.pipeline import RAGPipeline

# Setup logging - suppress INFO logs for cleaner output
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

def main():
    """Simple interface for asking questions"""
    print("🔒 Ḥimā - Securing Your Digital World")
    print("With Saudi heritage of protection")
    print()
    print("💬 What would you like to know about cyber threats?")
    print("=" * 50)
    
    # Check for Gemini API key
    api_key = os.getenv('GEMINI_API_KEY')
    if not api_key:
        print("\n⚠️  GEMINI_API_KEY not set!")
        print("💡 Set your API key: export GEMINI_API_KEY='your_api_key_here'")
        print("   Get API key from: https://makersuite.google.com/app/apikey")
        return
    
    try:
        # Initialize the RAG pipeline
        pipeline = RAGPipeline()
        os.environ['TOKENIZERS_PARALLELISM'] = 'false'  # Suppress tokenizer warnings
        
        init_success = pipeline.initialize()
        
        if not init_success:
            print("❌ Failed to initialize pipeline")
            return
        
        # Get user input
        query = input().strip()
        
        if not query:
            print("❌ No question provided. Exiting.")
            return
        
        # Process the query (suppress all internal logging)
        import warnings
        warnings.filterwarnings('ignore')
        
        # Use dynamic retrieval (automatically selects optimal number of documents)
        response = pipeline.query(query)
        
        if response.get('error'):
            print(f"\n❌ Error: {response.get('response')}")
        elif response.get('no_results'):
            print("\n⚠️  No results found for your question")
        else:
            # Display the clean response
            response_text = response.get('response', '')
            print(f"\n📝 Answer:")
            print("=" * 50)
            print(response_text)
            print("=" * 50)
            
            # Show suggested queries (clean format)
            suggested = response.get('suggested_queries', [])
            if suggested:
                print(f"\n💡 You might also ask:")
                for i, suggestion in enumerate(suggested[:3], 1):
                    print(f"   {i}. {suggestion}")
        
    except Exception as e:
        logger.error(f"Query failed: {e}")
        print(f"❌ Query failed: {e}")

if __name__ == "__main__":
    main()
