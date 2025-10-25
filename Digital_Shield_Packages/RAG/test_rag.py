#!/usr/bin/env python3
"""
Main RAG System Test File
Single test file for the Digital Shield RAG system - modify this file as needed
"""

import os
import sys
import logging
from pathlib import Path

# Add the project root to Python path (two levels up from RAG folder)
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Suppress logging for cleaner output
logging.basicConfig(level=logging.WARNING)

def test_setup():
    """Test if everything is set up correctly"""
    # Check .env file
    env_file = project_root / ".env"
    if env_file.exists():
        with open(env_file, 'r') as f:
            content = f.read()
            if 'GEMINI_API_KEY' not in content:
                print("❌ GEMINI_API_KEY not found in .env file")
                return False
    else:
        print("❌ .env file not found")
        return False
    
    # Test imports
    try:
        from .pipeline import RAGPipeline
    except ImportError as e:
        print(f"❌ Failed to import RAGPipeline: {e}")
        return False
    
    # Test Gemini API
    try:
        from .llm_integration import get_gemini_provider
        provider = get_gemini_provider()
        if not provider.is_available():
            print("❌ Gemini API not working")
            return False
    except Exception as e:
        print(f"❌ Gemini API test failed: {e}")
        return False
    
    return True

def test_rag_system():
    """Test the RAG system with user input"""
    try:
        from .pipeline import RAGPipeline
        
        pipeline = RAGPipeline()
        success = pipeline.initialize()
        
        if not success:
            print("❌ Failed to initialize pipeline")
            return False
        
        # Get user input for query
        
        query = input().strip()
        
        if not query:
            print("❌ No query provided. Exiting.")
            return False
        
        # Process the query with dynamic retrieval (automatically selects based on query type)
        response = pipeline.query(query)
        
        if response.get('error'):
            print(f"❌ Query failed: {response.get('response')}")
            return False
        
        # Display the response
        print("\n" + "=" * 50)
        print("📝 ANSWER:")
        print("=" * 50)
        response_text = response.get('response', '')
        print(response_text)
        print("=" * 50)
        
        # Show suggested queries
        suggested = response.get('suggested_queries', [])
        if suggested:
            print(f"\n💡 You might also ask:")
            for i, suggestion in enumerate(suggested[:3], 1):
                print(f"   {i}. {suggestion}")
        
        return True
        
    except Exception as e:
        print(f"❌ RAG system test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main test function"""
    print("🔒 Ḥimā - Securing Your Digital World")
    print("With Saudi heritage of protection")
    print()
    print("💬 What would you like to know about cyber threats?")
    print("=" * 60)
    
    # Test setup
    setup_ok = test_setup()
    
    if setup_ok:
        # Test RAG system with user input
        rag_ok = test_rag_system()
        
        if not rag_ok:
            print("\n⚠️  RAG system test failed")
            print("Check the error messages above for details")
    else:
        print("\n❌ Setup test failed")
        print("Please check the error messages above")

if __name__ == "__main__":
    main()
