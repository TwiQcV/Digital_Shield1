"""
Streamlit Integration Example for Digital Shield RAG System
This shows how to integrate the RAG system with your Streamlit app
"""

import streamlit as st
import sys
from pathlib import Path
from typing import Dict, Any

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from Digital_Shield_Packages.RAG.main import RAGSystem

@st.cache_resource
def get_rag_system():
    """
    Get or create the RAG system instance (cached for performance)
    """
    return RAGSystem()

def initialize_rag_system():
    """
    Initialize the RAG system with loading states
    """
    if 'rag_system' not in st.session_state:
        with st.spinner("🔒 Initializing Ḥimā RAG System..."):
            st.session_state.rag_system = get_rag_system()
            success = st.session_state.rag_system.initialize()
            
            if not success:
                st.error("❌ Failed to initialize RAG system. Please check your configuration.")
                st.stop()
    
    return st.session_state.rag_system

def display_rag_chat():
    """
    Display the RAG chat interface
    """
    st.title("🔒 Ḥimā - Digital Shield RAG System")
    st.caption("With Saudi heritage of protection")
    
    # Initialize RAG system
    rag_system = initialize_rag_system()
    
    # Chat interface
    st.subheader("💬 Ask about Cybersecurity")
    
    # Input area
    user_question = st.text_input(
        "What would you like to know about cyber threats?",
        placeholder="e.g., What are the most common attack types?",
        key="user_question"
    )
    
    # Query options
    col1, col2 = st.columns(2)
    with col1:
        top_k = st.slider("Number of documents to retrieve", 5, 50, 20)
    with col2:
        similarity_threshold = st.slider("Similarity threshold", 0.1, 1.0, 0.7)
    
    # Process query button
    if st.button("🔍 Ask Ḥimā", type="primary"):
        if user_question:
            with st.spinner("🔍 Analyzing your question..."):
                result = rag_system.query(
                    user_question,
                    top_k=top_k,
                    similarity_threshold=similarity_threshold
                )
            
            # Display results
            if result.get('error'):
                st.error(f"❌ {result.get('response')}")
            else:
                # Main response
                st.success("📝 Answer:")
                st.write(result.get('response', ''))
                
                # Metadata
                with st.expander("📊 Query Details"):
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Documents Retrieved", result.get('retrieved_documents_count', 0))
                    with col2:
                        st.metric("Context Length", result.get('context_length', 0))
                    with col3:
                        st.metric("Query Type", result.get('query_analysis', {}).get('intent', 'Unknown'))
                
                # Suggested queries
                suggested = result.get('suggested_queries', [])
                if suggested:
                    st.subheader("💡 You might also ask:")
                    for i, suggestion in enumerate(suggested[:3], 1):
                        if st.button(f"{i}. {suggestion}", key=f"suggestion_{i}"):
                            st.session_state.user_question = suggestion
                            st.rerun()
        else:
            st.warning("Please enter a question")

def display_system_status():
    """
    Display RAG system status
    """
    st.subheader("📊 System Status")
    
    rag_system = initialize_rag_system()
    status = rag_system.get_status()
    
    # System info
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("System Ready", "✅ Yes" if status.get('is_ready') else "❌ No")
    with col2:
        st.metric("Embedding Model", status.get('config', {}).get('embedding_model', 'Unknown'))
    with col3:
        st.metric("Vector DB", status.get('config', {}).get('vector_db', 'Unknown'))
    
    # Collection stats
    if 'collection_stats' in status:
        stats = status['collection_stats']
        st.subheader("📚 Data Statistics")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total Documents", stats.get('total_documents', 0))
        with col2:
            st.metric("Collection Name", status.get('config', {}).get('collection', 'Unknown'))

def display_sample_queries():
    """
    Display sample queries
    """
    st.subheader("💡 Sample Questions")
    
    rag_system = initialize_rag_system()
    sample_queries = rag_system.get_sample_queries()
    
    for i, query in enumerate(sample_queries[:5], 1):
        if st.button(f"{i}. {query}", key=f"sample_{i}"):
            st.session_state.user_question = query
            st.rerun()

# Example usage in your main Streamlit app
def main():
    """
    Example of how to integrate this into your main Streamlit app
    """
    st.set_page_config(
        page_title="Digital Shield RAG",
        page_icon="🔒",
        layout="wide"
    )
    
    # Sidebar
    with st.sidebar:
        st.title("🔒 Ḥimā RAG System")
        
        # Navigation
        page = st.selectbox(
            "Choose a page:",
            ["Chat", "System Status", "Sample Queries"]
        )
    
    # Main content
    if page == "Chat":
        display_rag_chat()
    elif page == "System Status":
        display_system_status()
    elif page == "Sample Queries":
        display_sample_queries()

if __name__ == "__main__":
    main()
