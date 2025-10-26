import streamlit as st
import os
import sys
from pathlib import Path
from typing import Dict, Any

# Add project root to path for RAG system imports
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Import RAG system
try:
    from Digital_Shield_Packages.RAG.main import RAGSystem
    RAG_AVAILABLE = True
except ImportError as e:
    st.warning(f"RAG system not available: {e}")
    RAG_AVAILABLE = False

# Configure the page
st.set_page_config(
    page_title="Ḥimā - Digital Shield",
    page_icon="🔒",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for professional styling
st.markdown("""
<style>
    .main-header {
        text-align: center;
        margin-bottom: 2rem;
    }
    
    .avatar-container {
        display: flex;
        justify-content: center;
        margin-bottom: 1rem;
    }
    
    .branding {
        text-align: center;
        margin-bottom: 2rem;
    }
    
    .main-title {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f4e79;
        margin-bottom: 0.5rem;
    }
    
    .subtitle {
        font-size: 1.2rem;
        color: #666;
        font-style: italic;
        margin-bottom: 1.5rem;
    }
    
    .quick-questions {
        background-color: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #1f4e79;
        margin-bottom: 2rem;
    }
    
    .quick-questions h3 {
        color: #1f4e79;
        margin-bottom: 1rem;
    }
    
    .quick-questions ul {
        margin: 0;
        padding-left: 1.5rem;
    }
    
    .quick-questions li {
        margin-bottom: 0.5rem;
        color: #333;
    }
    
    .chat-header {
        background: linear-gradient(90deg, #1f4e79, #2c5aa0);
        color: white;
        padding: 1rem;
        border-radius: 10px 10px 0 0;
        margin-bottom: 0;
    }
    
    .chat-subheader {
        background-color: #e8f4f8;
        padding: 0.5rem 1rem;
        border-radius: 0 0 10px 10px;
        margin-bottom: 1rem;
        color: #1f4e79;
        font-weight: 500;
    }
    
    .separator {
        height: 2px;
        background: linear-gradient(90deg, #1f4e79, #2c5aa0);
        margin: 1rem 0;
    }
    
    .chat-container {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 1rem;
        min-height: 400px;
        max-height: 500px;
        overflow-y: auto;
    }
    
    .user-message {
        background-color: #1f4e79;
        color: white;
        padding: 0.8rem;
        border-radius: 15px 15px 5px 15px;
        margin: 0.5rem 0 0.5rem 20%;
        text-align: right;
    }
    
    .assistant-message {
        background-color: white;
        color: #333;
        padding: 0.8rem;
        border-radius: 15px 15px 15px 5px;
        margin: 0.5rem 20% 0.5rem 0;
        border: 1px solid #e0e0e0;
    }
    
    .stChatInput > div > div > input {
        border-radius: 25px;
        border: 2px solid #1f4e79;
    }
    
    .stChatInput > div > div > input:focus {
        border-color: #2c5aa0;
        box-shadow: 0 0 0 0.2rem rgba(31, 78, 121, 0.25);
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Initialize RAG system
@st.cache_resource
def get_rag_system():
    """Get or create the RAG system instance (cached for performance)"""
    if RAG_AVAILABLE:
        return RAGSystem()
    return None

def initialize_rag_system():
    """Initialize the RAG system with loading states"""
    if RAG_AVAILABLE and 'rag_system' not in st.session_state:
        with st.spinner("🔒 Initializing Ḥimā RAG System..."):
            rag_system = get_rag_system()
            if rag_system:
                success = rag_system.initialize()
                if success:
                    st.session_state.rag_system = rag_system
                    st.session_state.rag_ready = True
                else:
                    st.session_state.rag_ready = False
            else:
                st.session_state.rag_ready = False
    
    return st.session_state.get('rag_system', None), st.session_state.get('rag_ready', False)

# Helper function to generate responses using RAG system with LLM
def generate_response(user_input):
    """Generate intelligent responses using RAG system with LLM fallback"""
    input_lower = user_input.lower()
    
    # Try to use RAG system first
    rag_system, rag_ready = initialize_rag_system()
    
    if rag_ready and rag_system:
        try:
            # Use RAG system to get intelligent response
            result = rag_system.query(
                user_input,
                top_k=20,
                similarity_threshold=0.7
            )
            
            if not result.get('error'):
                response = result.get('response', '')
                
                # Add system info if available
                if result.get('retrieved_documents_count', 0) > 0:
                    response += f"\n\n*📊 Based on analysis of {result.get('retrieved_documents_count', 0)} cybersecurity documents*"
                
                # Add suggested queries if available
                suggested = result.get('suggested_queries', [])
                if suggested:
                    response += "\n\n💡 **You might also ask:**"
                    for i, suggestion in enumerate(suggested[:3], 1):
                        response += f"\n{i}. {suggestion}"
                
                return response
            else:
                # RAG system error, fall back to hardcoded responses
                st.warning("RAG system encountered an error, using fallback responses.")
                
        except Exception as e:
            st.warning(f"RAG system error: {e}. Using fallback responses.")
    
    # If RAG system is not available, return a simple message
    return f"""I apologize, but I'm currently unable to process your request: "{user_input}"

Please try again later or contact support for assistance."""

# Main application layout
def main():
    # Top Section - Centered
    st.markdown('<div class="main-header">', unsafe_allow_html=True)
    
    # Avatar container
    st.markdown('<div class="avatar-container">', unsafe_allow_html=True)
    avatar_path = "Digital_Shield_Avatars/Welcome.jpg"
    if os.path.exists(avatar_path):
        st.image(avatar_path, width=150)
    else:
        st.markdown("🔒", style="font-size: 100px; text-align: center;")
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Branding
    st.markdown('<div class="branding">', unsafe_allow_html=True)
    st.markdown('<h1 class="main-title">🔒 Ḥimā - Securing Your Digital World</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">With Saudi heritage of protection</p>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Main Chat Area - Full Width
    st.markdown("""
    <div class="separator"></div>
    """, unsafe_allow_html=True)
    
    # Chat messages display area
    chat_container = st.container()
    
    with chat_container:
        
        if st.session_state.messages:
            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])
        else:
            # Welcome message
            with st.chat_message("assistant"):
                st.markdown("""
                **Welcome to Ḥimā!** 🔒
                
                I'm your digital security assistant, here to help protect your digital world with the wisdom of Saudi heritage.
                
                You can ask me about:
                - **Phishing protection** and how to identify threats
                - **Ransomware trends** and prevention strategies  
                - **Banking security** and financial protection
                - **Global cyber threat landscape** by country/region
                - Any other cybersecurity concerns
                
                How can I help secure your digital world today?
                """)
    
    # Chat input at bottom
    if prompt := st.chat_input("🔍 Ask about cyber threats..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate and display assistant response
        with st.chat_message("assistant"):
            response = generate_response(prompt)
            st.markdown(response)
        
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})

if __name__ == "__main__":
    main()
