import streamlit as st
import os
import sys
import time
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

# Custom CSS for modern chat interface
st.markdown("""
<style>
    /* Avatar container styling */
    .avatar-container {
        display: flex !important;
        justify-content: center !important;
        align-items: center !important;
        margin-bottom: 1rem !important;
        width: 100% !important;
    }
    
    /* Circular avatar styling */
    .avatar-container img,
    .stImage img {
        border-radius: 50% !important;
        object-fit: cover !important;
        border: 3px solid #1f4e79 !important;
        box-shadow: 0 4px 8px rgba(31, 78, 121, 0.2) !important;
        width: 120px !important;
        height: 120px !important;
    }
    
    /* Force circular styling on all images in avatar container */
    div[data-testid="stImage"] img {
        border-radius: 50% !important;
        object-fit: cover !important;
        border: 3px solid #1f4e79 !important;
        box-shadow: 0 4px 8px rgba(31, 78, 121, 0.2) !important;
        width: 120px !important;
        height: 120px !important;
    }
    
    /* Custom chat input styling */
    .stChatInput > div > div > input {
        border-radius: 25px;
        border: 2px solid #1f4e79;
    }
    
    .stChatInput > div > div > input:focus {
        border-color: #2c5aa0;
        box-shadow: 0 0 0 0.2rem rgba(31, 78, 121, 0.25);
    }
    
    /* Custom pills styling */
    .stPills {
        margin-top: 1rem;
    }
    
    /* Title styling */
    .stTitle {
        color: #1f4e79;
    }
    
    /* Button styling */
    .stButton > button {
        background-color: #1f4e79;
        color: white;
        border-radius: 20px;
        border: none;
        padding: 0.5rem 1rem;
    }
    
    .stButton > button:hover {
        background-color: #2c5aa0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []

if "current_state" not in st.session_state:
    st.session_state.current_state = "welcome"

if "success_start_time" not in st.session_state:
    st.session_state.success_start_time = None

# Function to get avatar based on current state
def get_avatar_for_state(state):
    """Get the appropriate avatar image based on the current state"""
    avatar_mapping = {
        "welcome": "Digital_Shield_Avatars/Welcome.jpg",
        "processing": "Digital_Shield_Avatars/Processing State.jpg",
        "success": "Digital_Shield_Avatars/Welcome.jpg",  # Use welcome for success
        "error": "Digital_Shield_Avatars/Error State.jpg"
    }
    return avatar_mapping.get(state, "Digital_Shield_Avatars/Welcome.jpg")

# Initialize RAG system
@st.cache_resource
def get_rag_system():
    """Get or create the RAG system instance (cached for performance)"""
    if RAG_AVAILABLE:
        return RAGSystem()
    return None

def initialize_rag_system():
    """Initialize the RAG system silently in the background"""
    if RAG_AVAILABLE and 'rag_system' not in st.session_state:
        # Initialize silently without showing spinner to users
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
def generate_response(user_input, avatar_placeholder):
    """Generate intelligent responses using RAG system with LLM fallback"""
    input_lower = user_input.lower()

    # Set state to processing and update avatar
    st.session_state.current_state = "processing"
    avatar_path = get_avatar_for_state(st.session_state.current_state)
    if os.path.exists(avatar_path):
        avatar_placeholder.image(avatar_path, width=150)
    else:
        avatar_placeholder.markdown("", style="font-size: 100px; text-align: center;")

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
                # Removed document count display as requested

                # Add suggested queries if available
                suggested = result.get('suggested_queries', [])
                if suggested:
                    response += "\n\n💡 **You might also ask:**"
                    for i, suggestion in enumerate(suggested[:3], 1):
                        response += f"\n{i}. {suggestion}"

                # Set state to success and update avatar
                st.session_state.current_state = "success"
                st.session_state.success_start_time = time.time()  # Start the success timer
                avatar_path = get_avatar_for_state(st.session_state.current_state)
                if os.path.exists(avatar_path):
                    avatar_placeholder.image(avatar_path, width=150)
                else:
                    avatar_placeholder.markdown("", style="font-size: 100px; text-align: center;")
                
                return response
            else:
                # RAG system error, fall back to hardcoded responses
                st.session_state.current_state = "error"
                avatar_path = get_avatar_for_state(st.session_state.current_state)
                if os.path.exists(avatar_path):
                    avatar_placeholder.image(avatar_path, width=150)
                else:
                    avatar_placeholder.markdown("", style="font-size: 100px; text-align: center;")
                st.warning("RAG system encountered an error, using fallback responses.")

        except Exception as e:
            st.session_state.current_state = "error"
            avatar_path = get_avatar_for_state(st.session_state.current_state)
            if os.path.exists(avatar_path):
                avatar_placeholder.image(avatar_path, width=150)
            else:
                avatar_placeholder.markdown("", style="font-size: 100px; text-align: center;")
            st.warning(f"RAG system error: {e}. Using fallback responses.")

    # If RAG system is not available, return a simple message
    st.session_state.current_state = "error"
    avatar_path = get_avatar_for_state(st.session_state.current_state)
    if os.path.exists(avatar_path):
        avatar_placeholder.image(avatar_path, width=150)
    else:
        avatar_placeholder.markdown("", style="font-size: 100px; text-align: center;")
    return f"""I apologize, but I'm currently unable to process your request: "{user_input}"
Please try again later or contact support for assistance."""

# Main application layout
def main():
    # Centered avatar and title
    col1, col2, col3 = st.columns([1, 3, 1])
    
    with col2:
        # Avatar container with placeholder for dynamic updates
        st.markdown('<div class="avatar-container">', unsafe_allow_html=True)
        avatar_placeholder = st.empty()
        
        # Handle success state timing
        if st.session_state.current_state == "success":
            if st.session_state.success_start_time is None:
                st.session_state.success_start_time = time.time()
            elif time.time() - st.session_state.success_start_time > 3.0:  # 3 second delay
                st.session_state.current_state = "welcome"
                st.session_state.success_start_time = None
        
        avatar_path = get_avatar_for_state(st.session_state.current_state)
        if os.path.exists(avatar_path):
            avatar_placeholder.image(avatar_path, width=120)
        else:
            avatar_placeholder.markdown("", style="font-size: 80px; text-align: center;")
        st.markdown('</div>', unsafe_allow_html=True)

        # Centered title and caption
        st.markdown('<div style="text-align: center;">', unsafe_allow_html=True)
        st.title("Ḥimā - Securing Your Digital World", anchor=False)
        st.caption("With Saudi heritage of protection")
        st.markdown('</div>', unsafe_allow_html=True)

    # Title row for restart button
    title_row = st.container(horizontal=True, vertical_alignment="bottom")

    # Check if user has interacted yet
    user_just_asked_initial_question = (
        "initial_question" in st.session_state and st.session_state.initial_question
    )
    
    user_just_clicked_suggestion = (
        "selected_suggestion" in st.session_state and st.session_state.selected_suggestion
    )
    
    user_first_interaction = (
        user_just_asked_initial_question or user_just_clicked_suggestion
    )
    
    has_message_history = (
        "messages" in st.session_state and len(st.session_state.messages) > 0
    )

    # Show welcome screen when user hasn't asked a question yet
    if not user_first_interaction and not has_message_history:
        st.session_state.messages = []

        with st.container():
            st.chat_input("🔍 Ask about cyber threats...", key="initial_question")

            # Example suggestions
            suggestions = [
                "How to protect my bank from cyber threats?",
                "What are the latest phishing techniques?",
                "How to secure my business from ransomware?",
                "What are the top cybersecurity trends in 2024?",
                "How to identify fake banking emails?"
            ]
            
            selected_suggestion = st.pills(
                label="💡 Example Questions",
                label_visibility="collapsed",
                options=suggestions,
                key="selected_suggestion",
            )

        st.stop()

    # Show chat input at the bottom when a question has been asked
    user_message = st.chat_input("🔍 Ask a follow-up question...")

    if not user_message:
        if user_just_asked_initial_question:
            user_message = st.session_state.initial_question
        if user_just_clicked_suggestion:
            user_message = st.session_state.selected_suggestion

    # Clear conversation button
    with title_row:
        def clear_conversation():
            st.session_state.messages = []
            st.session_state.initial_question = None
            st.session_state.selected_suggestion = None
            st.session_state.current_state = "welcome"

        st.button("🔄 Restart", on_click=clear_conversation)

    # Display chat messages from history
    for i, message in enumerate(st.session_state.messages):
        with st.chat_message(message["role"]):
            if message["role"] == "assistant":
                st.container()  # Fix ghost message bug
            st.markdown(message["content"])

    if user_message:
        # When the user posts a message...
        
        # Display user message
        with st.chat_message("user"):
            st.text(user_message)

        # Display assistant response
        with st.chat_message("assistant"):
            with st.spinner("🔒 Analyzing your security question..."):
                # Update avatar to processing state
                st.session_state.current_state = "processing"
                avatar_path = get_avatar_for_state(st.session_state.current_state)
                if os.path.exists(avatar_path):
                    avatar_placeholder.image(avatar_path, width=120)
                else:
                    avatar_placeholder.markdown("🔒", style="font-size: 80px; text-align: center;")
                
                # Generate response
                response = generate_response(user_message, avatar_placeholder)

            # Put everything after the spinner in a container to fix ghost message bug
            with st.container():
                # Display the response
                st.markdown(response)

                # Add messages to chat history
                st.session_state.messages.append({"role": "user", "content": user_message})
                st.session_state.messages.append({"role": "assistant", "content": response})

if __name__ == "__main__":
    main()
