"""
Gemini API Integration for RAG system
Simple integration with Google's Gemini API for intelligent responses
"""

import os
import logging
from typing import Dict, Any

# Load .env file if it exists
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    # dotenv not installed, that's okay
    pass

from .config import RAGConfig

logger = logging.getLogger(__name__)

class GeminiProvider:
    """Google Gemini API provider for intelligent responses"""
    
    def __init__(self, api_key: str = None, config: RAGConfig = None):
        self.api_key = api_key or os.getenv('GEMINI_API_KEY')
        self.model_name = "gemini-2.5-flash"
        self.client = None
        self.config = config or RAGConfig()
        
        if self.api_key:
            self._initialize_client()
    
    def _initialize_client(self):
        """Initialize the Gemini client"""
        try:
            import google.generativeai as genai
            genai.configure(api_key=self.api_key)
            self.client = genai.GenerativeModel(self.model_name)
            logger.info(f"Gemini client initialized successfully with model: {self.model_name}")
        except ImportError:
            logger.error("Google Generative AI package not installed. Install with: pip install google-generativeai")
        except Exception as e:
            logger.error(f"Error initializing Gemini client: {e}")
    
    def is_available(self) -> bool:
        """Check if Gemini is available"""
        return self.client is not None and self.api_key is not None
    
    def generate_response(self, query: str, context: str) -> str:
        """Generate response using Gemini API"""
        if not self.is_available():
            return self._fallback_response(query, context)
        
        try:
            # Create a comprehensive prompt for cybersecurity analysis with proper contextualization
            prompt = f"""
            You are a cybersecurity expert analyzing data from the Digital Shield dataset. 
            Provide a CLEAN, PROFESSIONAL answer to the user's question.
            
            **RULES:**
            - Answer directly and conversationally
            - NEVER mention similarity scores, document IDs, or retrieval details
            - NEVER show raw document chunks or technical metadata
            - Present findings in clear, organized bullet points
            - Include relevant statistics and examples naturally
            - Use emojis sparingly for visual clarity
            
            **Provide a polished, user-friendly response:**
            
            Based on the following context about cybersecurity incidents, answer the user's question.
            
            IMPORTANT CONTEXTUALIZATION GUIDELINES:
            - The Digital Shield dataset contains approximately 18,000+ cybersecurity incident records
            - The context provided below contains relevant incidents retrieved for this query
            - Present findings directly without mentioning sample sizes or disclaimers
            - Focus on actionable insights and recommendations
            - Avoid technical disclaimers about dataset limitations
            
            Context (Sample of relevant incidents):
            {context}
            
            User Question: {query}
            
            Please provide a detailed, informative response based on the data. Follow these guidelines:
            1. Present findings directly and confidently
            2. Include specific statistics and patterns from the data
            3. Focus on actionable insights and recommendations
            4. Use clear, professional language
            5. If the context doesn't contain enough information to fully answer the question, please say so
            6. Avoid mentioning sample sizes or dataset limitations
            
            Format your response in a clear, professional manner with bullet points or sections as appropriate.
            """
            
            # Generate response
            response = self.client.generate_content(prompt)
            return response.text
            
        except Exception as e:
            logger.error(f"Error generating Gemini response: {e}")
            return self._fallback_response(query, context)
    
    def _fallback_response(self, query: str, context: str) -> str:
        """Fallback response when Gemini is not available"""
        return f"""
        Based on the cybersecurity data, here's what I found regarding your query: '{query}'
        
        **Key Findings:**
        {context[:1500]}{'...' if len(context) > 1500 else ''}
        
        **Note:** For more intelligent analysis, please set your GEMINI_API_KEY environment variable.
        """

def get_gemini_provider(config: RAGConfig = None) -> GeminiProvider:
    """Get a Gemini provider instance"""
    return GeminiProvider(config=config)
