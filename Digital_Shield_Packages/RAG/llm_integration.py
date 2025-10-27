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
            # Create a focused cybersecurity expert prompt
            prompt = f"""
            You are a cybersecurity expert. ONLY answer questions related to cybersecurity, cyber threats, and digital security.
            
            **IMPORTANT:** If the question is NOT about cybersecurity, respond briefly: "I specialize in cybersecurity analysis. Please ask me about cyber threats, security incidents, or digital protection."
            
            **FOR CYBERSECURITY QUESTIONS:**
            - Answer as a knowledgeable cybersecurity expert
            - Use specific statistics and data points from the analysis below
            - Be concise and professional
            - Provide actionable insights
            - If data is not available, say so clearly
            - DO NOT reference specific incident numbers, IDs, or dataset identifiers
            - Focus on patterns, trends, and actionable recommendations
            
            **CYBERSECURITY ANALYSIS DATA:**
            {context}
            
            **USER QUESTION:** {query}
            
            **RESPONSE:**
            If this is a cybersecurity question, provide a focused, data-driven answer. If not, politely redirect to cybersecurity topics.
            """
            
            # Generate response
            response = self.client.generate_content(prompt)
            response_text = response.text
            
            # Remove any incident IDs and dataset identifiers that might be in the response
            import re
            # Remove patterns like #12345, #13478, etc. (but preserve other # usage)
            response_text = re.sub(r'#\d{4,}', '', response_text)  # Only remove 4+ digit IDs
            # Remove any document IDs or chunk references
            response_text = re.sub(r'Document \d+', '', response_text)
            response_text = re.sub(r'Chunk \d+', '', response_text)
            # Enhanced incident number removal - catch various formats
            response_text = re.sub(r'Incident \d+', '', response_text)
            response_text = re.sub(r'incident \d+', '', response_text)
            response_text = re.sub(r'Incident\s+\d+', '', response_text)
            response_text = re.sub(r'incident\s+\d+', '', response_text)
            # Remove standalone incident numbers that might appear
            response_text = re.sub(r'\bIncident\b\s*\d+', '', response_text)
            response_text = re.sub(r'\bincident\b\s*\d+', '', response_text)
            # Clean up multiple spaces but preserve single spaces
            response_text = re.sub(r' {2,}', ' ', response_text).strip()
            
            # Ensure response is complete and not truncated
            if response_text.endswith('Still') or response_text.endswith('still'):
                response_text = response_text[:-5].strip()
            
            return response_text
            
        except Exception as e:
            logger.error(f"Error generating Gemini response: {e}")
            return self._fallback_response(query, context)
    
    def _fallback_response(self, query: str, context: str) -> str:
        """Fallback response when Gemini is not available"""
        # Check if it's a cybersecurity question
        cybersecurity_terms = ['cyber', 'security', 'attack', 'threat', 'incident', 'vulnerability', 'malware', 'phishing', 'ransomware', 'ddos', 'breach', 'hack']
        
        if not any(term in query.lower() for term in cybersecurity_terms):
            return "I specialize in cybersecurity analysis. Please ask me about cyber threats, security incidents, or digital protection."
        
        # Remove incident IDs and dataset identifiers from context
        import re
        clean_context = re.sub(r'#\d{4,}', '', context)  # Only remove 4+ digit IDs
        clean_context = re.sub(r'Document \d+', '', clean_context)
        clean_context = re.sub(r'Chunk \d+', '', clean_context)
        # Enhanced incident number removal - catch various formats
        clean_context = re.sub(r'Incident \d+', '', clean_context)
        clean_context = re.sub(r'incident \d+', '', clean_context)
        clean_context = re.sub(r'Incident\s+\d+', '', clean_context)
        clean_context = re.sub(r'incident\s+\d+', '', clean_context)
        # Remove standalone incident numbers that might appear
        clean_context = re.sub(r'\bIncident\b\s*\d+', '', clean_context)
        clean_context = re.sub(r'\bincident\b\s*\d+', '', clean_context)
        clean_context = re.sub(r' {2,}', ' ', clean_context).strip()  # Only clean multiple spaces
        
        return f"""
        Based on cybersecurity analysis, here's what I found regarding your query: '{query}'
        
        **Key Findings:**
        {clean_context[:1500]}{'...' if len(clean_context) > 1500 else ''}
        
        **Note:** For more intelligent analysis, please set your GEMINI_API_KEY environment variable.
        """

def get_gemini_provider(config: RAGConfig = None) -> GeminiProvider:
    """Get a Gemini provider instance"""
    return GeminiProvider(config=config)
