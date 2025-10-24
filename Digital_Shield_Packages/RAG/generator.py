"""
Generator module for LLM response generation
Handles text generation and response formatting
"""

import json
from typing import List, Dict, Any, Optional
import logging
from datetime import datetime

from .config import RAGConfig
from .llm_integration import get_gemini_provider

logger = logging.getLogger(__name__)

class Generator:
    """Handles response generation and formatting"""
    
    def __init__(self, config: RAGConfig = None):
        self.config = config or RAGConfig()
        self.gemini_provider = get_gemini_provider(config=self.config)
        
    def generate_response(self, query: str, context: str, 
                        query_analysis: Dict[str, Any] = None,
                        statistical_summary: Dict[str, Any] = None) -> str:
        """Generate a response based on query and context"""
        try:
            # Determine response type based on query analysis
            if query_analysis and query_analysis.get('query_type') == 'statistical':
                return self._generate_statistical_response(query, context, statistical_summary)
            elif query_analysis and query_analysis.get('query_type') == 'comparative':
                return self._generate_comparative_response(query, context)
            elif query_analysis and query_analysis.get('query_type') == 'trend_analysis':
                return self._generate_trend_response(query, context)
            else:
                return self._generate_general_response(query, context)
                
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return self._generate_error_response()
    
    def _generate_statistical_response(self, query: str, context: str, 
                                     stats: Dict[str, Any] = None) -> str:
        """Generate a statistical response"""
        try:
            if not stats:
                return self._generate_general_response(query, context)
            
            response_parts = [
                f"**Key Statistics:**",
                f"**Total Incidents Analyzed:** {stats.get('total_documents', 'N/A')}",
                f"**Average Financial Loss:** ${stats.get('avg_financial_loss', 0):.2f} million",
                f"**Total Affected Users:** {stats.get('total_affected_users', 0):,}",
                f"**Average Similarity Score:** {stats.get('avg_similarity', 0):.3f}",
                ""
            ]
            
            # Add distribution data
            if stats.get('country_distribution'):
                response_parts.append("**Countries Affected:**")
                for country, count in list(stats['country_distribution'].items())[:5]:
                    response_parts.append(f"- {country}: {count} incidents")
                response_parts.append("")
            
            if stats.get('attack_type_distribution'):
                response_parts.append("**Attack Types:**")
                for attack_type, count in list(stats['attack_type_distribution'].items())[:5]:
                    response_parts.append(f"- {attack_type}: {count} incidents")
                response_parts.append("")
            
            if stats.get('severity_distribution'):
                response_parts.append("**Severity Distribution:**")
                for severity, count in stats['severity_distribution'].items():
                    response_parts.append(f"- {severity}: {count} incidents")
                response_parts.append("")
            
            # Add context summary
            response_parts.extend([
                "**Relevant Incidents:**",
                context[:1000] + "..." if len(context) > 1000 else context
            ])
            
            return "\n".join(response_parts)
            
        except Exception as e:
            logger.error(f"Error generating statistical response: {e}")
            return self._generate_general_response(query, context)
    
    def _generate_comparative_response(self, query: str, context: str) -> str:
        """Generate a comparative response"""
        try:
            response_parts = [
                "Here's a comparative analysis based on the cybersecurity data:",
                "",
                "**Key Comparisons:**",
                "",
                context[:1500] + "..." if len(context) > 1500 else context,
                "",
                "**Analysis:**",
                "The data shows variations in attack patterns, financial impacts, and resolution times across different categories. This comparison helps identify trends and patterns in cybersecurity incidents."
            ]
            
            return "\n".join(response_parts)
            
        except Exception as e:
            logger.error(f"Error generating comparative response: {e}")
            return self._generate_general_response(query, context)
    
    def _generate_trend_response(self, query: str, context: str) -> str:
        """Generate a trend analysis response"""
        try:
            response_parts = [
                "Here's a trend analysis based on the cybersecurity data:",
                "",
                "**Trend Analysis:**",
                "",
                context[:1500] + "..." if len(context) > 1500 else context,
                "",
                "**Key Insights:**",
                "- The data reveals patterns in attack frequency and impact over time",
                "- Financial losses and affected user counts show varying trends",
                "- Different attack types demonstrate different temporal patterns"
            ]
            
            return "\n".join(response_parts)
            
        except Exception as e:
            logger.error(f"Error generating trend response: {e}")
            return self._generate_general_response(query, context)
    
    def _generate_general_response(self, query: str, context: str) -> str:
        """Generate a general response using Gemini API"""
        try:
            # Use Gemini for intelligent response generation
            response = self.gemini_provider.generate_response(query, context)
            
            # Return the response directly without disclaimers
            return response
            
        except Exception as e:
            logger.error(f"Error generating general response: {e}")
            return self._generate_error_response()
    
    def _generate_error_response(self) -> str:
        """Generate an error response"""
        return (
            "I apologize, but I encountered an error while processing your query. "
            "Please try rephrasing your question or contact support if the issue persists."
        )
    
    def format_response(self, response: str, include_metadata: bool = True) -> Dict[str, Any]:
        """Format the response with metadata"""
        try:
            formatted_response = {
                'response': response,
                'timestamp': datetime.now().isoformat(),
                'response_length': len(response),
                'word_count': len(response.split())
            }
            
            if include_metadata:
                formatted_response['metadata'] = {
                    'generated_by': 'Digital Shield RAG System',
                    'version': '1.0',
                    'model': self.config.LLM_MODEL
                }
            
            return formatted_response
            
        except Exception as e:
            logger.error(f"Error formatting response: {e}")
            return {
                'response': response,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def generate_summary(self, documents: List[Dict[str, Any]]) -> str:
        """Generate a summary of retrieved documents"""
        try:
            if not documents:
                return "No documents found to summarize."
            
            # Extract key information
            countries = set()
            attack_types = set()
            severities = set()
            total_financial_loss = 0
            total_affected_users = 0
            
            for doc in documents:
                metadata = doc.get('metadata', {})
                if metadata.get('country'):
                    countries.add(metadata['country'])
                if metadata.get('attack_type'):
                    attack_types.add(metadata['attack_type'])
                if metadata.get('severity'):
                    severities.add(metadata['severity'])
                if metadata.get('financial_loss'):
                    total_financial_loss += metadata['financial_loss']
                if metadata.get('affected_users'):
                    total_affected_users += metadata['affected_users']
            
            summary_parts = [
                f"**Summary of {len(documents)} cybersecurity incidents (Sample from Digital Shield Dataset):**",
                "",
                f"**Scope Note:** This summary is based on a limited sample of {len(documents)} incidents from the Digital Shield dataset (18,000+ total records).",
                "",
                f"**Countries in Sample:** {', '.join(sorted(countries))}",
                f"**Attack Types in Sample:** {', '.join(sorted(attack_types))}",
                f"**Severity Levels in Sample:** {', '.join(sorted(severities))}",
                f"**Total Financial Loss (Sample):** ${total_financial_loss:.2f} million",
                f"**Total Affected Users (Sample):** {total_affected_users:,}",
                "",
                "**Key Incidents (Sample):**"
            ]
            
            # Add top 3 most relevant incidents
            for i, doc in enumerate(documents[:3]):
                summary_parts.append(f"{i+1}. {doc['text'][:200]}...")
            
            return "\n".join(summary_parts)
            
        except Exception as e:
            logger.error(f"Error generating summary: {e}")
            return "Error generating summary."
    
    def generate_insights(self, documents: List[Dict[str, Any]]) -> List[str]:
        """Generate insights from the documents"""
        try:
            insights = []
            
            if not documents:
                return ["No documents available for insights."]
            
            # Analyze patterns
            countries = [doc.get('metadata', {}).get('country') for doc in documents if doc.get('metadata', {}).get('country')]
            attack_types = [doc.get('metadata', {}).get('attack_type') for doc in documents if doc.get('metadata', {}).get('attack_type')]
            severities = [doc.get('metadata', {}).get('severity') for doc in documents if doc.get('metadata', {}).get('severity')]
            
            # Generate insights based on patterns with sample disclaimers
            if countries:
                most_common_country = max(set(countries), key=countries.count)
                insights.append(f"Most affected country in sample: {most_common_country}")
            
            if attack_types:
                most_common_attack = max(set(attack_types), key=attack_types.count)
                insights.append(f"Most common attack type in sample: {most_common_attack}")
            
            if severities:
                severity_dist = {}
                for severity in severities:
                    severity_dist[severity] = severity_dist.get(severity, 0) + 1
                most_common_severity = max(severity_dist, key=severity_dist.get)
                insights.append(f"Most common severity level in sample: {most_common_severity}")
            
            # Add general insights with sample disclaimers
            insights.extend([
                f"Analysis based on {len(documents)} relevant incidents (sample from 18,000+ total records)",
                "Sample data shows patterns in cybersecurity threat landscape",
                "Financial impact and user impact vary significantly across sample incidents",
                "Note: These insights are based on the specific sample retrieved and may not represent the full dataset"
            ])
            
            return insights
            
        except Exception as e:
            logger.error(f"Error generating insights: {e}")
            return ["Error generating insights."]
