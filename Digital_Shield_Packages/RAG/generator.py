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
        """Generate a response based on query and context - GENERAL APPROACH"""
        try:
            # Always enhance context with relevant statistical data for ANY query
            enhanced_context = self._enhance_context_with_statistics(context, query_analysis)
            
            # Use the general LLM response for ALL queries - no hardcoded logic
            return self._generate_general_response(query, enhanced_context)
                
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return self._generate_error_response()
    
    
    
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
    
    def _enhance_context_with_statistics(self, context: str, query_analysis: Dict[str, Any] = None) -> str:
        """Dynamically enhance context with relevant statistical data for ANY query"""
        try:
            # Get actual dataset statistics
            stats = self._get_actual_dataset_statistics()
            if not stats:
                return context
            
            # Create a natural cybersecurity analysis overview
            stats_context = []
            stats_context.append("**Cybersecurity Threat Analysis:**")
            stats_context.append("")
            
            # Always include key statistics that are useful for most queries
            country_dist = stats.get('country_distribution', {})
            if country_dist:
                stats_context.append("**Geographic Impact Analysis:**")
                for country, count in list(country_dist.items())[:5]:
                    stats_context.append(f"- {country}: {count} incidents")
                stats_context.append("")
            
            attack_dist = stats.get('attack_type_distribution', {})
            if attack_dist:
                stats_context.append("**Threat Landscape Overview:**")
                for attack_type, count in list(attack_dist.items())[:5]:
                    stats_context.append(f"- {attack_type}: {count} incidents")
                stats_context.append("")
            
            industry_dist = stats.get('industry_distribution', {})
            if industry_dist:
                stats_context.append("**Industry Vulnerability Assessment:**")
                for industry, count in list(industry_dist.items())[:5]:
                    stats_context.append(f"- {industry}: {count} incidents")
                stats_context.append("")
            
            severity_dist = stats.get('severity_distribution', {})
            if severity_dist:
                stats_context.append("**Incident Severity Distribution:**")
                for severity, count in severity_dist.items():
                    stats_context.append(f"- {severity}: {count} incidents")
                stats_context.append("")
            
            # Add general analysis info
            stats_context.append(f"**Analysis Summary:**")
            stats_context.append(f"- Total incidents analyzed: {stats.get('total_records', 'N/A')}")
            stats_context.append(f"- Average financial impact: ${stats.get('avg_financial_loss', 0):.2f} million")
            stats_context.append(f"- Total users affected: {stats.get('total_affected_users', 0):,}")
            stats_context.append(f"- Analysis period: {stats.get('year_range', ('N/A', 'N/A'))[0]} - {stats.get('year_range', ('N/A', 'N/A'))[1]}")
            stats_context.append("")
            
            # Combine with sample incidents
            enhanced_context = "\n".join(stats_context) + "\n**Recent Incident Examples:**\n" + context
            return enhanced_context
            
        except Exception as e:
            logger.error(f"Error enhancing context with statistics: {e}")
            return context

    def _get_actual_dataset_statistics(self) -> Dict[str, Any]:
        """Get actual statistics from the full dataset"""
        try:
            import pandas as pd
            from pathlib import Path
            
            # Load the actual dataset
            csv_file = Path("Digital_Shield_data/processed/Cleaned_Digital_Shield_with_severity.csv")
            if not csv_file.exists():
                logger.error(f"Dataset file not found: {csv_file}")
                return None
                
            df = pd.read_csv(csv_file)
            
            # Calculate actual statistics
            stats = {
                'total_records': len(df),
                'country_distribution': df['country'].value_counts().to_dict(),
                'attack_type_distribution': df['attack type'].value_counts().to_dict(),
                'industry_distribution': df['target industry'].value_counts().to_dict(),
                'severity_distribution': df['severity_kmeans'].value_counts().to_dict(),
                'avg_financial_loss': df['financial loss (in million $)'].mean(),
                'total_affected_users': df['number of affected users'].sum(),
                'year_range': (df['year'].min(), df['year'].max())
            }
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting actual dataset statistics: {e}")
            return None

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
