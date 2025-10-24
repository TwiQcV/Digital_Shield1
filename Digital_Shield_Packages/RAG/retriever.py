"""
Retriever module for semantic search and document retrieval
"""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple
import logging
from collections import Counter

from .config import RAGConfig
from .embed_store import EmbeddingStore
from .data_loader import DataLoader

logger = logging.getLogger(__name__)

class Retriever:
    """Handles semantic search and document retrieval"""
    
    def __init__(self, config: RAGConfig = None, embed_store: EmbeddingStore = None, data_loader: DataLoader = None):
        self.config = config or RAGConfig()
        self.embed_store = embed_store or EmbeddingStore(self.config)
        self.data_loader = data_loader or DataLoader(self.config)
        
    def retrieve_documents(self, query: str, top_k: int = None, 
                          similarity_threshold: float = None) -> List[Dict[str, Any]]:
        """Retrieve relevant documents with dynamic sampling based on query type"""
        similarity_threshold = similarity_threshold or self.config.SIMILARITY_THRESHOLD
        
        try:
            # Classify query type for dynamic retrieval
            query_type = self._classify_query_type(query)
            
            # Set dynamic top_k based on query type
            if query_type == "statistical":
                dynamic_top_k = self.config.STATISTICAL_TOP_K  # "most common", "trends", "overview"
                logger.info(f"Statistical query detected - using {dynamic_top_k} documents")
            elif query_type == "specific":
                dynamic_top_k = self.config.SPECIFIC_TOP_K   # "incident #123", "cases in Brazil"
                logger.info(f"Specific query detected - using {dynamic_top_k} documents")
            else:  # general queries
                dynamic_top_k = self.config.GENERAL_TOP_K   # balanced approach
                logger.info(f"General query detected - using {dynamic_top_k} documents")
            
            # Use dynamic top_k or override if specified
            final_top_k = top_k or dynamic_top_k
            
            # Search for documents (no more multiplication logic)
            all_results = self.embed_store.search_similar(query, final_top_k)
            
            if not all_results:
                return []
            
            # For statistical queries, use diverse sampling
            # For specific/general queries, use similarity-based selection
            if query_type == "statistical":
                diverse_results = self._apply_diverse_sampling(all_results, final_top_k)
            else:
                # For specific and general queries, just use the most similar documents
                diverse_results = all_results[:final_top_k]
            
            # Use distance-based filtering (more reliable than similarity)
            distance_threshold = 2.0  # Accept documents with distance < 2.0
            filtered_results = [
                result for result in diverse_results 
                if result['distance'] <= distance_threshold
            ]
            
            # Debug: Show country distribution
            countries = [r.get('metadata', {}).get('country', 'unknown') for r in filtered_results]
            country_counts = Counter(countries)
            logger.info(f"Retrieved {len(filtered_results)} documents for {query_type} query: '{query[:50]}...'")
            logger.info(f"Country distribution: {dict(country_counts)}")
            return filtered_results
            
        except Exception as e:
            logger.error(f"Error retrieving documents: {e}")
            return []
    
    def _apply_diverse_sampling(self, results: List[Dict[str, Any]], target_count: int) -> List[Dict[str, Any]]:
        """Apply diverse sampling to ensure balanced representation"""
        if len(results) <= target_count:
            return results
        
        # For statistical queries, use more diverse sampling
        # For specific queries, use similarity-based selection
        # For general queries, use balanced approach
        
        # Group by country for diversity analysis
        country_groups = {}
        for result in results:
            country = result.get('metadata', {}).get('country', 'unknown')
            if country not in country_groups:
                country_groups[country] = []
            country_groups[country].append(result)
        
        countries = list(country_groups.keys())
        if not countries:
            return results[:target_count]
        
        # Calculate proportional distribution based on actual data distribution
        total_docs = len(results)
        diverse_results = []
        
        # Take proportional samples from each country (not equal samples)
        for country in countries:
            country_docs = country_groups[country]
            country_proportion = len(country_docs) / total_docs
            country_target = max(1, int(target_count * country_proportion))
            
            # Sort by similarity and take best from this country
            country_docs.sort(key=lambda x: x.get('distance', 0))
            selected_docs = country_docs[:country_target]
            diverse_results.extend(selected_docs)
            
            logger.info(f"Country {country}: {len(country_docs)} total, {len(selected_docs)} selected (proportional)")
        
        # If we need more documents, fill with remaining best matches
        if len(diverse_results) < target_count:
            remaining_needed = target_count - len(diverse_results)
            used_docs = set(id(doc) for doc in diverse_results)
            
            for result in results:
                if len(diverse_results) >= target_count:
                    break
                if id(result) not in used_docs:
                    diverse_results.append(result)
        
        # Sort final results by similarity
        diverse_results.sort(key=lambda x: x.get('distance', 0))
        return diverse_results[:target_count]
    
    def retrieve_by_criteria(self, query: str, 
                           country: str = None,
                           attack_type: str = None,
                           severity: str = None,
                           year_range: Tuple[int, int] = None) -> List[Dict[str, Any]]:
        """Retrieve documents with additional filtering criteria"""
        try:
            # First get semantic search results
            semantic_results = self.retrieve_documents(query)
            
            # Apply additional filters if specified
            filtered_results = []
            for result in semantic_results:
                metadata = result.get('metadata', {})
                
                # Check country filter
                if country and metadata.get('country', '').lower() != country.lower():
                    continue
                    
                # Check attack type filter
                if attack_type and attack_type.lower() not in metadata.get('attack_type', '').lower():
                    continue
                    
                # Check severity filter
                if severity and severity.lower() not in metadata.get('severity', '').lower():
                    continue
                    
                # Check year range filter
                if year_range:
                    year = metadata.get('year')
                    if year and not (year_range[0] <= year <= year_range[1]):
                        continue
                
                filtered_results.append(result)
            
            logger.info(f"Filtered to {len(filtered_results)} documents with criteria")
            return filtered_results
            
        except Exception as e:
            logger.error(f"Error retrieving documents with criteria: {e}")
            return []
    
    def get_context_for_query(self, query: str, max_context_length: int = 50000) -> str:
        """Get formatted context for a query"""
        try:
            # Retrieve relevant documents
            results = self.retrieve_documents(query)
            
            if not results:
                return "No relevant documents found."
            
            # Format context
            context_parts = []
            current_length = 0
            
            for i, result in enumerate(results):
                text = result['text']
                similarity = result['similarity']
                
                # Add document with similarity score
                doc_text = f"Document {i+1} (Similarity: {similarity:.3f}):\n{text}\n\n"
                
                if current_length + len(doc_text) > max_context_length:
                    break
                    
                context_parts.append(doc_text)
                current_length += len(doc_text)
            
            context = "".join(context_parts)
            logger.info(f"Generated context of {len(context)} characters")
            return context
            
        except Exception as e:
            logger.error(f"Error generating context: {e}")
            return "Error generating context."
    
    def _classify_query_type(self, query: str) -> str:
        """Classify query type for dynamic retrieval"""
        try:
            query_lower = query.lower()
            
            # Statistical indicators - need comprehensive data
            statistical_terms = [
                'most common', 'trends', 'overview', 'statistics', 'distribution', 
                'patterns', 'frequently', 'average', 'top', 'highest', 'lowest',
                'compare', 'comparison', 'vs', 'versus', 'over time', 'yearly',
                'summary', 'analysis', 'insights', 'findings'
            ]
            
            # Specific indicators - need focused results
            specific_terms = [
                'incident #', 'case #', 'specific', 'particular', 'exactly',
                'in brazil', 'in china', 'in usa', 'in uk', 'in germany',
                'in 2023', 'in 2024', 'in 2022', 'exact', 'precise'
            ]
            
            # Check for statistical terms
            if any(term in query_lower for term in statistical_terms):
                return "statistical"
            
            # Check for specific terms
            if any(term in query_lower for term in specific_terms):
                return "specific"
            
            # Default to general
            return "general"
            
        except Exception as e:
            logger.error(f"Error classifying query type: {e}")
            return "general"
    
    def analyze_query_intent(self, query: str) -> Dict[str, Any]:
        """Analyze the intent and extract key terms from the query"""
        try:
            # Simple keyword extraction and intent analysis
            query_lower = query.lower()
            
            # Common cybersecurity terms
            attack_types = ['phishing', 'ransomware', 'ddos', 'malware', 'sql injection', 'man-in-the-middle']
            countries = ['china', 'uk', 'germany', 'france', 'usa', 'japan']
            industries = ['education', 'retail', 'it', 'healthcare', 'finance', 'government']
            severity_levels = ['critical', 'high', 'medium', 'low']
            
            # Extract detected terms
            detected_attack_types = [term for term in attack_types if term in query_lower]
            detected_countries = [term for term in countries if term in query_lower]
            detected_industries = [term for term in industries if term in query_lower]
            detected_severity = [term for term in severity_levels if term in query_lower]
            
            # Determine query type using the new classification method
            query_type = self._classify_query_type(query)
            
            return {
                'query_type': query_type,
                'detected_attack_types': detected_attack_types,
                'detected_countries': detected_countries,
                'detected_industries': detected_industries,
                'detected_severity': detected_severity,
                'query_length': len(query),
                'word_count': len(query.split())
            }
            
        except Exception as e:
            logger.error(f"Error analyzing query intent: {e}")
            return {'query_type': 'general', 'error': str(e)}
    
    def get_statistical_summary(self, query: str) -> Dict[str, Any]:
        """Get statistical summary based on retrieved documents"""
        try:
            results = self.retrieve_documents(query, top_k=20)  # Get more results for stats
            
            if not results:
                return {"error": "No documents found"}
            
            # Extract statistics from metadata
            countries = []
            attack_types = []
            severities = []
            financial_losses = []
            affected_users = []
            
            for result in results:
                metadata = result.get('metadata', {})
                if metadata.get('country'):
                    countries.append(metadata['country'])
                if metadata.get('attack_type'):
                    attack_types.append(metadata['attack_type'])
                if metadata.get('severity'):
                    severities.append(metadata['severity'])
                if metadata.get('financial_loss'):
                    financial_losses.append(metadata['financial_loss'])
                if metadata.get('affected_users'):
                    affected_users.append(metadata['affected_users'])
            
            # Calculate statistics
            stats = {
                'total_documents': len(results),
                'country_distribution': dict(Counter(countries)),
                'attack_type_distribution': dict(Counter(attack_types)),
                'severity_distribution': dict(Counter(severities)),
                'avg_financial_loss': np.mean(financial_losses) if financial_losses else 0,
                'total_affected_users': sum(affected_users) if affected_users else 0,
                'avg_similarity': np.mean([r['similarity'] for r in results])
            }
            
            return stats
            
        except Exception as e:
            logger.error(f"Error generating statistical summary: {e}")
            return {"error": str(e)}
    
    def suggest_related_queries(self, query: str) -> List[str]:
        """Suggest related queries based on the current query"""
        try:
            # Simple query suggestions based on common patterns
            suggestions = []
            
            query_lower = query.lower()
            
            # Add suggestions based on query type
            if 'phishing' in query_lower:
                suggestions.extend([
                    "What are the most common phishing attack targets?",
                    "How effective are different defense mechanisms against phishing?",
                    "What is the financial impact of phishing attacks?"
                ])
            elif 'ransomware' in query_lower:
                suggestions.extend([
                    "Which industries are most affected by ransomware?",
                    "What are the average resolution times for ransomware attacks?",
                    "How much data is typically breached in ransomware attacks?"
                ])
            elif 'financial' in query_lower or 'loss' in query_lower:
                suggestions.extend([
                    "What are the highest financial loss incidents?",
                    "Which countries have the highest financial losses?",
                    "How do financial losses correlate with attack severity?"
                ])
            else:
                # General suggestions
                suggestions.extend([
                    "What are the most common attack types?",
                    "Which countries are most affected by cyber attacks?",
                    "What are the trends in cybersecurity incidents over time?",
                    "How do different industries compare in terms of security incidents?"
                ])
            
            return suggestions[:5]  # Return top 5 suggestions
            
        except Exception as e:
            logger.error(f"Error generating query suggestions: {e}")
            return []
