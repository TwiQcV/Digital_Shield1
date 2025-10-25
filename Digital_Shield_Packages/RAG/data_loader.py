"""
Data loader for RAG module
Loads and preprocesses the Digital Shield cybersecurity dataset
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any
from pathlib import Path
import logging

from .config import RAGConfig

logger = logging.getLogger(__name__)

class DataLoader:
    """Handles loading and preprocessing of cybersecurity data for RAG"""
    
    def __init__(self, config: RAGConfig = None):
        self.config = config or RAGConfig()
        self.df = None
        self.text_chunks = []
        
    def load_data(self) -> pd.DataFrame:
        """Load the CSV data file"""
        try:
            if not self.config.CSV_FILE.exists():
                raise FileNotFoundError(f"Data file not found: {self.config.CSV_FILE}")
                
            self.df = pd.read_csv(self.config.CSV_FILE)
            logger.info(f"Loaded {len(self.df)} records from {self.config.CSV_FILE}")
            return self.df
            
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise
    
    def create_text_chunks(self) -> List[str]:
        """Convert structured data to searchable text chunks"""
        if self.df is None:
            self.load_data()
            
        chunks = []
        
        for idx, row in self.df.iterrows():
            # Create a comprehensive text chunk for each record
            chunk = self._create_record_chunk(row, idx)
            chunks.append(chunk)
            
        self.text_chunks = chunks
        logger.info(f"Created {len(chunks)} text chunks")
        return chunks
    
    def _create_record_chunk(self, row: pd.Series, idx: int) -> str:
        """Create a text chunk for a single record with enhanced semantic information"""
        # Create a more comprehensive and semantically rich text description
        country = row.get('country', 'Unknown')
        year = row.get('year', 'Unknown')
        attack_type = row.get('attack type', 'Unknown')
        target_industry = row.get('target industry', 'Unknown')
        financial_loss = row.get('financial loss (in million $)', 0)
        affected_users = row.get('number of affected users', 0)
        vulnerability = row.get('security vulnerability type', 'Unknown')
        defense = row.get('defense mechanism used', 'Unknown')
        resolution_time = row.get('incident resolution time (in hours)', 0)
        data_breach = row.get('data breach in gb', 0)
        severity = row.get('severity_kmeans', 'Unknown')
        
        # Create a more natural, semantic description
        chunk_parts = [
            f"Cybersecurity incident in {country} during {year}",
            f"Attack type: {attack_type} targeting {target_industry} industry",
            f"Security vulnerability: {vulnerability}",
            f"Defense mechanism: {defense}",
            f"Financial impact: ${financial_loss:.2f} million in losses",
            f"Users affected: {affected_users:,.0f} people",
            f"Resolution time: {resolution_time:.1f} hours",
            f"Data breach: {data_breach:.2f} GB of data compromised",
            f"Severity classification: {severity}",
            f"Incident details: {attack_type} attack on {target_industry} sector in {country}",
            f"Security measures: {defense} was used to defend against {vulnerability}",
            f"Impact assessment: {severity} severity with ${financial_loss:.2f}M financial loss"
        ]
        
        # Join with newlines and clean up
        chunk = "\n".join(chunk_parts)
        return chunk.strip()
    
    def get_chunk_metadata(self, chunk_idx: int) -> Dict[str, Any]:
        """Get comprehensive metadata for a specific chunk"""
        if self.df is None:
            self.load_data()
            
        if chunk_idx >= len(self.df):
            return {}
            
        row = self.df.iloc[chunk_idx]
        
        # Convert NumPy types to Python native types for ChromaDB compatibility
        def convert_to_python_type(value):
            """Convert NumPy types to Python native types"""
            if pd.isna(value):
                return None
            if hasattr(value, 'item'):  # NumPy scalar
                return value.item()
            return value
        
        # Enhanced metadata with all useful fields
        return {
            'chunk_id': int(chunk_idx),
            'country': str(row.get('country', '')),
            'year': convert_to_python_type(row.get('year')),
            'attack_type': str(row.get('attack type', '')),
            'target_industry': str(row.get('target industry', '')),
            'security_vulnerability': str(row.get('security vulnerability type', '')),
            'defense_mechanism': str(row.get('defense mechanism used', '')),
            'severity': str(row.get('severity_kmeans', '')),
            'financial_loss': convert_to_python_type(row.get('financial loss (in million $)')),
            'affected_users': convert_to_python_type(row.get('number of affected users')),
            'resolution_time': convert_to_python_type(row.get('incident resolution time (in hours)')),
            'data_breach_gb': convert_to_python_type(row.get('data breach in gb')),
            # Additional semantic fields for better retrieval
            'attack_category': str(row.get('attack type', '')).lower(),
            'industry_category': str(row.get('target industry', '')).lower(),
            'vulnerability_category': str(row.get('security vulnerability type', '')).lower(),
            'defense_category': str(row.get('defense mechanism used', '')).lower(),
            'severity_level': str(row.get('severity_kmeans', '')).lower()
        }
    
    def search_by_criteria(self, 
                          country: str = None,
                          attack_type: str = None, 
                          severity: str = None,
                          year_range: tuple = None) -> List[int]:
        """Search for chunks matching specific criteria"""
        if self.df is None:
            self.load_data()
            
        mask = pd.Series([True] * len(self.df))
        
        if country:
            mask &= self.df['country'].str.contains(country, case=False, na=False)
        if attack_type:
            mask &= self.df['attack type'].str.contains(attack_type, case=False, na=False)
        if severity:
            mask &= self.df['severity_kmeans'].str.contains(severity, case=False, na=False)
        if year_range:
            mask &= (self.df['year'] >= year_range[0]) & (self.df['year'] <= year_range[1])
            
        return self.df[mask].index.tolist()
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get dataset statistics"""
        if self.df is None:
            self.load_data()
            
        return {
            'total_records': len(self.df),
            'countries': self.df['country'].nunique(),
            'attack_types': self.df['attack type'].nunique(),
            'severity_levels': self.df['severity_kmeans'].value_counts().to_dict(),
            'year_range': (self.df['year'].min(), self.df['year'].max()),
            'avg_financial_loss': self.df['financial loss (in million $)'].mean(),
            'total_affected_users': self.df['number of affected users'].sum()
        }
    
    def get_sample_chunks(self, n: int = 3) -> List[str]:
        """Get sample text chunks for testing"""
        if not self.text_chunks:
            self.create_text_chunks()
            
        # Return random sample
        indices = np.random.choice(len(self.text_chunks), min(n, len(self.text_chunks)), replace=False)
        return [self.text_chunks[i] for i in indices]
