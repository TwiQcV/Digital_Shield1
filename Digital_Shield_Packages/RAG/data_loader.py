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
        """Create a text chunk for a single record"""
        # Format the record as a comprehensive text description
        chunk_parts = [
            f"Cybersecurity Incident #{idx + 1}",
            f"Country: {row.get('country', 'Unknown')}",
            f"Year: {row.get('year', 'Unknown')}",
            f"Attack Type: {row.get('attack type', 'Unknown')}",
            f"Target Industry: {row.get('target industry', 'Unknown')}",
            f"Financial Loss: ${row.get('financial loss (in million $)', 0):.2f} million",
            f"Affected Users: {row.get('number of affected users', 0):,.0f}",
            f"Security Vulnerability: {row.get('security vulnerability type', 'Unknown')}",
            f"Defense Mechanism: {row.get('defense mechanism used', 'Unknown')}",
            f"Resolution Time: {row.get('incident resolution time (in hours)', 0):.1f} hours",
            f"Data Breach Size: {row.get('data breach in gb', 0):.2f} GB",
            f"Severity Level: {row.get('severity_kmeans', 'Unknown')}"
        ]
        
        # Join with newlines and clean up
        chunk = "\n".join(chunk_parts)
        return chunk.strip()
    
    def get_chunk_metadata(self, chunk_idx: int) -> Dict[str, Any]:
        """Get metadata for a specific chunk"""
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
        
        return {
            'chunk_id': int(chunk_idx),
            'country': str(row.get('country', '')),
            'year': convert_to_python_type(row.get('year')),
            'attack_type': str(row.get('attack type', '')),
            'severity': str(row.get('severity_kmeans', '')),
            'financial_loss': convert_to_python_type(row.get('financial loss (in million $)')),
            'affected_users': convert_to_python_type(row.get('number of affected users'))
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
