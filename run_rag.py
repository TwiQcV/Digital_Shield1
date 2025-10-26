#!/usr/bin/env python3
"""
Simple RAG System Runner
Easy way to run the Digital Shield RAG system
"""

import sys
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Import and run the RAG system
from Digital_Shield_Packages.RAG.main import main

if __name__ == "__main__":
    main()
