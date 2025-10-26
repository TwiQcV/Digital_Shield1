#!/usr/bin/env python3
"""
Digital Shield Streamlit Chatbot Runner
Easy way to run the Ḥimā chatbot application
"""

import subprocess
import sys
import os
from pathlib import Path

def main():
    """Run the Streamlit chatbot application"""
    
    # Get the current directory
    current_dir = Path(__file__).parent
    
    # Path to the streamlit app
    app_path = current_dir / "streamlit_chatbot.py"
    
    # Check if the app file exists
    if not app_path.exists():
        print(f"Error: {app_path} not found!")
        sys.exit(1)
    
    print("🔒 Starting Ḥimā - Digital Shield Chatbot...")
    print("=" * 50)
    print("The application will open in your default web browser.")
    print("If it doesn't open automatically, check the terminal for the URL.")
    print("=" * 50)
    
    try:
        # Run streamlit
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", 
            str(app_path),
            "--server.port", "8501",
            "--server.address", "localhost",
            "--browser.gatherUsageStats", "false"
        ], cwd=current_dir)
    except KeyboardInterrupt:
        print("\n👋 Ḥimā chatbot stopped. Stay secure!")
    except Exception as e:
        print(f"Error running the application: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
