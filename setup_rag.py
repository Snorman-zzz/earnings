#!/usr/bin/env python3
"""
Setup script for RAG pipeline installation and testing.
Run this script to install dependencies and test the RAG system.
"""

import os
import sys
import subprocess
from pathlib import Path

def install_dependencies():
    """Install required dependencies from requirements.txt."""
    print("🔧 Installing RAG dependencies...")
    
    try:
        result = subprocess.run([
            sys.executable, "-m", "pip", "install", "-r", "requirements.txt"
        ], capture_output=True, text=True, check=True)
        
        print("✅ Dependencies installed successfully!")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install dependencies: {e}")
        print(f"STDOUT: {e.stdout}")
        print(f"STDERR: {e.stderr}")
        return False

def check_environment():
    """Check if required environment variables are set."""
    print("🔍 Checking environment variables...")
    
    required_vars = ["OPENROUTER_API_KEY"]
    missing_vars = []
    
    for var in required_vars:
        if not os.getenv(var):
            missing_vars.append(var)
    
    if missing_vars:
        print(f"❌ Missing environment variables: {', '.join(missing_vars)}")
        print("Please set these variables in your .env file or environment:")
        for var in missing_vars:
            print(f"  export {var}=your_api_key_here")
        return False
    
    print("✅ Environment variables are set!")
    return True

def test_imports():
    """Test if RAG components can be imported."""
    print("🧪 Testing RAG imports...")
    
    try:
        from src.react_agent.rag_config import RAGConfig
        print("✅ RAG configuration import successful!")
        
        from src.react_agent.rag_pdf_analyzer import RAGPDFAnalyzer
        print("✅ RAG analyzer import successful!")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False

def test_rag_initialization():
    """Test if RAG system can be initialized."""
    print("🚀 Testing RAG system initialization...")
    
    try:
        from src.react_agent.rag_pdf_analyzer import RAGPDFAnalyzer
        analyzer = RAGPDFAnalyzer()
        print("✅ RAG system initialized successfully!")
        return True
        
    except Exception as e:
        print(f"❌ RAG initialization failed: {e}")
        return False

def main():
    """Main setup and test routine."""
    print("🎯 RAG Pipeline Setup and Test")
    print("=" * 40)
    
    # Change to project directory
    project_dir = Path(__file__).parent
    os.chdir(project_dir)
    
    # Install dependencies
    if not install_dependencies():
        print("Setup failed during dependency installation.")
        return False
    
    # Check environment
    if not check_environment():
        print("Setup failed during environment check.")
        return False
    
    # Test imports
    if not test_imports():
        print("Setup failed during import testing.")
        return False
    
    # Test initialization
    if not test_rag_initialization():
        print("Setup failed during RAG initialization.")
        return False
    
    print("\n🎉 RAG Pipeline setup completed successfully!")
    print("You can now run the application with: streamlit run src/react_agent/financial_analyst_app.py")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)