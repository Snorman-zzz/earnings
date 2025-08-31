# FinSight AI: RAG-Powered Earnings Analysis & Stock Price Prediction

FinSight AI is a comprehensive tool for automating the analysis of corporate earnings reports using **Retrieval-Augmented Generation (RAG)** technology. It combines multiple retrieval strategies with Large Language Models (LLMs) to extract key information from earnings documents, analyze financial metrics, and generate insightful reports with price predictions.

## 🚀 NEW: RAG Pipeline Implementation

This version now uses a sophisticated RAG pipeline instead of direct Claude API calls, providing:
- **Multi-retriever architecture** (Vector, AutoMerging, BM25)
- **Comprehensive document processing** (text + tables extraction)
- **Evaluation-based response selection** (faithfulness + relevancy scoring)
- **Cost-efficient processing** with improved accuracy

## Project Overview

FinSight AI streamlines the financial analysis workflow by:

1. Analyzing earnings press releases and presentation slides uploaded as PDFs
2. Automatically extracting key financial metrics (EPS, revenue, margins, etc.)
3. Comparing reported figures to Wall Street expectations
4. Calculating year-over-year growth rates and surprise percentages
5. Generating comprehensive analysis reports with price predictions

## System Architecture

The system is built on a modular architecture with three main components:

### 1. RAG PDF Analysis Module (`rag_pdf_analyzer.py`)

The RAG PDF analyzer uses a multi-stage retrieval-augmented generation pipeline:
- **Document Ingestion**: Extracts text and tables from PDF files using multiple parsers
- **Multi-retriever Setup**: Implements Vector, AutoMerging, and BM25 retrievers
- **Evaluation Engine**: Uses faithfulness and relevancy scoring to select best responses
- **Response Generation**: Leverages OpenRouter LLMs for structured financial analysis

### 2. RAG Configuration (`rag_config.py`)

Centralized configuration management for the RAG pipeline:
- Model configurations (LLM and embedding models)
- Retrieval parameters and chunking strategies  
- Environment variable handling and validation
- Financial analysis prompt templates

### 3. Financial Data Integration (`financial_agents.py`)

The financial agents framework:
- Fetches real-time market data and analyst estimates using yfinance
- Provides tools for retrieving stock prices and consensus expectations
- Implements a custom agent architecture for financial reasoning
- Structures data into a coherent workflow

### 4. Web Application (`financial_analyst_app.py`)

The streamlit-based web application now features RAG integration:
- Provides an intuitive interface for document uploads
- Displays RAG-powered financial analysis with retrieval metrics
- Shows earnings tables, financial metrics, and price predictions
- Handles user inputs and validation with improved error handling

## Installation

### Quick Setup (Recommended)

1. Clone the repository:
```bash
git clone https://github.com/your-username/finsight-ai.git
cd finsight-ai
```

2. Run the automated setup script:
```bash
python setup_rag.py
```

This script will:
- Install all RAG dependencies
- Check environment variables
- Test the RAG system initialization
- Verify everything is working correctly

### Manual Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Set up environment variables:
Copy `.env.example` to `.env` and fill in your API keys:
```bash
cp .env.example .env
```

Required environment variables:
```
OPENROUTER_API_KEY=your_openrouter_api_key  # Required for RAG pipeline
ANTHROPIC_API_KEY=your_anthropic_api_key    # Optional fallback
OPENAI_API_KEY=your_openai_api_key          # Optional for financial agents
```

## Usage

1. Run the Streamlit application:
```bash
streamlit run src/react_agent/financial_analyst_app.py
```

2. Upload documents:
   - Earnings press release (PDF)
   - Earnings presentation slides (PDF)

3. Enter company information:
   - Company name
   - Stock ticker symbol

4. Click "Generate Comprehensive Analysis" to process the documents

5. Review the analysis:
   - Earnings Calls table (Expected vs. Reported)
   - Financials table (Current Quarter vs. Previous Year)
   - Analysis summary
   - Post-earnings price prediction

## Code Structure

```
finsight-ai/
├── src/react_agent/
│   ├── rag_pdf_analyzer.py      # RAG-based PDF analysis pipeline
│   ├── rag_config.py            # RAG configuration and model setup
│   ├── claude_pdf_analyzer.py   # Original Claude API analyzer (legacy)
│   ├── financial_agents.py      # Agent framework and financial tools
│   └── financial_analyst_app.py # Streamlit web application
├── setup_rag.py                 # Automated setup and testing script
├── .env.example                 # Environment variables template
├── requirements.txt             # Project dependencies (updated for RAG)
└── README.md                    # This file
```

## Key Features

### RAG Pipeline Features
- **Multi-retriever architecture**: Vector similarity, AutoMerging, and BM25 retrievers
- **Comprehensive document processing**: Text extraction + table extraction from PDFs
- **Evaluation-driven selection**: Faithfulness and relevancy scoring for best responses
- **Cost-efficient processing**: Reduced API calls with improved accuracy

### Financial Analysis Features  
- **Precise numerical extraction**: High accuracy in extracting financial metrics
- **Unit normalization**: Automatically handles different numerical formats (B vs M)
- **Structured output**: Well-formatted HTML tables and price prediction formulas
- **Market data integration**: Real-time analyst estimates and stock prices
- **Web-based interface**: Easy-to-use interface for document uploads and analysis

## Limitations and Future Work

- Currently optimized for quarterly earnings reports in standard formats
- Price predictions are based on earnings surprise and market data, not comprehensive market models
- Future improvements planned:
  - Support for more document types (annual reports, investor presentations)
  - Enhanced time-series analysis for better price predictions
  - Sector-specific analysis customizations
  - Expanded historical data integration

## Contributors

- Ziqi Shao: ML method development
- Zhixiao Wu: Refinement of two methods, dataset collection and evaluation
- Mingze Yuan: LLM method development

## License

This project was developed as part of the CS7180: Special Topics in Generative AI course at Northeastern University.

---

This AI-powered financial analysis tool is for informational purposes only and should not be considered financial advice.
