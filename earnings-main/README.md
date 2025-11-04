# 📊 FinSight AI - Automated Earnings Analysis

**Multi-Agent RAG System for Earnings Report Analysis**

FinSight AI is a comprehensive tool for automating the analysis of corporate earnings reports. It uses a Multi-Agent RAG (Retrieval-Augmented Generation) system with specialized AI agents to extract key financial metrics, verify accuracy, and generate insightful analysis reports.

> **🚀 Recent Update:** Migrated from Claude Files API to a full Multi-Agent RAG architecture with Docling PDF processing, hybrid retrieval, and verification workflows.

---

## 🎯 Project Overview

FinSight AI streamlines the financial analysis workflow by:

1. **Processing** earnings press releases and presentation slides (PDFs)
2. **Extracting** key financial metrics (EPS, revenue, margins, etc.) using AI
3. **Verifying** extracted data against source documents
4. **Comparing** reported figures to Wall Street expectations
5. **Calculating** year-over-year growth rates and surprise percentages
6. **Generating** comprehensive analysis reports

---

## ✨ Key Features

- ✅ **Multi-Agent System**: Research Agent + Verification Agent with workflow orchestration
- ✅ **Hybrid Retrieval**: BM25 (keyword) + Vector Search (semantic)
- ✅ **Table Preservation**: Financial tables kept intact during PDF processing
- ✅ **Unit Normalization**: Handles M (millions) vs B (billions) conversions
- ✅ **GAAP Classification**: Distinguishes GAAP vs Non-GAAP figures
- ✅ **Automated Verification**: Cross-checks all metrics against source documents
- ✅ **Analyst Estimates**: Fetches real-time data from Yahoo Finance
- ✅ **YoY Comparisons**: Automatic year-over-year growth calculations
- ✅ **Export Results**: Download reports in Markdown format

---

## 🚀 Quick Start

### 1. Install Dependencies

```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

*Note: Installation may take 5-10 minutes due to large dependencies (PyTorch, transformers, etc.)*

### 2. Set Up Environment

```bash
cp .env_example .env
# Edit .env and add your OPENAI_API_KEY
```

Required in `.env`:
```bash
OPENAI_API_KEY="sk-..."
```

### 3. Run the Application

```bash
streamlit run app.py
```

Open `http://localhost:8501` in your browser.

---

## 📚 Usage

### Web Interface

1. **Enter Company Information** (sidebar):
   - Stock Ticker (e.g., NVDA)
   - Company Name (e.g., NVIDIA Corporation)

2. **Upload Documents** (sidebar):
   - Earnings Press Release (PDF)
   - Earnings Presentation (PDF)

3. **Click "Analyze Earnings"**

4. **View Results**:
   - Earnings Call Summary Table
   - Key Financial Metrics Table
   - Price Impact Analysis
   - Data Verification Report

5. **Download** (optional):
   - Markdown report
   - Verification text file

---

## 🏗️ System Architecture

### Multi-Agent RAG Pipeline

```
PDF Upload
    ↓
Document Processing (Docling)
    ├── Markdown conversion
    └── Table structure preservation
    ↓
Chunking (1500 chars, 200 overlap)
    ↓
Hybrid Retriever
    ├── BM25 (40%): Keyword matching
    └── Vector Search (60%): Semantic similarity
    ↓
Research Agent (GPT-5)
    └── Extract metrics with structured output
    ↓
Market Data Tools (yfinance)
    └── Fetch analyst estimates
    ↓
Verification Agent (GPT-5)
    └── Cross-check accuracy
    ↓
Conditional Routing
    ├── If verified → Generate Report
    └── If issues → Re-extract
    ↓
Report Generator
    └── Markdown tables & analysis
```

### Components

#### 1. **Document Processor** (`document_processor/`)
- Uses Docling for PDF → Markdown conversion
- Preserves table structures
- SHA-256 caching (7-day expiration)
- Chunk size: 1500 characters

#### 2. **Retriever Builder** (`retriever/`)
- Hybrid approach: BM25 + Vector Search
- OpenAI `text-embedding-3-small` embeddings
- ChromaDB vector store
- Retrieval K: 20 documents

#### 3. **AI Agents** (`agents/`)
- **Research Agent**: Extracts metrics using GPT-5
- **Verification Agent**: Validates extracted data
- **Workflow**: LangGraph orchestration with conditional routing

#### 4. **Tools** (`tools/`)
- **Market Data**: `fetch_street_estimates()`, `fetch_stock_price()`
- **Calculations**: Unit normalization, surprise %, YoY growth

#### 5. **Web App** (`app.py`)
- Streamlit interface
- File uploads
- Result display
- Download options

---

## 📁 Project Structure

```
earnings-main/
├── app.py                      # Streamlit web application
├── config/                     # Configuration
│   ├── settings.py             # Central config (models, tokens, etc.)
│   ├── constants.py            # Financial metrics, units, GAAP types
│   └── __init__.py
├── document_processor/         # PDF processing
│   ├── financial_document_processor.py
│   └── __init__.py
├── retriever/                  # Hybrid retrieval
│   ├── financial_retriever_builder.py
│   └── __init__.py
├── agents/                     # AI agents
│   ├── financial_research_agent.py
│   ├── financial_verification_agent.py
│   ├── financial_workflow.py
│   └── __init__.py
├── tools/                      # LangChain tools
│   ├── market_data_tools.py
│   ├── calculation_tools.py
│   └── __init__.py
├── requirements.txt            # Python dependencies
├── .env_example                # Environment template
├── test_imports.py             # Diagnostic script
├── MIGRATION_SUMMARY.md        # Detailed migration docs
└── README.md                   # This file
```

---

## 🛠️ Technical Details

### Models
- **GPT-5** (`gpt-5`): Extraction and verification
- **Embeddings**: `text-embedding-3-small`

### Configuration
- Chunk size: 1500 characters (optimized for financial tables)
- Chunk overlap: 200 characters
- Retrieval K: 20 documents
- BM25 weight: 0.4
- Vector weight: 0.6

### Token Limits
- Research Agent: 2500 tokens
- Verification Agent: 1500 tokens
- (Conservative limits account for GPT-5's internal reasoning tokens)

### Caching
- Location: `~/.cache/earnings_rag/`
- Cache key: SHA-256 of PDF content
- Expiration: 7 days

---

## 🔧 Configuration

Edit `config/settings.py` to customize:
- Model selection
- Token limits
- Chunk size and overlap
- Retrieval parameters
- Cache settings

---

## 🚨 Troubleshooting

### Import Errors
```bash
# Install/upgrade packages
pip install -r requirements.txt --upgrade

# Run diagnostic script
python test_imports.py
```

### API Key Not Found
```bash
# Check .env file exists
ls -la .env

# Test key loading
python -c "from dotenv import load_dotenv; import os; load_dotenv(); print('Found' if os.getenv('OPENAI_API_KEY') else 'Missing')"
```

### Docling Installation Issues (macOS)
```bash
brew install libxml2 libxmlsec1
pip install docling --no-cache-dir
```

### OpenSSL Warning
If you see urllib3 OpenSSL warnings, they're informational and don't affect functionality.

---

## 📈 Comparison: Before vs After

| Feature | Before (Claude API) | After (RAG System) |
|---------|--------------------|--------------------|
| **Processing** | Base64 encoding | Docling PDF parsing |
| **Retrieval** | None (direct context) | Hybrid BM25 + Vector |
| **Agents** | Single-shot | Multi-agent workflow |
| **Verification** | Manual review | Automated cross-checking |
| **Caching** | None | SHA-256 with expiry |
| **Tables** | May break | Preserved |
| **Accuracy** | 93.8% baseline | Targeted to match/exceed |

---

## 📊 Output Examples

### Earnings Call Summary
| Metric | Reported | Estimated | Surprise |
|--------|----------|-----------|----------|
| EPS | $5.16 | $5.10 | +1.18% |
| Revenue | $35.08B | $33.16B | +5.79% |

### Key Financial Metrics
| Metric | Current Quarter | Prior Year | YoY Growth |
|--------|----------------|------------|------------|
| REVENUE | $35.08B | $18.12B | +93.60% |
| NET_INCOME | $19.31B | $11.92B | +62.00% |

---

## 📝 Requirements

- Python 3.9+
- OpenAI API key with GPT-5 access
- ~5GB disk space for dependencies
- 8GB+ RAM recommended

---

## 📚 Documentation

- **MIGRATION_SUMMARY.md**: Complete migration details, architecture decisions, troubleshooting
- **test_imports.py**: Diagnostic script for setup verification
- **.env_example**: Environment variable template

---

## 🤝 Contributors

- **Ziqi Shao**: ML method development
- **Zhixiao Wu**: Method refinement, dataset collection, evaluation
- **Mingze Yuan**: LLM method development

---

## 📄 License

This project was developed as part of the CS7180: Special Topics in Generative AI course at Northeastern University.

---

## ⚠️ Disclaimer

This AI-powered financial analysis tool is for informational and educational purposes only. It should not be considered financial advice. Always conduct your own research and consult with qualified financial professionals before making investment decisions.

---

## 🎯 Future Enhancements

- [ ] Support for annual reports and 10-K filings
- [ ] Multi-company batch processing
- [ ] Historical performance tracking
- [ ] Sector-specific analysis customizations
- [ ] Export to Excel/PDF formats
- [ ] API endpoints for programmatic access
- [ ] Enhanced price prediction models
- [ ] Real-time earnings call transcription

---

**Built with:** OpenAI GPT-5 • LangChain • LangGraph • Docling • Streamlit • ChromaDB • yfinance

**Migration Date:** 2025-11-05
