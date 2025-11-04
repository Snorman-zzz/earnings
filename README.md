# FinSight AI - Automated Earnings Analysis & Prediction System

A comprehensive financial analysis platform combining Multi-Agent RAG (Retrieval-Augmented Generation) for earnings report analysis with machine learning evaluation frameworks for stock price prediction.

> **Academic Project** - CS7180: Special Topics in Generative AI
> Northeastern University

---

## 📋 Project Overview

This repository contains two integrated systems for automated financial analysis:

1. **Multi-Agent RAG Earnings Analyzer** (`earnings-main/`)
   - Processes earnings PDFs using AI agents
   - Extracts financial metrics with verification
   - Compares to analyst estimates
   - Generates comprehensive analysis reports

2. **Evaluation Framework** (`eval-main/`)
   - Historical S&P 500 earnings & price data (1980-2025)
   - Benchmark datasets for model comparison
   - Gradient boost baseline model
   - Performance metrics (direction & regression)

---

## 🎯 Key Features

### Earnings Analysis System
- ✅ **Multi-Agent Architecture**: Research Agent + Verification Agent with LangGraph orchestration
- ✅ **Hybrid Retrieval**: BM25 (keyword) + Vector Search (semantic) with ChromaDB
- ✅ **Table Preservation**: Docling PDF processing maintains financial table structures
- ✅ **Unit Normalization**: Handles millions vs billions conversions automatically
- ✅ **GAAP Classification**: Distinguishes GAAP vs Non-GAAP figures
- ✅ **Automated Verification**: Cross-checks extracted metrics against source documents
- ✅ **Market Data Integration**: Real-time analyst estimates from Yahoo Finance

### Evaluation System
- ✅ **Historical Data**: 45 years of S&P 500 earnings and stock prices
- ✅ **Multiple Test Sets**: Symbol-based, time-based, and random sampling configurations
- ✅ **Baseline Models**: Gradient boost implementation for comparison
- ✅ **Comprehensive Metrics**: Direction accuracy and regression performance
- ✅ **Reproducible**: Standardized datasets and evaluation pipeline

---

## 🚀 Quick Start

### Prerequisites
- Python 3.9+
- OpenAI API key (GPT-5 access)
- ~5GB disk space for dependencies
- 8GB+ RAM recommended

### 1. Install Earnings Analyzer

```bash
cd earnings-main

# Setup environment
python3 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt

# Configure API key
cp .env_example .env
# Edit .env and add: OPENAI_API_KEY="sk-..."

# Run application
streamlit run app.py
```

Open `http://localhost:8501` to use the web interface.

### 2. Run Evaluation Framework

```bash
cd eval-main

# Evaluate predictions
python evaluate.py

# Train gradient boost model
python gradient_boost.py

# Regenerate datasets
python dataset.py
```

---

## 📁 Repository Structure

```
fs/
├── README.md                    # This file
├── genai_project_report.pdf     # Project requirements & design
│
├── earnings-main/               # Multi-Agent RAG Earnings Analyzer
│   ├── app.py                   # Streamlit web application
│   ├── agents/                  # AI agents (Research, Verification, Workflow)
│   ├── config/                  # Settings and constants
│   ├── document_processor/      # Docling PDF processing
│   ├── retriever/               # Hybrid BM25 + Vector retrieval
│   ├── tools/                   # Market data & calculation tools
│   ├── test_imports.py          # Diagnostic testing
│   ├── CLAUDE.md                # Developer guide for Claude Code
│   ├── MIGRATION_SUMMARY.md     # Technical migration docs
│   └── README.md                # Component documentation
│
└── eval-main/                   # Evaluation Framework
    ├── data/
    │   ├── combined_data.csv    # Historical earnings & prices
    │   ├── test/                # Test datasets (5 configurations)
    │   ├── predictions/         # Baseline predictions
    │   └── evaluation/          # Metrics results
    ├── dataset.py               # Dataset generation
    ├── evaluate.py              # Evaluation pipeline
    ├── gradient_boost.py        # ML baseline model
    └── README.md                # Evaluation docs
```

---

## 🏗️ System Architecture

### Multi-Agent RAG Pipeline

```
User Uploads PDFs
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
    └── Extract financial metrics
    ↓
Market Data Tools (yfinance)
    └── Fetch analyst estimates
    ↓
Verification Agent (GPT-5)
    └── Cross-check accuracy
    ↓
Conditional Routing
    ├── If verified → Generate Report
    └── If issues → Re-extract metrics
    ↓
Final Report Display
    ├── Earnings analysis tables
    ├── Verification report
    └── Download options
```

### Technology Stack

**AI & ML:**
- OpenAI GPT-5 (gpt-5) - Metric extraction & verification
- OpenAI text-embedding-3-small - Vector embeddings
- LangChain - RAG framework
- LangGraph - Workflow orchestration
- ChromaDB - Vector database
- Scikit-learn - Gradient boost models

**Document Processing:**
- Docling - PDF → Markdown with table preservation
- pypdf - PDF utilities

**Data & Analysis:**
- yfinance - Market data & analyst estimates
- pandas - Data manipulation
- numpy - Numerical computing

**Web Interface:**
- Streamlit - Interactive UI

---

## 📊 Evaluation Datasets

### Test Dataset Configurations

1. **by_symbol_random_10.csv**
   - Random 10 records per stock
   - Stocks: NVDA, GOOGL, AMZN, AAPL, MSFT, META, TSLA
   - Use case: Per-stock performance analysis

2. **by_symbol_time_10.csv**
   - Latest 10 records per stock
   - Same tech stocks as above
   - Use case: Recent trend analysis

3. **by_random_100.csv**
   - Random sample of 100 earnings events
   - Use case: Quick baseline testing

4. **by_time_100.csv**
   - Latest 100 earnings events
   - Use case: Current market conditions

5. **by_random_1000.csv**
   - Random sample of 1000 events
   - Use case: Comprehensive evaluation

### Metrics Evaluated

**Direction Metrics:**
- Accuracy of price movement direction prediction
- Precision, recall, F1-score

**Regression Metrics:**
- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- R² Score

---

## 🔧 Configuration

### Earnings Analyzer Settings

Edit `earnings-main/config/settings.py`:

```python
GPT5_MODEL = "gpt-5"                    # AI model
EMBEDDING_MODEL = "text-embedding-3-small"
CHUNK_SIZE = 1500                       # Optimized for tables
CHUNK_OVERLAP = 200
RETRIEVAL_K = 20                        # Chunks to retrieve
RESEARCH_AGENT_MAX_TOKENS = 2500
VERIFICATION_AGENT_MAX_TOKENS = 1500
```

### Financial Constants

Edit `earnings-main/config/constants.py`:

```python
FINANCIAL_METRICS = [
    "EPS", "Revenue", "Operating Income",
    "Net Income", "Gross Margin", "Operating Margin",
    "Free Cash Flow"
]

UNIT_MULTIPLIERS = {"M": 1, "B": 1000, "K": 0.001}
GAAP_TYPES = ["GAAP", "Non-GAAP", "Adjusted"]
```

---

## 📖 Usage Examples

### Analyzing Earnings Reports

1. Navigate to `http://localhost:8501`
2. Enter company information:
   - Ticker: `NVDA`
   - Name: `NVIDIA Corporation`
3. Upload PDF files:
   - Earnings press release
   - Earnings presentation
4. Click "🚀 Analyze Earnings"
5. Review results:
   - Earnings call summary table
   - Financial metrics comparison
   - Verification report
6. Download Markdown report

### Running Evaluations

```bash
cd eval-main

# Evaluate all test datasets
python evaluate.py

# Results saved to data/evaluation/
# - direction_metrics_*.csv
# - regression_metrics_*.csv
```

### Training Custom Models

```bash
cd eval-main

# Train gradient boost model
python gradient_boost.py

# Predictions saved to data/predictions/
```

---

## 🔍 Key Technical Patterns

### 1. Unit Normalization (Critical)

The system handles mixed units (millions vs billions) in financial reports:

```python
# tools/calculation_tools.py
@tool
def calculate_surprise_percentage(reported, expected,
                                  reported_unit="M",
                                  expected_unit="M"):
    # Normalize both to millions first
    reported_m = normalize_to_millions(reported, reported_unit)
    expected_m = normalize_to_millions(expected, expected_unit)

    surprise = ((reported_m - expected_m) / abs(expected_m)) * 100
    return round(surprise, 2)
```

**Why critical:** Prevents calculation errors when reported values are in billions but estimates are in millions.

### 2. Conditional Re-extraction

LangGraph workflow includes verification with automatic retry:

```python
# agents/financial_workflow.py
workflow.add_conditional_edges(
    "verify_metrics",
    self._decide_after_verification,
    {
        "reextract": "extract_metrics",  # Retry if verification fails
        "generate": "generate_report",    # Continue if verified
    }
)
```

Maximum 1 retry to prevent infinite loops.

### 3. Document Caching

SHA-256 based caching with 7-day expiration:

```python
# document_processor/financial_document_processor.py
cache_key = hashlib.sha256(pdf_bytes).hexdigest()
cache_path = CACHE_DIR / f"{cache_key}.md"

if cache_path.exists() and not _is_cache_expired(cache_path):
    return _load_from_cache(cache_path)
```

Avoids re-processing identical PDFs.

---

## 🚨 Troubleshooting

### Earnings Analyzer Issues

**"Could not import rank_bm25"**
```bash
cd earnings-main
source venv/bin/activate
pip install rank-bm25
```

**"OPENAI_API_KEY not found"**
```bash
cd earnings-main
ls -la .env  # Check file exists
# Edit .env and add: OPENAI_API_KEY="sk-..."
```

**Empty GPT-5 responses**
- Increase token limits in `config/settings.py`
- Current: 2500 (research), 1500 (verification)

**Table extraction issues**
- Verify `CHUNK_SIZE = 1500` in settings
- Check Docling has `do_table_structure=True`

### Evaluation Framework Issues

**Missing data files**
```bash
cd eval-main
python dataset.py  # Regenerate datasets
```

**Import errors**
```bash
pip install pandas numpy scikit-learn
```

---

## 📚 Documentation

### Main Documentation
- **README.md** (this file) - Project overview
- **genai_project_report.pdf** - Original requirements & design

### Component Documentation
- **earnings-main/README.md** - Earnings analyzer guide
- **earnings-main/CLAUDE.md** - Developer guide for Claude Code
- **earnings-main/MIGRATION_SUMMARY.md** - Migration details
- **eval-main/README.md** - Evaluation framework guide

### Testing & Diagnostics
- **earnings-main/test_imports.py** - Component initialization tests

---

## 🤝 Contributors

- **Ziqi Shao** - ML method development
- **Zhixiao Wu** - Method refinement, dataset collection, evaluation
- **Mingze Yuan** - LLM method development

---

## 📄 License

This project was developed as part of the CS7180: Special Topics in Generative AI course at Northeastern University.

---

## ⚠️ Disclaimer

This AI-powered financial analysis tool is for **informational and educational purposes only**. It should not be considered financial advice. Always conduct your own research and consult with qualified financial professionals before making investment decisions.

---

## 🎯 Project Achievements

- ✅ **Multi-Agent RAG System**: Successfully migrated from Claude Files API to comprehensive RAG architecture
- ✅ **Table Preservation**: Docling maintains financial table integrity during processing
- ✅ **Automated Verification**: Reduces manual checking with AI-powered cross-validation
- ✅ **Unit Normalization**: Prevents calculation errors in earnings surprise calculations
- ✅ **Comprehensive Evaluation**: 45 years of S&P 500 data for rigorous testing
- ✅ **Baseline Comparison**: Gradient boost model for performance benchmarking
- ✅ **Production Ready**: Streamlit interface with caching and error handling
- ✅ **Well Documented**: CLAUDE.md for future development, extensive README files

---

## 🔗 Resources

- **Repository**: https://github.com/Snorman-zzz/interview-prep-finsight
- **OpenAI GPT-5**: https://openai.com/
- **LangChain**: https://python.langchain.com/
- **LangGraph**: https://langchain-ai.github.io/langgraph/
- **Docling**: https://github.com/DS4SD/docling
- **Streamlit**: https://streamlit.io/

---

**Built with:** OpenAI GPT-5 • LangChain • LangGraph • Docling • Streamlit • ChromaDB • yfinance

**Last Updated:** 2025-11-05
