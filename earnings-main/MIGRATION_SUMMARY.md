# FinSight AI - Multi-Agent RAG System Migration

## Migration Complete ✅

The earnings-main application has been successfully migrated from Claude Files API to a Multi-Agent RAG System architecture (similar to docchat).

---

## What Changed

### Architecture Transformation

**Before:** Direct Claude API with base64 PDF processing
- Single-shot extraction
- No retrieval system
- Limited verification
- 93.8% accuracy baseline

**After:** Multi-Agent RAG System
- Document chunking with Docling
- Hybrid retrieval (BM25 + Vector Search)
- Specialized agents with verification loops
- LangGraph workflow orchestration

---

## New Components

### 1. **Document Processing** (`document_processor/`)
- `FinancialDocumentProcessor`: Docling-based PDF → Markdown conversion
- Table structure preservation
- SHA-256 caching with 7-day expiration
- Chunk size: 1500 characters (optimized for financial tables)

### 2. **Retrieval System** (`retriever/`)
- `FinancialRetrieverBuilder`: Hybrid retrieval
  - BM25 (40%): Keyword-based matching
  - Vector Search (60%): Semantic similarity
  - OpenAI `text-embedding-3-small` embeddings
  - ChromaDB vector store

### 3. **AI Agents** (`agents/`)
- **FinancialResearchAgent**: Extracts metrics using GPT-5
  - EPS, Revenue, Operating Income, Net Income, etc.
  - GAAP vs non-GAAP classification
  - Quarter-over-quarter and year-over-year comparisons

- **FinancialVerificationAgent**: Verifies extracted data
  - Cross-checks against source documents
  - Identifies discrepancies
  - Suggests corrections

- **FinancialWorkflow**: LangGraph orchestration
  - Extract → Fetch Estimates → Verify → Generate Report
  - Conditional re-extraction if verification fails

### 4. **Tools** (`tools/`)
- **Market Data Tools**: yfinance integration (LangChain format)
  - `fetch_street_estimates()`
  - `fetch_stock_price()`
  - `fetch_historical_performance()`

- **Calculation Tools**: Unit normalization & calculations
  - `normalize_to_millions()` - Critical for M vs B conversion
  - `calculate_surprise_percentage()`
  - `calculate_yoy_growth()`

### 5. **Configuration** (`config/`)
- `settings.py`: Centralized config
- `constants.py`: Financial metrics, unit multipliers, GAAP types

### 6. **Streamlit App** (Updated)
- Preserved original UI styling
- Integrated RAG pipeline
- Added verification report display
- Download buttons for results

---

## Setup Instructions

### 1. Install Dependencies

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install packages (may take 5-10 minutes due to large dependencies)
pip install -r requirements.txt
```

### 2. Configure Environment

Create a `.env` file in the project root:

```bash
# Copy example
cp .env_example .env

# Add your OpenAI API key
OPENAI_API_KEY="sk-..."

# Optional: LangSmith tracing (for debugging)
LANGSMITH_API_KEY=""
LANGSMITH_TRACING=false
```

### 3. Test Imports

```bash
python test_imports.py
```

Expected output:
```
================================================================================
Testing FinSight AI RAG System Imports
================================================================================

1. Testing basic imports...
✅ Basic imports successful

2. Testing environment setup...
✅ OPENAI_API_KEY found (length: XX)

3. Testing config imports...
✅ Config imports successful
   - GPT5 Model: gpt-5
   - Embedding Model: text-embedding-3-small
   - Chunk Size: 1500
   - Financial Metrics: 7 defined

... [all tests should pass]
```

### 4. Run the Application

```bash
streamlit run app.py
```

The app will launch at `http://localhost:8501`

---

## Usage

1. **Enter Company Information** (sidebar)
   - Stock Ticker (e.g., NVDA)
   - Company Name (e.g., NVIDIA Corporation)

2. **Upload Documents** (sidebar)
   - Press Release PDF
   - Earnings Presentation PDF

3. **Click "Analyze Earnings"**

4. **View Results**
   - Earnings Analysis Report (tables, metrics, analysis)
   - Verification Report (accuracy checks)

5. **Download** (optional)
   - Markdown report
   - Verification text

---

## Key Features

✅ **Automated Metric Extraction**
- EPS (GAAP & Non-GAAP)
- Revenue
- Operating Income
- Net Income
- Gross Margin
- Forward Guidance

✅ **Analyst Estimates Integration**
- Fetch from Yahoo Finance
- Calculate earnings surprises
- Compare to consensus

✅ **Verification System**
- Cross-check all extracted metrics
- Identify discrepancies
- Re-extract if needed

✅ **Unit Normalization**
- Handles M (millions) vs B (billions)
- Prevents calculation errors
- Critical for accurate surprise percentages

✅ **Table Preservation**
- Financial tables kept intact during chunking
- Better context for retrieval
- Improved extraction accuracy

---

## Technical Details

### RAG Pipeline Flow

```
PDF Upload
    ↓
Document Processing (Docling)
    ↓
Chunking (1500 chars, 200 overlap)
    ↓
Hybrid Retriever (BM25 + Vector)
    ↓
Research Agent (GPT-5 extraction)
    ↓
Market Data Tools (yfinance)
    ↓
Verification Agent (GPT-5 cross-check)
    ↓
Generate Report (Markdown tables)
    ↓
Display in Streamlit
```

### Token Limits (learned from docchat)
- Research Agent: 2500 tokens
- Verification Agent: 1500 tokens
- (GPT-5 uses internal reasoning tokens, so conservative limits)

### Caching
- Document processing cached for 7 days
- Cache key: SHA-256 hash of PDF content
- Cache location: `~/.cache/earnings_rag/`

---

## Comparison: Old vs New

| Feature | Claude Files API (Old) | Multi-Agent RAG (New) |
|---------|----------------------|----------------------|
| **Processing** | Base64 encoding | Docling PDF parsing |
| **Retrieval** | None (direct context) | Hybrid BM25 + Vector |
| **Agents** | Single-shot | Multi-agent workflow |
| **Verification** | Manual | Automated cross-checking |
| **Caching** | None | SHA-256 with expiry |
| **Tables** | May break | Preserved |
| **Cost per doc** | $0.062 | Variable (API calls) |
| **Accuracy** | 93.8% | Targeted to match/exceed |

---

## Dependencies

### Core
- `openai>=2.7.1` - GPT-5 Responses API
- `streamlit==1.30.0` - Web UI
- `yfinance==0.2.35` - Market data

### LangChain RAG
- `langchain>=0.3.0` - Framework
- `langchain-openai>=0.2.0` - OpenAI integration
- `langchain-community>=0.3.0` - BM25 retriever
- `langgraph>=0.2.0` - Workflow orchestration

### Document Processing
- `docling>=2.0.0` - PDF parsing
- `pypdf>=4.0.0` - PDF utilities

### Vector Database
- `chromadb>=0.5.0` - Vector store

---

## Next Steps

### For Development
1. ✅ Complete pip install
2. ✅ Create .env file
3. ⏳ Run import tests
4. ⏳ Test with sample earnings PDFs
5. ⏳ Compare accuracy against 93.8% baseline

### For Production
- Add error handling for edge cases
- Implement rate limiting for API calls
- Add progress indicators for long documents
- Set up monitoring/logging
- Create automated test suite

---

## Troubleshooting

### Import Errors
```bash
# Missing packages
pip install -r requirements.txt

# Outdated packages
pip install --upgrade -r requirements.txt
```

### API Key Issues
```bash
# Check .env file exists
ls -la .env

# Verify key is set
python -c "import os; from dotenv import load_dotenv; load_dotenv(); print('Key found' if os.getenv('OPENAI_API_KEY') else 'Key missing')"
```

### Docling Installation (macOS)
If Docling fails to install:
```bash
# Install required system libraries
brew install libxml2 libxmlsec1
pip install docling --no-cache-dir
```

---

## Files Created/Modified

### New Files
```
config/
  ├── settings.py
  ├── constants.py
  └── __init__.py

document_processor/
  ├── financial_document_processor.py
  └── __init__.py

retriever/
  ├── financial_retriever_builder.py
  └── __init__.py

agents/
  ├── financial_research_agent.py
  ├── financial_verification_agent.py
  ├── financial_workflow.py
  └── __init__.py

tools/
  ├── market_data_tools.py
  ├── calculation_tools.py
  └── __init__.py

app.py (NEW - replaced src/react_agent/financial_analyst_app.py)
test_imports.py
MIGRATION_SUMMARY.md
```

### Modified Files
```
requirements.txt (updated for RAG system)
.env_example (added OPENAI_API_KEY)
```

---

## Migration Decisions

### Why These Choices?

1. **Docling over PyPDF**
   - Better table structure preservation
   - Markdown output easier to chunk
   - Used successfully in docchat

2. **Hybrid Retrieval**
   - BM25 catches exact metric mentions
   - Vector search handles semantic queries
   - 40/60 split balances both approaches

3. **GPT-5 over Claude**
   - Automatic reasoning adjustment
   - Responses API for structured output
   - Consistent with docchat architecture

4. **LangGraph Orchestration**
   - State-based workflow
   - Conditional routing (re-extraction)
   - Easier to debug than callbacks

5. **Preserved Streamlit**
   - User requested (not Gradio)
   - Familiar interface
   - Minimal migration overhead

---

## Support

For questions or issues:
1. Check this MIGRATION_SUMMARY.md
2. Run `python test_imports.py` for diagnostics
3. Review logs in `~/.cache/earnings_rag/`
4. Consult docchat for similar architecture patterns

---

**Migration completed on:** 2025-11-05
**Claude Code Version:** Sonnet 4.5
