# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project: FinSight AI - Multi-Agent RAG Earnings Analyzer

This is a financial analysis system that uses Multi-Agent RAG (Retrieval-Augmented Generation) to analyze corporate earnings reports. It extracts financial metrics from PDF documents, verifies accuracy, and generates comprehensive analysis reports.

---

## Development Commands

### Setup
```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies (5-10 min due to PyTorch, transformers)
pip install -r requirements.txt

# Configure environment
cp .env_example .env
# Edit .env to add OPENAI_API_KEY
```

### Running the Application
```bash
# Start Streamlit app (main interface)
streamlit run app.py

# Or use venv Python directly
venv/bin/streamlit run app.py

# App runs at http://localhost:8501
```

### Testing
```bash
# Test all imports and component initialization
python test_imports.py

# Or with venv
venv/bin/python test_imports.py

# Expected: All ✅ checks passing
```

### Dependency Management
```bash
# Install single missing package
venv/bin/pip install <package-name>

# Update requirements.txt after adding packages
venv/bin/pip freeze > requirements.txt

# Important: rank-bm25 must be included for BM25 retriever
```

---

## Architecture Overview

### Multi-Agent RAG Pipeline

The system uses a **stateful LangGraph workflow** with conditional routing:

```
PDF Upload
    ↓
FinancialDocumentProcessor (Docling) → Markdown + chunking (1500 chars)
    ↓
FinancialRetrieverBuilder → Hybrid (BM25 40% + Vector 60%)
    ↓
FinancialWorkflow (LangGraph orchestration):
    1. Research Agent → Extract metrics (GPT-5)
    2. Market Data Tools → Fetch estimates (yfinance)
    3. Verification Agent → Cross-check (GPT-5)
    4. Conditional → Re-extract if verification fails
    5. Report Generator → Markdown tables
```

### Key Architectural Patterns

#### 1. **Document Processing with Table Preservation**
- **Location**: `document_processor/financial_document_processor.py`
- **Pattern**: Docling PDF → Markdown conversion with SHA-256 caching
- **Critical**: `do_table_structure=True` preserves financial tables during chunking
- **Chunk size**: 1500 chars (larger than typical 1000) for complete table rows
- **Cache**: `~/.cache/earnings_rag/` with 7-day expiration

#### 2. **Hybrid Retrieval System**
- **Location**: `retriever/financial_retriever_builder.py`
- **Pattern**: `EnsembleRetriever` combining BM25 and Vector Search
- **Weights**: BM25 (0.4) for exact keyword matches, Vector (0.6) for semantic similarity
- **Rationale**: BM25 catches exact metric mentions ("Q3 2024 EPS"), Vector handles semantic queries
- **Embeddings**: OpenAI `text-embedding-3-small`
- **Vector Store**: ChromaDB (ephemeral, rebuilt per analysis)
- **K=20**: Retrieves 20 chunks to ensure comprehensive context

#### 3. **LangGraph Workflow with Conditional Routing**
- **Location**: `agents/financial_workflow.py`
- **Pattern**: State-based workflow using `StateGraph` and `FinancialAgentState` TypedDict
- **State Management**: All data (metrics, estimates, verification) flows through shared state
- **Conditional Logic**: `_decide_after_verification()` routes to re-extraction if accuracy fails
- **Critical**: Maximum 1 re-extraction attempt to prevent infinite loops

#### 4. **Unit Normalization (CRITICAL)**
- **Location**: `tools/calculation_tools.py`
- **Pattern**: LangChain `@tool` decorated functions
- **Purpose**: Prevent calculation errors when mixing millions (M) and billions (B)
- **Functions**:
  - `normalize_to_millions()`: Converts all values to millions for comparison
  - `calculate_surprise_percentage()`: **ALWAYS normalizes both reported and expected before calculating**
- **Example**: If reported=5.16B and expected=5100M, normalize both to millions first
- **Importance**: This was a critical accuracy issue in the original system (93.8% baseline)

#### 5. **GPT-5 Responses API Integration**
- **Locations**: `agents/financial_research_agent.py`, `agents/financial_verification_agent.py`
- **Pattern**: OpenAI Responses API (not Chat Completions)
- **Token Limits**: Research (2500), Verification (1500) - conservative due to internal reasoning tokens
- **Output**: JSON-structured responses parsed with `json.loads(response.output_text)`
- **Prompts**: Detailed system prompts in `_create_extraction_prompt()` methods

---

## Configuration & Settings

### Central Configuration
- **Location**: `config/settings.py`
- **Pattern**: Singleton `Settings` class with class-level attributes
- **Usage**: `from config.settings import settings`
- **Key Settings**:
  - `GPT5_MODEL = "gpt-5"`
  - `CHUNK_SIZE = 1500` (financial table optimization)
  - `RETRIEVAL_K = 20`
  - `RESEARCH_AGENT_MAX_TOKENS = 2500`

### Financial Constants
- **Location**: `config/constants.py`
- **Contains**:
  - `FINANCIAL_METRICS`: List of extractable metrics (EPS, Revenue, etc.)
  - `UNIT_MULTIPLIERS`: {"M": 1, "B": 1000, "K": 0.001} for normalization
  - `GAAP_TYPES`: ["GAAP", "Non-GAAP", "Adjusted"]

### Environment Variables
- **File**: `.env` (not committed)
- **Required**: `OPENAI_API_KEY`
- **Optional**: `LANGSMITH_API_KEY`, `LANGSMITH_TRACING`, `LANGSMITH_PROJECT`

---

## Critical Code Patterns

### 1. Adding New Financial Metrics
When adding metrics to extract:
1. Add to `config/constants.py` → `FINANCIAL_METRICS` list
2. Update `agents/financial_research_agent.py` → `extract_all_metrics()` to include new query
3. Update `agents/financial_workflow.py` → `_generate_financials_table()` for display
4. **Important**: Ensure prompt specifies units and GAAP classification

### 2. Modifying Token Limits
If GPT-5 responses are truncated:
1. Edit `config/settings.py` → `RESEARCH_AGENT_MAX_TOKENS` or `VERIFICATION_AGENT_MAX_TOKENS`
2. **Note**: GPT-5 uses internal reasoning tokens, so output may be less than max_output_tokens
3. Learned from docchat: 50→200 was too low, 2500/1500 works well

### 3. Adjusting Retrieval Weights
To change BM25 vs Vector balance:
1. Edit `retriever/financial_retriever_builder.py` → `build_hybrid_retriever()`
2. Default: `weights=[0.4, 0.6]` (BM25, Vector)
3. More keyword focus: increase BM25 weight (e.g., [0.6, 0.4])
4. More semantic focus: increase Vector weight (e.g., [0.3, 0.7])

### 4. Handling Verification Failures
The workflow automatically re-extracts once if verification fails:
1. `agents/financial_verification_agent.py` → Sets `needs_reextraction=True` if issues found
2. `agents/financial_workflow.py` → `_decide_after_verification()` routes back to extraction
3. Maximum 1 retry (state tracks `needs_reextraction` to prevent loops)

### 5. Caching Behavior
Document processing is cached to avoid re-processing identical PDFs:
1. Cache key: SHA-256 hash of PDF bytes
2. Cache location: `~/.cache/earnings_rag/converted/`
3. Expiration: 7 days
4. **To clear cache**: Delete cache directory or modify `CACHE_EXPIRY_DAYS`

---

## Data Flow Details

### Input → Output Flow
1. **User uploads** press release PDF + presentation PDF via Streamlit
2. **app.py** saves to temp files, calls `processor.process_all_documents()`
3. **FinancialDocumentProcessor** converts PDFs → Markdown → chunks (cached)
4. **FinancialRetrieverBuilder** creates BM25 + Vector retriever from chunks
5. **FinancialWorkflow.run_analysis()** executes LangGraph workflow:
   - State initialized with ticker, company_name, retriever
   - Research Agent extracts metrics from retrieved chunks
   - Market data tools fetch analyst estimates
   - Verification Agent cross-checks extracted vs source
   - Report generator creates Markdown tables
6. **app.py** displays final_report and verification_report in Streamlit

### State Object Structure
```python
FinancialAgentState = {
    "ticker": str,                    # e.g., "NVDA"
    "company_name": str,              # e.g., "NVIDIA Corporation"
    "retriever": EnsembleRetriever,   # Hybrid BM25+Vector
    "extracted_metrics": Dict,        # {metric_key: {value, unit, gaap, ...}}
    "analyst_estimates": Dict,        # {eps_estimate, revenue_estimate, ...}
    "verification_report": Dict,      # {overall_verified, discrepancies, ...}
    "final_report": str,              # Markdown report
    "needs_reextraction": bool        # Triggers conditional routing
}
```

---

## Important Files

### Entry Point
- **app.py**: Streamlit UI, main application entry point

### Core RAG Components
- **document_processor/financial_document_processor.py**: PDF processing with Docling
- **retriever/financial_retriever_builder.py**: Hybrid BM25+Vector retrieval
- **agents/financial_research_agent.py**: Metric extraction (GPT-5)
- **agents/financial_verification_agent.py**: Accuracy verification (GPT-5)
- **agents/financial_workflow.py**: LangGraph orchestration

### Tools & Utilities
- **tools/market_data_tools.py**: yfinance integration (LangChain @tool format)
- **tools/calculation_tools.py**: Unit normalization, surprise %, YoY growth

### Configuration
- **config/settings.py**: Central config (models, tokens, chunks)
- **config/constants.py**: Financial metrics, unit multipliers, GAAP types

### Testing & Diagnostics
- **test_imports.py**: Tests all imports and component initialization
- **MIGRATION_SUMMARY.md**: Detailed migration documentation from Claude API

---

## Common Issues & Solutions

### "Could not import rank_bm25"
- **Cause**: Missing BM25 retriever dependency
- **Fix**: `venv/bin/pip install rank-bm25`
- **Prevention**: Ensure `rank-bm25>=0.2.2` in requirements.txt

### "OPENAI_API_KEY not found"
- **Cause**: Missing or incorrect .env configuration
- **Fix**: Create `.env` file with `OPENAI_API_KEY=sk-...`
- **Test**: `python -c "from dotenv import load_dotenv; import os; load_dotenv(); print(os.getenv('OPENAI_API_KEY'))"`

### Empty or Truncated GPT-5 Responses
- **Cause**: `max_output_tokens` too low for reasoning + output
- **Fix**: Increase token limits in `config/settings.py`
- **Current**: 2500 (research), 1500 (verification) - proven to work

### Table Extraction Issues
- **Cause**: Tables split across chunks
- **Fix**: Increase `CHUNK_SIZE` in settings (currently 1500, optimized for tables)
- **Alternative**: Verify `do_table_structure=True` in document processor

### Verification Always Failing
- **Check**: Verify retriever is returning relevant chunks
- **Debug**: Add logging in `_verify_single_metric()` to see retrieved context
- **Potential**: Adjust retrieval K or weights

---

## Migration Context

This codebase was **recently migrated** (2025-11-05) from:
- **Old**: Claude Files API with direct PDF processing
- **New**: Multi-Agent RAG with Docling, hybrid retrieval, verification loops

### Legacy Code (DO NOT USE)
- `src/react_agent/claude_pdf_analyzer.py` - Old Claude API approach
- `src/react_agent/financial_analyst_app.py` - Old Streamlit app
- **Use**: `app.py` (new) and components in root-level directories

### Migration Rationale
1. **Table preservation**: Docling maintains financial table structure
2. **Verification**: Multi-agent approach catches extraction errors
3. **Retrieval**: Hybrid BM25+Vector improves context relevance
4. **Accuracy**: Target 93.8%+ baseline with automated verification

See `MIGRATION_SUMMARY.md` for complete migration details.

---

## Development Notes

### When Adding Features
- Use LangChain `@tool` decorator for new tools
- Add to workflow as new node or integrate into existing agents
- Update `FinancialAgentState` TypedDict if new state fields needed
- Test with `test_imports.py` before full app run

### When Modifying Prompts
- Prompts live in agent classes (`_create_extraction_prompt()` methods)
- Always specify output format (JSON structure)
- Include examples for complex extractions
- Test with sample earnings PDFs

### When Debugging
1. Check Streamlit logs for errors
2. Run `test_imports.py` to verify setup
3. Inspect `~/.cache/earnings_rag/` for cached documents
4. Enable LangSmith tracing in `.env` for detailed workflow logs

### Performance Optimization
- Document processing is cached (7 days)
- Vector embeddings are ephemeral (rebuilt per analysis)
- Consider persistent ChromaDB for batch processing scenarios
- GPT-5 Responses API is faster than Chat Completions

---

## Python Environment

- **Python Version**: 3.9+ (tested with 3.13)
- **Virtual Environment**: Required (venv)
- **Key Dependencies**:
  - openai>=2.7.1 (GPT-5 Responses API)
  - langchain>=0.3.0 (RAG framework)
  - langgraph>=0.2.0 (workflow orchestration)
  - docling>=2.0.0 (PDF processing)
  - chromadb>=0.5.0 (vector store)
  - streamlit==1.30.0 (web UI)
  - yfinance==0.2.35 (market data)
  - rank-bm25>=0.2.2 (BM25 retriever)

---

## Additional Resources

- **README.md**: User-facing documentation, quick start guide
- **MIGRATION_SUMMARY.md**: Technical migration details, troubleshooting
- **genai_project_report.pdf**: Original project requirements and design
