"""
FinSight AI - Earnings Analysis with Multi-Agent RAG System
Main Streamlit Application
"""

import os
import streamlit as st
import tempfile
import logging
from pathlib import Path

# Add project root to path
import sys
sys.path.insert(0, str(Path(__file__).parent))

from document_processor.financial_document_processor import FinancialDocumentProcessor
from retriever.financial_retriever_builder import FinancialRetrieverBuilder
from agents.financial_workflow import FinancialWorkflow
from config.settings import settings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="FinSight AI - Earnings Analyzer",
    page_icon="📊",
    layout="wide"
)

# Custom CSS (preserved from original)
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .stButton>button {
        width: 100%;
        background-color: #1E88E5;
        color: white;
        font-size: 1.1rem;
        padding: 0.5rem 1rem;
        border-radius: 8px;
    }
    .earnings-report {
        background-color: #f8f9fa;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    .verification-box {
        background-color: #e8f5e9;
        padding: 15px;
        border-radius: 8px;
        border-left: 4px solid #4caf50;
        margin: 10px 0;
    }
    table {
        width: 100%;
        border-collapse: collapse;
        margin: 15px 0;
    }
    th {
        background-color: #1E88E5;
        color: white;
        padding: 12px;
        text-align: left;
    }
    td {
        padding: 10px;
        border-bottom: 1px solid #ddd;
    }
    tr:hover {
        background-color: #f5f5f5;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def initialize_system():
    """Initialize the RAG system components."""
    logger.info("Initializing FinSight AI RAG system...")

    try:
        processor = FinancialDocumentProcessor()
        retriever_builder = FinancialRetrieverBuilder()
        workflow = FinancialWorkflow()

        logger.info("✅ System initialized successfully")
        return processor, retriever_builder, workflow

    except Exception as e:
        logger.error(f"❌ System initialization failed: {e}")
        st.error(f"System initialization error: {str(e)}")
        st.stop()


def save_uploaded_file(uploaded_file):
    """Save uploaded file to temporary location."""
    try:
        # Create temporary file
        suffix = Path(uploaded_file.name).suffix
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            return tmp_file.name
    except Exception as e:
        logger.error(f"Error saving file: {e}")
        raise


def process_earnings_analysis(
    processor, retriever_builder, workflow,
    ticker, company_name, press_release_file, presentation_file
):
    """Process earnings analysis using RAG system."""

    try:
        # Save uploaded files
        with st.spinner("📄 Processing documents..."):
            pr_path = save_uploaded_file(press_release_file)
            pres_path = save_uploaded_file(presentation_file)

            logger.info(f"Processing documents for {company_name} ({ticker})")

            # Process documents (with caching)
            all_chunks = processor.process_all_documents(pr_path, pres_path)
            st.success(f"✅ Processed {len(all_chunks)} document chunks")

        # Build retriever
        with st.spinner("🔍 Building retrieval system..."):
            retriever = retriever_builder.build_hybrid_retriever(all_chunks)
            st.success("✅ Retrieval system ready")

        # Run analysis workflow
        with st.spinner("🤖 Running multi-agent analysis..."):
            result = workflow.run_analysis(
                ticker=ticker,
                company_name=company_name,
                retriever=retriever
            )
            st.success("✅ Analysis complete!")

        # Cleanup temporary files
        os.unlink(pr_path)
        os.unlink(pres_path)

        return result

    except Exception as e:
        logger.error(f"Analysis error: {e}", exc_info=True)
        raise


def main():
    """Main application."""

    # Header
    st.markdown('<div class="main-header">📊 FinSight AI</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">Automated Earnings Analysis with Multi-Agent RAG</div>',
        unsafe_allow_html=True
    )

    # Initialize system
    processor, retriever_builder, workflow = initialize_system()

    # Sidebar for inputs
    with st.sidebar:
        st.header("📋 Company Information")

        ticker = st.text_input(
            "Stock Ticker",
            value="NVDA",
            help="Enter the stock ticker symbol (e.g., NVDA, AAPL)"
        )

        company_name = st.text_input(
            "Company Name",
            value="NVIDIA Corporation",
            help="Enter the company name"
        )

        st.header("📁 Upload Documents")

        press_release = st.file_uploader(
            "Press Release PDF",
            type=["pdf"],
            help="Upload the earnings press release PDF"
        )

        presentation = st.file_uploader(
            "Earnings Presentation PDF",
            type=["pdf"],
            help="Upload the earnings presentation PDF"
        )

        st.divider()

        analyze_button = st.button(
            "🚀 Analyze Earnings",
            type="primary",
            use_container_width=True
        )

    # Main content area
    if analyze_button:
        if not ticker or not company_name:
            st.error("⚠️ Please provide both ticker and company name")
            return

        if not press_release or not presentation:
            st.error("⚠️ Please upload both PDF documents")
            return

        try:
            # Run analysis
            result = process_earnings_analysis(
                processor, retriever_builder, workflow,
                ticker, company_name, press_release, presentation
            )

            # Display results
            st.markdown("---")

            # Earnings Report
            col1, col2 = st.columns([3, 1])

            with col1:
                st.markdown("### 📈 Earnings Analysis Report")
                st.markdown(
                    f'<div class="earnings-report">{result["final_report"]}</div>',
                    unsafe_allow_html=True
                )

            with col2:
                st.markdown("### ✅ Verification")
                st.markdown(
                    f'<div class="verification-box"><pre>{result["verification_report"]}</pre></div>',
                    unsafe_allow_html=True
                )

            # Download options
            st.markdown("---")
            st.markdown("### 💾 Download Results")

            col1, col2 = st.columns(2)

            with col1:
                st.download_button(
                    label="📄 Download Report (Markdown)",
                    data=result["final_report"],
                    file_name=f"{ticker}_earnings_report.md",
                    mime="text/markdown"
                )

            with col2:
                st.download_button(
                    label="✅ Download Verification",
                    data=result["verification_report"],
                    file_name=f"{ticker}_verification.txt",
                    mime="text/plain"
                )

        except Exception as e:
            st.error(f"❌ Analysis failed: {str(e)}")
            logger.error(f"Analysis error: {e}", exc_info=True)

            with st.expander("🐛 Error Details"):
                st.code(str(e))

    else:
        # Welcome message
        st.info("""
        👋 **Welcome to FinSight AI!**

        This application uses a Multi-Agent RAG system to analyze earnings reports:

        **How it works:**
        1. 📄 Upload earnings documents (press release + presentation)
        2. 🔍 Documents are processed and chunked using Docling
        3. 🤖 Multiple AI agents extract and verify financial metrics
        4. 📊 Comprehensive earnings report is generated

        **Features:**
        - ✅ Automated metric extraction (EPS, Revenue, etc.)
        - ✅ Verification against source documents
        - ✅ Market data integration (analyst estimates)
        - ✅ Year-over-year comparisons
        - ✅ Surprise percentage calculations

        **Get started by filling in the sidebar and clicking "Analyze Earnings"!**
        """)

        # System info
        with st.expander("ℹ️ System Information"):
            st.markdown(f"""
            **RAG System Configuration:**
            - Document Processor: Docling (PDF → Markdown)
            - Retrieval: Hybrid (BM25 + Vector Search)
            - Embeddings: {settings.EMBEDDING_MODEL}
            - Research Agent: {settings.GPT5_MODEL}
            - Verification Agent: {settings.GPT5_MODEL}
            - Chunk Size: {settings.CHUNK_SIZE} characters
            - Retrieval K: {settings.RETRIEVAL_K} chunks
            """)


if __name__ == "__main__":
    main()
