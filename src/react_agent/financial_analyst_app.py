import os
import streamlit as st
import yfinance as yf
import time
import anthropic
import base64
import re
from tenacity import retry, stop_after_attempt, wait_exponential
from dotenv import load_dotenv
from .rag_pdf_analyzer import RAGPDFAnalyzer

# Load environment variables
load_dotenv()


# Initialize RAG PDF Analyzer
@st.cache_resource
def get_rag_analyzer():
    """Initialize RAG PDF Analyzer with error handling."""
    try:
        return RAGPDFAnalyzer()
    except Exception as e:
        st.error(f"Failed to initialize RAG analyzer: {str(e)}")
        st.error("Make sure OPENROUTER_API_KEY is set in your environment variables")
        st.stop()

# Initialize Anthropic client (keeping as fallback)
@st.cache_resource
def get_anthropic_client():
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        st.error("ANTHROPIC_API_KEY not found in environment variables")
        st.stop()
    return anthropic.Anthropic(api_key=api_key)


# =======================
# Market Data Handling
# =======================
@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, max=10))
def get_market_data(ticker: str):
    """Fetch market data with debug prints"""
    try:
        print(f"\n=== FETCHING MARKET DATA ===")
        # Use yfinance to get estimates
        stock = yf.Ticker(ticker)
        info = stock.info
        price = stock.history(period="1d")["Close"].iloc[-1] if not stock.history(period="1d").empty else 0.0

        return {
            "eps": info.get("epsCurrentYear", "N/A"),
            "revenue": info.get("revenueEstimate", "N/A"),
            "price": f"{price:.2f}" if isinstance(price, float) else 'N/A',
            "ticker": ticker
        }
    except Exception as e:
        st.error(f"Market data error: {str(e)}")
        print(f"\n!!! MARKET DATA ERROR !!!\n{str(e)}")
        return None


# =======================
# Claude PDF Analysis
# =======================
def encode_file(file_content):
    """Encode file content to base64."""
    return base64.b64encode(file_content).decode("utf-8")


def analyze_earnings_documents(rag_analyzer, press_release_bytes, presentation_bytes, ticker, company_name, market_data):
    """Analyze earnings documents using RAG pipeline."""
    try:
        print("\n=== RAG ANALYSIS STARTED ===")
        start_time = time.time()

        print("Starting RAG analysis...")
        
        # Use the RAG analyzer to process the documents
        results = rag_analyzer.analyze_earnings_documents(
            press_release=press_release_bytes,
            presentation=presentation_bytes,
            ticker=ticker,
            company_name=company_name,
            market_data=market_data
        )
        
        if "error" in results:
            print(f"\n!!! RAG ANALYSIS ERROR !!!\n{results['error']}")
            return {"error": f"Error analyzing documents with RAG: {results['error']}"}
        
        print(f"RAG analysis completed in {time.time() - start_time:.2f} seconds")
        
        # Log retrieval information if available
        if "retrieval_info" in results:
            info = results["retrieval_info"]
            print(f"Processed {info.get('total_nodes', 'unknown')} nodes")
            print(f"Used retrievers: {', '.join(info.get('retrievers_used', []))}")
        
        return {"full_analysis": results["full_analysis"]}

    except Exception as e:
        print(f"\n!!! RAG ANALYSIS ERROR !!!\n{str(e)}")
        return {"error": f"Error analyzing documents with RAG: {str(e)}"}


# =======================
# Add Custom CSS for Better Styling
# =======================
def add_custom_styling():
    """Add custom CSS for better styling"""
    st.markdown("""
    <style>
    /* Improve table styling */
    .styled-table {
        width: 100%;
        border-collapse: collapse;
        margin-bottom: 20px;
        font-size: 15px;
    }

    .styled-table th {
        background-color: #f1f8ff;
        font-weight: 600;
        text-align: left;
        padding: 12px 15px;
        border-bottom: 2px solid #ddd;
    }

    .styled-table td {
        padding: 10px 15px;
        border-bottom: 1px solid #ddd;
    }

    .styled-table tr:nth-child(even) {
        background-color: #f8f9fa;
    }

    /* Add styling for the analysis section */
    .earnings-analysis {
        padding: 20px;
        border-radius: 5px;
        border: 1px solid #e0e0e0;
        margin-top: 20px;
        margin-bottom: 30px;
        background-color: white;
    }

    /* Price prediction styling */
    .price-prediction {
        background-color: #f0f5ff;
        padding: 15px;
        border-radius: 5px;
        border-left: 3px solid #4b6cb7;
        margin-top: 20px;
        margin-bottom: 20px;
        font-weight: bold;
        font-size: 18px;
    }

    /* Section headers */
    .earnings-analysis h2 {
        color: #1a56db;
        padding-bottom: 10px;
        border-bottom: 1px solid #eaecef;
        margin-top: 30px;
        margin-bottom: 20px;
    }

    .earnings-analysis h3 {
        color: #2d3748;
        margin-top: 25px;
        margin-bottom: 15px;
    }

    /* Disclaimer text */
    .disclaimer {
        font-size: 12px;
        color: #6c757d;
        margin-top: 30px;
        padding-top: 10px;
        border-top: 1px solid #eaecef;
    }
    </style>
    """, unsafe_allow_html=True)


# =======================
# Main App Logic
# =======================
def main():
    st.title("RAG-Powered Earnings Analysis & Stock Price Predictor")

    # Add custom styling
    add_custom_styling()
    
    # Add information about RAG system
    st.info("🤖 This application now uses a Retrieval-Augmented Generation (RAG) pipeline instead of direct Claude API calls for improved document analysis and cost efficiency.")

    # =======================
    # UI Components
    # =======================
    st.sidebar.header("Required Documents")
    press_release = st.sidebar.file_uploader(
        "Upload Press Release (PDF)",
        type=["pdf"],
        help="Latest earnings press release document"
    )

    presentation = st.sidebar.file_uploader(
        "Upload Earnings Presentation (PDF)",
        type=["pdf"],
        help="Quarterly earnings presentation deck"
    )

    st.sidebar.header("Company Info")
    company_name = st.sidebar.text_input("Company Name", "NVIDIA Corporation")
    ticker_symbol = st.sidebar.text_input("Stock Ticker", "NVDA").strip().upper()

    # =======================
    # Main Execution
    # =======================
    if st.button("Generate Comprehensive Analysis"):
        # Validate inputs
        if not press_release or not presentation:
            st.error("Please upload both required documents")
            st.stop()

        if not ticker_symbol.isalpha() or len(ticker_symbol) > 5:
            st.error("Invalid ticker symbol format")
            st.stop()

        print("\n=== ANALYSIS JOB STARTED ===")

        with st.spinner("Analyzing market data..."):
            market_data = get_market_data(ticker_symbol)
            if not market_data:
                st.stop()

        # Initialize the RAG analyzer
        rag_analyzer = get_rag_analyzer()

        with st.spinner("Analyzing documents with RAG pipeline..."):
            # Read file contents
            press_release_bytes = press_release.getvalue()
            presentation_bytes = presentation.getvalue()

            # Call the RAG analyzer
            results = analyze_earnings_documents(
                rag_analyzer=rag_analyzer,
                press_release_bytes=press_release_bytes,
                presentation_bytes=presentation_bytes,
                ticker=ticker_symbol,
                company_name=company_name,
                market_data=market_data
            )

            if "error" in results:
                st.error(f"Analysis failed: {results['error']}")
                st.stop()

        # Display results
        st.header(f"{company_name} ({ticker_symbol}) Earnings Analysis")

        # Process markdown tables to ensure they render properly
        def fix_markdown_tables(content):
            # Look for table content with pipe characters
            table_pattern = r'\|(.*?)\|\s*\n'
            table_matches = re.findall(table_pattern, content, re.MULTILINE | re.DOTALL)

            if table_matches:
                # This content contains tables that need fixing

                # First, identify complete tables
                table_blocks = re.findall(r'(\|.*?\|.*?\n\|[-:|\s]+\|\n(?:\|.*?\|.*?\n)+)', content,
                                          re.MULTILINE | re.DOTALL)

                for table_block in table_blocks:
                    # Create a proper HTML table replacement
                    html_table = '<table class="styled-table">\n<thead>\n<tr>\n'

                    # Get the rows
                    rows = table_block.strip().split('\n')

                    # Process header row
                    header_cells = [cell.strip() for cell in rows[0].split('|') if cell.strip()]
                    for cell in header_cells:
                        html_table += f'<th>{cell}</th>\n'

                    html_table += '</tr>\n</thead>\n<tbody>\n'

                    # Skip the header and separator lines (rows[0] and rows[1])
                    for row in rows[2:]:
                        if '|' in row:  # Make sure it's a table row
                            html_table += '<tr>\n'
                            cells = [cell.strip() for cell in row.split('|') if cell]
                            for cell in cells:
                                html_table += f'<td>{cell}</td>\n'
                            html_table += '</tr>\n'

                    html_table += '</tbody>\n</table>'

                    # Replace the markdown table with the HTML table
                    content = content.replace(table_block, html_table)

            return content

        # Display the analysis
        if results.get("full_analysis"):
            analysis_content = results.get("full_analysis")

            # 1. Replace markdown headers with HTML headers
            analysis_content = re.sub(r'## ([^\n]+)', r'<h2>\1</h2>', analysis_content)
            analysis_content = re.sub(r'### ([^\n]+)', r'<h3>\1</h3>', analysis_content)

            # 2. Fix any tables in the content
            analysis_content = fix_markdown_tables(analysis_content)

            # 3. Format the price prediction section
            price_prediction_pattern = r'<h3>Price Prediction</h3>\s*(.+?)\s*(?=<h|$)'
            price_prediction_match = re.search(price_prediction_pattern, analysis_content, re.DOTALL)

            if price_prediction_match:
                price_text = price_prediction_match.group(1).strip()
                # Format the price prediction with proper styling
                formatted_price = f'<div class="price-prediction">{price_text}</div>'
                analysis_content = analysis_content.replace(price_prediction_match.group(0),
                                                            f'<h3>Price Prediction</h3>\n{formatted_price}\n\n')

            # Display the formatted analysis
            st.markdown(f'<div class="earnings-analysis">{analysis_content}</div>', unsafe_allow_html=True)
        else:
            st.error("No analysis was generated.")

    st.markdown("---")
    st.markdown(
        '<p class="disclaimer">This RAG-powered financial analysis is for informational purposes only and should not be considered financial advice. Results are generated using retrieval-augmented generation with multiple retriever strategies for comprehensive document analysis.</p>',
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()