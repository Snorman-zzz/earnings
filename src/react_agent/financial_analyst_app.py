# financial_analyst_app.py
import os
import streamlit as st
import yfinance as yf
import time
import re
import anthropic
import base64
from tenacity import retry, stop_after_attempt, wait_exponential
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


# Initialize Anthropic client
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


def analyze_earnings_documents(client, press_release_bytes, presentation_bytes, ticker, company_name, market_data):
    """Analyze earnings documents using Claude API."""
    try:
        print("\n=== CLAUDE ANALYSIS STARTED ===")
        start_time = time.time()

        # Encode both PDFs
        press_release_b64 = encode_file(press_release_bytes)
        presentation_b64 = encode_file(presentation_bytes)
        print("PDFs encoded successfully")

        # Format market data if provided
        market_context = ""
        if market_data:
            market_context = f"""
            Current market data:
            - Expected EPS: {market_data.get('eps', 'N/A')}
            - Expected Revenue: {market_data.get('revenue', 'N/A')}
            - Current Stock Price: {market_data.get('price', 'N/A')}

            Calculate the surprise percentage for EPS and Revenue as: ((Reported - Expected) / Expected) * 100%
            """

        # Create the prompt
        prompt = f"""
        Please analyze these earnings documents for {company_name} ({ticker}).
        The first document is the press release and the second is the presentation.
        {market_context}

        Extract the following information with precise numerical values:
        1. Reported EPS and Revenue for the current quarter
        2. Other key financial metrics (net income, operating income, gross margin, etc.)
        3. Year-over-Year (Y/Y) changes for all metrics
        4. Forward guidance for next quarter/year if available
        5. Any announced stock splits, dividends, or buybacks

        Format your response with EXACTLY these sections:

        1. Earnings Calls Table comparing:
           - Metric | Expected | Reported | Surprise
           - Include rows for EPS and Revenue with precise values and proper units

        2. Financials Table showing:
           - Metric | Current Quarter | Previous Year | Y/Y Change
           - Include rows for Revenue, Net Income, Diluted EPS, Operating Income, etc.

        3. A concise summary of key findings, performance against expectations, and earnings quality

        4. A price prediction based on the earnings data, using this LaTeX format:
           $\\text{{Price Prediction}} = \\text{{CurrentPrice}} \\times (1 + \\text{{AdjustmentFactor}}) = \\text{{Result}}$

        # IMPORTANT UNIT NORMALIZATION INSTRUCTIONS:
        Before calculating any percentage changes or surprises, normalize units first:
        1. For values with different units (like "11.89B" vs "39.33M"), convert both to the same unit first
        2. Convert all values to the same unit (millions or billions) before calculating percentages
        3. If expected revenue is in billions and reported is in millions, convert both to millions before
           calculating the surprise percentage
        4. For numbers with units (B for billions, M for millions), extract the number part and apply the scale:
           - 1B = 1000M (converting billions to millions)
           - 1M = 0.001B (converting millions to billions)

        Ensure proper spacing between numbers and units (like "13.51B" not "13.51billion").
        Use consistent decimal precision (two decimal places) for all numerical values.
        """

        # Make the API request with the correct document format
        print("Sending request to Claude API...")
        response = client.messages.create(
            model="claude-3-7-sonnet-20250219",
            max_tokens=4000,
            temperature=0,
            system="You are a financial analyst specialized in earnings report analysis. Be extremely precise with numbers, calculations, and ensure consistent formatting of tables.",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "document",
                            "source": {
                                "type": "base64",
                                "media_type": "application/pdf",
                                "data": press_release_b64
                            }
                        },
                        {
                            "type": "document",
                            "source": {
                                "type": "base64",
                                "media_type": "application/pdf",
                                "data": presentation_b64
                            }
                        }
                    ]
                }
            ]
        )

        analysis_text = response.content[0].text
        print(f"Claude analysis completed in {time.time() - start_time:.2f} seconds")

        # Parse the analysis to extract structured data
        result = parse_analysis(analysis_text)
        result["full_analysis"] = analysis_text

        return result

    except Exception as e:
        print(f"\n!!! CLAUDE API ERROR !!!\n{str(e)}")
        return {"error": f"Error analyzing documents: {str(e)}"}


def parse_analysis(analysis_text):
    """Parse the analysis text to extract structured data."""
    result = {
        "earnings_table": None,
        "financials_table": None,
        "summary": None,
        "price_prediction": None,
        "latex_formula": None
    }

    # Extract tables
    tables = {}
    pattern = r'### ([^\n]+)\s*\n\s*\|([^\n]+)\|\s*\n\s*\|[-:\s\|]+\|\s*\n((?:\|[^\n]+\|\s*\n)*)'

    for match in re.finditer(pattern, analysis_text):
        header = match.group(1).strip()
        table_header = match.group(2).strip()
        table_content = match.group(3).strip()

        # Format the full table
        headers = [h.strip() for h in table_header.split('|') if h.strip()]
        separator = '|' + '|'.join(['-' * (len(h) + 2) for h in headers]) + '|'
        full_table = f"| {' | '.join(headers)} |\n{separator}\n{table_content}"
        tables[header] = full_table

    # Store tables in result
    if "Earnings Calls" in tables:
        result["earnings_table"] = tables["Earnings Calls"]
    if "Financials" in tables:
        result["financials_table"] = tables["Financials"]

    # Extract summary (text outside of tables and formulas)
    summary = re.sub(r'### [^\n]+\s*\n\s*\|[^\n]+\|\s*\n\s*\|[-:\s\|]+\|\s*\n((?:\|[^\n]+\|\s*\n)*)', '', analysis_text)
    summary = re.sub(r'\$\$[^$]+\$\$|\$[^$]+\$', '', summary)  # Remove LaTeX
    summary = re.sub(r'###[^\n]*\n', '', summary)  # Remove headers
    summary = re.sub(r'\n{3,}', '\n\n', summary)  # Normalize spacing
    result["summary"] = summary.strip()

    # Extract LaTeX formula for price prediction
    latex_pattern = r'\$([^$]+)\$'
    latex_matches = re.findall(latex_pattern, analysis_text)
    if latex_matches:
        formula = latex_matches[0]
        result["latex_formula"] = formula

        # Try to extract the price prediction result
        result_match = re.search(r'= (\d+\.\d+)$', formula)
        if result_match:
            result["price_prediction"] = float(result_match.group(1))

    return result


# =======================
# Add Custom CSS for Better Styling
# =======================
def add_custom_styling():
    """Add custom CSS for better styling"""
    st.markdown("""
    <style>
    /* Improve table styling */
    table {
        width: 100%;
        border-collapse: collapse;
        margin-bottom: 20px;
    }

    th {
        background-color: #f1f8ff;
        font-weight: 600;
        text-align: left;
        padding: 12px 15px;
        border-bottom: 2px solid #ddd;
    }

    td {
        padding: 10px 15px;
        border-bottom: 1px solid #ddd;
    }

    tr:nth-child(even) {
        background-color: #f8f9fa;
    }

    /* Add styling for the summary section */
    .summary-section {
        background-color: #f8f9fa;
        padding: 15px;
        border-radius: 5px;
        border-left: 3px solid #4b6cb7;
        margin-bottom: 20px;
    }

    /* Style for the prediction box */
    .prediction-box {
        background-color: #f0f5ff;
        padding: 15px;
        border-radius: 5px;
        border-left: 3px solid #4b6cb7;
        margin-top: 20px;
        margin-bottom: 20px;
    }
    </style>
    """, unsafe_allow_html=True)


# =======================
# Main App Logic
# =======================
def main():
    st.title("Earnings Analysis & Post-Earnings Stock Price Predictor")

    # Add custom styling
    add_custom_styling()

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

        # Initialize the Anthropic client
        client = get_anthropic_client()

        with st.spinner("Analyzing documents with Claude..."):
            # Read file contents
            press_release_bytes = press_release.getvalue()
            presentation_bytes = presentation.getvalue()

            # Call the Claude analyzer
            results = analyze_earnings_documents(
                client=client,
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
        st.subheader(f"{company_name} ({ticker_symbol}) Earnings Analysis")

        # Display tables
        if results.get("earnings_table"):
            st.markdown("### Earnings Calls")
            st.markdown(results["earnings_table"])

        if results.get("financials_table"):
            st.markdown("### Financials")
            st.markdown(results["financials_table"])

        # Display summary
        if results.get("summary"):
            st.markdown("### Analysis Summary")
            st.markdown(f'<div class="summary-section">{results["summary"]}</div>', unsafe_allow_html=True)

        # Display price prediction
        if results.get("latex_formula"):
            st.markdown('<div class="prediction-box">', unsafe_allow_html=True)
            st.markdown("### Post-Earnings Price Prediction")
            st.latex(results["latex_formula"])
            st.markdown('</div>', unsafe_allow_html=True)

        # Show full analysis in expander
        with st.expander("View Full Analysis"):
            st.markdown(results.get("full_analysis", "No full analysis available"))

    st.markdown("---")
    st.caption(
        "This AI-powered financial analysis is for informational purposes only and should not be considered financial advice.")


if __name__ == "__main__":
    main()