import os
import streamlit as st
import yfinance as yf
import time
import anthropic
import base64
import re
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

        IMPORTANT: Your output MUST use HTML formatting for the entire response, including tables. Do NOT use markdown tables with pipe characters.

        Create your response with these EXACT sections:

        <h2>Earnings Summary</h2>

        <h3>Earnings Calls</h3>
        <table>
        <tr>
          <th>Metric</th>
          <th>Expected</th>
          <th>Reported</th>
          <th>Surprise</th>
        </tr>
        <tr>
          <td>EPS</td>
          <td>$X.XX</td>
          <td>$X.XX</td>
          <td>X.XX%</td>
        </tr>
        <tr>
          <td>Revenue</td>
          <td>$XX.XXB</td>
          <td>$XX.XXB</td>
          <td>X.XX%</td>
        </tr>
        </table>

        <h3>Financials</h3>
        <table>
        <tr>
          <th>Metric</th>
          <th>Current Quarter</th>
          <th>Previous Year</th>
          <th>Y/Y Change</th>
        </tr>
        <tr>
          <td>Revenue</td>
          <td>$XX.XXB</td>
          <td>$XX.XXB</td>
          <td>XX.XX%</td>
        </tr>
        <tr>
          <td>Net Income</td>
          <td>$X.XXB</td>
          <td>$X.XXB</td>
          <td>XX.XX%</td>
        </tr>
        <tr>
          <td>Diluted EPS</td>
          <td>$X.XX</td>
          <td>$X.XX</td>
          <td>XX.XX%</td>
        </tr>
        <tr>
          <td>Operating Income</td>
          <td>$X.XXB</td>
          <td>$X.XXB</td>
          <td>XX.XX%</td>
        </tr>
        <tr>
          <td>Gross Margin</td>
          <td>XX.X%</td>
          <td>XX.X%</td>
          <td>X.X pts</td>
        </tr>
        </table>

        <h3>Key Findings Summary</h3>
        <p>Write a concise summary of the key findings with proper sentences and spacing between words.</p>

        <h3>Price Prediction</h3>
        <p>Price Prediction = CurrentPrice × (1 + AdjustmentFactor) = NewPrice</p>

        For example:
        <p>Price Prediction = 110.93 × (1 + 0.10) = 122.02</p>

        # IMPORTANT UNIT NORMALIZATION INSTRUCTIONS:
        Before calculating any percentage changes or surprises, normalize units first:
        1. For values with different units (like "11.89B" vs "39.33M"), convert both to the same unit first
        2. Convert all values to the same unit (millions or billions) before calculating percentages
        3. For numbers with units (B for billions, M for millions), extract the number part and apply the scale:
           - 1B = 1000M (converting billions to millions)
           - 1M = 0.001B (converting millions to billions)

        # CRUCIAL FORMATTING REQUIREMENTS:
        1. Use proper spacing between all words and numbers (e.g., "39.33 billion" NOT "39.33billion")
        2. Insert spaces between each word in your text
        3. Format numbers consistently with proper units (e.g., "$13.51B" not "$13.51billion")
        4. Ensure proper spacing after each punctuation mark
        5. Maintain proper spacing between all words in sentences
        6. DO NOT use markdown tables with | characters - use proper HTML <table>, <tr>, <th>, <td> tags
        7. DO NOT use markdown for any part of your response - use HTML tags for everything
        """

        # Make the API request with the correct document format
        print("Sending request to Claude API...")
        response = client.messages.create(
            model="claude-3-7-sonnet-20250219",
            max_tokens=4000,
            temperature=0,
            system="""You are a financial analyst specialized in earnings report analysis with expertise in markdown formatting.

Your task is to analyze financial documents and present the results in perfectly formatted markdown.

CRITICALLY IMPORTANT: Your output must have proper spacing between ALL words. Never run words together.
Always insert spaces between words. Format all text as properly spaced plain text.

For example:
✓ "NVIDIA reported revenue of 39.33 billion for Q4 FY2025, representing a 78% increase."
✗ "NVIDIA reported revenueof 39.33billionforQ4FY2025, representinga78% increase."

Your analysis will be rendered directly in a web application, so proper markdown formatting is essential.
            """,
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

        return {"full_analysis": analysis_text}

    except Exception as e:
        print(f"\n!!! CLAUDE API ERROR !!!\n{str(e)}")
        return {"error": f"Error analyzing documents: {str(e)}"}


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
        '<p class="disclaimer">This AI-powered financial analysis is for informational purposes only and should not be considered financial advice.</p>',
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()