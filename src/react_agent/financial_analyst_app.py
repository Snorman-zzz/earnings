# financial_analyst_app.py
import os
import streamlit as st
from financial_agents import graph, fetch_street_estimates, fetch_stock_price, file_search
from langchain_core.messages import HumanMessage
import json
from openai import OpenAI
import time
import re
from pdf_data_extractor import PDFDataExtractor
from metrics_extractor import MetricsExtractor
from file_search import FileSearch

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


# =======================
# Document Processing
# =======================
def create_vector_store(press_release, presentation):
    """Create vector store with improved reliability"""
    try:
        print("\n=== DOCUMENT PROCESSING STARTED ===")
        print(f"Press Release File: {press_release.name}")
        print(f"Presentation File: {presentation.name}")

        # First, extract data from PDFs using extractor
        extractor = PDFDataExtractor()

        # Process both PDFs
        extraction_results = extractor.process_pdf_files([
            press_release.getvalue(),
            presentation.getvalue()
        ])

        # Additional metrics extraction
        metrics_extractor = MetricsExtractor()
        additional_metrics = metrics_extractor.extract_all_metrics(
            extraction_results.get("text", ""),
            extraction_results.get("tables", [])
        )

        # Merge additional metrics with extracted metrics
        for key, value in additional_metrics.items():
            if key not in extraction_results["metrics"] or not extraction_results["metrics"][key]:
                extraction_results["metrics"][key] = value

        print(
            f"\nExtracted {len(extraction_results.get('metrics', {}))} metrics and {len(extraction_results.get('tables', []))} tables")

        # Create the vector store
        vector_store = client.vector_stores.create(name="Earnings Analysis")
        print(f"\nVector Store Created - ID: {vector_store.id}")

        file_ids = []
        # Process press release
        pr_file = (press_release.name, press_release.getvalue())
        pr_response = client.files.create(file=pr_file, purpose="assistants")
        file_ids.append(pr_response.id)
        print(f"Uploaded Press Release - File ID: {pr_response.id}")

        # Process earnings presentation
        ep_file = (presentation.name, presentation.getvalue())
        ep_response = client.files.create(file=ep_file, purpose="assistants")
        file_ids.append(ep_response.id)
        print(f"Uploaded Presentation - File ID: {ep_response.id}")

        print("\nWaiting for file processing (15 seconds)...")
        time.sleep(15)

        for file_id in file_ids:
            client.vector_stores.files.create(
                vector_store_id=vector_store.id,
                file_id=file_id
            )
        print("Files added to vector store successfully")

        # Store the extracted data in session state for backup/fallback
        st.session_state.extracted_data = extraction_results
        st.session_state.extraction_successful = True

        return vector_store.id

    except Exception as e:
        st.error(f"Document processing failed: {str(e)}")
        print(f"\n!!! DOCUMENT PROCESSING ERROR !!!\n{str(e)}")
        st.session_state.extraction_successful = False
        return None


# =======================
# Market Data Handling
# =======================
def get_market_data(ticker: str):
    """Fetch market data with debug prints"""
    try:
        print("\n=== FETCHING MARKET DATA ===")
        estimates = fetch_street_estimates(ticker)
        price = fetch_stock_price(ticker)

        print(f"Ticker: {ticker}")
        print(f"EPS Estimate: {estimates.get('eps', 'N/A')}")
        print(f"Revenue Estimate: {estimates.get('revenue', 'N/A')}")
        print(f"Current Price: {price if isinstance(price, float) else 'N/A'}")

        return {
            "eps": estimates.get('eps', 'N/A'),
            "revenue": estimates.get('revenue', 'N/A'),
            "price": f"{price:.2f}" if isinstance(price, float) else 'N/A',
            "ticker": ticker
        }
    except Exception as e:
        st.error(f"Market data error: {str(e)}")
        print(f"\n!!! MARKET DATA ERROR !!!\n{str(e)}")
        return None


# =======================
# Analysis Workflow
# =======================
def run_agent_analysis(vector_store_id: str, market_data: dict, company: str):
    """Run analysis with fallback to extracted data"""
    try:
        print("\n=== AGENT ANALYSIS INIT ===")
        print(f"Company: {company}")
        print(f"Vector Store ID: {vector_store_id}")
        print("Market Data:")
        print(json.dumps(market_data, indent=2))

        # Add extraction data as backup
        extracted_data = {}
        if hasattr(st.session_state, 'extracted_data'):
            extracted_data = st.session_state.extracted_data
            print(f"Using extracted data as backup with {len(extracted_data.get('metrics', {}))} metrics")

        initial_state = {
            "messages": [HumanMessage(content=json.dumps({
                "company": company,
                "market_data": market_data,
                "vector_store_id": vector_store_id,
                "extracted_data": {
                    "metrics": extracted_data.get("metrics", {}),
                    "structured_data": extracted_data.get("structured_data", {})
                }
            }))],
            "step_count": 0,
            "final_answer": None,
            "vector_store_id": vector_store_id
        }

        print("\nINITIAL STATE:")
        print(f"Step Count: {initial_state['step_count']}")

        # Create a custom file_search function that falls back to extracted data
        if hasattr(st.session_state, 'extracted_data') and st.session_state.extracted_data:
            # Create improved search instance
            improved_search = FileSearch(client, st.session_state.extracted_data)

            # Monkey patch the file_search function in the global scope
            def patched_file_search(query, vector_store_id):
                return improved_search.search(query, vector_store_id)

            # Replace original file_search with patched version
            import builtins
            setattr(builtins, "file_search", patched_file_search)
            print("Improved file search function has been activated")

        # Run the agent with our patched search
        result = graph.invoke(initial_state)

        print("\n=== AGENT ANALYSIS RESULT ===")
        print(f"Total Steps Completed: {result.get('step_count', 0)}")

        # Get the final answer or use the last message content if no final answer
        final_answer = result.get("final_answer")
        if not final_answer and result.get("messages"):
            # Try to get the last AI message
            for msg in reversed(result.get("messages", [])):
                if msg.type == "ai":
                    final_answer = msg.content
                    break

        print("Final Answer Preview:")
        print((final_answer or "No answer generated")[:500] + "...")
        print("Vector Store ID Used:", result.get('vector_store_id', 'N/A'))

        return final_answer if final_answer else "Analysis incomplete"

    except Exception as e:
        error_msg = f"Analysis error: {str(e)}"
        print(f"\n!!! ANALYSIS ERROR !!!\n{error_msg}")
        return error_msg


def normalize_financial_unit(value_str):
    """
    Normalize financial values to a common unit (millions) for comparison.

    Args:
        value_str: String value like "11.89B" or "39.33M"

    Returns:
        float: Value converted to millions
    """
    if not value_str or not isinstance(value_str, str):
        return None

    # Extract the number and unit
    match = re.search(r'([\$]?)([\d\.,]+)\s*([BMKTbmkt])?', value_str)
    if not match:
        # Try alternative format with spelled out units
        match = re.search(r'([\$]?)([\d\.,]+)\s*(billion|million|thousand|trillion)', value_str, re.IGNORECASE)
        if not match:
            try:
                # Try to convert directly if it's just a number
                return float(value_str.replace('$', '').replace(',', '')) / 1000000  # Convert to millions
            except (ValueError, TypeError):
                return None

    # Extract components
    symbol = match.group(1)
    number = float(match.group(2).replace(',', ''))
    unit = match.group(3).lower() if match.group(3) else ''

    # Convert to millions based on unit
    if unit in ['b', 'billion']:
        return number * 1000  # Billions to millions
    elif unit in ['m', 'million']:
        return number  # Already in millions
    elif unit in ['k', 'thousand']:
        return number / 1000  # Thousands to millions
    elif unit in ['t', 'trillion']:
        return number * 1000000  # Trillions to millions
    else:
        # No unit specified, assume dollars/units and convert to millions
        return number / 1000000


def format_financial_value(value, unit='M', include_symbol=True, decimal_places=2):
    """
    Format a financial value with consistent units.

    Args:
        value: Numeric value to format
        unit: Target unit (M or B)
        include_symbol: Whether to include $ symbol
        decimal_places: Number of decimal places

    Returns:
        str: Formatted financial value
    """
    if value is None:
        return "N/A"

    # Convert to float if not already
    if isinstance(value, str):
        try:
            normalized = normalize_financial_unit(value)
            if normalized is None:
                return value  # Return original if we can't parse it
            value = normalized

            # Adjust unit based on size
            if unit == 'M' and value >= 1000:
                value = value / 1000
                unit = 'B'
        except (ValueError, TypeError):
            return value

    # Format the number with appropriate unit
    symbol = "$" if include_symbol else ""
    format_str = f"{symbol}{{:.{decimal_places}f}}{unit}"
    return format_str.format(value)


def calculate_percentage_change(new_value, old_value):
    """
    Calculate percentage change with unit normalization.

    Args:
        new_value: New value (string or number)
        old_value: Old value (string or number)

    Returns:
        float: Percentage change
    """
    # Normalize values to millions if they're strings
    if isinstance(new_value, str):
        new_normalized = normalize_financial_unit(new_value)
        if new_normalized is None:
            try:
                new_normalized = float(new_value.replace('$', '').replace('%', '').replace(',', ''))
            except (ValueError, TypeError):
                return None
    else:
        new_normalized = new_value

    if isinstance(old_value, str):
        old_normalized = normalize_financial_unit(old_value)
        if old_normalized is None:
            try:
                old_normalized = float(old_value.replace('$', '').replace('%', '').replace(',', ''))
            except (ValueError, TypeError):
                return None
    else:
        old_normalized = old_value

    # Calculate percentage change
    if old_normalized == 0:
        return None  # Avoid division by zero

    return ((new_normalized - old_normalized) / old_normalized) * 100


def extract_tables(markdown_text):
    """Extracts markdown tables from the text with improved formatting and unit normalization"""
    tables = {}

    # Match headers and their corresponding tables
    pattern = r'### ([^\n]+)\s*\n\s*\|([^\n]+)\|\s*\n\s*\|[-:\s\|]+\|\s*\n((?:\|[^\n]+\|\s*\n)*)'

    for match in re.finditer(pattern, markdown_text):
        header = match.group(1).strip()
        table_header = match.group(2).strip()
        table_content = match.group(3).strip()

        # Check if this is an Earnings Calls table with potential unit mismatches
        if header == 'Earnings Calls':
            # Process rows to fix surprise percentages
            fixed_rows = []
            rows = table_content.split('\n')

            for row in rows:
                cells = [cell.strip() for cell in row.split('|')[1:-1]]
                if len(cells) >= 4:
                    # Process revenue and EPS rows
                    if cells[0].lower() in ['revenue', 'eps']:
                        # Get expected and reported values
                        expected = cells[1]
                        reported = cells[2]

                        # Normalize to millions for comparison
                        expected_mm = normalize_financial_unit(expected)
                        reported_mm = normalize_financial_unit(reported)

                        if expected_mm is not None and reported_mm is not None:
                            # Recalculate surprise percentage
                            surprise_pct = ((reported_mm - expected_mm) / expected_mm) * 100
                            # Format to 2 decimal places
                            cells[3] = f"{surprise_pct:.2f}%"
                            # Rebuild the row
                            row = f"| {' | '.join(cells)} |"

                fixed_rows.append(row)

            # Rebuild the table content
            table_content = '\n'.join(fixed_rows)

        # Also fix Year-over-Year changes in the Financials table
        if header == 'Financials':
            fixed_rows = []
            rows = table_content.split('\n')

            for row in rows:
                cells = [cell.strip() for cell in row.split('|')[1:-1]]
                if len(cells) >= 4:
                    # Get current and previous values
                    current = cells[1]
                    previous = cells[2]

                    # Normalize units
                    current_mm = normalize_financial_unit(current)
                    previous_mm = normalize_financial_unit(previous)

                    if current_mm is not None and previous_mm is not None:
                        # Recalculate Y/Y percentage
                        yoy_pct = ((current_mm - previous_mm) / previous_mm) * 100
                        # Format to 2 decimal places
                        cells[3] = f"{yoy_pct:.2f}%"
                        # Rebuild the row
                        row = f"| {' | '.join(cells)} |"

                fixed_rows.append(row)

            # Rebuild the table content
            table_content = '\n'.join(fixed_rows)

        # Combine header and content with proper spacing
        headers = [h.strip() for h in table_header.split('|') if h.strip()]
        separator = '|' + '|'.join(['-' * (len(h) + 2) for h in headers]) + '|'

        # Format the full table
        full_table = f"| {' | '.join(headers)} |\n{separator}\n{table_content}"
        tables[header] = full_table

    # If we couldn't find tables with the pattern, try a more generic approach
    if not tables:
        # Find all table-like structures
        table_pattern = r'(\|[^\n]+\|\s*\n\s*\|[-:\s\|]+\|\s*\n(?:\|[^\n]+\|\s*\n)*)'
        table_matches = re.findall(table_pattern, markdown_text, re.DOTALL)

        for i, table in enumerate(table_matches):
            # Try to determine table type from content
            table_content = table.lower()
            if 'eps' in table_content or 'surprise' in table_content:
                tables['Earnings Calls'] = table.strip()
            elif 'quarter' in table_content or 'y/y' in table_content:
                tables['Financials'] = table.strip()
            else:
                tables[f'Table {i + 1}'] = table.strip()

    return tables


def extract_latex_formula(markdown_text):
    """Extracts LaTeX formula for price prediction with improved handling"""
    # Look for LaTeX formulas
    pattern = r'\$\$([^$]+)\$\$|\$([^$]+)\$'
    matches = re.findall(pattern, markdown_text)

    # Process all matches
    for display, inline in matches:
        formula = display if display else inline
        if formula and ('price' in formula.lower() or 'prediction' in formula.lower()):
            # Clean and format the formula
            # Ensure proper spacing around operators
            clean_formula = re.sub(r'([=×*+\-])', r' \1 ', formula)
            clean_formula = re.sub(r'\s+', ' ', clean_formula)  # Remove excess spaces

            return clean_formula

    # If no specific price formula found, return the first formula if any
    if matches:
        formula = matches[0][0] if matches[0][0] else matches[0][1]
        return formula

    return "\\text{Price Prediction} = \\text{CurrentPrice} \\times (1 + \\text{AdjustmentFactor})"


def extract_summary(markdown_text):
    """Extracts summary text with improved parsing"""
    # First, remove all tables
    no_tables = re.sub(r'### [^\n]+\s*\n\s*\|[^\n]+\|\s*\n\s*\|[-:\s\|]+\|\s*\n((?:\|[^\n]+\|\s*\n)*)', '',
                       markdown_text)

    # Now remove LaTeX formulas
    no_latex = re.sub(r'\$\$[^$]+\$\$|\$[^$]+\$', '', no_tables)

    # Remove headers
    no_headers = re.sub(r'###[^\n]*\n', '', no_latex)

    # Fix spacing issues in text
    fixed_text = re.sub(r'(\w)\.(\w)', r'\1. \2', no_headers)  # Add space after period
    fixed_text = re.sub(r'(\w),(\w)', r'\1, \2', fixed_text)  # Add space after comma
    fixed_text = re.sub(r'(\d+)(billion|million)', r'\1 \2', fixed_text)  # Add space between number and unit
    fixed_text = re.sub(r'(surpassing)the(expected)', r'\1 the \2', fixed_text)  # Fix specific spacing issues
    fixed_text = re.sub(r'against(the)', r'against \1', fixed_text)  # Fix specific spacing issues

    # Now look for paragraphs
    paragraphs = [p.strip() for p in fixed_text.split('\n\n') if p.strip()]

    # Join all remaining paragraphs
    return '\n\n'.join(paragraphs)


def fix_numerical_presentation(text):
    """Fix numerical presentation in outputs"""
    # Fix spacing between numbers and units
    text = re.sub(r'(\d+)([A-Za-z])', r'\1 \2', text)

    # Ensure consistent decimal places for percentages
    def format_percentage(match):
        try:
            value = float(match.group(1))
            return f"{value:.2f}%"
        except:
            return match.group(0)

    text = re.sub(r'(\d+\.\d+)\%', format_percentage, text)

    # Format dollar amounts consistently
    def format_dollar(match):
        try:
            value = float(match.group(1).replace(',', ''))
            if "B" in match.group(2):
                return f"${value:.2f}B"
            elif "M" in match.group(2):
                return f"${value:.2f}M"
            else:
                return f"${value:.2f}"
        except:
            return match.group(0)

    text = re.sub(r'\$([\d\.,]+)\s*([BM])', format_dollar, text)

    return text


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

        with st.spinner("Processing documents..."):
            vector_store_id = create_vector_store(press_release, presentation)
            if not vector_store_id:
                st.stop()

        with st.spinner("Analyzing market data..."):
            market_data = get_market_data(ticker_symbol)
            if not market_data:
                st.stop()

        with st.spinner("Running expert multi-agent analysis..."):
            analysis_text = run_agent_analysis(
                vector_store_id=vector_store_id,
                market_data=market_data,
                company=company_name
            )

        # Display results
        if not isinstance(analysis_text, str) or "error" in analysis_text.lower():
            st.error("Analysis failed to complete properly: " + str(analysis_text))
            st.stop()

        # Apply text fixes for better formatting
        analysis_text = fix_numerical_presentation(analysis_text)

        st.subheader(f"{company_name} ({ticker_symbol}) Earnings Analysis")

        # Extract and display tables
        tables = extract_tables(analysis_text)
        if "Earnings Calls" in tables:
            st.markdown("### Earnings Calls")
            st.markdown(tables["Earnings Calls"])

        if "Financials" in tables:
            st.markdown("### Financials")
            st.markdown(tables["Financials"])

        # Extract and display summary
        summary = extract_summary(analysis_text)
        if summary:
            st.markdown("### Analysis Summary")
            st.markdown(f'<div class="summary-section">{summary}</div>', unsafe_allow_html=True)

        # Extract and display price prediction
        latex_formula = extract_latex_formula(analysis_text)
        if latex_formula:
            st.markdown('<div class="prediction-box">', unsafe_allow_html=True)
            st.markdown("### Post-Earnings Price Prediction")
            st.latex(latex_formula)
            st.markdown('</div>', unsafe_allow_html=True)

        # Show full analysis in expander
        with st.expander("View Full Analysis"):
            st.markdown(analysis_text)

    st.markdown("---")
    st.caption(
        "This AI-powered financial analysis is for informational purposes only and should not be considered financial advice.")


if __name__ == "__main__":
    main()