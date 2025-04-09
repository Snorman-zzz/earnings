# financial_analyst_app.py
import os
import streamlit as st
from financial_agents import graph, fetch_street_estimates, fetch_stock_price, file_search
from langchain_core.messages import HumanMessage
import json
from openai import OpenAI
import time
import re

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

st.title("Earnings Analysis & Post-Earnings Stock Price Predictor")

# =======================
# Document Processing
# =======================
def create_vector_store(press_release, presentation):
    """Create vector store with debug prints"""
    try:
        print("\n=== DOCUMENT PROCESSING STARTED ===")
        print(f"Press Release File: {press_release.name}")
        print(f"Presentation File: {presentation.name}")
        
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
        
        return vector_store.id
    
    except Exception as e:
        st.error(f"Document processing failed: {str(e)}")
        print(f"\n!!! DOCUMENT PROCESSING ERROR !!!\n{str(e)}")
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
    """Run analysis with detailed debug prints"""
    try:
        print("\n=== AGENT ANALYSIS INIT ===")
        print(f"Company: {company}")
        print(f"Vector Store ID: {vector_store_id}")
        print("Market Data:")
        print(json.dumps(market_data, indent=2))
        
        initial_state = {
            "messages": [HumanMessage(content=json.dumps({
                "company": company,
                "market_data": market_data,
                "vector_store_id": vector_store_id
            }))],
            "step_count": 0,
            "final_answer": None,
            "vector_store_id": vector_store_id
        }

        print("\nINITIAL STATE:")
        print(f"Step Count: {initial_state['step_count']}")
        print("Initial Message Content:")
        print(initial_state["messages"][0].content[:500] + "...")
        
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

# =======================
# Extract Tables
# =======================
def extract_tables(markdown_text):
    """Extracts markdown tables from the text"""
    tables = {}
    
    # Match headers and their corresponding tables
    pattern = r'### ([^\n]+)\s*\n\s*\|([^\n]+)\|\s*\n\s*\|[-:\s\|]+\|\s*\n((?:\|[^\n]+\|\s*\n)*)'
    
    for match in re.finditer(pattern, markdown_text):
        header = match.group(1).strip()
        table_header = match.group(2).strip()
        table_content = match.group(3).strip()
        
        # Combine header and content
        full_table = f"| {table_header} |\n|{''.join(['-' if c != '|' else '|' for c in '|' + table_header + '|'])}|\n{table_content}"
        tables[header] = full_table
    
    return tables

# =======================
# Extract LaTeX Formula
# =======================
def extract_latex_formula(markdown_text):
    """Extracts LaTeX formula for price prediction"""
    # Look for inline LaTeX
    inline_pattern = r'\$([^$]+)\$'
    inline_matches = re.findall(inline_pattern, markdown_text)
    
    # Look for display LaTeX
    display_pattern = r'\$\$([^$]+)\$\$'
    display_matches = re.findall(display_pattern, markdown_text)
    
    # Combine matches with priority to display math
    all_matches = display_matches + inline_matches
    
    # Return the first formula that mentions price or prediction
    for formula in all_matches:
        if 'price' in formula.lower() or 'prediction' in formula.lower():
            return f"${formula}$"
    
    # If no specific price formula found, return the first formula if any
    return f"${all_matches[0]}$" if all_matches else ""

# =======================
# Extract Summary
# =======================
def extract_summary(markdown_text):
    """Extracts summary text between tables and price prediction"""
    # First, remove all tables
    no_tables = re.sub(r'### [^\n]+\s*\n\s*\|[^\n]+\|\s*\n\s*\|[-:\s\|]+\|\s*\n((?:\|[^\n]+\|\s*\n)*)', '', markdown_text)
    
    # Now remove LaTeX formulas
    no_latex = re.sub(r'\$\$[^$]+\$\$|\$[^$]+\$', '', no_tables)
    
    # Now look for paragraphs that might be summaries (not headers)
    paragraphs = [p.strip() for p in no_latex.split('\n\n') if p.strip() and not p.strip().startswith('#')]
    
    # Join all remaining paragraphs
    return '\n\n'.join(paragraphs)

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
        st.markdown(summary)
    
    # Extract and display price prediction
    latex_formula = extract_latex_formula(analysis_text)
    if latex_formula:
        st.markdown("### Post-Earnings Price Prediction")
        st.latex(latex_formula.strip('$'))
    
    # Show full analysis in expander
    with st.expander("View Full Analysis"):
        st.markdown(analysis_text)

st.markdown("---")
st.caption("This AI-powered financial analysis is for informational purposes only and should not be considered financial advice.")