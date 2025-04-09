# financial_analyst_app.py (revised)
import os
import streamlit as st
from financial_agents import graph, fetch_street_estimates, fetch_stock_price, file_search
from langchain_core.messages import HumanMessage
import tempfile
import json
from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

st.title("Earnings Analysis & Post-Earnings Stock Price Predictor")

# =======================
# Document Processing with OpenAI
# =======================
def create_vector_store(uploaded_files):
    """Create vector store with uploaded files using OpenAI API"""
    try:
        vector_store = client.vector_stores.create(name="Earnings Analysis")
        file_ids = []
        
        for uploaded_file in uploaded_files:
            # Create a proper file tuple with filename and content
            file_tuple = (uploaded_file.name, uploaded_file.getvalue())
            
            # Upload file with proper filename
            response = client.files.create(
                file=file_tuple,
                purpose="assistants"
            )
            file_ids.append(response.id)
        
        # Wait for files to process (simplified version)
        import time
        time.sleep(10)  # Wait for file processing
        
        # Add files to vector store
        for file_id in file_ids:
            client.vector_stores.files.create(
                vector_store_id=vector_store.id,
                file_id=file_id
            )
            
        return vector_store.id
    
    except Exception as e:
        st.error(f"File processing failed: {str(e)}")
        return None

# =======================
# Updated Market Data Handling
# =======================
def get_market_data(ticker: str):
    with st.sidebar:
        with st.spinner("Fetching market data..."):
            try:
                estimates = fetch_street_estimates.invoke(ticker)
                price = fetch_stock_price.invoke(ticker)
                return {
                    "eps": estimates.get('eps', 'N/A'),
                    "revenue": estimates.get('revenue', 'N/A'),
                    "price": f"{price:.2f}" if isinstance(price, float) else 'N/A',
                    "ticker": ticker
                }
            except Exception as e:
                st.error(f"Data fetch error: {str(e)}")
                return None

# =======================
# Enhanced Analysis Execution
# =======================
def run_agent_analysis(vector_store_id: str, market_data: dict, company: str):
    """Execute agent workflow with vector store access"""
    initial_state = {
        "messages": [HumanMessage(content=json.dumps({
            "company": company,
            "market_data": market_data,
            "vector_store_id": vector_store_id
        }))],
        "step_count": 0,
        "api_calls": 0,
        "final_answer": None,
        "vector_store_id": vector_store_id
    }
    
    try:
        result = graph.invoke(initial_state)
        return result.get("final_answer", "No analysis generated")
    except Exception as e:
        return f"Analysis failed: {str(e)}"

# =======================
# Main UI Components
# =======================
st.sidebar.header("Input Files")
press_release = st.sidebar.file_uploader(
    "Press Release", 
    type=["pdf", "txt", "md"],
    accept_multiple_files=False
)
presentation = st.sidebar.file_uploader(
    "Earnings Presentation", 
    type=["pdf", "txt", "md"],
    accept_multiple_files=False
)

st.sidebar.header("Company Information")
company_name = st.sidebar.text_input("Company Name", "NVIDIA Corporation")
ticker_symbol = st.sidebar.text_input("Stock Ticker", "NVDA").strip().upper()

# =======================
# Analysis Trigger
# =======================
if st.button("Generate Analysis"):
    if not press_release or not presentation:
        st.error("Please upload both required documents")
        st.stop()
    
    with st.spinner("Processing documents..."):
        # Create vector store with OpenAI
        vector_store_id = create_vector_store([press_release, presentation])
        
        if not vector_store_id:
            st.error("Failed to process documents")
            st.stop()
            
        # Get market data
        market_data = get_market_data(ticker_symbol)
        
        if not market_data:
            st.error("Failed to fetch market data")
            st.stop()
            
        # Run agent workflow
        analysis = run_agent_analysis(
            vector_store_id=vector_store_id,
            market_data=market_data,
            company=company_name
        )
        
        # Display results
        st.markdown("## Analysis Results")
        if isinstance(analysis, dict):
            st.table(analysis.get("comparisons", []))
            st.latex(analysis.get("price_prediction", ""))
            st.markdown(analysis.get("narrative", ""))
        else:
            st.markdown(analysis)
