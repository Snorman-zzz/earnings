# financial_agents.py (simplified)
from dotenv import load_dotenv
from typing import Annotated, Optional, TypedDict
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.graph import StateGraph
from langgraph.prebuilt import create_react_agent
from openai import OpenAI
import os
import yfinance as yf
from tenacity import retry, stop_after_attempt, wait_exponential

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

class AgentState(TypedDict):
    messages: list[BaseMessage]
    step_count: int
    final_answer: Optional[str]
    vector_store_id: str

# Core Financial Tools
@tool
def fetch_street_estimates(ticker: str) -> dict:
    """Fetch analyst estimates from Yahoo Finance"""
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, max=10))
    def _fetch(ticker: str):
        stock = yf.Ticker(ticker)
        info = stock.info
        return {
            "eps": info.get("epsCurrentYear", "N/A"),
            "revenue": info.get("revenueEstimate", "N/A")
        }
    return _fetch(ticker)

@tool
def fetch_stock_price(ticker: str) -> float:
    """Get current stock price"""
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, max=10))
    def _fetch(ticker: str):
        data = yf.Ticker(ticker).history(period="1d")
        return float(data["Close"].iloc[-1]) if not data.empty else 0.0
    return _fetch(ticker)

@tool
def file_search(query: Annotated[str, "Search query"], 
               vector_store_id: Annotated[str, "Vector store ID"]) -> str:
    """Search documents using OpenAI file search"""
    try:
        response = client.chat.completions.create(
            model="gpt-4-turbo",
            messages=[{"role": "user", "content": query}],
            tools=[{"type": "file_search", "vector_store_ids": [vector_store_id]}]
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Search error: {str(e)}"

# Agent Configuration
def create_agent(llm, tools, instructions):
    prompt = ChatPromptTemplate.from_messages([
        ("system", f"You are a financial analyst. {instructions}"),
        MessagesPlaceholder(variable_name="messages"),
        MessagesPlaceholder(variable_name="agent_scratchpad")
    ])
    return create_react_agent(llm, tools, prompt=prompt)

research_agent = create_agent(
    ChatOpenAI(model="gpt-4-turbo"),
    [fetch_street_estimates, fetch_stock_price],
    "Analyze market data and financial documents to generate insights."
)

analyst_agent = create_agent(
    ChatOpenAI(model="gpt-4-turbo"),
    [file_search],
    "Generate detailed reports with citations and financial analysis."
)

# Workflow Setup
workflow = StateGraph(AgentState)
workflow.add_node("researcher", research_agent.invoke)
workflow.add_node("analyst", analyst_agent.invoke)

workflow.set_entry_point("researcher")
workflow.add_conditional_edges("researcher", lambda s: "analyst" if not s.get("final_answer") else END)
workflow.add_conditional_edges("analyst", lambda s: "researcher" if not s.get("final_answer") else END)

graph = workflow.compile()

__all__ = ['graph', 'fetch_street_estimates', 'fetch_stock_price', 'file_search']
