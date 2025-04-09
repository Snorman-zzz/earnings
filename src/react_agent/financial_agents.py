# financial_agents.py
from dotenv import load_dotenv
from typing import Annotated, Optional, TypedDict, List, Dict, Any
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.graph import StateGraph, END
from openai import OpenAI
import os
import yfinance as yf
from tenacity import retry, stop_after_attempt, wait_exponential
import json

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

class AgentState(TypedDict):
    messages: list[BaseMessage]
    step_count: int
    final_answer: Optional[str]
    vector_store_id: str

# Debug print function
def _print_state(state: AgentState, agent_name: str):
    """Print state details for debugging"""
    print(f"\n{'='*40}")
    print(f"{agent_name.upper()} AGENT STATE (Step {state.get('step_count', 0)})")
    print(f"Vector Store ID: {state.get('vector_store_id', 'N/A')}")
    print("Messages:")
    for idx, msg in enumerate(state.get('messages', [])):
        content = msg.content if isinstance(msg.content, str) else str(msg.content)[:100]
        print(f"  [{idx}] {msg.type}: {content}...")
    print('='*40 + '\n')

# Core Financial Tools with debug prints
@tool
def fetch_street_estimates(ticker: str) -> dict:
    """Fetch analyst estimates from Yahoo Finance"""
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, max=10))
    def _fetch(ticker: str):
        print(f"\nFetching estimates for {ticker}")
        stock = yf.Ticker(ticker)
        info = stock.info
        return {
            "eps": info.get("epsCurrentYear", "N/A"),
            "revenue": info.get("revenueEstimate", "N/A") 
        }
    return _fetch(ticker)

@tool
def fetch_stock_price(ticker: str) -> float:
    """Get current stock price with debugging"""
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, max=10))
    def _fetch(ticker: str):
        print(f"\nFetching price for {ticker}")
        data = yf.Ticker(ticker).history(period="1d")
        return float(data["Close"].iloc[-1]) if not data.empty else 0.0
    return _fetch(ticker)

# Fixed file_search tool implementation
def file_search(query: str, vector_store_id: str) -> str:
    """Search documents using OpenAI file search with debug prints"""
    print(f"\nFILE SEARCH INITIATED")
    print(f"Query: {query}")
    print(f"Using Vector Store: {vector_store_id}")
    try:
        # Updated tool format for OpenAI API
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": query}],
            tools=[{
                "type": "file_search",
                "file_search": {
                    "vector_store_ids": [vector_store_id]
                }
            }]
        )
        content = response.choices[0].message.content
        print(f"Search Results Preview: {content[:200]}...")
        return content
    except Exception as e:
        print(f"Search Error: {str(e)}")
        return f"Search error: {str(e)}"

# Refactoring the agent creation approach
def create_custom_agent(llm, tools, system_message):
    """Create a custom agent that directly uses LLM for reasoning"""
    def agent_fn(state):
        # Extract messages from state
        messages = state.get("messages", [])
        
        # Add system message to the front
        system_msg = [("system", system_message)]
        
        # Create a prompt with the system message and user messages
        prompt = ChatPromptTemplate.from_messages(system_msg)
        
        # Prepare messages for the LLM
        formatted_messages = prompt.format_messages()
        formatted_messages.extend(messages)
        
        # Call the LLM
        response = llm.invoke(formatted_messages)
        
        # Add the response to messages
        messages.append(response)
        
        # Simple tool execution pattern
        content = response.content
        if "I need to use a tool" in content or "I'll use the" in content:
            for tool_name in ["fetch_street_estimates", "fetch_stock_price", "file_search"]:
                if tool_name in content:
                    # Extract arguments using a simple pattern
                    try:
                        args_start = content.find(tool_name) + len(tool_name)
                        args_text = content[args_start:].strip()
                        
                        # Extract the argument
                        start_idx = args_text.find("(")
                        end_idx = args_text.rfind(")")
                        
                        if start_idx != -1 and end_idx != -1:
                            arg = args_text[start_idx+1:end_idx].strip().strip('"\'')
                            print(f"Executing tool {tool_name} with arg: {arg}")
                            
                            # Execute the appropriate tool
                            if tool_name == "fetch_street_estimates":
                                result = fetch_street_estimates(arg)
                            elif tool_name == "fetch_stock_price":
                                result = fetch_stock_price(arg)
                            elif tool_name == "file_search":
                                result = file_search(arg, state.get("vector_store_id", ""))
                            
                            # Add tool result as a new message
                            messages.append(HumanMessage(content=f"Tool {tool_name} result: {result}"))
                            break
                    except Exception as e:
                        print(f"Error executing tool: {str(e)}")
                        messages.append(HumanMessage(content=f"Error executing tool {tool_name}: {str(e)}"))
        
        # Check if this is a final answer
        final_answer = None
        if "FINAL ANSWER:" in content:
            final_answer = content.split("FINAL ANSWER:")[1].strip()
        
        return {
            "messages": messages,
            "final_answer": final_answer
        }
    
    return agent_fn

# Research Agent - Updated for expert analysis
research_agent = create_custom_agent(
    ChatOpenAI(model="gpt-4o"),
    ["fetch_street_estimates", "fetch_stock_price"],
    """You are an expert financial analyst specializing in data extraction and analysis.

Your task is to extract key financial data from earnings documents with precision:

1. EXPECTED VALUES:
   - Expected EPS from analyst estimates
   - Expected quarterly revenue from analyst estimates 
   - Current stock price

2. REPORTED VALUES (use file_search to find these):
   - Reported EPS from the latest quarter
   - Reported quarterly revenue 
   - Other key quarterly financial metrics (net income, operating income, gross margin, etc.)
   - Year-over-Year (Y/Y) changes for all metrics

3. GUIDANCE (use file_search to find these):
   - Forward guidance for next quarter/year
   - Management commentary on growth trajectory
   - Any announced stock splits, dividends, or buybacks

Be extremely precise with numerical data, including exact figures with correct units (billions, millions, etc.).
Always calculate the surprise percentage for EPS and Revenue as: ((Reported - Expected) / Expected) * 100%

If you need to use a tool, clearly indicate: "I need to use a tool: [tool_name]([argument])".

When you have completed your research, provide your FINAL ANSWER: with all extracted data clearly organized.
"""
)

# Analyst Agent - Updated for expert report format
analyst_agent = create_custom_agent(
    ChatOpenAI(model="gpt-4o"),
    ["file_search"],
    """You are an expert financial analyst specializing in earnings report generation.

Based on the research data provided, create a professional earnings analysis with EXACTLY the following format:

1. **Two separate tables** in Markdown:
   * **Table 1: Earnings Calls** which compares:
      * Reported EPS vs. Expected EPS,
      * Reported Revenue vs. Expected Revenue,
      * and calculates the Surprise percentage.
   * **Table 2: Quarterly Financials** showing the quarterly financial metrics such as revenue, net income, diluted EPS, operating income, etc., with Year-over-Year (Y/Y) changes.

2. A brief explanation or summary below the tables.

3. A prediction for the appropriate post-earnings stock price based on guidance in the slides.
   * **Important**: Use **only LaTeX math with dollar signs** for any formulas. For example:
   $\\text{{Price Prediction}} = CurrentPrice \\times (1 + GrowthRate) = Result$
   * Do **not** use square brackets for LaTeX. Only use inline `$...$` or display `$$...$$` syntax.

Format the tables EXACTLY as:
- First table with header "### Earnings Calls" and columns "Metric | Expected | Reported | Surprise"
- Second table with header "### Financials" and columns "Metric | Current Quarter | Previous Year | Y/Y Change"

When you need to search for additional information, use: "I need to use a tool: file_search([query])".

When you have completed your report, provide your FINAL ANSWER: with the complete report formatted exactly as requested.
"""
)

# Wrapped agent invocations for debugging
def research_agent_invoke(state: AgentState):
    _print_state(state, "Research")
    
    # Extract the initial input data
    if len(state["messages"]) == 1 and state["messages"][0].type == "human":
        try:
            data = json.loads(state["messages"][0].content)
            # Create a more readable message for the agent
            ticker = data.get("market_data", {}).get("ticker")
            company = data.get("company")
            eps = data.get("market_data", {}).get("eps")
            price = data.get("market_data", {}).get("price")
            
            # Replace the JSON message with a more readable one
            state["messages"] = [HumanMessage(content=
                f"Analyze {company} (Ticker: {ticker}) with current EPS estimate of {eps} and stock price of ${price}. "
                f"Extract all financial metrics from the latest earnings documents using file_search tool. "
                f"Be very precise with numbers and calculations."
            )]
        except Exception as e:
            # If parsing fails, keep the original message
            print(f"Error parsing JSON: {str(e)}")
    
    result = research_agent(state)
    
    # Update the state with the result
    updated_state = {
        **state,
        "messages": result.get("messages", state.get("messages", [])),
        "step_count": state.get("step_count", 0) + 1
    }
    
    # Set final answer if available
    if result.get("final_answer"):
        updated_state["final_answer"] = result.get("final_answer")
    
    print("\nRESEARCH AGENT OUTPUT:")
    if updated_state["messages"]:
        latest_msg = updated_state["messages"][-1].content
        print(latest_msg[:200] + "..." if len(latest_msg) > 200 else latest_msg)
    
    return updated_state

def analyst_agent_invoke(state: AgentState):
    _print_state(state, "Analyst")
    
    # Format the request for the analyst agent
    if state["messages"] and state["messages"][-1].type == "ai":
        # Get the latest research result
        research_result = state["messages"][-1].content
        
        # Create a new request for the analyst
        analyst_request = (
            f"Based on the research findings below, create an expert financial analysis report "
            f"with the EXACT format specified in your instructions. Ensure you include two "
            f"properly formatted tables (Earnings Calls and Financials) and use LaTeX with dollar "
            f"signs for the price prediction formula.\n\n"
            f"RESEARCH FINDINGS:\n{research_result}"
        )
        
        # Replace the messages with just this request
        state["messages"] = [HumanMessage(content=analyst_request)]
    
    result = analyst_agent(state)
    
    # Update the state with the result
    updated_state = {
        **state,
        "messages": result.get("messages", state.get("messages", [])),
        "step_count": state.get("step_count", 0) + 1
    }
    
    # Set final answer if available
    if result.get("final_answer"):
        updated_state["final_answer"] = result.get("final_answer")
    
    print("\nANALYST AGENT OUTPUT:")
    if updated_state["messages"]:
        latest_msg = updated_state["messages"][-1].content
        print(latest_msg[:200] + "..." if len(latest_msg) > 200 else latest_msg)
    
    return updated_state

# Workflow Setup with state tracking
workflow = StateGraph(AgentState)
workflow.add_node("researcher", research_agent_invoke)
workflow.add_node("analyst", analyst_agent_invoke)

workflow.set_entry_point("researcher")
workflow.add_conditional_edges(
    "researcher",
    lambda s: "analyst" if not s.get("final_answer") else END
)
workflow.add_conditional_edges(
    "analyst",
    lambda s: END  # Always end after analyst
)

graph = workflow.compile()

__all__ = ['graph', 'fetch_street_estimates', 'fetch_stock_price', 'file_search']