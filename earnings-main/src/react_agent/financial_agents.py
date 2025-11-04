from dotenv import load_dotenv
from typing import Optional, TypedDict
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, END
import yfinance as yf
from tenacity import retry, stop_after_attempt, wait_exponential

load_dotenv()


# client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


class AgentState(TypedDict):
    messages: list[BaseMessage]
    step_count: int
    final_answer: Optional[str]
    # vector_store_id: str  # Commented out as part of removing file storage logic


# Debug print function
def _print_state(state: AgentState, agent_name: str):
    """Print state details for debugging"""
    print(f"\n{'=' * 40}")
    print(f"{agent_name.upper()} AGENT STATE (Step {state.get('step_count', 0)})")
    # print(f"Vector Store ID: {state.get('vector_store_id', 'N/A')}")  # Commented out
    print("Messages:")
    for idx, msg in enumerate(state.get('messages', [])):
        content = msg.content if isinstance(msg.content, str) else str(msg.content)[:100]
        print(f"  [{idx}] {msg.type}: {content}...")
    print('=' * 40 + '\n')


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


# Commented out the file_search tool implementation
"""
def file_search(query: str, vector_store_id: str) -> str:
    #Search documents using OpenAI file search with debug prints
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
"""


# Keep only the most necessary agent (the analyst agent) and comment out research agent
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
            for tool_name in ["fetch_street_estimates", "fetch_stock_price"]:  # Removed file_search
                if tool_name in content:
                    # Extract arguments using a simple pattern
                    try:
                        args_start = content.find(tool_name) + len(tool_name)
                        args_text = content[args_start:].strip()

                        # Extract the argument
                        start_idx = args_text.find("(")
                        end_idx = args_text.rfind(")")

                        if start_idx != -1 and end_idx != -1:
                            arg = args_text[start_idx + 1:end_idx].strip().strip('"\'')
                            print(f"Executing tool {tool_name} with arg: {arg}")

                            # Execute the appropriate tool
                            if tool_name == "fetch_street_estimates":
                                result = fetch_street_estimates(arg)
                            elif tool_name == "fetch_stock_price":
                                result = fetch_stock_price(arg)
                            # elif tool_name == "file_search":
                            #     result = file_search(arg, state.get("vector_store_id", ""))

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


# Commenting out the Research Agent to simplify the workflow
"""
research_agent = create_custom_agent(
    ChatOpenAI(model="gpt-4o"),
    ["fetch_street_estimates", "fetch_stock_price"],
    #...system message content...
)
"""

# Keep only the Analyst Agent with simplified tools
analyst_agent = create_custom_agent(
    ChatOpenAI(model="gpt-4o"),
    ["fetch_street_estimates", "fetch_stock_price"],  # Removed file_search
    """You are an expert financial analyst specializing in earnings report generation.

Based on the earnings documents for the company, create a professional earnings analysis with EXACTLY the following format:

1. **Two separate tables** in Markdown:
   * **Table 1: Earnings Calls** which compares:
      * Reported EPS vs. Expected EPS,
      * Reported Revenue vs. Expected Revenue,
      * and calculates the Surprise percentage.
   * **Table 2: Quarterly Financials** showing the quarterly financial metrics such as revenue, net income, diluted EPS, operating income, etc., with Year-over-Year (Y/Y) changes.

2. A brief explanation or summary below the tables.

3. A prediction for the appropriate post-earnings stock price based on guidance in the slides.
   * **Important**: Use **only LaTeX math with dollar signs** for any formulas. For example:
   $\\text{{Price Prediction}} = \\text{{CurrentPrice}} \\times (1 + \\text{{AdjustmentFactor}}) = \\text{{Result}}$
   * Do **not** use square brackets for LaTeX. Only use inline `$...$` or display `$$...$$` syntax.

# IMPORTANT UNIT NORMALIZATION INSTRUCTIONS:
Before calculating any percentage changes or surprises, you MUST normalize units first:
1. For values with different units (like "11.89B" vs "39.33M"), convert both to the same unit first
2. Convert all values to the same unit (millions or billions) before calculating percentages
3. If expected revenue is in billions and reported is in millions, convert both to millions before
   calculating the surprise percentage
4. For numbers with units (B for billions, M for millions), extract the number part and apply the scale:
   - 1B = 1000M (converting billions to millions)
   - 1M = 0.001B (converting millions to billions)

Surprise percentage formula with unit normalization:
1. Convert Expected and Reported values to the same unit (e.g., millions)
2. Calculate: ((Reported in millions - Expected in millions) / Expected in millions) * 100%

Format the tables EXACTLY as follows:

### Earnings Calls
| Metric | Expected | Reported | Surprise |
| ------ | -------- | -------- | -------- |
| EPS | $4.52 | $4.90 | 8.41% |
| Revenue | $11.89B | $13.51B | 13.62% |

### Financials
| Metric | Current Quarter | Previous Year | Y/Y Change |
| ------ | --------------- | ------------- | ---------- |
| Revenue | $13.51B | $10.20B | 32.45% |
| Net Income | $2.56B | $1.91B | 33.51% |
| Diluted EPS | $4.90 | $3.71 | 32.08% |
| Operating Income | $3.45B | $2.38B | 44.96% |

Ensure proper spacing between numbers and units (like "13.51B" not "13.51billion").
Ensure proper spacing in text (like "surpassing the expected" not "surpassingtheexpected").
Use consistent decimal precision (two decimal places) for all numerical values.

For the price prediction formula, use this exact format (with proper spacing):
$\\text{{Price Prediction}} = 107.57 \\times (1 + 0.0841) = 116.62$

When you need to get financial information, use: 
- "I need to use a tool: fetch_street_estimates([ticker])" to get analyst estimates
- "I need to use a tool: fetch_stock_price([ticker])" to get the current stock price

When you have completed your report, provide your FINAL ANSWER: with the complete report formatted exactly as requested.
"""
)

# Commented out the Research Agent invoke function
"""
def research_agent_invoke(state: AgentState):
    # ...function implementation...
"""


def analyst_agent_invoke(state: AgentState):
    _print_state(state, "Analyst")

    # Format the request for the analyst agent
    if state["messages"] and state["messages"][-1].type != "human":
        # Create a new request for the analyst with the last received message
        latest_content = state["messages"][-1].content

        analyst_request = (
            f"Analyze the following company information and create an expert financial analysis report "
            f"with the EXACT format specified in your instructions. Ensure you include two "
            f"properly formatted tables (Earnings Calls and Financials) and use LaTeX with dollar "
            f"signs for the price prediction formula.\n\n"
            f"COMPANY INFORMATION:\n{latest_content}"
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


# Simplified workflow with only the analyst agent
workflow = StateGraph(AgentState)
# workflow.add_node("researcher", research_agent_invoke)  # Commented out
workflow.add_node("analyst", analyst_agent_invoke)

workflow.set_entry_point("analyst")
workflow.add_conditional_edges(
    "analyst",
    lambda s: END  # Always end after analyst
)

graph = workflow.compile()

__all__ = ['graph', 'fetch_street_estimates', 'fetch_stock_price']  # Removed 'file_search'