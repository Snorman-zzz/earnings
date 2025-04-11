"""Define the agent workflow graph."""

from __future__ import annotations

from typing import Dict, List, Annotated, TypedDict, Optional

from langchain_core.tools import BaseTool
from langchain_core.runnables import RunnableConfig
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, END

from react_agent.configuration import Configuration
from react_agent.utils import load_chat_model, get_message_text


class AgentState(TypedDict):
    """The state for the agent."""
    messages: list[BaseMessage]
    vector_store_id: Optional[str]
    step_count: int
    final_answer: Optional[str]


def build_agent_graph(tools: List[BaseTool], config: RunnableConfig = None):
    """Build the agent graph with the specified tools and configuration."""
    # Load configuration from runnable config
    configuration = Configuration.from_runnable_config(config)

    # Load language model
    llm = load_chat_model(configuration.model)

    # Create the graph
    workflow = StateGraph(AgentState)

    # Define the agent function
    def agent_fn(state: AgentState):
        """Simple agent reasoning step."""
        messages = state["messages"]
        system_message = configuration.system_prompt

        # Create final prompt for LLM
        prompt = [("system", system_message)]
        for msg in messages:
            prompt.append((msg.type, get_message_text(msg)))

        # Call LLM
        response = llm.invoke(prompt)

        # Add response to messages
        messages.append(AIMessage(content=response.content))

        # Check if any tools should be called
        tool_match = False
        lower_content = response.content.lower()

        for tool in tools:
            if f"use {tool.name.lower()}" in lower_content or f"call {tool.name.lower()}" in lower_content:
                tool_match = True
                break

        # Check for final answer
        final_answer = None
        if "final answer:" in lower_content:
            parts = response.content.split("final answer:", 1)
            final_answer = parts[1].strip() if len(parts) > 1 else None

        return {
            "messages": messages,
            "step_count": state["step_count"] + 1,
            "final_answer": final_answer,
            "vector_store_id": state.get("vector_store_id")
        }

    # Add the agent node
    workflow.add_node("agent", agent_fn)

    # Implement tool execution
    def tool_executor(state: AgentState, tool_name: str):
        """Execute a tool."""
        messages = state["messages"]
        last_message = messages[-1]

        # Find the tool
        selected_tool = None
        for tool in tools:
            if tool.name.lower() == tool_name.lower():
                selected_tool = tool
                break

        if not selected_tool:
            # Tool not found - add error message
            messages.append(HumanMessage(content=f"Error: Tool '{tool_name}' not found."))
            return state

        # Try to extract arguments from the message
        content = get_message_text(last_message)
        try:
            # Simple argument extraction - look for the tool name followed by parameters
            start_idx = content.lower().find(tool_name.lower())
            if start_idx == -1:
                raise ValueError(f"Cannot find tool name '{tool_name}' in message.")

            arg_text = content[start_idx + len(tool_name):].strip()
            # Find the first opening parenthesis after the tool name
            open_paren = arg_text.find("(")
            close_paren = arg_text.rfind(")")

            if open_paren != -1 and close_paren != -1:
                arg = arg_text[open_paren + 1:close_paren].strip().strip('"\'')

                # Special case for file_search which needs vector_store_id
                if tool_name.lower() == "file_search":
                    result = selected_tool.invoke({"query": arg, "vector_store_id": state.get("vector_store_id", "")})
                else:
                    result = selected_tool.invoke(arg)

                # Add tool result to messages
                messages.append(HumanMessage(content=f"Tool {tool_name} result: {result}"))
            else:
                messages.append(HumanMessage(content=f"Error: Could not parse arguments for {tool_name}."))

        except Exception as e:
            # Add error message
            messages.append(HumanMessage(content=f"Error executing tool {tool_name}: {str(e)}"))

        return {
            "messages": messages,
            "step_count": state["step_count"],
            "final_answer": state.get("final_answer"),
            "vector_store_id": state.get("vector_store_id")
        }

    # Add tool nodes for each tool
    for tool in tools:
        workflow.add_node(tool.name, lambda state, tool=tool: tool_executor(state, tool.name))

    # Set entry point
    workflow.set_entry_point("agent")

    # Add conditional edges from agent
    def agent_router(state: AgentState):
        """Route based on the agent's output."""
        # If we have a final answer, we're done
        if state.get("final_answer"):
            return END

        # If we've hit the step limit, we're done
        if state["step_count"] >= 10:  # Limit to 10 steps
            return END

        # See if we should use a tool
        last_message = state["messages"][-1]
        content = get_message_text(last_message).lower()

        for tool in tools:
            tool_name = tool.name.lower()
            if f"use {tool_name}" in content or f"call {tool_name}" in content:
                return tool_name

        # No tool match, continue with agent
        return "agent"

    # Add conditional edges from agent
    workflow.add_conditional_edges("agent", agent_router)

    # Add edges from tools back to agent
    for tool in tools:
        workflow.add_edge(tool.name, "agent")

    # Compile the graph
    return workflow.compile()


# Default graph with no tools
graph = build_agent_graph([])