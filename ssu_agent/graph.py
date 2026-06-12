from __future__ import annotations

from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.tools import BaseTool
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import create_react_agent

from ssu_agent import config
from ssu_agent.mcp_client import create_mcp_client


async def build_graph(
    llm: BaseChatModel | None = None,
    tools: list[BaseTool] | None = None,
) -> CompiledStateGraph:
    """Build a ReAct agent graph.

    Args:
        llm: Override LLM (used in tests). Defaults to the configured chat model.
        tools: Override tool list (used in tests). Defaults to ssuMCP public tools.
    """
    if tools is None:
        client = create_mcp_client()
        tools = await client.get_tools()

    if llm is None:
        from langchain_google_genai import ChatGoogleGenerativeAI

        llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            google_api_key=config.GOOGLE_API_KEY,
        )

    return create_react_agent(llm, tools)


async def run_query(user_message: str) -> str:
    """Run a single user query through the agent and return the response text."""
    graph = await build_graph()
    result: dict[str, Any] = await graph.ainvoke(
        {"messages": [{"role": "user", "content": user_message}]}
    )
    messages = result.get("messages", [])
    if messages:
        last = messages[-1]
        return last.content if hasattr(last, "content") else str(last)
    return ""
