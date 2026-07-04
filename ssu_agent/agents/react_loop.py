"""
Shared manual bind_tools ReAct loop for the read-only sub-agents.

The academic and LMS sub-agents run the identical loop — bind the tools, let
the model call them for up to N turns, then return one tagged answer — differing
only by their tool set, system prompt, and display tag. This module holds that
loop once so the two agents can't drift apart. The library agent does NOT use it:
its HITL gate needs the intermediate prepare_* ToolMessages preserved in state,
whereas this loop intentionally returns only the final tagged answer.

Why a manual loop instead of create_react_agent: it enables per-provider fallback
across the LLM sequence and avoids the turn-2 looping observed with the prebuilt
agent (see the library agent's module docstring for the A/B detail).
"""

from __future__ import annotations

import json

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, SystemMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool

from ssu_agent.supervisor.state import SsuAgentState

_MAX_TOOL_TURNS = 6


def drop_routing_messages(messages: list) -> list:
    """Remove supervisor transfer_to_* tool calls and their ToolMessage results.

    When the supervisor routes to a sub-agent it leaves an AIMessage with a
    transfer_to_<agent> tool call + a ToolMessage("ROUTE_TO:<agent>") in the
    shared state. Groq llama-3.3-70b sees the trailing ToolMessage and produces a
    text completion instead of calling the sub-agent's tools. Strip those routing
    artifacts so the inner ReAct agent sees a clean user→ conversation.
    """
    routing_call_ids: set[str] = set()
    for msg in messages:
        if (
            isinstance(msg, AIMessage)
            and msg.tool_calls
            and all(tc.get("name", "").startswith("transfer_to_") for tc in msg.tool_calls)
        ):
            for tc in msg.tool_calls:
                routing_call_ids.add(tc.get("id", ""))

    result = []
    for msg in messages:
        if (
            isinstance(msg, AIMessage)
            and msg.tool_calls
            and all(tc.get("name", "").startswith("transfer_to_") for tc in msg.tool_calls)
        ):
            continue
        if isinstance(msg, ToolMessage) and msg.tool_call_id in routing_call_ids:
            continue
        result.append(msg)
    return result


async def run_react_loop(
    llm_seq: list[BaseChatModel],
    tools: list[BaseTool],
    system_prompt: str,
    tag: str,
    state: SsuAgentState,
    config: RunnableConfig,
) -> dict:
    """Run the bind_tools ReAct loop with per-provider fallback.

    Tries each LLM in ``llm_seq`` in order; on any provider error it advances to
    the next. Returns a single ``[{tag} ...]``-tagged AIMessage and clears
    ``active_agent`` so control returns to the supervisor.
    """
    messages = drop_routing_messages(state["messages"])
    input_messages = [SystemMessage(content=system_prompt), *messages]

    last_exc: Exception | None = None
    for _llm in llm_seq:
        try:
            llm_with_tools = _llm.bind_tools(tools)
            history = list(input_messages)

            for _ in range(_MAX_TOOL_TURNS):
                response = await llm_with_tools.ainvoke(history, config=config)
                history.append(response)

                if not response.tool_calls:
                    break

                for tc in response.tool_calls:
                    matched = next((t for t in tools if t.name == tc["name"]), None)
                    if matched is None:
                        history.append(
                            ToolMessage(
                                content=f"Tool '{tc['name']}' not found.",
                                tool_call_id=tc.get("id", ""),
                            )
                        )
                        continue
                    try:
                        result = await matched.ainvoke(tc.get("args", {}), config=config)
                        content = (
                            result
                            if isinstance(result, str)
                            else json.dumps(result, ensure_ascii=False)
                        )
                    except Exception as tool_exc:
                        content = f"Tool error: {tool_exc}"
                    history.append(ToolMessage(content=content, tool_call_id=tc.get("id", "")))

            last_ai = next(
                (
                    m
                    for m in reversed(history[len(input_messages):])
                    if isinstance(m, AIMessage) and m.content
                ),
                None,
            )
            tagged = AIMessage(
                content=f"[{tag}] {last_ai.content}" if last_ai else f"[{tag}] 처리 완료"
            )
            return {"messages": [tagged], "active_agent": None}
        except Exception as exc:
            last_exc = exc

    raise last_exc or RuntimeError("All LLM providers exhausted")
