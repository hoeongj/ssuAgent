"""
LMS sub-agent — assignments and LMS terms.

Uses direct bind_tools loop (not create_react_agent) to avoid turn-2 looping
and enable per-provider fallback.
Supports optional term_id for semester selection (LMS term bug fix, PR #61).
"""

from __future__ import annotations

import json

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, SystemMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool
from langgraph.graph import StateGraph

from ssu_agent.agents.library import _drop_routing_messages
from ssu_agent.llm_factory import create_llm, get_llm_sequence
from ssu_agent.supervisor.state import SsuAgentState

_SYSTEM_PROMPT_BASE = """당신은 숭실대학교 LMS(Canvas) 전문 AI 어시스턴트입니다.

담당 영역:
- 과제 목록 조회 (get_my_assignments): compact=true 옵션으로 요약 제공
- LMS 학기 목록 (get_my_lms_terms): 학기 선택 시 먼저 이 도구로 학기 ID를 확인하세요.

학기 관련 주의: Canvas API는 6월에 여름학기를 기본(default)으로 반환하므로,
1학기 과제를 조회할 때는 get_my_lms_terms로 학기 목록을 먼저 조회하고
올바른 term_id를 사용하세요."""


def _build_lms_prompt(mcp_session_id: str | None) -> str:
    prompt = _SYSTEM_PROMPT_BASE
    if mcp_session_id:
        prompt += (
            f'\n\n[인증 세션] mcp_session_id = "{mcp_session_id}"\n'
            "get_my_assignments, get_my_lms_terms "
            "호출 시 이 값을 mcp_session_id 파라미터로 반드시 포함하세요."
        )
    return prompt


def build_lms_agent(
    lms_tools: list[BaseTool],
    llm: BaseChatModel | None = None,
) -> StateGraph:
    """Build the LMS sub-agent graph."""
    llm_seq = [llm] if llm is not None else get_llm_sequence()
    if not llm_seq:
        llm_seq = [create_llm()]

    async def agent_node(state: SsuAgentState, config: RunnableConfig) -> dict:
        mcp_session_id = state.get("mcp_session_id")
        prompt = _build_lms_prompt(mcp_session_id)
        messages = _drop_routing_messages(state["messages"])
        input_messages = [SystemMessage(content=prompt), *messages]

        last_exc: Exception | None = None
        for _llm in llm_seq:
            try:
                llm_with_tools = _llm.bind_tools(lms_tools)
                history = list(input_messages)

                for _ in range(6):
                    response = await llm_with_tools.ainvoke(history, config=config)
                    history.append(response)

                    if not response.tool_calls:
                        break

                    for tc in response.tool_calls:
                        matched = next((t for t in lms_tools if t.name == tc["name"]), None)
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
                        for m in reversed(history[len(input_messages) :])
                        if isinstance(m, AIMessage) and m.content
                    ),
                    None,
                )
                tagged = AIMessage(
                    content=f"[LMS 에이전트] {last_ai.content}"
                    if last_ai
                    else "[LMS 에이전트] 처리 완료"
                )
                return {"messages": [tagged], "active_agent": None}
            except Exception as exc:
                last_exc = exc

        raise last_exc or RuntimeError("All LLM providers exhausted")

    graph = StateGraph(SsuAgentState)
    graph.add_node("agent", agent_node)
    graph.set_entry_point("agent")
    graph.set_finish_point("agent")
    return graph
