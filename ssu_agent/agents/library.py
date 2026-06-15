"""
Library sub-agent — custom StateGraph with HITL interrupt.

Architecture choice (why NOT create_react_agent):
  create_react_agent's tool-calling loop has no interception point between
  tool execution and the agent's next step. We need to pause the graph
  AFTER a prepare_* tool returns and BEFORE confirm_action executes,
  surfacing the action details for human approval.

HITL flow:
  agent_node
    → router: does any ToolMessage contain an actionId?
        yes → check_approval_node
                 1. calls interrupt(approval_request)  ← graph pauses here
                 2. FastAPI streams {type:"interrupt"} SSE to client
                 3. Client shows confirmation dialog; user approves/denies
                 4. Client POSTs to /agent/resume → Command(resume={approved, action_id})
                 5. Graph resumes inside check_approval_node, AFTER interrupt()
                 6. If approved: calls confirm_action → appends result message
                    If denied:  appends cancellation message
        no  → done_node (clear active_agent, return to parent)

Why interrupt() must be in a NODE (not a router/edge function):
  LangGraph saves state at node boundaries. interrupt() inside a node correctly
  checkpoints the state and resumes execution at the same node after resume().
  Calling interrupt() inside an add_conditional_edges router function would
  skip checkpointing, causing silent data loss on resume.

Tool split:
  Inner ReAct agent has ALL library tools EXCEPT confirm_action.
  The agent is encouraged to call prepare_* which returns an actionId.
  The graph layer enforces the approval gate before running confirm_action.
"""

from __future__ import annotations

import json
from typing import Literal

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.tools import BaseTool
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import create_react_agent
from langgraph.types import interrupt

from ssu_agent.llm_factory import create_llm, get_llm_sequence
from ssu_agent.supervisor.state import SsuAgentState

_SYSTEM_PROMPT_BASE = """당신은 숭실대학교 도서관 전문 AI 어시스턴트입니다.

CRITICAL RULES — MUST FOLLOW EXACTLY:
1. Reservation/swap/cancellation → call the matching prepare_* tool IMMEDIATELY.
   Write NO text before the tool call.
2. If the tool returns AUTH_REQUIRED → call start_auth(provider="library"),
   then show the returned loginUrl to the user.
3. After prepare_* succeeds → the system handles confirmation UI automatically.
   Do NOT call confirm_action yourself.

사용 가능한 도구:
- 좌석 현황 조회 / 추천 / 도서 검색 / 대출 현황
- 예약: prepare_reserve_library_seat
- 이석: prepare_swap_library_seat
- 반납: prepare_cancel_library_seat
- 인증 확인: get_auth_status | 로그인: start_auth(provider="library")

행동 규칙:
- 예약·이석·반납 요청이 오면 즉시 prepare_* 도구를 호출하세요. 재확인 금지.
- AUTH_REQUIRED 응답 → start_auth(provider="library") 호출 후 loginUrl 안내.
- prepare_* 호출 후 시스템이 승인 창을 자동 표시하고 confirm_action을 처리합니다.
- confirm_action은 직접 호출하지 마세요."""


def _build_library_prompt(mcp_session_id: str | None) -> str:
    prompt = _SYSTEM_PROMPT_BASE
    if mcp_session_id:
        prompt += (
            f'\n\n[인증 세션] mcp_session_id = "{mcp_session_id}"\n'
            "prepare_*, get_my_library_*, get_my_library_seat 등 인증이 필요한 도구 호출 시 "
            "이 값을 mcp_session_id 파라미터로 반드시 포함하세요."
        )
    return prompt


_CONFIRM_TOOL_NAMES = {"confirm_action"}
_PREPARE_TOOL_NAMES = {
    "prepare_reserve_library_seat",
    "prepare_swap_library_seat",
    "prepare_cancel_library_seat",
}


def _extract_action_id(messages: list) -> dict | None:
    """Scan recent ToolMessages for an actionId from a prepare_* call."""
    for msg in reversed(messages[-10:]):
        if isinstance(msg, ToolMessage):
            try:
                data = json.loads(msg.content) if isinstance(msg.content, str) else msg.content
                if isinstance(data, dict) and "data" in data:
                    inner = data["data"]
                    if isinstance(inner, dict) and "actionId" in inner:
                        return {"action_id": inner["actionId"], "details": inner}
            except (json.JSONDecodeError, TypeError):
                pass
    return None


def _has_pending_action(state: SsuAgentState) -> Literal["check_approval", "done"]:
    """Router: check if the agent produced a prepare_* result needing approval."""
    return "check_approval" if _extract_action_id(state["messages"]) else "done"


def build_library_agent(
    library_tools: list[BaseTool],
    llm: BaseChatModel | None = None,
) -> StateGraph:
    """Build the Library sub-agent graph (returns an UNCOMPILED StateGraph).

    Call .compile(checkpointer=...) on the result before use.
    """
    # llm_seq: single-item list when injected (tests), full sequence otherwise.
    llm_seq = [llm] if llm is not None else get_llm_sequence()
    if not llm_seq:
        llm_seq = [create_llm()]

    # Strip confirm_action — handled by HITL gate node
    agent_tools = [t for t in library_tools if t.name not in _CONFIRM_TOOL_NAMES]
    confirm_tool: BaseTool | None = next(
        (t for t in library_tools if t.name == "confirm_action"), None
    )

    # ── Nodes ─────────────────────────────────────────────────────────────────

    async def agent_node(state: SsuAgentState) -> dict:
        mcp_session_id = state.get("mcp_session_id")
        prompt = _build_library_prompt(mcp_session_id)
        last_exc: Exception | None = None
        for _llm in llm_seq:
            try:
                inner = create_react_agent(_llm, agent_tools, prompt=prompt)
                result = await inner.ainvoke({"messages": state["messages"]})
                return {"messages": result["messages"]}
            except Exception as exc:
                last_exc = exc
        raise last_exc or RuntimeError("All LLM providers exhausted")

    async def check_approval_node(state: SsuAgentState) -> dict:
        """HITL gate: interrupt for human approval, then execute or cancel."""
        action = _extract_action_id(state["messages"])
        if action is None:
            return {"active_agent": None}

        # ── interrupt() ──────────────────────────────────────────────────────
        # Execution pauses here; LangGraph serialises state to SQLite checkpoint.
        # FastAPI's astream_events yields an on_interrupt event, which main.py
        # streams as {"type": "interrupt", "data": {...}} SSE.
        # Client resumes via POST /agent/resume → Command(resume={approved, action_id}).
        resume = interrupt({"type": "library_reservation_approval", **action})
        # ────────────────────────────────────────────────────────────────────

        if resume.get("approved") and confirm_tool is not None:
            mcp_session_id = state.get("mcp_session_id")
            result = await confirm_tool.ainvoke({"mcp_session_id": mcp_session_id})
            msg = AIMessage(content=f"[도서관 에이전트] 예약 확정 완료: {result}")
        else:
            msg = AIMessage(content="[도서관 에이전트] 예약이 취소되었습니다.")

        return {"messages": [msg], "pending_action": None, "active_agent": None}

    def done_node(state: SsuAgentState) -> dict:
        return {"active_agent": None}

    # ── Graph ─────────────────────────────────────────────────────────────────

    graph = StateGraph(SsuAgentState)
    graph.add_node("agent", agent_node)
    graph.add_node("check_approval", check_approval_node)
    graph.add_node("done", done_node)

    graph.set_entry_point("agent")
    graph.add_conditional_edges(
        "agent",
        _has_pending_action,
        {"check_approval": "check_approval", "done": "done"},
    )
    graph.add_edge("check_approval", END)
    graph.add_edge("done", END)

    return graph
