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
  Inner ReAct loop has ALL library tools EXCEPT confirm_action.
  The agent is encouraged to call prepare_* which returns an actionId.
  The graph layer enforces the approval gate before running confirm_action.

Why manual bind_tools loop instead of create_react_agent:
  In controlled A/B testing (turn-2 path: prepare_* → AUTH_REQUIRED → start_auth),
  create_react_agent exhibited looping — it called prepare_reserve_library_seat
  twice instead of advancing to start_auth on the second turn. In the HITL flow
  this would produce two distinct actionIds; _extract_action_id scans recent
  ToolMessages and would gate on the wrong/stale action, breaking the approval gate.
  The manual loop's explicit break-after-actionId prevents this entirely.
  (A malformed <function=...> XML tool call was observed once in production logs,
  but was not reproducible in controlled testing — XML causation is unconfirmed.)
"""

from __future__ import annotations

import json
from typing import Literal

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, SystemMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool
from langgraph.graph import END, StateGraph
from langgraph.types import interrupt

from ssu_agent.agents.react_loop import drop_routing_messages
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
    else:
        # No auth session: reservation / personal library tools would only hit
        # AUTH_REQUIRED, and calling prepare_* here makes weak models emit filler
        # instead of a useful message. Answer directly with a login nudge.
        # (The library has its own login, separate from u-SAINT/LMS SmartID SSO.)
        prompt += (
            "\n\n[인증 세션 없음] 예약·이석·반납·대출 현황·내 좌석처럼 로그인이 필요한 "
            "기능은 지금 이용할 수 없습니다. prepare_*·get_my_library_* 도구를 호출하지 말고, "
            "'좌석 예약·대출 등은 도서관 로그인(연결) 후 이용할 수 있어요'라고 안내만 하세요. "
            "좌석 현황 조회·도서 검색 같은 공개 도구는 그대로 사용해 답하세요."
        )
    return prompt


_CONFIRM_TOOL_NAMES = {"confirm_action"}
_PREPARE_TOOL_NAMES = {
    "prepare_reserve_library_seat",
    "prepare_swap_library_seat",
    "prepare_cancel_library_seat",
}


def inner_react_tools(library_tools: list[BaseTool]) -> list[BaseTool]:
    """Tools the inner ReAct loop is allowed to call — everything EXCEPT
    confirm_action, which is run only by the HITL gate node after the human
    approves. Extracted as a pure function so the approval-gate invariant
    (the model can never call confirm_action itself) is directly unit-testable.
    """
    return [t for t in library_tools if t.name not in _CONFIRM_TOOL_NAMES]


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


def _extract_login_url(content: str) -> str | None:
    """Pull the loginUrl out of an AUTH_REQUIRED tool response (top-level or nested)."""
    try:
        data = json.loads(content) if isinstance(content, str) else content
    except (json.JSONDecodeError, TypeError):
        return None
    scopes = [data, data.get("data") if isinstance(data, dict) else None]
    for scope in scopes:
        if isinstance(scope, dict):
            url = scope.get("loginUrl") or scope.get("login_url")
            if isinstance(url, str) and url:
                return url
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
    llm_seq = [llm] if llm is not None else get_llm_sequence()
    if not llm_seq:
        llm_seq = [create_llm()]

    # Strip confirm_action — handled by HITL gate node
    agent_tools = inner_react_tools(library_tools)
    confirm_tool: BaseTool | None = next(
        (t for t in library_tools if t.name == "confirm_action"), None
    )

    # ── Nodes ─────────────────────────────────────────────────────────────────

    async def agent_node(state: SsuAgentState, config: RunnableConfig) -> dict:
        mcp_session_id = state.get("mcp_session_id")
        prompt = _build_library_prompt(mcp_session_id)
        messages = drop_routing_messages(state["messages"])
        input_messages = [SystemMessage(content=prompt), *messages]

        last_exc: Exception | None = None
        for _llm in llm_seq:
            try:
                llm_with_tools = _llm.bind_tools(agent_tools)
                history = list(input_messages)

                for _ in range(6):
                    response = await llm_with_tools.ainvoke(history, config=config)
                    history.append(response)

                    if not response.tool_calls:
                        break

                    hitl_triggered = False
                    for tc in response.tool_calls:
                        matched = next((t for t in agent_tools if t.name == tc["name"]), None)
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

                        # Deterministic auth guard: if an auth-required tool reports
                        # AUTH_REQUIRED, return a fixed login-needed message NOW and stop.
                        # The weak free LLM otherwise ignores the result and hallucinates a
                        # successful reservation ("예약되었습니다" with nothing reserved).
                        if "AUTH_REQUIRED" in content:
                            login_url = _extract_login_url(content)
                            notice = (
                                "좌석 예약·대출 같은 기능은 도서관 로그인(연결)이 필요해요. "
                                "먼저 도서관에 로그인해 주세요."
                            )
                            if login_url:
                                notice += f"\n로그인: {login_url}"
                            return {
                                "messages": [AIMessage(content=f"[도서관 에이전트] {notice}")],
                                "active_agent": None,
                            }

                        # If prepare_* returned an actionId let HITL router take over
                        if tc["name"] in _PREPARE_TOOL_NAMES:
                            try:
                                data = json.loads(content)
                                if (
                                    isinstance(data, dict)
                                    and "data" in data
                                    and isinstance(data["data"], dict)
                                    and "actionId" in data["data"]
                                ):
                                    hitl_triggered = True
                            except (json.JSONDecodeError, TypeError):
                                pass

                    if hitl_triggered:
                        break

                return {"messages": history[len(input_messages) :]}
            except Exception as exc:
                last_exc = exc

        raise last_exc or RuntimeError("All LLM providers exhausted")

    async def check_approval_node(state: SsuAgentState) -> dict:
        """HITL gate: interrupt for human approval, then execute or cancel."""
        action = _extract_action_id(state["messages"])
        if action is None:
            return {"active_agent": None}

        # ── interrupt() ──────────────────────────────────────────────────────
        # Execution pauses here; LangGraph serialises state to the checkpointer
        # (prod=Postgres, local=SQLite). The pause surfaces in astream_events as an
        # on_chain_stream chunk carrying __interrupt__ (NOT an on_interrupt event);
        # main._extract_interrupt forwards the Interrupt value as {"type":"interrupt"} SSE.
        # Client resumes via POST /agent/resume → Command(resume={approved, action_id}).
        resume = interrupt({"type": "library_reservation_approval", **action})
        # ────────────────────────────────────────────────────────────────────

        if resume.get("approved") and confirm_tool is not None:
            mcp_session_id = state.get("mcp_session_id")
            result = await confirm_tool.ainvoke({"mcp_session_id": mcp_session_id})
            msg = AIMessage(content=f"[도서관 에이전트] 예약 확정 완료: {result}")
        else:
            msg = AIMessage(content="[도서관 에이전트] 예약이 취소되었습니다.")

        return {"messages": [msg], "active_agent": None}

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
