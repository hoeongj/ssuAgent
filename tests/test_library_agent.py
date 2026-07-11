"""
Tests for the Library sub-agent — HITL interrupt logic.

Key scenarios:
1. HITL: prepare_reserve_library_seat result triggers interrupt()
2. No HITL: tool results without actionId pass through without interruption
3. execute_confirm_node calls confirm_action after approval
"""

from __future__ import annotations

import json

import pytest
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool

from ssu_agent.agents.library import (
    _extract_action_id,
    build_library_agent,
    inner_react_tools,
)
from ssu_agent.supervisor.state import SsuAgentState

# ── Mock tools ────────────────────────────────────────────────────────────────


@tool
def get_library_available_seats() -> str:
    """도서관 좌석 현황"""
    return '{"floors": [{"floor": 2, "available": 10}]}'


@tool
def prepare_reserve_library_seat(mcp_session_id: str, seat_id: int) -> str:
    """예약 준비"""
    return json.dumps({"status": "OK", "data": {"actionId": 99, "seatLabel": "B-007"}})


@tool
def confirm_action(mcp_session_id: str, action_id: int) -> str:
    """예약 확정"""
    return '{"status": "OK", "message": "예약 완료"}'


LIBRARY_TOOLS = [get_library_available_seats, prepare_reserve_library_seat, confirm_action]


# ── Unit: _extract_action_id ──────────────────────────────────────────────────


def test_extract_action_id_from_tool_message():
    msgs = [
        HumanMessage(content="B-007 좌석 예약해줘"),
        AIMessage(content=""),
        ToolMessage(
            content=json.dumps({"status": "OK", "data": {"actionId": 99, "seatLabel": "B-007"}}),
            tool_call_id="tc-1",
        ),
    ]
    result = _extract_action_id(msgs)
    assert result is not None
    assert result["action_id"] == 99
    assert result["details"]["seatLabel"] == "B-007"


def test_extract_action_id_returns_none_when_no_action():
    msgs = [
        HumanMessage(content="좌석 현황 알려줘"),
        AIMessage(content=""),
        ToolMessage(
            content=json.dumps({"status": "OK", "data": {"floors": []}}),
            tool_call_id="tc-1",
        ),
    ]
    assert _extract_action_id(msgs) is None


def test_extract_action_id_handles_malformed_json():
    msgs = [
        ToolMessage(content="not-json-{{{", tool_call_id="tc-1"),
    ]
    # Should not raise; returns None
    assert _extract_action_id(msgs) is None


# ── Unit: library graph builds ────────────────────────────────────────────────


class _MockLibraryLLM(FakeMessagesListChatModel):
    """Fake LLM: first call returns a prepare_reserve tool call, second returns final text."""

    def bind_tools(self, tools, **kwargs):
        return self


def _make_library_llm() -> _MockLibraryLLM:
    """Two-step response: tool call → synthesis (matches ReAct loop)."""
    return _MockLibraryLLM(
        responses=[
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "id": "tc-1",
                        "name": "prepare_reserve_library_seat",
                        "args": {"mcp_session_id": "sess-001", "seat_id": 42},
                        "type": "tool_call",
                    }
                ],
            ),
            AIMessage(content="예약 준비 완료. 승인 대기 중입니다."),
        ]
    )


def test_library_agent_excludes_confirm_action():
    """The inner ReAct loop must NOT be able to call confirm_action — it is run
    only by the HITL gate after human approval. Assert the actual tool split."""
    inner_names = {t.name for t in inner_react_tools(LIBRARY_TOOLS)}

    # confirm_action is present in the full tool set but withheld from the model.
    assert "confirm_action" in {t.name for t in LIBRARY_TOOLS}
    assert "confirm_action" not in inner_names
    # Read/prepare tools that the model IS allowed to call survive the split.
    assert "prepare_reserve_library_seat" in inner_names
    assert "get_library_available_seats" in inner_names


def test_library_agent_graph_compiles():
    graph = build_library_agent(LIBRARY_TOOLS, llm=_make_library_llm())
    assert graph.compile() is not None


# ── Integration: HITL interrupt triggers on prepare result ────────────────────


@pytest.mark.asyncio
async def test_library_agent_interrupt_on_prepare():
    """Library agent should pause with __interrupt__ when prepare_* result contains actionId.

    LangGraph 1.2.4: ainvoke catches GraphInterrupt internally and returns
    {'__interrupt__': [Interrupt(value=..., id=...)]} in the result dict.
    (Raising GraphInterrupt only happens when the node runs outside a compiled graph.)
    """
    from langgraph.checkpoint.memory import MemorySaver

    graph = build_library_agent(LIBRARY_TOOLS, llm=_make_library_llm()).compile(
        checkpointer=MemorySaver()
    )

    initial: SsuAgentState = {
        "messages": [HumanMessage(content="B-007 예약해줘")],
        "mcp_session_id": "sess-001",
        "active_agent": "library",
    }
    config = {"configurable": {"thread_id": "lib-test-001"}}

    result = await graph.ainvoke(initial, config=config)

    # ainvoke surfaces the interrupt via __interrupt__ key, not by raising
    assert "__interrupt__" in result, "Expected graph to pause with __interrupt__"
    interrupt_val = result["__interrupt__"][0].value
    assert interrupt_val["type"] == "library_reservation_approval"
    assert interrupt_val["action_id"] == 99


# ── Integration: AUTH_REQUIRED deterministic guard ────────────────────────────


@tool
def prepare_reserve_needs_auth(mcp_session_id: str, seat_id: int) -> str:
    """예약 준비 (returns AUTH_REQUIRED)"""
    return json.dumps(
        {
            "status": "AUTH_REQUIRED",
            "provider": "library",
            "loginUrl": "https://ssumcp.duckdns.org/api/auth/library/start",
            "data": None,
        }
    )


class _AuthRequiredLLM(FakeMessagesListChatModel):
    def bind_tools(self, tools, **kwargs):
        return self


@pytest.mark.asyncio
async def test_library_auth_required_returns_login_message_not_hallucination():
    """When a reservation tool returns AUTH_REQUIRED, the agent must deterministically
    return a 'log in first' message + loginUrl — NOT let the weak LLM hallucinate a
    successful reservation ("예약되었습니다" with nothing actually reserved)."""
    from langgraph.checkpoint.memory import MemorySaver

    llm = _AuthRequiredLLM(
        responses=[
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "id": "tc-1",
                        "name": "prepare_reserve_needs_auth",
                        "args": {"mcp_session_id": "", "seat_id": 2},
                        "type": "tool_call",
                    }
                ],
            ),
            # The model WOULD hallucinate success on its next turn — the guard fires first.
            AIMessage(content="네, 도서관 2층 좌석이 예약되었습니다."),
        ]
    )
    graph = build_library_agent([prepare_reserve_needs_auth], llm=llm).compile(
        checkpointer=MemorySaver()
    )
    initial: SsuAgentState = {
        "messages": [HumanMessage(content="도서관 2층 예약해줘")],
        "mcp_session_id": None,
        "active_agent": "library",
    }
    result = await graph.ainvoke(initial, config={"configurable": {"thread_id": "auth-req-1"}})

    final = result["messages"][-1].content
    assert "도서관 로그인" in final  # deterministic login nudge
    assert "예약되었습니다" not in final  # hallucination suppressed
    assert "ssumcp.duckdns.org" in final  # loginUrl surfaced to the user
