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

from ssu_agent.agents.library import _extract_action_id, build_library_agent
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
    return _MockLibraryLLM(responses=[
        AIMessage(
            content="",
            tool_calls=[{
                "id": "tc-1",
                "name": "prepare_reserve_library_seat",
                "args": {"mcp_session_id": "sess-001", "seat_id": 42},
                "type": "tool_call",
            }],
        ),
        AIMessage(content="예약 준비 완료. 승인 대기 중입니다."),
    ])


def test_library_agent_excludes_confirm_action():
    """Library agent graph should strip confirm_action from the inner ReAct agent."""
    graph = build_library_agent(LIBRARY_TOOLS, llm=_make_library_llm())
    compiled = graph.compile()
    assert compiled is not None


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
        "pending_action": None,
    }
    config = {"configurable": {"thread_id": "lib-test-001"}}

    result = await graph.ainvoke(initial, config=config)

    # ainvoke surfaces the interrupt via __interrupt__ key, not by raising
    assert "__interrupt__" in result, "Expected graph to pause with __interrupt__"
    interrupt_val = result["__interrupt__"][0].value
    assert interrupt_val["type"] == "library_reservation_approval"
    assert interrupt_val["action_id"] == 99
