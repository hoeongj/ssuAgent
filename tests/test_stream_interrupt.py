"""Regression: the library HITL approval must reach the SSE client.

langgraph 1.2.4's astream_events(version="v2") does NOT emit an on_interrupt
event. When a node calls interrupt(), the graph pauses and the interrupt rides
inside an on_chain_stream chunk as {"__interrupt__": (Interrupt(value=...),)}.
An earlier version of _stream_graph matched `event == "on_interrupt"`, a branch
that never fired — so a paused graph reached the client as a plain `done`, the
approval card never showed, and /agent/resume was never called (HITL broken).

These tests drive the REAL _stream_graph over a REAL interrupting langgraph graph
(mirroring the prod topology: a parent graph embedding a compiled subgraph whose
node interrupts), so a regression to the dead-branch behaviour fails here. The
prior fake-graph tests couldn't catch it: they synthesised an on_interrupt event
that real langgraph never produces.
"""

from __future__ import annotations

import json
from typing import Annotated, TypedDict

import pytest
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import StructuredTool, tool
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.types import Command, interrupt
from pydantic import BaseModel

from ssu_agent import main
from ssu_agent.supervisor.graph import build_supervisor_graph

# Same shape the real library agent passes to interrupt(): a type tag plus the
# extracted action ({"action_id", "details"}).
_PAYLOAD = {
    "type": "library_reservation_approval",
    "action_id": 77,
    "details": {"seatCode": "B-007", "roomName": "3층", "actionType": "RESERVE"},
}


class _State(TypedDict):
    messages: Annotated[list, add_messages]


def _build_interrupting_graph():
    def ask(state: _State):
        decision = interrupt(_PAYLOAD)
        approved = bool(decision.get("approved")) if isinstance(decision, dict) else False
        return {"messages": [{"role": "assistant", "content": f"확정:{approved}"}]}

    sub = StateGraph(_State)
    sub.add_node("ask", ask)
    sub.add_edge(START, "ask")
    sub.add_edge("ask", END)

    parent = StateGraph(_State)
    parent.add_node("library_agent", sub.compile())
    parent.add_edge(START, "library_agent")
    parent.add_edge("library_agent", END)
    return parent.compile(checkpointer=MemorySaver())


async def _collect(input_data, config) -> list[dict]:
    events: list[dict] = []
    async for sse in main._stream_graph(input_data, config):
        assert sse.startswith("data: ") and sse.endswith("\n\n")
        events.append(json.loads(sse[len("data: ") :].strip()))
    return events


@pytest.fixture
def _graph(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(main, "_graph", _build_interrupting_graph())


async def test_interrupt_surfaces_as_sse(_graph) -> None:
    config = {"configurable": {"thread_id": "hitl-1"}}
    events = await _collect({"messages": [{"role": "user", "content": "예약해줘"}]}, config)

    types = [e["type"] for e in events]
    assert "interrupt" in types, f"HITL interrupt never reached the client: {types}"
    # The stream pauses on interrupt — it must NOT fall through to done.
    assert "done" not in types, f"stream should pause on interrupt, not complete: {types}"

    interrupt_evt = next(e for e in events if e["type"] == "interrupt")
    assert interrupt_evt["data"] == _PAYLOAD


async def test_resume_completes_without_reinterrupting(_graph) -> None:
    config = {"configurable": {"thread_id": "hitl-2"}}
    await _collect({"messages": [{"role": "user", "content": "예약해줘"}]}, config)

    resumed = await _collect(Command(resume={"approved": True, "action_id": 77}), config)
    types = [e["type"] for e in resumed]
    assert "handoff" not in types, f"resume must not emit a fresh handoff: {types}"
    assert "interrupt" not in types, f"resume must not re-interrupt: {types}"
    assert "done" in types, f"resumed stream must complete with done: {types}"


class _HitlLLM(FakeMessagesListChatModel):
    def bind_tools(self, tools, **kwargs):
        return self


class _PrepareReserveArgs(BaseModel):
    mcp_session_id: str
    seat_id: int


async def _content_and_artifact_prepare(mcp_session_id: str, seat_id: int):
    payload = json.dumps(
        {"status": "OK", "data": {"actionId": 314, "seatLabel": "B-007"}},
        ensure_ascii=False,
    )
    return ([{"type": "text", "text": payload}], None)


def _make_content_and_artifact_prepare_tool() -> StructuredTool:
    return StructuredTool(
        name="prepare_reserve_library_seat",
        description="예약 준비",
        args_schema=_PrepareReserveArgs,
        coroutine=_content_and_artifact_prepare,
        response_format="content_and_artifact",
    )


def _make_recording_confirm_tool(confirm_calls: list[dict]):
    @tool
    def confirm_action(mcp_session_id: str, action_id: int) -> str:
        """예약 확정"""
        confirm_calls.append({"mcp_session_id": mcp_session_id, "action_id": action_id})
        return json.dumps(
            {"status": "OK", "data": "예약 요청을 접수했습니다. intentId=314."},
            ensure_ascii=False,
        )

    return confirm_action


def _make_full_hitl_llm() -> _HitlLLM:
    return _HitlLLM(
        responses=[
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "id": "route-1",
                        "name": "transfer_to_library_agent",
                        "args": {"query": "B-007 예약해줘"},
                        "type": "tool_call",
                    }
                ],
            ),
            AIMessage(content="도서관 에이전트에게 전달했습니다."),
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "id": "prepare-1",
                        "name": "prepare_reserve_library_seat",
                        "args": {"mcp_session_id": "stale-session", "seat_id": 42},
                        "type": "tool_call",
                    }
                ],
            ),
        ]
    )


async def _build_full_hitl_graph(confirm_calls: list[dict]):
    return await build_supervisor_graph(
        all_tools=[
            _make_content_and_artifact_prepare_tool(),
            _make_recording_confirm_tool(confirm_calls),
        ],
        llm=_make_full_hitl_llm(),
        checkpointer=MemorySaver(),
    )


def _initial_full_hitl_state():
    return {
        "messages": [HumanMessage(content="B-007 예약해줘")],
        "mcp_session_id": "stale-session",
        "library_connected": True,
        "active_agent": None,
    }


async def test_full_graph_resume_approved_invokes_confirm_and_streams_message(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    confirm_calls: list[dict] = []
    monkeypatch.setattr(main, "_graph", await _build_full_hitl_graph(confirm_calls))
    config = {"configurable": {"thread_id": "full-hitl-approve"}}

    interrupted = await _collect(_initial_full_hitl_state(), config)

    interrupt_evt = next(e for e in interrupted if e["type"] == "interrupt")
    assert interrupt_evt["data"]["action_id"] == 314

    req = main.ResumeRequest(
        thread_id="full-hitl-approve",
        approved=True,
        action_id=314,
        mcp_session_id="fresh-session",
        library_connected=True,
    )
    resumed = await _collect(main.build_resume_command(req), config)

    assert confirm_calls == [{"mcp_session_id": "fresh-session", "action_id": 314}]
    assert any(
        e["type"] == "text" and "예약 요청을 접수했습니다. intentId=314." in e["content"]
        for e in resumed
    )
    assert resumed[-1]["type"] == "done"


async def test_full_graph_resume_denied_streams_cancel_without_confirm(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    confirm_calls: list[dict] = []
    monkeypatch.setattr(main, "_graph", await _build_full_hitl_graph(confirm_calls))
    config = {"configurable": {"thread_id": "full-hitl-deny"}}

    interrupted = await _collect(_initial_full_hitl_state(), config)
    assert any(e["type"] == "interrupt" for e in interrupted)

    req = main.ResumeRequest(
        thread_id="full-hitl-deny",
        approved=False,
        action_id=314,
        mcp_session_id="fresh-session",
        library_connected=True,
    )
    resumed = await _collect(main.build_resume_command(req), config)

    assert confirm_calls == []
    assert any(e["type"] == "text" and "예약이 취소되었습니다." in e["content"] for e in resumed)
    assert resumed[-1]["type"] == "done"
