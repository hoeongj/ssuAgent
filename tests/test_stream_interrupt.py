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
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.types import Command, interrupt

from ssu_agent import main

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
    assert "interrupt" not in types, f"resume must not re-interrupt: {types}"
    assert "done" in types, f"resumed stream must complete with done: {types}"
