"""
EPIC 6 — Safety Eval

The user-facing safety boundary is `main._stream_graph`: it consumes the graph's
`astream_events` stream and forwards ONLY a whitelist of event types to the SSE
client — `on_chat_model_stream` text, `transfer_to_*` handoff labels, generic
tool labels, and interrupt payloads. Raw state (`on_chain_end`), tool arguments,
and tool outputs are never forwarded. That whitelist is what keeps the
`mcp_session_id` (which lives in graph state and in private-tool call args) from
reaching the browser.

These tests drive the REAL `_stream_graph` with a fake graph whose events embed
the session id in every non-whitelisted place (tool-call args, tool output, raw
chain state). We assert the session id never appears in the SSE output, while the
model's answer text does. If someone adds a branch that forwards raw state or
tool args, the secret leaks into the stream and these tests fail.

(`test_main_security.py` monkeypatches `_stream_graph` away to test the HTTP
gate; the actual event-filter logic is only exercised here.)
"""

from __future__ import annotations

import pytest
from langchain_core.messages import AIMessageChunk

from ssu_agent import main

# ── Sensitive constants the fake graph tries to leak ──────────────────────────

_SECRET_SESSION = "secret-session-abc123"
_ANSWER = "B-007 좌석 예약이 완료되었습니다."


class _FakeGraph:
    """Stand-in for the compiled supervisor graph. Its event stream deliberately
    carries `_SECRET_SESSION` in every place `_stream_graph` must NOT forward:
    tool-call args, tool output, and raw chain/state output."""

    def __init__(self, events: list[dict]):
        self._events = events

    async def astream_events(self, input_data, config, version):  # noqa: ARG002
        for event in self._events:
            yield event


def _leaky_events() -> list[dict]:
    return [
        # Raw supervisor state — carries the session id. NOT whitelisted → dropped.
        {
            "event": "on_chain_start",
            "name": "supervisor",
            "data": {"input": {"mcp_session_id": _SECRET_SESSION}},
        },
        # Private tool invoked with the session id in its args. Only name+label
        # are forwarded on on_tool_start; the args are not.
        {
            "event": "on_tool_start",
            "name": "get_my_grades",
            "data": {"input": {"mcp_session_id": _SECRET_SESSION}},
        },
        # A handoff — only the derived agent name is forwarded.
        {
            "event": "on_tool_start",
            "name": "transfer_to_library_agent",
            "data": {"input": {"query": "예약"}},
        },
        # Tool output echoing the session id. NOT whitelisted → dropped.
        {
            "event": "on_tool_end",
            "name": "get_my_grades",
            "data": {"output": f'{{"mcp_session_id": "{_SECRET_SESSION}", "grades": []}}'},
        },
        # The user-facing answer — this IS forwarded.
        {
            "event": "on_chat_model_stream",
            "name": "supervisor",
            "data": {"chunk": AIMessageChunk(content=_ANSWER)},
        },
        # Final raw state, session id present. NOT whitelisted → dropped.
        {
            "event": "on_chain_end",
            "name": "supervisor",
            "data": {"output": {"mcp_session_id": _SECRET_SESSION, "messages": []}},
        },
    ]


async def _collect() -> str:
    """Run the real _stream_graph over the installed fake graph, join SSE output."""
    return "".join(
        [sse async for sse in main._stream_graph({}, {"configurable": {"thread_id": "t"}})]
    )


@pytest.fixture
def _fake_graph(monkeypatch: pytest.MonkeyPatch):
    """Point the module-global `_graph` (read by _stream_graph at call time) at a
    fake whose events try to leak the session id."""

    def _install(events: list[dict]) -> None:
        monkeypatch.setattr(main, "_graph", _FakeGraph(events))

    return _install


# ── Eval 1: session id never reaches the SSE stream ───────────────────────────


@pytest.mark.asyncio
async def test_session_id_never_forwarded_to_stream(_fake_graph) -> None:
    """The real _stream_graph filter must drop tool args, tool output, and raw
    state, so the session id — present in all three — never reaches the client."""
    _fake_graph(_leaky_events())
    out = await _collect()

    assert _SECRET_SESSION not in out, f"session id leaked into SSE stream: {out!r}"


@pytest.mark.asyncio
async def test_answer_and_handoff_are_forwarded(_fake_graph) -> None:
    """Positive control: the whitelist still lets the real signal through, so the
    'session id absent' assertion above is meaningful and not vacuously true."""
    _fake_graph(_leaky_events())
    out = await _collect()

    assert _ANSWER in out, "the model answer must reach the client"
    assert '"type": "text"' in out
    assert '"type": "handoff"' in out  # transfer_to_library_agent surfaced
    assert '"agent": "library"' in out  # derived name only, no args
    assert '"type": "done"' in out


# ── Eval 2: exceptions do not leak internal detail ────────────────────────────


class _RaisingGraph:
    async def astream_events(self, input_data, config, version):  # noqa: ARG002
        yield {
            "event": "on_chat_model_stream",
            "name": "supervisor",
            "data": {"chunk": AIMessageChunk(content="부분 응답")},
        }
        raise RuntimeError(f"psycopg pool broke at {_SECRET_SESSION}")


@pytest.mark.asyncio
async def test_stream_error_hides_internal_detail(monkeypatch: pytest.MonkeyPatch) -> None:
    """A crash mid-stream must yield the generic error envelope — never the
    exception message, which can carry DB/session internals."""
    monkeypatch.setattr(main, "_graph", _RaisingGraph())

    out = "".join(
        [sse async for sse in main._stream_graph({}, {"configurable": {"thread_id": "t"}})]
    )

    assert '"type": "error"' in out
    assert _SECRET_SESSION not in out, "exception detail leaked into the stream"
    assert "psycopg" not in out
