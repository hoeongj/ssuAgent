"""
EPIC 6 — Routing Eval

Verifies the supervisor's routing machinery end-to-end WITHOUT a live LLM:

    fake LLM emits a transfer_to_* tool call
        → the real supervisor ReAct agent executes that routing tool
        → the tool returns a "ROUTE_TO:X" marker (from `_make_routing_tools`)
        → `_post_supervisor` extracts the marker and routes to the sub-agent node

The one thing that genuinely needs a live model is the *tool-selection* step
(query text → which transfer_to_* the LLM decides to call). That is simulated by
the fake LLM. Everything downstream of that decision — tool registration, the
ReAct execution loop, the marker each routing tool emits, the marker parser, and
the goto target `_post_supervisor` produces — is the REAL production code path.

This is deliberately not the old shape, where a `ROUTE_TO:library_agent`
ToolMessage was hand-injected and the test asserted the parser echoed it back.
That tautology could not catch a routing tool emitting the wrong marker, a
prefix mismatch, or an unregistered transfer tool; this test can.
"""

from __future__ import annotations

import pytest
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import END
from langgraph.prebuilt import create_react_agent

from ssu_agent.supervisor.graph import (
    _ROUTE_PREFIX,
    _SUPERVISOR_PROMPT,
    _make_routing_tools,
    _post_supervisor,
)
from ssu_agent.supervisor.state import SsuAgentState

# ── Fake tool-calling supervisor LLM ──────────────────────────────────────────


class _FakeRoutingLLM(FakeMessagesListChatModel):
    """Fake supervisor LLM. `bind_tools` is a no-op so the ReAct agent keeps the
    canned responses; the actual tools are executed by create_react_agent's
    ToolNode, not by the model."""

    def bind_tools(self, tools, **kwargs):
        return self


def _transfer_call(tool_name: str, query: str) -> AIMessage:
    """An AIMessage that calls one routing tool — what the LLM would emit."""
    return AIMessage(
        content="",
        tool_calls=[
            {
                "id": "tc-route-1",
                "name": tool_name,
                "args": {"query": query},
                "type": "tool_call",
            }
        ],
    )


async def _run_supervisor(responses: list[AIMessage], query: str) -> SsuAgentState:
    """Build the REAL supervisor ReAct agent (same routing tools + prompt the graph
    uses) driven by a fake LLM, run it on `query`, and return the resulting state."""
    llm = _FakeRoutingLLM(responses=responses)
    react = create_react_agent(llm, _make_routing_tools(), prompt=_SUPERVISOR_PROMPT)
    result = await react.ainvoke({"messages": [HumanMessage(content=query)]})
    # Shape a supervisor state around the produced messages for _post_supervisor.
    return {
        "messages": result["messages"],
        "mcp_session_id": "eval-session",
        "active_agent": None,
    }


# ── Parametrized routing eval ─────────────────────────────────────────────────


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("query", "tool_name", "expected_agent"),
    [
        ("도서관 자리 잡아줘", "transfer_to_library_agent", "library_agent"),
        ("빈 좌석 알려줘", "transfer_to_library_agent", "library_agent"),
        ("장학금 기준 알려줘", "transfer_to_academic_agent", "academic_agent"),
        ("졸업요건 확인해줘", "transfer_to_academic_agent", "academic_agent"),
        ("LMS 과제 확인해줘", "transfer_to_lms_agent", "lms_agent"),
        ("강의 자료 내려받아줘", "transfer_to_lms_agent", "lms_agent"),
    ],
)
async def test_eval_routing_tool_drives_correct_node(
    query: str, tool_name: str, expected_agent: str
) -> None:
    """When the LLM calls a transfer tool, the real ReAct + parser chain routes to
    the matching sub-agent node. Fails if the routing tool emits the wrong marker,
    the prefix drifts, or the tool is not registered on the supervisor."""
    state = await _run_supervisor([_transfer_call(tool_name, query)], query)

    # The real routing tool actually executed and produced its marker.
    markers = [
        msg.content
        for msg in state["messages"]
        if isinstance(getattr(msg, "content", None), str)
        and f"{_ROUTE_PREFIX}{expected_agent}" in msg.content
    ]
    assert markers, f"expected a {_ROUTE_PREFIX}{expected_agent} marker for {tool_name!r}"

    cmd = _post_supervisor(state)
    assert cmd.goto == expected_agent, f"'{query}' should route to {expected_agent}"
    assert cmd.update["active_agent"] == expected_agent


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("query", "answer"),
    [
        ("오늘 학식 뭐야", "오늘 학식은 제육볶음입니다."),
        ("캠퍼스 시설 안내해줘", "도서관, 체육관 등이 있습니다."),
        ("안녕", "안녕하세요! 무엇을 도와드릴까요?"),
    ],
)
async def test_eval_direct_answer_ends_without_routing(query: str, answer: str) -> None:
    """When the LLM answers directly (no transfer tool call), no marker is emitted
    and the supervisor ends without handing off to a sub-agent."""
    state = await _run_supervisor([AIMessage(content=answer)], query)

    assert not any(
        _ROUTE_PREFIX in getattr(msg, "content", "")
        for msg in state["messages"]
        if isinstance(getattr(msg, "content", None), str)
    ), "a direct answer must not produce a routing marker"

    cmd = _post_supervisor(state)
    assert cmd.goto is END, f"'{query}' should end without sub-agent routing"


# ── Marker-parser edge cases (real parser, not tautological) ───────────────────


def test_post_supervisor_unknown_marker_goes_to_end() -> None:
    """No routing marker anywhere in recent messages → END."""
    state: SsuAgentState = {
        "messages": [
            HumanMessage(content="???"),
            AIMessage(content="알 수 없는 요청"),
        ],
        "mcp_session_id": None,
        "active_agent": None,
    }
    assert _post_supervisor(state).goto is END


def test_post_supervisor_marker_survives_surrounding_text() -> None:
    """A marker embedded in longer tool output is still extracted."""
    from langchain_core.messages import ToolMessage

    state: SsuAgentState = {
        "messages": [
            HumanMessage(content="도서관 예약"),
            AIMessage(content=""),
            ToolMessage(
                content=f"some preamble {_ROUTE_PREFIX}library_agent trailing text",
                tool_call_id="tc-embed",
            ),
        ],
        "mcp_session_id": None,
        "active_agent": None,
    }
    assert _post_supervisor(state).goto == "library_agent"
