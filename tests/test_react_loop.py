"""
Tests for the shared sub-agent ReAct loop (ssu_agent.agents.react_loop).

Focus: the latency fix — a turn's tool calls run concurrently (asyncio.gather),
tool results stay ordered, and the loop is bounded by _MAX_TOOL_TURNS.
"""

from __future__ import annotations

import asyncio
import time

import pytest
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import tool

from ssu_agent.agents.react_loop import _MAX_TOOL_TURNS, run_react_loop
from ssu_agent.supervisor.state import SsuAgentState


class _SeqLLM(FakeMessagesListChatModel):
    """Fake model returning a fixed response sequence; bind_tools is a no-op."""

    def bind_tools(self, tools, **kwargs):
        return self


def _state() -> SsuAgentState:
    return {
        "messages": [HumanMessage(content="질문")],
        "mcp_session_id": None,
        "active_agent": "x",
    }


@pytest.mark.asyncio
async def test_tool_calls_in_a_turn_run_concurrently():
    """Two 0.2s tools in one turn should finish in ~0.2s (parallel), not ~0.4s."""

    @tool
    async def slow_a(x: str) -> str:
        """slow tool a"""
        await asyncio.sleep(0.2)
        return "a-done"

    @tool
    async def slow_b(x: str) -> str:
        """slow tool b"""
        await asyncio.sleep(0.2)
        return "b-done"

    llm = _SeqLLM(
        responses=[
            AIMessage(
                content="",
                tool_calls=[
                    {"name": "slow_a", "args": {"x": "1"}, "id": "c1"},
                    {"name": "slow_b", "args": {"x": "2"}, "id": "c2"},
                ],
            ),
            AIMessage(content="두 조회를 종합한 답변입니다."),
        ]
    )

    started = time.perf_counter()
    result = await run_react_loop(
        [llm], [slow_a, slow_b], "시스템 프롬프트", "테스트", _state(), {}
    )
    elapsed = time.perf_counter() - started

    assert elapsed < 0.4, f"tools ran sequentially ({elapsed:.2f}s)"
    assert "[테스트]" in result["messages"][-1].content
    assert result["active_agent"] is None


@pytest.mark.asyncio
async def test_loop_is_bounded_by_max_tool_turns():
    """A model that always emits a tool call must stop at _MAX_TOOL_TURNS."""
    calls = {"n": 0}

    @tool
    async def counter(x: str) -> str:
        """counting tool"""
        calls["n"] += 1
        return "ok"

    # Single response with a tool call → FakeMessagesListChatModel repeats it,
    # so the loop only stops when it hits the turn cap.
    llm = _SeqLLM(
        responses=[
            AIMessage(
                content="",
                tool_calls=[{"name": "counter", "args": {"x": "1"}, "id": "c1"}],
            )
        ]
    )

    result = await run_react_loop([llm], [counter], "시스템", "테스트", _state(), {})

    assert calls["n"] == _MAX_TOOL_TURNS
    # No AIMessage carried content, so the loop returns the fallback tag.
    assert result["messages"][-1].content == "[테스트] 처리 완료"
