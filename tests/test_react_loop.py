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
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool
from pydantic import Field

from ssu_agent.agents.react_loop import (
    _MAX_TOOL_TURNS,
    EMPTY_RESPONSE_FALLBACK,
    drop_routing_messages,
    latest_turn_messages,
    run_react_loop,
)
from ssu_agent.supervisor.state import SsuAgentState


class _SeqLLM(FakeMessagesListChatModel):
    """Fake model returning a fixed response sequence; bind_tools is a no-op."""

    def bind_tools(self, tools, **kwargs):
        return self


class _CapturingSeqLLM(_SeqLLM):
    seen_inputs: list[list] = Field(default_factory=list)

    async def ainvoke(self, input, config=None, **kwargs):
        self.seen_inputs.append(list(input) if isinstance(input, list) else input)
        return await super().ainvoke(input, config=config, **kwargs)


def _state() -> SsuAgentState:
    return {
        "messages": [HumanMessage(content="질문")],
        "mcp_session_id": None,
        "active_agent": "x",
    }


def test_drop_routing_messages_removes_supervisor_ai_message():
    messages = [
        HumanMessage(content="도서관 좌석 예약해줘"),
        AIMessage(content="도서관 에이전트에게 전달했습니다.", name="supervisor"),
        AIMessage(
            content="",
            tool_calls=[
                {
                    "id": "route-1",
                    "name": "transfer_to_library_agent",
                    "args": {"query": "도서관 좌석 예약해줘"},
                    "type": "tool_call",
                }
            ],
        ),
        ToolMessage(content="ROUTE_TO:library_agent", tool_call_id="route-1"),
        AIMessage(content="[도서관 에이전트] 실제 답변"),
    ]

    cleaned = drop_routing_messages(messages)

    assert [getattr(m, "content", "") for m in cleaned] == [
        "도서관 좌석 예약해줘",
        "[도서관 에이전트] 실제 답변",
    ]


def test_drop_routing_messages_preserves_completed_supervisor_turns():
    messages = [
        HumanMessage(content="도서관 5층 빈 자리 있어?"),
        AIMessage(content="[도서관 에이전트] 도서관 로그인 후 확인할 수 있습니다."),
        HumanMessage(content="오늘 학식 뭐야?"),
        AIMessage(
            content="",
            name="supervisor",
            tool_calls=[
                {
                    "id": "meal-1",
                    "name": "get_today_meal",
                    "args": {},
                    "type": "tool_call",
                }
            ],
        ),
        ToolMessage(content="오늘 학식: 제육볶음", tool_call_id="meal-1"),
        AIMessage(content="오늘 학식은 제육볶음입니다.", name="supervisor"),
        HumanMessage(content="내 졸업요건 알려줘"),
        AIMessage(
            content="학사 에이전트에게 전달하겠습니다.",
            name="supervisor",
            tool_calls=[
                {
                    "id": "route-academic-1",
                    "name": "transfer_to_academic_agent",
                    "args": {"query": "내 졸업요건 알려줘"},
                    "type": "tool_call",
                }
            ],
        ),
        ToolMessage(content="ROUTE_TO:academic_agent", tool_call_id="route-academic-1"),
        AIMessage(content="학사 에이전트로 전환했습니다.", name="supervisor"),
    ]

    cleaned = drop_routing_messages(messages)

    assert [getattr(m, "content", "") for m in cleaned] == [
        "도서관 5층 빈 자리 있어?",
        "[도서관 에이전트] 도서관 로그인 후 확인할 수 있습니다.",
        "오늘 학식 뭐야?",
        "",
        "오늘 학식: 제육볶음",
        "오늘 학식은 제육볶음입니다.",
        "내 졸업요건 알려줘",
    ]


def test_latest_turn_messages_drops_unrelated_completed_turns():
    messages = [
        HumanMessage(content="오늘 학식 뭐야?"),
        AIMessage(content="오늘 학식은 비빔밥입니다.", name="supervisor"),
        HumanMessage(content="내 졸업요건 알려줘"),
    ]

    assert latest_turn_messages(messages, agent_tag="학사 에이전트") == [messages[-1]]


def test_latest_turn_messages_keeps_immediately_previous_same_agent_turn():
    messages = [
        HumanMessage(content="내 졸업요건 알려줘"),
        AIMessage(content="[학사 에이전트] 전공 학점이 부족해요."),
        HumanMessage(content="몇 학점 남았어?"),
    ]

    assert latest_turn_messages(messages, agent_tag="학사 에이전트") == messages


def test_latest_turn_messages_uses_agent_name_instead_of_display_prefix():
    messages = [
        HumanMessage(content="내 성적 알려줘"),
        AIMessage(content="어느 학기를 볼까요?", name="academic_agent"),
        HumanMessage(content="지난학기요"),
    ]

    assert latest_turn_messages(messages, agent_tag="학사 에이전트") == messages


def test_latest_turn_messages_drops_previous_different_named_agent_turn():
    messages = [
        HumanMessage(content="도서관 빈자리 알려줘"),
        AIMessage(content="5층에 17석 있어요.", name="library_agent"),
        HumanMessage(content="지난학기요"),
    ]

    assert latest_turn_messages(messages, agent_tag="학사 에이전트") == [messages[-1]]


def test_drop_routing_messages_handles_mixed_public_and_transfer_calls():
    messages = [
        HumanMessage(content="오늘 학식과 내 졸업요건 알려줘"),
        AIMessage(
            content="학식은 조회하고 학사 에이전트로 전환하겠습니다.",
            name="supervisor",
            tool_calls=[
                {
                    "id": "meal-mixed-1",
                    "name": "get_today_meal",
                    "args": {},
                    "type": "tool_call",
                },
                {
                    "id": "route-mixed-1",
                    "name": "transfer_to_academic_agent",
                    "args": {"query": "내 졸업요건 알려줘"},
                    "type": "tool_call",
                },
            ],
        ),
        ToolMessage(content="오늘 학식: 제육볶음", tool_call_id="meal-mixed-1"),
        ToolMessage(content="ROUTE_TO:academic_agent", tool_call_id="route-mixed-1"),
        AIMessage(content="학사 에이전트로 전환했습니다.", name="supervisor"),
    ]

    cleaned = drop_routing_messages(messages)

    assert [getattr(m, "content", "") for m in cleaned] == [
        "오늘 학식과 내 졸업요건 알려줘",
        "",
        "오늘 학식: 제육볶음",
    ]
    assert isinstance(cleaned[1], AIMessage)
    assert [tc["name"] for tc in cleaned[1].tool_calls] == ["get_today_meal"]


def test_drop_routing_messages_scopes_reused_call_ids_to_their_user_turn():
    messages = [
        HumanMessage(content="내 졸업요건 알려줘"),
        AIMessage(
            content="",
            name="supervisor",
            tool_calls=[
                {
                    "id": "reused-call-id",
                    "name": "transfer_to_academic_agent",
                    "args": {"query": "내 졸업요건 알려줘"},
                    "type": "tool_call",
                }
            ],
        ),
        ToolMessage(content="ROUTE_TO:academic_agent", tool_call_id="reused-call-id"),
        AIMessage(content="[학사 에이전트] 로그인 후 확인할 수 있습니다."),
        HumanMessage(content="오늘 학식 뭐야?"),
        AIMessage(
            content="",
            name="supervisor",
            tool_calls=[
                {
                    "id": "reused-call-id",
                    "name": "get_today_meal",
                    "args": {},
                    "type": "tool_call",
                }
            ],
        ),
        ToolMessage(content="오늘 학식: 제육볶음", tool_call_id="reused-call-id"),
        AIMessage(content="오늘 학식은 제육볶음입니다.", name="supervisor"),
    ]

    cleaned = drop_routing_messages(messages)

    assert "ROUTE_TO:academic_agent" not in [getattr(m, "content", "") for m in cleaned]
    assert "오늘 학식: 제육볶음" in [getattr(m, "content", "") for m in cleaned]


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
    assert result["messages"][-1].id is None


@pytest.mark.asyncio
async def test_terminal_tool_result_skips_an_extra_model_round():
    calls = {"n": 0}

    @tool
    async def create_download() -> str:
        """Create a browser download URL."""
        calls["n"] += 1
        return '{"downloadUrl":"https://example.com/download?token=test"}'

    llm = _SeqLLM(
        responses=[
            AIMessage(
                content="",
                tool_calls=[{"name": "create_download", "args": {}, "id": "download-1"}],
            )
        ]
    )

    result = await run_react_loop(
        [llm],
        [create_download],
        "시스템",
        "테스트",
        _state(),
        {},
        terminal_tool_result_formatter=lambda name, _content: (
            "[파일 다운로드](https://example.com/download?token=test)"
            if name == "create_download"
            else None
        ),
    )

    assert calls["n"] == 1
    assert result["messages"][-1].content == (
        "[테스트] [파일 다운로드](https://example.com/download?token=test)"
    )
    assert result["messages"][-1].id is None


@pytest.mark.asyncio
async def test_empty_final_content_uses_fallback():
    llm = _SeqLLM(responses=[AIMessage(content=" \n ", id="blank-final")])

    result = await run_react_loop([llm], [], "시스템", "테스트", _state(), {})

    assert result["messages"][-1].content == f"[테스트] {EMPTY_RESPONSE_FALLBACK}"
    assert result["messages"][-1].id is None
    assert result["messages"][-1].id != "blank-final"


@pytest.mark.asyncio
async def test_non_empty_final_content_is_untouched():
    llm = _SeqLLM(responses=[AIMessage(content="정상 답변입니다.")])

    result = await run_react_loop([llm], [], "시스템", "테스트", _state(), {})

    assert result["messages"][-1].content == "[테스트] 정상 답변입니다."


@pytest.mark.asyncio
async def test_content_block_final_answer_is_flattened_and_reuses_model_id():
    llm = _SeqLLM(
        responses=[
            AIMessage(
                content=[{"type": "text", "text": "졸업 요건은 130학점입니다.", "index": 0}],
                id="claude-final-1",
            )
        ]
    )

    result = await run_react_loop([llm], [], "시스템", "학사 에이전트", _state(), {})
    tagged = result["messages"][-1]

    assert tagged.content == "[학사 에이전트] 졸업 요건은 130학점입니다."
    assert tagged.id == "claude-final-1"
    assert "{" not in tagged.content


@pytest.mark.asyncio
async def test_structured_invalid_session_stops_before_model_synthesis():
    @tool
    def private_lookup() -> str:
        """Private lookup fixture."""
        return (
            '{"status":"INVALID_SESSION","mcpSessionId":null,'
            '"developerMessage":"Call start_auth with mcp_session_id"}'
        )

    llm = _SeqLLM(
        responses=[
            AIMessage(
                content="",
                tool_calls=[{"name": "private_lookup", "args": {}, "id": "private-1"}],
            ),
            AIMessage(content="MCP session ID를 알려주세요."),
        ]
    )

    result = await run_react_loop(
        [llm],
        [private_lookup],
        "시스템",
        "학사 에이전트",
        _state(),
        {},
        auth_required_message="화면 상단의 연결에서 다시 연결해 주세요.",
    )

    assert result["messages"][-1].content == (
        "[학사 에이전트] 화면 상단의 연결에서 다시 연결해 주세요."
    )


@pytest.mark.asyncio
async def test_auth_status_name_inside_normal_data_does_not_trigger_guard():
    @tool
    def public_policy() -> str:
        """Public policy fixture."""
        return '{"status":"OK","data":{"note":"AUTH_REQUIRED is a documented status"}}'

    llm = _SeqLLM(
        responses=[
            AIMessage(
                content="",
                tool_calls=[{"name": "public_policy", "args": {}, "id": "public-1"}],
            ),
            AIMessage(content="공개 정책 답변입니다."),
        ]
    )

    result = await run_react_loop(
        [llm],
        [public_policy],
        "시스템",
        "학사 에이전트",
        _state(),
        {},
        auth_required_message="연결해 주세요.",
    )

    assert result["messages"][-1].content == "[학사 에이전트] 공개 정책 답변입니다."


@pytest.mark.asyncio
async def test_tool_exception_details_never_reach_next_model_turn():
    @tool
    def failing_lookup() -> str:
        """Fail with a sensitive upstream detail."""
        raise RuntimeError("backend rejected session raw-exception-secret")

    llm = _CapturingSeqLLM(
        responses=[
            AIMessage(
                content="",
                tool_calls=[{"name": "failing_lookup", "args": {}, "id": "failure-1"}],
            ),
            AIMessage(content="조회에 실패했습니다."),
        ]
    )

    await run_react_loop([llm], [failing_lookup], "시스템", "테스트", _state(), {})

    second_turn = repr(llm.seen_inputs[1])
    assert "raw-exception-secret" not in second_turn
    assert "Tool error: upstream tool failed." in second_turn


@pytest.mark.asyncio
async def test_cross_agent_history_is_removed_before_model_invocation():
    session_id = "library-history-secret"
    auth_url = "https://ssumcp.duckdns.org/api/mcp/auth/library/start?state=secret"
    state: SsuAgentState = {
        "messages": [
            HumanMessage(content="도서관 5층 빈 자리 있어?"),
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "get_my_library_seat",
                        "args": {"mcp_session_id": session_id},
                        "id": "library-history-1",
                    }
                ],
            ),
            ToolMessage(
                content=(
                    '{"status":"OK","mcpSessionId":"library-history-secret",'
                    '"loginUrl":"https://ssumcp.duckdns.org/api/mcp/auth/library/'
                    'start?state=secret","data":{"available":8}}'
                ),
                tool_call_id="library-history-1",
            ),
            AIMessage(content=f"로그인: {auth_url}"),
            HumanMessage(content="일반 졸업 기준 알려줘"),
        ],
        "mcp_session_id": session_id,
        "library_connected": True,
        "active_agent": "academic",
    }
    llm = _CapturingSeqLLM(responses=[AIMessage(content="공개 기준 답변")])

    await run_react_loop([llm], [], "시스템", "학사 에이전트", state, {})

    model_input = repr(llm.seen_inputs[0])
    assert session_id not in model_input
    assert auth_url not in model_input
    assert "mcp_session_id" not in model_input
    assert '"available": 8' not in model_input
    assert "도서관 5층" not in model_input
    assert "일반 졸업 기준 알려줘" in model_input
