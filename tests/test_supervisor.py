"""
Tests for the supervisor graph — routing, state, and tool categorisation.

Design: All tests use mock LLM and mock MCP tools so no real network call is made.
MemorySaver provides in-memory checkpoint isolation.
"""

from __future__ import annotations

import pytest
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import tool

from ssu_agent.supervisor.graph import build_supervisor_graph, categorise_tools
from ssu_agent.supervisor.state import SsuAgentState

# ── Fixtures ──────────────────────────────────────────────────────────────────


@tool
def get_today_meal() -> str:
    """오늘 학식 조회"""
    return "오늘 학식: 제육볶음"


@tool
def get_my_grades(mcp_session_id: str) -> str:
    """성적 조회"""
    return '{"grades": []}'


@tool
def get_my_assignments(mcp_session_id: str) -> str:
    """과제 조회"""
    return '{"items": []}'


@tool
def get_lms_dashboard(mcp_session_id: str) -> str:
    """LMS dashboard"""
    return '{"dashboard": []}'


@tool
def get_library_available_seats() -> str:
    """도서관 좌석 현황"""
    return '{"floors": []}'


@tool
def prepare_reserve_library_seat(mcp_session_id: str, seat_id: int) -> str:
    """예약 준비"""
    return '{"status": "OK", "data": {"actionId": 42, "seatLabel": "A-001"}}'


@tool
def confirm_action(mcp_session_id: str, action_id: int) -> str:
    """예약 확정"""
    return '{"status": "OK"}'


@tool
def start_auth(provider: str) -> str:
    """인증 시작"""
    return '{"loginUrl": "https://example.com/login"}'


@tool
def get_my_lms_courses(mcp_session_id: str) -> str:
    """수강 과목 목록 조회"""
    return '{"courses": []}'


@tool
def get_my_lms_materials(mcp_session_id: str) -> str:
    """비영상 자료 목록 조회"""
    return '{"materials": []}'


@tool
def prepare_lms_material_export(mcp_session_id: str) -> str:
    """내보낼 자료 검증 및 확인 요청 생성"""
    return '{"status": "OK"}'


@tool
def confirm_lms_material_export(mcp_session_id: str) -> str:
    """확인 후 ZIP 내보내기 시작"""
    return '{"downloadUrl": "https://example.com/download"}'


MOCK_TOOLS = [
    get_today_meal,
    get_my_grades,
    get_my_assignments,
    get_lms_dashboard,
    get_library_available_seats,
    prepare_reserve_library_seat,
    confirm_action,
    start_auth,
    get_my_lms_courses,
    get_my_lms_materials,
    prepare_lms_material_export,
    confirm_lms_material_export,
]


# ── Unit: tool categorisation ─────────────────────────────────────────────────


def test_categorise_splits_library_tools():
    cats = categorise_tools(MOCK_TOOLS)
    lib_names = {t.name for t in cats["library"]}
    assert "get_library_available_seats" in lib_names
    assert "prepare_reserve_library_seat" in lib_names
    assert "confirm_action" in lib_names


def test_categorise_splits_academic_tools():
    cats = categorise_tools(MOCK_TOOLS)
    academic_names = {t.name for t in cats["academic"]}
    assert "get_my_grades" in academic_names


def test_categorise_splits_lms_tools():
    cats = categorise_tools(MOCK_TOOLS)
    lms_names = {t.name for t in cats["lms"]}
    library_names = {t.name for t in cats["library"]}

    assert "get_my_assignments" in lms_names
    assert "get_lms_dashboard" in lms_names
    # New LMS export tools must reach LMS agent
    assert "get_my_lms_courses" in lms_names
    assert "get_my_lms_materials" in lms_names
    assert "prepare_lms_material_export" in lms_names  # THE BUG FIX TEST
    assert "confirm_lms_material_export" in lms_names

    # Library tools must NOT be mis-routed
    assert "prepare_reserve_library_seat" in library_names
    assert "confirm_action" in library_names

    # prepare_lms_material_export must NOT be in library (the bug this PR fixes)
    assert "prepare_lms_material_export" not in library_names


def test_categorise_public_tools():
    cats = categorise_tools(MOCK_TOOLS)
    pub_names = {t.name for t in cats["public"]}
    assert "get_today_meal" in pub_names


def test_categorise_auth_tools():
    cats = categorise_tools(MOCK_TOOLS)
    auth_names = {t.name for t in cats["auth"]}
    assert "start_auth" in auth_names


# ── Unit: SsuAgentState structure ─────────────────────────────────────────────


def test_state_has_required_keys():
    state: SsuAgentState = {
        "messages": [HumanMessage(content="안녕")],
        "mcp_session_id": "test-session",
        "active_agent": None,
        "pending_action": None,
    }
    assert state["mcp_session_id"] == "test-session"
    assert state["active_agent"] is None


# ── Integration: graph builds and runs with mock LLM ──────────────────────────


class _MockLLM(FakeMessagesListChatModel):
    """Fake LLM that always returns a direct answer (no tool calls, no routing)."""

    def bind_tools(self, tools, **kwargs):
        return self


def _make_mock_llm() -> _MockLLM:
    return _MockLLM(
        responses=[
            AIMessage(content="테스트 응답: 오늘 학식은 제육볶음입니다."),
            AIMessage(content="테스트 응답: 오늘 학식은 제육볶음입니다."),
        ]
    )


@pytest.mark.asyncio
async def test_graph_builds_with_mock_llm():
    """Graph compiles without error using mock tools and MemorySaver."""
    from langgraph.checkpoint.memory import MemorySaver

    graph = await build_supervisor_graph(
        all_tools=MOCK_TOOLS,
        llm=_make_mock_llm(),
        checkpointer=MemorySaver(),
    )
    assert graph is not None


@pytest.mark.asyncio
async def test_graph_initial_state_has_mcp_session():
    """State is correctly initialised with mcp_session_id from request."""
    from langgraph.checkpoint.memory import MemorySaver

    graph = await build_supervisor_graph(
        all_tools=MOCK_TOOLS,
        llm=_make_mock_llm(),
        checkpointer=MemorySaver(),
    )
    config = {"configurable": {"thread_id": "thread-test-001"}}
    result = await graph.ainvoke(
        {
            "messages": [HumanMessage(content="안녕")],
            "mcp_session_id": "session-abc",
            "active_agent": None,
            "pending_action": None,
        },
        config=config,
    )
    # State should carry mcp_session_id through
    assert result.get("mcp_session_id") == "session-abc"


# ── Integration: routing markers ──────────────────────────────────────────────


def test_route_marker_regex():
    """post_supervisor correctly extracts routing markers from messages."""
    from langchain_core.messages import ToolMessage

    from ssu_agent.supervisor.graph import _ROUTE_PREFIX, _post_supervisor

    state: SsuAgentState = {
        "messages": [
            HumanMessage(content="도서관 좌석 찾아줘"),
            AIMessage(content=""),
            ToolMessage(content=f"{_ROUTE_PREFIX}library_agent", tool_call_id="tc-1"),
        ],
        "mcp_session_id": None,
        "active_agent": None,
        "pending_action": None,
    }
    cmd = _post_supervisor(state)
    assert cmd.goto == "library_agent"
    assert cmd.update["active_agent"] == "library_agent"


def test_no_route_marker_goes_to_end():
    """post_supervisor routes to END when no marker is present."""
    from ssu_agent.supervisor.graph import _post_supervisor

    state: SsuAgentState = {
        "messages": [
            HumanMessage(content="오늘 학식 뭐야"),
            AIMessage(content="오늘 학식은 제육볶음입니다."),
        ],
        "mcp_session_id": None,
        "active_agent": None,
        "pending_action": None,
    }
    from langgraph.graph import END

    cmd = _post_supervisor(state)
    assert cmd.goto is END
