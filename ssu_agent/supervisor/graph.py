"""
Supervisor graph — multi-agent router for ssuAgent.

Architecture: Custom StateGraph with routing-marker pattern.

Why NOT create_react_agent with Command-returning tools:
  LangGraph 1.2.4's create_react_agent does not propagate Command returns from
  tool functions to the parent graph. This means handoff tools cannot directly
  transition the state machine. (Verified: inspect.getsource shows no Command
  handling in the prebuilt agent executor for this version.)

Why NOT pure conditional-edges on supervisor LLM output:
  Fragile string parsing. Structured output with Pydantic + a routing node is
  cleaner and gives us a single typed decision object.

Chosen pattern — "Route Marker + Post-Router":
  1. supervisor_react: create_react_agent with public tools (meal, notice,
     campus, auth) + lightweight routing tools that return a "ROUTE_TO:X" marker.
     The marker tools do NO work; they're lightweight signals for step 2.
  2. post_supervisor_node: scans state for routing markers and returns
     Command(goto=target) to transition the parent graph to a sub-agent node.
  3. Sub-agent nodes (library_agent, academic_agent, lms_agent) are embedded as
     compiled subgraphs. Interrupt() inside a subgraph node propagates up through
     the parent graph correctly — this is why sub-agents must be NODES, not tool
     invocations.

State flow:
  START → supervisor_react → post_supervisor
                                  ↓ (routing marker found)
        library_agent / academic_agent / lms_agent → END
                                  ↓ (no marker: supervisor answered directly)
                                 END

Parent-Child State design:
  All nodes share SsuAgentState (single TypedDict). The messages channel uses
  add_messages reducer so all agents append to the same conversation thread.
  active_agent is set by post_supervisor and cleared by sub-agents on return.

MCP session lifecycle:
  thread_id (LangGraph checkpoint key) maps 1:1 with a FastAPI client
  connection. mcp_session_id (ssuMCP private tool auth) is passed in the initial
  request body and stored in SsuAgentState. Sub-agents receive it via state and
  include it in private tool calls as instructed by their system prompts.

Streaming:
  FastAPI calls graph.astream_events(version="v2") and filters:
  - on_chat_model_stream → text chunks (token-by-token output)
  - on_tool_start where name starts with "transfer_to_" → handoff status UX
  - on_interrupt → HITL payload for library approval
"""

from __future__ import annotations

import re

from langchain_core.language_models import BaseChatModel
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool, tool
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import create_react_agent
from langgraph.types import Command

from ssu_agent.agents.academic import build_academic_agent
from ssu_agent.agents.library import build_library_agent
from ssu_agent.agents.lms import build_lms_agent
from ssu_agent.llm_factory import get_llm_sequence
from ssu_agent.supervisor.state import SsuAgentState

# ── Tool-name categorisation ──────────────────────────────────────────────────

_LIBRARY_PREFIXES = (
    "get_library",
    "recommend_library",
    "search_library",
    "get_my_library",
    "wait_for_library",
    "get_library_wait",
    "cancel_library_wait",
    "get_room_available",
)
_LIBRARY_NAMES = {
    "prepare_reserve_library_seat",
    "prepare_swap_library_seat",
    "prepare_cancel_library_seat",
    "confirm_action",
}
_ACADEMIC_NAMES = {
    "get_my_grades",
    "check_graduation_requirements",
    "simulate_gpa",
    "evaluate_graduation_with_policy",
    "classify_academic_question",
    "search_academic_policy_sources",
    "get_academic_policy_brief",
    "check_scholarship_policy",
    "list_academic_policy_sources",
    "get_academic_calendar",
    "find_academic_calendar_events",
    "get_my_schedule",
    "get_my_chapel_info",
    "get_my_scholarships",
}
_LMS_NAMES = {
    "get_my_assignments",
    "get_my_lms_terms",
    "get_lms_dashboard",
    # new in Phase C
    "get_my_lms_courses",
    "get_my_lms_materials",
    "prepare_lms_material_export",
    "confirm_lms_material_export",
}
_AUTH_NAMES = {"start_auth", "get_auth_status", "logout_provider", "logout_all"}


def categorise_tools(all_tools: list[BaseTool]) -> dict[str, list[BaseTool]]:
    """Split MCP tools into domain buckets used by the supervisor and sub-agents."""
    cats: dict[str, list[BaseTool]] = {
        "library": [],
        "academic": [],
        "lms": [],
        "auth": [],
        "public": [],
    }
    for t in all_tools:
        name = t.name
        if any(name.startswith(p) for p in _LIBRARY_PREFIXES) or name in _LIBRARY_NAMES:
            cats["library"].append(t)
        elif name in _ACADEMIC_NAMES:
            cats["academic"].append(t)
        elif name in _LMS_NAMES:
            cats["lms"].append(t)
        elif name in _AUTH_NAMES:
            cats["auth"].append(t)
        else:
            cats["public"].append(t)
    return cats


# ── Routing tools (lightweight markers) ──────────────────────────────────────

_ROUTE_PREFIX = "ROUTE_TO:"


def _make_routing_tools() -> list[BaseTool]:
    @tool
    def transfer_to_library_agent(query: str) -> str:
        """Transfer to Library Agent.

        Use for: seat availability, seat recommendation, book search,
        loan status, library seat reservation, swap, or cancellation requests.
        Provide `query` with the user's specific request.
        """
        return f"{_ROUTE_PREFIX}library_agent"

    @tool
    def transfer_to_academic_agent(query: str) -> str:
        """Transfer to Academic Agent.

        Use for: grades, GPA simulation, graduation requirements, academic
        calendar, chapel attendance, scholarships, and academic policy questions
        (credits, graduation criteria, scholarship eligibility).
        Provide `query` with the user's specific request.
        """
        return f"{_ROUTE_PREFIX}academic_agent"

    @tool
    def transfer_to_lms_agent(query: str) -> str:
        """Transfer to LMS Agent.

        Use for: assignments, LMS terms, deadlines, LMS dashboard
        (학사 대시보드 - 과제 마감·시험 일정·공지 통합 조회), LMS course list (과목 목록 조회),
        LMS materials list (강의자료 목록 조회), and LMS non-video material ZIP export
        (비영상 자료 ZIP 내보내기).
        Provide `query` with the user's specific request.
        """
        return f"{_ROUTE_PREFIX}lms_agent"

    return [transfer_to_library_agent, transfer_to_academic_agent, transfer_to_lms_agent]


_ROUTE_RE = re.compile(r"ROUTE_TO:(\w+)")

_SUPERVISOR_PROMPT = """당신은 숭실대학교 AI 어시스턴트입니다.

역할:
1. 식단(meal), 공지(notice), 캠퍼스 시설(facility), 인증(auth) 관련
   간단한 질문은 직접 도구를 호출해 답합니다.
2. 도서관(library), 학사(academic), LMS 관련 전문 질문은 해당 에이전트로 전달합니다:
   - 도서관 좌석/예약/도서 → transfer_to_library_agent
   - 성적/졸업/장학/학칙 → transfer_to_academic_agent
   - LMS 과제, 마감일, 시험 일정, 공지사항(대시보드), 강의자료 조회 및
     비영상 자료 ZIP 내보내기(LMS 다운로드) → transfer_to_lms_agent

LMS 강의자료 내보내기 플로우 안내:
사용자가 LMS 강의자료 다운로드나 내보내기를 요청하면 transfer_to_lms_agent로
전달하십시오. 해당 에이전트는 다음 단계를 거칩니다:
get_my_lms_courses → get_my_lms_materials → prepare_lms_material_export
→ confirm_lms_material_export → 다운로드 링크 제공 (유효기간 20분).

전달 시 사용자의 원래 질문을 query에 그대로 포함하세요.
이미 하위 에이전트 답변([도서관/학사/LMS 에이전트])이 대화에 있다면
별도 도구 호출 없이 답변을 요약해 사용자에게 전달하세요.
"""


# ── Post-supervisor routing node ──────────────────────────────────────────────


def _post_supervisor(state: SsuAgentState) -> Command:
    """Check if the supervisor's last tool result is a routing marker."""
    for msg in reversed(state["messages"][-8:]):
        content = getattr(msg, "content", "")
        if isinstance(content, str):
            m = _ROUTE_RE.search(content)
            if m:
                target = m.group(1)
                return Command(goto=target, update={"active_agent": target})
    return Command(goto=END)


# ── Graph builder ─────────────────────────────────────────────────────────────


async def build_supervisor_graph(
    all_tools: list[BaseTool] | None = None,
    llm: BaseChatModel | None = None,
    checkpointer=None,
):
    """Build and compile the full multi-agent supervisor graph.

    Args:
        all_tools: MCP tool list. Fetched from ssuMCP if None.
        llm: Override LLM (used in tests).
        checkpointer: LangGraph checkpointer. Caller owns its lifecycle. In prod
            an AsyncPostgresSaver (langgraph-checkpoint-postgres) is opened in the
            FastAPI lifespan and passed in (see main.py / ADR 003).
            If None, uses MemorySaver (no persistence — development only).

    Returns:
        Compiled StateGraph with the provided checkpointer.

    Checkpointer lifecycle note (important for HITL):
        The prod AsyncPostgresSaver is backed by a connection pool opened in the
        lifespan handler and kept alive for the app's lifetime. If the pool
        closes, HITL resume fails because the checkpoint can't be read.
    """
    from ssu_agent.mcp_client import create_mcp_client

    if all_tools is None:
        client = create_mcp_client()
        all_tools = await client.get_tools()

    if checkpointer is None:
        from langgraph.checkpoint.memory import MemorySaver

        checkpointer = MemorySaver()

    cats = categorise_tools(all_tools)
    routing_tools = _make_routing_tools()

    # Supervisor: public tools (meal/notice/campus) + auth + lightweight routing tools
    supervisor_tools = [*cats["public"], *cats["auth"], *routing_tools]
    llm_seq = [llm] if llm is not None else get_llm_sequence()

    async def supervisor_node(state: SsuAgentState, config: RunnableConfig) -> dict:
        last_exc: Exception | None = None
        for _llm in llm_seq:
            try:
                react = create_react_agent(_llm, supervisor_tools, prompt=_SUPERVISOR_PROMPT)
                result = await react.ainvoke({"messages": state["messages"]}, config=config)
                return {"messages": result["messages"]}
            except Exception as exc:
                last_exc = exc
        raise last_exc or RuntimeError("All LLM providers exhausted")

    # Sub-agent subgraphs — embedded as nodes so interrupt() propagates correctly
    library_subgraph = build_library_agent([*cats["library"], *cats["auth"]], llm).compile()
    academic_subgraph = build_academic_agent(
        [*cats["academic"], *cats["auth"]], llm
    ).compile()
    lms_subgraph = build_lms_agent([*cats["lms"], *cats["auth"]], llm).compile()

    # Parent graph assembly
    builder = StateGraph(SsuAgentState)

    builder.add_node("supervisor", supervisor_node)
    builder.add_node("post_supervisor", _post_supervisor)
    builder.add_node("library_agent", library_subgraph)
    builder.add_node("academic_agent", academic_subgraph)
    builder.add_node("lms_agent", lms_subgraph)

    builder.add_edge(START, "supervisor")
    builder.add_edge("supervisor", "post_supervisor")

    # post_supervisor returns Command(goto=target|END) — LangGraph handles routing
    builder.add_edge("library_agent", END)
    builder.add_edge("academic_agent", END)
    builder.add_edge("lms_agent", END)

    return builder.compile(checkpointer=checkpointer)
