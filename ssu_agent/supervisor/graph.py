"""
Supervisor graph — multi-agent router for ssuAgent.

Architecture: Custom StateGraph with routing-marker pattern.

Why retain marker tools instead of changing the handoff contract:
  The parent graph already owns domain transitions and persisted state. Keeping
  routing tools as side-effect-free marker producers lets the explicit
  post-supervisor node perform that transition without coupling the parent graph
  to the inner agent executor's Command semantics (ADR 0001).

Why NOT pure conditional-edges on supervisor LLM output:
  Fragile string parsing. Structured output with Pydantic + a routing node is
  cleaner and gives us a single typed decision object.

Chosen pattern — "Route Marker + Post-Router":
  1. supervisor_react: create_agent with public tools (meal, notice,
     campus) + lightweight routing tools that return a "ROUTE_TO:X" marker.
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
  connection. mcp_session_id (ssuMCP private tool auth) is stored in state but
  redacted at every model boundary. Auth lifecycle UX stays browser-owned;
  session-bound sub-agent tools inject the handle only during execution.

Streaming:
  FastAPI calls graph.astream_events(version="v2") and filters:
  - on_chat_model_stream → candidate answer text (routing/pre-tool narration is suppressed)
  - on_tool_start where name starts with "transfer_to_" → handoff status UX
  - on_chain_stream carrying __interrupt__ → HITL payload for library approval
    (langgraph 1.2.4 does not emit an on_interrupt event; see main._extract_interrupt)
"""

from __future__ import annotations

import logging
import re

from langchain.agents import create_agent
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool, tool
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command

from ssu_agent.agents.academic import build_academic_agent
from ssu_agent.agents.auth_guard import (
    contains_internal_auth_guidance,
    sanitize_messages_for_model,
    tools_for_model,
)
from ssu_agent.agents.library import build_library_agent
from ssu_agent.agents.lms import build_lms_agent
from ssu_agent.agents.react_loop import drop_routing_messages, latest_turn_messages
from ssu_agent.llm_factory import get_llm_sequence
from ssu_agent.supervisor.state import SsuAgentState
from ssu_agent.tool_results import content_to_text, sanitize_tool_pairing

logger = logging.getLogger(__name__)

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
    "export_all_lms_materials",
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
_LIBRARY_ROUTE_KEYWORDS = ("도서관", "열람실", "좌석")
_LMS_DIRECT_ROUTE_RE = re.compile(
    r"(?<![A-Za-z0-9])lms(?![A-Za-z0-9])|(?:강의|수업)\s*(?:파일|자료)"
    r"|(?:강의|수업).{0,12}(?:다운|내려받)"
    r"|(?:다운|내려받).{0,12}(?:강의|수업)",
    re.IGNORECASE,
)
_LMS_DIRECT_ROUTE_CONFLICTS = (
    "도서관",
    "열람실",
    "좌석",
    "예약",
    "성적",
    "졸업",
    "학점",
    "시간표",
    "등록금",
    "장학",
    "채플",
    "학식",
    "식단",
    "메뉴",
    "공지",
    "도서",
    "대출",
    "캠퍼스",
    "시설",
)
_LIBRARY_CONTINUATION_MAX_CHARS = 20
_LIBRARY_AGENT_PREFIX_RE = re.compile(r"^\s*\[도서관 에이전트\]\s*")
_LIBRARY_LOGIN_GATE_RE = re.compile(
    r"도서관.*로그인.*필요|로그인.*필요.*도서관|도서관.*로그인해 주세요"
)
_LIBRARY_COMPLETED_OUTCOME_MARKERS = (
    "예약 완료",
    "예약이 취소되었습니다",
    "예약 실패",
    "접수했습니다",
)
_STRONG_OTHER_DOMAIN_KEYWORDS = {
    "성적",
    "졸업",
    "학점",
    "수강",
    "시간표",
    "강의",
    "과제",
    "시험",
    "학식",
    "식단",
    "장학",
    "채플",
    "등록금",
}
_LIBRARY_CLARIFICATION_CUES = (
    "어디",
    "어느",
    "몇층",
    "몇 층",
    "열람실",
    "좌석",
    "자리",
    "선호",
    "원하",
    "괜찮",
)

_SUPERVISOR_PROMPT = """당신은 숭실대학교 AI 어시스턴트입니다.

역할:
1. 식단(meal), 공지(notice), 캠퍼스 시설(facility) 관련
   간단한 질문은 직접 도구를 호출해 답합니다.
2. 도서관(library), 학사(academic), LMS 관련 전문 질문은 해당 에이전트로 전달합니다:
   - 도서관 좌석/예약/도서 → transfer_to_library_agent
   - 성적/졸업/장학/학칙 → transfer_to_academic_agent
   - LMS 과제, 마감일, 시험 일정, 공지사항(대시보드), 강의자료 조회 및
     비영상 자료 ZIP 내보내기(LMS 다운로드) → transfer_to_lms_agent

LMS 강의자료 내보내기 플로우 안내:
사용자가 LMS 강의자료 다운로드나 내보내기를 요청하면 transfer_to_lms_agent로
전달하십시오. 전체 과목·모든 자료 요청은 export_all_lms_materials →
confirm_lms_material_export로 처리하고, 특정 과목 요청은 get_my_lms_courses →
prepare_lms_material_export → confirm_lms_material_export로 처리합니다.

전달 시 가장 최근 사용자 질문을 query에 그대로 포함하세요. 과거에 끝난 다른 요청이나 답변을
다시 요약하지 마세요. 현재 질문이 짧은 후속 표현일 때만 함께 제공된 직전 하위 에이전트 답변을
문맥으로 사용하세요.

라우팅 규칙(중요): 인사·잡담(예: "안녕", "뭐해")이나 도서관/학사/LMS 어디에도
해당하지 않는 범위 밖 질문은 transfer 도구를 호출하지 말고 당신이 직접 간단히
답하세요. 전문 에이전트는 실제로 해당 도메인 데이터가 필요할 때만 호출합니다.
인증 시작·세션 전달·로그인 URL 생성은 채팅 모델의 역할이 아닙니다. 사용자에게 내부 세션
값이나 서버 로그인 URL을 요청하거나 보여주지 마세요.
"""

_SUPERVISOR_AUTH_FALLBACK = (
    "학교 서비스 연결은 화면 상단의 ‘연결’에서 진행해 주세요. 로그인 정보나 내부 인증 값은 "
    "채팅에 입력하지 않아도 돼요."
)
_STANDALONE_DOMAIN_REQUEST_RE = re.compile(
    r"도서관|좌석|예약|대출|도서|성적|졸업|장학|채플|학점|gpa|"
    r"lms|과제|강의자료|수업자료|다운로드|학식|메뉴|식당|공지|캠퍼스|시설",
    re.IGNORECASE,
)


def _supervisor_model_messages(messages: list) -> list:
    latest_text = _latest_human_message_text(messages).strip()
    include_previous_turn = len(latest_text) <= 40 and not _STANDALONE_DOMAIN_REQUEST_RE.search(
        latest_text
    )
    return latest_turn_messages(
        drop_routing_messages(messages),
        include_previous_turn=include_previous_turn,
    )


# ── Post-supervisor routing node ──────────────────────────────────────────────


def _latest_human_message_text(messages: list) -> str:
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            return content_to_text(msg.content)
        if isinstance(msg, dict) and msg.get("role") in {"user", "human"}:
            return content_to_text(msg.get("content"))
    return ""


def _last_ai_message_text(messages: list) -> str:
    for msg in reversed(messages):
        if isinstance(msg, AIMessage):
            return content_to_text(msg.content)
        if isinstance(msg, dict) and msg.get("role") in {"assistant", "ai"}:
            return content_to_text(msg.get("content"))
    return ""


def _strip_library_agent_prefix(text: str) -> str:
    return _LIBRARY_AGENT_PREFIX_RE.sub("", text, count=1).strip()


def _is_library_reservation_clarification(text: str) -> bool:
    return ("?" in text or "까요" in text or "세요" in text) and any(
        cue in text for cue in _LIBRARY_CLARIFICATION_CUES
    )


def _is_library_awaiting_user_input(raw_text: str) -> bool:
    if not raw_text:
        return False
    text = _strip_library_agent_prefix(raw_text)
    if any(marker in text for marker in _LIBRARY_COMPLETED_OUTCOME_MARKERS):
        return False
    if _LIBRARY_LOGIN_GATE_RE.search(text):
        return True
    from_library = raw_text.strip().startswith("[도서관 에이전트]")
    return from_library and _is_library_reservation_clarification(text)


def _deterministic_route(state: SsuAgentState) -> str | None:
    """Conservatively pre-route obvious library or LMS turns before the supervisor."""
    messages = state.get("messages", [])
    user_text = _latest_human_message_text(messages).strip()
    if not user_text:
        return None

    if _LMS_DIRECT_ROUTE_RE.search(user_text) and not any(
        keyword in user_text for keyword in _LMS_DIRECT_ROUTE_CONFLICTS
    ):
        return "lms_agent"

    if any(keyword in user_text for keyword in _STRONG_OTHER_DOMAIN_KEYWORDS):
        return None

    if any(keyword in user_text for keyword in _LIBRARY_ROUTE_KEYWORDS):
        return "library_agent"

    ai_text = _last_ai_message_text(messages)
    if len(user_text) <= _LIBRARY_CONTINUATION_MAX_CHARS and _is_library_awaiting_user_input(
        ai_text
    ):
        return "library_agent"

    return None


def _post_supervisor(state: SsuAgentState) -> Command:
    """Check the current user turn for a supervisor routing marker."""
    for msg in reversed(state["messages"]):
        if isinstance(msg, HumanMessage) or (
            isinstance(msg, dict) and msg.get("role") in {"user", "human"}
        ):
            break
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
    from ssu_agent.mcp_client import create_mcp_client, wrap_mcp_tools_for_retry

    if all_tools is None:
        client = create_mcp_client()
        all_tools = wrap_mcp_tools_for_retry(await client.get_tools())

    if checkpointer is None:
        from langgraph.checkpoint.memory import MemorySaver

        checkpointer = MemorySaver()

    cats = categorise_tools(all_tools)
    routing_tools = _make_routing_tools()

    # Browser-owned auth lifecycle tools and session-bearing tools never cross
    # the supervisor model boundary.
    supervisor_tools = tools_for_model([*cats["public"], *routing_tools], None)
    llm_seq = [llm] if llm is not None else get_llm_sequence()
    # Build one ReAct agent per provider ONCE. The tools and prompt are static for
    # the graph's lifetime, so constructing create_agent inside the node
    # re-compiled the identical graph on every request (and again per provider in
    # the fallback loop). Pre-building keeps the same fallback order.
    supervisor_reacts = [
        create_agent(_llm, supervisor_tools, system_prompt=_SUPERVISOR_PROMPT) for _llm in llm_seq
    ]

    async def supervisor_node(state: SsuAgentState, config: RunnableConfig) -> dict:
        last_exc: Exception | None = None
        config = config or {}
        supervisor_config = {
            **config,
            "tags": [*(config.get("tags") or []), "supervisor_llm"],
        }
        input_messages = sanitize_tool_pairing(
            sanitize_messages_for_model(
                _supervisor_model_messages(state["messages"]),
                state.get("mcp_session_id"),
            )
        )
        input_message_count = len(input_messages)
        for idx, react in enumerate(supervisor_reacts):
            try:
                result = await react.ainvoke(
                    {"messages": input_messages},
                    config=supervisor_config,
                )
                new_messages = sanitize_messages_for_model(
                    result["messages"][input_message_count:],
                    state.get("mcp_session_id"),
                )
                for msg in new_messages:
                    if isinstance(msg, AIMessage):
                        msg.name = "supervisor"
                        if contains_internal_auth_guidance(content_to_text(msg.content)):
                            msg.content = "" if msg.tool_calls else _SUPERVISOR_AUTH_FALLBACK
                return {"messages": new_messages}
            except Exception as exc:
                # Same rationale as react_loop: surface WHY each provider failed
                # instead of only re-raising the last one.
                logger.warning(
                    "[supervisor] provider #%d failed: type=%s",
                    idx,
                    type(exc).__name__,
                )
                last_exc = exc
        raise last_exc or RuntimeError("All LLM providers exhausted")

    # Sub-agent subgraphs — embedded as nodes so interrupt() propagates correctly
    library_subgraph = build_library_agent([*cats["library"], *cats["auth"]], llm).compile()
    academic_subgraph = build_academic_agent([*cats["academic"], *cats["auth"]], llm).compile()
    lms_subgraph = build_lms_agent([*cats["lms"], *cats["auth"]], llm).compile()

    # Parent graph assembly
    builder = StateGraph(SsuAgentState)

    builder.add_node("supervisor", supervisor_node)
    builder.add_node("post_supervisor", _post_supervisor)
    builder.add_node("library_agent", library_subgraph)
    builder.add_node("academic_agent", academic_subgraph)
    builder.add_node("lms_agent", lms_subgraph)

    def route_from_start(state: SsuAgentState) -> str:
        return _deterministic_route(state) or "supervisor"

    builder.add_conditional_edges(
        START,
        route_from_start,
        {
            "library_agent": "library_agent",
            "lms_agent": "lms_agent",
            "supervisor": "supervisor",
        },
    )
    builder.add_edge("supervisor", "post_supervisor")

    # post_supervisor returns Command(goto=target|END) — LangGraph handles routing
    builder.add_edge("library_agent", END)
    builder.add_edge("academic_agent", END)
    builder.add_edge("lms_agent", END)

    return builder.compile(checkpointer=checkpointer)
