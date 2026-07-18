"""
Academic sub-agent — u-SAINT + official policy RAG.

Uses direct bind_tools loop (not create_react_agent) to avoid turn-2 looping
and enable per-provider fallback. No HITL needed (no write actions).

Covers: grades, GPA simulation, graduation requirements, academic calendar,
chapel attendance, scholarships, and ssuMCP's embedded academic policy RAG
(classify_academic_question, search/get/evaluate policy sources).
"""

from __future__ import annotations

import re

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool
from langgraph.graph import StateGraph

from ssu_agent.agents.auth_guard import ProviderLinkState, check_provider_link, tools_for_model
from ssu_agent.agents.react_loop import (
    UPSTREAM_TOOL_UNAVAILABLE_MESSAGE,
    drop_routing_messages,
    latest_turn_messages,
    run_react_loop,
)
from ssu_agent.llm_factory import create_llm, get_llm_sequence
from ssu_agent.supervisor.state import SsuAgentState
from ssu_agent.tool_results import content_to_text

_SYSTEM_PROMPT_BASE = """당신은 숭실대학교 학사 정보 전문 AI 어시스턴트입니다.

담당 영역:
- 성적 조회 및 GPA 시뮬레이션
- 졸업 요건 확인 및 정책 근거 RAG 검색
- 학사 일정 조회
- 채플 이수 현황
- 장학금 조회 및 장학 정책 검색
- 공식 학칙·졸업·장학 출처 기반 근거 제시

정책 답변 시 반드시 공식 출처(search_academic_policy_sources 또는
get_academic_policy_brief)를 조회하여 근거를 포함하세요.
추측이나 일반 지식으로 답하지 마세요.

대화 규칙: 가장 최근 사용자 메시지의 학사 요청만 답하세요. 이전 턴에서 이미 답한 식단·도서관
등의 요청을 다시 정리하거나 담당 범위가 아니라고 언급하지 마세요.

효율 규칙(중요): 질문에 맞는 도구를 최소 횟수로 호출하세요. 졸업 요건처럼
복합 판단이 필요하면 evaluate_graduation_with_policy 같은 복합 도구를 우선 1회
호출하고, 필요한 도구가 여러 개면 한 번에 함께 호출하세요(병렬 실행됨). 같은
도구를 반복 호출하거나 결과가 나온 뒤 불필요하게 재확인하지 마세요."""

_ACADEMIC_LOGIN_MESSAGE = (
    "개인별로 남은 졸업 요건과 성적·학점 현황은 u-SAINT 연결 후 확인할 수 있어요. "
    "화면 상단의 ‘연결’을 열어 u-SAINT를 연결한 뒤 같은 질문을 다시 보내 주세요. "
    "학번·비밀번호 같은 로그인 정보는 채팅에 입력하지 않아도 돼요.\n\n"
    '로그인 없이 일반 기준만 보고 싶다면 "일반 졸업 기준 알려줘"라고 말해 주세요.'
)
_ACADEMIC_STATUS_UNAVAILABLE_MESSAGE = (
    "u-SAINT 연결 상태를 지금 확인하지 못했어요. 잠시 후 다시 보내거나 화면 상단의 ‘연결’에서 "
    "u-SAINT 상태를 확인해 주세요. 로그인 정보는 채팅에 입력하지 않아도 돼요."
)
_ACADEMIC_SERVICE_UNAVAILABLE_MESSAGE = (
    "u-SAINT 연결은 확인됐지만 학사 정보 서비스에서 요청한 정보를 가져오지 못했어요. "
    "잠시 후 다시 보내 주세요. "
    "개인 학사 조회가 계속 실패하면 화면 상단의 ‘연결’에서 u-SAINT를 다시 연결해 주세요. "
    "로그인 정보는 채팅에 입력하지 않아도 돼요."
)

_ACADEMIC_DATA_RE = re.compile(
    r"졸업|성적|학점|gpa|평점|채플|장학|시간표|수강|학적|전공|교양",
    re.IGNORECASE,
)
_PERSONAL_SUBJECT_RE = re.compile(
    r"(?:^|[\s,?!])(?:내|나의|내가|나는|저의|제가|저는|제)"
    r"(?=$|[\s,?!]|졸업|성적|학점|gpa|평점|채플|장학|시간표|수강|학적|전공|교양)",
    re.IGNORECASE,
)
_PERSONAL_LOOKUP_RE = re.compile(
    r"(?:성적|평점|gpa).{0,12}(?:보여|조회|확인|알려|계산|나왔)"
    r"|(?:이번학기|현재|금학기).{0,12}(?:시간표|성적|학점|수강)"
    r"|(?:시간표|수강내역|수강목록).{0,12}(?:보여|조회|확인|알려)"
    r"|채플.{0,12}(?:몇|남았|남은|출석|이수|현황)"
    r"|채플.{0,12}(?:얼마나|더).{0,12}(?:들어|이수|남)"
    r"|장학(?:금)?.{0,12}(?:받았|받은|내역|조회|현황)"
    r"|졸업.{0,24}(?:남았|남은|진단|판정|요건확인|할수있|가능해|가능한가)"
    r"|(?:학점|전공|교양).{0,12}(?:남았|남은|부족|현황)"
    r"|(?:지난|저번|이전)학기.{0,12}(?:성적|평점|gpa|학점)"
    r"|(?:졸업사정표|취득학점).{0,12}(?:보여|조회|확인|알려)"
    r"|(?:누적학점|누적평점|현재성적|현재학점|이수현황|채플출석|받은장학)",
    re.IGNORECASE,
)
_PUBLIC_POLICY_RE = re.compile(
    r"계산법|산정(?:법|방식)|(?:일반적인?|공통)\s*(?:기준|요건)|기준|규정|정책|방법|뜻",
    re.IGNORECASE,
)
_PUBLIC_ACADEMIC_REQUEST_RE = re.compile(r"학사\s*일정|학사일정|캘린더|달력", re.IGNORECASE)


def _last_human_message_text(messages: list) -> str:
    for message in reversed(messages):
        if isinstance(message, HumanMessage):
            return content_to_text(message.content)
        if isinstance(message, dict) and message.get("role") in {"user", "human"}:
            return content_to_text(message.get("content"))
    return ""


def _requires_private_academic_data(text: str) -> bool:
    """Distinguish personal status lookups from public policy questions."""
    if not _ACADEMIC_DATA_RE.search(text):
        return False
    compact = re.sub(r"\s+", "", text).lower()
    has_personal_subject = bool(_PERSONAL_SUBJECT_RE.search(text))
    if not has_personal_subject and _PUBLIC_POLICY_RE.search(text):
        return False
    return bool(has_personal_subject or _PERSONAL_LOOKUP_RE.search(compact))


def _is_ambiguous_academic_follow_up(text: str) -> bool:
    """Return whether a short utterance needs the immediately prior academic turn."""
    compact = re.sub(r"[\s?!.,]+", "", text).lower()
    if not compact or len(compact) > 24:
        return False
    if _PUBLIC_POLICY_RE.search(text) or _PUBLIC_ACADEMIC_REQUEST_RE.search(text):
        return False
    if compact in {"네", "응", "맞아", "이제알려줘", "다시알려줘"} or compact.startswith(
        ("그", "지난학기", "저번학기")
    ):
        return True
    return not _ACADEMIC_DATA_RE.search(text) and compact.endswith(
        ("알려줘", "보여줘", "확인해줘", "계산해줘")
    )


def _requires_private_academic_context(messages: list) -> bool:
    """Classify the current request, inheriting only an ambiguous academic follow-up."""
    current_text = _last_human_message_text(messages)
    if _requires_private_academic_data(current_text):
        return True
    if not _is_ambiguous_academic_follow_up(current_text):
        return False

    contextual = latest_turn_messages(
        drop_routing_messages(messages),
        agent_tag="학사 에이전트",
    )
    human_texts = [
        content_to_text(message.content)
        for message in contextual
        if isinstance(message, HumanMessage)
    ]
    return len(human_texts) >= 2 and _requires_private_academic_data(human_texts[-2])


def _build_academic_prompt(authenticated: bool) -> str:
    prompt = _SYSTEM_PROMPT_BASE
    if authenticated:
        prompt += (
            "\n\n[인증 연결 있음] 인증 값은 시스템이 도구에 자동으로 주입합니다. "
            "절대 로그인을 다시 요청하거나 '로그인이 필요하다/필요할 수 있다'고 답하지 마세요. "
            "성적·졸업요건·GPA·시간표·채플·장학 질문에는 해당 도구(get_my_grades, "
            "check_graduation_requirements, simulate_gpa, get_my_schedule, get_my_chapel_info, "
            "get_my_scholarships, evaluate_graduation_with_policy 등)를 지금 즉시 호출하고, "
            "그 결과의 실제 값으로 답하세요. 내부 인증 값이나 로그인 링크를 사용자에게 "
            "보여주거나 직접 알려 달라고 요청하지 마세요."
        )
    else:
        # No auth session: personal-data tools would only fail (and start_auth
        # here just burns turns/latency). Answer directly with a login nudge.
        prompt += (
            "\n\n[인증 세션 없음] 개인 데이터가 필요한 도구(get_my_grades, "
            "check_graduation_requirements, simulate_gpa, get_my_schedule, "
            "get_my_chapel_info, get_my_scholarships)나 start_auth를 호출하지 마세요. "
            "공식 출처 검색 같은 공개 도구만 쓰고, 개인 정보는 'u-SAINT 연결(로그인) 후 "
            "확인할 수 있다'고 안내만 하세요."
        )
    return prompt


def build_academic_agent(
    academic_tools: list[BaseTool],
    llm: BaseChatModel | None = None,
) -> StateGraph:
    """Build the Academic sub-agent graph."""
    llm_seq = [llm] if llm is not None else get_llm_sequence()
    if not llm_seq:
        llm_seq = [create_llm()]

    async def agent_node(state: SsuAgentState, config: RunnableConfig) -> dict:
        mcp_session_id = state.get("mcp_session_id")
        requires_private_data = _requires_private_academic_context(state.get("messages", []))
        confirmed_session_id: str | None = None
        provider_state: ProviderLinkState | None = None
        if not mcp_session_id and requires_private_data:
            return {
                "messages": [AIMessage(content=f"[학사 에이전트] {_ACADEMIC_LOGIN_MESSAGE}")],
                "active_agent": None,
            }

        if mcp_session_id and requires_private_data:
            provider_state = await check_provider_link(
                academic_tools,
                mcp_session_id,
                "SAINT",
                config,
            )
            if provider_state is ProviderLinkState.DISCONNECTED:
                return {
                    "messages": [AIMessage(content=f"[학사 에이전트] {_ACADEMIC_LOGIN_MESSAGE}")],
                    "active_agent": None,
                }
            if provider_state in {
                ProviderLinkState.UNAVAILABLE,
                ProviderLinkState.UNSUPPORTED,
            }:
                return {
                    "messages": [
                        AIMessage(content=f"[학사 에이전트] {_ACADEMIC_STATUS_UNAVAILABLE_MESSAGE}")
                    ],
                    "active_agent": None,
                }
            confirmed_session_id = mcp_session_id

        prompt = _build_academic_prompt(bool(confirmed_session_id))
        return await run_react_loop(
            llm_seq,
            tools_for_model(academic_tools, confirmed_session_id),
            prompt,
            "학사 에이전트",
            state,
            config,
            auth_required_message=_ACADEMIC_LOGIN_MESSAGE,
            upstream_failure_message=(
                _ACADEMIC_SERVICE_UNAVAILABLE_MESSAGE
                if confirmed_session_id
                else UPSTREAM_TOOL_UNAVAILABLE_MESSAGE
            ),
            private_tool_call_budget=(1 if provider_state is ProviderLinkState.DEGRADED else None),
        )

    graph = StateGraph(SsuAgentState)
    graph.add_node("agent", agent_node)
    graph.set_entry_point("agent")
    graph.set_finish_point("agent")
    return graph
