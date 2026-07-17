"""
LMS sub-agent — assignments and LMS terms.

Uses direct bind_tools loop (not create_react_agent) to avoid turn-2 looping
and enable per-provider fallback.
Supports optional term_id for semester selection (LMS term bug fix, PR #61).
"""

from __future__ import annotations

import json
import re
from typing import Any
from urllib.parse import parse_qs, urlsplit

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool
from langgraph.graph import StateGraph

from ssu_agent import config as agent_config
from ssu_agent.agents.auth_guard import ProviderLinkState, check_provider_link, tools_for_model
from ssu_agent.agents.react_loop import run_react_loop
from ssu_agent.llm_factory import create_llm, get_llm_sequence
from ssu_agent.supervisor.state import SsuAgentState

_SYSTEM_PROMPT_BASE = """당신은 숭실대학교 LMS(Canvas) 전문 AI 어시스턴트입니다.

담당 영역:
- 과제 목록 조회 (get_my_assignments): compact=true 옵션으로 요약 제공
- LMS 학기 목록 (get_my_lms_terms): 학기 선택 시 먼저 이 도구로 학기 ID를 확인하세요.
- get_lms_dashboard: 미제출 과제·퀴즈 마감, 학사일정(시험·수강신청), 진행 중 공지를 한 번에 조회.
  선택 파라미터: term_id (생략 시 현재 학기 자동).
- 강의자료 내보내기 플로우:
  - 사용자가 현재 학기 전체 과목·모든 자료를 명시하면 export_all_lms_materials로 한 번에
    미리보기를 만든 뒤 confirm_lms_material_export를 호출하세요.
  - 특정 과목 요청은 get_my_lms_courses로 과목과 content_id를 조회한 뒤
    prepare_lms_material_export → confirm_lms_material_export 순서로 처리하세요.
  - confirm_lms_material_export는 ZIP 작업을 시작하고 브라우저 다운로드 링크를 반환합니다.
영상·오디오 파일은 절대 포함되지 않음.

대화 규칙: 가장 최근 사용자 메시지의 LMS 요청만 답하세요. 이전 턴에서 이미 답한 다른 영역의
요청을 다시 정리하거나 담당 범위가 아니라고 언급하지 마세요.

학기 관련 주의: Canvas API는 6월에 여름학기를 기본(default)으로 반환하므로,
1학기 과제를 조회할 때는 get_my_lms_terms로 학기 목록을 먼저 조회하고
올바른 term_id를 사용하세요.

효율 규칙(중요): 질문에 맞는 도구를 최소 횟수로 호출하세요. 필요한 도구가
여러 개면 한 번에 함께 호출하세요(병렬 실행됨). 같은 도구를 반복 호출하거나
결과가 나온 뒤 불필요하게 재확인하지 마세요. 단, LMS 자료 내보내기에서는
export_all_lms_materials 또는 prepare_lms_material_export의 결과를 먼저 받은 뒤
confirm_lms_material_export를 다음 턴에 호출하세요. prepare/export와 confirm을 같은
도구 호출 묶음에 넣으면 안 됩니다."""

_LMS_LOGIN_MESSAGE = (
    "LMS 개인 데이터는 u-SAINT·LMS 연결 후 확인할 수 있어요. 화면 상단의 ‘연결’을 열어 "
    "u-SAINT와 LMS를 연결한 뒤 다시 요청해 주세요. 학번·비밀번호 같은 로그인 정보는 "
    "채팅에 입력하지 않아도 돼요."
)
_LMS_STATUS_UNAVAILABLE_MESSAGE = (
    "LMS 연결 상태를 지금 확인하지 못했어요. 잠시 후 다시 보내거나 화면 상단의 ‘연결’에서 "
    "LMS 상태를 확인해 주세요. 로그인 정보는 채팅에 입력하지 않아도 돼요."
)
_LMS_EXPORT_PATH_RE = re.compile(r"^/api/lms/exports/[^/]+/download$")


def _format_bytes(byte_count: int) -> str:
    units = ((1024**3, "GB"), (1024**2, "MB"), (1024, "KB"))
    for unit_bytes, unit_label in units:
        if byte_count >= unit_bytes:
            value = byte_count / unit_bytes
            number = f"{value:.0f}" if value >= 10 or value.is_integer() else f"{value:.1f}"
            return f"{number} {unit_label}"
    return f"{byte_count} B"


def _export_confirmation_data(content: str) -> dict[str, Any] | None:
    try:
        payload = json.loads(content)
    except (json.JSONDecodeError, TypeError):
        return None
    if not isinstance(payload, dict):
        return None

    if payload.get("status") != "OK":
        return None
    data = payload.get("data")
    return data if isinstance(data, dict) else None


def _format_lms_export_confirmation(tool_name: str, content: str) -> str | None:
    """Turn a successful confirm result into the final answer without another LLM call."""
    if tool_name != "confirm_lms_material_export":
        return None

    data = _export_confirmation_data(content)
    if data is None:
        return None
    download_url = data.get("downloadUrl")
    if not isinstance(download_url, str):
        return None
    if any(character in download_url for character in "\r\n []()<>\"'"):
        return None

    parsed = urlsplit(download_url)
    trusted_mcp_url = urlsplit(agent_config.SSUMCP_URL)
    token_values = parse_qs(parsed.query, keep_blank_values=True).get("token")
    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
        or parsed.scheme.lower() != trusted_mcp_url.scheme.lower()
        or parsed.netloc.lower() != trusted_mcp_url.netloc.lower()
        or not _LMS_EXPORT_PATH_RE.fullmatch(parsed.path)
        or not token_values
        or not token_values[0]
    ):
        return None

    details: list[str] = []
    file_count = data.get("fileCount")
    if isinstance(file_count, int) and not isinstance(file_count, bool) and file_count > 0:
        details.append(f"파일 {file_count}개")
    estimated_bytes = data.get("estimatedBytes")
    if (
        isinstance(estimated_bytes, int)
        and not isinstance(estimated_bytes, bool)
        and estimated_bytes > 0
    ):
        details.append(f"약 {_format_bytes(estimated_bytes)}")

    summary = f" ({' · '.join(details)})" if details else ""
    return (
        f"강의자료 다운로드를 준비했어요{summary}.\n\n"
        f"[강의 파일 다운로드]({download_url})\n\n"
        "링크를 지금 열어 두면 압축 상태를 확인하고, 준비되는 즉시 다운로드할 수 있어요."
    )


def _build_lms_prompt(authenticated: bool) -> str:
    prompt = _SYSTEM_PROMPT_BASE
    if authenticated:
        prompt += (
            "\n\n[인증 연결 있음] 사용자는 이미 로그인되어 있고 인증 값은 시스템이 도구에 "
            "자동으로 주입합니다. "
            "절대 로그인을 다시 요청하지 마세요. 과제·강의자료·대시보드 질문에는 해당 도구"
            "(get_my_assignments, get_my_lms_terms, get_lms_dashboard, get_my_lms_courses, "
            "get_my_lms_materials, export_all_lms_materials, prepare_lms_material_export, "
            "confirm_lms_material_export)를 "
            "지금 즉시 호출하고, 그 결과로 답하세요. 내부 인증 값이나 로그인 링크를 사용자에게 "
            "보여주거나 직접 알려 달라고 요청하지 마세요."
        )
    else:
        # No auth session: every LMS tool needs the session. Answer with a login
        # nudge instead of burning turns on calls that can only fail.
        prompt += (
            "\n\n[인증 세션 없음] LMS 도구는 모두 인증이 필요합니다. 도구를 "
            "호출하지 말고 'LMS 연결(로그인) 후 확인할 수 있다'고 안내만 하세요."
        )
    return prompt


def build_lms_agent(
    lms_tools: list[BaseTool],
    llm: BaseChatModel | None = None,
) -> StateGraph:
    """Build the LMS sub-agent graph."""
    llm_seq = [llm] if llm is not None else get_llm_sequence()
    if not llm_seq:
        llm_seq = [create_llm()]

    async def agent_node(state: SsuAgentState, config: RunnableConfig) -> dict:
        mcp_session_id = state.get("mcp_session_id")
        if not mcp_session_id:
            return {
                "messages": [AIMessage(content=f"[LMS 에이전트] {_LMS_LOGIN_MESSAGE}")],
                "active_agent": None,
            }

        provider_state = await check_provider_link(
            lms_tools,
            mcp_session_id,
            "LMS",
            config,
        )
        if provider_state is ProviderLinkState.DISCONNECTED:
            return {
                "messages": [AIMessage(content=f"[LMS 에이전트] {_LMS_LOGIN_MESSAGE}")],
                "active_agent": None,
            }
        if provider_state in {
            ProviderLinkState.UNAVAILABLE,
            ProviderLinkState.UNSUPPORTED,
        }:
            return {
                "messages": [
                    AIMessage(content=f"[LMS 에이전트] {_LMS_STATUS_UNAVAILABLE_MESSAGE}")
                ],
                "active_agent": None,
            }

        prompt = _build_lms_prompt(True)
        return await run_react_loop(
            llm_seq,
            tools_for_model(lms_tools, mcp_session_id),
            prompt,
            "LMS 에이전트",
            state,
            config,
            auth_required_message=_LMS_LOGIN_MESSAGE,
            terminal_tool_result_formatter=_format_lms_export_confirmation,
            standalone_tool_names={"confirm_lms_material_export"},
        )

    graph = StateGraph(SsuAgentState)
    graph.add_node("agent", agent_node)
    graph.set_entry_point("agent")
    graph.set_finish_point("agent")
    return graph
