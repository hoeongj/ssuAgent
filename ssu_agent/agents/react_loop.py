"""
Shared manual bind_tools ReAct loop for the academic and LMS sub-agents.

The academic and LMS sub-agents run the identical loop — bind the tools, let
the model call them for up to N turns, then return one tagged answer — differing
only by their tool set, system prompt, and display tag. This module holds that
loop once so the two agents can't drift apart. The library agent does NOT use it:
its HITL gate needs the intermediate prepare_* ToolMessages preserved in state,
whereas this loop intentionally returns only the final tagged answer.

Why a manual loop instead of create_react_agent: it enables per-provider fallback
across the LLM sequence and avoids the turn-2 looping observed with the prebuilt
agent (see the library agent's module docstring for the A/B detail).
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from collections.abc import Callable

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool

from ssu_agent.agents.auth_guard import (
    auth_denial_status,
    contains_internal_auth_guidance,
    sanitize_messages_for_model,
    sanitize_tool_result_for_model,
)
from ssu_agent.supervisor.state import SsuAgentState
from ssu_agent.tool_results import content_to_text, sanitize_tool_pairing, tool_result_to_text

logger = logging.getLogger(__name__)

# Kept low on purpose: each turn is a sequential LLM round-trip, and the whole
# sub-agent answer must reach the browser inside the Vercel proxy's 60s cap
# (ssuAI app/api/agent/stream). 4 turns covers legitimate multi-tool answers
# while stopping exploratory re-call storms that used to push latency past 60s.
_MAX_TOOL_TURNS = 4
EMPTY_RESPONSE_FALLBACK = "요청을 처리하지 못했어요. 다시 한 번 구체적으로 말씀해 주세요."
_AGENT_NAMES_BY_TAG = {
    "학사 에이전트": "academic_agent",
    "도서관 에이전트": "library_agent",
    "LMS 에이전트": "lms_agent",
}
TerminalToolResultFormatter = Callable[[str, str], str | None]


def _provider_label(llm: BaseChatModel) -> str:
    """Human-readable model id for latency logging (Groq vs Gemini vs …)."""
    return getattr(llm, "model_name", None) or getattr(llm, "model", None) or type(llm).__name__


async def _run_tool_call(
    tc: dict,
    tools: list[BaseTool],
    config: RunnableConfig,
    mcp_session_id: str | None = None,
) -> ToolMessage:
    """Execute one tool call and return its ToolMessage. Never raises so the
    surrounding asyncio.gather resolves for every call in the turn."""
    call_id = tc.get("id", "")
    name = tc.get("name", "")
    matched = next((t for t in tools if t.name == name), None)
    if matched is None:
        return ToolMessage(content=f"Tool '{name}' not found.", tool_call_id=call_id)
    started = time.perf_counter()
    try:
        result = await matched.ainvoke(tc.get("args", {}), config=config)
        content = sanitize_tool_result_for_model(
            tool_result_to_text(result),
            mcp_session_id,
        )
    except Exception as tool_exc:
        logger.warning(
            "tool %s failed: type=%s",
            name,
            type(tool_exc).__name__,
        )
        content = "Tool error: upstream tool failed."
    logger.info("tool %s finished in %.2fs", name, time.perf_counter() - started)
    return ToolMessage(content=content, tool_call_id=call_id)


async def _run_tool_call_with_batch_policy(
    tc: dict,
    tools: list[BaseTool],
    config: RunnableConfig,
    mcp_session_id: str | None,
    *,
    defer_standalone: bool,
) -> ToolMessage:
    """Reject a standalone-only call when the model batches it with dependencies."""
    if defer_standalone:
        name = str(tc.get("name", ""))
        logger.warning("tool %s deferred because it must run in a standalone turn", name)
        return ToolMessage(
            content=json.dumps(
                {
                    "status": "INVALID_TOOL_SEQUENCE",
                    "message": "Call this tool alone after the preceding tool result is available.",
                }
            ),
            tool_call_id=str(tc.get("id", "")),
        )
    return await _run_tool_call(tc, tools, config, mcp_session_id)


def drop_routing_messages(messages: list) -> list:
    """Remove routing artifacts without erasing completed supervisor turns.

    When the supervisor routes to a sub-agent it leaves an AIMessage with a
    transfer_to_<agent> tool call + a ToolMessage("ROUTE_TO:<agent>") in the
    shared state. Groq llama-3.3-70b sees the trailing ToolMessage and produces a
    text completion instead of calling the sub-agent's tools. Narration from the
    same routed user turn must also be stripped because it can make the sub-agent
    think the request was already answered.

    Do not remove every message named ``supervisor``: those messages also contain
    completed direct answers such as meal results. Erasing only those answers
    leaves consecutive HumanMessages in history, so the next domain agent treats
    an already-answered question as a second pending request.
    """
    routing_call_ids: set[tuple[int, str]] = set()
    message_turns: list[int] = []
    routed_turns: set[int] = set()
    turn = -1

    for msg in messages:
        if isinstance(msg, HumanMessage):
            turn += 1
        message_turns.append(turn)
        routing_calls = (
            [tc for tc in msg.tool_calls if tc.get("name", "").startswith("transfer_to_")]
            if isinstance(msg, AIMessage)
            else []
        )
        if routing_calls:
            routed_turns.add(turn)
            for tc in routing_calls:
                if call_id := tc.get("id"):
                    routing_call_ids.add((turn, call_id))

    result = []
    for msg, message_turn in zip(messages, message_turns, strict=True):
        if isinstance(msg, AIMessage):
            routing_calls = [
                tc for tc in msg.tool_calls if tc.get("name", "").startswith("transfer_to_")
            ]
            if routing_calls:
                non_routing_calls = [tc for tc in msg.tool_calls if tc not in routing_calls]
                if non_routing_calls:
                    result.append(
                        msg.model_copy(update={"content": "", "tool_calls": non_routing_calls})
                    )
                continue
            if msg.name == "supervisor" and message_turn in routed_turns:
                if msg.tool_calls:
                    result.append(msg.model_copy(update={"content": ""}))
                continue
        if isinstance(msg, ToolMessage) and (message_turn, msg.tool_call_id) in routing_call_ids:
            continue
        result.append(msg)
    return result


def latest_turn_messages(
    messages: list,
    *,
    agent_tag: str | None = None,
    include_previous_turn: bool = False,
) -> list:
    """Keep the current request plus, at most, one relevant sub-agent turn.

    A specialist must not receive unrelated completed meal/library/academic
    turns just because they share a checkpoint. Consecutive follow-ups still
    need the immediately preceding answer from the same specialist. The
    supervisor can opt into one preceding completed turn for short ambiguous
    follow-ups such as "자료구조요", "지난학기요", or "그럼 내일은?".
    """
    latest_human_index = next(
        (
            index
            for index in range(len(messages) - 1, -1, -1)
            if isinstance(messages[index], HumanMessage)
        ),
        None,
    )
    if latest_human_index is None:
        return list(messages)

    current_turn = list(messages[latest_human_index:])
    if agent_tag is None and not include_previous_turn:
        return current_turn

    previous_human_index: int | None = None
    previous_answer: AIMessage | None = None
    for index in range(latest_human_index - 1, -1, -1):
        message = messages[index]
        if previous_answer is None and isinstance(message, AIMessage):
            if content_to_text(message.content).strip() and not message.tool_calls:
                previous_answer = message
        if isinstance(message, HumanMessage):
            previous_human_index = index
            break

    if previous_human_index is None or previous_answer is None:
        return current_turn

    previous_text = content_to_text(previous_answer.content)
    if agent_tag is not None:
        expected_name = _AGENT_NAMES_BY_TAG.get(agent_tag)
        has_matching_name = expected_name is not None and previous_answer.name == expected_name
        has_legacy_prefix = previous_text.lstrip().startswith(f"[{agent_tag}]")
        if not has_matching_name and not has_legacy_prefix:
            return current_turn

    return [*messages[previous_human_index:latest_human_index], *current_turn]


def _content_is_blank(content: object) -> bool:
    if isinstance(content, str):
        return not content.strip()
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict) and "text" in item:
                parts.append(str(item["text"]))
            else:
                parts.append(str(item))
        return not "".join(parts).strip()
    return not content


def apply_empty_response_fallback(messages: list) -> None:
    """Replace a blank final assistant answer without changing tool-call turns."""
    for msg in reversed(messages):
        if isinstance(msg, AIMessage):
            if msg.tool_calls:
                return
            if _content_is_blank(msg.content):
                msg.content = EMPTY_RESPONSE_FALLBACK
            return


async def run_react_loop(
    llm_seq: list[BaseChatModel],
    tools: list[BaseTool],
    system_prompt: str,
    tag: str,
    state: SsuAgentState,
    config: RunnableConfig,
    auth_required_message: str | None = None,
    terminal_tool_result_formatter: TerminalToolResultFormatter | None = None,
    standalone_tool_names: set[str] | None = None,
) -> dict:
    """Run the bind_tools ReAct loop with per-provider fallback.

    Tries each LLM in ``llm_seq`` in order; on any provider error it advances to
    the next. Returns a single ``[{tag} ...]``-tagged AIMessage and clears
    ``active_agent`` so control returns to the supervisor. A domain may provide
    ``terminal_tool_result_formatter`` when a successful tool result already
    contains the complete user answer; this avoids a redundant model round and
    checkpoints the result before the browser stream deadline. Tools named in
    ``standalone_tool_names`` are not executed when a model batches them with
    another call; the model receives a sequence error and may retry after the
    dependency result is available.
    """
    messages = sanitize_messages_for_model(
        latest_turn_messages(
            drop_routing_messages(state["messages"]),
            agent_tag=tag,
        ),
        state.get("mcp_session_id"),
    )
    input_messages = sanitize_tool_pairing([SystemMessage(content=system_prompt), *messages])

    last_exc: Exception | None = None
    for _llm in llm_seq:
        provider = _provider_label(_llm)
        try:
            llm_with_tools = _llm.bind_tools(tools)
            history = list(input_messages)

            for turn in range(_MAX_TOOL_TURNS):
                turn_started = time.perf_counter()
                response = await llm_with_tools.ainvoke(history, config=config)
                history.append(response)

                if not response.tool_calls:
                    if auth_required_message and contains_internal_auth_guidance(
                        content_to_text(response.content)
                    ):
                        return {
                            "messages": [AIMessage(content=f"[{tag}] {auth_required_message}")],
                            "active_agent": None,
                        }
                    logger.info(
                        "[%s] provider=%s turn=%d final (%.2fs)",
                        tag,
                        provider,
                        turn,
                        time.perf_counter() - turn_started,
                    )
                    break

                # Fan the turn's tool calls out concurrently. u-SAINT scrapes are
                # the dominant cost; running N of them in parallel collapses the
                # per-turn latency from sum-of-tools to slowest-tool. gather keeps
                # result order aligned with response.tool_calls, so each ToolMessage
                # still trails its AIMessage tool call in the expected order.
                logger.info(
                    "[%s] provider=%s turn=%d calling %d tool(s): %s",
                    tag,
                    provider,
                    turn,
                    len(response.tool_calls),
                    [tc.get("name") for tc in response.tool_calls],
                )
                standalone_batch = len(response.tool_calls) > 1
                tool_messages = await asyncio.gather(
                    *(
                        _run_tool_call_with_batch_policy(
                            tc,
                            tools,
                            config,
                            state.get("mcp_session_id"),
                            defer_standalone=(
                                standalone_batch
                                and tc.get("name") in (standalone_tool_names or set())
                            ),
                        )
                        for tc in response.tool_calls
                    )
                )
                if auth_required_message and any(
                    auth_denial_status(content_to_text(message.content)) is not None
                    for message in tool_messages
                ):
                    return {
                        "messages": [AIMessage(content=f"[{tag}] {auth_required_message}")],
                        "active_agent": None,
                    }

                if terminal_tool_result_formatter is not None:
                    for tool_call, tool_message in zip(
                        response.tool_calls,
                        tool_messages,
                        strict=True,
                    ):
                        try:
                            terminal_text = terminal_tool_result_formatter(
                                str(tool_call.get("name", "")),
                                content_to_text(tool_message.content),
                            )
                        except Exception as formatter_exc:
                            logger.warning(
                                "[%s] terminal tool formatter failed: type=%s",
                                tag,
                                type(formatter_exc).__name__,
                            )
                            continue
                        if terminal_text:
                            logger.info(
                                "[%s] provider=%s turn=%d terminal tool result",
                                tag,
                                provider,
                                turn,
                            )
                            return {
                                "messages": [
                                    AIMessage(
                                        content=f"[{tag}] {terminal_text}",
                                        id=None,
                                        name=_AGENT_NAMES_BY_TAG.get(tag),
                                    )
                                ],
                                "active_agent": None,
                            }

                history.extend(
                    sanitize_messages_for_model(
                        list(tool_messages),
                        state.get("mcp_session_id"),
                    )
                )

            apply_empty_response_fallback(history[len(input_messages) :])
            last_ai = next(
                (
                    m
                    for m in reversed(history[len(input_messages) :])
                    if isinstance(m, AIMessage) and not _content_is_blank(m.content)
                ),
                None,
            )
            text = content_to_text(last_ai.content) if last_ai else ""
            fallback_applied = (
                last_ai is not None
                and content_to_text(last_ai.content).strip() == EMPTY_RESPONSE_FALLBACK.strip()
            )
            tagged = AIMessage(
                content=f"[{tag}] {text}" if text.strip() else f"[{tag}] 처리 완료",
                # id reuse is a dedup optimization valid only when the tagged
                # text equals the streamed text. The empty-response fallback
                # deliberately diverges, so it needs a fresh id or SSE id-dedup
                # drops it (regression from ef0dff4).
                id=None if last_ai is None or fallback_applied else last_ai.id,
                name=_AGENT_NAMES_BY_TAG.get(tag),
            )
            return {"messages": [tagged], "active_agent": None}
        except Exception as exc:
            # Log every provider failure — the fallback used to swallow all but
            # the last exception, hiding WHY the earlier (preferred) providers
            # failed when diagnosing quota/schema errors in prod.
            logger.warning(
                "[%s] provider=%s failed: type=%s",
                tag,
                provider,
                type(exc).__name__,
            )
            last_exc = exc

    raise last_exc or RuntimeError("All LLM providers exhausted")
