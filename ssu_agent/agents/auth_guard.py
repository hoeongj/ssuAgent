"""Deterministic authentication boundaries for model-driven sub-agents.

The browser owns login UX and ssuMCP owns authorization. Models may choose which
domain tool to call, but they must never receive or reproduce the raw MCP session
handle. This module binds that handle outside the model-visible tool schema and
parses authentication outcomes as structured data.
"""

from __future__ import annotations

import asyncio
import copy
import json
import logging
import re
from enum import StrEnum
from typing import Any

from langchain_core.callbacks.manager import AsyncCallbackManagerForToolRun
from langchain_core.messages import AIMessage, BaseMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool
from pydantic import PrivateAttr, create_model

from ssu_agent.tool_results import content_to_text, tool_result_to_text

logger = logging.getLogger(__name__)

_SESSION_ARGUMENT_NAMES = {"mcp_session_id", "mcpSessionId"}
_AUTH_LIFECYCLE_TOOL_NAMES = {
    "get_auth_status",
    "start_auth",
    "logout_provider",
    "logout_all",
}
_AUTH_DENIAL_STATUSES = {
    "AUTH_REQUIRED",
    "NO_SESSION",
    "INVALID_SESSION",
    "SESSION_MISMATCH",
}
_INTERNAL_AUTH_GUIDANCE_RE = re.compile(
    r"mcp[_\s-]*session[_\s-]*(?:id|아이디)"
    r"|mcp_session_id"
    r"|세션\s*(?:id|ID|아이디)"
    r"|start_auth"
    r"|(?:https?://[^\s`\"'<>()]+)?/api/(?:mcp/)?auth/"
    r"(?:saint|lms|library)/start(?:\?[^\s`\"'<>()]*)?"
    r"|(?:로그인\s*)?버튼.{0,16}(?:누르|클릭)"
    r"|(?:누르|클릭).{0,16}(?:로그인\s*)?버튼",
    re.IGNORECASE,
)
_AUTH_START_URL_RE = re.compile(
    r"(?:https?://[^\s`\"'<>()]+)?/api/(?:mcp/)?auth/"
    r"(?:saint|lms|library)/start(?:\?[^\s`\"'<>()]*)?",
    re.IGNORECASE,
)
_SESSION_ASSIGNMENT_RE = re.compile(
    r"[\"']?(?:mcp_session_id|mcpSessionId)[\"']?\s*(?::|=)\s*"
    r"[\"']?[A-Za-z0-9._:-]{6,}[\"']?",
    re.IGNORECASE,
)
_SESSION_REFERENCE_RE = re.compile(
    r"(?:mcp[_\s-]*session(?:[_\s-]*(?:id|아이디))?"
    r"|session(?:[_\s-]*(?:id|아이디))?"
    r"|세션(?:\s*(?:id|ID|아이디))?)"
    r"\s*(?::|=|is\s+|was\s+|값은?\s*)?"
    r"[\"']?[A-Za-z0-9][A-Za-z0-9._:-]{5,}[\"']?",
    re.IGNORECASE,
)
_SESSION_DESCRIPTION_RE = re.compile(
    r"mcp_session_id|mcpSessionId|MCP\s+session\s+ID",
    re.IGNORECASE,
)
_LMS_EXPORT_CAPABILITY_URL_RE = re.compile(
    r"https?://[^\s`\"'<>()]+/api/lms/exports/[^/\s`\"'<>()]+/download\?"
    r"(?=[^\s`\"'<>()]*\btoken=)[^\s`\"'<>()]+",
    re.IGNORECASE,
)
_AUTH_STATUS_TIMEOUT_SECONDS = 5.0


class ProviderLinkState(StrEnum):
    CONNECTED = "connected"
    DISCONNECTED = "disconnected"
    UNAVAILABLE = "unavailable"
    UNSUPPORTED = "unsupported"


def _session_argument_names(tool: BaseTool) -> set[str]:
    schema = tool.args_schema
    if isinstance(schema, dict):
        properties = schema.get("properties", {})
        return _SESSION_ARGUMENT_NAMES.intersection(properties)
    fields = getattr(schema, "model_fields", None)
    if isinstance(fields, dict):
        return _SESSION_ARGUMENT_NAMES.intersection(fields)
    return set()


def _without_session_schema(tool: BaseTool) -> type | dict | None:
    schema = tool.args_schema
    if isinstance(schema, dict):
        reduced = copy.deepcopy(schema)
        properties = reduced.get("properties", {})
        for name in _SESSION_ARGUMENT_NAMES:
            properties.pop(name, None)
        if isinstance(reduced.get("required"), list):
            reduced["required"] = [
                name for name in reduced["required"] if name not in _SESSION_ARGUMENT_NAMES
            ]
        return reduced

    fields = getattr(schema, "model_fields", None)
    if not isinstance(fields, dict):
        return schema
    visible_fields = {
        name: (field.annotation, copy.copy(field))
        for name, field in fields.items()
        if name not in _SESSION_ARGUMENT_NAMES
    }
    model_name = re.sub(r"\W+", "_", f"{tool.name}_session_bound_input")
    return create_model(model_name, **visible_fields)


class _SessionBoundTool(BaseTool):
    """A model-visible tool that injects its session argument at execution time."""

    _wrapped: BaseTool = PrivateAttr()
    _mcp_session_id: str = PrivateAttr()

    def __init__(self, wrapped: BaseTool, mcp_session_id: str) -> None:
        description = _SESSION_DESCRIPTION_RE.sub("internal authentication", wrapped.description)
        description += (
            "\n인증은 시스템이 자동으로 처리합니다. 사용자에게 세션 값이나 로그인 링크를 "
            "요청하지 마세요."
        )
        super().__init__(
            name=wrapped.name,
            description=description,
            args_schema=_without_session_schema(wrapped),
            return_direct=wrapped.return_direct,
            verbose=wrapped.verbose,
            callbacks=wrapped.callbacks,
            tags=wrapped.tags,
            metadata=wrapped.metadata,
            handle_tool_error=wrapped.handle_tool_error,
            handle_validation_error=wrapped.handle_validation_error,
            response_format=wrapped.response_format,
        )
        self._wrapped = wrapped
        self._mcp_session_id = mcp_session_id

    def _run(self, *args: Any, **kwargs: Any) -> Any:
        msg = "Session-bound MCP tools support async invocation only"
        raise NotImplementedError(msg)

    async def _arun(
        self,
        *args: Any,
        config: RunnableConfig | None = None,
        run_manager: AsyncCallbackManagerForToolRun | None = None,
        **kwargs: Any,
    ) -> Any:
        for name in _SESSION_ARGUMENT_NAMES:
            kwargs.pop(name, None)
        kwargs["mcp_session_id"] = self._mcp_session_id
        return await self._wrapped._arun(  # noqa: SLF001
            *args,
            config=config,
            run_manager=run_manager,
            **kwargs,
        )


def tools_for_model(tools: list[BaseTool], mcp_session_id: str | None) -> list[BaseTool]:
    """Remove auth lifecycle tools and bind private-tool sessions outside the LLM."""
    visible: list[BaseTool] = []
    for tool in tools:
        if tool.name in _AUTH_LIFECYCLE_TOOL_NAMES:
            continue
        if _session_argument_names(tool):
            if mcp_session_id:
                visible.append(_SessionBoundTool(tool, mcp_session_id))
            continue
        visible.append(tool)
    return visible


def _json_object(text: str) -> dict[str, Any] | None:
    try:
        value = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return None
    return value if isinstance(value, dict) else None


def redact_internal_auth_artifacts(text: str, mcp_session_id: str | None = None) -> str:
    """Redact internal session handles and server-owned auth URLs from text."""
    redacted = text
    if mcp_session_id and len(mcp_session_id) >= 8:
        redacted = redacted.replace(mcp_session_id, "[internal authentication redacted]")
    redacted = _SESSION_ASSIGNMENT_RE.sub("[internal authentication redacted]", redacted)
    # Checkpoint history may contain a session that predates the current request.
    # Redact values by their authentication context instead of knowing only the
    # latest handle; unrelated UUID-shaped domain identifiers remain intact.
    redacted = _SESSION_REFERENCE_RE.sub("[internal authentication redacted]", redacted)
    redacted = _AUTH_START_URL_RE.sub("[authentication link redacted]", redacted)
    return redacted


def _redact_model_capability_urls(text: str) -> str:
    """Hide bearer-style download URLs only when copying history to a model."""
    return _LMS_EXPORT_CAPABILITY_URL_RE.sub("[download capability redacted]", text)


def _sanitize_nested_auth(value: Any, mcp_session_id: str | None) -> Any:
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.startswith(("{", "[")):
            try:
                parsed = json.loads(value)
            except json.JSONDecodeError:
                pass
            else:
                return json.dumps(
                    _sanitize_nested_auth(parsed, mcp_session_id),
                    ensure_ascii=False,
                )
        return _redact_model_capability_urls(redact_internal_auth_artifacts(value, mcp_session_id))
    if isinstance(value, list):
        return [_sanitize_nested_auth(item, mcp_session_id) for item in value]
    if not isinstance(value, dict):
        return value

    return {
        key: _sanitize_nested_auth(item, mcp_session_id)
        for key, item in value.items()
        if key not in _SESSION_ARGUMENT_NAMES
        and key not in {"loginUrl", "login_url", "developerMessage", "expiresAt"}
    }


def sanitize_messages_for_model(
    messages: list,
    mcp_session_id: str | None = None,
) -> list:
    """Return a model-safe history without cross-agent auth artifacts.

    Checkpointed library turns can contain the model's original tool arguments
    and raw MCP ToolMessages. Copy and redact them at every LLM boundary while
    leaving the durable checkpoint untouched for HITL routing.
    """
    sanitized: list = []
    auth_call_ids: set[str] = set()
    for message in messages:
        if isinstance(message, ToolMessage):
            if message.tool_call_id in auth_call_ids:
                continue
            content = _redact_model_capability_urls(
                sanitize_tool_result_for_model(
                    tool_result_to_text(message.content),
                    mcp_session_id,
                )
            )
            sanitized.append(message.model_copy(update={"content": content}))
            continue

        if isinstance(message, AIMessage):
            auth_calls = [
                tool_call
                for tool_call in message.tool_calls
                if tool_call.get("name") in _AUTH_LIFECYCLE_TOOL_NAMES
            ]
            auth_call_ids.update(
                tool_call_id for tool_call in auth_calls if (tool_call_id := tool_call.get("id"))
            )
            visible_tool_calls = [
                tool_call for tool_call in message.tool_calls if tool_call not in auth_calls
            ]
            visible_invalid_tool_calls = [
                tool_call
                for tool_call in message.invalid_tool_calls
                if tool_call.get("name") not in _AUTH_LIFECYCLE_TOOL_NAMES
            ]
            additional_kwargs = _sanitize_nested_auth(
                message.additional_kwargs,
                mcp_session_id,
            )
            if isinstance(additional_kwargs, dict):
                additional_kwargs.pop("tool_calls", None)
            updates: dict[str, Any] = {
                "content": _sanitize_nested_auth(
                    "" if auth_calls else message.content,
                    mcp_session_id,
                ),
                "tool_calls": [
                    _sanitize_nested_auth(tool_call, mcp_session_id)
                    for tool_call in visible_tool_calls
                ],
                "invalid_tool_calls": [
                    _sanitize_nested_auth(tool_call, mcp_session_id)
                    for tool_call in visible_invalid_tool_calls
                ],
                "additional_kwargs": additional_kwargs,
            }
            safe_message = message.model_copy(update=updates)
            if (
                not auth_calls
                or safe_message.tool_calls
                or safe_message.invalid_tool_calls
                or content_to_text(safe_message.content).strip()
            ):
                sanitized.append(safe_message)
            continue

        if isinstance(message, BaseMessage):
            sanitized.append(
                message.model_copy(
                    update={"content": _sanitize_nested_auth(message.content, mcp_session_id)}
                )
            )
            continue

        sanitized.append(_sanitize_nested_auth(copy.deepcopy(message), mcp_session_id))
    return sanitized


async def check_provider_link(
    tools: list[BaseTool],
    mcp_session_id: str,
    provider: str,
    config: RunnableConfig,
) -> ProviderLinkState:
    """Read provider linkage without exposing the status payload to the model."""
    status_tool = next((tool for tool in tools if tool.name == "get_auth_status"), None)
    if status_tool is None:
        return ProviderLinkState.UNSUPPORTED
    try:
        result = await asyncio.wait_for(
            status_tool.ainvoke(
                {"mcp_session_id": mcp_session_id},
                config=config,
            ),
            timeout=_AUTH_STATUS_TIMEOUT_SECONDS,
        )
        payload = _json_object(tool_result_to_text(result))
    except Exception as exc:
        logger.warning(
            "Provider authentication status check failed: provider=%s type=%s",
            provider,
            type(exc).__name__,
        )
        return ProviderLinkState.UNAVAILABLE

    if payload is None:
        return ProviderLinkState.UNAVAILABLE
    status = str(payload.get("status", "")).upper()
    if status in {"AUTH_REQUIRED", "NO_SESSION", "INVALID_SESSION", "SESSION_MISMATCH"}:
        return ProviderLinkState.DISCONNECTED
    if status != "OK":
        return ProviderLinkState.UNAVAILABLE
    providers = payload.get("providers")
    if not isinstance(providers, list):
        return ProviderLinkState.UNAVAILABLE
    for entry in providers:
        if not isinstance(entry, dict):
            continue
        if str(entry.get("provider", "")).upper() != provider.upper():
            continue
        if entry.get("linked") is not True:
            return ProviderLinkState.DISCONNECTED
        health = str(entry.get("health", "UNKNOWN")).upper()
        if health == "EXPIRED":
            return ProviderLinkState.DISCONNECTED
        if health == "ERROR":
            return ProviderLinkState.UNAVAILABLE
        return ProviderLinkState.CONNECTED
    return ProviderLinkState.DISCONNECTED


def auth_denial_status(content: str) -> str | None:
    payload = _json_object(content)
    if payload is None:
        return None
    status = str(payload.get("status", "")).upper()
    return status if status in _AUTH_DENIAL_STATUSES else None


def contains_internal_auth_guidance(content: object) -> bool:
    if isinstance(content, str):
        return bool(_INTERNAL_AUTH_GUIDANCE_RE.search(content))
    if isinstance(content, list):
        return any(
            contains_internal_auth_guidance(item.get("text", ""))
            for item in content
            if isinstance(item, dict)
        )
    return False


def sanitize_tool_result_for_model(
    content: str,
    mcp_session_id: str | None = None,
) -> str:
    """Remove session handles and auth procedures from structured MCP results."""
    payload = _json_object(content)
    if payload is None:
        return redact_internal_auth_artifacts(content, mcp_session_id)

    def sanitize(value: Any) -> Any:
        if isinstance(value, str):
            return redact_internal_auth_artifacts(value, mcp_session_id)
        if isinstance(value, list):
            return [sanitize(item) for item in value]
        if not isinstance(value, dict):
            return value

        status = str(value.get("status", "")).upper()
        private_envelope = any(name in value for name in _SESSION_ARGUMENT_NAMES) or any(
            name in value for name in {"loginUrl", "developerMessage"}
        )
        sanitized: dict[str, Any] = {}
        for key, item in value.items():
            if key in _SESSION_ARGUMENT_NAMES or key in {
                "loginUrl",
                "login_url",
                "developerMessage",
                "expiresAt",
            }:
                continue
            if key == "message" and (private_envelope or status in _AUTH_DENIAL_STATUSES):
                continue
            if key == "userMessage" and status in _AUTH_DENIAL_STATUSES:
                continue
            sanitized[key] = sanitize(item)
        return sanitized

    return redact_internal_auth_artifacts(
        json.dumps(sanitize(payload), ensure_ascii=False),
        mcp_session_id,
    )
