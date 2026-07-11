from __future__ import annotations

import asyncio
import logging
from collections.abc import Iterable
from typing import Any

import httpx
from langchain_core.callbacks.manager import AsyncCallbackManagerForToolRun
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool
from langchain_mcp_adapters.client import MultiServerMCPClient
from mcp.shared.exceptions import McpError
from mcp.types import CONNECTION_CLOSED
from pydantic import PrivateAttr

from ssu_agent import config

logger = logging.getLogger(__name__)


_DEFAULT_RETRY_BACKOFF_SECONDS = 0.1
_RETRYABLE_HTTP_STATUS_CODES = {502, 503, 504}
_RETRYABLE_MCP_ERROR_MESSAGES = (
    "connection closed",
    "session terminated",
    "session not found",
    "session expired",
    "invalid or expired session",
    "timed out while waiting for response",
)
_HTTPX_TRANSPORT_EXCEPTIONS = (
    httpx.ConnectError,
    httpx.ReadError,
    httpx.RemoteProtocolError,
    httpx.ConnectTimeout,
    httpx.ReadTimeout,
    httpx.WriteError,
    httpx.WriteTimeout,
    httpx.PoolTimeout,
)


def create_mcp_client(*, timeout_seconds: float | None = None) -> MultiServerMCPClient:
    connection: dict[str, Any] = {
        "url": config.SSUMCP_URL,
        "transport": "streamable_http",
    }
    if timeout_seconds is not None:
        connection["timeout"] = timeout_seconds
        connection["sse_read_timeout"] = timeout_seconds

    return MultiServerMCPClient(
        {
            "ssuMCP": connection,
        }
    )


def wrap_mcp_tools_for_retry(tools: Iterable[BaseTool]) -> list[BaseTool]:
    return [wrap_mcp_tool_for_retry(tool) for tool in tools]


def wrap_mcp_tool_for_retry(
    tool: BaseTool,
    *,
    retry_backoff_seconds: float = _DEFAULT_RETRY_BACKOFF_SECONDS,
) -> BaseTool:
    if isinstance(tool, _RetryingMCPTool):
        return tool
    return _RetryingMCPTool(tool, retry_backoff_seconds=retry_backoff_seconds)


def _is_retryable_mcp_transport_error(exc: BaseException) -> bool:
    for current in _walk_exception_tree(exc):
        if isinstance(current, _HTTPX_TRANSPORT_EXCEPTIONS):
            return True

        if isinstance(current, httpx.HTTPStatusError):
            response = current.response
            if response is not None and response.status_code in _RETRYABLE_HTTP_STATUS_CODES:
                return True

        if isinstance(current, McpError):
            message = str(current).lower()
            code = current.error.code
            if code in {CONNECTION_CLOSED, httpx.codes.REQUEST_TIMEOUT}:
                return True
            if any(fragment in message for fragment in _RETRYABLE_MCP_ERROR_MESSAGES):
                return True

    return False


def _walk_exception_tree(exc: BaseException) -> Iterable[BaseException]:
    seen: set[int] = set()
    stack: list[BaseException] = [exc]
    while stack:
        current = stack.pop()
        if id(current) in seen:
            continue
        seen.add(id(current))
        yield current

        if isinstance(current, BaseExceptionGroup):
            stack.extend(current.exceptions)
        if current.__cause__ is not None:
            stack.append(current.__cause__)
        if current.__context__ is not None:
            stack.append(current.__context__)


class _RetryingMCPTool(BaseTool):
    _wrapped: BaseTool = PrivateAttr()
    _retry_backoff_seconds: float = PrivateAttr()

    def __init__(self, wrapped: BaseTool, *, retry_backoff_seconds: float) -> None:
        super().__init__(
            name=wrapped.name,
            description=wrapped.description,
            args_schema=wrapped.args_schema,
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
        self._retry_backoff_seconds = retry_backoff_seconds

    def _run(self, *args: Any, **kwargs: Any) -> Any:
        msg = "MCP retry wrapper supports async tool invocation only"
        raise NotImplementedError(msg)

    async def _arun(
        self,
        *args: Any,
        config: RunnableConfig | None = None,
        run_manager: AsyncCallbackManagerForToolRun | None = None,
        **kwargs: Any,
    ) -> Any:
        try:
            return await self._invoke_wrapped(args, kwargs, config, run_manager)
        except Exception as exc:
            if not _is_retryable_mcp_transport_error(exc):
                raise
            logger.warning(
                "Retrying MCP tool %s once after transport/session failure: %s",
                self.name,
                exc,
                exc_info=True,
            )

        await asyncio.sleep(self._retry_backoff_seconds)
        return await self._invoke_wrapped(args, kwargs, config, run_manager)

    async def _invoke_wrapped(
        self,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        config: RunnableConfig | None,
        run_manager: AsyncCallbackManagerForToolRun | None,
    ) -> Any:
        return await self._wrapped._arun(  # noqa: SLF001
            *args,
            config=config,
            run_manager=run_manager,
            **kwargs,
        )
