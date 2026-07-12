"""
Unwrap a LangChain tool invocation result down to its text payload.

Why this exists: langchain_mcp_adapters builds every MCP tool as a
StructuredTool(response_format="content_and_artifact"). When a tool is invoked
with a bare args dict (no "type": "tool_call" wrapper, so LangChain has no
tool_call_id to attach the artifact to), StructuredTool._format_output has
nowhere to put the artifact and falls back to returning the raw content
list — e.g. [{"type": "text", "text": "{...actual JSON payload...}"}] —
instead of the inner JSON string a plain @tool function would return. Every
call site that turns a tool result into ToolMessage content, or re-parses a
tool result looking for a field like actionId, must unwrap this shape first;
otherwise it ends up parsing/stringifying the wrapping list instead of the
payload itself.
"""

from __future__ import annotations

import json

from langchain_core.messages import AIMessage, ToolMessage


def tool_result_to_text(result: object) -> str:
    """Return the text payload of a tool invocation result.

    - str  -> returned as-is.
    - list -> concatenate every dict block with block.get("type") == "text"
      (also tolerates non-dict items exposing a `.text` attribute); if no text
      could be extracted (e.g. an artifact-only / non-text block list), falls
      back to json.dumps(result, ensure_ascii=False).
    - dict -> json.dumps(result, ensure_ascii=False).
    - anything else -> str(result).
    """
    if isinstance(result, str):
        return result
    if isinstance(result, list):
        parts: list[str] = []
        for block in result:
            if isinstance(block, dict):
                if block.get("type") == "text" and isinstance(block.get("text"), str):
                    parts.append(block["text"])
            else:
                text = getattr(block, "text", None)
                if isinstance(text, str):
                    parts.append(text)
        if parts:
            return "".join(parts)
        return json.dumps(result, ensure_ascii=False)
    if isinstance(result, dict):
        return json.dumps(result, ensure_ascii=False)
    return str(result)


def content_to_text(content: object) -> str:
    """Flatten an AIMessage.content (str or content-block list) to plain text."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and isinstance(item.get("text"), str):
                parts.append(item["text"])
            else:
                text = getattr(item, "text", None)
                if isinstance(text, str):
                    parts.append(text)
        return " ".join(parts)
    return ""


def _tool_call_id(tool_call: dict) -> str | None:
    value = tool_call.get("id")
    return value if isinstance(value, str) and value else None


def _content_has_text(content: object) -> bool:
    if isinstance(content, str):
        return bool(content.strip())
    if isinstance(content, list):
        for item in content:
            if isinstance(item, str) and item.strip():
                return True
            if isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str) and text.strip():
                    return True
            text = getattr(item, "text", None)
            if isinstance(text, str) and text.strip():
                return True
        return False
    return bool(content)


def sanitize_tool_pairing(messages: list) -> list:
    """Return an Anthropic-valid copy of an LLM message history.

    Anthropic requires each contiguous ToolMessage group to answer the tool_calls
    on the immediately preceding AIMessage, and every declared tool_call must
    have a result before the next non-tool message. The checkpointed state is
    left untouched: original message objects are reused when already valid, and
    copied only when dangling tool_calls need to be stripped.
    """
    sanitized: list = []
    i = 0
    while i < len(messages):
        msg = messages[i]

        if isinstance(msg, AIMessage) and msg.tool_calls:
            group: list[ToolMessage] = []
            j = i + 1
            while j < len(messages) and isinstance(messages[j], ToolMessage):
                group.append(messages[j])
                j += 1

            declared_ids = {
                tool_call_id for tc in msg.tool_calls if (tool_call_id := _tool_call_id(tc))
            }
            valid_tool_messages = [
                tool_msg for tool_msg in group if tool_msg.tool_call_id in declared_ids
            ]
            answered_ids = {tool_msg.tool_call_id for tool_msg in valid_tool_messages}
            matched_tool_calls = [
                tool_call
                for tool_call in msg.tool_calls
                if (tool_call_id := _tool_call_id(tool_call)) in answered_ids
            ]

            if len(matched_tool_calls) == len(msg.tool_calls):
                sanitized.append(msg)
            else:
                copied = msg.model_copy(update={"tool_calls": matched_tool_calls})
                if matched_tool_calls or _content_has_text(copied.content):
                    sanitized.append(copied)

            sanitized.extend(valid_tool_messages)
            i = j
            continue

        if isinstance(msg, ToolMessage):
            while i < len(messages) and isinstance(messages[i], ToolMessage):
                i += 1
            continue

        sanitized.append(msg)
        i += 1

    return sanitized
