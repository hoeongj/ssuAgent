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
