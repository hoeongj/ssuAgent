"""
Tests for tool_result_to_text — the MCP content-block unwrap helper.

langchain_mcp_adapters tools are StructuredTool(response_format=
"content_and_artifact"); invoked with a bare args dict (no tool_call_id) they
return the raw content-block list instead of the inner text. Every case here
matches a shape this helper must handle to keep that list from leaking into
ToolMessage content or defeating downstream json.loads() dict checks.
"""

from __future__ import annotations

import json

from ssu_agent.tool_results import tool_result_to_text


def test_str_passthrough():
    assert tool_result_to_text("hello") == "hello"


def test_block_list_single_text_block():
    payload = json.dumps({"status": "OK", "data": {"actionId": 42}})
    result = tool_result_to_text([{"type": "text", "text": payload}])
    assert result == payload


def test_multi_block_concat():
    blocks = [
        {"type": "text", "text": "ab"},
        {"type": "text", "text": "cd"},
    ]
    assert tool_result_to_text(blocks) == "abcd"


def test_non_text_blocks_ignored_among_text_blocks():
    blocks = [
        {"type": "text", "text": "kept"},
        {"type": "image", "data": "base64..."},
    ]
    assert tool_result_to_text(blocks) == "kept"


def test_empty_list_falls_back_to_json_dumps():
    blocks = [{"type": "image", "data": "base64..."}]
    result = tool_result_to_text(blocks)
    assert result == json.dumps(blocks, ensure_ascii=False)


def test_empty_list_literal_falls_back_to_json_dumps():
    assert tool_result_to_text([]) == json.dumps([], ensure_ascii=False)


def test_dict_result_is_json_dumped():
    payload = {"status": "OK", "data": {"actionId": 1}}
    assert tool_result_to_text(payload) == json.dumps(payload, ensure_ascii=False)


def test_object_with_text_attribute():
    class _TextBlock:
        def __init__(self, text: str) -> None:
            self.text = text

    result = tool_result_to_text([_TextBlock("from-object")])
    assert result == "from-object"


def test_mixed_dict_and_object_blocks_concat_in_order():
    class _TextBlock:
        def __init__(self, text: str) -> None:
            self.text = text

    blocks = [{"type": "text", "text": "first-"}, _TextBlock("second")]
    assert tool_result_to_text(blocks) == "first-second"


def test_non_string_non_list_non_dict_falls_back_to_str():
    assert tool_result_to_text(42) == "42"
