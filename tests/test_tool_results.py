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

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from ssu_agent.tool_results import content_to_text, sanitize_tool_pairing, tool_result_to_text


def test_str_passthrough():
    assert tool_result_to_text("hello") == "hello"


def test_content_to_text_reuses_tool_result_flattening():
    assert content_to_text([{"type": "text", "text": "assistant text", "index": 0}]) == (
        "assistant text"
    )


def test_content_to_text_flattens_message_content_string_blocks():
    assert content_to_text(["hello", {"text": "there"}]) == "hello there"


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


def test_sanitize_tool_pairing_drops_orphan_tool_message():
    human = HumanMessage(content="질문")
    orphan = ToolMessage(content="orphan", tool_call_id="missing")

    sanitized = sanitize_tool_pairing([human, orphan])

    assert sanitized == [human]


def test_sanitize_tool_pairing_strips_dangling_tool_calls_without_mutating():
    ai = AIMessage(
        content="도구를 호출합니다.",
        tool_calls=[
            {"id": "call-a", "name": "tool_a", "args": {}, "type": "tool_call"},
            {"id": "call-b", "name": "tool_b", "args": {}, "type": "tool_call"},
        ],
    )
    tool_result = ToolMessage(content="A result", tool_call_id="call-a")
    next_user = HumanMessage(content="다음 질문")

    sanitized = sanitize_tool_pairing([ai, tool_result, next_user])

    assert sanitized[0] is not ai
    assert sanitized[0].content == ai.content
    assert [tc["id"] for tc in sanitized[0].tool_calls] == ["call-a"]
    assert sanitized[1] is tool_result
    assert sanitized[2] is next_user
    assert [tc["id"] for tc in ai.tool_calls] == ["call-a", "call-b"]


def test_sanitize_tool_pairing_keeps_well_formed_history_unchanged():
    human = HumanMessage(content="질문")
    ai = AIMessage(
        content="",
        tool_calls=[{"id": "call-a", "name": "tool_a", "args": {}, "type": "tool_call"}],
    )
    tool_result = ToolMessage(content="A result", tool_call_id="call-a")
    final = AIMessage(content="답변")
    messages = [human, ai, tool_result, final]

    sanitized = sanitize_tool_pairing(messages)

    assert sanitized == messages
    assert sanitized[0] is human
    assert sanitized[1] is ai
    assert sanitized[2] is tool_result
    assert sanitized[3] is final


def test_sanitize_tool_pairing_prod_failure_shape():
    ai = AIMessage(
        content="",
        tool_calls=[{"id": "call-a", "name": "tool_a", "args": {}, "type": "tool_call"}],
    )
    valid_result = ToolMessage(content="A result", tool_call_id="call-a")
    orphan_result = ToolMessage(content="B result", tool_call_id="call-b")

    sanitized = sanitize_tool_pairing([ai, valid_result, orphan_result])

    assert sanitized == [ai, valid_result]
