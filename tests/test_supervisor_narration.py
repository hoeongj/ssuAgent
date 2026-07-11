"""Supervisor hand-off narration must be suppressed; a direct answer must be shown.

After routing via ``transfer_to_*``, the supervisor LLM tends to also emit a filler
narration ("...에이전트에게 전달했습니다") that is NOT the real answer — the sub-agent's
reply is. ``_stream_graph`` holds supervisor text, drops it when a transfer fires, and
flushes it only when the supervisor answered directly (no routing).
"""

from __future__ import annotations

import json
from typing import AsyncIterator

from langchain_core.callbacks import AsyncCallbackManagerForLLMRun, CallbackManagerForLLMRun
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult

from ssu_agent import main
from ssu_agent.supervisor.graph import build_supervisor_graph


class _Chunk:
    def __init__(self, text: str) -> None:
        self.content = text


class _FakeGraph:
    """Minimal stand-in that replays a fixed astream_events sequence."""

    def __init__(self, events: list[dict]) -> None:
        self._events = events

    async def astream_events(self, input_data, config, version):  # noqa: ARG002
        for event in self._events:
            yield event


class _StreamingMessagesListChatModel(BaseChatModel):
    responses: list[AIMessage]
    i: int = 0

    def bind_tools(self, tools, **kwargs):
        return self

    def _next_response(self) -> AIMessage:
        response = self.responses[self.i]
        self.i = self.i + 1 if self.i < len(self.responses) - 1 else 0
        return response

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs,
    ) -> ChatResult:
        return ChatResult(generations=[ChatGeneration(message=self._next_response())])

    async def _astream(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: AsyncCallbackManagerForLLMRun | None = None,
        **kwargs,
    ) -> AsyncIterator[ChatGenerationChunk]:
        response = self._next_response()
        chunks: list[AIMessageChunk] = []
        if response.content:
            chunks.append(AIMessageChunk(content=response.content))
        if response.tool_calls:
            chunks.append(
                AIMessageChunk(
                    content="",
                    tool_call_chunks=[
                        {
                            "name": tc["name"],
                            "args": json.dumps(tc.get("args", {}), ensure_ascii=False),
                            "id": tc.get("id"),
                            "index": idx,
                        }
                        for idx, tc in enumerate(response.tool_calls)
                    ],
                )
            )
        if not chunks:
            chunks.append(AIMessageChunk(content=""))
        chunks[-1].chunk_position = "last"
        for chunk in chunks:
            yield ChatGenerationChunk(message=chunk)

    @property
    def _llm_type(self) -> str:
        return "streaming-messages-list-chat-model"


def _model(node: str, text: str, tags: list[str] | None = None) -> dict:
    return {
        "event": "on_chat_model_stream",
        "name": "",
        "tags": tags or [],
        "metadata": {"langgraph_node": node},
        "data": {"chunk": _Chunk(text)},
    }


def _transfer(agent: str) -> dict:
    return {
        "event": "on_tool_start",
        "name": f"transfer_to_{agent}_agent",
        "metadata": {},
        "data": {},
    }


async def _collect(
    input_data: dict | object | None = None,
    config: dict | None = None,
) -> list[dict]:
    out: list[dict] = []
    async for sse in main._stream_graph(input_data or {"messages": []}, config or {}):
        out.append(json.loads(sse[len("data: ") :].strip()))
    return out


async def test_supervisor_handoff_narration_is_dropped(monkeypatch) -> None:
    events = [
        _model(
            "agent",
            "도서관 2층 예약은 도서관 에이전트에게 전달했습니다.",
            tags=["supervisor_llm"],
        ),
        _transfer("library"),
        _model("library_agent", "좌석 예약은 도서관 로그인 후 이용할 수 있어요."),
    ]
    monkeypatch.setattr(main, "_graph", _FakeGraph(events))
    out = await _collect()

    text = "".join(e["content"] for e in out if e["type"] == "text")
    assert "전달했습니다" not in text  # supervisor narration suppressed
    assert "로그인 후 이용" in text  # sub-agent's real answer shown
    assert any(e["type"] == "handoff" for e in out)


async def test_supervisor_direct_answer_is_kept(monkeypatch) -> None:
    # No routing: the supervisor's own answer IS the response and must be shown.
    events = [_model("agent", "안녕하세요! 무엇을 도와드릴까요?", tags=["supervisor_llm"])]
    monkeypatch.setattr(main, "_graph", _FakeGraph(events))
    out = await _collect()

    text = "".join(e["content"] for e in out if e["type"] == "text")
    assert "무엇을 도와드릴까요" in text


async def test_real_supervisor_graph_suppresses_handoff_narration(monkeypatch) -> None:
    from langgraph.checkpoint.memory import MemorySaver

    llm = _StreamingMessagesListChatModel(
        responses=[
            AIMessage(
                content="도서관 좌석 예약은 도서관 에이전트에게 전달했습니다.",
                tool_calls=[
                    {
                        "id": "call-transfer-library",
                        "name": "transfer_to_library_agent",
                        "args": {"query": "도서관 2층 좌석 예약해줘"},
                        "type": "tool_call",
                    }
                ],
            ),
            AIMessage(content="도서관 에이전트에게 전달했습니다."),
            AIMessage(content="[도서관 에이전트] 좌석 예약은 도서관 로그인 후 이용할 수 있어요."),
        ]
    )
    graph = await build_supervisor_graph(
        all_tools=[],
        llm=llm,
        checkpointer=MemorySaver(),
    )
    monkeypatch.setattr(main, "_graph", graph)

    out = await _collect(
        {
            "messages": [HumanMessage(content="도서관 2층 좌석 예약해줘")],
            "mcp_session_id": None,
            "active_agent": None,
        },
        {"configurable": {"thread_id": "real-supervisor-narration"}},
    )

    text = "".join(e["content"] for e in out if e["type"] == "text")
    assert "전달했습니다" not in text
    assert "좌석 예약은 도서관 로그인 후 이용할 수 있어요" in text
    assert any(e["type"] == "handoff" and e["agent"] == "library" for e in out)
