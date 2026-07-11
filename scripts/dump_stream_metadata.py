from __future__ import annotations

import asyncio
import json
from typing import AsyncIterator

from langchain_core.callbacks import AsyncCallbackManagerForLLMRun, CallbackManagerForLLMRun
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult

from ssu_agent.supervisor.graph import build_supervisor_graph


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


async def main() -> None:
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
    graph = await build_supervisor_graph(all_tools=[], llm=llm)
    input_data = {
        "messages": [HumanMessage(content="도서관 2층 좌석 예약해줘")],
        "mcp_session_id": None,
        "active_agent": None,
    }
    config = {"configurable": {"thread_id": "dump-stream-metadata"}}

    async for event in graph.astream_events(input_data, config=config, version="v2"):
        metadata = event.get("metadata", {})
        print(
            "event={event} name={name} tags={tags} langgraph_node={node} checkpoint_ns={ns}".format(
                event=event.get("event"),
                name=event.get("name"),
                tags=event.get("tags"),
                node=metadata.get("langgraph_node"),
                ns=metadata.get("checkpoint_ns"),
            )
        )


if __name__ == "__main__":
    asyncio.run(main())
