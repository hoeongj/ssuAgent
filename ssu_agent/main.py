"""
FastAPI app — SSE streaming entry point for ssuAgent.

Endpoints:
  POST /agent/stream   — start or continue a conversation, stream SSE
  POST /agent/resume   — resume after HITL interrupt (user approval/denial)
  GET  /health         — liveness check

SSE event types emitted:
  {"type": "text",    "content": "..."}    — LLM token chunk
  {"type": "handoff", "agent": "library"}  — sub-agent routing started
  {"type": "tool",    "name": "..."}       — any tool call started (debug)
  {"type": "interrupt","data": {...}}       — HITL payload awaiting user decision
  {"type": "done"}                          — graph reached END

MCP session lifecycle (thread_id ↔ mcp_session_id):
  Every FastAPI request carries a `thread_id` (stable per user/device) used
  as the LangGraph checkpoint key. The `mcp_session_id` (ssuMCP private tool
  auth token) is passed in the request body and stored in SsuAgentState so
  sub-agents can include it in private MCP tool calls.

  The two IDs are intentionally separate:
  - thread_id: conversation persistence (Postgres checkpoint)
  - mcp_session_id: ssuMCP auth (externally managed by ssuAI login flow)
  A single thread can switch mcp_session_id across requests (e.g. re-login),
  so the graph always takes the latest value from the request rather than
  reading it from checkpoint.

Checkpointer (Postgres):
  Uses AsyncPostgresSaver from langgraph-checkpoint-postgres backed by
  an AsyncConnectionPool (psycopg3). autocommit=True is required by LangGraph.
  setup() creates the checkpoint tables on first startup.

Streaming optimisation (Gemini suggestion applied):
  astream_events(version="v2") yields rich event dicts. We filter:
  - on_chat_model_stream   → text chunks (user sees typing)
  - on_tool_start          → handoff/tool events (user sees "routing...")
  - on_interrupt           → HITL payload (client shows approval dialog)
  Skipping on_chain_* and on_retriever_* avoids SSE noise.
"""

from __future__ import annotations

import json
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from psycopg_pool import AsyncConnectionPool
from pydantic import BaseModel

from ssu_agent import config
from ssu_agent.supervisor.graph import build_supervisor_graph

# Graph reference — set during lifespan startup
_graph = None


@asynccontextmanager
async def _lifespan(app: FastAPI):
    """FastAPI lifespan: open Postgres connection pool, build graph, keep alive."""
    global _graph
    async with AsyncConnectionPool(
        conninfo=config.DATABASE_URL,
        max_size=5,
        kwargs={"autocommit": True, "prepare_threshold": 0},
    ) as pool:
        checkpointer = AsyncPostgresSaver(pool)
        await checkpointer.setup()
        _graph = await build_supervisor_graph(checkpointer=checkpointer)
        yield


app = FastAPI(title="ssuAgent", version="0.2.0", lifespan=_lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request / response models ─────────────────────────────────────────────────


class AgentRequest(BaseModel):
    message: str
    thread_id: str = ""  # "" → new conversation
    mcp_session_id: str | None = None


class ResumeRequest(BaseModel):
    thread_id: str
    approved: bool
    action_id: int | None = None
    mcp_session_id: str | None = None


# ── SSE helpers ───────────────────────────────────────────────────────────────


def _sse(payload: dict) -> str:
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


async def _stream_graph(input_data: dict | object, config: dict):
    """Yield SSE strings from graph.astream_events."""
    async for event in _graph.astream_events(input_data, config=config, version="v2"):
        etype = event.get("event", "")
        name = event.get("name", "")

        if etype == "on_chat_model_stream":
            chunk = event["data"]["chunk"]
            content = chunk.content if hasattr(chunk, "content") else str(chunk)
            if content:
                yield _sse({"type": "text", "content": content})

        elif etype == "on_tool_start":
            if name.startswith("transfer_to_"):
                agent = name.replace("transfer_to_", "").replace("_agent", "")
                yield _sse(
                    {
                        "type": "handoff",
                        "agent": agent,
                        "status": "routing",
                        "message": f"{agent} 에이전트로 전환 중...",
                    }
                )
            else:
                yield _sse({"type": "tool", "name": name})

        elif etype == "on_interrupt":
            interrupt_data = event.get("data", {})
            yield _sse({"type": "interrupt", "data": interrupt_data})
            return  # Pause SSE; client waits for /agent/resume

    yield _sse({"type": "done"})


# ── Endpoints ─────────────────────────────────────────────────────────────────


@app.post("/agent/stream")
async def stream_agent(req: AgentRequest):
    """Start or continue a conversation. Streams SSE events."""
    thread_id = req.thread_id or str(uuid.uuid4())
    initial_state = {
        "messages": [{"role": "user", "content": req.message}],
        "mcp_session_id": req.mcp_session_id,
        "active_agent": None,
        "pending_action": None,
    }
    config = {"configurable": {"thread_id": thread_id}}

    return StreamingResponse(
        _stream_graph(initial_state, config),
        media_type="text/event-stream",
        headers={
            "X-Thread-Id": thread_id,
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        },
    )


@app.post("/agent/resume")
async def resume_agent(req: ResumeRequest):
    """Resume a graph paused by a library HITL interrupt.

    The client sends {approved: bool, action_id: int} after the user decides.
    LangGraph resumes the library_agent's execute_confirm node if approved,
    or short-circuits to done if denied.
    """
    from langgraph.types import Command

    resume_payload = {
        "approved": req.approved,
        "action_id": req.action_id,
        "mcp_session_id": req.mcp_session_id,
    }
    config = {"configurable": {"thread_id": req.thread_id}}

    return StreamingResponse(
        _stream_graph(Command(resume=resume_payload), config),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
    )


@app.get("/health")
async def health():
    return {"status": "UP", "version": app.version}
