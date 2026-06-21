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
import logging
import secrets
import uuid
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from psycopg_pool import AsyncConnectionPool
from pydantic import BaseModel

from ssu_agent import config
from ssu_agent.supervisor.graph import build_supervisor_graph

logger = logging.getLogger(__name__)

# Graph reference — set during lifespan startup
_graph = None

_TOOL_LABELS: dict[str, str] = {
    "prepare_reserve_library_seat": "좌석 예약 준비 중...",
    "prepare_swap_library_seat": "좌석 이석 준비 중...",
    "prepare_cancel_library_seat": "좌석 반납 준비 중...",
    "confirm_action": "예약 확정 중...",
    "get_library_available_seats": "이용 가능 좌석 조회 중...",
    "get_library_seat_status": "좌석 상태 확인 중...",
    "get_library_seat_catalog": "좌석 목록 조회 중...",
    "recommend_library_seats": "좌석 추천 중...",
    "get_my_library_seat": "내 좌석 확인 중...",
    "get_my_library_loans": "대출 현황 조회 중...",
    "search_library_book": "도서 검색 중...",
    "get_auth_status": "인증 상태 확인 중...",
    "start_auth": "로그인 시작 중...",
    "get_my_grades": "성적 조회 중...",
    "get_my_schedule": "시간표 조회 중...",
    "get_my_chapel_info": "채플 정보 조회 중...",
    "get_my_scholarships": "장학금 조회 중...",
    "simulate_gpa": "GPA 시뮬레이션 중...",
    "check_graduation_requirements": "졸업 요건 확인 중...",
    "get_my_assignments": "과제 목록 조회 중...",
    "get_today_meal": "오늘 식단 조회 중...",
    "get_meal_by_date": "식단 조회 중...",
    "search_campus_facilities": "캠퍼스 시설 검색 중...",
}


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
    allow_origins=config.ALLOWED_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Auth dependency ─────────────────────────────────────────────────────────────


async def verify_agent_key(x_agent_key: str | None = Header(default=None)) -> None:
    """Opt-in API-key gate for /agent endpoints.

    No-op when config.AGENT_API_KEY is empty (default — prod behavior unchanged).
    When set, the request must carry an X-Agent-Key header equal to it.
    config is read live (not bound at import time) so the gate reflects the
    current env / test overrides. compare_digest guards against timing attacks;
    the `not x_agent_key` short-circuit avoids a TypeError when the header is
    missing (compare_digest rejects None).
    """
    expected = config.AGENT_API_KEY
    if not expected:
        return
    if not x_agent_key or not secrets.compare_digest(x_agent_key, expected):
        raise HTTPException(status_code=401, detail="Invalid or missing X-Agent-Key")


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
    try:
        async for event in _graph.astream_events(input_data, config=config, version="v2"):
            etype = event.get("event", "")
            name = event.get("name", "")

            if etype == "on_chat_model_stream":
                chunk = event["data"]["chunk"]
                content = chunk.content if hasattr(chunk, "content") else str(chunk)
                if isinstance(content, list):
                    content = "".join(
                        item["text"] if isinstance(item, dict) and "text" in item else str(item)
                        for item in content
                    )
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
                    label = _TOOL_LABELS.get(name, name)
                    yield _sse({"type": "tool", "name": name, "label": label})

            elif etype == "on_interrupt":
                interrupt_data = event.get("data", {})
                yield _sse({"type": "interrupt", "data": interrupt_data})
                return  # Pause SSE; client waits for /agent/resume

    except Exception as exc:
        logger.exception("agent stream failed")
        yield _sse({"type": "error", "message": str(exc)})
        return

    yield _sse({"type": "done"})


# ── Endpoints ─────────────────────────────────────────────────────────────────


@app.post("/agent/stream", dependencies=[Depends(verify_agent_key)])
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


@app.post("/agent/resume", dependencies=[Depends(verify_agent_key)])
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
