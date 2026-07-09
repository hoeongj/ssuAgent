"""
FastAPI app — SSE streaming entry point for ssuAgent.

Endpoints:
  POST /agent/stream   — start or continue a conversation, stream SSE
  POST /agent/resume   — resume after HITL interrupt (user approval/denial)
  GET  /health         — liveness check

SSE event types emitted:
  {"type": "text",    "content": "..."}    — LLM token chunk
  {"type": "handoff", "agent": "library"}  — sub-agent routing started
  {"type": "tool",    "name": "..."}       — tool call started (Korean UX label via _TOOL_LABELS)
  {"type": "interrupt","data": {...}}       — HITL payload awaiting user decision
  {"type": "done"}                          — graph reached END

MCP session lifecycle (thread_id ↔ mcp_session_id ↔ principal):
  Every FastAPI request carries a `thread_id` (stable per user/device) used
  as the LangGraph checkpoint key. The `mcp_session_id` (ssuMCP private tool
  auth token) is passed in the request body and stored in SsuAgentState so
  sub-agents can include it in private MCP tool calls.

  The three concepts are intentionally separate:
  - thread_id: conversation persistence (Postgres checkpoint)
  - mcp_session_id: ssuMCP auth (externally managed by ssuAI login flow,
    ROTATES on every re-login — never a stable per-user key)
  - principal: OPTIONAL stable per-user subject supplied by the caller (e.g. a
    frontend JWT subject). ssuAgent does not derive this itself — see ADR 0011
    for why (ssuMCP's get_auth_status deliberately never returns a student id
    / principalKey; resolving one would require a cross-repo ssuMCP change).

  A thread's ownership is claimed/verified by claim_or_verify_thread_owner:
  - principal present -> bound to the (hashed) principal. Stable across
    mcp_session_id rotation: the same principal from a different session still
    resolves to the same thread. A different principal is rejected (403).
  - principal absent, mcp_session_id present -> bound to that session only
    (legacy behavior, unchanged): a different session is rejected (403).
  - neither present -> anonymous thread (owner NULL), open to any caller, same
    as before ADR 0011.
  A pre-existing session-owned thread is lazily upgraded to principal
  ownership the first time its rightful session presents a principal (see ADR
  0011 "마이그레이션 규칙"). The graph still takes the latest mcp_session_id
  from the request for MCP tool calls after ownership is verified.

Checkpointer (Postgres):
  Uses AsyncPostgresSaver from langgraph-checkpoint-postgres backed by
  an AsyncConnectionPool (psycopg3). autocommit=True is required by LangGraph.
  setup() creates the checkpoint tables on first startup. The same pool also
  creates thread_owners, which binds client-supplied thread_id values to the
  creating mcp_session_id or, when supplied, a stable principal (ADR 0011).

Streaming optimisation:
  astream_events(version="v2") yields rich event dicts. We filter:
  - on_chat_model_stream   → text chunks (user sees typing)
  - on_tool_start          → handoff/tool events (user sees "routing...")
  - on_chain_stream        → HITL payload when a chunk carries __interrupt__
                             (client shows approval dialog). langgraph 1.2.4 does
                             NOT emit an on_interrupt event — the interrupt rides
                             inside an on_chain_stream chunk. See _extract_interrupt.
  Other on_chain_* / on_retriever_* chunks are dropped (SSE noise, and raw state
  can carry mcp_session_id — never forwarded).
"""

from __future__ import annotations

import hashlib
import json
import logging
import secrets
import uuid
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from psycopg_pool import AsyncConnectionPool
from pydantic import BaseModel, Field
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from ssu_agent import config
from ssu_agent.supervisor.graph import build_supervisor_graph

# uvicorn does not attach a handler to the root logger, so ssu_agent's INFO-level
# latency instrumentation (react_loop per-turn provider + per-tool timing) would
# fall through to logging.lastResort (WARNING) and vanish. Attach our own stream
# handler at INFO so those records reliably reach the container logs.
_ssu_logger = logging.getLogger("ssu_agent")
_ssu_logger.setLevel(logging.INFO)
if not any(isinstance(h, logging.StreamHandler) for h in _ssu_logger.handlers):
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter("%(levelname)s:%(name)s:%(message)s"))
    _ssu_logger.addHandler(_handler)
_ssu_logger.propagate = False

logger = logging.getLogger(__name__)


def _client_ip(request: Request) -> str:
    """Per-IP key for rate limiting. Behind the k3s ingress every request shares
    the ingress socket address, so prefer the left-most X-Forwarded-For hop (the
    real client) and fall back to the socket address. Mirrors ssuMCP
    ClientIpResolver (ADR 0061)."""
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        first = forwarded.split(",")[0].strip()
        if first:
            return first
    return get_remote_address(request)


# Per-IP inbound throttle on /agent/* (mirrors ssuMCP ADR 0061): the endpoints
# fan out to paid LLM providers, so an unauthenticated flood is a cost/DoS risk.
# In-memory storage = per-process (prod runs a single replica; documented caveat).
limiter = Limiter(key_func=_client_ip)

# Graph and pool references — set during lifespan startup
_graph = None
_pool: AsyncConnectionPool | None = None

_THREAD_OWNER_FORBIDDEN_DETAIL = "이 대화는 현재 세션의 소유가 아닙니다."

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
    "get_meal_weekly": "주간 식단 조회 중...",
    "get_dorm_weekly_meal": "기숙사 주간 식단 조회 중...",
    "search_campus_facilities": "캠퍼스 시설 검색 중...",
    # LMS (강의/과제/자료 내보내기)
    "get_my_lms_courses": "수강 강의 조회 중...",
    "get_my_lms_materials": "강의자료 조회 중...",
    "get_my_lms_terms": "학기 목록 조회 중...",
    "get_lms_dashboard": "LMS 대시보드 조회 중...",
    "prepare_lms_material_export": "강의자료 내보내기 준비 중...",
    "confirm_lms_material_export": "강의자료 내보내기 확정 중...",
    # 학사일정 · 학칙/졸업/장학 근거
    "get_academic_calendar": "학사일정 조회 중...",
    "find_academic_calendar_events": "학사일정 검색 중...",
    "get_academic_policy_brief": "학칙 요약 조회 중...",
    "search_academic_policy_sources": "학칙 근거 검색 중...",
    "check_scholarship_policy": "장학 정책 확인 중...",
    "evaluate_graduation_with_policy": "졸업 요건 평가 중...",
    # 도서관 대기 · 열람실 좌석
    "get_room_available_seats": "열람실 좌석 조회 중...",
    "get_library_wait_status": "대기 현황 조회 중...",
    "wait_for_library_seat": "좌석 대기 등록 중...",
    "cancel_library_wait": "대기 취소 중...",
    # 세션
    "logout_provider": "로그아웃 중...",
    "logout_all": "전체 로그아웃 중...",
}


@asynccontextmanager
async def _lifespan(app: FastAPI):
    """FastAPI lifespan: open Postgres connection pool, build graph, keep alive."""
    global _graph, _pool
    async with AsyncConnectionPool(
        conninfo=config.DATABASE_URL,
        # Pool ceiling ~= concurrent streams x checkpointer ops. Five fits the
        # current single-pod dozens-of-users shape; raise with replicas/HPA per load test.
        max_size=config.AGENT_PG_POOL_MAX_SIZE,
        kwargs={"autocommit": True, "prepare_threshold": 0},
    ) as pool:
        try:
            checkpointer = AsyncPostgresSaver(pool)
            await checkpointer.setup()
            await _setup_thread_owners(pool)
            _pool = pool
            _graph = await build_supervisor_graph(checkpointer=checkpointer)
            yield
        finally:
            _graph = None
            _pool = None


async def _setup_thread_owners(pool: AsyncConnectionPool) -> None:
    async with pool.connection() as conn:
        async with conn.cursor() as cur:
            await cur.execute(
                """
                CREATE TABLE IF NOT EXISTS thread_owners (
                    thread_id TEXT PRIMARY KEY,
                    owner TEXT,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
                )
                """
            )
            # ADR 0011: owner_kind distinguishes a stable-principal owner from a
            # legacy/session-scoped owner. ADD COLUMN IF NOT EXISTS keeps this
            # additive over the ADR 0010 table already live in prod — existing
            # rows get owner_kind = NULL, which claim_or_verify_thread_owner
            # treats identically to owner_kind = 'session' (see docstring there).
            await cur.execute("ALTER TABLE thread_owners ADD COLUMN IF NOT EXISTS owner_kind TEXT")


app = FastAPI(title="ssuAgent", version="0.2.0", lifespan=_lifespan)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

app.add_middleware(
    CORSMiddleware,
    allow_origins=config.ALLOWED_ORIGINS,
    # Narrowed from "*": the API only serves POST /agent/* and GET /health.
    allow_methods=["GET", "POST"],
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


def _hash_principal(principal: str) -> str:
    """One-way digest of a caller-supplied stable principal before it touches storage.

    ssuMCP's get_auth_status deliberately never returns a raw student id (see ADR
    0011), so ssuAgent never derives `principal` itself — it only ever receives
    whatever value a future caller chooses to send. Hashing it before it reaches
    `thread_owners` means a DB dump never reveals the plaintext subject, while
    equality comparisons (the only operation ownership binding needs) still work
    identically on the digest.
    """
    return hashlib.sha256(principal.encode("utf-8")).hexdigest()


async def claim_or_verify_thread_owner(
    thread_id: str,
    mcp_session_id: str | None,
    principal: str | None = None,
) -> None:
    """Bind a new thread to its owner, or verify the current caller against it.

    ADR 0011. `principal` is an optional stable per-user subject (e.g. a future
    frontend JWT subject) — see the module docstring. Resolution order:

    1. `principal` present -> the thread is owned by hash(principal). This
       survives `mcp_session_id` rotation (re-login): the same principal from a
       *different* session still matches. A *different* principal is rejected.
    2. `principal` absent, `mcp_session_id` present -> owned by that session only
       (ADR 0010 behavior, unchanged): a different session is rejected.
    3. Neither present -> anonymous thread (owner NULL), open to any caller,
       unchanged from ADR 0010.

    Lazy migration: a thread claimed under rule 2 (session-owned) is upgraded to
    rule 1 (principal-owned) the moment its rightful session presents a
    `principal` — i.e. on the first verified access from that session after the
    caller starts sending one. See docs/adr/0011 for why lazy beats a batch
    migration here (there is no batch of principals to backfill from — the
    value only exists once a caller starts sending it).
    """
    if _pool is None:
        raise HTTPException(status_code=503, detail="Agent storage is not ready")

    hashed_principal = _hash_principal(principal) if principal else None

    if hashed_principal is not None:
        claim_owner, claim_kind = hashed_principal, "principal"
    elif mcp_session_id is not None:
        claim_owner, claim_kind = mcp_session_id, "session"
    else:
        claim_owner, claim_kind = None, None

    async with _pool.connection() as conn:
        async with conn.cursor() as cur:
            await cur.execute(
                """
                INSERT INTO thread_owners (thread_id, owner, owner_kind)
                VALUES (%s, %s, %s)
                ON CONFLICT (thread_id) DO NOTHING
                """,
                (thread_id, claim_owner, claim_kind),
            )
            await cur.execute(
                "SELECT owner, owner_kind FROM thread_owners WHERE thread_id = %s",
                (thread_id,),
            )
            row = await cur.fetchone()

            if row is None:
                raise HTTPException(status_code=503, detail="Agent storage is not ready")

            stored_owner, stored_kind = row

            if stored_owner is None:
                return  # Anonymous thread — open to any caller (ADR 0010).

            if stored_kind == "principal":
                if hashed_principal is not None and hashed_principal == stored_owner:
                    return
                raise HTTPException(status_code=403, detail=_THREAD_OWNER_FORBIDDEN_DETAIL)

            # stored_kind == "session", or NULL for rows written before ADR 0011
            # shipped (pre-existing ADR 0010 rows never had an owner_kind column
            # value) — both mean "owned by the mcp_session_id in `owner`".
            if stored_owner != mcp_session_id:
                raise HTTPException(status_code=403, detail=_THREAD_OWNER_FORBIDDEN_DETAIL)

            # Verified as the rightful session. If this request now carries a
            # principal, lazily upgrade the thread from session- to
            # principal-ownership so a future re-login (new mcp_session_id, same
            # principal) still finds it. Runs at most once per thread: after
            # this UPDATE, stored_kind is "principal" and the branch above
            # handles all later calls.
            if hashed_principal is not None:
                await cur.execute(
                    """
                    UPDATE thread_owners
                    SET owner = %s, owner_kind = 'principal'
                    WHERE thread_id = %s
                    """,
                    (hashed_principal, thread_id),
                )


# ── Request / response models ─────────────────────────────────────────────────


class AgentRequest(BaseModel):
    # Oversized-payload guard: cap the free-text message (config-tunable).
    message: str = Field(max_length=config.AGENT_MAX_MESSAGE_CHARS)
    thread_id: str = ""  # "" → new conversation
    mcp_session_id: str | None = None
    # ADR 0011: optional stable per-user subject (e.g. a frontend JWT subject),
    # independent of the rotating mcp_session_id. Absent today from every known
    # caller — see ADR 0011 for the follow-up this unblocks — so it MUST default
    # to None and every code path MUST keep working when it is never sent.
    principal: str | None = None


class ResumeRequest(BaseModel):
    thread_id: str
    approved: bool
    action_id: int | None = None
    mcp_session_id: str | None = None
    principal: str | None = None


# ── SSE helpers ───────────────────────────────────────────────────────────────


def _sse(payload: dict) -> str:
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


def _extract_interrupt(chunk: object) -> dict | None:
    """Return the HITL payload if this astream_events chunk carries an interrupt.

    langgraph 1.2.4's astream_events(version="v2") does NOT emit a dedicated
    on_interrupt event. When a node calls interrupt(), the graph pauses and the
    interrupt surfaces inside an on_chain_stream chunk shaped like
    {"__interrupt__": (Interrupt(value=<payload>, ...),)}. We forward only the
    first Interrupt's .value (the developer-controlled approval payload), never
    the surrounding chunk, so raw graph state is not leaked.
    """
    if isinstance(chunk, dict):
        interrupts = chunk.get("__interrupt__")
        if interrupts:
            first = interrupts[0]
            return getattr(first, "value", first)
    return None


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

            elif etype == "on_chain_stream":
                # langgraph surfaces an interrupt() pause inside a chain-stream
                # chunk (not via a dedicated event). Forward the HITL payload and
                # stop; the client shows the approval card and calls /agent/resume.
                interrupt_data = _extract_interrupt(event.get("data", {}).get("chunk"))
                if interrupt_data is not None:
                    yield _sse({"type": "interrupt", "data": interrupt_data})
                    return  # Pause SSE; client waits for /agent/resume

    except Exception:
        # Do not reflect the exception detail to the client — it can carry
        # internal stack / DB context. The full traceback is logged server-side.
        logger.exception("agent stream failed")
        yield _sse(
            {"type": "error", "message": "처리 중 오류가 발생했습니다. 잠시 후 다시 시도해 주세요."}
        )
        return

    yield _sse({"type": "done"})


# ── Endpoints ─────────────────────────────────────────────────────────────────


@app.post("/agent/stream", dependencies=[Depends(verify_agent_key)])
@limiter.limit(lambda: config.AGENT_RATE_LIMIT)
async def stream_agent(request: Request, req: AgentRequest):
    """Start or continue a conversation. Streams SSE events."""
    thread_id = req.thread_id or str(uuid.uuid4())
    await claim_or_verify_thread_owner(thread_id, req.mcp_session_id, req.principal)
    initial_state = {
        "messages": [{"role": "user", "content": req.message}],
        "mcp_session_id": req.mcp_session_id,
        "active_agent": None,
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
@limiter.limit(lambda: config.AGENT_RATE_LIMIT)
async def resume_agent(request: Request, req: ResumeRequest):
    """Resume a graph paused by a library HITL interrupt.

    The client sends {approved: bool, action_id: int} after the user decides.
    LangGraph resumes the library_agent's execute_confirm node if approved,
    or short-circuits to done if denied.
    """
    from langgraph.types import Command

    await claim_or_verify_thread_owner(req.thread_id, req.mcp_session_id, req.principal)
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
