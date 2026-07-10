# ssuAgent

[![CI](https://github.com/ghdtjdwn/ssuAgent/actions/workflows/ci.yml/badge.svg)](https://github.com/ghdtjdwn/ssuAgent/actions/workflows/ci.yml)

**한국어** [README.md](README.md) · **English** (this document)

> 🧩 **Soongsil Campus AI Platform** (1 of 4 services) · [ssuMCP](https://github.com/ghdtjdwn/ssuMCP) · [ssuAI](https://github.com/ghdtjdwn/ssuAI) · **ssuAgent** · [ssu-ai-service](https://github.com/ghdtjdwn/ssu-ai-service) · 🟢 [Live](https://ssuai.vercel.app)

A LangGraph-based **multi-agent** campus AI agent for Soongsil University that connects to the university's MCP server ([ssuMCP](https://github.com/ghdtjdwn/ssuMCP)). It integrates with the [ssuAI](https://github.com/ghdtjdwn/ssuAI) web chat UI via SSE streaming.

🟢 **Live** — try it in the chat: <https://ssuai.vercel.app/chat> (this agent answers over SSE)

## Architecture

```
User Query
    │
    ▼
Supervisor (LangGraph StateGraph) ── classifies the query → routes by domain
    ├── academic agent   (academics / grades / graduation / scholarships)
    ├── library agent    (seat recommendation & reservation, prepare/confirm HITL)
    └── lms agent        (courses / assignments / material export)
    │  Streamable HTTP (MCP 2025-03-26)
    ▼
ssuMCP Server (Spring Boot 4)
    ├── Pyxis (library)
    ├── u-SAINT (academics / grades)
    └── LMS (courses / assignments)
```

- **Multi-provider LLM fallback**: `llm_factory.get_llm_sequence()` falls back in the order Groq (llama-3.3-70b, free 14,400 req/day) → Gemini → OpenRouter, removing the single point of failure. Each provider is added to the sequence only when its API key is set — `GROQ_API_KEY` for Groq, `GOOGLE_API_KEY` for Gemini, `OPENROUTER_API_KEY` for OpenRouter. If no key is set at all, `create_llm()` raises an explicit `RuntimeError` (no silent misbehavior). Groq uses `ChatGroq` instead of the generic `ChatOpenAI` wrapper — the generic wrapper serializes assistant content as a list, which makes Groq return 400 on the second tool call.
- **State persistence**: conversation state is persisted with the LangGraph Postgres checkpointer.
- **Thread ownership binding**: the `thread_owners` table binds each `thread_id` to the `mcp_session_id` that first created it, so other sessions cannot read or resume the same checkpoint.
- **HITL safeguard**: library write actions only ever run through the two-step flow `prepare_*` → user approval → `confirm_action`.

### Key components

| Component | File | Role |
|---|---|---|
| Supervisor | `supervisor/graph.py` | Classifies the query → routes to a domain via a `ROUTE_TO:X` marker. `create_react_agent` in LangGraph 1.2.4 does not propagate a `Command` returned by a tool up to the parent graph, so the workaround is a pattern where the routing tool returns a marker string and a `post_supervisor` node scans it and emits `Command(goto=X)` (ADR 0001) |
| Domain agents | `agents/{academic,library,lms}.py` | Per-domain MCP tool bundle + manual `bind_tools` fallback loop (removes the single-provider point of failure) |
| MCP client | `mcp_client.py` | Connects to ssuMCP over Streamable HTTP (MCP 2025-03-26), loads tools dynamically |
| LLM factory | `llm_factory.py` | `get_llm_sequence()` — Groq → Gemini → OpenRouter priority fallback |
| Checkpointer | LangGraph Postgres | Persists conversation state across turns |

## Why LangGraph?

| Approach | Reasoning |
|------|------|
| LangChain LCEL | Fine for simple chains. Hard to express state, loops, and branching |
| Raw function calling | Orchestration code managed by hand. Multi-step complexity grows |
| **LangGraph** (chosen) | Expresses state, branching, and loops as an explicit StateGraph. Easy to extend to the Phase 2 multi-agent design |

## Setup

```bash
pip install uv
uv sync --extra dev
```

## Run

At least one LLM provider key is required (any single one is enough; if all three are set, they are used in fallback order):

```bash
export GROQ_API_KEY=<your-groq-key>        # 1st priority (optional)
export GOOGLE_API_KEY=<your-gemini-key>    # 2nd priority (optional)
export OPENROUTER_API_KEY=<your-or-key>    # 3rd priority (optional)
export SSUMCP_URL=https://ssumcp.duckdns.org/mcp  # optional, this is the default
# Run the FastAPI app (SSE streaming endpoint)
uv run uvicorn ssu_agent.main:app --host 0.0.0.0 --port 8000

# Call it from another terminal (add -H "X-Agent-Key: <key>" if AGENT_API_KEY is set)
curl -N -X POST http://localhost:8000/agent/stream \
  -H "Content-Type: application/json" \
  -d '{"message": "오늘 학식 알려줘"}'   # "What's on the cafeteria menu today?"
```

## Security / configuration

Key runtime environment variables (all optional; defaults preserve the existing prod behavior):

| Env var | Default | Role |
|---|---|---|
| `ALLOWED_ORIGINS` | `*` (allow all) | CORS allow-list. Comma-separated list of origins (parsed in `config.py` → `CORSMiddleware` in `main.py`). A single `*` keeps the previous allow-all behavior. Narrowing it to the actual frontend origins enables CORS protection. |
| `AGENT_API_KEY` | empty (gate off) | **Opt-in** API key gate for the `/agent/*` endpoints (the `verify_agent_key` dependency in `main.py`). Empty means no-op (existing prod behavior unchanged). When set, every `/agent` request must send a matching `X-Agent-Key` header (compared with `secrets.compare_digest` to defend against timing attacks); a missing or wrong key gets 401. |
| `AGENT_RATE_LIMIT` | `30/minute` | Per-IP inbound rate limit for `/agent/stream` and `/agent/resume` (slowapi syntax, the `limiter` in `main.py`). Keyed by the leftmost X-Forwarded-For hop (the real client IP behind the ingress). Exceeding it returns 429. Background in ADR 0009. |
| `AGENT_MAX_MESSAGE_CHARS` | `8000` | Maximum character count of a single request `message` (pydantic `Field(max_length=…)`). Exceeding it returns 422 (oversized-payload guard, ADR 0009). |
| LLM keys | — | Of `GROQ_API_KEY`/`GOOGLE_API_KEY`/`OPENROUTER_API_KEY`, only the ones that are set join the fallback sequence (see Architecture above). |
| `GEMINI_MODEL` | `gemini-2.5-flash` | Model name used by the Gemini provider (`llm_factory.py`; only used when `GOOGLE_API_KEY` is set). |

### Thread ownership binding

`/agent/stream` and `/agent/resume` check the `thread_owners` table before running the graph. A new `thread_id` stores the `mcp_session_id` of the first request as its owner; any subsequent access to the same `thread_id` from a different `mcp_session_id` returns 403. Anonymous threads without an `mcp_session_id` keep a `NULL` owner, allowing the existing no-session flow.

Checkpoints created before this was deployed have no owner row, so the first requester after deployment claims ownership. Since `mcp_session_id` changes on re-login, ssuAI must clear `ssuagent_thread_id` on logout to avoid a self-403. See `docs/adr/0010-agent-thread-ownership-binding.md` (Korean) for the full decision background.

### `/agent` endpoint authentication

`/agent/*` is protected by an API key gate. ssuAgent enforces an `X-Agent-Key` header matching `AGENT_API_KEY` (`verify_agent_key` in `main.py`, 401 on mismatch), and the ssuAI frontend injects the key in a server-only proxy (`lib/server/agentProxy.ts`) — the browser only calls same-origin `/api/agent/*`, so the key is never exposed to the client. CORS is restricted to the frontend origins via `ALLOWED_ORIGINS`. See `docs/adr/0009-agent-edge-hardening.md` (Korean) for the design background and verification steps.

## Test

```bash
uv run pytest
```

## Phase Roadmap

| Phase | Scope | Status |
|-------|------|------|
| 1 | Single ReAct agent, 3 public tools (meals / library / notices) | ✅ Done |
| 2 | Per-domain supervisor multi-agent, authenticated library reservation tools (HITL), streaming responses | ✅ Done |
| 3 | ssuAI frontend integration (web UI chat, SSE) | ✅ Done |
| Security hardening | LLM provider key guard, env-based CORS (`ALLOWED_ORIGINS`), `/agent` API key gate (`AGENT_API_KEY`), thread ownership binding | ✅ Done |

> Implementation note: due to a looping issue with `create_react_agent`, the domain agents were switched to a manual `bind_tools` fallback loop (removing the single-provider point of failure). See `docs/adr/` (Korean) for the rationale and alternatives considered.
