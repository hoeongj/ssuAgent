# AGENTS.md - ssuAgent

Python LangGraph campus assistant agent connecting to ssuMCP.

## Workflow

- Design/review and execution roles are coordinated by the mp root workflow.
- Authorship: hoengj <seongjuice999@gmail.com>. No AI attribution anywhere.
  No attribution trailers or assistant/vendor names in commits, PRs, code comments, docs.
- Decisions: web search first, evaluate portfolio value > trend fit > completion, confirm with user.
- Docs: Korean/English mix in README/docs. LLM-facing files (this file, prompts) = English only.

## Commands

- Install: `uv sync --extra dev`
- Test: `uv run pytest`
- Lint: `uv run ruff check . && uv run ruff format --check .`
- Run: set at least one LLM provider key (`GROQ_API_KEY` / `GOOGLE_API_KEY` / `OPENROUTER_API_KEY`), then call `ssu_agent.graph.run_query(...)` from Python.

## LLM providers (llm_factory.py)

- Fallback order: Groq -> Gemini -> OpenRouter. Each provider is added to the sequence ONLY when its API key is set (`GROQ_API_KEY` / `GOOGLE_API_KEY` / `OPENROUTER_API_KEY`). `create_llm()` raises `RuntimeError` if no provider key is configured.

## Security / configuration (Wave 4, shipped)

- `ALLOWED_ORIGINS` (default `*`): comma-separated CORS allow-list parsed in `config.py`, applied in `main.py` CORSMiddleware. Lone `*` keeps wide-open behavior; set real origin to narrow.
- `AGENT_API_KEY` (default empty): opt-in API-key gate on `/agent/*` (`verify_agent_key` dependency in `main.py`). Empty => no-op (prod behavior unchanged). When set, requests must send matching `X-Agent-Key` header (`secrets.compare_digest`); else 401.
- Thread ownership binding: `thread_owners` binds client-supplied `thread_id` to the first creating `mcp_session_id`; different sessions get 403 on `/agent/stream` and `/agent/resume`. Anonymous threads keep `owner NULL`. See ADR 0010.
- DONE (`/agent` auth active in prod): ssuAgent enforces it via `verify_agent_key` (`main.py`, 401 on missing/wrong `X-Agent-Key`; wired on `/agent/stream` and `/agent/resume`); ssuAI injects the key server-side in `lib/server/agentProxy.ts`, so the browser only hits same-origin `/api/agent/*` (PR #205 `c891ba6`). Prod 3-way verified: no key => 401, correct key direct => 422 (auth passes, body validation), via proxy => 422. Remaining follow-up: narrow `ALLOWED_ORIGINS` from `*` to the real frontend origin.

## Phase Roadmap (Phases 1-4 complete)

- Phase 1 (DONE): single ReAct agent, public ssuMCP tools (meal/library/notice), scaffolding
- Phase 2 (DONE): supervisor multi-agent sub-graphs per domain (academic/library/lms), auth tools (library reservation HITL), streaming
- Phase 3 (DONE): ssuAI frontend integration (web UI for agent chat, SSE)
- Phase 4 (DONE): LlamaIndex official-source RAG (SimpleVectorStore + RelevancyEvaluator)
- Wave 4 security hardening (SHIPPED): LLM provider key guards, env CORS (`ALLOWED_ORIGINS`), `/agent` API-key gate (`AGENT_API_KEY`), thread ownership binding. `/agent` auth is now active in prod (PR #205 `c891ba6`); only CORS narrowing remains as follow-up (see Security section).

## Commit Convention

Conventional Commits: feat/fix/refactor/chore/docs + scope (agent/mcp/graph/test)
Branch: feat/fix/... + kebab-case. PR required; merge via local fast-forward only.
