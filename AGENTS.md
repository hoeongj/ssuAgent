# AGENTS.md - ssuAgent

Python LangGraph campus assistant agent connecting to ssuMCP.

## Workflow

- Design/review and execution roles are coordinated by the mp root workflow.
- Authorship: ghdtjdwn <seongjuice999@gmail.com>. No AI attribution anywhere.
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
- TODO (follow-up, NOT done): to activate `/agent` auth, set `AGENT_API_KEY` on ssuAgent AND make ssuAI send the `X-Agent-Key` header (both sides, or all requests 401). Separately, narrow `ALLOWED_ORIGINS` from `*` to the real frontend origin.

## Phase Roadmap (Phases 1-4 complete)

- Phase 1 (DONE): single ReAct agent, public ssuMCP tools (meal/library/notice), scaffolding
- Phase 2 (DONE): supervisor multi-agent sub-graphs per domain (academic/library/lms), auth tools (library reservation HITL), streaming
- Phase 3 (DONE): ssuAI frontend integration (web UI for agent chat, SSE)
- Phase 4 (DONE): LlamaIndex official-source RAG (SimpleVectorStore + RelevancyEvaluator)
- Wave 4 security hardening (SHIPPED): LLM provider key guards, env CORS (`ALLOWED_ORIGINS`), opt-in `/agent` API-key gate (`AGENT_API_KEY`). Activation of `/agent` auth + CORS narrowing remain as follow-up (see Security section).

## Commit Convention

Conventional Commits: feat/fix/refactor/chore/docs + scope (agent/mcp/graph/test)
Branch: feat/fix/... + kebab-case. PR required; merge via local fast-forward only.
