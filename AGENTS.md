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
- Run: set `GOOGLE_API_KEY`, then call `ssu_agent.graph.run_query(...)` from Python.

## Phase Roadmap (Phases 1-4 complete)

- Phase 1 (DONE): single ReAct agent, public ssuMCP tools (meal/library/notice), scaffolding
- Phase 2 (DONE): supervisor multi-agent sub-graphs per domain (academic/library/lms), auth tools (library reservation HITL), streaming
- Phase 3 (DONE): ssuAI frontend integration (web UI for agent chat, SSE)
- Phase 4 (DONE): LlamaIndex official-source RAG (SimpleVectorStore + RelevancyEvaluator)

## Commit Convention

Conventional Commits: feat/fix/refactor/chore/docs + scope (agent/mcp/graph/test)
Branch: feat/fix/... + kebab-case. PR required; merge via local fast-forward only.
