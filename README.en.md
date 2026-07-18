# ssuAgent

[![CI](https://github.com/ghdtjdwn/ssuAgent/actions/workflows/ci.yml/badge.svg)](https://github.com/ghdtjdwn/ssuAgent/actions/workflows/ci.yml)
[![Security](https://github.com/ghdtjdwn/ssuAgent/actions/workflows/security.yml/badge.svg)](https://github.com/ghdtjdwn/ssuAgent/actions/workflows/security.yml)

[한국어](README.md) · **English**

A FastAPI/LangGraph orchestration service that routes ssuAI requests to domain-specific MCP tools
and manages conversation state, SSE streams, and user-approval flows.

[Live chat](https://ssuai.vercel.app/chat) ·
[Platform case study](https://seongju.vercel.app/en/projects/ssu-platform/) · [Documentation map](docs/README.md)

## Role in the platform

| Service | Responsibility | Repository |
| --- | --- | --- |
| ssuAI | User interface, same-origin BFF, authentication state, and SSE/HITL UX | [ghdtjdwn/ssuAI](https://github.com/ghdtjdwn/ssuAI) |
| **ssuAgent** | **Intent routing, tool composition, conversation checkpoints, and HITL orchestration** | This repository |
| ssuMCP | Campus domain logic, MCP/REST contracts, authentication, and state changes | [ghdtjdwn/ssuMCP](https://github.com/ghdtjdwn/ssuMCP) |
| ssu-ai-service | Isolated embedding-request gateway | [ghdtjdwn/ssu-ai-service](https://github.com/ghdtjdwn/ssu-ai-service) |

This service never calls university systems directly. It consumes domain capabilities only through
MCP. The ssuAI BFF verifies browser authentication and forwards only the minimum principal and MCP
session context.

## Architecture

![ssuAgent orchestration architecture showing trust boundaries, LangGraph routing, checkpoints, MCP, and LLM fallback](docs/assets/architecture.svg)

```text
browser
  → ssuAI same-origin agent proxy
  → FastAPI stream/resume boundary
  → LangGraph supervisor
  → academic · library · LMS specialist
  → ssuMCP tools over Streamable HTTP
```

- The supervisor routes the current user turn to a domain specialist; each specialist loads only
  the MCP tools it needs.
- A LangGraph PostgreSQL checkpointer persists conversations and interrupts. The verified principal
  hash is bound to `thread_id`, preventing another user from reading or resuming the checkpoint.
- Library writes interrupt after `prepare_*` and resume with `confirm_action` only after explicit
  approval in ssuAI.
- Only configured providers join the Anthropic → Groq → Gemini → OpenRouter sequence. Each agent's
  manual `bind_tools` loop controls provider-specific tool-call differences and fallback. Groq uses
  `ChatGroq` instead of the generic `ChatOpenAI` wrapper for tool-call content compatibility.
  Pricing, model availability, and organization quotas are external runtime constraints; this order
  does not claim a cost or accuracy advantage.

See the [architecture document](docs/architecture.md) for request, trust, and state boundaries and
the current single-replica constraint.

## Engineering evidence

| Problem | Implementation and verification |
| --- | --- |
| Another user resuming an existing conversation | Stable principal hash and thread-owner binding — [ADR 0010](docs/adr/0010-agent-thread-ownership-binding.md) · [security tests](tests/test_main_security.py) |
| Broken resume semantics after stream reconnect or interrupt | Stable thread and explicit `resume` event ordering — [stream contract tests](tests/test_stream_interrupt.py) |
| A clear LMS export timing out in another LLM turn | Conservative direct routing and deterministic link generation — [ADR 0022](docs/adr/0022-deterministic-lms-export-download.md) · [LMS tests](tests/test_lms_agent.py) |
| Provider-specific tool-call formats and cascading failures | Configuration-driven provider sequence and agent-local fallback — [factory tests](tests/test_llm_factory.py) · [ADR 0004](docs/adr/004-multi-provider-llm-fallback.md) |
| Handoffs duplicating answers or choosing the wrong tool | Routing and safety evaluation sets — [routing eval](tests/test_eval_routing.py) · [safety eval](tests/test_eval_safety.py) |
| Deployment of an unverified image | ARM64 image publication follows Ruff, formatting, and pytest — [CI workflow](.github/workflows/ci.yml) · [deployment guide](docs/deploy.md) |

The main stack is Python 3.12, FastAPI, LangGraph, LangChain, PostgreSQL checkpointer, MCP
Streamable HTTP, SSE, uv, Ruff, pytest, Docker, Helm, and ArgoCD.

## Local development and verification

PostgreSQL and at least one LLM provider key are required. Keep real values in `.env` and never
commit them.

```bash
git clone https://github.com/ghdtjdwn/ssuAgent.git
cd ssuAgent
cp .env.example .env
set -a && source .env && set +a

uv sync --extra dev
uv run uvicorn ssu_agent.main:app --host 0.0.0.0 --port 8000
```

```bash
uv run ruff check .
uv run ruff format --check .
uv run pytest
uv run pytest tests/test_eval_routing.py
```

See the [configuration guide](docs/configuration.md) for each variable's security meaning and the
difference between local and production settings.

The versioned routing corpus contains nine prompts: six domain handoffs and
three direct answers, classified against four failure types. The test uses a fake chat model while
executing the real routing tools, markers, parser, and graph destinations after tool selection. It
is not a live-model tool-selection accuracy result. See [`evals/README.md`](evals/README.md) for the
evidence boundary.

## Documentation

- [Documentation map](docs/README.md)
- [Architecture and trust boundaries](docs/architecture.md) (Korean)
- [Configuration and environment variables](docs/configuration.md) (Korean)
- [GitOps deployment and operational verification](docs/deploy.md) (Korean)
- [Architecture decision records](docs/adr/) (Korean)

## Scope and limitations

- Production currently runs one replica and inbound rate limiting is process-local. A shared limiter
  and checkpoint concurrency verification are required before scaling out.
- CI stream/HITL unit tests use `MemorySaver`; the current gate does not include a container-backed
  restart-and-resume test against real PostgreSQL.
- The service depends on LLM providers and university systems. On tool failure, it reports the
  limitation or required connection instead of guessing.
- Production `/agent/*` endpoints require a server-to-server API key. Public users access them
  through ssuAI.

## License

[MIT](LICENSE)
