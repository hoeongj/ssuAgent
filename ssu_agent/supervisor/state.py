"""
SsuAgentState — LangGraph multi-agent shared state.

Design decision (ADR): single shared TypedDict across supervisor + sub-agents.
- messages: Annotated[list, add_messages] is the merge channel (all agents append here).
- active_agent: set by supervisor when routing, cleared when sub-agent finishes.
- pending_action: private to Library agent; holds prepare_* result awaiting HITL approval.
- mcp_session_id: passed from the FastAPI client, threaded to all private MCP tool calls.

Why a single state rather than per-agent TypedDicts:
  LangGraph subgraphs that share a parent state use channel-level merging via reducers.
  For this project, only `messages` needs cross-agent merging (add_messages is the reducer).
  Other fields are updated by exactly one owner, so a plain override is correct.
  A separate per-agent TypedDict would require explicit input/output transforms at every
  subgraph boundary — unnecessary complexity at this scale.
"""

from __future__ import annotations

from typing import Annotated

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict


class SsuAgentState(TypedDict):
    # ── Conversation ──────────────────────────────────────────────────────────
    messages: Annotated[list[BaseMessage], add_messages]

    # ── Session binding ───────────────────────────────────────────────────────
    # Lifecycle: FastAPI thread_id (LangGraph) ↔ mcp_session_id (ssuMCP auth).
    # The client passes mcp_session_id on every request so the agent can pass
    # it to private MCP tools (library reservation, SAINT, LMS) as a parameter.
    mcp_session_id: str | None

    # ── Routing ───────────────────────────────────────────────────────────────
    # Set by supervisor before routing; cleared by sub-agent on return.
    # Used to detect re-entry to supervisor after sub-agent completion.
    active_agent: str | None

    # ── HITL (Library write actions) ──────────────────────────────────────────
    # Library agent populates this when a prepare_* tool returns an action_id.
    # interrupt() pauses the graph; FastAPI streams the payload to the client.
    # Resume path: client sends /agent/resume with {approved: bool, action_id: int}.
    pending_action: dict | None
