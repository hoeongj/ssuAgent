"""Tests for the optional local-RAG agent tool (SSUAGENT_LOCAL_RAG)."""

from __future__ import annotations

from ssu_agent import config
from ssu_agent.rag import tool as rag_tool


def test_disabled_by_default(monkeypatch):
    monkeypatch.setattr(config, "LOCAL_RAG_ENABLED", False)
    assert rag_tool.build_local_rag_tools() == []


def test_enabled_adds_named_tool(monkeypatch):
    # No OPENAI_API_KEY in CI -> engine builds with MockEmbedding (offline-safe).
    monkeypatch.setattr(config, "LOCAL_RAG_ENABLED", True)
    tools = rag_tool.build_local_rag_tools()
    assert len(tools) == 1
    assert tools[0].name == "search_local_academic_rag"


def test_enabled_tool_returns_text(monkeypatch):
    monkeypatch.setattr(config, "LOCAL_RAG_ENABLED", True)
    tool = rag_tool.build_local_rag_tools()[0]
    out = tool.invoke({"question": "졸업 학점 기준"})
    assert isinstance(out, str) and out
