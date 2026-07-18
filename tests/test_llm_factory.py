"""
Tests for the LLM factory provider guarding.

Each provider must be gated by its own API key so an unset key never adds a
client that crashes at call time (the Gemini bug this fixes). config attributes
are monkeypatched so no real key/network is needed; client construction with a
dummy key is offline (langchain defers the network call to invoke time).
"""

from __future__ import annotations

import pytest

from ssu_agent import config, llm_factory


def _clear_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(config, "ANTHROPIC_API_KEY", "")
    monkeypatch.setattr(config, "GROQ_API_KEY", "")
    monkeypatch.setattr(config, "GOOGLE_API_KEY", "")
    monkeypatch.setattr(config, "OPENROUTER_API_KEY", "")


def test_no_keys_returns_empty_sequence(monkeypatch: pytest.MonkeyPatch):
    _clear_keys(monkeypatch)
    assert llm_factory.get_llm_sequence() == []


def test_create_llm_raises_runtime_error_when_no_keys(monkeypatch: pytest.MonkeyPatch):
    _clear_keys(monkeypatch)
    with pytest.raises(RuntimeError, match="No LLM provider API key configured"):
        llm_factory.create_llm()


def test_only_gemini_key_yields_gemini_model(monkeypatch: pytest.MonkeyPatch):
    from langchain_google_genai import ChatGoogleGenerativeAI

    _clear_keys(monkeypatch)
    monkeypatch.setattr(config, "GOOGLE_API_KEY", "test-google-key")

    sequence = llm_factory.get_llm_sequence()
    assert len(sequence) == 1
    assert isinstance(sequence[0], ChatGoogleGenerativeAI)
    # create_llm now succeeds because the sequence is non-empty.
    assert isinstance(llm_factory.create_llm(), ChatGoogleGenerativeAI)


def test_gemini_not_added_without_key(monkeypatch: pytest.MonkeyPatch):
    """Only Groq configured → sequence has exactly the Groq client."""
    from langchain_groq import ChatGroq

    _clear_keys(monkeypatch)
    monkeypatch.setattr(config, "GROQ_API_KEY", "test-groq-key")

    sequence = llm_factory.get_llm_sequence()
    assert len(sequence) == 1
    assert isinstance(sequence[0], ChatGroq)


def test_anthropic_key_prepends_claude_before_groq(monkeypatch: pytest.MonkeyPatch):
    from langchain_anthropic import ChatAnthropic
    from langchain_groq import ChatGroq

    _clear_keys(monkeypatch)
    monkeypatch.setattr(config, "ANTHROPIC_API_KEY", "test-anthropic-key")
    monkeypatch.setattr(config, "ANTHROPIC_MODEL", "claude-haiku-4-5")
    monkeypatch.setattr(config, "GROQ_API_KEY", "test-groq-key")

    sequence = llm_factory.get_llm_sequence()
    assert [type(llm) for llm in sequence] == [ChatAnthropic, ChatGroq]
    assert hasattr(sequence[0], "bind_tools")


def test_all_keys_yield_groq_gemini_openrouter_in_order(monkeypatch: pytest.MonkeyPatch):
    """All three keys set → fallback priority is exactly Groq → Gemini → OpenRouter.

    Guards the explicit operational provider order without assuming a stable
    external quota; the OpenRouter client points at the OpenRouter base_url.
    """
    from langchain_google_genai import ChatGoogleGenerativeAI
    from langchain_groq import ChatGroq
    from langchain_openai import ChatOpenAI

    _clear_keys(monkeypatch)
    monkeypatch.setattr(config, "GROQ_API_KEY", "test-groq-key")
    monkeypatch.setattr(config, "GOOGLE_API_KEY", "test-google-key")
    monkeypatch.setattr(config, "OPENROUTER_API_KEY", "test-openrouter-key")

    sequence = llm_factory.get_llm_sequence()
    assert [type(llm) for llm in sequence] == [ChatGroq, ChatGoogleGenerativeAI, ChatOpenAI]
