"""
LLM factory with multi-provider fallback.

Provider priority:
  0. Anthropic Claude Haiku 4.5 — used first when ANTHROPIC_API_KEY is set
     (paid; temporary dev/testing). Falls back to the configured providers below on
     error or when the key is unset.
  1. Groq Llama-3.3-70b
  2. Google Gemini
  3. OpenRouter

Provider pricing, model availability, and organization-level quotas change
outside this repository. They are runtime constraints, not part of the tested
fallback-order contract.

NOTE: Use ChatGroq (not ChatOpenAI with Groq base_url) for Groq — the
generic ChatOpenAI wrapper serializes assistant content as a list of
content blocks, which Groq's API rejects with a 400 on the second tool
call turn. ChatGroq handles the string-content conversion internally.

NOTE: langchain_core 1.4.x RunnableWithFallbacks lacks bind_tools, so
with_fallbacks() breaks when create_react_agent calls model.bind_tools()
internally. Use get_llm_sequence() + per-agent retry loops instead.
"""

from __future__ import annotations

from langchain_core.language_models import BaseChatModel

from ssu_agent import config


def get_llm_sequence() -> list[BaseChatModel]:
    """Return LLMs in priority order for per-request fallback loops."""
    llms: list[BaseChatModel] = []

    if config.ANTHROPIC_API_KEY:
        from langchain_anthropic import ChatAnthropic

        llms.append(
            ChatAnthropic(
                model=config.ANTHROPIC_MODEL,
                api_key=config.ANTHROPIC_API_KEY,
                max_tokens=2048,
                # The low-tier paid key 429s under normal prod load; the SDK
                # honors Retry-After between retries, so a few retries here
                # ride out a transient rate limit instead of immediately
                # falling through to the free provider chain.
                max_retries=3,
            )
        )

    if config.GROQ_API_KEY:
        from langchain_groq import ChatGroq

        llms.append(
            ChatGroq(
                model="llama-3.3-70b-versatile",
                api_key=config.GROQ_API_KEY,
                max_retries=1,
            )
        )

    if config.GOOGLE_API_KEY:
        from langchain_google_genai import ChatGoogleGenerativeAI

        llms.append(
            ChatGoogleGenerativeAI(
                model=config.GEMINI_MODEL,
                google_api_key=config.GOOGLE_API_KEY,
                max_retries=1,
            )
        )

    if config.OPENROUTER_API_KEY:
        from langchain_openai import ChatOpenAI

        llms.append(
            ChatOpenAI(
                model="meta-llama/llama-3.3-70b-instruct:free",
                base_url="https://openrouter.ai/api/v1",
                api_key=config.OPENROUTER_API_KEY,
                max_retries=1,
            )
        )

    return llms


def create_llm() -> BaseChatModel:
    """Return the highest-priority available LLM (for static agent builds)."""
    sequence = get_llm_sequence()
    if not sequence:
        raise RuntimeError(
            "No LLM provider API key configured "
            "(set GROQ_API_KEY / GOOGLE_API_KEY / OPENROUTER_API_KEY)"
        )
    return sequence[0]
