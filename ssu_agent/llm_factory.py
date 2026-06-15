"""
LLM factory with multi-provider fallback.

Provider priority (Groq first — higher free-tier quota):
  1. Groq Llama-3.3-70b  — 14,400 req/day free, very fast
  2. Google Gemini        — 20 req/day free tier, high quality
  3. OpenRouter           — catch-all aggregator, many free models

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

    if config.GROQ_API_KEY:
        from langchain_groq import ChatGroq

        llms.append(
            ChatGroq(
                model="llama-3.3-70b-versatile",
                api_key=config.GROQ_API_KEY,
                max_retries=1,
            )
        )

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
    return get_llm_sequence()[0]
