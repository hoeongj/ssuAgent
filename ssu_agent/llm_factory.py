"""
LLM factory with multi-provider fallback.

Provider priority (each falls back to the next on any exception):
  1. Google Gemini  — primary, best quality for Korean
  2. Groq Llama     — free tier, very fast, OpenAI-compatible
  3. OpenRouter     — catch-all aggregator, many free models

max_retries=1 on each provider so quota exhaustion propagates quickly
to the next fallback instead of blocking for tenacity back-off.
"""

from __future__ import annotations

from langchain_core.language_models import BaseChatModel

from ssu_agent import config


def create_llm() -> BaseChatModel:
    """Return primary LLM with automatic fallback to Groq then OpenRouter."""
    from langchain_google_genai import ChatGoogleGenerativeAI

    primary = ChatGoogleGenerativeAI(
        model=config.GEMINI_MODEL,
        google_api_key=config.GOOGLE_API_KEY,
        max_retries=1,
    )

    fallbacks: list[BaseChatModel] = []

    if config.GROQ_API_KEY:
        from langchain_openai import ChatOpenAI

        fallbacks.append(
            ChatOpenAI(
                model="llama-3.3-70b-versatile",
                base_url="https://api.groq.com/openai/v1",
                api_key=config.GROQ_API_KEY,
                max_retries=1,
            )
        )

    if config.OPENROUTER_API_KEY:
        from langchain_openai import ChatOpenAI

        fallbacks.append(
            ChatOpenAI(
                model="meta-llama/llama-3.3-70b-instruct:free",
                base_url="https://openrouter.ai/api/v1",
                api_key=config.OPENROUTER_API_KEY,
                max_retries=1,
            )
        )

    if fallbacks:
        return primary.with_fallbacks(fallbacks)
    return primary
