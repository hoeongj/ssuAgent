import os

# Strip whitespace/CRLF from env vars — secrets copied on Windows can carry
# a trailing \r, which produces an illegal HTTP Authorization header and
# causes httpcore.LocalProtocolError on every request to Groq/OpenRouter.
SSUMCP_URL: str = os.getenv("SSUMCP_URL", "https://ssumcp.duckdns.org/mcp").strip()
GOOGLE_API_KEY: str = os.getenv("GOOGLE_API_KEY", "").strip()
GEMINI_MODEL: str = os.getenv("GEMINI_MODEL", "gemini-2.5-flash").strip()
GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "").strip()
OPENROUTER_API_KEY: str = os.getenv("OPENROUTER_API_KEY", "").strip()
DATABASE_URL: str = os.getenv(
    "DATABASE_URL",
    "postgresql://ssuai:dev@localhost:5432/ssuai",
).strip()

# CORS allow-list. Comma-separated origins; a lone "*" means allow all.
# Default "*" preserves the previous wide-open behavior until configured.
ALLOWED_ORIGINS: list[str] = [
    origin.strip()
    for origin in os.getenv("ALLOWED_ORIGINS", "*").strip().split(",")
    if origin.strip()
]

# Optional API key gate for /agent endpoints. When empty (default), the gate
# is a no-op so existing prod behavior is preserved; when set, requests must
# send a matching X-Agent-Key header.
AGENT_API_KEY: str = os.getenv("AGENT_API_KEY", "").strip()

# Per-IP inbound rate limit for /agent/* (slowapi syntax, e.g. "30/minute").
# Mirrors ssuMCP ADR 0061: these endpoints fan out to paid LLM providers, so an
# unauthenticated request flood is a cost-exhaustion / DoS vector.
AGENT_RATE_LIMIT: str = os.getenv("AGENT_RATE_LIMIT", "30/minute").strip()

# Max characters accepted in a single agent message (oversized-payload guard).
AGENT_MAX_MESSAGE_CHARS: int = int(os.getenv("AGENT_MAX_MESSAGE_CHARS", "8000"))
