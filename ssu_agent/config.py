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
