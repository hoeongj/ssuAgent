import os

SSUMCP_URL: str = os.getenv("SSUMCP_URL", "https://ssumcp.duckdns.org/mcp")
GOOGLE_API_KEY: str = os.getenv("GOOGLE_API_KEY", "")
GEMINI_MODEL: str = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
DATABASE_URL: str = os.getenv(
    "DATABASE_URL",
    "postgresql://ssuai:dev@localhost:5432/ssuai",
)
