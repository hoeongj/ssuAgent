import os

SSUMCP_URL: str = os.getenv("SSUMCP_URL", "https://ssumcp.duckdns.org/mcp")
GOOGLE_API_KEY: str = os.getenv("GOOGLE_API_KEY", "")
SQLITE_DB_PATH: str = os.getenv("SQLITE_DB_PATH", "ssu_agent_checkpoints.db")
