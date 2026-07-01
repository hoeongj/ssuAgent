"""Optional local academic RAG, exposed to the academic agent as a tool.

OFF by default (`SSUAGENT_LOCAL_RAG`). In production the authoritative academic-policy
source is ssuMCP's server-side RAG (`search_academic_policy_sources`); this local
LlamaIndex pipeline runs over a small bundled fixture corpus and is supplementary.
It is kept off by default to avoid a split answer source (see ADR 0008).

Meaningful retrieval needs a real embedding model — set `OPENAI_API_KEY` — otherwise a
MockEmbedding (random vectors) is used and results are demo-only.
"""

from __future__ import annotations

import logging
import os

from langchain_core.tools import BaseTool, StructuredTool

from ssu_agent import config

log = logging.getLogger(__name__)


def _build_engine():
    from ssu_agent.rag.academic_rag import AcademicRagEngine
    from ssu_agent.rag.fixtures import ACADEMIC_FIXTURES

    embed_model = None
    if os.getenv("OPENAI_API_KEY", "").strip():
        try:
            from llama_index.embeddings.openai import OpenAIEmbedding

            embed_model = OpenAIEmbedding(model="text-embedding-3-small")
        except Exception:  # pragma: no cover - optional dependency
            log.warning("OpenAIEmbedding unavailable; falling back to MockEmbedding")
    return AcademicRagEngine.from_documents(ACADEMIC_FIXTURES, embed_model=embed_model)


def build_local_rag_tools() -> list[BaseTool]:
    """Return the local-RAG tool list — empty unless SSUAGENT_LOCAL_RAG is enabled."""
    if not config.LOCAL_RAG_ENABLED:
        return []
    try:
        engine = _build_engine()
    except Exception:
        log.exception("local academic RAG engine build failed; tool disabled")
        return []

    def search_local_academic_rag(question: str) -> str:
        """Search the bundled local academic-policy corpus (supplementary)."""
        result = engine.query(question)
        if not result.source_texts:
            return "로컬 학칙 코퍼스에서 관련 근거를 찾지 못했습니다."
        lines: list[str] = []
        if result.answer.strip():
            lines.append(result.answer.strip())
        for text, meta in zip(result.source_texts, result.source_metadata):
            source = meta.get("source", "출처 미상")
            lines.append(f"- ({source}) {text}")
        return "\n".join(lines)

    tool = StructuredTool.from_function(
        func=search_local_academic_rag,
        name="search_local_academic_rag",
        description=(
            "보조 로컬 학칙 RAG(번들 코퍼스). 공식·권위 있는 근거는 "
            "search_academic_policy_sources를 우선 사용하고, 이 도구는 로컬 파이프라인의 "
            "보완 검색용으로만 쓴다."
        ),
    )
    return [tool]
