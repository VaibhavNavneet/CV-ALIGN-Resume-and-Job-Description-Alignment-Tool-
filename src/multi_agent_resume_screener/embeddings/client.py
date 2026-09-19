"""Embedding client factory for resume-evidence retrieval.

Retrieval always embeds with Gemini's embedding API, independent of
``LLM_PROVIDER`` (Groq, the other supported chat provider, has no embeddings
API). ``GOOGLE_API_KEY`` is therefore required for retrieval even when the
chat model itself runs on Groq.
"""

from __future__ import annotations

from langchain_core.embeddings import Embeddings

from multi_agent_resume_screener.settings import Settings, get_settings

# gemini-embedding-001: current stable, GA, free-tier Gemini text-embedding
# model. text-embedding-004, referenced by older tutorials, was deprecated
# and shut down by Google on 2026-01-14 -- do not use it.
DEFAULT_EMBEDDING_MODEL = "gemini-embedding-001"


class EmbeddingConfigError(RuntimeError):
    """Raised when the embedding provider is misconfigured (e.g. missing API key)."""


def get_embedder(settings: Settings | None = None) -> Embeddings:
    """Build the Gemini embeddings client used for resume-evidence retrieval.

    Args:
        settings: Optional settings override (useful in tests). Falls back to
            the cached application settings.

    Returns:
        A LangChain ``Embeddings`` instance ready to ``.embed_documents()`` /
        ``.embed_query()``.

    Raises:
        EmbeddingConfigError: If ``GOOGLE_API_KEY`` is not set.
    """
    settings = settings or get_settings()
    if not settings.google_api_key:
        raise EmbeddingConfigError(
            "GOOGLE_API_KEY is not set. Evidence retrieval uses Gemini's "
            "free-tier embedding API regardless of LLM_PROVIDER; get a free "
            "key at https://aistudio.google.com/apikey and add it to your "
            ".env file."
        )
    # Imported lazily, mirroring llm/client.py's provider imports.
    from langchain_google_genai import GoogleGenerativeAIEmbeddings

    model = settings.embedding_model or DEFAULT_EMBEDDING_MODEL
    return GoogleGenerativeAIEmbeddings(model=model, google_api_key=settings.google_api_key)
