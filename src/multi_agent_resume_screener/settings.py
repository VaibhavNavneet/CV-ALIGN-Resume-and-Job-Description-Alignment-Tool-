"""Application configuration, loaded from environment / .env file.

All settings live here so there is a single source of truth. Secrets (API keys)
are read from the environment and never hard-coded. See `.env.example` for the
available variables.
"""

from __future__ import annotations

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Strongly-typed application settings sourced from the environment."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # --- LLM provider selection ---
    # Which provider to use: "gemini" (default) or "groq".
    llm_provider: str = Field(default="gemini")
    # Optional explicit model name. If unset, a per-provider default is used.
    llm_model: str | None = Field(default=None)

    # --- Embedding model (resume-evidence retrieval) ---
    # Retrieval always uses Gemini's embedding API regardless of LLM_PROVIDER
    # (Groq has no embeddings API), so GOOGLE_API_KEY is required for it.
    # Optional explicit model name; defaults to "gemini-embedding-001" if unset.
    embedding_model: str | None = Field(default=None)

    # --- Provider API keys ---
    google_api_key: str | None = Field(default=None)
    groq_api_key: str | None = Field(default=None)

    # --- Application ---
    app_env: str = Field(default="development")
    log_level: str = Field(default="INFO")
    db_path: str = Field(default="runs.db")
    database_url: str | None = Field(default=None, repr=False)
    vercel: bool = Field(default=False)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached Settings instance (the .env file is read only once)."""
    return Settings()
