"""ASGI entrypoint for Vercel's native FastAPI runtime."""

from multi_agent_resume_screener.api.main import app

__all__ = ["app"]
