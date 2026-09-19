FROM python:3.12-slim

# Don't write .pyc files; flush stdout/stderr immediately.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8000 \
    DB_PATH=/tmp/runs.db

WORKDIR /app

# Install dependencies first for better layer caching.
# Both supported providers are included in the runtime dependencies.
COPY pyproject.toml requirements.txt README.md LICENSE ./
COPY src ./src
RUN pip install --no-cache-dir .

EXPOSE 8000

# Shell form so $PORT (set by most PaaS hosts, e.g. Render) is expanded.
CMD uvicorn multi_agent_resume_screener.api.main:app --host 0.0.0.0 --port ${PORT}
