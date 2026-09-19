"""SQLite (local) and PostgreSQL (Vercel) persistence for screening runs.

Uses ``sqlite3`` locally and ``psycopg`` for PostgreSQL. Each screening run
is stored as a row with its full JSON payload, so a run can be retrieved later
via ``GET /runs/{id}`` and recent runs can be listed. A fresh connection is
opened per operation, which keeps the store safe to call from FastAPI's worker
threads.
"""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from multi_agent_resume_screener.state import ScreeningResult

_SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    id          TEXT PRIMARY KEY,
    created_at  TEXT NOT NULL,
    mode        TEXT NOT NULL,
    job_title   TEXT,
    n_candidates INTEGER NOT NULL,
    result_json TEXT NOT NULL
);
"""


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class RunStore:
    """A small SQLite-backed store for screening runs."""

    _placeholder = "?"

    def __init__(self, db_path: str | Path = "runs.db") -> None:
        self.db_path = str(db_path)
        self._init_db()

    @contextmanager
    def _connect(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            with conn:
                yield conn
        finally:
            conn.close()

    def _init_db(self) -> None:
        # ":memory:" databases do not persist across connections, so creating a
        # fresh one per operation would lose data; guard against that footgun.
        if self.db_path == ":memory:":
            raise ValueError(
                "RunStore needs a file path; ':memory:' is not supported "
                "because each operation opens a new connection."
            )
        with self._connect() as conn:
            conn.execute(_SCHEMA)

    def initialize(self) -> None:
        """Create the schema explicitly before serving production traffic."""
        self._init_db()

    def save(self, result: ScreeningResult) -> str:
        """Persist a screening result and return its run id.

        Mutates ``result`` in place to set ``run_id`` and ``created_at`` if they
        are not already populated.
        """
        run_id = result.run_id or uuid4().hex
        created_at = result.created_at or _now_iso()
        result.run_id = run_id
        result.created_at = created_at

        with self._connect() as conn:
            conn.execute(
                "INSERT INTO runs "
                "(id, created_at, mode, job_title, n_candidates, result_json) "
                f"VALUES ({', '.join([self._placeholder] * 6)}) "
                "ON CONFLICT (id) DO UPDATE SET "
                "created_at = excluded.created_at, mode = excluded.mode, "
                "job_title = excluded.job_title, "
                "n_candidates = excluded.n_candidates, "
                "result_json = excluded.result_json",
                (
                    run_id,
                    created_at,
                    result.mode,
                    result.job_title,
                    len(result.candidates),
                    result.model_dump_json(),
                ),
            )
        return run_id

    def get(self, run_id: str) -> ScreeningResult | None:
        """Return a stored run by id, or ``None`` if not found."""
        with self._connect() as conn:
            row = conn.execute(
                f"SELECT result_json FROM runs WHERE id = {self._placeholder}",
                (run_id,),
            ).fetchone()
        if row is None:
            return None
        return ScreeningResult.model_validate_json(row["result_json"])

    def list_runs(self, limit: int = 50) -> list[dict]:
        """Return summaries of recent runs, newest first."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT id, created_at, mode, job_title, n_candidates "
                f"FROM runs ORDER BY created_at DESC LIMIT {self._placeholder}",
                (limit,),
            ).fetchall()
        return [dict(row) for row in rows]


class PostgresRunStore(RunStore):
    """Durable serverless storage, with one short-lived connection per operation.

    Use the database provider's pooled connection URL. Schema creation is an
    explicit deployment step, not a cold-start side effect.
    """

    _placeholder = "%s"

    def __init__(self, database_url: str) -> None:
        self.database_url = database_url

    @contextmanager
    def _connect(self):
        import psycopg
        from psycopg.rows import dict_row

        with psycopg.connect(
            self.database_url,
            row_factory=dict_row,
            connect_timeout=10,
            options="-c statement_timeout=15000",
            prepare_threshold=None,
        ) as conn:
            yield conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(_SCHEMA)
