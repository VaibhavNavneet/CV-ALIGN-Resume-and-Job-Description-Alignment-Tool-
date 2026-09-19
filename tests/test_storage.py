"""Tests for the SQLite run store."""

from __future__ import annotations

import pytest

from multi_agent_resume_screener.state import CandidateResult, ScreeningResult
from multi_agent_resume_screener.storage.runs import RunStore


def _result(mode="recruiter") -> ScreeningResult:
    return ScreeningResult(
        mode=mode,
        job_title="Backend",
        candidates=[
            CandidateResult(filename="a.pdf", score=0.8, verdict="strong_fit"),
            CandidateResult(filename="b.pdf", score=0.4, verdict="weak_fit"),
        ],
    )


def test_save_assigns_run_id_and_created_at(tmp_path):
    store = RunStore(tmp_path / "runs.db")
    result = _result()
    assert result.run_id is None

    run_id = store.save(result)
    assert run_id
    assert result.run_id == run_id
    assert result.created_at is not None


def test_get_round_trips(tmp_path):
    store = RunStore(tmp_path / "runs.db")
    result = _result()
    run_id = store.save(result)

    fetched = store.get(run_id)
    assert fetched is not None
    assert fetched.run_id == run_id
    assert fetched.job_title == "Backend"
    assert len(fetched.candidates) == 2
    assert fetched.candidates[0].filename == "a.pdf"


def test_get_unknown_returns_none(tmp_path):
    store = RunStore(tmp_path / "runs.db")
    assert store.get("does-not-exist") is None


def test_list_runs_newest_first(tmp_path):
    store = RunStore(tmp_path / "runs.db")
    id1 = store.save(ScreeningResult(mode="recruiter", created_at="2026-01-01T00:00:00",
                                     job_title="A"))
    id2 = store.save(ScreeningResult(mode="candidate", created_at="2026-02-01T00:00:00",
                                     job_title="B"))

    runs = store.list_runs()
    assert [r["id"] for r in runs] == [id2, id1]
    assert runs[0]["job_title"] == "B"


def test_persists_across_store_instances(tmp_path):
    db = tmp_path / "runs.db"
    run_id = RunStore(db).save(_result())
    # A new store pointing at the same file can read the run back.
    assert RunStore(db).get(run_id) is not None


def test_memory_path_rejected():
    with pytest.raises(ValueError):
        RunStore(":memory:")


def test_save_existing_run_updates_without_duplicates(tmp_path):
    store = RunStore(tmp_path / "runs.db")
    result = _result()
    run_id = store.save(result)
    result.job_title = "Updated title"
    assert store.save(result) == run_id
    assert len(store.list_runs()) == 1
    assert store.get(run_id).job_title == "Updated title"


def test_postgres_uses_bound_parameters_and_closes_connection(monkeypatch):
    from unittest.mock import MagicMock

    import psycopg

    from multi_agent_resume_screener.storage.runs import PostgresRunStore

    connection = MagicMock()
    connection.__enter__.return_value = connection
    connect = MagicMock(return_value=connection)
    monkeypatch.setattr(psycopg, "connect", connect)
    store = PostgresRunStore("postgresql://example.invalid/test")
    connect.assert_not_called()  # no connections or DDL on cold start
    result = _result()
    result.job_title = "O'Reilly; DROP TABLE runs"
    store.save(result)
    sql, params = connection.execute.call_args.args
    assert sql.count("%s") == 6
    assert result.job_title not in sql
    assert params[3] == result.job_title
    assert connect.call_args.kwargs["prepare_threshold"] is None
    connection.__exit__.assert_called_once()

    connection.execute.return_value.fetchone.return_value = {
        "result_json": result.model_dump_json()
    }
    assert store.get(result.run_id) == result
    connection.execute.return_value.fetchall.return_value = [{"id": result.run_id}]
    assert store.list_runs(5) == [{"id": result.run_id}]
    assert connection.execute.call_args.args[1] == (5,)
