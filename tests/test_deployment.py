"""Vercel configuration and storage selection regressions."""

import importlib
import json
import tomllib
from pathlib import Path

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from multi_agent_resume_screener.api.main import _default_store
from multi_agent_resume_screener.settings import Settings
from multi_agent_resume_screener.storage.runs import PostgresRunStore, RunStore

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def configure_store(monkeypatch):
    _default_store.cache_clear()

    def configure(**kwargs):
        settings = Settings(_env_file=None, **kwargs)
        monkeypatch.setattr(
            "multi_agent_resume_screener.api.main.get_settings", lambda: settings
        )

    yield configure
    _default_store.cache_clear()


def test_vercel_requires_durable_storage(configure_store):
    configure_store(vercel=True, database_url=None)
    with pytest.raises(HTTPException) as error:
        _default_store()
    assert error.value.status_code == 503
    assert "DATABASE_URL" in error.value.detail


def test_vercel_uses_postgres_without_connecting_at_import(configure_store):
    configure_store(vercel=True, database_url="postgresql://example.invalid/test")
    assert isinstance(_default_store(), PostgresRunStore)


def test_local_storage_still_works(configure_store, tmp_path):
    configure_store(vercel=False, database_url=None, db_path=str(tmp_path / "runs.db"))
    assert isinstance(_default_store(), RunStore)
    assert _default_store().list_runs() == []


def test_vercel_entrypoint_and_frontend():
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    module_name, name = project["tool"]["vercel"]["entrypoint"].split(":")
    deployed_app = getattr(importlib.import_module(module_name), name)
    config = json.loads((ROOT / "vercel.json").read_text(encoding="utf-8"))
    assert config["functions"][f"{module_name}.py"]["maxDuration"] == 300
    with TestClient(deployed_app) as client:
        for path in ("/", "/health", "/static/style.css", "/static/app.js", "/docs"):
            assert client.get(path).status_code == 200
