"""Tests for the FastAPI app (offline, fake LLM via dependency override)."""

from __future__ import annotations

import math
import re
import zlib
from io import BytesIO

import pytest
from fastapi.testclient import TestClient
from pypdf import PdfWriter

from multi_agent_resume_screener.agents.critic import _CriticLLMOutput
from multi_agent_resume_screener.api.main import app, get_embedder, get_llm, get_store
from multi_agent_resume_screener.state import (
    MatchResult,
    StructuredJD,
    StructuredResume,
    SubScore,
)
from multi_agent_resume_screener.storage.runs import RunStore

_TOKEN_RE = re.compile(r"[a-z0-9]+")


class _FakeEmbedder:
    """Deterministic bag-of-words embedder for offline tests (see
    tests/test_retrieval.py for the retrieval-specific tests of this fake)."""

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vectorize(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vectorize(text)

    @staticmethod
    def _vectorize(text: str, dim: int = 64) -> list[float]:
        vec = [0.0] * dim
        for tok in _TOKEN_RE.findall(text.lower()):
            vec[zlib.crc32(tok.encode("utf-8")) % dim] += 1.0
        norm = math.sqrt(sum(v * v for v in vec))
        return [v / norm for v in vec] if norm else vec


class _RoutingRunnable:
    def __init__(self, schema, results):
        self._schema = schema
        self._results = results

    def invoke(self, messages):
        return self._results[self._schema.__name__]


class _RoutingFakeLLM:
    def __init__(self, results: dict):
        self.results = results

    def with_structured_output(self, schema, **kwargs):
        return _RoutingRunnable(schema, self.results)


def _fake_llm() -> _RoutingFakeLLM:
    return _RoutingFakeLLM(
        {
            "StructuredResume": StructuredResume(name="Cand", skills=["python"]),
            "StructuredJD": StructuredJD(title="Backend", required_skills=["python"]),
            "MatchResult": MatchResult(
                sub_scores=[SubScore(section="skills", score=0.8)],
                overall_evidence_quality=0.9,
            ),
            "_CriticLLMOutput": _CriticLLMOutput(
                gaps=[], suggestions=["Add a link"], verdict="moderate_fit",
                confidence_in_scoring=0.95,
            ),
        }
    )


def _blank_pdf_bytes() -> bytes:
    writer = PdfWriter()
    writer.add_blank_page(width=200, height=200)
    buf = BytesIO()
    writer.write(buf)
    return buf.getvalue()


@pytest.fixture
def client(tmp_path):
    store = RunStore(tmp_path / "test_runs.db")
    app.dependency_overrides[get_llm] = _fake_llm
    app.dependency_overrides[get_embedder] = _FakeEmbedder
    app.dependency_overrides[get_store] = lambda: store
    yield TestClient(app)
    app.dependency_overrides.clear()


def test_health():
    c = TestClient(app)
    resp = c.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_root_serves_frontend():
    c = TestClient(app)
    resp = c.get("/")
    assert resp.status_code == 200
    assert "text/html" in resp.headers["content-type"]
    assert "multi-agent-resume-screener" in resp.text


def test_static_assets_served():
    c = TestClient(app)
    css = c.get("/static/style.css")
    assert css.status_code == 200
    js = c.get("/static/app.js")
    assert js.status_code == 200


def test_screen_recruiter_ranks_candidates(client):
    pdf = _blank_pdf_bytes()
    files = [
        ("resumes", ("a.pdf", pdf, "application/pdf")),
        ("resumes", ("b.pdf", pdf, "application/pdf")),
    ]
    resp = client.post(
        "/screen",
        data={"jd": "Backend engineer, Python required", "mode": "recruiter"},
        files=files,
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["mode"] == "recruiter"
    assert body["job_title"] == "Backend"
    assert len(body["candidates"]) == 2
    scores = [c["score"] for c in body["candidates"]]
    assert scores == sorted(scores, reverse=True)


def test_screen_candidate_mode(client):
    files = [("resumes", ("a.pdf", _blank_pdf_bytes(), "application/pdf"))]
    resp = client.post(
        "/screen",
        data={"jd": "Backend engineer", "mode": "candidate"},
        files=files,
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["mode"] == "candidate"
    assert len(body["candidates"]) == 1
    assert "suggestions" in body["candidates"][0]


def test_screen_empty_jd_rejected(client):
    files = [("resumes", ("a.pdf", _blank_pdf_bytes(), "application/pdf"))]
    resp = client.post("/screen", data={"jd": "   "}, files=files)
    assert resp.status_code == 422


def test_screen_invalid_mode_rejected(client):
    files = [("resumes", ("a.pdf", _blank_pdf_bytes(), "application/pdf"))]
    resp = client.post(
        "/screen",
        data={"jd": "Backend", "mode": "not_a_mode"},
        files=files,
    )
    assert resp.status_code == 422


def test_screen_requires_resume_file(client):
    resp = client.post("/screen", data={"jd": "Backend"})
    # FastAPI returns 422 when the required file field is missing.
    assert resp.status_code == 422


def test_screen_persists_run_and_is_retrievable(client):
    files = [("resumes", ("a.pdf", _blank_pdf_bytes(), "application/pdf"))]
    resp = client.post(
        "/screen", data={"jd": "Backend engineer", "mode": "recruiter"}, files=files
    )
    assert resp.status_code == 200
    run_id = resp.json()["run_id"]
    assert run_id

    # The run is retrievable via GET /runs/{id}.
    fetched = client.get(f"/runs/{run_id}")
    assert fetched.status_code == 200
    assert fetched.json()["run_id"] == run_id

    # And it shows up in the recent-runs listing.
    listing = client.get("/runs")
    assert listing.status_code == 200
    assert any(r["id"] == run_id for r in listing.json()["runs"])


def test_get_unknown_run_returns_404(client):
    resp = client.get("/runs/nonexistent-id")
    assert resp.status_code == 404


def test_screen_rejects_large_upload_before_pipeline(client):
    resp = client.post(
        "/screen",
        data={"jd": "Backend"},
        files=[("resumes", ("large.pdf", b"x" * 4_000_001, "application/pdf"))],
    )
    assert resp.status_code == 413
    assert client.get("/runs").json() == {"runs": []}


def test_screen_rejects_aggregate_upload_size(client):
    # Padding after EOF preserves this valid PDF while testing the batch limit.
    pdf = _blank_pdf_bytes().ljust(2_000_001, b" ")
    resp = client.post(
        "/screen",
        data={"jd": "Backend"},
        files=[("resumes", (name, pdf, "application/pdf")) for name in ("a.pdf", "b.pdf")],
    )
    assert resp.status_code == 413


def test_screen_rejects_too_many_resumes(client):
    resp = client.post(
        "/screen",
        data={"jd": "Backend"},
        files=[
            ("resumes", (f"{i}.pdf", _blank_pdf_bytes(), "application/pdf"))
            for i in range(11)
        ],
    )
    assert resp.status_code == 422


def test_screen_rejects_long_jd(client):
    resp = client.post(
        "/screen",
        data={"jd": "x" * 20_001},
        files=[("resumes", ("a.pdf", _blank_pdf_bytes(), "application/pdf"))],
    )
    assert resp.status_code == 422


@pytest.mark.parametrize("limit", [-1, 0, 101])
def test_runs_limit_is_bounded(client, limit):
    assert client.get("/runs", params={"limit": limit}).status_code == 422


def test_database_errors_do_not_expose_connection_details(client):
    from psycopg import OperationalError

    class UnavailableStore:
        def list_runs(self, limit):
            raise OperationalError("sensitive connection details")

    app.dependency_overrides[get_store] = UnavailableStore
    response = client.get("/runs")
    assert response.status_code == 503
    assert "sensitive" not in response.text
