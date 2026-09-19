"""Tests for the evidence-retrieval (RAG) layer (offline, using a fake embedder)."""

from __future__ import annotations

import math
import re
import zlib

import pytest

from multi_agent_resume_screener.embeddings.client import EmbeddingConfigError, get_embedder
from multi_agent_resume_screener.pipeline.retrieval import build_chunks, retrieve_evidence
from multi_agent_resume_screener.settings import Settings
from multi_agent_resume_screener.state import (
    EducationItem,
    ExperienceItem,
    ProjectItem,
    StructuredJD,
    StructuredResume,
)

_TOKEN_RE = re.compile(r"[a-z0-9]+")

# The production default (retrieval.py's _MIN_SIMILARITY, currently 0.80) is
# calibrated against real Gemini embedding scores and is meaningless against
# this fake's crude lexical hashing, which produces a much lower and
# differently-shaped score range. Tests that need to distinguish "a real
# match was found" from "nothing was found" pass this low threshold
# explicitly instead of silently inheriting the production-tuned default.
_FAKE_MIN_SIMILARITY = 0.05


class _FakeEmbedder:
    """Deterministic bag-of-words embedder for offline tests.

    Real retrieval uses Gemini's embedding API in production (see
    embeddings/client.py); this fake stands in for it in tests so the suite
    never needs network access or GOOGLE_API_KEY, while still producing
    vectors that reflect keyword overlap closely enough to prove the
    chunking/ranking logic.
    """

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        # Mirrors the real Gemini API, which rejects blank text with a 400
        # (EmbedContentRequest.content contains an empty Part) -- strict on
        # purpose so retrieval code that ever sends blank text fails loudly
        # here instead of only in production.
        for t in texts:
            if not t.strip():
                raise ValueError("embed_documents() received blank text")
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


def _resume_with_separable_facts() -> StructuredResume:
    """A synthetic resume with clearly separable facts for ranking tests."""
    return StructuredResume(
        name="Jordan Candidate",
        skills=["python", "fastapi", "sql", "postgresql"],
        experience=[
            ExperienceItem(
                company="Acme Web Co",
                role="Backend Engineer",
                dates="2022-2024",
                bullets=[
                    "Built REST APIs in Python using FastAPI",
                    "Optimized PostgreSQL queries for a SQL-heavy reporting service",
                ],
            ),
        ],
        projects=[
            ProjectItem(
                name="Riverside Bridge Load Analysis",
                description="Structural load analysis for a pedestrian bridge, "
                "civil engineering coursework project.",
                tech=["AutoCAD", "structural analysis"],
            ),
        ],
        education=[
            EducationItem(degree="B.Tech Civil Engineering", institute="State University", year="2021"),
        ],
    )


# --------------------------------------------------------------------------- #
# Chunking
# --------------------------------------------------------------------------- #
def test_build_chunks_one_per_item_plus_one_skills_chunk():
    resume = _resume_with_separable_facts()
    chunks = build_chunks(resume)

    # 1 experience + 1 project + 1 education + 1 skills = 4
    assert len(chunks) == 4
    sections = [c.section for c in chunks]
    assert sections.count("experience") == 1
    assert sections.count("projects") == 1
    assert sections.count("education") == 1
    assert sections.count("skills") == 1

    by_section = {c.section: c for c in chunks}
    assert by_section["experience"].title == "Backend Engineer at Acme Web Co"
    assert "FastAPI" in by_section["experience"].text
    assert by_section["projects"].title == "Riverside Bridge Load Analysis"
    assert by_section["education"].title == "State University"
    assert by_section["skills"].title is None
    assert "python" in by_section["skills"].text


def test_build_chunks_skips_fully_empty_items():
    resume = StructuredResume(
        experience=[ExperienceItem()],  # every field blank
        projects=[ProjectItem()],
        education=[EducationItem()],
    )
    chunks = build_chunks(resume)
    assert chunks == []


def test_build_chunks_no_skills_chunk_when_skills_empty():
    resume = StructuredResume(skills=[])
    chunks = build_chunks(resume)
    assert all(c.section != "skills" for c in chunks)


# --------------------------------------------------------------------------- #
# Embedder determinism
# --------------------------------------------------------------------------- #
def test_fake_embedder_vectorize_is_deterministic():
    embedder = _FakeEmbedder()
    a = embedder.embed_documents(["Python FastAPI backend"])[0]
    b = embedder.embed_documents(["Python FastAPI backend"])[0]
    assert a == b


def test_get_embedder_raises_without_api_key():
    settings = Settings(google_api_key=None, llm_provider="groq")
    with pytest.raises(EmbeddingConfigError):
        get_embedder(settings=settings)


# --------------------------------------------------------------------------- #
# Retrieval ranking
# --------------------------------------------------------------------------- #
def test_retrieval_ranks_relevant_chunk_highest():
    resume = _resume_with_separable_facts()
    chunks = build_chunks(resume)
    jd = StructuredJD(
        title="Backend Engineer",
        required_skills=["python", "fastapi"],
        responsibilities=["Build Python backend services"],
    )

    evidence = retrieve_evidence(chunks, jd, embedder=_FakeEmbedder(), top_k=4, min_similarity=_FAKE_MIN_SIMILARITY)

    assert evidence["experience"], "expected the Python/FastAPI experience chunk to be retrieved"
    top = evidence["experience"][0]
    assert "FastAPI" in top.chunk.text
    assert top.chunk.title == "Backend Engineer at Acme Web Co"


def test_retrieval_returns_empty_when_no_relevant_evidence():
    resume = _resume_with_separable_facts()
    chunks = build_chunks(resume)
    # This resume has no machine-learning content anywhere.
    jd = StructuredJD(
        title="ML Engineer",
        required_skills=["machine learning", "tensorflow"],
        responsibilities=["Train and deploy machine learning models"],
    )

    evidence = retrieve_evidence(chunks, jd, embedder=_FakeEmbedder(), top_k=4)

    assert evidence["skills"] == []


def test_retrieval_skips_section_with_empty_jd_query_without_crashing():
    # A JD with no `qualifications` produces an empty query string for the
    # education section. The real Gemini embeddings API rejects an empty
    # content Part with a 400 -- this exact combination (resume has an
    # education chunk, JD has no qualifications) crashed a live run before
    # retrieve_evidence() learned to skip sections with an empty query.
    resume = _resume_with_separable_facts()
    chunks = build_chunks(resume)
    jd = StructuredJD(required_skills=["python"], responsibilities=["Python backend work"])
    assert jd.qualifications == []  # the condition that triggers an empty query

    evidence = retrieve_evidence(chunks, jd, embedder=_FakeEmbedder(), top_k=4)

    assert evidence["education"] == []  # no crash, no fabricated evidence


def test_retrieval_respects_top_k():
    chunks = build_chunks(
        StructuredResume(
            experience=[
                ExperienceItem(company=f"Company {i}", role="Engineer", bullets=[f"Did Python task {i}"])
                for i in range(6)
            ]
        )
    )
    jd = StructuredJD(required_skills=["python"], responsibilities=["Python engineering"])

    evidence = retrieve_evidence(
        chunks, jd, embedder=_FakeEmbedder(), top_k=2, min_similarity=_FAKE_MIN_SIMILARITY
    )

    # Exactly top_k, not just "<= top_k" -- all 6 candidates match similarly
    # well, so this actually exercises truncation rather than passing
    # vacuously if nothing had cleared the threshold.
    assert len(evidence["experience"]) == 2


def test_retrieval_no_chunks_returns_empty_sections():
    evidence = retrieve_evidence([], StructuredJD(required_skills=["python"]), embedder=_FakeEmbedder())
    assert all(v == [] for v in evidence.values())
