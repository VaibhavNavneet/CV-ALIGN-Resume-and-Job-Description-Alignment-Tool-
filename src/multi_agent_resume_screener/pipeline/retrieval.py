"""Evidence retrieval: chunk the resume, embed it, retrieve top-k per section.

This is the RAG layer that grounds the matcher's per-section scoring in
specific resume excerpts rather than the full resume alone. It is entirely
deterministic *except* for the embedding call itself, which is delegated to
an injected ``Embeddings`` client (Gemini in production; a fake in tests --
see :mod:`multi_agent_resume_screener.embeddings.client`).

Chunking is one unit per resume item (one per experience/project/education
entry, one for the whole skills list) so each chunk is a meaningful,
self-contained piece of evidence with real metadata (company/role, project
name, institute). Retrieval builds one query per section from the relevant
``StructuredJD`` fields, embeds it, and ranks that section's own chunks by
cosine similarity against it -- dropping anything below a relevance floor so
a section with no genuinely relevant evidence returns an empty list rather
than padded, irrelevant top-k results.
"""

from __future__ import annotations

import math

from langchain_core.embeddings import Embeddings

from multi_agent_resume_screener.embeddings.client import get_embedder
from multi_agent_resume_screener.state import (
    SECTIONS,
    ResumeChunk,
    RetrievedChunk,
    Section,
    StructuredJD,
    StructuredResume,
)

# Below this cosine similarity, a chunk is not considered relevant evidence.
#
# Calibrated empirically against the real Gemini embedder (see
# scripts/inspect_similarity.py), not guessed: same-domain resume/JD matches
# scored 0.850-0.914 across skills/experience/projects, while every
# cross-domain (partial or zero skill overlap) pairing topped out at 0.789 --
# dense embeddings compress short resume/JD text into a much narrower,
# higher similarity band than a lexical model would, so a threshold tuned
# for one is meaningless for the other. 0.80 sits in that observed gap.
# This is a small empirical sample (a handful of domain pairs); revisit if
# production usage shows the matcher frequently getting "no evidence
# retrieved" on things that should plausibly match.
_MIN_SIMILARITY = 0.80


def build_chunks(resume: StructuredResume) -> list[ResumeChunk]:
    """Split a structured resume into retrievable, semantically meaningful chunks.

    One chunk per experience/project/education item (fully-empty items are
    skipped), plus one chunk for the whole skills list.
    """
    chunks: list[ResumeChunk] = []

    for exp in resume.experience:
        if not (exp.company or exp.role or exp.bullets):
            continue
        title = " at ".join(p for p in (exp.role, exp.company) if p) or None
        text = "\n".join(filter(None, [title, exp.dates, *exp.bullets]))
        chunks.append(ResumeChunk(section="experience", title=title, text=text))

    for proj in resume.projects:
        if not (proj.name or proj.description or proj.tech):
            continue
        text = "\n".join(filter(None, [proj.description, ", ".join(proj.tech)]))
        chunks.append(ResumeChunk(section="projects", title=proj.name, text=text))

    for edu in resume.education:
        if not (edu.degree or edu.institute or edu.year or edu.gpa):
            continue
        text = " ".join(filter(None, [edu.degree, edu.institute, edu.year, edu.gpa]))
        chunks.append(ResumeChunk(section="education", title=edu.institute, text=text))

    if resume.skills:
        chunks.append(
            ResumeChunk(section="skills", title=None, text=", ".join(resume.skills))
        )

    return chunks


def _section_query(section: Section, jd: StructuredJD) -> str:
    """Build the retrieval query text for one section from the structured JD."""
    if section == "skills":
        return " ".join(jd.required_skills + jd.nice_to_have_skills)
    if section == "experience":
        parts = list(jd.responsibilities)
        if jd.title:
            parts.append(jd.title)
        if jd.min_experience_years:
            parts.append(f"{jd.min_experience_years} years of experience")
        return " ".join(parts)
    if section == "projects":
        return " ".join(jd.responsibilities + jd.required_skills)
    return " ".join(jd.qualifications)  # education


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


def retrieve_evidence(
    chunks: list[ResumeChunk],
    jd: StructuredJD,
    embedder: Embeddings | None = None,
    top_k: int = 4,
    min_similarity: float = _MIN_SIMILARITY,
) -> dict[str, list[RetrievedChunk]]:
    """Retrieve the top-k most relevant resume chunks per section.

    Args:
        chunks: The resume's chunks (see :func:`build_chunks`).
        jd: The structured job description.
        embedder: Optional embeddings client (dependency injection for
            tests). Falls back to the configured Gemini embedder.
        top_k: Max chunks to return per section.
        min_similarity: Chunks scoring below this are dropped rather than
            padding the results with irrelevant evidence.

    Returns:
        One entry per :data:`~multi_agent_resume_screener.state.SECTIONS`,
        each a list of :class:`RetrievedChunk` sorted by descending score
        (possibly empty if no chunk in that section met the relevance floor).
    """
    embedder = embedder or get_embedder()

    by_section: dict[str, list[ResumeChunk]] = {section: [] for section in SECTIONS}
    for chunk in chunks:
        by_section[chunk.section].append(chunk)

    sections_with_chunks = [s for s in SECTIONS if by_section[s]]
    result: dict[str, list[RetrievedChunk]] = {section: [] for section in SECTIONS}
    if not sections_with_chunks:
        return result

    # Batch both embedding calls (all chunk texts once, all section queries
    # once) rather than one call per chunk/query, to minimise API usage.
    # Blank text is excluded from both batches: Gemini's embedding API
    # rejects an empty content Part outright (400 error), and an empty JD
    # query has no signal to search with anyway, so those sections
    # correctly yield no evidence rather than crashing the whole request.
    all_chunks = [c for s in sections_with_chunks for c in by_section[s] if c.text.strip()]
    if not all_chunks:
        return result
    chunk_vectors = embedder.embed_documents([c.text for c in all_chunks])
    vectors_by_id = dict(zip((id(c) for c in all_chunks), chunk_vectors))
    chunked_sections = {c.section for c in all_chunks}

    queryable_sections = [
        s for s in sections_with_chunks if s in chunked_sections and _section_query(s, jd).strip()
    ]
    if not queryable_sections:
        return result

    queries = [_section_query(s, jd) for s in queryable_sections]
    query_vectors = embedder.embed_documents(queries)

    for section, query_vector in zip(queryable_sections, query_vectors):
        scored = sorted(
            (
                RetrievedChunk(chunk=c, score=_cosine(query_vector, vectors_by_id[id(c)]))
                for c in by_section[section]
                if id(c) in vectors_by_id
            ),
            key=lambda rc: rc.score,
            reverse=True,
        )
        result[section] = [rc for rc in scored[:top_k] if rc.score >= min_similarity]

    return result
