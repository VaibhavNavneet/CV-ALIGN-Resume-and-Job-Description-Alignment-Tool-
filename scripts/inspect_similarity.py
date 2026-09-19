"""Print raw, UNFILTERED cosine similarity scores for a sample of golden
queries against the real embedder, to calibrate `min_similarity` in
pipeline/retrieval.py.

The min_similarity=0.3 threshold was tuned against the crude offline lexical
fake embedder (tests/test_retrieval.py's bag-of-words hash vectorizer). Real
dense embedding models (Gemini included) are known to produce a much higher
baseline similarity between *any* two pieces of natural language than a
bag-of-words model does, so that threshold may not be discriminating
anything against the real API. This script bypasses retrieve_evidence()'s
threshold/top-k filtering entirely and prints every candidate chunk's raw
score, sorted descending, so we can see the real distribution before picking
a new number.

Usage: python scripts/inspect_similarity.py   (needs GOOGLE_API_KEY in .env)
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from golden_dataset import BACKEND_RESUME, DEVOPS_RESUME, FRONTEND_RESUME, JD_BACKEND, JD_DEVOPS, JD_ML  # noqa: E402

from dotenv import load_dotenv

from multi_agent_resume_screener.embeddings.client import get_embedder
from multi_agent_resume_screener.pipeline.retrieval import _cosine, _section_query, build_chunks
from multi_agent_resume_screener.state import Section, StructuredJD, StructuredResume


def _inspect(label: str, resume: StructuredResume, jd: StructuredJD, section: Section, embedder) -> None:
    chunks = [c for c in build_chunks(resume) if c.section == section]
    query = _section_query(section, jd)
    print(f"\n=== {label} ({section}) ===")
    print(f"query: {query!r}")
    if not query.strip() or not chunks:
        print("  (no query or no chunks for this section -- skipped)")
        return

    query_vec = embedder.embed_documents([query])[0]
    chunk_vecs = embedder.embed_documents([c.text for c in chunks])

    scored = sorted(
        ((c, _cosine(query_vec, v)) for c, v in zip(chunks, chunk_vecs)),
        key=lambda pair: pair[1],
        reverse=True,
    )
    for chunk, score in scored:
        title = chunk.title or "(skills/no title)"
        preview = chunk.text[:60].replace("\n", " ")
        print(f"  score={score:.4f}  [{title}]  {preview}...")


def main() -> int:
    load_dotenv()
    embedder = get_embedder()

    # Case 1: positive query with a real decoy in the same section --
    # confirms the relevant item scores meaningfully higher than the decoy.
    _inspect("backend resume vs backend JD (has a decoy item)", BACKEND_RESUME, JD_BACKEND, "experience", embedder)
    _inspect("backend resume vs backend JD (has a decoy item)", BACKEND_RESUME, JD_BACKEND, "projects", embedder)
    _inspect("backend resume vs backend JD (true match, skills has no decoy)", BACKEND_RESUME, JD_BACKEND, "skills", embedder)
    _inspect("backend resume vs backend JD (true match, education has no decoy)", BACKEND_RESUME, JD_BACKEND, "education", embedder)

    # Case 2: negative control where the two domains share ONE real skill
    # (Python) -- the "leak" flagged in the eval might be legitimate partial
    # relevance rather than a bug.
    _inspect("backend resume vs ML JD (shares 'Python')", BACKEND_RESUME, JD_ML, "skills", embedder)
    _inspect("backend resume vs ML JD (shares 'Python')", BACKEND_RESUME, JD_ML, "experience", embedder)

    # Case 3: negative control with NO shared skill at all -- the cleanest
    # test of whether the threshold discriminates anything for real.
    _inspect("frontend resume vs DevOps JD (zero skill overlap)", FRONTEND_RESUME, JD_DEVOPS, "skills", embedder)
    _inspect("frontend resume vs DevOps JD (zero skill overlap)", FRONTEND_RESUME, JD_DEVOPS, "experience", embedder)

    # Case 4: same as case 3 but with a different unrelated pair, for a
    # second data point on the "totally unrelated" baseline.
    _inspect("DevOps resume vs Backend JD (zero skill overlap)", DEVOPS_RESUME, JD_BACKEND, "skills", embedder)

    print(
        "\nDone. Compare the RELEVANT item's score against the decoy's score in "
        "case 1, and look at how high cases 3-4's scores are (should be clearly "
        "lower than case 1's relevant item if the embedding space discriminates "
        "unrelated domains). Paste this full output back for a threshold recommendation."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
