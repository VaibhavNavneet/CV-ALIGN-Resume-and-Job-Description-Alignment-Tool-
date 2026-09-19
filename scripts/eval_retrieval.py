"""Evaluate the evidence-retrieval (RAG) layer with Recall@K over a golden set.

Usage (from the project root, with the virtualenv active and GOOGLE_API_KEY
set in .env):

    python scripts/eval_retrieval.py                # real Gemini embeddings
    python scripts/eval_retrieval.py --top-k 3
    python scripts/eval_retrieval.py --fake          # dry run, no API calls

By default this hits the real Gemini embeddings API (the same one the
production pipeline uses), because a metric worth putting on a resume has to
be measured against the real system, not the offline fake embedder the test
suite uses for hermetic CI. Use --fake only to sanity-check the harness
itself without spending API quota.

Recall@K here = (# gold-relevant chunks that appear in the top-K retrieved
results) / (# gold-relevant chunks that exist for that section), averaged
across every (resume, JD, section) query where at least one relevant chunk
exists. Queries with zero relevant chunks (the negative controls) are
reported separately as a no-fabrication check, since recall is undefined
(0/0) for them.
"""

from __future__ import annotations

import argparse
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from golden_dataset import GOLDEN_QUERIES, NEGATIVE_CONTROLS, GoldenQuery  # noqa: E402

from dotenv import load_dotenv

from multi_agent_resume_screener.embeddings.client import EmbeddingConfigError, get_embedder
from multi_agent_resume_screener.pipeline.retrieval import build_chunks, retrieve_evidence


def _group_by_resume_jd(queries: list[GoldenQuery]) -> list[dict]:
    """Group golden queries sharing a (resume, JD) pair so retrieval only
    runs once per pair, not once per section -- fewer embedding API calls."""
    groups: dict[tuple[int, int], dict] = {}
    order: list[tuple[int, int]] = []
    for q in queries:
        key = (id(q.resume), id(q.jd))
        if key not in groups:
            groups[key] = {"resume": q.resume, "jd": q.jd, "queries": []}
            order.append(key)
        groups[key]["queries"].append(q)
    return [groups[k] for k in order]


_RETRY_DELAY_RE = re.compile(r"retry in ([\d.]+)s", re.IGNORECASE)
_MAX_RETRIES = 5
_DEFAULT_RETRY_DELAY = 40.0  # seconds; safely past a free-tier per-minute window
_PACING_DELAY = 1.5  # seconds between (resume, JD) groups, to avoid tripping the limit at all


def _retrieve_with_retry(chunks, jd, embedder, top_k, min_similarity):
    """Call retrieve_evidence(), retrying on a free-tier rate limit (429).

    The eval script fires through many (resume, JD) groups back-to-back,
    which can trip the free tier's per-minute embedding quota even though a
    single real screening request (1-2 calls) never would. Retries using the
    API's own suggested delay when present, else a fixed fallback.

    ``min_similarity=None`` means "use retrieve_evidence()'s own production
    default" (kept as the single source of truth in retrieval.py rather than
    duplicated here) -- used for real Gemini runs. --fake runs pass an
    explicit low threshold instead, since the fake embedder's score scale
    isn't comparable to the real one the production default was calibrated
    against (see retrieval.py's _MIN_SIMILARITY comment).
    """
    kwargs = {"top_k": top_k}
    if min_similarity is not None:
        kwargs["min_similarity"] = min_similarity
    for attempt in range(1, _MAX_RETRIES + 1):
        try:
            return retrieve_evidence(chunks, jd, embedder=embedder, **kwargs)
        except Exception as exc:  # noqa: BLE001 - only rate limits are retried, others re-raise
            message = str(exc)
            is_rate_limit = "429" in message or "quota" in message.lower()
            if not is_rate_limit or attempt == _MAX_RETRIES:
                raise
            match = _RETRY_DELAY_RE.search(message)
            delay = float(match.group(1)) + 5 if match else _DEFAULT_RETRY_DELAY
            print(f"  [rate limited, attempt {attempt}/{_MAX_RETRIES}] waiting {delay:.0f}s before retry...")
            time.sleep(delay)


class _FakeEmbedder:
    """Deterministic bag-of-words embedder for --fake dry runs (mirrors the
    one used in the offline test suite; never used for the published metric).
    """

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        import math
        import re
        import zlib

        # Mirrors the real Gemini API, which rejects blank text with a 400 --
        # strict on purpose so a --fake dry run catches empty-query bugs too.
        for t in texts:
            if not t.strip():
                raise ValueError("embed_documents() received blank text")

        def vectorize(text: str, dim: int = 64) -> list[float]:
            vec = [0.0] * dim
            for tok in re.findall(r"[a-z0-9]+", text.lower()):
                vec[zlib.crc32(tok.encode("utf-8")) % dim] += 1.0
            norm = math.sqrt(sum(v * v for v in vec))
            return [v / norm for v in vec] if norm else vec

        return [vectorize(t) for t in texts]


_FAKE_MIN_SIMILARITY = 0.05  # the fake's score scale isn't comparable to real Gemini's


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--top-k", type=int, default=4, help="Retrieval top-k (default: 4, matches PipelineConfig default)")
    parser.add_argument("--fake", action="store_true", help="Use a deterministic fake embedder instead of the real Gemini API")
    parser.add_argument(
        "--min-similarity", type=float, default=None,
        help="Override the relevance floor. Defaults to retrieval.py's production value for real "
             f"runs, or {_FAKE_MIN_SIMILARITY} for --fake (the two embedders' score scales aren't comparable).",
    )
    args = parser.parse_args()

    load_dotenv()

    if args.fake:
        embedder = _FakeEmbedder()
        min_similarity = args.min_similarity if args.min_similarity is not None else _FAKE_MIN_SIMILARITY
        print("Using FAKE embedder (dry run) -- these numbers are NOT for publishing.\n")
    else:
        try:
            embedder = get_embedder()
        except EmbeddingConfigError as exc:
            print(f"[CONFIG ERROR] {exc}")
            return 2
        min_similarity = args.min_similarity  # None -> retrieve_evidence()'s own production default
        print("Using the real Gemini embedder configured in your environment.\n")

    all_queries = GOLDEN_QUERIES + NEGATIVE_CONTROLS
    groups = _group_by_resume_jd(all_queries)

    per_query_recall: list[tuple[str, int, int]] = []  # id, hits, total_relevant
    negative_control_leaks: list[tuple[str, int]] = []  # id, # retrieved (should be 0)

    print(f"{'query':<38} {'section':<12} {'result':<18}")
    print("-" * 70)

    for i, group in enumerate(groups):
        if i > 0 and not args.fake:
            time.sleep(_PACING_DELAY)  # spread real API calls to avoid tripping the quota
        chunks = build_chunks(group["resume"])
        try:
            evidence = _retrieve_with_retry(chunks, group["jd"], embedder, args.top_k, min_similarity)
        except Exception as exc:  # noqa: BLE001 - surface any provider/network error
            print(f"[REQUEST FAILED] {type(exc).__name__}: {exc}")
            return 1

        for q in group["queries"]:
            section_chunks = [c for c in chunks if c.section == q.section]
            relevant_chunks = [c for c in section_chunks if q.relevant(c.text)]
            retrieved_texts = {rc.chunk.text for rc in evidence.get(q.section, [])}

            if relevant_chunks:
                hits = sum(1 for c in relevant_chunks if c.text in retrieved_texts)
                per_query_recall.append((q.id, hits, len(relevant_chunks)))
                print(f"{q.id:<38} {q.section:<12} recall={hits}/{len(relevant_chunks)}")
            else:
                leaked = len(retrieved_texts)
                negative_control_leaks.append((q.id, leaked))
                status = "OK (nothing retrieved)" if leaked == 0 else f"LEAK ({leaked} retrieved!)"
                print(f"{q.id:<38} {q.section:<12} {status}")

    print("-" * 70)

    macro_recall = sum(h / t for _, h, t in per_query_recall) / len(per_query_recall)
    micro_hits = sum(h for _, h, _ in per_query_recall)
    micro_total = sum(t for _, _, t in per_query_recall)
    micro_recall = micro_hits / micro_total

    print(f"\nRecall@{args.top_k} (macro-avg over {len(per_query_recall)} queries): {macro_recall:.3f}")
    print(f"Recall@{args.top_k} (micro-avg, {micro_hits}/{micro_total} relevant chunks retrieved): {micro_recall:.3f}")

    leaks = sum(1 for _, n in negative_control_leaks if n > 0)
    print(f"\nNegative-control check ({len(negative_control_leaks)} unrelated-JD queries): "
          f"{len(negative_control_leaks) - leaks}/{len(negative_control_leaks)} correctly retrieved nothing")

    print(
        f"\nSuggested line: \"Built and evaluated a RAG evidence-retrieval layer "
        f"against a {len(per_query_recall)}-query golden dataset, achieving "
        f"Recall@{args.top_k} = {macro_recall:.2f}.\""
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
