# Architecture — CV-Align | Multi-Agent Resume Screener

## Problem

Given one or more resumes (PDF) and a single job description (JD), produce a
ranked shortlist where every candidate's score is **explainable**: which
sections matched, what evidence supported the score, what gaps exist, and what
the candidate could improve.

The earlier single-pass design returned a number with no traceable reasoning.
This version decomposes the work into specialized agents so each stage is
independently inspectable, testable, and improvable.

## Function vs. agent

We use a plain Python **function** when the task is deterministic, and an
**agent** (LLM call) only when the task needs judgment. This keeps cost down and
keeps behavior reproducible wherever possible.

| Task | Implementation |
|------|----------------|
| PDF → text | function (`pypdf`) |
| Text → structured resume | agent |
| JD text → structured requirements | agent |
| Resume chunking + evidence retrieval | function (chunking is deterministic; ranks chunks by embedding similarity via the Gemini embeddings API — not an LLM reasoning call) |
| Resume ↔ JD section matching | agent (now grounded in retrieved evidence) |
| Sub-scores → final score | function (weighted sum) |
| Objective resume-quality checks (hygiene) | function (rule-based) |
| Gaps / suggestions / verdict | agent |

## Pipeline

```
┌─────────────────┐     ┌─────────────────┐
│  Resume PDF     │     │ Job Description │
└────────┬────────┘     └────────┬────────┘
         │                       │
         ▼                       ▼
   ┌──────────┐            ┌──────────┐
   │  Parser  │            │   JD     │
   │  Agent   │            │  Parser  │
   └────┬─────┘            └────┬─────┘
        │  StructuredResume     │  StructuredJD
        └───────────┬───────────┘
                    ▼
              ┌──────────┐
              │Retriever │  chunks resume + retrieves top-k
              │ Function │  evidence per section (Gemini embeddings)
              └────┬─────┘
                    ▼
              ┌──────────┐
              │ Matcher  │  per-section sub-scores + evidence,
              │  Agent   │  grounded in retrieved excerpts
              └────┬─────┘ ◄──────────────┐
                   ▼                      │ (self-correction,
              ┌──────────┐                │  max 1 retry)
              │  Scorer  │  weighted      │
              │ Function │  final score   │
              └────┬─────┘                │
                   ▼                      │
              ┌──────────┐                │
              │  Critic  │  gaps,         │
              │  Agent   │  suggestions,  │
              └────┬─────┘  needs_rescore─┘
                   ▼
              ┌──────────┐
              │  Report  │  score + rationale + gaps + verdict
              └──────────┘
```

> The deterministic **hygiene** function runs between the scorer and the critic;
> it is omitted from the box diagram above for space but feeds the critic and the
> final report.

## Evidence retrieval (RAG layer)

**Why.** Before this layer existed, the matcher scored each section by reading
the *entire* structured resume as one JSON blob. There was no explicit step
saying "here is the specific resume evidence most relevant to this JD
requirement" before scoring — the matcher had to find it itself, buried in
everything else. The retriever makes that step explicit: it surfaces the
top-matching resume excerpts per section *before* the matcher runs, so scoring
is grounded in retrieved evidence rather than the matcher's own unaided
search through the full document.

**Chunking.** `pipeline/retrieval.py::build_chunks` splits a `StructuredResume`
into one chunk per meaningful item: one per experience entry, one per project,
one per education entry, plus a single chunk for the whole (flat) skills list.
Fully-empty items are skipped. Each chunk carries `section`, an optional
`title` (company/role, project name, institute) for display, and the
reconstructed `text`. This granularity avoids both extremes — a single
chunk-per-section (too coarse to distinguish which specific project/role is
relevant) and one-chunk-per-sentence (arbitrary fragments with no standalone
meaning).

**Embeddings.** Retrieval embeds chunk text and JD-derived queries with
Gemini's `gemini-embedding-001` model via `langchain-google-genai` —
already a required dependency (used for the Gemini chat-model option), so
this adds **zero new packages**. This makes retrieval genuinely *semantic*
(the goal), not a keyword/hash match. The tradeoff: this couples the
retrieval layer to `GOOGLE_API_KEY` even when `LLM_PROVIDER=groq` (the
repo's actual default per `.env.example`) — Groq has no embeddings API, so
there is no way to keep retrieval semantic without depending on some
embeddings provider, and Gemini's free tier was the pragmatic choice given
the project's existing Gemini integration. `text-embedding-004`, the model
referenced by most older tutorials, was deprecated and shut down by Google
on 2026-01-14; `gemini-embedding-001` is the current stable, GA, free-tier
replacement (verified against Google's docs while building this feature, not
assumed). The model is overridable via the `EMBEDDING_MODEL` setting.

**Vector store.** There isn't a persistent one. Chunk embeddings are computed
inside the retrieval node for one resume's screening run and held only in
local variables — never written to `PipelineState`, never persisted, never
loaded as a long-lived index. This fits Vercel's request-based execution:
nothing embedding-related is kept in memory beyond a single request.

**Retrieval.** For each of the four sections, `retrieve_evidence` builds one
query from the relevant `StructuredJD` fields (skills from
`required_skills`/`nice_to_have_skills`; experience from `responsibilities`/
`title`/`min_experience_years`; projects from `responsibilities`/
`required_skills`; education from `qualifications`), embeds it, and ranks
that section's own chunks by cosine similarity. Both the batch of all chunk
texts and the batch of all section queries are embedded in one API call each
(not one call per chunk/query), so a typical resume costs about two
embedding requests, not dozens. Results below `min_similarity` (0.80) are
dropped rather than padded with irrelevant top-k chunks — a section with no
genuinely relevant evidence returns an empty list. The matcher's system
prompt explicitly distinguishes "no evidence retrieved" from "candidate
lacks the skill" and instructs it to never invent evidence not present in
the resume.

`min_similarity` was recalibrated from an initial guess of 0.3 to 0.80 after
running the real Gemini embedder against a sample of golden-dataset queries
(`scripts/inspect_similarity.py`): dense embeddings compress short resume/JD
text into a much narrower, higher similarity band than the offline lexical
fake embedder does, so a threshold tuned against the fake was not
discriminating anything against the real API — every negative-control query
was retrieving evidence it shouldn't have. Empirically, same-domain matches
scored 0.850-0.914 across skills/experience/projects, while every
cross-domain pairing (including ones sharing a single overlapping skill)
topped out at 0.789; 0.80 sits in that gap. This is a small empirical
sample, not a proven-optimal constant — worth revisiting if production usage
shows the matcher frequently getting "no evidence retrieved" on things that
should plausibly match. Because the two embedders' score distributions
aren't comparable, offline tests pass their own low threshold explicitly
(`tests/test_retrieval.py::_FAKE_MIN_SIMILARITY`) rather than inheriting
this production-tuned default.

A full run of the 65-query golden dataset against the real threshold landed
at **Recall@4 = 0.89** (58/65), with **14/14 negative controls retrieving
nothing** — every miss was in the `education` section specifically, never
skills/experience/projects (100% each). Investigating with
`inspect_similarity.py` showed why: a genuine education match (e.g. "B.Tech
Computer Science" against a JD asking for "Bachelor's degree in Computer
Science") scored 0.784 — *inside* the same band as cross-domain false
positives (0.759-0.789) measured elsewhere, not clearly above it. This is a
deliberate, understood tradeoff rather than a bug to keep chasing: lowering
the threshold enough to catch that education match would very likely
reintroduce false positives in other sections, since the score bands
overlap. Education also carries the lowest weight in `DEFAULT_WEIGHTS`
(0.10), and the matcher always sees the full structured resume regardless of
retrieval, so a missed education-evidence highlight doesn't remove
information the matcher could otherwise use — it only means that one
section's score isn't retrieval-grounded. Precision (zero fabricated
cross-domain evidence) was chosen over recall in the one section where the
cost of missing it is lowest.

**Grounding.** `retrieve_evidence_node` (in `pipeline/graph.py`) runs once per
resume, between `parse_jd` and `match`, storing chunks and retrieved evidence
on `PipelineState`. `match_node` passes `state.retrieved_evidence` into
`agents/matcher.py::match()`, which appends a `## RETRIEVED EVIDENCE` block
to its prompt (per section: score, title, quoted text, or an explicit
"no evidence retrieved" line) — in addition to, not instead of, the full
structured resume it already received, so no existing behavior is lost.

**Self-correction.** The critic → matcher retry edge (`prepare_retry → match`)
bypasses `retrieve_evidence` entirely in the graph topology, so a retry reuses
the same retrieved evidence (and does not re-embed anything) — the matcher
just gets another attempt at reasoning from the same grounded evidence, plus
the critic's feedback.

**Testing.** All of the above except the embedding call itself is pure,
deterministic Python. The embedding call is isolated behind
`embeddings/client.py::get_embedder()` (mirrors `llm/client.py`'s provider
factory) and injected as a parameter everywhere the chat model already is
(`build_pipeline`, `run_pipeline`, `screen`, the `/screen` FastAPI dependency).
Offline tests inject a small deterministic hashing/bag-of-words fake embedder
instead of calling Gemini — good enough to prove ranking logic, never used in
production. No test in the suite requires `GOOGLE_API_KEY` or network access.

## Orchestration & personas

A higher layer (`pipeline/screen.py`) serves two personas from one engine:

- **candidate** — one resume, full critic feedback to improve the CV.
- **recruiter** — many resumes ranked against one JD. In `full` critic mode every
  candidate is critiqued; in `fast` mode the resumes are scored first and only
  the **top-K** are critiqued, saving LLM calls at scale.

The JD is parsed once and reused across all resumes (the parser nodes are
idempotent), and resumes are processed concurrently with `asyncio`. The
`enable_critic` flag drives a conditional edge after the hygiene node so fast
mode can skip the inline critic and critique only the top-K ranked candidates.

## Persistence

Each screening run is stored by `storage/runs.py` (PostgreSQL on Vercel, local
SQLite otherwise) as a row containing the full JSON result, retrievable via
`GET /runs/{id}` and listed via `GET /runs`. A fresh connection is opened per
operation, keeping the store safe to call from FastAPI's worker threads.

## Design decisions

- **Per-resume pipeline, parallelized.** Each resume runs the full pipeline
  independently (`asyncio.gather`), then a final pass ranks them. This is
  debuggable, retry-friendly, and avoids long-context recency bias from stuffing
  many resumes into one call.
- **Matcher returns sub-scores + evidence, not a final number.** Each section
  (skills, experience, projects, education) is scored independently with quoted
  evidence and short reasoning.
- **Scorer is deterministic.** `final = Σ weightᵢ · sub_scoreᵢ`. Weights are
  hardcoded defaults with an optional per-request override. The math is provable
  and tunable without touching prompts.
- **Critic drives a single self-correction loop.** If the critic's confidence in
  the scoring is low, it sets `needs_rescore` and the graph loops back to the
  matcher once (`retry_count < max_retries`). One conditional edge justifies
  using LangGraph over plain function calls.
- **Configurable critic.** `fast` mode runs the critic only on the top-K ranked
  resumes; `full` mode runs it on every resume.
- **Configurable retrieval.** `retrieval_top_k` (default 4, per request) caps
  how many evidence chunks are surfaced to the matcher per section, mirroring
  `critic_top_k`.

## Shared state

All nodes read from and write to a single `PipelineState` (Pydantic). It
accumulates structured outputs as agents run, tracks retry bookkeeping, and
maintains a `trace` list — one entry per agent invocation — which becomes the
explainability/audit log persisted to the run store.

See `src/multi_agent_resume_screener/state.py` (added in Step 4) for the concrete schema.

## Scoring weights (defaults)

| Section | Weight |
|---------|--------|
| Skills | 0.35 |
| Experience | 0.30 |
| Projects | 0.25 |
| Education | 0.10 |

Overridable per request via the API `config`.
