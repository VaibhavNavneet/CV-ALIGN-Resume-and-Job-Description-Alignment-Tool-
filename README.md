# CV-Align | Multi-Agent Resume Screener

> Multi-agent resume screening pipeline built with **LangGraph** and **Gemini**, served via **FastAPI**.

CV-Align screens resumes against a job description using a pipeline of
specialized agents (a **parser**, a **JD parser**, a **retriever**, a **matcher**, a
deterministic **scorer**, a deterministic **hygiene** checker, and a **critic**),
with a self-correction loop and a full audit trail for explainability. The
retriever grounds the matcher's scoring in the resume excerpts most relevant to
each JD requirement, retrieved via semantic (Gemini-embedding) search over the
resume — a small, genuine RAG layer, not just a full-document dump.

It serves two personas from one engine:

- **Candidate mode**: "How well does my CV fit this job, and how do I improve it?"
  Feedback comes in two tracks: **build over time** (skills/experience to develop
  for this role) and **fix right now** (immediate, deterministic CV edits).
- **Recruiter mode**: "Rank these resumes for this job, with reasons."
  Every candidate carries the matcher's per-section reasoning + evidence as the
  explicit reason for the ranking.

## Why a multi-agent design?

A single LLM call that "scores a resume" is a black box: you can't tell *why* a
candidate ranked where they did, and you can't improve one stage without
risking the others. CV-Align splits the job into focused stages, using a
plain function where the task is deterministic and an LLM agent only where the
task needs judgment:

| Stage | Type | Responsibility |
|-------|------|----------------|
| Parser | agent | Resume PDF text → structured fields (skills, projects, experience) |
| JD Parser | agent | Job description → structured requirements |
| Retriever | function | Chunks the resume + retrieves top-k relevant excerpts per section (Gemini embeddings, cosine similarity) |
| Matcher | agent | Per-section sub-scores + quoted evidence, grounded in retrieved excerpts, vs **this** JD |
| Scorer | function | Deterministic weighted score from sub-scores |
| Hygiene | function | Deterministic "fix right now" rules: links, quantified bullets, weak verbs, over-long bullets, first-person pronouns, buzzwords, generic names… |
| Critic | agent | JD gaps, "build over time" skill-building advice, verdict, and a confidence check |

If the critic is not confident the score is well-supported, it loops back to the
matcher once for a re-evaluation (capped to avoid infinite loops).

```
                resume.pdf          job description
                    │                     │
                    ▼                     ▼
                 parser ─────────────► jd_parser
                    └──────────┬──────────┘
                               ▼
                          retriever (Gemini embeddings)
                               │
                    ┌────► matcher ──► scorer ──► hygiene ──► critic ─┐
                    │                                                  │
                    └──────── self-correction (≤1 retry) ◄────────────┘
                                          │
                                          ▼
                          ranked, explainable results
```

## Candidate feedback: two tracks

When you screen your own CV (candidate mode), the feedback is deliberately split
into two buckets, because the two kinds of improvement have very different time
horizons:

- **Build over time, for this role.** JD requirements you don't clearly meet
  (`gaps`) plus the critic's forward-looking, skill-building advice
  (`suggestions`): technologies to learn and the kind of experience/projects to
  build next. These are things you *grow into*, not edits you make today.
- **Fix right now, quick CV edits.** Objective, rule-based issues from the
  deterministic hygiene checker (`hygiene_issues`): add impact numbers, lead with
  strong action verbs, condense over-long bullets, drop first-person pronouns and
  buzzwords, add missing links. These are instant, no-LLM, and reproducible, and
  the UI shows them in a separate section below.

Because the "fix right now" bucket is deterministic, it's cheap and defensible
("we combine objective rule-based checks with LLM judgment, not one big prompt")
and adds no latency.

## Status

✅ Core engine + HTTP API complete and tested (94 tests, fully offline). The
core pipeline has been live-verified against Gemini; the evidence-retrieval
layer's offline tests use a fake embedder and have not yet been run against
the live Gemini embeddings API. See `docs/architecture.md` for the full design.

## Tech stack

- Python 3.12
- FastAPI + Uvicorn
- LangGraph + LangChain (agent orchestration)
- Google Gemini (free tier), provider-abstracted so Groq is swappable
- pypdf (PDF text extraction)
- PostgreSQL on Vercel; SQLite for local run history

## Getting started

```bash
# 1. Clone
git clone https://github.com/VaibhavNavneet/CV-ALIGN: Resume and Job Description Alignment Tool.git
cd multi-agent-resume-screener

# 2. Create & activate a virtual environment
python -m venv .venv
# Windows
.venv\Scripts\Activate.ps1
# macOS/Linux
# source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure environment
copy .env.example .env        # Windows  (use `cp` on macOS/Linux)
# then edit .env and set GOOGLE_API_KEY and GROQ_API_KEY
# (the example selects LLM_PROVIDER=groq)
```

For the default Groq setup, keep `LLM_PROVIDER=groq` and supply both keys.
Google's key is required for embeddings even when Groq handles screening.
To use Gemini for screening instead, set `LLM_PROVIDER=gemini` and remove
`LLM_MODEL` so the correct provider default is selected.
Local development uses SQLite automatically when `DATABASE_URL` is unset.

## Usage

### Run the API

```bash
uvicorn multi_agent_resume_screener.api.main:app --reload
# open http://127.0.0.1:8000/docs for interactive Swagger UI
```

Screen resumes against a job description:

```bash
# Recruiter mode: rank multiple resumes
curl -X POST http://127.0.0.1:8000/screen \
  -F "jd=Backend engineer. Required: Python, Go, PostgreSQL, 3+ years." \
  -F "mode=recruiter" \
  -F "resumes=@alice.pdf" \
  -F "resumes=@bob.pdf"

# Candidate mode: feedback to improve one CV
curl -X POST http://127.0.0.1:8000/screen \
  -F "jd=Backend engineer..." \
  -F "mode=candidate" \
  -F "resumes=@my_cv.pdf"

# Retrieve a stored run later
curl http://127.0.0.1:8000/runs/<run_id>
```

### Command-line scripts

```bash
python scripts/check_llm.py                       # verify your LLM key works
python scripts/parse_resume.py resume.pdf         # PDF → structured resume
python scripts/screen.py resume.pdf job.txt       # full pipeline + agent trace
```

## API endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Liveness check |
| POST | `/screen` | Screen resume PDF(s) against a JD; returns ranked results |
| GET | `/runs/{id}` | Retrieve a stored screening run |
| GET | `/runs` | List recent runs |

`/screen` form fields: `jd` (text), `mode` (`candidate`\|`recruiter`),
`critic_mode` (`fast`\|`full`), `critic_top_k` (int), `resumes` (PDF file(s)).

## Testing

```bash
pip install -e ".[dev]"
pytest -q                       # offline tests (fake LLM + fake embedder)
ruff check src tests scripts    # lint
python -m build                 # package including frontend assets
```

## Project layout

```
src/multi_agent_resume_screener/
├── state.py         # Shared Pydantic state + public result models
├── settings.py      # Typed config from .env
├── pdf.py           # Deterministic PDF → text
├── llm/             # Provider-abstracted LLM client (Gemini/Groq)
├── embeddings/      # Gemini embedding client (resume-evidence retrieval)
├── agents/          # parser, jd_parser, matcher, critic
├── pipeline/        # scorer, hygiene, retrieval (deterministic), graph, screen
├── api/             # FastAPI app
└── storage/         # PostgreSQL / local SQLite run persistence
```

## Deployment

Vercel runs the existing FastAPI application as a Python function through
`app.py`. The HTML/CSS/JS, agent pipeline, and API URLs remain the same.
No `package.json`, Node application, custom build command, output directory,
or persistent Uvicorn start command is needed. Use the **FastAPI** framework
preset, the repository root, and **Fluid compute**. Python 3.12 is selected by
`.python-version`; `vercel.json` sets a 300-second function duration.

### Database and environment

Create a PostgreSQL database (for example through the Vercel Marketplace).
Use the provider's pooled, TLS-enabled connection URL as `DATABASE_URL`.
Each storage operation opens and closes its own connection; no database or
uploaded PDF is persisted on the function filesystem. SQLite remains available
locally when `DATABASE_URL` is absent. On Vercel, missing `DATABASE_URL` returns
503 instead of silently losing history in `/tmp`.

Set the following in Vercel's environment variables:

| Variable | Value |
|----------|-------|
| `DATABASE_URL` | PostgreSQL URL, with the provider's TLS settings |
| `GOOGLE_API_KEY` | Required for Gemini embeddings, including when using Groq |
| `LLM_PROVIDER` | `groq` or `gemini` |
| `GROQ_API_KEY` | Required only for Groq |
| `LLM_MODEL` | Optional; defaults to the selected provider's model |
| `EMBEDDING_MODEL` | Optional; defaults to `gemini-embedding-001` |

Never expose these as browser variables or commit `.env`. Both LLM providers
are installed by default. Vercel sets `VERCEL=1` automatically.
Use a separate database for Preview deployments.

### Step 1: Prepare accounts and credentials

- Sign in to your Vercel account.
- Create a PostgreSQL database and obtain its pooled connection URL with TLS.
- Obtain a Google API key for embeddings.
- Obtain a Groq API key for the screening agents.

The steps below use Groq. To use Gemini instead, enter `gemini` for
`LLM_PROVIDER` and skip `GROQ_API_KEY`. Google embeddings require
`GOOGLE_API_KEY` in either case.

### Step 2: Install prerequisites and dependencies

Install **Python 3.12** and **Node.js 22.12 or newer**.
Open PowerShell in the project root (the folder containing `pyproject.toml`
and `vercel.json`), then run:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
npm install --global vercel@latest
```

### Step 3: Connect the project to Vercel

```powershell
vercel login
vercel link
```

Complete browser authentication, select your Vercel account, and create or
link a project. Use these project settings:

| Setting | Value |
|---------|-------|
| Root directory | This repository's root |
| Framework preset | FastAPI |
| Build command | Leave the framework default; no custom override |
| Output directory | Leave the framework default; no custom override |
| Fluid compute | Enabled |

If the CLI reports that your saved login is no longer authorized, run
`vercel login` again before continuing.

### Step 4: Add production environment variables

Run each command and paste the corresponding value at its prompt:

```powershell
vercel env add DATABASE_URL production
vercel env add GOOGLE_API_KEY production
vercel env add LLM_PROVIDER production
vercel env add GROQ_API_KEY production
```

Enter `groq` for `LLM_PROVIDER`. For `DATABASE_URL`, use your database provider's
pooled PostgreSQL URL, including its TLS configuration. Enter real keys at the
CLI prompts, never in source files.

### Step 5: Initialize the database

Download the production variables to a gitignored file and create the table:

```powershell
vercel env pull .env.local --environment=production
.\.venv\Scripts\python.exe -m dotenv -f .env.local run -- .\.venv\Scripts\python.exe scripts/init_db.py
```

Expected output:

```text
Run-history schema is ready.
```

`init_db.py` is idempotent and must run once per database before accepting
screening requests. It creates the `runs` table without touching existing rows.
Existing local SQLite history is not automatically imported.
The downloaded `.env.local` contains secrets; do not commit or share it.

### Step 6: Verify locally and deploy

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m ruff check src tests scripts app.py
.\.venv\Scripts\python.exe -m build
vercel --prod
```

The tests use fake models and do not validate live provider credentials or
database connectivity. Open the deployment URL printed by Vercel to complete
the live checks below.

### Step 7: Verify the deployed application

1. Visit `/health`; it should return `"status": "ok"`.
2. Open `/` and confirm the form loads with its styling.
3. Paste a job description and upload one small, text-based PDF.
4. Submit the form and confirm screening results appear.
5. Visit `/runs` to confirm the result was saved.
6. Use its `id` to visit `/runs/{id}` and confirm the result can be retrieved.
7. Check `/docs` for the interactive API documentation.

If screening times out, try fewer PDFs or select **Fast** critic mode.
If the API reports that `DATABASE_URL` is missing, check the Production
environment variables and redeploy. If run history is unavailable, check the
database connection settings and confirm Step 5 completed successfully.

For future code or production environment changes, run `vercel --prod` again.
For Preview deployments (`vercel` without `--prod`), add the variables to the
Preview environment and initialize its separate database first.

### Request limits and behavior

- At most 10 PDFs, 4 MB total PDF bytes, and 20,000 job-description characters
  per request. These leave room for multipart overhead below Vercel's 4.5 MB
  request limit. The UI and API validate these limits.
- Screening stays inside the request; there are no background tasks that
  depend on a surviving server process. Slow providers or large batches can
  still exceed the 300-second duration. The UI suggests a smaller batch or
  Fast critic mode after a timeout. Larger workloads require a durable job
  queue and separate worker, outside this synchronous deployment.
- Frontend requests use same-origin `/screen`, so no CORS allowlist or backend
  hostname is necessary. PDF extraction uses memory; multipart spooling uses
  the runtime's temporary directory and FastAPI cleans up upload files.
- Run endpoints retain the original shared-history behavior without per-user
  authentication. Keep the deployment access-restricted if processing private
  resumes; do not treat it as a multi-tenant service.

See [Vercel FastAPI support](https://vercel.com/docs/frameworks/backend/fastapi)
and [function limits](https://vercel.com/docs/functions/limitations).
The optional `Dockerfile` remains for local container use; Vercel does not use it.

## License

[MIT](LICENSE) © 2026 Shreyansh Dutt Mehra
