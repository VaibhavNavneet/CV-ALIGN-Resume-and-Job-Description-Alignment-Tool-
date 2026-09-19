"use strict";

const form = document.getElementById("screen-form");
const statusEl = document.getElementById("status");
const resultsEl = document.getElementById("results");
const submitBtn = document.getElementById("submit-btn");

const SECTION_ORDER = ["skills", "experience", "projects", "education"];

/* Evergreen recruiter quick-wins shown to candidates (things you can do right
   now that a text-only checker can't verify from an extracted PDF). */
const EVERGREEN_QUICKWINS = [
  "Bold the tools, metrics, and keywords a recruiter scans for.",
  "Lead each bullet with a strong action verb and a concrete number.",
  "Keep each bullet to a single line; cut filler words.",
  "Mirror the exact skill terms from the job description where they apply.",
];

/* Small DOM helper that sets text safely (no innerHTML for untrusted data). */
function el(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined && text !== null) node.textContent = text;
  return node;
}

function showStatus(message, isError) {
  statusEl.hidden = false;
  statusEl.textContent = message;
  statusEl.classList.toggle("error", Boolean(isError));
}

function clearStatus() {
  statusEl.hidden = true;
  statusEl.textContent = "";
  statusEl.classList.remove("error");
}

function pct(value) {
  return Math.round((Number(value) || 0) * 100);
}

function renderBreakdown(breakdown) {
  const wrap = el("div", "breakdown");
  SECTION_ORDER.forEach((section) => {
    const value = breakdown && section in breakdown ? breakdown[section] : 0;
    const row = el("div", "bar-row");
    row.appendChild(el("span", "bar-label", section));

    const track = el("div", "bar-track");
    const fill = el("div", "bar-fill");
    fill.style.width = pct(value) + "%";
    track.appendChild(fill);
    row.appendChild(track);

    row.appendChild(el("span", "bar-value", pct(value) + "%"));
    wrap.appendChild(row);
  });
  return wrap;
}

function renderList(title, items, emptyText) {
  const frag = document.createDocumentFragment();
  frag.appendChild(el("p", "section-title", title));
  if (items && items.length) {
    const ul = el("ul");
    items.forEach((item) => ul.appendChild(el("li", null, item)));
    frag.appendChild(ul);
  } else {
    frag.appendChild(el("p", "empty", emptyText));
  }
  return frag;
}

function renderReasons(candidate) {
  const reasons = candidate.section_reasons || {};
  const evidence = candidate.section_evidence || {};
  const sections = SECTION_ORDER.filter((s) => reasons[s]);
  if (!sections.length) return null;

  const wrap = el("div", "reasons");
  wrap.appendChild(el("p", "section-title", "Why this ranking"));
  sections.forEach((section) => {
    const row = el("div", "reason-row");
    row.appendChild(el("span", "reason-section", section));
    const body = el("div", "reason-body");
    body.appendChild(el("span", "reason-text", reasons[section]));
    const ev = evidence[section];
    if (ev && ev.length) {
      body.appendChild(el("span", "reason-evidence", "Evidence: " + ev.join("; ")));
    }
    row.appendChild(body);
    wrap.appendChild(row);
  });
  return wrap;
}

function renderCandidate(candidate, index, mode) {
  const card = el("div", "candidate");

  const head = el("div", "candidate-head");
  const titleWrap = el("div");
  const name = el("h3", "candidate-name", candidate.candidate_name || "Candidate");
  const file = el("span", "candidate-file", "  " + (candidate.filename || ""));
  name.appendChild(file);
  titleWrap.appendChild(name);
  head.appendChild(titleWrap);

  const scoreWrap = el("div");
  scoreWrap.appendChild(el("span", "score", pct(candidate.score) + "%  "));
  const verdict = candidate.verdict || "weak_fit";
  scoreWrap.appendChild(el("span", "badge " + verdict, verdict.replace("_", " ")));
  head.appendChild(scoreWrap);

  card.appendChild(head);
  card.appendChild(renderBreakdown(candidate.breakdown));

  const reasons = renderReasons(candidate);
  if (reasons) card.appendChild(reasons);

  // Bucket 1 — Build over time: skills/experience to develop for this role.
  const build = el("div", "bucket build-over-time");
  build.appendChild(el("p", "bucket-title", "Build over time — for this role"));
  build.appendChild(
    renderList("Gaps vs the role", candidate.gaps, "No major gaps found.")
  );
  build.appendChild(
    renderList(
      "Skills & experience to build",
      candidate.suggestions,
      "No suggestions."
    )
  );
  card.appendChild(build);

  // Bucket 2 — Fix right now: immediate, deterministic CV edits, shown below.
  const fix = el("div", "bucket fix-now");
  fix.appendChild(el("p", "bucket-title", "Fix right now — quick CV edits"));
  if (candidate.hygiene_score !== null && candidate.hygiene_score !== undefined) {
    fix.appendChild(
      el("p", "hygiene-line", "Resume hygiene: " + pct(candidate.hygiene_score) + "%")
    );
  }
  const issues = candidate.hygiene_issues || [];
  if (issues.length) {
    const ul = el("ul");
    issues.forEach((i) => ul.appendChild(el("li", null, i)));
    fix.appendChild(ul);
  } else {
    fix.appendChild(el("p", "empty", "No quick fixes detected — clean resume."));
  }
  if (mode === "candidate") {
    fix.appendChild(el("p", "tips-title", "Evergreen recruiter quick-wins"));
    const tips = el("ul", "tips");
    EVERGREEN_QUICKWINS.forEach((t) => tips.appendChild(el("li", null, t)));
    fix.appendChild(tips);
  }
  card.appendChild(fix);

  return card;
}

function renderResults(data) {
  resultsEl.innerHTML = "";
  const candidates = data.candidates || [];

  const summary = el(
    "p",
    "results-summary",
    `${candidates.length} result(s) · mode: ${data.mode}` +
      (data.job_title ? ` · role: ${data.job_title}` : "")
  );
  resultsEl.appendChild(summary);

  candidates.forEach((c, i) => resultsEl.appendChild(renderCandidate(c, i, data.mode)));
}

async function handleSubmit(event) {
  event.preventDefault();
  resultsEl.innerHTML = "";

  const files = document.getElementById("resumes").files;
  if (!files || files.length === 0) {
    showStatus("Please choose at least one resume PDF.", true);
    return;
  }

  const formData = new FormData(form);
  if (files.length > 10) {
    showStatus("Choose at most 10 resumes per request.", true);
    return;
  }
  if (Array.from(files).reduce((total, file) => total + file.size, 0) > 4000000) {
    showStatus("Resume PDFs must total at most 4 MB. Choose fewer or smaller files.", true);
    return;
  }
  if (String(formData.get("jd") || "").length > 20000) {
    showStatus("Keep the job description under 20,000 characters.", true);
    return;
  }
  submitBtn.disabled = true;
  showStatus("Screening... this can take a few minutes.");

  try {
    const resp = await fetch("/screen", { method: "POST", body: formData });
    if (!resp.ok) {
      let detail = resp.status === 504
        ? "Screening timed out. Try fewer resumes or Fast critic mode."
        : resp.status === 413
        ? "Upload is too large. Choose fewer or smaller PDFs."
        : "Request failed (" + resp.status + ").";
      try {
        const err = await resp.json();
        if (err.detail) detail = typeof err.detail === "string"
          ? err.detail
          : JSON.stringify(err.detail);
      } catch (_) {
        /* ignore JSON parse errors */
      }
      showStatus(detail, true);
      return;
    }
    const data = await resp.json();
    clearStatus();
    renderResults(data);
  } catch (e) {
    showStatus("Network error: " + e.message, true);
  } finally {
    submitBtn.disabled = false;
  }
}

form.addEventListener("submit", handleSubmit);
