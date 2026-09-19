"""LangGraph orchestration wiring the agents into one screening pipeline.

Flow (per resume)::

    START
      → parse_resume        (agent, skipped if already structured)
      → parse_jd            (agent, skipped if already structured)
      → retrieve_evidence   (chunks resume + retrieves top-k evidence per
                              section via Gemini embeddings; skipped if
                              already computed, deterministic otherwise)
      → match               (agent; grounded in retrieved evidence, uses
                              critic feedback on retry)
      → score               (deterministic)
      → hygiene             (deterministic)
      → critic              (agent; sets needs_rescore deterministically)
      → [conditional]
            ├─ prepare_retry → match     (one self-correction loop; retrieval
                                           is not rebuilt on retry)
            └─ END

The single conditional edge (loop back to the matcher when the critic is
low-confidence, capped by ``config.max_retries``) is what makes this a genuine
LangGraph state machine rather than a straight-line function call.

Each node appends to ``state.trace`` by returning an extended list, so the audit
log accumulates without needing a custom channel reducer.
"""

from __future__ import annotations

from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from langgraph.graph import END, START, StateGraph

from multi_agent_resume_screener.agents.critic import critique
from multi_agent_resume_screener.agents.jd_parser import parse_jd
from multi_agent_resume_screener.agents.matcher import match
from multi_agent_resume_screener.agents.parser import parse_resume
from multi_agent_resume_screener.pipeline.hygiene import check_hygiene
from multi_agent_resume_screener.pipeline.retrieval import build_chunks, retrieve_evidence
from multi_agent_resume_screener.pipeline.scorer import score
from multi_agent_resume_screener.state import (
    JDRaw,
    PipelineConfig,
    PipelineState,
    ResumeRaw,
    TraceEntry,
)


def build_pipeline(llm: BaseChatModel | None = None, embedder: Embeddings | None = None):
    """Build and compile the screening pipeline graph.

    Args:
        llm: Optional chat model injected into every agent (used by tests to run
            the graph offline). When ``None``, each agent resolves the configured
            provider itself.
        embedder: Optional embeddings client injected into the retrieval node
            (used by tests to run retrieval offline). When ``None``, it resolves
            the configured Gemini embedder itself.

    Returns:
        A compiled LangGraph runnable. Invoke it with a :class:`PipelineState`
        (or an equivalent dict) and it returns the final state.
    """

    def _trace(state: PipelineState, agent: str, note: str = "") -> list[TraceEntry]:
        return state.trace + [TraceEntry(agent=agent, note=note)]

    def parse_resume_node(state: PipelineState) -> dict:
        if state.resume_structured is not None:
            return {}
        structured = parse_resume(state.resume_raw, llm=llm)
        return {
            "resume_structured": structured,
            "trace": _trace(state, "parser", f"{len(structured.skills)} skills"),
        }

    def parse_jd_node(state: PipelineState) -> dict:
        if state.jd_structured is not None:
            return {}
        structured = parse_jd(state.jd_raw, llm=llm)
        return {
            "jd_structured": structured,
            "trace": _trace(state, "jd_parser", structured.title or ""),
        }

    def retrieve_evidence_node(state: PipelineState) -> dict:
        if state.resume_chunks:
            return {}
        chunks = build_chunks(state.resume_structured)
        evidence = retrieve_evidence(
            chunks, state.jd_structured, embedder=embedder, top_k=state.config.retrieval_top_k
        )
        hits = sum(len(v) for v in evidence.values())
        return {
            "resume_chunks": chunks,
            "retrieved_evidence": evidence,
            "trace": _trace(state, "retriever", f"{len(chunks)} chunks, {hits} retrieved"),
        }

    def match_node(state: PipelineState) -> dict:
        # On a retry pass, feed the critic's suggestions back to the matcher.
        feedback = None
        if state.retry_count > 0 and state.critique is not None:
            feedback = state.critique.suggestions
        result = match(
            state.resume_structured,
            state.jd_structured,
            llm=llm,
            feedback=feedback,
            retrieved_evidence=state.retrieved_evidence,
        )
        note = f"retry={state.retry_count}" if feedback else "initial"
        return {"match_result": result, "trace": _trace(state, "matcher", note)}

    def score_node(state: PipelineState) -> dict:
        final = score(state.match_result, state.config)
        return {
            "final_score": final,
            "trace": _trace(state, "scorer", f"score={final.score:.2f}"),
        }

    def hygiene_node(state: PipelineState) -> dict:
        report = check_hygiene(state.resume_structured, state.resume_raw.text)
        return {
            "hygiene": report,
            "trace": _trace(state, "hygiene", f"{len(report.issues)} issues"),
        }

    def critic_node(state: PipelineState) -> dict:
        result = critique(
            state.resume_structured,
            state.jd_structured,
            state.match_result,
            state.final_score,
            state.config,
            hygiene=state.hygiene,
            llm=llm,
        )
        note = f"{result.verdict}, rescore={result.needs_rescore}"
        return {"critique": result, "trace": _trace(state, "critic", note)}

    def prepare_retry_node(state: PipelineState) -> dict:
        return {
            "retry_count": state.retry_count + 1,
            "trace": _trace(state, "self_correction", "looping back to matcher"),
        }

    def route_after_critic(state: PipelineState) -> str:
        if (
            state.critique is not None
            and state.critique.needs_rescore
            and state.can_retry()
        ):
            return "retry"
        return "end"

    def route_after_hygiene(state: PipelineState) -> str:
        # Fast mode (enable_critic=False) ends after scoring + hygiene; the
        # critic is run separately on only the top-K ranked candidates.
        return "critic" if state.config.enable_critic else "end"

    graph = StateGraph(PipelineState)
    graph.add_node("parse_resume", parse_resume_node)
    graph.add_node("parse_jd", parse_jd_node)
    graph.add_node("retrieve_evidence", retrieve_evidence_node)
    graph.add_node("match", match_node)
    graph.add_node("score", score_node)
    graph.add_node("check_hygiene", hygiene_node)
    graph.add_node("critic", critic_node)
    graph.add_node("prepare_retry", prepare_retry_node)

    graph.add_edge(START, "parse_resume")
    graph.add_edge("parse_resume", "parse_jd")
    graph.add_edge("parse_jd", "retrieve_evidence")
    graph.add_edge("retrieve_evidence", "match")
    graph.add_edge("match", "score")
    graph.add_edge("score", "check_hygiene")
    graph.add_conditional_edges(
        "check_hygiene",
        route_after_hygiene,
        {"critic": "critic", "end": END},
    )
    graph.add_conditional_edges(
        "critic",
        route_after_critic,
        {"retry": "prepare_retry", "end": END},
    )
    graph.add_edge("prepare_retry", "match")

    return graph.compile()


def run_pipeline(
    resume_raw: ResumeRaw,
    jd_raw: JDRaw,
    config: PipelineConfig | None = None,
    llm: BaseChatModel | None = None,
    embedder: Embeddings | None = None,
) -> PipelineState:
    """Run one resume through the full pipeline and return the final state.

    Args:
        resume_raw: Resume filename + extracted text.
        jd_raw: Raw job-description text.
        config: Optional pipeline config (mode, weights, retries, ...).
        llm: Optional chat model (dependency injection for tests).
        embedder: Optional embeddings client (dependency injection for tests).

    Returns:
        The final :class:`PipelineState` with structured data, scores, hygiene,
        critique, and the accumulated trace.
    """
    pipeline = build_pipeline(llm=llm, embedder=embedder)
    initial = PipelineState(
        resume_raw=resume_raw,
        jd_raw=jd_raw,
        config=config or PipelineConfig(),
    )
    result = pipeline.invoke(initial)
    # LangGraph may return a dict-like; normalise back to a PipelineState.
    if isinstance(result, PipelineState):
        return result
    return PipelineState.model_validate(result)
