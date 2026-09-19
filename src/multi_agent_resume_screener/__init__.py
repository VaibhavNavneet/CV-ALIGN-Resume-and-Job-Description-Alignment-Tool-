"""multi-agent-resume-screener: a multi-agent resume screening pipeline.

Agents (parser, jd_parser, matcher, critic) are orchestrated with LangGraph
around a shared :class:`multi_agent_resume_screener.state.PipelineState`. Scoring is a
deterministic function so the final number is always reproducible and auditable.
"""

__version__ = "0.1.0"
