"""Wires CrewAI's LLM abstraction to Anthropic.

Kept as its own tiny module so the rest of the crew package doesn't care
which provider is behind ``crewai.LLM`` — swap this one function and
everything upstream (agents, tasks, orchestration) is unaffected.
"""

from __future__ import annotations

import os

DEFAULT_MODEL = "claude-sonnet-4-6"


def build_llm(model: str | None = None):
    """Build a ``crewai.LLM`` configured for Anthropic.

    Lazily imports ``crewai`` so importing this module doesn't force the
    dependency on callers who only want the retrieval layer or the pure
    parsing helpers in :mod:`cairn.crew.parsing`.
    """
    from crewai import LLM  # type: ignore

    resolved_model = model or os.environ.get("CAIRN_MODEL", DEFAULT_MODEL)
    return LLM(model=f"anthropic/{resolved_model}")
