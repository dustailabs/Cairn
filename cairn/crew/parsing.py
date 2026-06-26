"""Pure, framework-free helpers for parsing and validating crew output.

Deliberately kept free of any ``crewai`` import. LLM output is unreliable
in exactly the same ways Concord's router output is — sometimes wrapped in
markdown code fences, sometimes with trailing commentary — so this module
is the equivalent of Concord's "classification parsing, including
malformed and markdown-fenced model output" tests.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)


class CrewOutputError(ValueError):
    """Raised when the crew's final output can't be parsed or validated."""


@dataclass(frozen=True)
class CrewAnswer:
    answer: str
    citations: list[str]
    unresolved: bool


def strip_code_fence(raw: str) -> str:
    """Pull JSON out of a ```json ... ``` fence if present, else return as-is."""
    match = _FENCE_RE.search(raw)
    return match.group(1).strip() if match else raw.strip()


def parse_critic_output(raw: str) -> CrewAnswer:
    """Parse the Critic agent's final JSON payload into a ``CrewAnswer``.

    Raises ``CrewOutputError`` on anything malformed rather than silently
    returning a half-populated answer — a RAG system that can't ground its
    answer should say so, not guess.
    """
    cleaned = strip_code_fence(raw)
    try:
        payload = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        raise CrewOutputError(f"Critic output was not valid JSON: {exc}") from exc

    if "answer" not in payload:
        raise CrewOutputError("Critic output missing required 'answer' field")

    citations = payload.get("citations", [])
    if not isinstance(citations, list):
        raise CrewOutputError("Critic output 'citations' must be a list")

    return CrewAnswer(
        answer=str(payload["answer"]),
        citations=[str(c) for c in citations],
        unresolved=bool(payload.get("unresolved", False)),
    )


def validate_citations(citations: list[str], valid_chunk_ids: set[str]) -> tuple[list[str], list[str]]:
    """Split citations into (valid, dropped).

    A citation pointing at a chunk id that was never actually retrieved is
    worse than no citation — it's a hallucinated source. We drop it and
    surface that fact in the trace rather than let it through silently.
    """
    valid = [c for c in citations if c in valid_chunk_ids]
    dropped = [c for c in citations if c not in valid_chunk_ids]
    return valid, dropped


def format_retrieved_context(chunks: list[tuple[str, str, str]]) -> str:
    """Render retrieved chunks as the context block handed to agents.

    Each tuple is ``(chunk_id, source, text)``. Format is intentionally
    plain and explicit about chunk ids, since the Critic must cite them
    verbatim.
    """
    lines = []
    for chunk_id, source, text in chunks:
        lines.append(f"[{chunk_id}] (source: {source})\n{text}")
    return "\n\n".join(lines)
