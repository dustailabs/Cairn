"""Structured trace events emitted at every pipeline step.

Same idea as Concord's agent trace: every step — retrieval, each crew
agent's turn, validation — emits a ``TraceEvent`` before and after it runs.
That's what feeds the "show your work" view in the API response and the
CLI demo, and it's what makes a RAG answer auditable instead of a black
box.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass(frozen=True)
class TraceEvent:
    step: str
    phase: str  # "start" | "end" | "error"
    detail: dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)


class Tracer:
    """Collects ``TraceEvent``s and optionally forwards them to a sink."""

    def __init__(self, on_event: Callable[[TraceEvent], None] | None = None) -> None:
        self._events: list[TraceEvent] = []
        self._on_event = on_event

    def emit(self, step: str, phase: str, **detail: Any) -> TraceEvent:
        event = TraceEvent(step=step, phase=phase, detail=detail)
        self._events.append(event)
        if self._on_event:
            self._on_event(event)
        return event

    @property
    def events(self) -> list[TraceEvent]:
        return list(self._events)

    def as_dicts(self) -> list[dict[str, Any]]:
        return [
            {"step": e.step, "phase": e.phase, "detail": e.detail, "timestamp": e.timestamp}
            for e in self._events
        ]
