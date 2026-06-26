"""``ResearchCrew``: the top-level orchestrator for Cairn's agentic RAG pipeline.

Wires together the hybrid retriever, the Planner/Researcher/Critic crew,
output parsing, citation validation, and tracing into one ``run()`` call.
"""

from __future__ import annotations

from dataclasses import dataclass

from cairn.retrieval import HybridRetriever
from cairn.trace import Tracer

from .agents import build_agents
from .llm import build_llm
from .parsing import CrewAnswer, CrewOutputError, parse_critic_output, validate_citations
from .tools import build_retrieval_tool


@dataclass(frozen=True)
class CrewResult:
    answer: str
    citations: list[str]
    dropped_citations: list[str]
    unresolved: bool
    trace: list[dict]


class ResearchCrew:
    """Runs the Planner -> Researcher -> Critic crew against a query."""

    def __init__(self, retriever: HybridRetriever, llm=None, top_k: int = 5) -> None:
        self._retriever = retriever
        self._llm = llm
        self._top_k = top_k

    def run(self, query: str) -> CrewResult:
        from crewai import Crew, Process  # type: ignore

        tracer = Tracer()
        llm = self._llm or build_llm()
        retrieval_tool = build_retrieval_tool(self._retriever, tracer, top_k=self._top_k)
        agents = build_agents(llm, retrieval_tool)

        from .tasks import build_tasks

        tasks = build_tasks(agents, query)
        crew = Crew(agents=list(agents.values()), tasks=tasks, process=Process.sequential, verbose=False)

        tracer.emit("crew", "start", query=query)
        raw_output = str(crew.kickoff())
        tracer.emit("crew", "end")

        return self._finalize(raw_output, tracer)

    def _finalize(self, raw_output: str, tracer: Tracer) -> CrewResult:
        """Parse + validate the crew's raw output. Split out so the
        integration test can exercise this path with a canned string
        instead of a real ``crew.kickoff()`` call."""
        valid_chunk_ids = self._retriever.chunk_ids

        tracer.emit("validation", "start")
        try:
            parsed: CrewAnswer = parse_critic_output(raw_output)
        except CrewOutputError as exc:
            tracer.emit("validation", "error", reason=str(exc))
            return CrewResult(
                answer="The crew's output couldn't be parsed, so no grounded answer is available.",
                citations=[],
                dropped_citations=[],
                unresolved=True,
                trace=tracer.as_dicts(),
            )

        valid, dropped = validate_citations(parsed.citations, valid_chunk_ids)
        tracer.emit("validation", "end", valid_citations=valid, dropped_citations=dropped)

        return CrewResult(
            answer=parsed.answer,
            citations=valid,
            dropped_citations=dropped,
            unresolved=parsed.unresolved or bool(dropped) and not valid,
            trace=tracer.as_dicts(),
        )
