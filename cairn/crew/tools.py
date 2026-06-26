"""CrewAI tool wrapping :class:`cairn.retrieval.HybridRetriever`.

This is the only point of contact between the crew and the retrieval
layer: the Researcher agent calls this tool, never the index directly,
so every lookup it makes is automatically traced and chunk-id-tagged.
"""

from __future__ import annotations

from cairn.retrieval import HybridRetriever
from cairn.trace import Tracer

from .parsing import format_retrieved_context


def build_retrieval_tool(retriever: HybridRetriever, tracer: Tracer, top_k: int = 5):
    """Build a CrewAI tool bound to a specific retriever + tracer.

    Built as a closure-based ``@tool`` rather than a stateful ``BaseTool``
    subclass so each crew run gets a tool bound to *that* run's tracer —
    CrewAI tool instances are otherwise stateless by convention.
    """
    from crewai.tools import tool  # type: ignore

    @tool("Retrieve grounded context")
    def retrieve_context(query: str) -> str:
        """Search the ingested knowledge base for passages relevant to a
        sub-question. Returns passages tagged with their chunk id and
        source — cite the chunk id exactly when you use a passage."""
        tracer.emit("retrieval", "start", query=query)
        results = retriever.retrieve(query, top_k=top_k)
        tracer.emit(
            "retrieval",
            "end",
            query=query,
            hit_count=len(results),
            chunk_ids=[r.chunk.id for r in results],
        )
        if not results:
            return "No relevant passages were found for this query."
        triples = [(r.chunk.id, r.chunk.source, r.chunk.text) for r in results]
        return format_retrieved_context(triples)

    return retrieve_context
