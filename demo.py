#!/usr/bin/env python3
"""Cairn CLI demo.

Ingests a small sample corpus, then either:
  - runs the full Planner -> Researcher -> Critic crew (if ANTHROPIC_API_KEY
    is set), printing the grounded answer, citations, and trace; or
  - falls back to a retrieval-only preview (no API key required) so anyone
    can see hybrid retrieval working without spending a single token.

Usage:
    python demo.py "What does Cairn use for keyword retrieval?"
"""

from __future__ import annotations

import json
import os
import sys

from cairn.crew import ResearchCrew
from cairn.retrieval import DocumentStore, HashingEmbedder, HybridRetriever

SAMPLE_CORPUS = [
    (
        "cairn-readme",
        "Cairn combines BM25 keyword search with dense vector search, fusing the two "
        "ranked lists with Reciprocal Rank Fusion (RRF) rather than blending raw scores. "
        "This avoids the scale-mismatch problem between lexical and semantic scores.",
    ),
    (
        "cairn-crew",
        "Cairn's agentic layer is a three-agent CrewAI crew: a Planner that decomposes "
        "the question into sub-questions, a Researcher that calls the retrieval tool for "
        "each sub-question, and a Critic that writes the final answer using only "
        "retrieved passages, citing a chunk id for every claim.",
    ),
    (
        "cairn-validation",
        "Every citation the Critic produces is checked against the chunk ids that were "
        "actually retrieved during the run. Citations pointing at chunks that were never "
        "retrieved are dropped and logged in the trace as a hallucinated source, rather "
        "than silently passed through to the user.",
    ),
    (
        "cairn-embeddings",
        "Cairn ships two embedders behind the same interface: a HashingEmbedder for fast, "
        "dependency-free offline use in tests and demos, and a SentenceTransformerEmbedder "
        "for real semantic search in production.",
    ),
]


def build_demo_retriever() -> HybridRetriever:
    store = DocumentStore()
    for source, text in SAMPLE_CORPUS:
        store.add_document(text=text, source=source)
    return HybridRetriever(store.all_chunks(), HashingEmbedder())


def run_retrieval_only(retriever: HybridRetriever, query: str) -> None:
    print("ANTHROPIC_API_KEY not set — showing retrieval-only preview.\n")
    results = retriever.retrieve(query, top_k=5)
    if not results:
        print("No matching passages found.")
        return
    for rank, hit in enumerate(results, start=1):
        print(f"#{rank}  fused_score={hit.fused_score:.4f}  "
              f"bm25_rank={hit.bm25_rank}  vector_rank={hit.vector_rank}")
        print(f"   [{hit.chunk.id}] ({hit.chunk.source}) {hit.chunk.text}\n")


def run_full_crew(retriever: HybridRetriever, query: str) -> None:
    crew = ResearchCrew(retriever=retriever)
    result = crew.run(query)

    print(f"Answer: {result.answer}\n")
    print(f"Citations: {result.citations}")
    if result.dropped_citations:
        print(f"Dropped (hallucinated) citations: {result.dropped_citations}")
    print(f"Unresolved: {result.unresolved}\n")
    print("Trace:")
    print(json.dumps(result.trace, indent=2))


def main() -> None:
    query = sys.argv[1] if len(sys.argv) > 1 else "How does Cairn fuse keyword and vector search?"
    print(f"Query: {query}\n{'-' * 60}\n")

    retriever = build_demo_retriever()
    if os.environ.get("ANTHROPIC_API_KEY"):
        run_full_crew(retriever, query)
    else:
        run_retrieval_only(retriever, query)


if __name__ == "__main__":
    main()
