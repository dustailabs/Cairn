"""FastAPI app exposing Cairn's ingest + query pipeline.

Kept deliberately small: app state holds one in-memory ``DocumentStore``
and rebuilds the ``HybridRetriever`` on each query against the current
chunk set. Fine for a demo/single-tenant deployment; a multi-tenant
production deployment would key the store/retriever by workspace and
persist chunks externally.
"""

from __future__ import annotations

import os

from fastapi import FastAPI, HTTPException

from cairn.crew import ResearchCrew
from cairn.retrieval import DocumentStore, HashingEmbedder, HybridRetriever

from .schemas import IngestRequest, IngestResponse, QueryRequest, QueryResponse

app = FastAPI(
    title="Cairn",
    description="Agentic RAG with hybrid retrieval, a Planner/Researcher/Critic crew, and a citation-checked answer.",
    version="0.1.0",
)

_store = DocumentStore()
_embedder = HashingEmbedder()  # swap for SentenceTransformerEmbedder() in production


def _build_retriever() -> HybridRetriever:
    return HybridRetriever(_store.all_chunks(), _embedder)


@app.post("/ingest", response_model=IngestResponse)
def ingest(payload: IngestRequest) -> IngestResponse:
    document = _store.add_document(text=payload.text, source=payload.source)
    return IngestResponse(
        doc_id=document.id,
        chunk_count=len(document.chunks),
        total_chunks=_store.chunk_count,
    )


@app.post("/query", response_model=QueryResponse)
def query(payload: QueryRequest) -> QueryResponse:
    if _store.chunk_count == 0:
        raise HTTPException(status_code=400, detail="No documents have been ingested yet.")

    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise HTTPException(
            status_code=503,
            detail="ANTHROPIC_API_KEY is not set, so the crew can't run. "
            "Set it in your environment, or use ResearchCrew directly with a fake LLM for testing.",
        )

    retriever = _build_retriever()
    crew = ResearchCrew(retriever=retriever, top_k=payload.top_k)
    result = crew.run(payload.query)

    return QueryResponse(
        answer=result.answer,
        citations=result.citations,
        dropped_citations=result.dropped_citations,
        unresolved=result.unresolved,
        trace=result.trace,
    )


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "documents": len(_store), "chunks": _store.chunk_count}
