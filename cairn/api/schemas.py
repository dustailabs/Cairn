from __future__ import annotations

from pydantic import BaseModel, Field


class IngestRequest(BaseModel):
    text: str = Field(..., min_length=1)
    source: str = Field(..., min_length=1)


class IngestResponse(BaseModel):
    doc_id: str
    chunk_count: int
    total_chunks: int


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1)
    top_k: int = Field(default=5, ge=1, le=20)


class TraceEventOut(BaseModel):
    step: str
    phase: str
    detail: dict
    timestamp: float


class QueryResponse(BaseModel):
    answer: str
    citations: list[str]
    dropped_citations: list[str]
    unresolved: bool
    trace: list[TraceEventOut]
