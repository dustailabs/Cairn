"""Document and chunk models, plus a small in-memory store.

Cairn is deliberately storage-agnostic at the core: ``DocumentStore`` is an
in-memory reference implementation good enough for the demo and for tests.
Swapping in Postgres/pgvector, Qdrant, or Elasticsearch means implementing
the same three methods against a real backend — nothing else in the
pipeline needs to change.
"""

from __future__ import annotations

import re
import uuid
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Chunk:
    """A single retrievable unit of text."""

    id: str
    doc_id: str
    source: str
    text: str
    position: int


@dataclass
class Document:
    id: str
    source: str
    text: str
    chunks: list[Chunk] = field(default_factory=list)


def chunk_text(text: str, max_chars: int = 800, overlap: int = 120) -> list[str]:
    """Split text into overlapping chunks on paragraph/sentence boundaries.

    A simple sliding window over sentence-aware splits. Good enough for
    the demo corpus sizes Cairn targets; swap in a token-aware splitter
    for production workloads with long, dense documents.
    """
    if max_chars <= overlap:
        raise ValueError("max_chars must be greater than overlap")

    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    chunks: list[str] = []
    current = ""

    for sentence in sentences:
        if not sentence:
            continue
        candidate = f"{current} {sentence}".strip() if current else sentence
        if len(candidate) <= max_chars:
            current = candidate
            continue

        if current:
            chunks.append(current)
        # carry the tail of the previous chunk forward for context overlap
        tail = current[-overlap:] if current else ""
        current = f"{tail} {sentence}".strip()

    if current:
        chunks.append(current)

    return chunks or [text.strip()]


class DocumentStore:
    """In-memory store for documents and their chunks."""

    def __init__(self) -> None:
        self._documents: dict[str, Document] = {}
        self._chunks: dict[str, Chunk] = {}

    def add_document(self, text: str, source: str, doc_id: str | None = None) -> Document:
        doc_id = doc_id or str(uuid.uuid4())
        pieces = chunk_text(text)
        chunks = [
            Chunk(id=f"{doc_id}::{i}", doc_id=doc_id, source=source, text=piece, position=i)
            for i, piece in enumerate(pieces)
        ]
        document = Document(id=doc_id, source=source, text=text, chunks=chunks)
        self._documents[doc_id] = document
        for chunk in chunks:
            self._chunks[chunk.id] = chunk
        return document

    def all_chunks(self) -> list[Chunk]:
        return list(self._chunks.values())

    def get_chunk(self, chunk_id: str) -> Chunk:
        return self._chunks[chunk_id]

    def __len__(self) -> int:
        return len(self._documents)

    @property
    def chunk_count(self) -> int:
        return len(self._chunks)
