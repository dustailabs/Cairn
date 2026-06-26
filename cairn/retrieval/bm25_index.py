"""Lexical (keyword) retrieval leg, backed by BM25."""

from __future__ import annotations

import re
from dataclasses import dataclass

from rank_bm25 import BM25Okapi

from .store import Chunk

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


@dataclass(frozen=True)
class ScoredChunk:
    chunk: Chunk
    score: float


class BM25Index:
    """Thin wrapper around ``rank_bm25`` keyed by chunk id."""

    def __init__(self, chunks: list[Chunk]) -> None:
        self._chunks = chunks
        corpus = [tokenize(c.text) for c in chunks]
        # rank_bm25 errors on a fully empty corpus; guard the demo's
        # "no documents ingested yet" case explicitly.
        self._bm25 = BM25Okapi(corpus) if corpus else None

    def search(self, query: str, top_k: int = 10) -> list[ScoredChunk]:
        if self._bm25 is None or not self._chunks:
            return []
        scores = self._bm25.get_scores(tokenize(query))
        ranked = sorted(zip(self._chunks, scores), key=lambda pair: pair[1], reverse=True)
        return [ScoredChunk(chunk=c, score=float(s)) for c, s in ranked[:top_k] if s > 0]
