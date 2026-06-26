"""Dense (vector) retrieval leg, and the hybrid fusion that combines it
with BM25.

The fusion strategy is Reciprocal Rank Fusion (RRF) — it combines two
ranked lists by their *rank position* rather than raw scores, which sidesteps
the usual headache of BM25 and cosine-similarity scores living on
incomparable scales. It's the same technique Elasticsearch and several
production hybrid-search systems use for exactly this reason.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .bm25_index import BM25Index, ScoredChunk, tokenize  # noqa: F401  (tokenize re-exported)
from .embeddings import Embedder, cosine_sim_matrix
from .store import Chunk


@dataclass(frozen=True)
class RetrievedChunk:
    chunk: Chunk
    fused_score: float
    bm25_rank: int | None
    vector_rank: int | None


class VectorIndex:
    """Brute-force cosine-similarity index over an Embedder's vectors.

    Fine for the corpus sizes Cairn's demo targets (hundreds to low
    thousands of chunks). For larger corpora, swap in an ANN index
    (FAISS, HNSWlib, pgvector) behind the same ``search`` signature.
    """

    def __init__(self, chunks: list[Chunk], embedder: Embedder) -> None:
        self._chunks = chunks
        self._embedder = embedder
        self._matrix = embedder.embed([c.text for c in chunks]) if chunks else np.zeros((0, embedder.dim))

    def search(self, query: str, top_k: int = 10) -> list[ScoredChunk]:
        if not self._chunks:
            return []
        query_vec = self._embedder.embed([query])[0]
        sims = cosine_sim_matrix(query_vec, self._matrix)
        ranked = sorted(zip(self._chunks, sims), key=lambda pair: pair[1], reverse=True)
        return [ScoredChunk(chunk=c, score=float(s)) for c, s in ranked[:top_k]]


def reciprocal_rank_fusion(
    bm25_results: list[ScoredChunk],
    vector_results: list[ScoredChunk],
    k: int = 60,
) -> list[RetrievedChunk]:
    """Merge two ranked lists into one fused ranking.

    ``k`` is RRF's standard smoothing constant (60 is the commonly cited
    default from the original RRF paper) — it dampens the influence of a
    chunk's exact rank so the fusion isn't dominated by whichever list
    happens to rank one chunk #1 vs #2.
    """
    bm25_rank = {sc.chunk.id: i for i, sc in enumerate(bm25_results)}
    vector_rank = {sc.chunk.id: i for i, sc in enumerate(vector_results)}
    chunks_by_id = {sc.chunk.id: sc.chunk for sc in [*bm25_results, *vector_results]}

    fused_scores: dict[str, float] = {}
    for chunk_id in chunks_by_id:
        score = 0.0
        if chunk_id in bm25_rank:
            score += 1.0 / (k + bm25_rank[chunk_id] + 1)
        if chunk_id in vector_rank:
            score += 1.0 / (k + vector_rank[chunk_id] + 1)
        fused_scores[chunk_id] = score

    ordered_ids = sorted(fused_scores, key=lambda cid: fused_scores[cid], reverse=True)
    return [
        RetrievedChunk(
            chunk=chunks_by_id[cid],
            fused_score=fused_scores[cid],
            bm25_rank=bm25_rank.get(cid),
            vector_rank=vector_rank.get(cid),
        )
        for cid in ordered_ids
    ]


class HybridRetriever:
    """Combines :class:`BM25Index` and :class:`VectorIndex` via RRF."""

    def __init__(self, chunks: list[Chunk], embedder: Embedder) -> None:
        self._chunks = chunks
        self.bm25 = BM25Index(chunks)
        self.vector = VectorIndex(chunks, embedder)

    def retrieve(self, query: str, top_k: int = 5, candidate_k: int = 20) -> list[RetrievedChunk]:
        bm25_hits = self.bm25.search(query, top_k=candidate_k)
        vector_hits = self.vector.search(query, top_k=candidate_k)
        fused = reciprocal_rank_fusion(bm25_hits, vector_hits)
        return fused[:top_k]

    @property
    def chunk_ids(self) -> set[str]:
        """All chunk ids known to this retriever, for citation validation."""
        return {c.id for c in self._chunks}
