"""Embedding backends for Cairn's dense retrieval leg.

Two implementations ship out of the box:

- ``SentenceTransformerEmbedder``: real semantic embeddings
  (``sentence-transformers``, lazy-imported so the package stays optional).
- ``HashingEmbedder``: a fast, dependency-free, deterministic embedder used
  in tests and the offline demo so CI never needs to download a model or
  hit the network.

Both satisfy the ``Embedder`` protocol, so the rest of the pipeline never
cares which one is wired in.
"""

from __future__ import annotations

import hashlib
import re
from typing import Protocol, Sequence

import numpy as np

_TOKEN_RE = re.compile(r"[a-z0-9]+")


class Embedder(Protocol):
    """Anything that turns text into a fixed-size float vector."""

    dim: int

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        """Return an (n, dim) float32 matrix, one row per input text."""
        ...


class HashingEmbedder:
    """Deterministic bag-of-hashed-tokens embedder.

    Not state-of-the-art, but it is stable, instantaneous, and needs no
    model weights — ideal for unit tests, CI, and the offline demo where
    we want predictable retrieval scores without a network call.
    """

    def __init__(self, dim: int = 256) -> None:
        self.dim = dim

    def _vector(self, text: str) -> np.ndarray:
        vec = np.zeros(self.dim, dtype=np.float32)
        tokens = _TOKEN_RE.findall(text.lower())
        for tok in tokens:
            digest = hashlib.sha256(tok.encode("utf-8")).digest()
            idx = int.from_bytes(digest[:4], "big") % self.dim
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vec[idx] += sign
        norm = float(np.linalg.norm(vec))
        if norm > 0:
            vec /= norm
        return vec

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        return np.stack([self._vector(t) for t in texts]) if texts else np.zeros((0, self.dim), dtype=np.float32)


class SentenceTransformerEmbedder:
    """Real semantic embeddings via ``sentence-transformers``.

    Imported lazily so installing/running Cairn's core logic and tests
    never requires pulling down a transformer model.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore
        except ImportError as exc:  # pragma: no cover - exercised only without extras
            raise ImportError(
                "SentenceTransformerEmbedder requires the 'semantic' extra: "
                "pip install 'cairn[semantic]'"
            ) from exc

        self._model = SentenceTransformer(model_name)
        self.dim = self._model.get_sentence_embedding_dimension()

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        vectors = self._model.encode(list(texts), normalize_embeddings=True)
        return np.asarray(vectors, dtype=np.float32)


def cosine_sim_matrix(query_vec: np.ndarray, doc_vecs: np.ndarray) -> np.ndarray:
    """Cosine similarity between one query vector and many doc vectors.

    Vectors from both embedders above are already L2-normalized, but we
    guard against zero vectors defensively rather than assuming that.
    """
    if doc_vecs.shape[0] == 0:
        return np.zeros((0,), dtype=np.float32)
    q_norm = float(np.linalg.norm(query_vec)) or 1.0
    d_norms = np.linalg.norm(doc_vecs, axis=1)
    d_norms[d_norms == 0] = 1.0
    sims = (doc_vecs @ query_vec) / (d_norms * q_norm)
    return sims


def is_finite_vector(vec: np.ndarray) -> bool:
    return bool(np.all(np.isfinite(vec))) if vec.size else True


__all__ = [
    "Embedder",
    "HashingEmbedder",
    "SentenceTransformerEmbedder",
    "cosine_sim_matrix",
    "is_finite_vector",
]
