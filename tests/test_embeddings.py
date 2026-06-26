import numpy as np

from cairn.retrieval.embeddings import HashingEmbedder, cosine_sim_matrix
from cairn.retrieval.hybrid import VectorIndex
from cairn.retrieval.store import DocumentStore


def test_hashing_embedder_is_deterministic():
    embedder = HashingEmbedder(dim=64)
    a = embedder.embed(["the quick brown fox"])
    b = embedder.embed(["the quick brown fox"])
    assert np.allclose(a, b)


def test_hashing_embedder_distinguishes_different_text():
    embedder = HashingEmbedder(dim=64)
    a = embedder.embed(["kafka event streaming"])[0]
    b = embedder.embed(["banana potassium fruit"])[0]
    sim = cosine_sim_matrix(a, np.array([b]))[0]
    assert sim < 0.9  # not identical topics, shouldn't look near-identical


def test_hashing_embedder_empty_input():
    embedder = HashingEmbedder(dim=64)
    out = embedder.embed([])
    assert out.shape == (0, 64)


def test_vector_index_ranks_semantically_closer_text_higher():
    store = DocumentStore()
    store.add_document(text="Kafka topics, partitions, and consumer groups.", source="a")
    store.add_document(text="A recipe for baking sourdough bread.", source="b")

    index = VectorIndex(store.all_chunks(), HashingEmbedder(dim=128))
    results = index.search("Kafka consumer group rebalancing", top_k=2)

    assert results
    assert "Kafka" in results[0].chunk.text


def test_vector_index_empty_corpus():
    index = VectorIndex([], HashingEmbedder(dim=32))
    assert index.search("anything") == []
