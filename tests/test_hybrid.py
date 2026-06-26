from cairn.retrieval.bm25_index import ScoredChunk
from cairn.retrieval.hybrid import HybridRetriever, reciprocal_rank_fusion
from cairn.retrieval.embeddings import HashingEmbedder
from cairn.retrieval.store import Chunk, DocumentStore


def _chunk(cid: str) -> Chunk:
    return Chunk(id=cid, doc_id="doc", source="src", text=f"text {cid}", position=0)


def test_rrf_favors_chunk_ranked_top_in_both_lists():
    a, b, c = _chunk("a"), _chunk("b"), _chunk("c")
    bm25_results = [ScoredChunk(chunk=a, score=10.0), ScoredChunk(chunk=b, score=5.0)]
    vector_results = [ScoredChunk(chunk=a, score=0.9), ScoredChunk(chunk=c, score=0.8)]

    fused = reciprocal_rank_fusion(bm25_results, vector_results)

    assert fused[0].chunk.id == "a"  # ranked #1 in both lists
    assert {f.chunk.id for f in fused} == {"a", "b", "c"}


def test_rrf_tracks_rank_provenance():
    a = _chunk("a")
    fused = reciprocal_rank_fusion([ScoredChunk(chunk=a, score=1.0)], [])
    assert fused[0].bm25_rank == 0
    assert fused[0].vector_rank is None


def test_rrf_empty_inputs():
    assert reciprocal_rank_fusion([], []) == []


def test_hybrid_retriever_end_to_end():
    store = DocumentStore()
    store.add_document(text="Kafka brokers replicate partitions across the cluster.", source="kafka-doc")
    store.add_document(text="A sourdough starter needs daily feeding.", source="baking-doc")

    retriever = HybridRetriever(store.all_chunks(), HashingEmbedder())
    results = retriever.retrieve("How does Kafka replicate partitions?", top_k=3)

    assert results
    assert "Kafka" in results[0].chunk.text
    assert retriever.chunk_ids == {c.id for c in store.all_chunks()}
