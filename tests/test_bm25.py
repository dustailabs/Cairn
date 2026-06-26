from cairn.retrieval.bm25_index import BM25Index, tokenize
from cairn.retrieval.store import DocumentStore


def test_tokenize_lowercases_and_strips_punctuation():
    assert tokenize("Hello, World! 123") == ["hello", "world", "123"]


def test_bm25_ranks_exact_keyword_match_higher():
    store = DocumentStore()
    store.add_document(text="Kafka is a distributed event streaming platform.", source="a")
    store.add_document(text="Bananas are a good source of potassium.", source="b")
    store.add_document(text="The weather today is mild with light wind.", source="c")

    index = BM25Index(store.all_chunks())
    results = index.search("Kafka streaming", top_k=5)

    assert results, "expected at least one hit"
    assert "Kafka" in results[0].chunk.text


def test_bm25_empty_index_returns_no_results():
    index = BM25Index([])
    assert index.search("anything") == []


def test_bm25_no_match_returns_empty():
    store = DocumentStore()
    store.add_document(text="Completely unrelated content about gardening.", source="a")
    index = BM25Index(store.all_chunks())
    assert index.search("quantum chromodynamics") == []
