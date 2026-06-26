from cairn.crew.crew import ResearchCrew
from cairn.retrieval import DocumentStore, HashingEmbedder, HybridRetriever
from cairn.trace import Tracer


def _build_retriever():
    store = DocumentStore()
    store.add_document(text="Kafka brokers replicate partitions across the cluster.", source="kafka-doc")
    return HybridRetriever(store.all_chunks(), HashingEmbedder()), store


def test_finalize_accepts_valid_citation():
    retriever, store = _build_retriever()
    crew = ResearchCrew(retriever=retriever)
    real_chunk_id = store.all_chunks()[0].id

    raw = f'{{"answer": "Kafka replicates partitions.", "citations": ["{real_chunk_id}"], "unresolved": false}}'
    result = crew._finalize(raw, Tracer())  # noqa: SLF001 — exercising the finalize path directly

    assert result.answer == "Kafka replicates partitions."
    assert result.citations == [real_chunk_id]
    assert result.dropped_citations == []
    assert result.unresolved is False


def test_finalize_drops_hallucinated_citation():
    retriever, _ = _build_retriever()
    crew = ResearchCrew(retriever=retriever)

    raw = '{"answer": "Some claim.", "citations": ["never-retrieved::0"], "unresolved": false}'
    result = crew._finalize(raw, Tracer())  # noqa: SLF001

    assert result.citations == []
    assert result.dropped_citations == ["never-retrieved::0"]
    assert result.unresolved is True  # no valid citation survived, so it's marked unresolved


def test_finalize_handles_unparseable_output_gracefully():
    retriever, _ = _build_retriever()
    crew = ResearchCrew(retriever=retriever)

    result = crew._finalize("not json at all", Tracer())  # noqa: SLF001

    assert result.unresolved is True
    assert result.citations == []
    assert any(e["phase"] == "error" for e in result.trace)


def test_finalize_emits_validation_trace_events():
    retriever, store = _build_retriever()
    crew = ResearchCrew(retriever=retriever)
    real_chunk_id = store.all_chunks()[0].id
    tracer = Tracer()

    raw = f'{{"answer": "x", "citations": ["{real_chunk_id}"]}}'
    crew._finalize(raw, tracer)  # noqa: SLF001

    steps = [e.step for e in tracer.events]
    assert "validation" in steps
