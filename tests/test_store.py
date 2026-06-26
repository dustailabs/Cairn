from cairn.retrieval.store import DocumentStore, chunk_text


def test_chunk_text_respects_max_chars():
    text = "Sentence one. Sentence two. Sentence three. " * 20
    chunks = chunk_text(text, max_chars=100, overlap=20)
    assert len(chunks) > 1
    # allow a little slack for the carried-over overlap tail
    assert all(len(c) <= 140 for c in chunks)


def test_chunk_text_short_input_returns_single_chunk():
    chunks = chunk_text("Just one short sentence.", max_chars=800, overlap=120)
    assert chunks == ["Just one short sentence."]


def test_document_store_add_and_retrieve():
    store = DocumentStore()
    doc = store.add_document(text="A. B. C. " * 50, source="unit-test")
    assert len(store) == 1
    assert store.chunk_count == len(doc.chunks)
    first_chunk = doc.chunks[0]
    assert store.get_chunk(first_chunk.id) == first_chunk


def test_document_store_multiple_documents_have_distinct_chunk_ids():
    store = DocumentStore()
    store.add_document(text="Document one content.", source="a")
    store.add_document(text="Document two content.", source="b")
    chunk_ids = {c.id for c in store.all_chunks()}
    assert len(chunk_ids) == store.chunk_count
