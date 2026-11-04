from src.ingestion.loader import Document
from src.chunking.chunker import chunk_document, chunk_documents


def doc(text="First paragraph.\\n\\nSecond paragraph.", sid="s1"):
    return Document(text=text, metadata={"source_id": sid, "title": "T", "url": "https://groww.in/x", "content_hash": "hash"})


def test_short_document_one_chunk():
    chunks = chunk_document(doc("Short text"))
    assert len(chunks) == 1 and chunks[0].text == "Short text"


def test_chunks_respect_size_for_long_blocks():
    text = "word " * 500
    chunks = chunk_document(doc(text), chunk_size=100, chunk_overlap=10)
    assert len(chunks) > 1
    assert all(len(c.text) <= 100 for c in chunks)


def test_metadata_preserved_and_index_added():
    chunks = chunk_document(doc("A\\n\\nB\\n\\nC"), chunk_size=3, chunk_overlap=0)
    assert chunks[0].metadata["source_id"] == "s1"
    assert chunks[0].metadata["chunk_index"] == 0
    assert chunks[0].metadata["url"].startswith("https://")


def test_empty_document_returns_no_chunks():
    assert chunk_document(doc("  ")) == []


def test_invalid_chunk_settings():
    import pytest
    with pytest.raises(ValueError):
        chunk_document(doc(), chunk_size=0)
    with pytest.raises(ValueError):
        chunk_document(doc(), chunk_size=10, chunk_overlap=10)


def test_chunk_multiple_documents():
    result = chunk_documents([doc("One", "a"), doc("Two", "b")])
    assert len(result) == 2
    assert result[0].metadata["source_id"] != result[1].metadata["source_id"]
