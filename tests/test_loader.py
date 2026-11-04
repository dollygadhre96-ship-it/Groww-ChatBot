import json
from pathlib import Path

from src.ingestion.loader import Document, load_documents, normalize_text


def sample(source_id="s1", text="  hello\r\n\r\n\r\nworld  ", content_hash="h1"):
    return {
        "source_id": source_id, "title": "Title", "url": "https://groww.in/page",
        "retrieved_at": "2026-01-01T00:00:00Z", "content_hash": content_hash,
        "text": text,
    }


def test_normalize_text_preserves_paragraphs():
    assert normalize_text(" a  \r\n\r\n\r\n b ") == "a\n\nb"


def test_load_valid_document(tmp_path):
    (tmp_path / "doc.json").write_text(json.dumps(sample()), encoding="utf-8")
    docs = load_documents(tmp_path)
    assert len(docs) == 1
    assert isinstance(docs[0], Document)
    assert docs[0].text == "hello\n\nworld"
    assert docs[0].metadata["source_id"] == "s1"
    assert docs[0].metadata["url"] == "https://groww.in/page"


def test_skips_scrape_report(tmp_path):
    (tmp_path / "scrape_report.json").write_text(json.dumps({"success": [], "failed": []}))
    (tmp_path / "doc.json").write_text(json.dumps(sample()), encoding="utf-8")
    assert len(load_documents(tmp_path)) == 1


def test_skips_empty_and_missing_fields(tmp_path):
    empty = sample(text="  ")
    missing = {"source_id": "x"}
    (tmp_path / "empty.json").write_text(json.dumps(empty), encoding="utf-8")
    (tmp_path / "missing.json").write_text(json.dumps(missing), encoding="utf-8")
    assert load_documents(tmp_path) == []


def test_skips_duplicate_source_and_content(tmp_path):
    (tmp_path / "a.json").write_text(json.dumps(sample()), encoding="utf-8")
    (tmp_path / "b.json").write_text(json.dumps(sample(source_id="s2")), encoding="utf-8")
    assert len(load_documents(tmp_path)) == 1


def test_missing_directory_raises(tmp_path):
    import pytest
    with pytest.raises(FileNotFoundError):
        load_documents(tmp_path / "absent")
