"""Load and validate scraped JSON documents from data/processed.

This module does not scrape URLs. It consumes only the JSON files produced by
src.ingestion.scraper, excluding the scrape report.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = ROOT / "data" / "processed"
REPORT_NAME = "scrape_report.json"

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Document:
    """Normalized document object passed downstream to chunking."""

    text: str
    metadata: dict[str, Any]


REQUIRED_FIELDS = ("source_id", "title", "url", "retrieved_at", "content_hash", "text")


def normalize_text(text: str) -> str:
    """Normalize line endings and whitespace while preserving paragraph breaks."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = [line.strip() for line in text.split("\n")]
    # Keep at most one blank line between content blocks.
    normalized: list[str] = []
    previous_blank = False
    for line in lines:
        blank = not line
        if blank and previous_blank:
            continue
        normalized.append(line)
        previous_blank = blank
    return "\n".join(normalized).strip()


def load_documents(directory: Path | str = PROCESSED_DIR) -> list[Document]:
    """Read valid scraped document JSON files; log and skip invalid files."""
    directory = Path(directory)
    if not directory.exists():
        raise FileNotFoundError(f"Processed documents directory not found: {directory}")
    if not directory.is_dir():
        raise NotADirectoryError(f"Not a directory: {directory}")

    documents: list[Document] = []
    seen_source_ids: set[str] = set()
    seen_hashes: set[str] = set()

    for path in sorted(directory.glob("*.json")):
        if path.name == REPORT_NAME:
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(payload, dict):
                raise ValueError("JSON root must be an object")
            missing = [field for field in REQUIRED_FIELDS if field not in payload]
            if missing:
                raise ValueError(f"Missing required fields: {', '.join(missing)}")

            text = payload.get("text")
            if not isinstance(text, str) or not text.strip():
                raise ValueError("Document text is empty or invalid")
            source_id = payload.get("source_id")
            title = payload.get("title")
            url = payload.get("url")
            if not all(isinstance(v, str) and v.strip() for v in (source_id, title, url)):
                raise ValueError("source_id, title, and url must be non-empty strings")

            normalized = normalize_text(text)
            if not normalized:
                raise ValueError("Document has no text after normalization")
            if source_id in seen_source_ids:
                log.warning("Skipping duplicate source_id=%s in %s", source_id, path.name)
                continue
            content_hash = str(payload["content_hash"])
            if content_hash in seen_hashes:
                log.warning("Skipping duplicate content hash in %s", path.name)
                continue

            metadata = {
                "source_id": source_id,
                "title": title.strip(),
                "url": url.strip(),
                "retrieved_at": str(payload["retrieved_at"]),
                "content_hash": content_hash,
            }
            documents.append(Document(text=normalized, metadata=metadata))
            seen_source_ids.add(source_id)
            seen_hashes.add(content_hash)
            log.info("Loaded %s (%s characters)", title, len(normalized))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError, TypeError) as exc:
            log.warning("Skipping invalid document %s: %s", path.name, exc)

    return documents


if __name__ == "__main__":
    docs = load_documents()
    print(f"Loaded {len(docs)} valid document(s)")
    for doc in docs:
        print(f"- {doc.metadata['title']}: {len(doc.text)} characters")
