"""Content-aware text chunking for loaded ingestion Document objects."""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Iterable

from src.ingestion.loader import Document


@dataclass(frozen=True)
class Chunk:
    """A chunk with text, stable ID, and inherited source metadata."""

    chunk_id: str
    text: str
    metadata: dict[str, Any]


def _split_paragraphs(text: str) -> list[str]:
    # Loader preserves paragraph breaks; use them as the first split boundary.
    return [p.strip() for p in text.split("\n\n") if p.strip()]


def _split_long_block(block: str, chunk_size: int) -> list[str]:
    """Split oversized blocks near sentence/word boundaries, with hard fallback."""
    pieces: list[str] = []
    remaining = block.strip()
    while len(remaining) > chunk_size:
        cut = remaining.rfind(" ", 0, chunk_size + 1)
        # Prefer a sentence boundary in the latter half of the chunk window.
        sentence_cut = max(
            remaining.rfind(". ", 0, chunk_size + 1),
            remaining.rfind("? ", 0, chunk_size + 1),
            remaining.rfind("! ", 0, chunk_size + 1),
        )
        if sentence_cut >= chunk_size // 2:
            cut = sentence_cut + 1
        if cut < chunk_size // 2:
            cut = chunk_size
        pieces.append(remaining[:cut].strip())
        remaining = remaining[cut:].strip()
    if remaining:
        pieces.append(remaining)
    return pieces


def chunk_document(
    document: Document,
    chunk_size: int = 800,
    chunk_overlap: int = 120,
) -> list[Chunk]:
    """Chunk one document using paragraph-aware packing and bounded overlap.

    Sizes are character counts (not tokens). Metadata from the source is copied
    to each chunk, with chunk index and document hash added.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero")
    if chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be >= 0 and < chunk_size")
    if not document.text.strip():
        return []

    blocks = _split_paragraphs(document.text)
    blocks = [piece for block in blocks for piece in _split_long_block(block, chunk_size)]
    chunks_text: list[str] = []
    current = ""

    for block in blocks:
        candidate = f"{current}\n\n{block}" if current else block
        if len(candidate) <= chunk_size:
            current = candidate
            continue
        if current:
            chunks_text.append(current.strip())
        # Carry a small overlap from previous chunk, then append next block.
        overlap = chunks_text[-1][-chunk_overlap:].strip() if chunk_overlap and chunks_text else ""
        current = f"{overlap}\n\n{block}".strip() if overlap else block
        # If overlap + block is still too large, keep block intact (already split).
        if len(current) > chunk_size:
            current = block

    if current.strip():
        chunks_text.append(current.strip())

    source_id = str(document.metadata.get("source_id", "unknown"))
    doc_hash = str(document.metadata.get("content_hash", ""))
    output: list[Chunk] = []
    for index, text in enumerate(chunks_text):
        stable = hashlib.sha256(
            f"{source_id}:{doc_hash}:{index}:{text}".encode("utf-8")
        ).hexdigest()[:20]
        metadata = dict(document.metadata)
        metadata.update({"chunk_index": index, "document_hash": doc_hash})
        output.append(Chunk(chunk_id=f"{source_id}-{stable}", text=text, metadata=metadata))
    return output


def chunk_documents(
    documents: Iterable[Document],
    chunk_size: int = 800,
    chunk_overlap: int = 120,
) -> list[Chunk]:
    """Chunk a collection of loaded documents."""
    result: list[Chunk] = []
    for document in documents:
        result.extend(chunk_document(document, chunk_size, chunk_overlap))
    return result
