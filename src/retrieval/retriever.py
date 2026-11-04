"""Question retrieval and source-aware context assembly."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Sequence, TYPE_CHECKING
from src.vector_store.store import SearchResult
if TYPE_CHECKING:
    from src.embeddings.service import EmbeddingService
    from src.vector_store.store import ChromaStore

@dataclass
class RetrievedChunk:
    id: str
    text: str
    metadata: dict[str, Any]
    distance: float
    relevance: float

@dataclass
class RetrievalResponse:
    query: str
    chunks: list[RetrievedChunk]
    context: str
    sources: list[dict[str, str]]
    no_evidence: bool
    threshold: float

class Retriever:
    """Embed a question, search Chroma, filter cosine distance, assemble evidence."""
    def __init__(self, store: ChromaStore, embeddings: EmbeddingService, top_k: int = 5,
                 relevance_threshold: float = 0.45, max_context_chars: int = 12000):
        if top_k < 1: raise ValueError("top_k must be >= 1")
        if not 0 <= relevance_threshold <= 2: raise ValueError("cosine distance threshold must be between 0 and 2")
        if max_context_chars < 1: raise ValueError("max_context_chars must be >= 1")
        self.store, self.embeddings = store, embeddings
        self.top_k, self.threshold, self.max_context_chars = top_k, float(relevance_threshold), max_context_chars

    def retrieve(self, question: str, where: dict[str, Any] | None = None) -> RetrievalResponse:
        if not isinstance(question, str) or not question.strip():
            raise ValueError("question must be a non-empty string")
        query = question.strip()
        vector = self.embeddings.embed_query(query)
        hits: Sequence[SearchResult] = self.store.query(vector, top_k=self.top_k, where=where)

        selected: list[RetrievedChunk] = []
        seen_text: set[str] = set()
        for hit in hits:
            distance = float(hit.distance)  # Chroma cosine distance: 0 is closest.
            if distance < 0 or distance > self.threshold: continue
            text = (hit.text or "").strip()
            normalized = " ".join(text.split()).casefold()
            if not normalized or normalized in seen_text: continue
            seen_text.add(normalized)
            selected.append(RetrievedChunk(str(hit.id), text, dict(hit.metadata or {}), distance,
                                         max(0.0, 1.0 - distance)))

        # Only evidence that fits completely in the context is exposed downstream.
        included, context_parts, source_map = [], [], {}
        used = 0
        for candidate in selected:
            meta = candidate.metadata
            title = str(meta.get("title") or meta.get("source_id") or "Groww source")
            url = str(meta.get("url") or "")
            evidence = f"[Evidence {len(included) + 1} | {title} | {url}]\n{candidate.text}"
            separator = "\n\n" if context_parts else ""
            remaining = self.max_context_chars - used - len(separator)
            if remaining <= 0 or len(evidence) > remaining: continue
            context_parts.append(evidence)
            used += len(separator) + len(evidence)
            included.append(candidate)
            source_map.setdefault(url or title, {"title": title, "url": url})

        context = "\n\n".join(context_parts)
        return RetrievalResponse(query, included, context, list(source_map.values()),
                                not bool(included), self.threshold)
