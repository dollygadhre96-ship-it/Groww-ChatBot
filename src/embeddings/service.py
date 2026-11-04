"""Embedding wrapper using sentence-transformers."""
from __future__ import annotations
from typing import Sequence
from sentence_transformers import SentenceTransformer
from src.chunking.chunker import Chunk

class EmbeddingService:
    """Embeds chunks and query text with the same normalized model."""
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)

    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        if any(not isinstance(t, str) or not t.strip() for t in texts):
            raise ValueError("Embedding inputs must be non-empty strings")
        vectors = self.model.encode(list(texts), batch_size=32, show_progress_bar=False,
                                   convert_to_numpy=True, normalize_embeddings=True)
        return [[float(value) for value in row] for row in vectors]

    def embed_chunks(self, chunks: Sequence[Chunk]) -> list[dict]:
        vectors = self.embed_texts([c.text for c in chunks])
        return [{"chunk_id": c.chunk_id, "text": c.text, "metadata": c.metadata,
                 "embedding": v} for c, v in zip(chunks, vectors)]

    def embed_query(self, query: str) -> list[float]:
        return self.embed_texts([query])[0]
