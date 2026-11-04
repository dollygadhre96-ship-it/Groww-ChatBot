"""Persistent ChromaDB storage for document chunks and embeddings."""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

@dataclass
class SearchResult:
    id: str
    text: str
    metadata: dict[str, Any]
    distance: float

class ChromaStore:
    """Persistent cosine-distance Chroma collection. Lower distance is better."""
    def __init__(self, persist_directory: str | Path = "data/chroma", collection_name: str = "hdfc_mf_faq", embedding_model: str | None = None):
        self.persist_directory = Path(persist_directory)
        self.persist_directory.mkdir(parents=True, exist_ok=True)
        self.collection_name, self.embedding_model = collection_name, embedding_model
        try:
            import chromadb
        except ImportError as exc:
            raise RuntimeError("ChromaDB is required to initialize ChromaStore; install requirements.txt") from exc
        self.client = chromadb.PersistentClient(path=str(self.persist_directory))
        self.collection = self.client.get_or_create_collection(name=collection_name, metadata={"hnsw:space": "cosine", "embedding_model": embedding_model or "unspecified"})

    @staticmethod
    def _metadata(metadata: dict[str, Any]) -> dict[str, Any]:
        clean = {}
        for key, value in metadata.items():
            if value is None: continue
            if isinstance(value, (str, int, float, bool)): clean[str(key)] = value
            elif isinstance(value, (list, tuple)): clean[str(key)] = ", ".join(map(str, value))
            else: clean[str(key)] = str(value)
        return clean

    def upsert_chunks(self, chunks: Sequence[Any], embeddings: Sequence[Sequence[float]]) -> int:
        """Upsert Chunk objects with id, text and metadata alongside vectors."""
        if len(chunks) != len(embeddings): raise ValueError("chunks and embeddings must have equal lengths")
        if not chunks: return 0
        ids, docs, metas, vectors = [], [], [], []
        for chunk, vector in zip(chunks, embeddings):
            cid, text, metadata = getattr(chunk, "id", getattr(chunk, "chunk_id", None)), getattr(chunk, "text", None), getattr(chunk, "metadata", {})
            if not cid or not isinstance(text, str) or not text.strip(): raise ValueError("each chunk needs a non-empty id and text")
            clean_metadata = self._metadata(dict(metadata or {}))
            ids.append(str(cid)); docs.append(text); metas.append(clean_metadata or None); vectors.append([float(x) for x in vector])
        dims = {len(v) for v in vectors}
        if len(dims) != 1 or 0 in dims: raise ValueError("embeddings must have one consistent, non-zero dimension")
        self.collection.upsert(ids=ids, documents=docs, metadatas=metas, embeddings=vectors)
        return len(ids)

    def query(self, query_embedding: Sequence[float], top_k: int = 5, where: dict[str, Any] | None = None) -> list[SearchResult]:
        """Search nearest chunks; returned distances are cosine distances (lower is closer)."""
        if top_k < 1: raise ValueError("top_k must be at least 1")
        if not query_embedding or self.count() == 0: return []
        args = {"query_embeddings": [[float(x) for x in query_embedding]], "n_results": min(top_k, self.count()), "include": ["documents", "metadatas", "distances"]}
        if where: args["where"] = where
        result = self.collection.query(**args)
        ids, docs, metas, distances = tuple((result.get(k) or [[]])[0] for k in ("ids", "documents", "metadatas", "distances"))
        return [SearchResult(str(i), d or "", m or {}, float(dist)) for i,d,m,dist in zip(ids,docs,metas,distances)]

    def count(self) -> int: return int(self.collection.count())

    def reset(self) -> None:
        """Destructively clear and recreate this collection."""
        self.client.delete_collection(self.collection_name)
        self.collection = self.client.get_or_create_collection(name=self.collection_name, metadata={"hnsw:space": "cosine", "embedding_model": self.embedding_model or "unspecified"})
