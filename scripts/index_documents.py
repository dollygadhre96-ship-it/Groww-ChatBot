"""Build/update the local ChromaDB index from scraped Groww JSON documents.

Run from the project root with: python scripts/index_documents.py
This script never deletes the existing collection; it uses stable IDs + upsert.
"""
from __future__ import annotations
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import yaml
from src.ingestion.loader import load_documents
from src.chunking.chunker import chunk_documents
from src.embeddings.service import EmbeddingService
from src.vector_store.store import ChromaStore


def main() -> int:
    config_path = ROOT / "config" / "config.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    chunk_cfg = config.get("chunking", {})
    emb_cfg = config.get("embeddings", {})
    store_cfg = config.get("vector_store", {})

    print("[1/4] Loading scraped documents...")
    documents = load_documents(ROOT / "data" / "processed")
    if not documents:
        print("ERROR: No valid scraped documents found in data/processed.")
        print("Run first: python -m src.ingestion.scraper")
        return 1
    print(f"      Loaded {len(documents)} document(s).")

    print("[2/4] Chunking documents...")
    chunks = chunk_documents(
        documents,
        chunk_size=int(chunk_cfg.get("chunk_size", 800)),
        chunk_overlap=int(chunk_cfg.get("chunk_overlap", 120)),
    )
    if not chunks:
        print("ERROR: Chunking produced no chunks.")
        return 1
    print(f"      Created {len(chunks)} chunk(s).")

    print("[3/4] Loading embedding model and generating vectors...")
    model_name = emb_cfg.get("model", "sentence-transformers/all-MiniLM-L6-v2")
    embedder = EmbeddingService(model_name)
    vectors = embedder.embed_texts([chunk.text for chunk in chunks])
    print(f"      Generated {len(vectors)} embedding(s) using {model_name}.")

    print("[4/4] Upserting into persistent ChromaDB...")
    persist_dir = ROOT / store_cfg.get("persist_directory", "data/chroma")
    store = ChromaStore(
        persist_directory=persist_dir,
        collection_name=store_cfg.get("collection_name", "hdfc_mf_faq"),
        embedding_model=model_name,
    )
    upserted = store.upsert_chunks(chunks, vectors)
    print(f"      Upserted {upserted} chunk(s).")
    print(f"SUCCESS: ChromaDB collection now contains {store.count()} record(s).")
    print(f"Collection: {store.collection_name}\nPersist directory: {store.persist_directory}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
