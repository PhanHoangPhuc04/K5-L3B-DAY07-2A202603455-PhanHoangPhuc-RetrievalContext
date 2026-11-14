"""Chroma persistent store (on disk). One collection per strategy, cosine space."""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import chromadb

from .models import Chunk

_ALLOWED_FILTERS = {"version_year", "is_current", "document_id", "chapter"}


class ChromaStore:
    def __init__(self, path: Path, dimensions: int, model_name: str) -> None:
        path.mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(path=str(path))
        self.dimensions = dimensions
        self.model_name = model_name

    def collection(self, strategy: str):
        col = self.client.get_or_create_collection(
            name=f"strategy_{strategy}",
            metadata={"hnsw:space": "cosine", "embedding_model": self.model_name,
                      "dimensions": self.dimensions})
        if col.metadata.get("embedding_model") != self.model_name:
            raise RuntimeError(f"collection {strategy} built with another model: {col.metadata}")
        return col

    def upsert(self, strategy: str, chunks: list[Chunk], vectors: list[list[float]]) -> dict:
        """Stable IDs = doc:article:index. Re-ingesting the same text changes nothing;
        chunks of a re-ingested document that no longer exist are deleted."""
        if len(chunks) != len(vectors):
            raise ValueError("chunks and vectors differ in length")
        if any(len(v) != self.dimensions for v in vectors):
            raise ValueError("vector dimension mismatch")
        col = self.collection(strategy)
        before = col.count()
        docs = {c.document_id for c in chunks}
        new_ids = {c.chunk_id for c in chunks}
        stale = []
        for d in docs:
            old = col.get(where={"document_id": d}, include=[])["ids"]
            stale += [i for i in old if i not in new_ids]
        if stale:
            col.delete(ids=stale)
        metas = []
        for c in chunks:
            m = {k: v for k, v in c.metadata.items() if v is not None}
            m.update({"document_id": c.document_id, "source": c.source, "page": c.page,
                      "content_sha1": hashlib.sha1(c.text.encode()).hexdigest()})
            metas.append(m)
        for i in range(0, len(chunks), 500):
            col.upsert(ids=[c.chunk_id for c in chunks[i:i + 500]], embeddings=vectors[i:i + 500],
                       documents=[c.text for c in chunks[i:i + 500]], metadatas=metas[i:i + 500])
        return {"before": before, "after": col.count(), "deleted_stale": len(stale)}

    def search(self, strategy: str, query_vector: list[float], top_k: int = 3,
               filters: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        where = None
        if filters:
            bad = set(filters) - _ALLOWED_FILTERS
            if bad:
                raise ValueError(f"filter field not allowed: {bad}")
            clauses = [{k: v} for k, v in filters.items()]
            where = clauses[0] if len(clauses) == 1 else {"$and": clauses}
        res = self.collection(strategy).query(query_embeddings=[query_vector], n_results=top_k,
                                              where=where,
                                              include=["documents", "metadatas", "distances"])
        return [{"rank": r + 1, "chunk_id": cid, "score": round(1 - dist, 4),
                 "text": doc, "metadata": meta}
                for r, (cid, doc, meta, dist) in enumerate(zip(
                    res["ids"][0], res["documents"][0], res["metadatas"][0], res["distances"][0]))]
