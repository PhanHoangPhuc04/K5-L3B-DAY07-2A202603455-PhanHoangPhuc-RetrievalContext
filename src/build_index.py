"""Dựng lại kho Day 6 (cấu hình B) và xuất chunk theo định dạng Day 7.

    python -m src.build_index            # units.jsonl -> chunk B -> data/chunks.json + Chroma store/chroma
    python -m src.build_index --ingest   # parse lại HTML gốc data/raw trước (giống `src.run ingest` Day 6)

Chunking, embedding (intfloat/multilingual-e5-base, 'passage: '/'query: ', L2-normalize, cosine)
và Chroma giữ nguyên code Day 6 (src/chunking.py, src/embeddings.py, src/store.py).
Upsert theo ID ổn định nên chạy lại không nhân đôi chunk.
"""
from __future__ import annotations

import argparse
import json
import time
from importlib.metadata import version

from .chunking import by_paragraph
from .common import CHUNKS, COLLECTION, RESULTS, ROOT, STORE, UNITS, dump_json, sha256
from .models import DocumentPage

MAX_SIZE = 1000  # giống Day 6 cấu hình B


def load_pages() -> list[DocumentPage]:
    return [DocumentPage(**json.loads(line)) for line in UNITS.read_text(encoding="utf-8").splitlines()]


def to_day7(chunk) -> dict:
    meta = {k: v for k, v in chunk.metadata.items() if v is not None}
    return {"id": chunk.chunk_id, "text": chunk.text, "source_id": chunk.document_id,
            "source": chunk.source, "page": chunk.page, "metadata": meta}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ingest", action="store_true", help="parse lại data/raw/*.html")
    args = ap.parse_args()
    if args.ingest:
        from .ingestion import load_corpus, write_clean
        errors: list[dict] = []
        write_clean(load_corpus(errors), UNITS)
        print(f"ingest errors={errors}")
    pages = load_pages()
    chunks = [c for p in pages for c in by_paragraph(p, MAX_SIZE)]
    ids = [c.chunk_id for c in chunks]
    if len(set(ids)) != len(ids):
        raise SystemExit("chunk ID trùng")
    dump_json(CHUNKS, [to_day7(c) for c in chunks])

    from .embeddings import E5Embedder
    from .store import ChromaStore
    t0 = time.perf_counter()
    emb = E5Embedder()
    store = ChromaStore(STORE, emb.dimensions, emb.model_name)
    up = store.upsert(COLLECTION, chunks, emb.embed_documents([c.text for c in chunks]))
    info = {"chunks": len(chunks), "units": len(pages), "max_size": MAX_SIZE,
            "collection": f"strategy_{COLLECTION}", "store": str(STORE.relative_to(ROOT)),
            "embedding_model": emb.model_name, "dimensions": emb.dimensions,
            "max_seq_length": emb.max_seq_length, "query_prefix": "query: ",
            "document_prefix": "passage: ", "normalized": True, "space": "cosine",
            "chromadb": version("chromadb"), "sentence_transformers": version("sentence-transformers"),
            "upsert": up, "duplicates_after_upsert": up["after"] - len(chunks),
            "seconds": round(time.perf_counter() - t0, 1), "cost_usd": 0,
            "units_sha256": sha256(UNITS), "chunks_sha256": sha256(CHUNKS)}
    dump_json(RESULTS / "index_info.json", info)
    print(json.dumps(info, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
