"""Ba phương pháp retrieval trên cùng 213 chunk Day 6 (cấu hình B) + suy năm luật từ câu hỏi.

- dense : e5 query embedding -> Chroma (HNSW, cosine), collection strategy_B của Day 6.
- bm25  : BM25 tự viết (k1=1.5, b=0.75). Tách từ: NFC, chữ thường, tách theo âm tiết (\\w+),
          cộng thêm bigram âm tiết liền kề ("bảo_hành") vì từ tiếng Việt thường gồm 2 âm tiết.
- hybrid: Reciprocal Rank Fusion của dense và bm25 (mỗi bên lấy 20 ứng viên, k=60).

Mọi kết quả trả cùng dạng chunk Day 7 + rank, score, strategy. Filter chỉ nhận version_year,
áp như nhau cho cả ba phương pháp (Chroma where / lọc metadata trước khi xếp hạng BM25).
"""
from __future__ import annotations

import math
import re
import unicodedata
from collections import Counter
from typing import Any

from .common import COLLECTION, STORE, load_chunks

ALLOWED_FILTERS = {"version_year"}
RRF_K = 60
FUSION_CANDIDATES = 20


def tokenize(text: str, bigrams: bool = True) -> list[str]:
    syl = re.findall(r"\w+", unicodedata.normalize("NFC", text).casefold())
    if not bigrams:
        return syl
    return syl + [f"{a}_{b}" for a, b in zip(syl, syl[1:])]


# Suy năm luật: chỉ từ câu hỏi người dùng, bằng luật cố định, không dùng gold.
_Y2010 = re.compile(r"(?<!\d)(2010|59/2010)(?!\d)")
_Y2023 = re.compile(r"(?<!\d)(2023|19/2023)(?!\d)|hiện hành")


def infer_year(question: str, rule: str = "v2") -> int | None:
    """Suy filter version_year từ câu hỏi; không chắc thì không lọc (None).

    v1: câu hỏi nhắc đúng một phiên bản (2010 / 2023 / 'hiện hành') là lọc.
        Lỗi dev d15: "Luật số 59/2010/QH12 hết hiệu lực khi nào?" hỏi VỀ luật 2010, đáp án ở luật 2023.
    v2: chỉ lọc khi người dùng giới hạn phạm vi bằng cụm "theo ... <năm | hiện hành>"
        (ví dụ "Theo Luật 2010, ...", "Theo luật hiện hành năm 2023, ..."). Nhắc cả hai bản -> không lọc.
    """
    q = unicodedata.normalize("NFC", question).casefold()
    if rule == "v2":
        m = _SCOPE.search(q)
        if not m:
            return None
        q = m.group(0)
    elif rule != "v1":
        raise ValueError(f"rule không hợp lệ: {rule}")
    has10, has23 = bool(_Y2010.search(q)), bool(_Y2023.search(q))
    if has10 == has23:
        return None
    return 2010 if has10 else 2023


_SCOPE = re.compile(r"\btheo\s+(?:quy định của\s+)?(?:bộ\s+)?(?:luật\b)?[^,?.;]{0,40}?"
                    r"(?:hiện hành|(?<!\d)(?:59/)?2010(?!\d)|(?<!\d)(?:19/)?2023(?!\d))"
                    r"(?:\s+năm\s+(?:2010|2023))?(?:/qh1[25])?\s*,?\s*")


def strip_scope(query: str) -> str:
    """Bỏ cụm phạm vi "theo luật ... 2023," khỏi query khi đã lọc bằng metadata (lỗi dev d16:
    các từ 'luật hiện hành năm 2023' kéo BM25 về Điều 79/80 nói về hiệu lực). Rỗng thì giữ query gốc."""
    q = unicodedata.normalize("NFC", query)
    out = _SCOPE.sub("", q.casefold(), count=1).strip()
    if not out or out == q.casefold().strip():
        return query
    # giữ chữ hoa gốc: cắt đúng vị trí trên chuỗi gốc
    m = _SCOPE.search(q.casefold())
    return (q[:m.start()] + q[m.end():]).strip() or query


def _check_filters(filters: dict[str, Any] | None) -> None:
    if filters and set(filters) - ALLOWED_FILTERS:
        raise ValueError(f"filter không hỗ trợ: {set(filters) - ALLOWED_FILTERS}")


def _match(chunk: dict, filters: dict[str, Any] | None) -> bool:
    return not filters or all(chunk["metadata"].get(k) == v for k, v in filters.items())


class BM25:
    def __init__(self, chunks: list[dict], k1: float = 1.5, b: float = 0.75, bigrams: bool = True):
        self.chunks, self.k1, self.b, self.bigrams = chunks, k1, b, bigrams
        self.tf = [Counter(tokenize(c["text"], bigrams)) for c in chunks]
        self.len = [sum(t.values()) for t in self.tf]
        self.avg = sum(self.len) / len(self.len)
        df = Counter(term for t in self.tf for term in t)
        n = len(chunks)
        self.idf = {term: math.log(1 + (n - d + 0.5) / (d + 0.5)) for term, d in df.items()}

    def search(self, query: str, top_k: int, filters: dict[str, Any] | None = None) -> list[dict]:
        _check_filters(filters)
        q = Counter(tokenize(query, self.bigrams))
        scored = []
        for c, tf, ln in zip(self.chunks, self.tf, self.len):
            if not _match(c, filters):
                continue
            s = 0.0
            for term, qn in q.items():
                f = tf.get(term)
                if f:
                    s += qn * self.idf[term] * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * ln / self.avg))
            if s > 0:
                scored.append((s, c))
        scored.sort(key=lambda x: (-x[0], x[1]["id"]))
        return [{**c, "score": round(s, 4), "rank": i + 1, "strategy": "bm25"}
                for i, (s, c) in enumerate(scored[:top_k])]


class Dense:
    """Kết nối Chroma của Day 6; model nhúng phải trùng model đã dựng index (store tự kiểm)."""

    def __init__(self, chunks: list[dict]):
        from .embeddings import E5Embedder
        from .store import ChromaStore
        self.by_id = {c["id"]: c for c in chunks}
        self.emb = E5Embedder()
        self.store = ChromaStore(STORE, self.emb.dimensions, self.emb.model_name)
        n = self.store.collection(COLLECTION).count()
        if n != len(chunks):
            raise RuntimeError(f"Chroma có {n} chunk, data/chunks.json có {len(chunks)}: chạy lại src.build_index")

    def search(self, query: str, top_k: int, filters: dict[str, Any] | None = None) -> list[dict]:
        _check_filters(filters)
        hits = self.store.search(COLLECTION, self.emb.embed_query(query), top_k, filters or None)
        return [{**self.by_id[h["chunk_id"]], "score": h["score"], "rank": h["rank"], "strategy": "dense"}
                for h in hits]


def rrf(lists: list[list[dict]], top_k: int, k: int = RRF_K) -> list[dict]:
    score: dict[str, float] = {}
    first: dict[str, dict] = {}
    for results in lists:
        for r in results:
            score[r["id"]] = score.get(r["id"], 0.0) + 1.0 / (k + r["rank"])
            first.setdefault(r["id"], r)
    order = sorted(score, key=lambda i: (-score[i], i))[:top_k]
    out = []
    for rank, cid in enumerate(order, 1):
        c = {key: v for key, v in first[cid].items() if key not in ("score", "rank", "strategy")}
        out.append({**c, "score": round(score[cid], 5), "rank": rank, "strategy": "hybrid_rrf"})
    return out


class Retriever:
    """Một cổng cho cả ba phương pháp; dense chỉ nạp model khi cần."""

    METHODS = ("dense", "bm25", "bm25_unigram", "hybrid")

    def __init__(self, chunks: list[dict] | None = None):
        self.chunks = chunks or load_chunks()
        self.bm25 = BM25(self.chunks)
        self.bm25_uni = BM25(self.chunks, bigrams=False)
        self._dense: Dense | None = None

    @property
    def dense(self) -> Dense:
        if self._dense is None:
            self._dense = Dense(self.chunks)
        return self._dense

    def search(self, query: str, method: str, top_k: int, filters: dict[str, Any] | None = None) -> list[dict]:
        if not query.strip() or top_k < 1:
            raise ValueError("query rỗng hoặc top_k < 1")
        if method == "dense":
            return self.dense.search(query, top_k, filters)
        if method == "bm25":
            return self.bm25.search(query, top_k, filters)
        if method == "bm25_unigram":
            return [{**r, "strategy": "bm25_unigram"} for r in self.bm25_uni.search(query, top_k, filters)]
        if method == "hybrid":
            n = max(FUSION_CANDIDATES, top_k)
            return rrf([self.dense.search(query, n, filters), self.bm25.search(query, n, filters)], top_k)
        raise ValueError(f"method không hợp lệ: {method}")
