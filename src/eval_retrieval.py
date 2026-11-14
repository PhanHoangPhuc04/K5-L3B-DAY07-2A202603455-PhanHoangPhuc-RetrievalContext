"""Thí nghiệm retrieval-only (không gọi model) trên một split, dùng query gốc.

    python -m src.eval_retrieval --split dev

Mỗi cấu hình ghi results/retrieval/<split>/<config>.json (trace từng câu: top-5, score, rank,
nguồn, Điều, cờ hit) và results/retrieval/<split>/summary.json. Hit@1 / Hit@3 tính trên top-3;
'context_hit' = evidence nằm trong top-5 sau khi đóng gói theo ngân sách ký tự của agent.
"""
from __future__ import annotations

import argparse
import statistics
import time

from .agent import pack
from .common import RESULTS, dump_json, hit, split_questions
from .retrieval import Retriever, infer_year, strip_scope

CONFIGS = {
    "dense": {"method": "dense", "year": False},
    "bm25": {"method": "bm25", "year": False},
    "bm25_unigram": {"method": "bm25_unigram", "year": False},
    "hybrid": {"method": "hybrid", "year": False},
    "dense_year_v1": {"method": "dense", "year": "v1", "strip": False},
    "hybrid_year_v1": {"method": "hybrid", "year": "v1", "strip": False},
    "dense_year": {"method": "dense", "year": "v2", "strip": True},
    "hybrid_year": {"method": "hybrid", "year": "v2", "strip": True},
}
TOP_K, BUDGET = 5, 8000


def run(split: str, names: list[str]) -> dict:
    qs, _ = split_questions(split)
    r = Retriever()
    summary = {}
    for name in names:
        cfg = CONFIGS[name]
        traces, times = [], []
        for q in qs:
            year = infer_year(q["question"], cfg["year"]) if cfg["year"] else None
            filters = {"version_year": year} if year else None
            query = strip_scope(q["question"]) if filters and cfg.get("strip") else q["question"]
            t0 = time.perf_counter()
            res = r.search(query, cfg["method"], TOP_K, filters)
            times.append((time.perf_counter() - t0) * 1000)
            _, ids = pack(res, BUDGET)
            flags = [hit(c, q["gold"]) for c in res]
            traces.append({"query_id": q["id"], "question": q["question"], "query_used": query, "filter": filters,
                           "results": [{"id": c["id"], "rank": c["rank"], "score": c["score"],
                                        "source_id": c["source_id"], "page": c["page"], "hit": f}
                                       for c, f in zip(res, flags)],
                           "gold_pages": sorted({(g["source_id"], g["page"]) for g in q["gold"]}),
                           "hit1": bool(flags[:1] and flags[0]), "hit3": any(flags[:3]),
                           "context_hit": any(f and c["id"] in ids for c, f in zip(res, flags))})
        n = len(traces)
        s = {"config": name, **cfg, "split": split, "N": n, "top_k_logged": TOP_K, "budget_chars": BUDGET,
             "hit1": sum(t["hit1"] for t in traces), "hit3": sum(t["hit3"] for t in traces),
             "context_hit_top5": sum(t["context_hit"] for t in traces),
             "filtered_questions": sum(bool(t["filter"]) for t in traces),
             "median_ms": round(statistics.median(times), 1), "cost_usd": 0}
        summary[name] = s
        dump_json(RESULTS / "retrieval" / split / f"{name}.json", traces)
        print(f"{name:13s} hit@1 {s['hit1']}/{n}  hit@3 {s['hit3']}/{n}  ctx(top5) {s['context_hit_top5']}/{n}"
              f"  filter {s['filtered_questions']}  {s['median_ms']} ms")
    dump_json(RESULTS / "retrieval" / split / "summary.json", summary)
    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=("dev", "test"), required=True)
    ap.add_argument("--configs", nargs="*", default=list(CONFIGS))
    args = ap.parse_args()
    run(args.split, args.configs)


if __name__ == "__main__":
    main()
