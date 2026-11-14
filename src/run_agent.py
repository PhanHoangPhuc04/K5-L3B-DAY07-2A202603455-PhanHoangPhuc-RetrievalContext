"""Chạy agent thật trên một split và ghi trace theo định dạng starter/agent_metrics.py.

    python -m src.run_agent --split dev --configs A_dense B_bm25 C_hybrid D_dense_year E_dense_year_multi
    python -m src.run_agent --split test --configs A_dense <cấu_hình_đã_chọn>

Ghi results/agent/<split>/<config>/traces.json (mỗi câu: tool call, query, filter, kết quả, context cuối,
câu trả lời, citation, số lượt gọi model, token, thời gian, lỗi) và summary.json (chỉ số tự động:
tool use, phân bố số lượt, Context Evidence Rate, trạng thái, token, thời gian).
Gold chỉ được đọc ở bước chấm CER sau khi agent đã trả lời; không đi vào agent hay retrieval.
"""
from __future__ import annotations

import argparse
import statistics
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

from . import llm
from .agent import run_agent
from .common import RESULTS, dump_json, hit, load_json, split_questions
from .retrieval import Retriever

BASE = {"top_k": 5, "budget": 8000, "year": None, "strip": False, "max_searches": 1}
CONFIGS = {
    "A_dense": {**BASE, "method": "dense"},
    "B_bm25": {**BASE, "method": "bm25"},
    "C_hybrid": {**BASE, "method": "hybrid"},
    "D_dense_year": {**BASE, "method": "dense", "year": "v2", "strip": True},
    "E_dense_year_multi": {**BASE, "method": "dense", "year": "v2", "strip": True, "max_searches": 3},
}


def score_context(trace: dict, gold: list[dict], by_id: dict) -> bool:
    return any(hit(by_id[cid], gold) for cid in trace["context_ids"])


def run(split: str, name: str, retriever: Retriever, workers: int) -> dict:
    cfg = CONFIGS[name]
    qs, cases = split_questions(split)
    gold = {q["id"]: q["gold"] for q in qs}
    by_id = {c["id"]: c for c in retriever.chunks}
    retriever.dense  # nạp model nhúng trước khi chạy song song
    t0 = time.perf_counter()

    def one(case):
        out = run_agent(case["question"], retriever, cfg)
        return {"query_id": case["id"], "split": split, "config": name, "model": llm.MODEL,
                "question": case["question"], **out}

    with ThreadPoolExecutor(workers) as ex:
        traces = list(ex.map(one, cases))
    wall = round(time.perf_counter() - t0, 1)
    for t in traces:  # chấm tự động sau khi chạy xong
        t["context_has_evidence"] = score_context(t, gold[t["query_id"]], by_id) if t["query_id"] in gold else None
    ok = [t for t in traces if t["status"] != "error"]
    ans = [t for t in ok if t["query_id"] in gold]
    una = [t for t in ok if t["query_id"] not in gold]
    tok_in = sum(t["usage"]["prompt_tokens"] for t in traces)
    tok_out = sum(t["usage"]["completion_tokens"] for t in traces)
    summary = {
        "config": name, **cfg, "split": split, "model": llm.MODEL,
        "run_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "cases": len(traces),
        "errors": [{"query_id": t["query_id"], "error": t["error"]} for t in traces if t["status"] == "error"],
        "status": dict(Counter(t["status"] for t in traces)),
        "tool_use": {"n": sum(any(e["event"] == "search" for e in t["events"]) for t in ok), "N": len(ok)},
        "searches_per_case": dict(sorted(Counter(t["searches"] for t in ok).items())),
        "model_calls_per_case": dict(sorted(Counter(t["model_calls"] for t in ok).items())),
        "context_evidence_rate": {"n": sum(bool(t["context_has_evidence"]) for t in ans), "N": len(gold)},
        "answered_answerable": {"n": sum(t["status"] == "answered" for t in ans), "N": len(ans)},
        "abstained_unanswerable": {"n": sum(t["status"] == "abstain" for t in una), "N": len(una)},
        "invalid_citation_abstain": sum(any(e["event"] == "invalid_or_missing_citations" for e in t["events"]) for t in ok),
        "tokens": {"prompt": tok_in, "completion": tok_out,
                   "cache_hit": sum(t["usage"]["prompt_cache_hit_tokens"] for t in traces),
                   "mean_per_case": round((tok_in + tok_out) / len(traces), 1)},
        "model_calls_total": sum(t["model_calls"] for t in traces),
        "latency_s": {"median_case": statistics.median(t["elapsed_s"] for t in traces),
                      "max_case": max(t["elapsed_s"] for t in traces), "wall_clock": wall, "workers": workers},
        "mean_context_chars": round(statistics.mean(t["context_chars"] for t in ok), 1) if ok else None,
        "cost": "token lấy từ trường usage của API; đơn giá: xem trang giá DeepSeek",
    }
    out = RESULTS / "agent" / split / name
    dump_json(out / "traces.json", sorted(traces, key=lambda t: t["query_id"]))
    dump_json(out / "summary.json", summary)
    print(f"{name:20s} CER {summary['context_evidence_rate']}  status {summary['status']}  "
          f"abstain-unans {summary['abstained_unanswerable']}  calls {summary['model_calls_total']}  "
          f"tok {tok_in}+{tok_out}  wall {wall}s  errors {len(summary['errors'])}")
    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=("dev", "test"), required=True)
    ap.add_argument("--configs", nargs="+", required=True, choices=list(CONFIGS))
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()
    if args.split == "test":
        lock = load_json(RESULTS.parent / "benchmark" / "lock.json")
        print("benchmark lock:", lock["sha256"])
    r = Retriever()
    for name in args.configs:
        run(args.split, name, r, args.workers)


if __name__ == "__main__":
    main()
