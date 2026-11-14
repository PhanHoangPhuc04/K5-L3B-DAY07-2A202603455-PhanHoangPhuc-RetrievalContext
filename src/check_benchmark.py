"""Kiểm benchmark và khoá test.

    python -m src.check_benchmark          # kiểm schema + mọi evidence nằm trọn trong 1 chunk đúng nguồn/Điều
    python -m src.check_benchmark --lock   # ghi sha256 vào benchmark/lock.json (làm MỘT lần, trước khi chạy test)

Lần sau chạy không có --lock sẽ so sha256 hiện tại với lock: lệch thì báo lỗi (test đã bị sửa sau khi khoá).
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone

from .common import CASES, LOCK, QUESTIONS, dump_json, hit, load_chunks, load_json, sha256


def check() -> list[str]:
    chunks = load_chunks()
    qs, cases = load_json(QUESTIONS), load_json(CASES)
    problems = []
    ids = [q["id"] for q in qs]
    if len(set(ids)) != len(ids):
        problems.append("question ID trùng")
    case_ids = {c["id"] for c in cases}
    if len(case_ids) != len(cases):
        problems.append("case ID trùng")
    for q in qs:
        if q["id"] not in case_ids:
            problems.append(f"{q['id']}: thiếu trong cases.json")
        for k in ("id", "split", "question", "expected_answer", "gold"):
            if not q.get(k):
                problems.append(f"{q['id']}: thiếu {k}")
        for g in q["gold"]:
            n = sum(hit(c, [g]) for c in chunks)
            if n == 0:
                problems.append(f"{q['id']}: evidence không nằm trọn trong chunk nào của {g['source_id']} Điều {g['page']}")
    for c in cases:
        if c["split"] not in ("dev", "test") or not isinstance(c.get("answerable"), bool):
            problems.append(f"{c['id']}: split/answerable sai")
        if not c["answerable"]:
            if c.get("expected_behavior") != "search_and_abstain" or not c.get("absence_reason"):
                problems.append(f"{c['id']}: câu thiếu evidence cần expected_behavior + absence_reason")
            if c["id"] in ids:
                problems.append(f"{c['id']}: câu thiếu evidence không được có gold")
    return problems


def counts() -> dict:
    cases = load_json(CASES)
    return {s: {"answerable": sum(c["split"] == s and c["answerable"] for c in cases),
                "unanswerable": sum(c["split"] == s and not c["answerable"] for c in cases)}
            for s in ("dev", "test")}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lock", action="store_true")
    args = ap.parse_args()
    problems = check()
    print("split:", counts())
    if problems:
        print("\n".join(problems)); sys.exit(1)
    print("Mọi evidence nằm trọn trong một chunk đúng nguồn/Điều.")
    now = {"questions.json": sha256(QUESTIONS), "cases.json": sha256(CASES)}
    if args.lock:
        if LOCK.exists():
            sys.exit("lock.json đã tồn tại: không khoá lại (test phải giữ nguyên)")
        dump_json(LOCK, {"locked_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                         "sha256": now, "counts": counts(),
                         "note": "Khoá trước khi chạy bất kỳ cấu hình nào trên test."})
        print("Đã khoá:", now)
    elif LOCK.exists():
        locked = load_json(LOCK)["sha256"]
        if locked != now:
            sys.exit(f"benchmark khác lock: {locked} != {now}")
        print("Khớp lock:", now)


if __name__ == "__main__":
    main()
