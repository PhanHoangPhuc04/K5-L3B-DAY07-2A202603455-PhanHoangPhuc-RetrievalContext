"""Đường dẫn, nạp dữ liệu Day 7 và định nghĩa hit dùng chung cho mọi cấu hình.

Chunk (data/chunks.json): id, text, source_id, source, page, metadata — đúng data/README.md.
page = số Điều trong văn bản source_id (corpus HTML không có trang vật lý, giữ quy ước Day 6).
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
CHUNKS = ROOT / "data" / "chunks.json"
UNITS = ROOT / "data" / "clean" / "units.jsonl"
STORE = ROOT / "store" / "chroma"
QUESTIONS = ROOT / "benchmark" / "questions.json"   # câu có đáp án (schema starter/baseline.py)
CASES = ROOT / "benchmark" / "cases.json"           # mọi câu, gồm câu thiếu evidence
LOCK = ROOT / "benchmark" / "lock.json"
RESULTS = ROOT / "results"
COLLECTION = "B"  # Day 6 cấu hình B: chunk theo khoản, tối đa 1000 ký tự, 213 chunk


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def dump_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_chunks() -> list[dict[str, Any]]:
    if not CHUNKS.exists():
        raise SystemExit("Thiếu data/chunks.json: chạy `python -m src.build_index` trước")
    return load_json(CHUNKS)


def hit(chunk: dict[str, Any], gold: list[dict[str, Any]]) -> bool:
    """Giống starter/baseline.py: đúng nguồn, đúng Điều và chứa trọn evidence (khớp nguyên văn)."""
    return any(chunk["source_id"] == g["source_id"] and int(chunk["page"]) == int(g["page"])
               and g["evidence"] in chunk["text"] for g in gold)


def split_questions(split: str) -> tuple[list[dict], list[dict]]:
    """Trả (câu có đáp án, mọi case) của một split. Không dùng ngoài code chấm điểm."""
    qs = [q for q in load_json(QUESTIONS) if q["split"] == split]
    cases = [c for c in load_json(CASES) if c["split"] == split]
    return qs, cases
