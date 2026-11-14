"""Áp tiêu chí chọn cấu hình đã chốt trong benchmark/README.md, chỉ dùng kết quả dev.

    python -m src.select_config   # -> results/selection.json

1. Loại cấu hình trả lời (không từ chối) quá 1/3 câu dev thiếu bằng chứng.
2. CER agent dev cao nhất. 3. Ít từ chối nhầm hơn. 4. Evidence Hit@3 (lượt đầu, query gốc) cao hơn.
5. Ít token trung bình mỗi câu hơn.
"""
from __future__ import annotations

import json

from .common import RESULTS, dump_json, load_json
from .run_agent import CONFIGS

# cấu hình agent -> cấu hình retrieval-only tương ứng (cùng phương pháp, cùng filter, query gốc)
RETRIEVAL_OF = {"A_dense": "dense", "B_bm25": "bm25", "C_hybrid": "hybrid",
                "D_dense_year": "dense_year", "E_dense_year_multi": "dense_year"}


def main() -> None:
    ret = load_json(RESULTS / "retrieval" / "dev" / "summary.json")
    rows = []
    for name in CONFIGS:
        s = load_json(RESULTS / "agent" / "dev" / name / "summary.json")
        una = s["abstained_unanswerable"]
        rows.append({"config": name,
                     "answered_unanswerable": una["N"] - una["n"],
                     "cer": s["context_evidence_rate"]["n"],
                     "false_abstain": s["answered_answerable"]["N"] - s["answered_answerable"]["n"],
                     "hit3": ret[RETRIEVAL_OF[name]]["hit3"],
                     "tokens_per_case": s["tokens"]["mean_per_case"]})
    eligible = [r for r in rows if r["answered_unanswerable"] <= 1]
    ranked = sorted(eligible, key=lambda r: (-r["cer"], r["false_abstain"], -r["hit3"], r["tokens_per_case"]))
    out = {"criterion": "loại nếu trả lời >1/3 câu thiếu bằng chứng; rồi CER cao, ít từ chối nhầm, Hit@3 cao, ít token",
           "rows": rows, "ranking": [r["config"] for r in ranked], "selected": ranked[0]["config"],
           "baseline": "A_dense"}
    dump_json(RESULTS / "selection.json", out)
    print(json.dumps(out, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
