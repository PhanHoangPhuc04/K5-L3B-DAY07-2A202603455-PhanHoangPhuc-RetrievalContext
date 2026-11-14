"""Kiểm hành vi code Day 7 (offline). Model và retriever ở đây là giả, chỉ để kiểm code, không đo chất lượng."""
import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.agent import ABSTAIN, pack, parse_citations, run_agent  # noqa: E402
from src.check_benchmark import check  # noqa: E402
from src.llm import LLMError  # noqa: E402
from src.retrieval import BM25, infer_year, rrf, strip_scope, tokenize  # noqa: E402


def chunk(cid, text, year=2023, page=1):
    return {"id": cid, "text": text, "source_id": f"luat-{year}", "source": "x", "page": page,
            "metadata": {"version_year": year, "law_number": f"{year}/QH"}}


CHUNKS = [chunk("luat-a:1:0", "Điều 1. Bảo hành sản phẩm được tính lại từ thời điểm đổi mới."),
          chunk("luat-a:2:0", "Điều 2. Hòa giải viên là người dân tộc thiểu số.", 2010, 2),
          chunk("luat-a:3:0", "Điều 3. Thương lượng trong 07 ngày làm việc.", 2023, 3)]


class RetrievalTests(unittest.TestCase):
    def test_tokenize_nfc_lower_bigram(self):
        decomposed = "Bảo hành"  # dấu tổ hợp -> NFC
        self.assertEqual(tokenize(decomposed, bigrams=False), ["bảo", "hành"])
        self.assertIn("bảo_hành", tokenize("Bảo hành"))

    def test_bm25_ranks_matching_chunk_and_filters(self):
        bm = BM25(CHUNKS)
        res = bm.search("thời điểm bảo hành", 3)
        self.assertEqual(res[0]["id"], "luat-a:1:0")
        self.assertEqual([r["rank"] for r in res], list(range(1, len(res) + 1)))
        only2010 = bm.search("hòa giải viên bảo hành", 3, {"version_year": 2010})
        self.assertTrue(only2010 and all(r["metadata"]["version_year"] == 2010 for r in only2010))
        with self.assertRaises(ValueError):
            bm.search("x", 3, {"chapter": "I"})

    def test_rrf_rewards_agreement(self):
        a = [{**CHUNKS[0], "rank": 1, "score": 0.9}, {**CHUNKS[1], "rank": 2, "score": 0.8}]
        b = [{**CHUNKS[1], "rank": 1, "score": 5.0}, {**CHUNKS[2], "rank": 2, "score": 4.0}]
        fused = rrf([a, b], 3)
        self.assertEqual(fused[0]["id"], "luat-a:2:0")  # có mặt ở cả hai danh sách
        self.assertEqual([f["rank"] for f in fused], [1, 2, 3])
        self.assertAlmostEqual(fused[0]["score"], round(1 / 62 + 1 / 61, 5))

    def test_infer_year_rules(self):
        self.assertEqual(infer_year("Theo Luật 2010, ai quyết định lỗi?"), 2010)
        self.assertEqual(infer_year("Theo luật hiện hành năm 2023, người tiêu dùng là ai?"), 2023)
        self.assertIsNone(infer_year("Luật số 59/2010/QH12 hết hiệu lực khi nào?"))  # v2: hỏi VỀ luật
        self.assertEqual(infer_year("Luật số 59/2010/QH12 hết hiệu lực khi nào?", "v1"), 2010)
        self.assertIsNone(infer_year("So sánh luật 2010 và 2023", "v1"))
        self.assertIsNone(infer_year("Ngày quyền người tiêu dùng là ngày nào?"))

    def test_strip_scope(self):
        self.assertEqual(strip_scope("Theo luật hiện hành năm 2023, người tiêu dùng là ai?"), "người tiêu dùng là ai?")
        self.assertEqual(strip_scope("Theo Luật 2010, biên bản cần chữ ký ai?"), "biên bản cần chữ ký ai?")
        self.assertEqual(strip_scope("bảo hành tính thế nào"), "bảo hành tính thế nào")


class FakeRetriever:
    def __init__(self, rows):
        self.rows, self.calls = rows, []

    def search(self, query, method, top_k, filters=None):
        self.calls.append((query, method, top_k, filters))
        return [{**c, "rank": i + 1, "score": 1.0} for i, c in enumerate(self.rows[:top_k])]


def fake_chat(script):
    it = iter(script)
    seen = []

    def chat(messages, tools=None, tool_choice=None):
        seen.append(tool_choice)
        return next(it), {"usage": {"prompt_tokens": 10, "completion_tokens": 2}, "latency_s": 0.0,
                          "finish_reason": "stop", "attempts": 1, "model": "fake"}
    chat.seen = seen
    return chat


def tool(query, i=0):
    return {"role": "assistant", "content": "", "tool_calls": [
        {"id": f"c{i}", "type": "function", "function": {"name": "search_law", "arguments": json.dumps({"query": query})}}]}


CFG = {"method": "dense", "year": "v2", "strip": True, "max_searches": 1, "top_k": 2, "budget": 2000}


class AgentTests(unittest.TestCase):
    def test_answer_with_context_citation(self):
        chat = fake_chat([tool("bảo hành"), {"content": "Tính lại từ lúc đổi [luat-a:1:0]."}])
        r = FakeRetriever(CHUNKS)
        out = run_agent("Theo Luật 2010, bảo hành tính thế nào?", r, CFG, chat=chat)
        self.assertEqual(out["status"], "answered")
        self.assertEqual(out["citations"], ["luat-a:1:0"])
        self.assertEqual(r.calls[0][3], {"version_year": 2010})  # filter suy từ câu hỏi gốc
        self.assertEqual(chat.seen, ["required", "none"])  # lượt đầu bắt buộc tìm, hết lượt thì buộc trả lời
        self.assertEqual(sum(e["event"] == "search" for e in out["events"]), 1)

    def test_citation_outside_context_abstains(self):
        chat = fake_chat([tool("x"), {"content": "Bịa [luat-a:3:0]."}])
        out = run_agent("câu hỏi", FakeRetriever(CHUNKS[:1]), CFG, chat=chat)
        self.assertEqual(out["status"], "abstain")
        self.assertEqual(out["citations"], [])
        self.assertTrue(any(e["event"] == "invalid_or_missing_citations" for e in out["events"]))

    def test_model_abstain_marker(self):
        chat = fake_chat([tool("x"), {"content": f"{ABSTAIN}: không có mức phạt."}])
        out = run_agent("mức phạt?", FakeRetriever(CHUNKS), CFG, chat=chat)
        self.assertEqual(out["status"], "abstain")

    def test_repeated_query_not_searched_twice(self):
        cfg = {**CFG, "max_searches": 3}
        chat = fake_chat([tool("bảo hành", 0), tool("Bảo  hành!", 1), {"content": "Đổi mới [luat-a:1:0]"}])
        r = FakeRetriever(CHUNKS)
        out = run_agent("q", r, cfg, chat=chat)
        self.assertEqual(len(r.calls), 1)
        self.assertTrue(any(e["event"] == "repeated_query" for e in out["events"]))
        self.assertEqual(out["status"], "answered")

    def test_budget_and_dedup(self):
        text, ids = pack(CHUNKS + CHUNKS, budget=10_000)
        self.assertEqual(ids, [c["id"] for c in CHUNKS])
        self.assertIn("[luat-a:1:0 | Luật 2023", text)
        _, small = pack(CHUNKS, budget=len(text) // 2)
        self.assertLess(len(small), len(CHUNKS))

    def test_model_error_is_recorded_not_faked(self):
        def broken(*a, **k):
            raise LLMError("HTTP 503")
        out = run_agent("q", FakeRetriever(CHUNKS), CFG, chat=broken)
        self.assertEqual(out["status"], "error")
        self.assertIn("503", out["error"])
        self.assertEqual(out["answer"], "")

    def test_parse_citations(self):
        self.assertEqual(parse_citations("a [luat-19-2023-QH15:13:0][luat-59-2010-QH12:2:1], b [luat-19-2023-QH15:13:0, x]"),
                         ["luat-19-2023-QH15:13:0", "luat-59-2010-QH12:2:1"])


class BenchmarkTests(unittest.TestCase):
    def test_every_evidence_inside_one_chunk(self):
        self.assertEqual(check(), [])


if __name__ == "__main__":
    unittest.main()
