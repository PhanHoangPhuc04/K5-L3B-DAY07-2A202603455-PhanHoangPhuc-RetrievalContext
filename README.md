# Bài làm Day 07 — Phan Hoàng Phúc (2A202603455, L3A)

Agentic RAG trên kho Day 6: Luật Bảo vệ quyền lợi người tiêu dùng 2010 (59/2010/QH12) và 2023 (19/2023/QH15), 131 Điều, 213 chunk theo khoản (cấu hình B của Day 6). Embedding `intfloat/multilingual-e5-base` (768 chiều) chạy local, vector store Chroma. Agent dùng model DeepSeek `deepseek-flash` qua tool calling thật. Chỉ làm phần cơ bản, không làm thử thách nâng cao.

Báo cáo: [report/individual/2A202603455_PhanHoangPhuc.md](report/individual/2A202603455_PhanHoangPhuc.md)

## Cách chạy (từ thư mục gốc repo, Python 3.11–3.13)

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env                      # điền DEEPSEEK_API_KEY (hoặc export biến môi trường)

python -m src.build_index                 # chunk B -> data/chunks.json + Chroma store/chroma (thêm --ingest để parse lại HTML gốc)
python -m src.check_benchmark             # kiểm evidence nằm trọn trong 1 chunk đúng nguồn/Điều, so sha256 với benchmark/lock.json
python -m src.eval_retrieval --split dev  # dense / bm25 / bm25_unigram / hybrid / lọc năm v1, v2 (không gọi model)
python starter/baseline.py --chunks data/chunks.json --questions benchmark/questions.json --split dev --output results/retrieval/dev/tfidf_starter
python -m src.run_agent --split dev --configs A_dense B_bm25 C_hybrid D_dense_year E_dense_year_multi
python -m src.select_config               # áp tiêu chí chọn đã chốt -> results/selection.json
python -m src.eval_retrieval --split test --configs dense dense_year
python -m src.run_agent --split test --configs A_dense D_dense_year
python starter/agent_metrics.py --cases benchmark/cases.json --traces results/agent/test/D_dense_year/traces.json \
  --reviews results/agent/test/D_dense_year/reviews.json --split test --output results/agent/test/D_dense_year/agent_metrics.json
python -m unittest discover -s tests -v
```

Lần đầu `build_index` tải model nhúng (~1 GB). Lỗi model hoặc vector store được ghi vào trace với `status: "error"`, không có kết quả giả. Chạy lại `run_agent` sẽ gọi API lại và ghi đè trace (model ở temperature 0 nhưng kết quả vẫn có thể lệch đôi chút).

| Thư mục / file | Nội dung |
|---|---|
| `src/` | `build_index`, `retrieval` (dense, BM25, RRF, suy năm), `agent` (vòng lặp tool calling), `llm` (client DeepSeek), `run_agent`, `eval_retrieval`, `check_benchmark`, `select_config`; `chunking/embeddings/store/ingestion/models` sao chép nguyên từ Day 6 |
| `data/` | `raw/` (HTML gốc), `clean/units.jsonl`, `sources.json`, `manifest.json` (từ Day 6), `chunks.json` (213 chunk, schema Day 7) |
| `benchmark/` | `questions.json` (30 câu có đáp án), `cases.json` (36 câu, gồm 6 câu thiếu bằng chứng), `lock.json`, `README.md` (split, định nghĩa chỉ số, tiêu chí chọn chốt trước) |
| `results/retrieval/` | trace top-5 từng câu và `summary.json` cho từng phương pháp |
| `results/agent/<split>/<cấu hình>/` | `traces.json`, `summary.json`, `reviews.json` (nhãn tay), `agent_metrics.json` (từ `starter/agent_metrics.py`) |
| `results/selection.json`, `results/index_info.json` | quyết định chọn cấu hình; thông tin index |
| `tests/test_day7.py` | test BM25, RRF, suy năm, bỏ cụm phạm vi, vòng lặp agent (model giả), citation, benchmark |

`store/`, `.env` không đưa lên repo; dựng lại store bằng `python -m src.build_index`.

---

# Đề bài gốc

# Day 7 — Agentic RAG

**Bài cá nhân · Python 3.11+ · tiếp nối Day 6**

Dùng embedding và vector store từ Day 6 để xây một agent biết tìm bằng chứng, trả lời có nguồn và nhận biết khi không đủ thông tin. Bạn tự chọn kiến trúc, số tool và cách tổ chức retrieval.

> **Tùy chọn:** [Gợi ý dùng AI để lập trình](AGENT_SETUP.md) giúp bạn làm lab; phần này không bắt buộc và không tính điểm.

## Bài cần làm

![Agent tìm tài liệu trong kho Day 6 và trả lời có nguồn](assets/retrieval-pipeline.svg)

1. Kết nối dữ liệu, embedding và vector store Day 6.
2. Xây agent dùng model thật và ít nhất một khả năng truy xuất dữ liệu. Agent có thể dùng nhiều tool và nhiều lượt gọi.
3. Thử **ít nhất 3 phương pháp retrieval khác nhau**, trong đó có dense retrieval dùng vector store Day 6. Tự chọn các phương pháp còn lại; xem [gợi ý](GUIDE.md#2-thử-các-phương-pháp-retrieval).
4. Thử nhiều cấu hình, benchmark, phân tích lỗi và chọn cấu hình tốt trên **dev**; sau đó đánh giá trên **test**.

Benchmark tối thiểu có **24 câu**: dev gồm 15 câu có đáp án và 2 câu thiếu bằng chứng; test gồm 5 câu có đáp án và 2 câu thiếu bằng chứng. Xem [benchmark](BENCHMARK.md) và [rubric](RUBRIC.md).

## Thử thách nâng cao · +15 điểm

Index một tài liệu thực tế từ **100 trang** có bảng, ảnh và biểu đồ; xây agent truy xuất và trả lời các câu hỏi về nội dung đó. Hoàn thành [thử thách nâng cao](ADVANCED.md) và đạt **ít nhất 90% benchmark test** để nhận 15 điểm bonus. Phần cơ bản tối đa 85 điểm; toàn bài tối đa 100 điểm.

## Bắt đầu

1. Đọc [hướng dẫn](GUIDE.md), [rubric](RUBRIC.md) và [quy tắc](RULES.md).
2. Tạo repo cá nhân theo [cách nộp bài](SUBMISSION.md), chuẩn bị đầu ra Day 6 và model hỗ trợ tool calling.
3. Dùng hoặc thay thế [starter](starter/README.md); bạn cần kết nối model và vector store thật.
4. Chạy `python -m unittest discover -s tests -v`. Test mock chỉ kiểm tra code, không phải benchmark agent thật.

![Các bước thử nghiệm trên dev và đánh giá trên test](assets/experiment-loop.svg)

Nộp code, cách chạy, cấu hình, benchmark, trace và [báo cáo](report/individual/MSSV_HoTen.md). Thời gian ở [checkpoints](CHECKPOINTS.md) là gợi ý, có thể điều chỉnh theo lớp và phạm vi thử nghiệm.
