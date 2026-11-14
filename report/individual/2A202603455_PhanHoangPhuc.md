# Báo cáo Day 7

Học viên: Phan Hoàng Phúc · MSSV 2A202603455 · L3A
Thời điểm chạy agent: dev 2026-11-14 13:17–13:18 UTC, test 13:21 UTC · macOS arm64, Python 3.13.15, CPU

## Dữ liệu và benchmark

- **Nguồn, quyền sử dụng, số trang và số chunk:**
  - Tôi dùng lại nguyên corpus Day 6: Luật Bảo vệ quyền lợi người tiêu dùng 2010 (59/2010/QH12, 51 Điều) và 2023 (19/2023/QH15, 80 Điều). Snapshot HTML lấy từ caselaw.vn, sha256 ghi trong `data/manifest.json`.
  - Văn bản quy phạm pháp luật không thuộc đối tượng bảo hộ quyền tác giả (Điều 15 khoản 2 Luật Sở hữu trí tuệ), nên tôi để cả dữ liệu thô lẫn dữ liệu sạch trong repo.
  - HTML không có trang. Như Day 6, tôi lấy `page` = số Điều: 131 Điều, 139.258 ký tự, tương đương 92 trang nếu tính 1.500 ký tự/trang.
  - Chunk dùng cấu hình B của Day 6 (theo khoản, tối đa 1.000 ký tự, mỗi chunk mở đầu bằng "Điều N. Tên điều"), được **213 chunk**. File `data/chunks.json` theo đúng schema `id, text, source_id, source, page, metadata`. ID ổn định dạng `luat-19-2023-QH15:13:0` (văn bản:Điều:thứ tự).
- **Vector store, embedding model, phiên bản và số chiều:**
  - Embedding `intfloat/multilingual-e5-base`, 768 chiều, max 512 token, chạy local trên CPU. Tiền tố `query: ` cho câu hỏi, `passage: ` cho tài liệu, chuẩn hoá L2.
  - Vector store chromadb 1.5.9 (PersistentClient, HNSW, cosine), collection `strategy_B` của Day 6, sentence-transformers 6.3.0.
  - `python -m src.build_index` dựng lại từ `data/clean/units.jsonl`. Lần chạy cuối: upsert 213 → 213, 0 chunk trùng, 16,6 giây, chi phí 0 (`results/index_info.json`).
  - `Dense` kiểm số chunk trong Chroma phải khớp `data/chunks.json`. Store tự báo lỗi nếu collection được dựng bằng model khác.
- **Agent model, cách chạy và phân chia dev/test:**
  - Model `deepseek-flash` qua API tương thích OpenAI (`https://api.deepseek.com`): tắt chế độ thinking, temperature 0, `max_tokens` 800.
  - Key đọc lúc chạy từ biến môi trường hoặc `.env` (đã gitignore). Repo chỉ có `.env.example` rỗng.
  - Dev 25 câu: 22 có đáp án và 3 câu thiếu bằng chứng. 22 câu có đáp án gồm 16 câu dev Day 6, cộng 6 câu test Day 6 (`d17`–`d22`). Day 6 đã chạy và báo cáo trên 6 câu này nên tôi không coi chúng là test độc lập nữa.
  - Test viết mới cho Day 7, 11 câu: 8 câu có đáp án lấy từ các Điều chưa dùng (2023: Điều 31, 46, 58, 66, 69; 2010: Điều 36, 37) và 3 câu thiếu bằng chứng. Trong đó 3 câu nêu năm luật, 3 câu ngoại lệ hoặc điều kiện.
  - Tôi khoá test bằng sha256 (`benchmark/lock.json`, 13:11:42 UTC), trước lần chạy retrieval đầu tiên. `src.check_benchmark` so lại lock trước khi chạy test.
- **Cách xác định câu có/thiếu evidence và tiêu chí chấm:**
  - Câu có đáp án: evidence phải là một đoạn khớp nguyên văn, nằm trọn trong một chunk đúng `source_id` và đúng Điều. Hàm `hit` dùng chung với `starter/baseline.py`, và `src.check_benchmark` kiểm cả 30 câu đều đạt điều kiện này.
  - Câu thiếu bằng chứng là câu gần chủ đề: tiền phạt, bảo hành tối thiểu, số tổng đài (dev); phí hòa giải, tỷ lệ hoa hồng đa cấp, thời hạn khiếu nại của người tiêu dùng (test).
  - Với mỗi câu thiếu bằng chứng, `absence_check` ghi các chuỗi tôi đã tìm trên toàn corpus và các Điều tôi đã đọc hết. Ví dụ "phạt tiền" 0 kết quả; "xử phạt" chỉ có ở Luật 2010 Điều 11, giao Chính phủ quy định chi tiết.
  - Định nghĩa chỉ số và tiêu chí chọn cấu hình tôi viết vào `benchmark/README.md` trước khi chạy test.

## Retrieval và thử nghiệm

Ba phương pháp chạy trên cùng 213 chunk, cùng bộ câu và nhãn dev, dùng câu hỏi gốc làm query:

- **dense**: dùng e5 trên Chroma.
- **BM25**: tôi tự viết với k1 = 1,5, b = 0,75. Tách từ bằng NFC, chữ thường, tách theo âm tiết `\w+`, rồi thêm bigram âm tiết liền kề (`bảo_hành`).
- **hybrid**: ghép dense top-20 và BM25 top-20 bằng RRF, k = 60.

Hit@1 và Evidence Hit@3 lấy trên top-3. Agent nhận top-5 mỗi lượt tìm. Context budget là 8.000 ký tự, tính cả nhãn nguồn `[id | Luật năm số hiệu | Điều N]`.

| Cấu hình | Phương pháp, top-k, filter, rerank, context budget | Giả thuyết / thay đổi | Split | Hit@1 | Evidence Hit@3 | Context Evidence Rate | Thời gian / chi phí |
|---|---|---|---|---|---|---|---|
| Baseline (A_dense) | dense, top-5/lượt, không filter, không rerank, 1 lượt tìm, 8.000 ký tự | Dense Day 6 làm mốc | dev | 14/22 (63,6%) | 22/22 (100%) | 22/22 (100%) | retrieval 36,7 ms/câu; agent 2,40 s/câu; 3.643 token/câu |
| Phương pháp 2 (B_bm25) | BM25 âm tiết + bigram, như trên | Văn bản luật lặp nhiều cụm từ chuyên ngành nên khớp từ khoá sẽ xếp hạng 1 tốt hơn | dev | 16/22 (72,7%) | 21/22 (95,5%) | 21/22 (95,5%) | 0,7 ms; 2,56 s; 3.601 token |
| Phương pháp 3 (C_hybrid) | RRF(dense 20, BM25 20), k = 60 | Ghép hai tín hiệu để lấy phần mạnh của mỗi bên | dev | 15/22 (68,2%) | 21/22 (95,5%) | 20/22 (90,9%) | 37,7 ms; 2,73 s; 3.594 token |
| Cải thiện 1 (D_dense_year) | dense + lọc `version_year` suy từ câu hỏi (luật v2) + bỏ cụm "theo luật … năm" khỏi query | Câu nêu năm luật bị lẫn Điều tương tự của bản kia | dev | 18/22 (81,8%) | 22/22 (100%) | 22/22 (100%) | 32,2 ms; 2,33 s; 3.604 token |
| Cải thiện 2 (E_dense_year_multi) | như D, tối đa 3 lượt tìm | Cho agent tìm lại khi lượt đầu chưa đủ | dev | 18/22* | 22/22* | 21/22 (95,5%) | 2,39 s; 4.834 token/câu (+33%) |
| Cấu hình đã chọn (D) | như dòng Cải thiện 1 | đã chốt trên dev | test | 8/8 (100%) | 8/8 (100%) | 8/8 (100%) | 34,7 ms; 2,20 s; 3.268 token/câu |
| Baseline (A) | như dòng Baseline | | test | 7/8 (87,5%) | 8/8 (100%) | 8/8 (100%) | 39,7 ms; 2,79 s; 3.312 token/câu |

\* Hit của E đo bằng câu hỏi gốc ở lượt đầu nên trùng với D. Các query sau do model tự viết, xem trace.

Đo thêm trên dev, cũng dùng query gốc (`results/retrieval/dev/summary.json`):

| Biến thể | Hit@1 | Hit@3 |
|---|---|---|
| BM25 chỉ âm tiết, không bigram | 16/22 | 21/22 |
| dense + lọc năm v1 | 17/22 | 21/22 |
| hybrid + lọc năm v1 | 19/22 | 20/22 |
| hybrid + lọc năm v2 + bỏ cụm phạm vi | 18/22 | 21/22 |
| TF-IDF của starter, top-3, context 800 ký tự | 17/22 | 19/22 |

TF-IDF của starter chỉ có 10/22 câu đưa được evidence vào context, vì budget 800 ký tự không chứa nổi một chunk dài.

Chi phí của từng cấu hình là token lấy từ trường `usage` của API. Đơn giá: xem trang giá DeepSeek, tôi không tự quy ra tiền. Retrieval và embedding chạy local nên chi phí bằng 0.

**Lỗi dev 1: lọc năm v1 làm mất d15.**
- Câu hỏi: "Luật … số 59/2010/QH12 hết hiệu lực khi nào?"
- Luật v1 thấy "2010" là lọc `version_year = 2010`. Nhưng đáp án nằm ở Luật 2023 Điều 79, nên top-3 chỉ còn 2010 Điều 50, 5, 2.
- Nguyên nhân: câu này hỏi **về** luật 2010, không phải hỏi **theo** luật 2010.
- Luật v2 chỉ lọc khi người dùng giới hạn phạm vi bằng cụm "theo … 2010 / 2023 / hiện hành". Số câu dev bị lọc giảm từ 9 xuống 7.
- Đo lại: dense + lọc năm từ 17/22 Hit@1 và 21/22 Hit@3 (v1) lên 18/22 và 22/22 (v2). Hybrid + lọc năm từ Hit@3 20/22 lên 21/22. Với d15, v2 không lọc và top-1 là đúng 2023 Điều 79.

**Lỗi dev 2: BM25 và hybrid trượt d16.**
- Câu hỏi: "Theo luật hiện hành năm 2023, người tiêu dùng được định nghĩa là ai?"
- BM25 trả 2023 Điều 79, 12, 80. Cụm "luật hiện hành năm 2023" khớp mạnh các điều nói về hiệu lực, còn chunk Điều 3 dài và không chứa từ "định nghĩa".
- Tôi thử bỏ cụm phạm vi khỏi query khi đã lọc bằng metadata. Hybrid vẫn trượt (top-3 là Điều 5, 6, 69). Thay đổi này **không** sửa được lỗi, vì nguyên nhân gốc là lệch từ vựng.
- Dense tìm đúng ngay. Ở agent B và C, câu này bị từ chối nhầm. Model muốn tìm lần hai nhưng đã hết lượt, nên in ra chuỗi gọi tool dạng văn bản; hậu kiểm citation chuyển kết quả thành từ chối thay vì để lọt một câu trả lời không có nguồn.

**Lỗi dev 3: query do agent tự viết làm mất d04 ở C và E.**
- Câu hỏi về bán hàng "đến tận nhà". Model viết query "quyền đơn phương chấm dứt hợp đồng và hoàn trả hàng hóa…", bỏ mất ý "tận cửa".
- Context vì vậy không có Điều 44. Agent trả lời theo Điều 47 (bán hàng không tại địa điểm giao dịch thường xuyên) và Điều 38 (giao dịch từ xa).
- Retrieval bằng câu hỏi gốc thì cả dense lẫn hybrid đều có Điều 44 trong top-3. Cấu hình E tìm 2 lượt cũng không lấy lại được.
- Tôi ghi lỗi này là giới hạn, chưa sửa. Hướng sửa: luôn tìm thêm một lượt bằng câu hỏi gốc.

**Chọn cấu hình.** Tiêu chí đã chốt trước khi chạy test (`benchmark/README.md`): loại cấu hình trả lời quá 1/3 câu thiếu bằng chứng, rồi xét lần lượt CER cao hơn → ít từ chối nhầm hơn → Hit@3 cao hơn → ít token hơn.
- Không cấu hình nào bị loại.
- A và D hoà ở CER 22/22, 0 câu từ chối nhầm và Hit@3 22/22. D thắng ở bước cuối vì ít token hơn: 3.604 so với 3.643 token/câu (`results/selection.json`).
- Mức chênh này rất nhỏ, nên trên dev tôi coi A và D gần như tương đương. Điểm D hơn rõ là Hit@1 (18/22 so với 14/22) và context không lẫn bản luật khác ở câu có nêu năm, nhưng Hit@1 không nằm trong tiêu chí đã chốt.
- E tốn thêm 33% token mà CER lại giảm 1 câu, nên tìm nhiều lượt không đáng với bộ câu này.

## Agent và đánh giá

- **Tool, số lượt gọi, điều kiện dừng, cách chọn context và xử lý lỗi:**
  - **Tool và lượt gọi:**
    - Agent có một tool `search_law(query)`. Phương pháp tìm (dense, BM25 hay hybrid) do cấu hình quyết định, model không được chọn, để so sánh công bằng.
    - Filter năm chỉ suy từ câu hỏi gốc của người dùng, không suy từ query model viết và không dùng gold.
    - Lượt gọi model đầu tiên đặt `tool_choice=required`, nên câu nào cũng có ít nhất một lần retrieval.
  - **Điều kiện dừng:**
    - Hết lượt tìm (`max_searches` 1 hoặc 3) thì gọi model thêm một lần với `tool_choice=none` để buộc trả lời.
    - Query trùng sau khi chuẩn hoá thì không tìm lại, chỉ báo cho model biết. Trùng 2 lần cũng buộc trả lời.
    - Trần số lượt gọi model là `max_searches + 3`.
  - **Chọn context:**
    - Khử trùng theo ID, giữ thứ tự hạng, gắn nhãn nguồn cho từng đoạn.
    - Ngừng thêm đoạn khi chạm 8.000 ký tự. Model chỉ thấy những đoạn đã được nhận vào context.
  - **Hậu kiểm:**
    - Câu trả lời phải có `[chunk-id]` và mọi ID phải nằm trong context; nếu không thì chuyển thành từ chối.
    - Model ghi "KHÔNG ĐỦ BẰNG CHỨNG" thì tính là từ chối.
    - System prompt ghi rõ nội dung tài liệu là dữ liệu, không phải chỉ dẫn.
  - **Xử lý lỗi:**
    - Lỗi mạng, 429 và 5xx được thử lại tối đa 3 lần. Lỗi model hoặc vector store ghi `status: "error"` kèm thông báo thật.
    - Ở các lần chạy đã ghi kết quả, không có lỗi nào.

| Cấu hình / split | Tool use | Số lượt gọi | Grounded answer 0/1/2 và trung bình | Citation hợp lệ | Câu có citation | Abstention đúng | Từ chối nhầm |
|---|---|---|---|---|---|---|---|
| A_dense / dev | 25/25 | 50 lượt model (2/câu); 1 lượt tìm/câu | 0 / 1 / 21 → 1,95/2 | 30/32 (93,8%) | 22/22 | 3/3 | 0/22 |
| B_bm25 / dev | 25/25 | 50; 1 tìm/câu | 1 / 1 / 20 → 1,86/2 | 27/29 (93,1%) | 21/21 | 3/3 | 1/22 |
| C_hybrid / dev | 25/25 | 50; 1 tìm/câu | 2 / 0 / 20 → 1,82/2 | 26/28 (92,9%) | 21/21 | 3/3 | 1/22 |
| D_dense_year / dev | 25/25 | 50; 1 tìm/câu | 0 / 0 / 22 → 2,00/2 | 32/32 (100%) | 22/22 | 3/3 | 0/22 |
| E_dense_year_multi / dev | 25/25 | 51; tìm 1 lượt: 2 câu, 2 lượt: 22, 3 lượt: 1 | 1 / 0 / 21 → 1,91/2 | 35/38 (92,1%) | 22/22 | 3/3 | 0/22 |
| A_dense / test | 11/11 | 22; 1 tìm/câu | 0 / 0 / 8 → 2,00/2 | 9/9 | 8/8 | 3/3 | 0/8 |
| D_dense_year / test | 11/11 | 22; 1 tìm/câu | 0 / 0 / 8 → 2,00/2 | 9/9 | 8/8 | 3/3 | 0/8 |

Số liệu bảng trên do `starter/agent_metrics.py` tổng hợp từ `results/agent/<split>/<cấu hình>/{traces,reviews}.json`, kết quả ghi ở `agent_metrics.json`.

Context của 8 câu test có đáp án, tôi chấm theo thang 0/1/2 (`results/agent/test/context_labels.json`): A được 8 câu mức 2, D cũng 8 câu mức 2, không câu nào mức 0 hay 1. Ở A, context của t04 và t07 lẫn 2023 Điều 65/66 cạnh đoạn 2010 cần dùng. Agent vẫn chọn đúng bản 2010, nhưng D sạch hơn vì đã lọc.

| Query ID | Loại câu | Tool call và kết quả | Câu trả lời và citation | Nhãn, lý do |
|---|---|---|---|---|
| t04 (D, test) | có đáp án, nêu năm | `search_law("biên bản hòa giải chữ ký các bên")`, filter 2010 suy từ "Theo Luật 2010"; top-1 `luat-59-2010-QH12:36:0` | Chữ ký các bên tham gia và chữ ký xác nhận của tổ chức, cá nhân tiến hành hòa giải [luat-59-2010-QH12:36:0] | Grade 2, citation đúng: Điều 36 khoản 2 nói nguyên văn. Context 2 |
| t02 (D, test) | có đáp án, ngoại lệ | 1 lượt tìm; context có 31:0 và 31:1 | Không bắt buộc với doanh nghiệp siêu nhỏ [31:0][31:1] | Grade 2; 31:0 nêu nghĩa vụ chung, 31:1 nêu ngoại lệ, cả hai đều đúng |
| tu3 (D, test) | thiếu bằng chứng | 1 lượt tìm; context có 2023 Điều 31, 57, 2010 Điều 31 | "KHÔNG ĐỦ BẰNG CHỨNG": chỉ có thời hạn phía doanh nghiệp (03 và 07 ngày làm việc), không có thời hạn khiếu nại của người tiêu dùng | Từ chối đúng; không thêm con số nào ngoài tài liệu |
| d04 (C, dev) | có đáp án, lỗi | Query model viết bỏ ý "tận cửa"; context 38:2, 38:1, 42:1, 23:1, 47:2, không có Điều 44 | Trả lời 03 ngày kể từ ngày nhận hợp đồng [47:2] và 30 ngày giao dịch từ xa [38:1] | Grade 0: sai căn cứ; 2 citation không hỗ trợ (Điều 47 là loại bán hàng khác, Điều 38 thiếu điều kiện) |
| d16 (B, dev) | có đáp án, từ chối nhầm | BM25 không ra Điều 3; model in chuỗi gọi tool giả sau khi hết lượt | Hậu kiểm chuyển thành "KHÔNG ĐỦ BẰNG CHỨNG" | Grade 0; hệ thống an toàn (không bịa) nhưng mất câu |

- **Đường dẫn trace, nhãn, lệnh chạy, thời gian và chi phí:**
  - Trace nằm ở `results/agent/{dev,test}/<cấu hình>/traces.json`. Mỗi câu ghi đủ tool call, query model viết, query thực dùng, filter, top-5 kèm score và hạng, đoạn thêm hoặc bỏ do budget, context cuối, câu trả lời gốc, citation, từng lượt gọi model (độ trễ, usage) và lỗi.
  - Nhãn tay nằm ở `reviews.json` cùng thư mục và `results/agent/test/context_labels.json`. Lệnh chạy xem README.
  - Tổng cộng: dev 251 lượt gọi model, 443.861 token vào (193.145 token trùng cache) và 38.064 token ra. Test 44 lượt gọi, 66.327 token vào và 6.056 token ra.
  - Ngoài ra có 4 lượt gọi thử API và 2 câu chạy thử agent trước khi chạy benchmark; các lượt này không lưu trace.
  - Agent chạy 6 luồng song song: median 2,2–2,8 giây/câu, cả bộ dev mỗi cấu hình mất 11–13 giây. Retrieval dense khoảng 35 ms/câu, BM25 dưới 1 ms/câu.
- **Kết quả trước/sau, ví dụ lỗi, đánh đổi và giới hạn:**
  - Trước/sau: lọc năm v1 → v2 nâng Hit@3 dev của dense + lọc năm từ 21/22 lên 22/22, Hit@1 từ 17 lên 18. Bỏ cụm phạm vi không cứu được d16 với hybrid.
  - Trên test, A và D bằng nhau ở mọi chỉ số của agent (8/8 câu grade 2, 3/3 từ chối đúng). D chỉ hơn ở Hit@1 (8/8 so với 7/8) và context sạch hơn ở 3 câu có nêu năm. Test 8 câu quá nhỏ để tách hai cấu hình.
  - Đánh đổi: hybrid và BM25 không hơn dense trên corpus nhỏ, một chủ đề này; hybrid còn làm CER của agent thấp hơn. Nhiều lượt tìm tốn thêm 33% token mà không tăng CER.
  - Giới hạn:
    - Nhãn grounded, citation và abstention do chính tôi chấm, có thể thiên lệch.
    - Benchmark nhỏ: 22 + 8 câu có đáp án, đa số chỉ cần một đoạn evidence.
    - Đáp án chuẩn và câu hỏi do tôi viết sau khi đã đọc luật.
    - Query do model viết có thể làm mất ý chính của câu hỏi (d04).
    - Chạy lại API có thể cho câu chữ khác dù temperature 0.
    - Luật suy năm là luật thủ công, chỉ bắt các mẫu "theo … năm".

## Thử thách nâng cao (nếu thực hiện)

Không thực hiện.

## Công cụ hỗ trợ

Tôi có dùng trợ lý lập trình để soạn nháp code và tài liệu. Code, số liệu và nhãn tôi tự chạy lại và đối chiếu với file kết quả.
