# Benchmark Day 7

| Split | Có đáp án | Thiếu bằng chứng | Tổng |
|---|---:|---:|---:|
| dev | 22 (`d01`–`d22`) | 3 (`du1`–`du3`) | 25 |
| test | 8 (`t01`–`t08`) | 3 (`tu1`–`tu3`) | 11 |

- `questions.json`: câu có đáp án, đúng schema `starter/baseline.py` (`id`, `split`, `question`, `expected_answer`, `gold[{source_id, page, evidence}]`). `page` = số Điều. Các phần tử `gold` là evidence thay thế (đúng một trong số đó là đủ).
- `cases.json`: mọi câu (`answerable`, `expected_behavior`); câu thiếu bằng chứng có `absence_reason` và `absence_check` (cách tôi đã kiểm trong corpus).
- `lock.json`: sha256 của hai file trên, ghi **trước** khi chạy bất kỳ cấu hình nào trên test.

## Nguồn câu hỏi và chia split

- `d01`–`d16`: 16 câu dev của Day 6.
- `d17`–`d22`: 6 câu test Day 6 (`t01`–`t06` cũ). Day 6 đã chạy và báo cáo trên chúng nên tôi không coi là test độc lập nữa, chuyển sang dev.
- `t01`–`t08`: test mới viết cho Day 7, lấy từ các Điều chưa có trong bộ câu Day 6 (2023: Điều 31, 46, 58, 66, 69; 2010: Điều 36, 37). Có 3 câu nêu năm luật (`t04`, `t05`, `t07`), câu ngoại lệ/điều kiện (`t02`, `t03`, `t05`). Rủi ro gần chủ đề với dev: `t04` (chữ ký biên bản hòa giải 2010) gần `d08` (chữ ký kết quả thương lượng 2023); `t01` gần `d20` (thương lượng 07 ngày).
- Câu thiếu bằng chứng gần chủ đề: tiền phạt, bảo hành tối thiểu, số tổng đài (dev); phí hòa giải, tỷ lệ hoa hồng đa cấp, thời hạn khiếu nại của người tiêu dùng (test). Mỗi câu ghi chuỗi đã tìm và Điều đã đọc toàn văn; tôi không kết luận "thiếu" chỉ vì top-k không thấy.

`python -m src.check_benchmark` kiểm mỗi evidence nằm trọn (khớp nguyên văn) trong ít nhất một chunk đúng nguồn và đúng Điều của `data/chunks.json`, và so sha256 với `lock.json`.

## Định nghĩa chỉ số (chốt trước khi chạy)

- **Hit@1 / Evidence Hit@3**: chunk ở hạng 1 / một trong 3 hạng đầu có đúng `source_id`, đúng `page` và chứa trọn một evidence của câu. Mẫu số = số câu có đáp án. Dùng query gốc (câu hỏi), filter ghi riêng.
- **Context Evidence Rate (CER)**: context cuối mà agent gửi model (gộp mọi lượt tìm, sau khi cắt theo ngân sách) có chunk hit / số câu có đáp án.
- Câu thiếu bằng chứng không vào mẫu số retrieval.

## Tiêu chí chọn cấu hình (chốt trước khi chạy test)

Chỉ dùng dev:

1. Loại cấu hình nào trả lời (không từ chối) quá 1 trong 3 câu dev thiếu bằng chứng.
2. Chọn CER của agent trên dev cao nhất.
3. Hòa thì ít câu có đáp án bị từ chối nhầm hơn.
4. Hòa tiếp thì Evidence Hit@3 (lượt tìm đầu, query gốc) cao hơn.
5. Hòa tiếp thì ít token trung bình mỗi câu hơn.

Nhãn tay (grounded answer, citation) dùng để báo cáo, không dùng để chọn, vì nhãn là bản tôi tự chấm. Sau khi chốt, chạy đúng một lần baseline và cấu hình chọn trên test; không chỉnh gì theo kết quả test.
