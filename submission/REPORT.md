# Lab 21 — Báo cáo đánh giá fine-tuning

**Họ tên**: Lê Mạnh Cường
**MSSV**: 2A202602604
**Ngày chạy full eval**: 07/10/2026
**Tier / GPU**: T4 / Tesla T4 (log báo 14.6 GB VRAM khả dụng)
**Base model**: `unsloth/Qwen3.5-4B`
**Độ chính xác huấn luyện**: fp16

## 1. Thiết lập

| Mục | Giá trị |
|---|---|
| Dataset | 250 ticket CSKH tiếng Việt → JSON triage |
| Train / validation | 225 / 25 (seed 42) |
| Target eval | 50 mẫu |
| Regression eval | 15 mẫu |
| `MASK_MODE` | `assistant-only` |
| Epochs / `max_steps` | 2 epochs / 30 bước mỗi run |
| P95 token / NB1 gợi ý | 98 / `max_length=256` |
| `max_length` thực tế NB3 | 1024 |

Tôi dùng model và corpus mặc định của tier T4 để làm thí nghiệm có metric khách quan
trên ticket CSKH, không cần LLM judge. Cấu hình T4 của repo đặt `max_length=1024`,
đúng như log NB3; NB1 đo p95=98 và gợi ý 256. Tôi giữ cấu hình tier có sẵn, nhưng đây
là khoảng cách lớn so với độ dài quan sát được và không được so sánh với một run ở 256.
Vì vậy kết quả không chứng minh rằng 1024 tốt hơn 256; lần thử tiếp theo nên giảm độ
dài sau khi kiểm tra truncation và so sánh trong cùng điều kiện.

## 2. Template và loss mask (NB1)

Template kiểm tra giữ lại khối `<think>` (`ok=true`, `open_tag_present=true`,
`body_present=true`). Điều này chỉ xác nhận template hỗ trợ reasoning trace; corpus
mặc định chứa câu trả lời JSON, không có trace reasoning thực. Do đó
`valid_trace_rate=0.0` trong kết quả không phải phép đo chất lượng suy luận trên bộ
dữ liệu này.

| Kiểm tra | Kết quả |
|---|---:|
| `supervised_fraction` | 0.4149 (39/94 token trong ví dụ proof) |
| Câu trả lời nằm trong loss | Có |
| Câu hỏi bị mask khỏi loss | Có |
| `token_stats.json` p95 / p99 / max | 98 / 100 / 101 |

Hai assert trong `mask_proof.json` đều đúng. Phần được giám sát chứa câu trả lời JSON;
system prompt và ticket người dùng nằm trong phần mask. Điều này xác nhận model được
train để sinh đáp án, không phải học lại prompt.

## 3. Baseline và kết quả bốn nhóm (NB2/NB5)

Baseline được đóng băng trước khi train; SHA của optimized prompt khớp với prompt
trong repo. Lần chạy mới ghi `eval_limit=null`, `smoke_mode=false`, `n_target=50` và
`n_regression=15`, tức là dùng đủ hai tập eval được cung cấp.

| Run | Target | Regression | Format | Latency (ms/mẫu) |
|---|---:|---:|---:|---:|
| (a) Base + naive prompt | 0.0000 | 0.7911 | 0.0000 | 3288.9 |
| (b) Base + optimized prompt | 0.7650 | 0.7911 | 1.0000 | 1076.1 |
| (c) LoRA fine-tune | 0.9700 | 0.5222 | 1.0000 | 1448.5 |

Prompt (b) cải thiện target so với (a), từ 0 lên 0.765, và format từ 0 lên 1.0. So với
(b), LoRA tăng target `+0.2050` (20.5 điểm phần trăm) và giữ format ở 1.0, nhưng
regression giảm `-0.2689` (26.89 điểm phần trăm). Latency của LoRA tăng khoảng
372.4 ms/mẫu, tương đương 34.6% so với (b). Như vậy model học tốt hơn tác vụ hẹp nhưng
đánh đổi đáng kể khả năng tổng quát theo regression set.

## 4. Đối chứng NB4

| Run | Thay đổi so với `correct` | Rank | Trainable params | Loss cuối | VRAM đỉnh (GB) | Target NB5 (50 mẫu) |
|---|---|---:|---:|---:|---:|---:|
| `correct` | all text-linear, LR 1e-4, 16-bit | 16 | 32,464,896 | 0.6257* | 8.78 | 0.970 |
| `attn_only` | chỉ q,v; rank tăng để khớp ngân sách | 283 | 32,456,704 | 0.5377 | 8.79 | 0.970 |
| `wrong_lr` | LR giảm 10 lần xuống 1e-5 | 16 | 32,464,896 | 1.5702 | 8.78 | 0.000 |
| `qlora` | trọng số base 4-bit | 16 | 32,464,896 | 0.7058 | 3.86 | 0.940 |

`*` `runs.csv` có ba dòng `correct` cùng cấu hình, cùng 30 bước, với thời gian 398.9,
404.1 và 418.2 giây và loss lần lượt 0.6257, 0.6255 và 0.6263. Bảng dùng dòng đầu;
NB5 chấm adapter `correct` đang có trong ZIP. File CSV không liên kết rõ adapter cuối
với một trong ba dòng lặp, nên không diễn giải khác biệt nhỏ giữa các loss như một
thí nghiệm độc lập.

Cả bốn cấu hình có cùng step budget 30. `attn_only` có số tham số train được lệch
`correct` khoảng 0.025%, đạt target bằng `correct` và có format bằng 1.0. Với tác vụ
này, kết quả chưa cho thấy việc gắn adapter vào nhiều lớp hơn có lợi thêm so với
attention-only khi ngân sách tham số được khớp. `wrong_lr` có loss cao nhất và target
0.000, cho thấy LR 1e-5 không học được tác vụ trong ngân sách 30 bước này. QLoRA giảm
VRAM từ 8.78 xuống 3.86 GB, tiết kiệm 4.92 GB (khoảng 56%), trong khi target thấp hơn
`correct` 0.03; đây là đánh đổi quan sát được trên run cụ thể, không phải kết luận
tổng quát về mọi cấu hình QLoRA.

## 5. Phán quyết

**Kết quả regression gate: FAILED.** Target delta là `+0.2050`, nhưng regression delta
là `-0.2689`; ngưỡng cho phép của gate chỉ là giảm tối đa 0.020.

Đây là kết quả đầy đủ trên 50 target và 15 regression mẫu, không phải smoke verdict.
Không nên triển khai adapter hiện tại chỉ vì target đạt 0.970: regression giảm từ
0.7911 xuống 0.5222, vượt xa mức suy giảm chấp nhận được. Cách diễn giải hợp lý là
fine-tune chuyên môn hóa tốt cho triage nhưng làm quên một phần đáng kể khả năng được
đo bởi bộ regression. Theo gợi ý của lab, hướng tiếp theo là thử thêm 1–5% dữ liệu
replay tổng quát, sau đó train và đánh giá lại trên cùng các tập đã đóng băng. Cần giữ
nguyên baseline (b), checksum và tiêu chí gate; không sửa eval hoặc nới tolerance để
đổi FAILED thành PASSED. Cho đến khi một run mới vừa giữ được target gain vừa giảm
regression trong giới hạn, phán quyết triển khai của tôi là **không deploy**.

## 6. Phân tích định tính

`qualitative.json` có 50 mẫu, điểm fine-tune là 1.00 ở 44 mẫu và 0.75 ở 6 mẫu. Sáu
trường hợp 0.75 đều hiển thị ba trường đầu khớp nhãn và urgency dự đoán là
`trung_binh` trong khi nhãn là `thap`; do điểm tổng là 0.75/4, lỗi quan sát được nằm ở
urgency. Ví dụ tốt có điểm 1.00. ZIP không lưu output baseline (b) theo từng ticket và
`ft_pred` chỉ lưu chuỗi dự đoán bị cắt ngắn, vì vậy không thể trung thực khẳng định
ticket cụ thể nào fine-tune thắng hoặc thua baseline (b). Bảng dưới so sánh dự đoán
fine-tune với nhãn chuẩn; đây không phải paired comparison với baseline.

| # | Ticket (rút gọn) | Nhãn chuẩn | Quan sát fine-tune | Điểm |
|---:|---|---|---|---:|
| 1 | Hoàn tiền nồi chiên; “Khi nào tiện”, “Quá tệ” | `hoan_tien / thap / nồi chiên không dầu / tieu_cuc` | `urgency=trung_binh` (sai; các trường hiển thị khác khớp) | 0.75 |
| 2 | Giao hàng chậm, đèn bàn LED; khách cảm ơn | `van_chuyen / thap / đèn bàn LED / tich_cuc` | `urgency=trung_binh` (sai) | 0.75 |
| 3 | Đèn bàn LED sai màu; khách nói shop hỗ trợ tốt | `san_pham_loi / thap / đèn bàn LED / tich_cuc` | `urgency=trung_binh` (sai) | 0.75 |
| 4 | Máy xay sinh tố, yêu cầu hoàn/đổi | `doi_tra / trung_binh / máy xay sinh tố / tich_cuc` | Bốn trường khớp nhãn theo điểm mẫu | 1.00 |
| 5 | Hỏi còn hàng balo laptop không | `hoi_thong_tin / thap / balo laptop / tich_cuc` | Bốn trường khớp nhãn theo điểm mẫu | 1.00 |

Các lỗi được lưu gợi ý model hay gán urgency trung bình cho câu có “khi nào tiện” dù
nhãn là thấp. Đây là lỗi dự đoán theo ticket; chưa thể kết luận nó là lỗi mới do
fine-tuning hay baseline (b) cũng mắc cùng lỗi. Để hoàn tất tiêu chí so sánh định tính
của rubric, cần lưu output baseline (b) và fine-tune đầy đủ cho cùng ít nhất 5 ticket,
bao gồm ít nhất 2 trường hợp fine-tune thực sự thua baseline.

## 7. Kết luận và điều học được

Trên full eval, LoRA cải thiện target từ 0.765 lên 0.970 và giữ format ở 1.000, nhưng
regression giảm từ 0.7911 xuống 0.5222. Do mức giảm 0.2689 lớn hơn tolerance 0.020,
regression gate trả về FAILED và tôi không khuyến nghị deploy adapter này. Kết quả cho
thấy chỉ tối ưu tác vụ mục tiêu là chưa đủ: một model phân loại ticket chính xác hơn
nhưng quên nhiều khả năng phổ thông có thể không phù hợp nếu sản phẩm cần cả hai. Các
đối chứng bổ sung cho thấy attention-only với rank được khớp ngân sách đạt cùng target
với all-text-linear; do đó số liệu này không ủng hộ việc tăng placement như một lợi ích
độc lập trên tác vụ đã thử. LR thấp hơn 10 lần không học được target trong 30 bước,
còn QLoRA tiết kiệm khoảng 56% VRAM với mức giảm target 0.03 so với `correct`. Thí
nghiệm tiếp theo nên tập trung vào replay dữ liệu tổng quát để xử lý regression,
giữ nguyên full eval và kiểm tra lại cả target lẫn regression. Tôi cũng sẽ lưu dự đoán
paired theo từng ticket để biết lỗi urgency của adapter có phải lỗi riêng của nó hay
đã có ở baseline prompt tốt.

**Ba điều tôi học được**

1. Phải đánh giá đồng thời target và regression; target cao không đủ để quyết định deploy.
2. So sánh adapter cần giữ cùng số bước và ngân sách tham số; `attn_only` được khớp
   ngân sách nên phép thử placement có ý nghĩa hơn so sánh rank cố định.
3. Log dự đoán theo từng ví dụ là cần thiết để phân biệt lỗi chung với lỗi do
   fine-tuning; aggregate score không trả lời câu hỏi đó.

## 8. Tình trạng và việc tiếp theo

- Full eval đã chạy: 50 target, 15 regression; `eval_limit=null`, `smoke_mode=false`.
- `results/verdict.json` ghi FAILED vì regression giảm quá tolerance. Đây là phán quyết
  cần báo cáo, không phải lỗi cần “sửa” bằng cách đổi ngưỡng.
- Cần lưu paired outputs của baseline (b) và fine-tune nếu muốn đáp ứng đầy đủ yêu cầu
  định tính “ít nhất 2 ví dụ fine-tune thua”.
- `runs.csv` ghi ba lần `correct` cùng cấu hình; giữ nguyên các dòng để minh bạch.
- Chạy lại `make verify` trong môi trường Linux/Colab sau khi đóng gói; báo cáo này đã
  được điền họ tên/MSSV và không còn placeholder trong mẫu.
