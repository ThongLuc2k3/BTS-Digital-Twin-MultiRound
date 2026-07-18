# BTS Digital Twin — Multi-Round Progressive Refinement

Pipeline Novel View Synthesis (3D Gaussian Splatting) cho bài thi **Viettel AI Race
2026 — Bài 1 BTS Digital Twin**. Kiến trúc: **train nhiều vòng liên tiếp** — Vòng 1 ra
baseline, Vòng 2+ tự đo vùng ảnh còn lỗi (so với ảnh train GT thật) rồi tinh chỉnh có
hướng dẫn, lặp lại tới khi hết ngân sách thời gian hoặc hết cải thiện.

**Đọc trước khi làm bất cứ gì:** [`docs/00_MASTER_PLAN.md`](docs/00_MASTER_PLAN.md) —
kiến trúc, lý do thiết kế, deadline thật (Vòng 1: **30/07/2026**). Sau đó đọc
[`docs/PORTED_KNOWLEDGE.md`](docs/PORTED_KNOWLEDGE.md) — danh sách bug đã tìm ra + sửa
ở repo tiền nhiệm (`BTS-Digital-Twin`), bắt buộc không lặp lại.

Trạng thái/tiến độ chi tiết: xem `docs/MILESTONE_*.md` (mỗi file 1 hạng mục công việc,
luôn có mục "Trạng thái hiện tại" + "Bước tiếp theo" cập nhật — đọc file mới nhất theo
số thứ tự để biết đang ở đâu nếu bị ngắt quãng).

Baseline engine: [`graphdeco-inria/gaussian-splatting`](https://github.com/graphdeco-inria/gaussian-splatting)
(đúng baseline BTC gợi ý trong đề bài).
