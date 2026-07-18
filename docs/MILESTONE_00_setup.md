# Milestone 00 — Khởi tạo repo + nền tảng dùng chung

## Trạng thái hiện tại
**HOÀN TẤT.** Repo skeleton + tài liệu kiến trúc + `pipeline/common/` đã port và verify
xong. Sẵn sàng cho các milestone tiếp theo (01, 02, 03) chạy song song.

## Bước tiếp theo
Xem `docs/MILESTONE_01_round1_baseline.md`, `docs/MILESTONE_02_refine_pipeline.md`,
`docs/MILESTONE_03_submission_and_testing.md` — 3 hạng mục làm song song bởi 3 agent
riêng, mỗi agent tự cập nhật đúng file milestone của mình.

## Lịch sử

### 2026-07-18 — Khởi tạo
- Tạo repo GitHub mới `ThongLuc2k3/BTS-Digital-Twin-MultiRound` (private) bằng `gh repo
  create`, clone về `/home/thongluc/Khóa Luận Tốt Nghiệp/BTS-Digital-Twin-MultiRound`.
- Đọc lại `Đề_bài.md` gốc (không suy đoán từ trí nhớ) — phát hiện deadline thật Vòng 1:
  **30/07/2026** (lúc viết file này còn ~12 ngày). Ghi vào `docs/00_MASTER_PLAN.md`.
- Viết `docs/00_MASTER_PLAN.md` (kiến trúc multi-round, tóm tắt đề bài, phân công) và
  `docs/PORTED_KNOWLEDGE.md` (toàn bộ bug/bài học thật đã tìm ra ở repo `BTS-Digital-
  Twin` — bug scale COLMAP, bug antialiasing/cfg_args, bug spatial_lr_scale khi resume
  từ .ply, cạm bẫy Jupyter/IPython, quy tắc đóng gói submission...).
- Port `pipeline/common/` (`scenes.py`, `poses.py`, `colmap_runner.py`,
  `logging_utils.py`, `alignment.py`, `__init__.py`) nguyên vẹn từ `BTS-Digital-Twin`
  — đây là code đã kiểm chứng thật (7/7 scene COLMAP chạy sạch, holdout split verify
  byte-exact). Verify lại: `py_compile` sạch + import + gọi `get_scene()` thành công.
  Không cần sửa gì (độ sâu thư mục `pipeline/common/` giống hệt repo gốc nên
  `Path(__file__).resolve().parents[2]` trong `scenes.py` vẫn đúng).
- Tạo `.gitignore` (dataset/checkpoint/gaussian-splatting clone/zip — không lưu trong
  git, quá nặng) + `README.md` trỏ vào tài liệu.
- **User đã cấp toàn quyền tự chủ**: không cần hỏi lại, tự tạo agent kiểm tra liên tục
  cho tới khi có 3 lần XÁC NHẬN LIÊN TIẾP không lỗi (qua mô phỏng, vì không có GPU cục
  bộ) mới coi là xong.
