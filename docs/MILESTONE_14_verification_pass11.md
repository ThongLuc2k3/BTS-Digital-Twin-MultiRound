# Milestone 14 — Verification pass #11 (agent bị ngắt giữa chừng do hết session quota, điều phối viên hoàn thiện)

## Trạng thái hiện tại
**HOÀN TẤT — FAIL.** Bộ đếm "3 lần sạch liên tiếp" reset về 0/3.

## Bước tiếp theo
Chạy verification pass #12 (agent mới, độc lập).

## Lịch sử

### 2026-07-19 — Pass #11 tìm 2 bug thật rồi bị ngắt giữa chừng (hết session quota)
Agent verification #11 đọc đủ tài liệu bắt buộc (kể cả `docs/VERIFICATION_LOOP_
PROTOCOL.md` mới thêm), rồi đi đúng hướng đã gợi ý (đối chiếu SCENE người dùng chọn với
nội dung checkpoint thật tải về, và biên độ số học của `MAX_ERROR_WEIGHT`) — tìm ra
**2 bug thật, cả 2 đều verify bằng thực thi thật**:

1. **Tràn số (overflow) khi `--max_weight` vượt trần biểu diễn được của mask 16-bit
   PNG** — `05_generate_error_mask.py` mã hoá `pixel = round(weight * 1000)` vào
   `uint16` (trần 65535 → `weight` tối đa biểu diễn được là `65.535`). Agent verify
   bằng thực thi thật: `--max_weight=70` → ghi `uint16=4464` → đọc lại (đúng cách
   `apply_error_refine_patch.py` đọc) ra `weight=4.464` — THẤP HƠN NHIỀU so với 70 yêu
   cầu, không có lỗi/crash nào báo; `--max_weight=65.536` (chỉ nhích quá trần 0.001) →
   `uint16=0` → giải mã ra `weight=0.0` — NGƯỢC HẲN ý đồ "ưu tiên cao nhất" thành
   "trọng số thấp nhất có thể". Đã sửa: thêm validate cứng ở đầu `main()` — chặn
   `--max_weight > 65.535` (trần biểu diễn được) VÀ `--max_weight < 1.0` (dưới 1.0 sẽ
   làm GIẢM trọng số đúng ở vùng lỗi cao nhất, ngược thiết kế), báo lỗi rõ ràng thay vì
   để âm thầm sai.
2. **Không đối chiếu `SCENE` (biến notebook) với scene THẬT của checkpoint vừa tải từ
   Drive** — nếu người dùng dán NHẦM link Drive của 1 scene KHÁC (dễ xảy ra khi copy
   nhiều link cho 7 scene, hoặc mở nhầm tab trình duyệt), mọi assert sẵn có (`cfg_args`/
   `pipeline_train_flags.json` tồn tại) vẫn qua bình thường — refine sẽ ÂM THẦM chạy
   với hình học (Gaussian) của 1 scene KHÁC ghép vào camera/error-mask của `SCENE`
   trong notebook, không có lỗi/crash rõ ràng, chỉ lộ ra muộn qua Score cực kỳ tệ ở
   Bước 11 mà khó chẩn đoán đúng nguyên nhân. Đã sửa: đọc `cfg_args.source_path` của
   checkpoint vừa tải (luôn có dạng `.../work/<SCENE>/colmap/dense` — cả
   `02_train_baseline.sh` lẫn `06_train_refine.sh` đều dùng chung đúng quy ước này) để
   suy ra tên scene THẬT, đối chiếu với biến `SCENE` — chặn cứng nếu lệch, kèm thông
   báo rõ nguyên nhân khả dĩ.

Agent áp cả 2 fix vào `kaggle_round2_refine.ipynb` xong thì **bị dừng đột ngột do hết
hạn mức phiên làm việc** (session limit, KHÔNG phải lỗi kỹ thuật) ngay khi đang chuẩn
bị áp fix #2 (cross-check scene) sang `kaggle_round3_refine.ipynb` — để lại 2 file
uncommitted: `pipeline/scripts/05_generate_error_mask.py` (cả 2 fix, hoàn chỉnh) và
`pipeline/kaggle_round2_refine.ipynb` (chỉ fix #2, hoàn chỉnh).

### 2026-07-19 — Điều phối viên hoàn thiện
- Verify lại `05_generate_error_mask.py`: `python -m py_compile` sạch; chạy thật (cài
  `opencv-python-headless`+`lpips` vào venv riêng) với `--max_weight 70` và
  `--max_weight 0.5` — cả 2 đều bị chặn đúng như thiết kế, thông báo lỗi rõ ràng, không
  crash mù mờ.
- Áp fix #2 (cross-check `SCENE`) sang `kaggle_round3_refine.ipynb` — lấy NGUYÊN nội
  dung cell đã sửa ở `kaggle_round2_refine.ipynb` (2 notebook dùng chung logic Bước 6
  gần như y hệt, chỉ khác số thứ tự vòng), tìm đúng cell tương ứng (nhận diện qua chuỗi
  `"gdown --folder"` duy nhất trong file) rồi thay thế.
- Nhân tiện chuẩn hoá lại định dạng: cell mà pass #11 sửa ở `kaggle_round2_refine.ipynb`
  bị lưu dưới dạng **1 chuỗi lớn duy nhất** (`source` là string) thay vì **list nhiều
  dòng** (`source` là list các dòng, kèm `\n`) như MỌI cell khác trong cả 4 notebook —
  cả 2 dạng đều hợp lệ theo chuẩn `nbformat` (không phải bug chức năng) nhưng KHÔNG
  nhất quán, dễ gây khó đọc/khó diff sau này. Đã chuẩn hoá lại cả 2 file (round2 và
  round3) về đúng dạng list-nhiều-dòng.
- `nbformat.validate()` cả 2 notebook sau khi sửa — hợp lệ.
- Chạy lại toàn bộ `tests/` — sạch, không có regression.

**Kết luận: 2 bug thật đã tìm + sửa xong hoàn chỉnh (cả 2 notebook Vòng 2/3 đều nhất
quán) — pass này tính là FAIL.** Bộ đếm 3 lần sạch liên tiếp reset về 0/3, cần pass #12
(agent mới) chạy lại từ đầu.
