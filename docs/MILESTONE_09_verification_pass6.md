# Milestone 09 — Verification pass #6 (agent bị ngắt giữa chừng do hết session quota, điều phối viên hoàn thiện)

## Trạng thái hiện tại
**HOÀN TẤT — FAIL** (tìm + sửa thêm bug thật). Bộ đếm "3 lần sạch liên tiếp" reset về 0.
Xem lịch sử để biết chi tiết.

## Bước tiếp theo
Chạy verification pass #7 (agent mới, độc lập).

## Lịch sử

### 2026-07-18 — Pass #6 bị ngắt giữa chừng (hết session quota, không phải lỗi kỹ thuật)
- Agent verification #6 đọc đủ tài liệu bắt buộc, xác nhận lại các fix của pass #5
  (`--fuzzy` đã bị xoá khỏi MỌI lệnh `gdown`, cả dạng file đơn lẫn `--folder`), rồi bắt
  đầu Part C (kiểm tra các gói pip khác chưa pin phiên bản) thì **bị dừng đột ngột do
  hết hạn mức phiên làm việc** (session limit, reset 19h giờ Bangkok — KHÔNG phải phát
  hiện lỗi kỹ thuật nào khiến nó dừng). Trước khi bị ngắt, agent đã kịp áp 1 fix (chưa
  commit): pin `gdown` thành `"gdown>=6,<7"` ở cả 4 notebook (quyết định hợp lý theo
  đúng gợi ý trong yêu cầu — pin phiên bản chắc chắn còn `--fuzzy` KHÔNG tồn tại nữa
  thay vì chỉ sửa cú pháp cho phiên bản mới nhất HIỆN TẠI, dễ tái diễn nếu `gdown` đổi
  tiếp trước deadline 30/07).

### 2026-07-18 — Điều phối viên tiếp tục hoàn thiện Part C + D
- Verify lại bằng `gdown --help` thật (cài bản 6.1.0 mới nhất) — xác nhận `--fuzzy`
  **không tồn tại trong TOÀN BỘ CLI**, kể cả chế độ `--folder` — nên việc pass #5 bỏ cờ
  này khỏi cả 2 dạng lệnh (file đơn + `--folder`) là ĐÚNG và ĐỦ, không cần sửa gì thêm
  cho `--folder`.
- Rà các gói pip khác trong cùng dòng lệnh cài đặt (`pycolmap`, `scikit-image`,
  `lpips`, `plyfile`, `tqdm`) — phát hiện `pycolmap` **không có version floor** (khác
  với `requirements.txt` của repo tiền nhiệm, vốn ghi rõ `pycolmap>=3.10` — API
  `pycolmap` từng có thay đổi giữa các bản, xem lịch sử debug ở repo tiền nhiệm). Đã
  thêm lại đúng floor đã được kiểm chứng: `"pycolmap>=3.10"`.
- **Bug thật tìm thêm (nghiêm trọng, độc lập với gdown)**: `pipeline/scripts/
  05_generate_error_mask.py` dùng `import cv2` (Gaussian blur + ghi PNG 16-bit) nhưng
  **KHÔNG notebook nào cài `opencv-python`** — verify bằng cách kiểm tra thật dòng
  lệnh `!pip install` ở cả 4 notebook (không có "opencv" ở đâu cả) + test cục bộ
  `import cv2` trong môi trường Python sạch (`ModuleNotFoundError`). Không thể giả định
  Kaggle có cài sẵn `cv2` hay không (dù nhiều khả năng CÓ, ảnh Docker chuẩn của Kaggle
  thường có sẵn OpenCV — nhưng đây là SUY ĐOÁN, không phải xác nhận, đúng tinh thần
  "test thật, không suy đoán" của `docs/PORTED_KNOWLEDGE.md` mục 6). Nếu không cài sẵn,
  cell Vòng 2/3 sẽ crash `ModuleNotFoundError` ngay khi sinh error mask. Đã thêm
  `opencv-python-headless` (bản không phụ thuộc GUI/X11, phù hợp môi trường server như
  Kaggle) vào dòng `!pip install` của cả 4 notebook (thêm ở cả 4 cho nhất quán, dù
  `kaggle_round1_baseline.ipynb`/`kaggle_submission.ipynb` không trực tiếp gọi
  `05_generate_error_mask.py` — cùng nguyên tắc "cài đủ superset các gói cả họ script
  cần" đã áp dụng sẵn cho `pycolmap`/`lpips` trong chính các notebook đó).
- Fresh clone thật `graphdeco-inria/gaussian-splatting` (không dùng lại clone cũ nào)
  hôm nay, checkout đúng commit đã pin, áp `apply_error_refine_patch.py` — 6/6 chỗ vá
  sạch, `py_compile` sau vá không lỗi. Xác nhận: pin theo COMMIT HASH (khác pin theo
  version pip như `gdown`) **miễn nhiễm hoàn toàn** với việc upstream đổi code sau này
  — checkout cùng 1 hash luôn cho nội dung file giống hệt bất kể lúc nào clone. Không
  cần lo lắng thêm cho hướng rủi ro này.
- Đánh giá claim "timeout 600 giây" (Đề_bài.md, mục "Chi tiết vòng thi") — kết luận:
  đây nhiều khả năng là thời gian chờ xử lý phía hạ tầng chấm điểm của BTC (sau khi
  thí sinh đã nộp `submission.zip` hoàn chỉnh qua portal web), KHÔNG áp dụng cho
  notebook train/render của thí sinh (không có bước nào trong pipeline này chạy "bên
  trong" 1 cửa sổ chờ do BTC kiểm soát — thí sinh tự chạy Kaggle, tự tạo file, rồi mới
  upload). `docs/00_MASTER_PLAN.md` đã ghi chú đúng mức độ thận trọng cần thiết (coi là
  giả định chưa xác nhận) — không cần sửa code gì thêm cho việc này, chỉ là xác nhận
  lại cách hiểu hiện tại hợp lý.
- Verify lại toàn bộ 4 notebook bằng `nbformat.validate()` sau các thay đổi trên — hợp
  lệ cả 4.

**Kết luận: bug thật đã tìm thêm (thiếu `opencv-python`) — pass này tính là FAIL, dù
phần "gdown pinning" agent gốc làm là 1 cải tiến hợp lý chứ chưa hẳn là bug độc lập.**
Bộ đếm 3 lần sạch liên tiếp reset về 0/3, cần pass #7 (agent mới) chạy lại từ đầu.
