# Milestone 01 — Vòng 1 (baseline): pipeline + notebook

## Trạng thái hiện tại
**HOÀN TẤT.** Toàn bộ pipeline Vòng 1 (baseline) đã port + refactor xong, verify cục
bộ đầy đủ (không có GPU cục bộ nên chỉ verify được cú pháp/logic/luồng điều khiển —
train/render thật CHỈ verify được trên Kaggle, xem `docs/PORTED_KNOWLEDGE.md` mục 6).

File đã tạo (thuộc sở hữu milestone này, KHÔNG đụng vào file của agent Vòng 2+/agent
submission-packaging):
- `pipeline/scripts/00_make_holdout_split.py` — tạo holdout nội bộ từ ảnh train.
- `pipeline/scripts/01_run_colmap.py` — undistort COLMAP (dùng sparse có sẵn của BTC).
- `pipeline/scripts/02_train_baseline.sh` — train 3DGS 1 lần/scene (đổi tên/đơn giản
  hoá từ `03_train_3dgs.sh` của repo tiền nhiệm — bỏ hẳn antenna-focus/depth-prior/
  exposure-comp, chỉ giữ antialiasing vì đã đo có lợi thật).
- `pipeline/scripts/03_render_test_poses.py` — render pose tuỳ ý (test thật hoặc
  holdout) từ checkpoint đã train.
- `pipeline/scripts/04_eval_metrics.py` — tự chấm PSNR/SSIM/LPIPS/Score trên holdout.
- `pipeline/kaggle_round1_baseline.ipynb` — notebook Kaggle chạy tuần tự cả pipeline
  cho 1 scene/lần, hỗ trợ `MODE="holdout"` (đo Score) và `MODE="final"` (checkpoint
  nộp bài/làm input Vòng 2+).

## Đã verify (cục bộ, không cần GPU)
- `python -m py_compile` sạch cho cả 4 file `.py` (`00_make_holdout_split.py`,
  `01_run_colmap.py`, `03_render_test_poses.py`, `04_eval_metrics.py`).
- `bash -n pipeline/scripts/02_train_baseline.sh` — cú pháp sạch.
- `nbformat.validate()` cho `pipeline/kaggle_round1_baseline.ipynb` — hợp lệ (22 cell).
- **`02_train_baseline.sh` test end-to-end bằng `train.py` GIẢ** (mock: in tiến độ giả,
  ghi `.ply`/`cfg_args` giả, hỗ trợ `CRASH_AT=<iter>` để mô phỏng crash), chạy trong
  thư mục scratch riêng (`/tmp`, đã dọn sạch sau khi test xong, không để lại rác trong
  repo). Các kịch bản đã test:
  1. **Thành công, ITERATIONS nhỏ (20)** — checkpoint cuối được tạo, `cfg_args` +
     `pipeline_train_flags.json` được ghi đúng schema
     (`{"antialiasing": true, "depth_prior": false, "exposure_comp": false,
     "antenna_focus": false}`), `colmap/dense/images/` bị xoá dọn đĩa sau khi xong.
  2. **Thành công, ITERATIONS=15000 (khớp mốc checkpoint thật)** — xác nhận CẢ 2
     checkpoint giữa chừng (`iteration_7000/`, `iteration_15000/`) đều được tạo đúng
     (không chỉ checkpoint cuối) — verify cơ chế `SAVE_ITERATIONS` hoạt động đúng.
  3. **`ANTIALIASING=0`** — `pipeline_train_flags.json` ghi đúng `antialiasing: false`.
  4. **Crash giữa chừng SAU khi đã qua 1 checkpoint** (`CRASH_AT=9000`, đã qua
     checkpoint 7000) — script in `[LỖI]` + 50 dòng cuối log ra stderr, in đúng
     `[CỨU ĐƯỢC]` trỏ tới checkpoint `iteration_7000` còn dùng được, **exit code
     truyền đúng lên trên** (khớp exit code thật của `train.py`, ở đây test với exit
     127 và exit 1 tuỳ tình huống).
  5. **Crash giữa chừng TRƯỚC khi có checkpoint nào** (`CRASH_AT=2000`) — không in
     dòng `[CỨU ĐƯỢC]` (đúng, vì chưa có checkpoint nào để cứu), vẫn exit lỗi đúng.
  6. **Thiếu biến `GS_REPO`** — báo lỗi rõ, exit 1, không chạy tiếp.
  7. **`GS_REPO` trỏ sai đường dẫn** (thiếu `train.py`) — báo lỗi rõ, exit 1.
  8. **Scene thiếu `colmap/dense/sparse/0`** — in `[BỎ QUA]` cho đúng scene đó, KHÔNG
     làm chết cả lệnh (quan trọng khi truyền nhiều scene trong 1 lần gọi).
  9. **`ANTIALIASING=1` nhưng `$GS_REPO` là bản clone CŨ** (không có cờ
     `--antialiasing` trong `arguments/__init__.py`) — báo lỗi rõ, exit 1, không âm
     thầm train sai cấu hình.
  10. **Xác nhận `|| true` sau `grep` trong vòng lặp poll tiến độ hoạt động đúng**:
      script bật `set -euo pipefail`, nếu không có `|| true` thì lần poll đầu tiên
      (log chưa có dòng tiến độ nào, grep exit 1) sẽ giết chết cả script — đã test
      thực nghiệm cơ chế poll chạy hết vòng lặp tới khi train xong ở tất cả kịch bản
      trên mà không bị chết giữa chừng, xác nhận `|| true` hoạt động đúng như tài
      liệu ở `docs/PORTED_KNOWLEDGE.md` mục 6 mô tả.

## Giới hạn đã biết (chưa/không thể verify cục bộ)
- **Train/render 3DGS thật, chất lượng Score thật trên dữ liệu thật** — cần GPU CUDA,
  chỉ chạy được trên Kaggle. Bản thân `train.py` thật của
  `graphdeco-inria/gaussian-splatting` chưa từng chạy qua trong milestone này (chỉ
  mock) — rủi ro còn lại: sai lệch API/tham số dòng lệnh nếu commit pin
  `54c035f7834b564019656c3e3fcc3646292f727d` có thay đổi tên cờ nào đó (đã đối chiếu
  bằng cách đọc lại đúng phần liên quan trong `docs/PORTED_KNOWLEDGE.md`/code cũ đã
  chạy thật ở repo tiền nhiệm — không tự suy đoán mới).
- `pycolmap`/`lpips`/`scikit-image` chưa cài trong môi trường phát triển cục bộ hiện
  tại của milestone này nên `00_make_holdout_split.py`/`01_run_colmap.py`/
  `04_eval_metrics.py` mới verify được `py_compile` (cú pháp), KHÔNG chạy thực tế với
  dữ liệu COLMAP thật cục bộ (khác một phần triết lý lý tưởng ở
  `docs/PORTED_KNOWLEDGE.md` mục 6 — mục đó khuyến nghị test bằng dữ liệu thật nếu có
  sẵn cục bộ, nhưng dataset thi nặng, tải riêng, môi trường máy dev không có GPU/cài
  đủ dependency nặng này). Logic 2 script này gần như PORT NGUYÊN VẸN (không đổi công
  thức toán/luồng xử lý) từ bản đã chạy thật 7/7 scene ở repo tiền nhiệm — rủi ro thấp
  nhưng CHƯA đạt mức "verify thật" tuyệt đối, cần chạy thật lần đầu trên Kaggle rồi
  đối chiếu số liệu (n_train/n_holdout, num_reg_images...) có hợp lý không.
- Notebook `kaggle_round1_baseline.ipynb` chưa chạy thật trên Kaggle (chỉ
  `nbformat.validate()` — đảm bảo JSON hợp lệ, KHÔNG đảm bảo mọi cell chạy đúng khi
  Run All thật). `GDRIVE_URL` copy nguyên từ `kaggle_private.ipynb` của repo tiền
  nhiệm (dataset thật, đã xác nhận đúng ở đó) — chưa tự tải lại để verify link còn
  sống ở thời điểm viết milestone này.

## Bước tiếp theo (cho người/agent chạy Kaggle thật)
1. Mở `pipeline/kaggle_round1_baseline.ipynb` trên Kaggle, Settings → GPU T4 x2/P100 +
   Internet On.
2. Điền `GITHUB_TOKEN` vào Kaggle Secrets (nếu repo giữ Private) — xem hướng dẫn ở
   Bước 3 trong notebook.
3. Chạy `MODE="holdout"` cho từng scene trong 7 scene trước, ghi lại Score từng scene
   (đối chiếu khoảng Score đã từng đo ở repo tiền nhiệm trong
   `docs/PORTED_KNOWLEDGE.md` để phát hiện sớm nếu có gì bất thường — vd Score thấp
   bất thường có thể do bug cấu hình/dữ liệu chưa lộ ra khi chỉ test cục bộ).
4. Sau khi xác nhận baseline hợp lý, chạy `MODE="final"` cho từng scene, tải NGUYÊN
   thư mục `gs_model/` lên Google Drive theo đúng hướng dẫn ở Bước 6 của notebook
   (không chỉ `.ply` — xem `docs/PORTED_KNOWLEDGE.md` mục 4 lý do bắt buộc).
5. Bàn giao link Drive từng scene cho Vòng 2+ (`kaggle_round2_refine.ipynb`, do agent
   khác phụ trách) hoặc dùng trực tiếp cho `kaggle_submission.ipynb` nếu không chạy
   thêm vòng nào.
6. Nếu train thật trên Kaggle phát hiện sai lệch API so với commit pin (vd tên cờ
   `train.py` đổi khác), cập nhật `02_train_baseline.sh` + `03_render_test_poses.py`
   + ghi lại phát hiện mới vào `docs/PORTED_KNOWLEDGE.md` (không chỉ sửa âm thầm).

## Lịch sử

### 2026-07-18 — Port + xây dựng + verify cục bộ Vòng 1 (baseline)
- Đọc `docs/00_MASTER_PLAN.md`, `docs/PORTED_KNOWLEDGE.md` (đầy đủ, không suy đoán từ
  trí nhớ) và skim `pipeline/common/scenes.py`/`poses.py` trước khi viết code, đúng
  theo yêu cầu STEP 0.
- Port + refactor 5 script từ repo tiền nhiệm
  (`00_make_holdout_split.py`, `01_run_colmap.py`, `03_train_3dgs.sh` ->
  `02_train_baseline.sh`, `04_render_test_poses.py` -> `03_render_test_poses.py`,
  `05_eval_metrics.py` -> `04_eval_metrics.py`) sang đúng tên file theo số thứ tự mới
  đã thống nhất với các agent song song khác (không đụng file `05_*`/`06_*`/`07_*`
  của agent khác — đã kiểm tra `git status` trước và sau khi xong, xác nhận không
  chồng lấn).
- `02_train_baseline.sh`: bỏ hẳn nhánh antenna-focus/depth-prior/exposure-comp (đã đo
  thật ở repo tiền nhiệm là KHÔNG cải thiện Score, xem `docs/PORTED_KNOWLEDGE.md` mục
  2) — giữ nguyên antialiasing (mặc định BẬT), OOM-safety env var
  (`PYTORCH_CUDA_ALLOC_CONF`), progress-polling (`|| true` sau `grep`, tránh bug
  `set -e`/`pipefail` giết script giữa chừng), multi-checkpoint (7000/15000/final),
  và ghi `pipeline_train_flags.json` đúng schema cũ (`depth_prior`/`exposure_comp`/
  `antenna_focus` luôn `false`) để code downstream (render/refine của agent khác) đọc
  key không bị lỗi thiếu key.
- Viết `pipeline/kaggle_round1_baseline.ipynb` bằng `nbformat` (22 cell), mirror cấu
  trúc/tông giọng `kaggle_private.ipynb` của repo tiền nhiệm — dùng đúng bản
  flexible-folder-detection đã fix (không bắt buộc tên thư mục dataset), hỗ trợ
  `MODE="holdout"`/`"final"`, `GITHUB_TOKEN`/Kaggle Secrets, Bước cuối hướng dẫn tải
  NGUYÊN thư mục `gs_model/` lên Drive.
- Test cục bộ toàn diện (xem mục "Đã verify" ở trên) — `py_compile` cho 4 file `.py`,
  `nbformat.validate()` cho notebook, và quan trọng nhất: viết `train.py` giả (mock,
  hỗ trợ `CRASH_AT` để mô phỏng OOM/crash) chạy `02_train_baseline.sh` thật trong thư
  mục scratch (`/tmp`, dọn sạch sau khi xong) qua 10 kịch bản (thành công, thất bại
  giữa chừng có/không checkpoint để cứu, thiếu/sai `GS_REPO`, scene thiếu sparse,
  guard antialiasing với repo cũ) — cả nhánh thành công lẫn nhánh lỗi đều test, đúng
  triết lý bắt buộc ở `docs/PORTED_KNOWLEDGE.md` mục 6.
- Không sửa `WORKLOG.md` (file đó thuộc repo tiền nhiệm `BTS-Digital-Twin`, không
  phải repo này) — milestone log này (`docs/MILESTONE_01_round1_baseline.md`) đóng
  vai trò tương đương cho repo `BTS-Digital-Twin-MultiRound`, theo đúng quy ước ghi ở
  `docs/00_MASTER_PLAN.md` mục 4.
