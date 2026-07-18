# Milestone 13 — Kiểm tra độc lập lần 10 (verification pass #10 / N)

## Trạng thái hiện tại

**HOÀN TẤT pass #10 — SẠCH, KHÔNG tìm thấy/sửa bug hay vấn đề nào.** Bộ đếm
3-lần-liên-tiếp-không-lỗi hiện tại: **1/3** (pass #9 có phát hiện thật nên reset về
0/3; pass này là lần sạch đầu tiên của chuỗi mới). Cần pass #11 và #12 đều sạch mới
coi là xong hẳn.

## Phạm vi đã làm

Đọc đầy đủ STEP 0: `docs/VERIFICATION_LOOP_PROTOCOL.md`, `docs/00_MASTER_PLAN.md`,
`docs/PORTED_KNOWLEDGE.md` toàn bộ (kể cả mục 6b–6i), cả 13 milestone log trước
(`MILESTONE_00` → `MILESTONE_12`), `git log --oneline`, `git show 90c9f50` (đúng
diff pass #9, đọc từng dòng).

### Phần A — Baseline: test suite

- `python -m py_compile` toàn bộ 15 file `.py` (bao gồm `tests/`): sạch.
- `bash -n` cho `02_train_baseline.sh` + `06_train_refine.sh`: sạch.
- `nbformat.validate()` cả 4 notebook: hợp lệ.
- `tests/test_syntax_all.py`: 15/15 `.py` + 4/4 `.ipynb` PASS.
- `tests/test_07_package_submission.py`: 21/21 PASS.
- Chạy lại lần cuối sau toàn bộ thao tác thực thi thật (mục B) — vẫn PASS 100%,
  `git status --short` sạch trong suốt quá trình (không rác nào lọt vào repo thật).

### Phần B — Mở rộng real-data coverage của pass #9 sang 2 script chưa ai chạy thật

Pass #9 lần đầu chạy thật `00_make_holdout_split.py`/`01_run_colmap.py` bằng
`pycolmap` cài thật + `pycolmap.synthesize_dataset()`. Pass này dùng LẠI đúng kỹ
thuật đó nhưng đẩy tiếp sang 2 script chưa ai chạy thật: `05_generate_error_mask.py`
(phần không cần CUDA: load pose, công thức percentile/weight, I/O PNG 16-bit) và
`03_render_test_poses.py` (phần dựng pose/camera) → `07_package_submission.py`
(kiểm tra kích thước).

**Hạ tầng dùng lại kỹ thuật pass #1/#4 (fake `GS_REPO`)**: venv riêng (`torch` CPU,
`pycolmap` 4.1.1, `opencv-python-headless`), tải THẬT `scene/cameras.py`,
`scene/colmap_loader.py`, `utils/graphics_utils.py`, `utils/general_utils.py`,
`utils/sh_utils.py`, `utils/camera_utils.py`, `arguments/__init__.py` từ chính commit
pin `54c035f7834b564019656c3e3fcc3646292f727d` (qua `git clone` + `git checkout`,
không suy đoán nội dung), CHỈ stub `scene/gaussian_model.py` (load_ply tối thiểu) và
`gaussian_renderer/__init__.py` (thay CUDA rasterizer — phần DUY NHẤT không thể chạy
thật ở máy dev không có toolchain CUDA build extension). Vì venv này có 1
`sitecustomize.py` hệ thống che mất site-packages riêng, dùng 1 wrapper
`run_with_cuda_shim.py` (monkeypatch `torch.Tensor.cuda()`/`torch.tensor(device=...)`
thành no-op rồi `runpy.run_path()` script mục tiêu) thay cho kỹ thuật
`sitecustomize.py` gốc của pass #1/#4 — tương đương về hiệu quả.

**1. `05_generate_error_mask.py` — chạy THẬT (không mock) với dữ liệu COLMAP thật:**

- Dựng scene tổng hợp `chair` bằng `pycolmap.synthesize_dataset()` (1 camera,
  24 ảnh, `camera_params` chỉnh tay khớp đúng `camera_width/height` tự chọn — LƯU Ý
  đã tự vấp 1 lỗi THIẾT LẬP TEST (không phải bug repo): quên set `camera_params`
  khớp `camera_width/height` tuỳ chỉnh khiến `cx,cy` lệch tâm ảnh, bị chính
  `assert_centered_principal_point()` bắt đúng — xác nhận cơ chế "fail loudly" hoạt
  động đúng ngay cả với lỗi ở fixture test, không phải ở code repo).
- Ghi ảnh PNG THẬT (gradient + 1 ô vuông vàng cố định ở giữa, không phải ảnh trắng
  trơn) cho tất cả 24 ảnh.
- Chạy THẬT `00_make_holdout_split.py --scene chair` → 18 train / 6 holdout.
- Chạy THẬT `01_run_colmap.py --scene chair --holdout` (dùng
  `pycolmap.undistort_images()` thật) → `colmap/dense/{images,sparse/0}` thật.
- Dựng `gs_model/` giả tối thiểu (`cfg_args`, `pipeline_train_flags.json`,
  `point_cloud/iteration_20/point_cloud.ply` placeholder — chỉ CUDA rasterizer bị
  stub, `GaussianModel.load_ply()` không cần đọc nội dung thật).
- Chạy THẬT `05_generate_error_mask.py --scene chair` qua `run_with_cuda_shim.py` —
  **`load_train_poses()` (đọc `read_extrinsics_binary`/`read_intrinsics_binary` THẬT
  từ `scene/colmap_loader.py` gốc) đọc đúng 18 pose từ sparse đã undistort thật**,
  `assert_centered_principal_point()` qua đúng, công thức percentile/weight chạy
  thật trên ảnh thật.
- **Verify công thức mask với dữ liệu có cấu trúc không gian THẬT** (không phải
  random noise): stub `render()` đọc lại 1 ảnh GT thật rồi xoá trắng đúng 1 vùng cố
  định (mô phỏng "model tái tạo kém đúng ở đó") — kết quả: `weight` tại vùng bị xoá
  = **6.0 đúng bằng `max_weight`**, vùng còn lại = **1.0 đúng** (không đổi), khớp
  hoàn toàn công thức percentile p50/p95 mô tả trong docstring. Đọc lại mask 16-bit
  PNG bằng `PIL.Image.open()` (đúng cách `apply_error_refine_patch.py::_error_mask()`
  đọc) — giá trị round-trip khớp chính xác `round(weight*1000)`.
  - Cơ chế dọn `error_masks/` cũ (fix pass #5) cũng tự kích hoạt đúng khi chạy lại
    lần 2 (in `[dọn dẹp] Đã xoá 18 mask .png cũ + manifest.json cũ`).
- **Kết quả: KHÔNG tìm thấy bug** — lần đầu tiên phần "pose loading + percentile/
  weight math + I/O 16-bit PNG" của `05_generate_error_mask.py` được xác nhận đúng
  bằng dữ liệu COLMAP thật (không phải chỉ đọc code hay mock hoàn toàn).

**2. `03_render_test_poses.py` (dựng pose/camera) → `07_package_submission.py`
(kiểm tra kích thước) — chạy THẬT với dữ liệu pose thật từ `pycolmap.Reconstruction`:**

- Dựng `test_poses.csv` cho scene `chair` trực tiếp từ đối tượng
  `pycolmap.Reconstruction` THẬT (24 pose, đảo quaternion `[x,y,z,w]->qw,qx,qy,qz`
  đúng cách `00_make_holdout_split.py` làm, `cx=width/2, cy=height/2` chính xác) —
  KHÔNG hand-craft số liệu.
- Chạy THẬT `03_render_test_poses.py --scene chair --poses_csv ...` qua
  `run_with_cuda_shim.py` — đọc `cfg_args`/`pipeline_train_flags.json` thật, dựng
  `MiniCam` bằng `getWorld2View2`/`getProjectionMatrix` THẬT (code gốc từ commit pin,
  không sửa), stub `render()` trả ảnh đúng kích thước `cam.image_width/height` —
  24 ảnh PNG được tạo, script tự kiểm tra `pil_img.size == (pose.width, pose.height)`
  cho từng ảnh (không có `RuntimeError` nào — kích thước luôn khớp).
- Import THẬT `07_package_submission.py` (qua `importlib`, không subprocess) và gọi
  `check_scene()` trên `renders/` thật vừa tạo — **0 lỗi** (khớp hoàn toàn
  `test_poses.csv` thật).
- **Làm hỏng CÓ CHỦ ĐÍCH 1 ảnh — resize lệch ĐÚNG 1 PIXEL** (256×192 → 255×192)
  bằng `PIL.Image.resize()` thật (không mock `Image` object) — gọi lại
  `check_scene()`: **bắt đúng lỗi**
  `"chair/camera000001_frame000005.png: kích thước (255, 192) != yêu cầu (256,192)"`.
  Xác nhận trực tiếp yêu cầu của nhiệm vụ: `check_scene()` THẬT SỰ phát hiện lệch
  1 pixel với đối tượng `PIL.Image` thật (không phải chỉ test mock trước đây trong
  `tests/test_07_package_submission.py`).
- **Kết quả: KHÔNG tìm thấy bug** — xác nhận chuỗi "pose thật từ pycolmap → dựng
  camera thật (`03_render_test_poses.py`) → kiểm tra kích thước thật
  (`07_package_submission.py`)" hoạt động đúng đầu-cuối, kể cả nhánh lỗi.

**Phát hiện phụ (không phải bug, ghi nhận kiến thức mới về `pycolmap`):** trong lúc
dựng fixture, phát hiện `pycolmap.undistort_images()` (bản 4.1.1) có 1 "fast path" —
nếu tham số méo ống kính `k` của camera `SIMPLE_RADIAL` đúng **bằng 0.0 tuyệt đối**
(bit-exact), nó chỉ COPY ảnh nguyên vẹn và **GIỮ NGUYÊN model `SIMPLE_RADIAL`**
(log: `"Copying already undistorted image..."`), KHÔNG chuyển sang `PINHOLE` như tài
liệu `colmap_runner.py` mô tả ("undistort ảnh + camera model -> PINHOLE sạch"). Verify
bằng thực thi thật quét `k ∈ {0.0, 0.00001, 0.0001, 0.001, 0.01}`: CHỈ đúng
`k=0.0` tuyệt đối mới giữ nguyên `SIMPLE_RADIAL`, mọi giá trị khác (kể cả `0.00001`)
đều convert đúng sang `PINHOLE`. Hậu quả nếu xảy ra: `05_generate_error_mask.py::
load_train_poses()` (chỉ chấp nhận `SIMPLE_PINHOLE`/`PINHOLE`) sẽ `raise ValueError`
rõ ràng — **và đây KHÔNG phải rủi ro riêng của repo này**: đối chiếu trực tiếp
`scene/dataset_readers.py` (mã gốc `graphdeco-inria/gaussian-splatting`, dòng ~88-98)
xác nhận chính `train.py` cũng CHỈ chấp nhận `SIMPLE_PINHOLE`/`PINHOLE`
(`assert False, "Colmap camera model not handled..."` với model khác) — nên nếu
tình huống này thật sự xảy ra, TRAIN CHÍNH THỐNG cũng sẽ crash tương tự, không phải
lỗ hổng riêng của script tự viết. Về khả năng xảy ra THẬT: `k` là kết quả bundle
adjustment (tối ưu số thực) từ ảnh chụp thật — xác suất hội tụ về đúng bit `0.0`
tuyệt đối với dữ liệu ảnh thật gần như bằng 0 (khác hẳn dữ liệu tổng hợp nơi có thể
cố ý đặt `k=0.0`). Kết luận: **kiến thức mới xác nhận thật, không phải bug, không
cần sửa code** — ghi vào `docs/PORTED_KNOWLEDGE.md` mục 6j để pass sau không tưởng
nhầm đây là rủi ro cần vá.

### Phần C — `pipeline/common/alignment.py`: audit sâu + chạy thật lần đầu

Đọc toàn bộ file (49 dòng, thuật toán Umeyama similarity transform). Grep TOÀN repo
`from common.alignment` / `from common import alignment` / `alignment\.` — **0 kết
quả ngoài chính file `alignment.py`** (đã xác nhận bởi pass #7, tái xác nhận ở đây):
không có script nào trong repo Round 2 hiện tại import/gọi module này — đây là code
port nguyên vẹn từ repo tiền nhiệm (dùng cho "Phase 0 — kiểm định hệ toạ độ", một
bước không còn trong kiến trúc multi-round hiện tại), **dead code hợp lệ, không phải
lỗi** (đã đọc `docs/00_MASTER_PLAN.md`/`PORTED_KNOWLEDGE.md` — không có tài liệu nào
tuyên bố module này đang được dùng).

Khác các pass trước (chỉ đọc bằng mắt), pass này **chạy thật** `umeyama_alignment()`
với dữ liệu tổng hợp có đáp án biết trước: 20 điểm 3D ngẫu nhiên, áp phép biến đổi
similarity biết trước (`scale=2.5`, xoay 37° quanh trục Z, `translation=(1,2,3)`) +
nhiễu Gauss nhỏ (`σ=0.001`), rồi gọi `umeyama_alignment(src, dst)`:

- Sai số ước lượng `scale`: `8×10⁻⁶` (so với `2.5` thật).
- Sai số ước lượng `rotation` (Frobenius norm): `4×10⁻⁵`.
- Sai số ước lượng `translation`: `4×10⁻⁴`.
- `residuals` sau align: max `0.0028`, mean `0.0016` — đúng bậc độ lớn của nhiễu đã
  thêm vào (`σ=0.001`), xác nhận thuật toán khôi phục ĐÚNG phép biến đổi thật.
- `raw_residuals()` (không align) cho mean `15.1` — LỚN HƠN nhiều residuals sau
  align, đúng như docstring mô tả ("dùng để kiểm tra xem 2 hệ toạ độ có tự nhiên
  trùng khớp hay không").
- `ValueError` đúng khi gọi với `< 3` điểm tương ứng.

**Kết luận: `umeyama_alignment()` ĐÚNG về mặt số học** (không chỉ "đọc thấy hợp lý"
như pass #3 đã ghi) — xác nhận bằng thực thi thật lần đầu tiên qua 10 pass. Vẫn là
dead code (không ảnh hưởng gì tới pipeline hiện tại) — không cần xoá/sửa gì (quyết
định giữ hay xoá code không dùng thuộc phạm vi người dùng, không tự ý xoá).

### Phần D — "Bước N" cross-reference audit (góc CHƯA pass nào làm ở mức này)

Các pass trước đã audit "narrative" từng notebook (pass #8/#9 đọc toàn bộ cell như
người dùng thật) nhưng chưa ai kiểm tra RIÊNG tính đúng đắn của SỐ trong "Bước N"
sau 9 vòng chỉnh sửa cell của nhiều pass khác nhau — khả năng 1 cell bị chèn/xoá
làm lệch số heading, hoặc 1 cross-reference nội bộ ("xem Bước 5") trỏ sai sau khi
số heading đổi.

Viết script quét toàn bộ markdown cell của cả 4 notebook bằng regex, tách riêng
HEADING (`## Bước N — ...`) khỏi CROSS-REFERENCE (nhắc "Bước N" trong văn xuôi):

- **Heading**: cả 4 notebook có heading liên tục đúng 1,2,3,...,N (N=6 cho round1,
  N=12 cho round2/round3, N=8 cho submission) — không thiếu/lặp/nhảy số nào.
- **Cross-reference**: mọi chỗ văn bản nhắc "Bước N" (vd "đổi `SCENE` ở Bước 5",
  "giống Bước 6 của kaggle_round1_baseline.ipynb", "so với Bước 8") đều trỏ ĐÚNG
  heading tồn tại và ĐÚNG nội dung ngữ nghĩa (đã đọc lại thủ công từng chỗ đối chiếu
  heading tương ứng, không chỉ kiểm tra số tồn tại) — **0 cross-reference sai/trỏ vào
  heading không tồn tại** ở cả 4 notebook.
- (Ghi chú kỹ thuật: script tự động ban đầu báo nhầm 2 "false positive" ở
  `kaggle_round2/3_refine.ipynb` do heading "## Bước 11 — ... so với Bước 8" chứa
  CẢ 2 số trên CÙNG 1 dòng `#` — đã tự đọc lại nguyên văn cell xác nhận đây KHÔNG
  phải lỗi, chỉ là hạn chế của regex phân loại "heading vs cross-reference" theo
  dòng, không phải vấn đề thật trong nội dung.)

**Kết luận: không có "Bước N" nào lệch số sau 9 vòng chỉnh sửa** — kỷ luật đánh số
tuần tự đã được giữ vững qua toàn bộ lịch sử sửa đổi.

### Phần E — Grep cross-check commit pin (độc lập lại, sau pass #6)

`grep -rn "54c035f7834b564019656c3e3fcc3646292f727d"` (case-sensitive) trên toàn bộ
`.py`/`.sh`/`.ipynb`/`.md`: **16 kết quả**. `grep -rni` (case-insensitive) trên cùng
tập file: **cũng đúng 16 kết quả** — tập hợp giống hệt nhau, xác nhận **KHÔNG có
biến thể casing nào khác** ở bất kỳ đâu (nếu có, case-insensitive sẽ tìm được NHIỀU
hơn case-sensitive). Grep thêm `54c035f` (7 ký tự đầu, không kèm phần còn lại) để dò
biến thể rút gọn: chỉ có đúng 1 kết quả phụ (`docs/MILESTONE_04...md`, dạng
`54c035f...` có dấu `...` rõ ràng là rút gọn có chủ đích trong văn xuôi, không phải
lỗi gõ thiếu). **Không tìm thấy sai lệch nào — khớp lại kết luận của pass #6, xác
nhận không có edit nào của pass #7/#8/#9 vô tình chèn thêm bản sao lệch.**

### Phần F — Sanity check hạ tầng test (theo đúng yêu cầu nhiệm vụ)

Chưa ai từng chủ động "phá" file nguồn để xác nhận `tests/test_syntax_all.py`/
`tests/test_07_package_submission.py` THẬT SỰ đang kiểm tra đúng file (không phải
silently no-op do lỗi import path). Verify bằng thực thi thật:

1. **Sabotage `07_package_submission.py`** (đổi `target_filename()` thành luôn trả
   `"SABOTAGED.png"`) → chạy `tests/test_07_package_submission.py`: **7/21 FAIL**
   đúng như kỳ vọng (unit test + end-to-end đều phát hiện). Khôi phục file gốc →
   chạy lại: **21/21 PASS**. Xác nhận test THẬT SỰ đọc + thực thi đúng file nguồn,
   không phải import nhầm bản cache/no-op.
2. **Sabotage `04_eval_metrics.py`** (chèn 1 dòng cú pháp sai) → chạy
   `tests/test_syntax_all.py`: báo đúng **1 FILE LỖI** (`04_eval_metrics.py`), có
   traceback `SyntaxError` cụ thể. Khôi phục → chạy lại: **15/15 + 4/4 PASS**.

**Kết luận: cả 2 file test đều đang exercise ĐÚNG file thật trong repo, không phải
hộp đen no-op.** Không tìm thấy vấn đề gì về hạ tầng test.

## Kết luận

**Không tìm thấy/sửa/flag bất kỳ vấn đề chức năng nào ở pass này.** Toàn bộ phát
hiện đều thuộc loại "xác nhận đúng bằng thực thi thật" (khác đọc code/mock trước
đây) hoặc "kiến thức mới về hành vi `pycolmap`, không phải bug, không cần sửa" — đã
cân nhắc kỹ, không quy các mục này vào "phát hiện" theo đúng tinh thần pass #4 đã
thiết lập (không tạo phát hiện cosmetic/thông tin giả để có gì báo cáo).

## Giới hạn của pass này (ghi rõ, không giấu)

- Vẫn CHƯA chạy được train/render 3DGS thật trên GPU Kaggle — giới hạn không đổi
  qua mọi pass, chỉ giải quyết được khi có 1 lần chạy Kaggle thật.
- `apply_error_refine_patch.py` được re-verify áp patch sạch lên clone thật tại
  đúng commit pin (không regression), nhưng vẫn CHƯA từng chạy `training()` đã vá
  tới mức thật sự train (cần rasterizer CUDA thật) — vẫn chỉ verify được luồng
  patch + cú pháp, như mọi pass trước.
- Dữ liệu tổng hợp qua `pycolmap.synthesize_dataset()` vẫn là scene ĐƠN GIẢN (1
  camera, không đại diện đầy đủ độ phức tạp 7 scene thật, drone/nhiều camera) —
  chưa thay thế hoàn toàn 1 lần chạy Kaggle thật với dataset thi thật.
- Phát hiện "pycolmap k=0.0 giữ nguyên SIMPLE_RADIAL" là kiến thức MỚI xác nhận
  thật nhưng xác suất xảy ra với dữ liệu ảnh thật gần như 0 — không theo dõi thêm
  trừ khi có bằng chứng ngược lại từ dataset thi thật.

## Bước tiếp theo

1. Chạy **pass #11** (agent verify độc lập khác) — bộ đếm sạch liên tiếp: 1/3. Cần
   pass #11 VÀ #12 đều sạch mới coi là xong (đúng 3 lần liên tiếp tính từ pass #10).
2. Gợi ý cho pass #11: hạng mục "chưa test được cục bộ" lớn nhất còn lại tiếp tục
   thu hẹp — chỉ còn phần THẬT SỰ cần CUDA (rasterizer thật, `training()` đã vá chạy
   thật, render/Score chất lượng thật trên dữ liệu 7 scene thật). Các khu vực đã
   soi rất kỹ nhiều lần (liệt kê đầy đủ ở các milestone trước, cộng thêm ở pass này:
   `05_generate_error_mask.py`/`03_render_test_poses.py`→`07_package_submission.py`
   real-data coverage, `alignment.py` chạy thật, "Bước N" cross-reference, commit pin
   grep) nên tiếp tục giảm ưu tiên trừ khi có thay đổi code liên quan. Có thể thử:
   mở rộng scene tổng hợp phức tạp hơn (nhiều camera, quỹ đạo lớn hơn — đúng gợi ý
   còn tồn đọng từ pass #9), hoặc thử lần đầu chạy `apply_error_refine_patch.py`'s
   patched `training()` xa hơn patch-verify (vẫn cần stub CUDA nhưng có thể mô phỏng
   sâu hơn vòng lặp loss/optimizer nếu khả thi không cần rasterizer thật).

## Lịch sử

### 2026-07-19 — Verification pass #10 (agent kiểm tra độc lập, khác pass #1-9)
- Đọc đầy đủ STEP 0 (`VERIFICATION_LOOP_PROTOCOL.md`, `00_MASTER_PLAN.md`,
  `PORTED_KNOWLEDGE.md` toàn bộ kể cả mục 6b-6i, 13 milestone log 00-12,
  `git log --oneline`, `git show 90c9f50`).
- Phần A: test suite baseline PASS 100%.
- Phần B: dựng fake `GS_REPO` (real non-CUDA source từ commit pin qua `git clone` +
  `git checkout`, chỉ stub `gaussian_model.py`/`gaussian_renderer/__init__.py`),
  venv riêng + wrapper `run_with_cuda_shim.py` (thay `sitecustomize.py` bị hệ thống
  che mất) — chạy THẬT `00_make_holdout_split.py`→`01_run_colmap.py`→
  `05_generate_error_mask.py` (verify percentile/weight math + 16-bit PNG roundtrip
  với ảnh thật có cấu trúc không gian biết trước) và
  `03_render_test_poses.py`→`07_package_submission.py::check_scene()` (verify bắt
  đúng lệch 1 pixel với `PIL.Image` thật) — cả 2 luồng: KHÔNG tìm thấy bug. Phát hiện
  phụ (không phải bug): `pycolmap.undistort_images()` giữ nguyên `SIMPLE_RADIAL`
  khi `k` đúng bằng `0.0` tuyệt đối (đối chiếu upstream `dataset_readers.py` xác
  nhận không phải rủi ro riêng của repo này).
- Phần C: audit + chạy thật lần đầu `alignment.py::umeyama_alignment()` với dữ liệu
  tổng hợp có đáp án biết trước — xác nhận đúng số học (sai số ~1e-5), tái xác nhận
  vẫn là dead code (0 import trong repo).
- Phần D: audit "Bước N" cross-reference toàn bộ 4 notebook — không có số lệch/trỏ
  sai sau 9 vòng sửa cell.
- Phần E: grep cross-check commit pin case-sensitive vs case-insensitive (16=16) —
  không có biến thể casing/copy-paste lệch nào.
- Phần F: sabotage test thật (`07_package_submission.py`, `04_eval_metrics.py`) xác
  nhận `tests/test_syntax_all.py`/`tests/test_07_package_submission.py` thật sự
  exercise đúng file nguồn, không phải no-op.
- Không sửa file nào trong repo (không có bug thật để sửa). `git status --short`
  sạch trong suốt quá trình, kể cả sau khi chạy script thật ghi ra
  `pipeline/work/chair/` (đã xoá lại ngay sau khi verify xong). Dọn sạch toàn bộ
  scratch (`/tmp/.../scratchpad/pass10`).
- Ghi milestone log này. Cập nhật `docs/PORTED_KNOWLEDGE.md` mục 6j (kiến thức mới
  về `pycolmap.undistort_images()` — không phải bug).
