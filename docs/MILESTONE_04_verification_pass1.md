# Milestone 04 — Kiểm tra độc lập lần 1 (verification pass #1 / 3)

## Trạng thái hiện tại

**HOÀN TẤT pass #1.** Theo yêu cầu của user: cần **3 lần XÁC NHẬN LIÊN TIẾP không lỗi**
mới coi repo là xong (`docs/MILESTONE_00_setup.md`). Pass này tìm ra **4 bug thật** (đã
sửa cả 4) + 2 vấn đề tài liệu (đã sửa). **Vì có bug thật, pass này KHÔNG tính là "sạch"
— bộ đếm 3-lần-liên-tiếp reset về 0.** Cần tiếp tục chạy pass #2, #3, #4 (liên tiếp,
không lỗi) mới coi là xong.

## Phạm vi đã làm

Agent verify độc lập, không thuộc 4 agent xây dựng ban đầu (điều phối + 3 sub-agent
Round-1/refine/submission-testing). Đọc đầy đủ `docs/00_MASTER_PLAN.md`,
`docs/PORTED_KNOWLEDGE.md`, cả 4 milestone log (00-03), `git log`, rồi làm 4 phần theo
đúng yêu cầu:

### Phần A — Chạy lại toàn bộ test suite có sẵn
- `tests/test_syntax_all.py`: 15/15 `.py` PASS + 4/4 `.ipynb` PASS.
- `tests/test_07_package_submission.py`: 21/21 PASS.
- `bash -n` cho cả 2 file `.sh` (`02_train_baseline.sh`, `06_train_refine.sh`): sạch.
- Không có `python -m py_compile`/`nbformat.validate()` nào thiếu trong bộ test có sẵn
  so với yêu cầu — không cần viết thêm gì cho Phần A.

### Phần B — Audit nhất quán chéo file (kết quả: tìm ra 2/4 bug thật)

Đối chiếu tường minh từng cặp file ghi/đọc, từng notebook, theo đúng checklist yêu cầu:

- **Script filename mà 4 notebook gọi (`!python .../scripts/...`, `!bash .../scripts/...`)
  đều tồn tại đúng đường dẫn** — grep hết, đối chiếu `ls pipeline/scripts/`: KHỚP 100%,
  không có tham chiếu file không tồn tại.
- **Schema `pipeline_train_flags.json`** — `02_train_baseline.sh` ghi 4 key
  (`antialiasing`/`depth_prior`/`exposure_comp`/`antenna_focus`), `06_train_refine.sh`
  đọc lại đúng 4 key đó + tự thêm `refine_history` (không xoá key cũ, dùng
  `flags.setdefault(...)`) — **KHỚP**, không có mismatch (đây là bug nguy hiểm nhất theo
  `PORTED_KNOWLEDGE.md`, đã kiểm tra kỹ, không có).
- **`Scene` API (`pipeline/common/scenes.py`)** — `get_scene()`/`all_scenes()`/
  `.test_poses_csv`/`.train_images_dir`/`.provided_sparse_dir`/
  `.has_valid_provided_sparse()` — grep toàn bộ script dùng đến, tất cả dùng đúng tên
  thuộc tính, không có lỗi gõ nhầm/API cũ. **KHỚP**.
- **`REPO_URL`/`GIT_BRANCH`** cả 4 notebook — cùng
  `https://github.com/ThongLuc2k3/BTS-Digital-Twin-MultiRound.git`, `GIT_BRANCH="main"`.
  **KHỚP**.
- **`error_masks/manifest.json`** — `05_generate_error_mask.py` ghi `iteration`/
  `max_weight` (+ các field khác), `06_train_refine.sh` đọc đúng đúng 2 key này để đối
  chiếu. **KHỚP**.
- **`!lệnh {biểu_thức}`** — quét toàn bộ 4 notebook, mọi chỗ nội suy chỉ dùng tên biến
  đơn giản (`{SCENE}`, `{scene}`) hoặc subscript đơn giản (`{os.environ['GS_REPO']}`) —
  đúng khuyến cáo `PORTED_KNOWLEDGE.md` mục 4, không có biểu thức phức tạp.
- **f-string thiếu tiền tố `f`** — quét bằng regex + đọc tay toàn bộ cell code của cả 4
  notebook: KHÔNG tìm thấy lỗi loại này (milestone 02 đã tự sửa 1 lỗi loại này trước
  khi agent verify này chạy).

**Bug thật #1 (MỚI, tìm ở pass này) — sai tên file `eval_metrics.txt` vs
`eval_metrics.csv`:** `pipeline/scripts/04_eval_metrics.py` ghi kết quả ra
`renders_root/<scene>/eval_metrics.csv` (dòng `write_csv(...)`), nhưng
`kaggle_round2_refine.ipynb` VÀ `kaggle_round3_refine.ipynb` (cả 2 file, ở Bước 8 và
Bước 11) lại đọc/copy từ `eval_metrics.txt` — file này KHÔNG BAO GIỜ tồn tại. Nếu chạy
thật trên Kaggle, `shutil.copy(...)` ở Bước 8 sẽ crash `FileNotFoundError` ngay sau khi
train baseline xong, chặn đứng toàn bộ luồng đo Score trước/sau của Vòng 2/3 (đúng phần
CỐT LÕI của cơ chế multi-round). Đây là bug cross-file kinh điển mà không sub-agent riêng
lẻ nào tự bắt được (agent viết `04_eval_metrics.py` không đọc lại code notebook của agent
refine, và ngược lại). **Đã sửa**: đổi cả 4 chỗ tham chiếu (`eval_metrics.txt` ->
`.csv`, `eval_metrics_BEFORE_round{2,3}.txt` -> `.csv`) ở cả 2 notebook.

**Bug thật #2 (MỚI, tìm ở pass này) — thiếu `git submodule update --init --recursive`
sau khi pin commit trong `kaggle_submission.ipynb`:** cả 3 notebook train
(`kaggle_round1_baseline.ipynb`, `kaggle_round2_refine.ipynb`,
`kaggle_round3_refine.ipynb`) đều có dòng `!git submodule update --init --recursive`
NGAY SAU `!git checkout 54c035f...` (đúng comment trong `02_train_baseline.sh`: "re-sync
submodule đúng theo commit vừa checkout") — nhưng `kaggle_submission.ipynb` THIẾU dòng
này. `git clone --recursive` chỉ đồng bộ submodule theo commit HEAD lúc clone (nhánh
mặc định), sau đó `git checkout <commit pin>` đổi con trỏ submodule ghi trong cây
nhưng KHÔNG tự cập nhật working tree của submodule — nếu commit pin tham chiếu version
submodule khác với HEAD mặc định, `diff-gaussian-rasterization`/`simple-knn` bị build
sai version, có thể lệch API với `train.py`/`gaussian_renderer` ở đúng chỗ quan trọng
nhất (bước cuối cùng trước khi nộp bài). **Đã sửa**: thêm dòng này vào Bước 2 của
`kaggle_submission.ipynb`.

**Vấn đề tài liệu (không phải bug chức năng, đã sửa luôn cho chính xác):**
- `pipeline/common/colmap_runner.py`: docstring ghi "sparse/0/ hợp lệ ở cả 13/13 scene"
  và "scene HCM0249" — cả 2 đều là tàn dư copy từ repo tiền nhiệm (round trước có 13
  scene, có scene tên `HCM0249`; round này (`docs/00_MASTER_PLAN.md`) chỉ có **7 scene**,
  KHÔNG có `HCM0249`). Không ảnh hưởng logic (default `camera_model="SIMPLE_RADIAL"` vẫn
  đúng, chỉ là câu giải thích lý do sai). Đã sửa thành "7/7 scene" và bỏ tên scene sai.
- `pipeline/common/logging_utils.py`: docstring "chạy hàng loạt 13 scene" -> sửa "7
  scene" (cùng lý do).

## Phần C — Mô phỏng end-to-end thống nhất (chưa ai làm, giá trị cao nhất)

Dựng **1 fake `GS_REPO` thống nhất** dùng chung cho toàn bộ chuỗi (không phải mock rời
rạc từng file như 3 agent ban đầu đã làm riêng lẻ):
- `train.py` mock (baseline + refine trong 1 file, có marker `error-refine` mô phỏng
  trạng thái đã vá).
- `scene/cameras.py`, `scene/colmap_loader.py`, `utils/graphics_utils.py`,
  `utils/general_utils.py`: **lấy nguyên văn từ `graphdeco-inria/gaussian-splatting` tại
  đúng commit pin `54c035f7834b564019656c3e3fcc3646292f727d`** (tải qua raw.githubusercontent.com
  — các file này thuần Python/numpy/cv2, KHÔNG phụ thuộc CUDA) — để test bằng đúng code
  thật, không phải hàng tự chế.
- `scene/gaussian_model.py`, `gaussian_renderer/__init__.py`: stub tối thiểu (thay cho
  phần PHỤ THUỘC CUDA thật — `simple_knn._C`/`diff_gaussian_rasterization` — không build
  được ở máy dev vì thiếu `nvcc`, đã xác nhận `nvcc: command not found`).
- Cài `torch` bản CPU-only (nhẹ, không cần build) trong venv riêng + 1
  `sitecustomize.py` monkeypatch `torch.Tensor.cuda()`/`torch.tensor(device="cuda")`
  thành no-op CHỈ trong harness test này (không đụng code thật trong repo) — máy dev có
  GPU thật (GTX 1650 4GB, xác nhận qua `nvidia-smi`) nhưng thiếu toolchain CUDA để build
  extension, nên vẫn phải mock ở mức rasterizer, đúng như `docs/PORTED_KNOWLEDGE.md`/
  hướng dẫn task đã lường trước.

**Đã re-verify độc lập (không chỉ tin lại milestone 02):** áp `apply_error_refine_patch.py`
thật lên **`train.py` GỐC THẬT** tải trực tiếp từ commit pin (không phải mock) — 6/6 patch
khớp, `py_compile` sạch sau vá, cờ chống vá 2 lần hoạt động đúng (chạy lại lần 2 tự nhận
diện "đã vá", không vá đè).

**Chuỗi mô phỏng đã chạy THẬT (không phải hand-craft file trung gian) bằng chính script
thật trong repo, qua `subprocess`/`bash` thật:**
1. Dựng `colmap/dense/sparse/0/{cameras.txt,images.txt}` (format COLMAP text chuẩn, đọc
   bằng `scene/colmap_loader.py` THẬT) + `colmap/dense/images/*.png` giả cho scene
   `chair` (3 ảnh train).
2. `02_train_baseline.sh chair` (ITERATIONS=20) -> `gs_model/cfg_args` +
   `pipeline_train_flags.json` (`antialiasing:true` đúng) + `point_cloud/iteration_20/`.
   `dense/images/` bị tự xoá đúng như thiết kế (dọn đĩa).
3. Tái tạo `dense/images/` (đúng quy trình thật: "chạy lại 01_run_colmap.py" — ở đây
   dựng lại fixture tương đương) -> `05_generate_error_mask.py --scene chair
   --iteration 20` (script THẬT, chỉ stub rasterizer) -> `error_masks/manifest.json`
   (`iteration:20, max_weight:6.0`) + 3 mask PNG.
4. `06_train_refine.sh chair` (REFINE_ITERATIONS=5, tự dò CKPT_ITERATION=20) -> checkpoint
   MỚI tại `iteration_25` (=20+5, đúng luỹ kế), `refine_history` có 1 phần tử,
   `point_cloud/iteration_20/point_cloud.ply` **còn nguyên byte-for-byte** (đối chiếu
   md5sum).
5. **Lặp lại vòng 3** (đúng yêu cầu trọng tâm — "nếu có bug chỉ lộ ở vòng 2" của task):
   tái tạo `dense/images/` -> `05_generate_error_mask.py --iteration 25` (manifest tự
   cập nhật `iteration:25`) -> `06_train_refine.sh` (REFINE_ITERATIONS=5 **CÙNG giá trị
   như vòng trước** — cố ý test đúng kịch bản bug đã tìm+sửa ở milestone 02) -> checkpoint
   MỚI tại `iteration_30` (=25+5), **KHÔNG va chạm tên thư mục** với `iteration_25` cũ,
   `refine_history` có 2 phần tử đúng thứ tự, `iteration_20`+`iteration_25` cả 2 vẫn còn
   nguyên (md5sum khớp). Xác nhận: fix bug "đường dẫn có dấu cách"/"va chạm thư mục" ở
   milestone 02 hoạt động đúng trong dây chuyền thật (không chỉ trong test mock độc lập
   của chính milestone đó), kể cả khi chạy đúng trên đường dẫn dự án có dấu cách
   ("Khóa Luận Tốt Nghiệp") như môi trường thật.
6. `03_render_test_poses.py --scene chair` (script THẬT) -> tự dò `iteration_30` (checkpoint
   cuối), đọc đúng `antialiasing=True` từ `pipeline_train_flags.json`, render 2 ảnh test
   đúng kích thước yêu cầu.
7. `07_package_submission.py` (script THẬT, cần đủ 7 scene vì không có cờ `--scene` —
   dựng thêm fixture renders nhanh cho 6 scene còn lại) -> `submission.zip` build +
   `verify_zip()` tự chấm OK, đúng cấu trúc `scene/tên_ảnh.JPG`, nội dung mã hoá lại
   đúng JPEG thật (không phải đổi tên suông).

**Kết quả Phần C: KHÔNG phát hiện bug mới nào trong luồng chạy chuỗi thật** (khác Phần
B, nơi tìm ra 2 bug). Điều này cho thấy các lớp bug nguy hiểm nhất mà `PORTED_KNOWLEDGE.md`
liệt kê (schema `pipeline_train_flags.json`, va chạm thư mục iteration, đường dẫn dấu
cách, `spatial_lr_scale`) đã được xử lý đúng trong code thật — nhưng **KHÔNG thay thế
được 1 lần chạy GPU Kaggle thật** (rasterizer/render chất lượng thật vẫn chưa verify
được, chỉ verify được luồng điều khiển/schema/tên file).

## Giới hạn của Phần C (ghi rõ, không giấu)
- KHÔNG verify được chất lượng render/Score thật (rasterizer bị stub hoàn toàn).
- KHÔNG chạy `00_make_holdout_split.py`/`01_run_colmap.py` thật (cần `pycolmap` +
  logic COLMAP thật) — chỉ dựng fixture thay thế đầu ra của chúng. Rủi ro còn lại nằm
  ở CHÍNH 2 script đó (không đổi gì ở pass này, milestone 01 đã ghi nhận đây là phần
  chưa test cục bộ được).
- `apply_error_refine_patch.py` chỉ verify PATCH ÁP ĐƯỢC + cú pháp sạch, KHÔNG chạy
  `train.py` đã vá thật với `--refine_from_iteration` (cần rasterizer thật).

## Bước tiếp theo
1. Chạy **pass #2** (agent verify độc lập khác, hoặc lặp lại quy trình này) — vì pass #1
   tìm ra bug thật, bộ đếm reset về 0, cần 3 lần liên tiếp KHÔNG lỗi mới coi xong.
2. Các hạng mục "CHƯA test được" liệt kê ở milestone 01/02/03 (train/render GPU thật
   trên Kaggle) vẫn còn nguyên, không nằm trong phạm vi verify cục bộ.

## Lịch sử

### 2026-07-18 — Verification pass #1 (agent kiểm tra độc lập)
- Đọc đầy đủ STEP 0 (`00_MASTER_PLAN.md`, `PORTED_KNOWLEDGE.md`, 4 milestone log,
  `git log`).
- Phần A: chạy lại `tests/test_syntax_all.py` (15/15 + 4/4 PASS), `tests/test_07_package_submission.py`
  (21/21 PASS), `bash -n` 2 file `.sh` — tất cả sạch, không cần thêm gì.
- Phần B: audit chéo toàn bộ điểm nối giữa các file theo checklist — tìm+sửa 2 bug thật
  (`eval_metrics.txt` sai tên file ở `kaggle_round2_refine.ipynb`/`kaggle_round3_refine.ipynb`;
  thiếu `git submodule update --init --recursive` ở `kaggle_submission.ipynb`) + 2 vấn đề
  tài liệu (số scene/tên scene tàn dư từ repo tiền nhiệm ở `colmap_runner.py`/
  `logging_utils.py`).
- Phần C: dựng 1 fake `GS_REPO` thống nhất (dùng code thật không-CUDA từ commit pin +
  stub tối thiểu phần CUDA), chạy chuỗi Vòng 1 -> Vòng 2 -> Vòng 3 -> render -> đóng gói
  bằng CHÍNH script thật trong repo (không hand-craft file trung gian) — xác nhận không
  có bug mới, đặc biệt xác nhận lại 2 bug đã fix ở milestone 02 (đường dẫn dấu cách, va
  chạm thư mục iteration) hoạt động đúng khi chạy 2 vòng refine liên tiếp thật.
- Dọn sạch toàn bộ thư mục scratch (`/tmp/.../scratchpad/partc`, `patch_verify`,
  `real_ref`, `venv`, `bin`) sau khi xong — không để lại rác trong repo (`git status`
  xác nhận sạch, chỉ còn đúng các file sửa).
- Chạy lại toàn bộ test suite lần cuối sau khi sửa — xác nhận vẫn PASS 100%.
