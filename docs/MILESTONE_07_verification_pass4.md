# Milestone 07 — Kiểm tra độc lập lần 4 (verification pass #4 / N)

## Trạng thái hiện tại

**HOÀN TẤT pass #4 — SẠCH, KHÔNG tìm thêm bug/vấn đề nào.** Đây là lần XÁC NHẬN
KHÔNG LỖI ĐẦU TIÊN sau 3 pass liên tiếp đều tìm ra vấn đề thật (pass #1: 4, pass #2: 1,
pass #3: 2). Theo yêu cầu 3-lần-liên-tiếp-không-lỗi của user, bộ đếm hiện tại: **1/3**.
Cần tiếp tục pass #5 và #6 (đều phải sạch) mới coi là xong hẳn.

## Phạm vi đã làm

Đọc đầy đủ STEP 0: `docs/00_MASTER_PLAN.md`, `docs/PORTED_KNOWLEDGE.md` toàn bộ (kể cả
mục 6b/6c/6d), cả 6 milestone log trước (`MILESTONE_00` → `MILESTONE_06`), `git log
--oneline`, `git show` đầy đủ diff của cả 3 commit verify trước (`92b6623`, `0571dbc`,
`59a0f7d`) — không chỉ đọc commit message mà xem trực tiếp từng dòng đã đổi.

### Phần A — Baseline: chạy lại toàn bộ test suite

- `tests/test_syntax_all.py`: 15/15 `.py` PASS + 4/4 `.ipynb` PASS.
- `tests/test_07_package_submission.py`: 21/21 PASS.
- `bash -n` cho `02_train_baseline.sh` + `06_train_refine.sh`: sạch.
- Chạy lại lần cuối ở cuối pass (không có gì để re-verify vì không sửa gì) — vẫn PASS
  100%.

### Phần B — Re-audit trực tiếp các fix của pass #1–#3 (đọc code thật, không chỉ tin
milestone log)

1. **Symlink dataset (fix pass #3)** — đọc trực tiếp cell "Bước 4" của cả 4 notebook
   bằng script (không đọc mắt): xác nhận cả 4 file dùng ĐÚNG 1 pattern
   `if target.is_symlink(): unlink() elif target.exists(): rmtree()` rồi LUÔN gọi
   `os.symlink(...)` sau đó — an toàn giống hệt nhau cả 4 file, không còn bản
   `target.unlink() if target.is_symlink() else None` cũ. **Xác nhận đúng.**
   - Phát hiện phụ (KHÔNG phải bug, chỉ ghi nhận): dòng NGAY SAU đó —
     `os.environ["BTS_DATASET_ROOT"] = str(target)` ở 3 notebook train, nhưng
     `= str(target.resolve())` ở `kaggle_submission.ipynb` — khác nhau về mặt text,
     nhưng **không khác nhau về hành vi**: `target` là 1 symlink hợp lệ ngay tại thời
     điểm gán (vừa `os.symlink()` xong ở dòng trên), và mọi thao tác `Path`/mở file
     sau đó (`pipeline/common/scenes.py::DATASET_ROOT`, chỉ dùng đúng 1 chỗ để nối
     đường dẫn con, không so sánh `realpath`/identity ở đâu khác — đã grep toàn repo
     xác nhận `BTS_DATASET_ROOT` chỉ xuất hiện 2 chỗ: docstring + dòng gán trong
     `scenes.py`) đều tự động theo symlink dưới POSIX, dù dùng đường dẫn symlink hay
     đường dẫn đã resolve. Không có so sánh path bằng chuỗi/identity nào trong repo phụ
     thuộc vào việc path có resolve hay không. Kết luận: đây là 1 sai khác nhỏ về
     STYLE (thừa `.resolve()` ở 1 trong 4 file) do 2 agent khác nhau viết độc lập,
     nhưng KHÔNG gây sai lệch hành vi thật — không đủ điều kiện tính là bug (đã cân
     nhắc kỹ theo đúng yêu cầu "không tạo ra phát hiện cosmetic giả").
2. **GPU-check `raise SystemExit` (fix pass #3)** — dump cell "Bước 1" cả 4 notebook
   bằng script, so sánh chuỗi: **giống hệt nhau 100% byte-for-byte** (đã in độ dài
   chuỗi = 1233 ký tự cả 4 file). Không có f-string/nội suy nào trong khối message
   (toàn bộ là chuỗi tĩnh nối bằng dấu `+`/liền kề, không có `{...}`) — không có nguy
   cơ lỗi interpolation. `nbformat.validate()` vẫn PASS sau khi thêm — notebook hợp lệ.
   **Vị trí cell**: xác nhận lại bằng cách đọc thứ tự cell thật (không chỉ tin theo
   heading) — cell GPU-check nằm ở "Bước 1 — Cài đặt", TRƯỚC "Bước 2 — Clone + build
   3D Gaussian Splatting" (tốn 2-5 phút build CUDA extension) và TRƯỚC "Bước 4 — Tải
   dataset từ Google Drive" (vài trăm MB–vài GB) ở cả 4 notebook — đúng yêu cầu chặn
   SỚM trước khi tốn thời gian/quota, không phải chỉ tồn tại đâu đó muộn hơn.
3. **`_latest_iteration_dir()` backport (fix pass #2)** — grep TOÀN REPO (không chỉ 2
   file `.sh` đã biết) cho pattern `sort` bất kỳ áp lên đường dẫn: chỉ còn đúng 2 định
   nghĩa hàm này (`02_train_baseline.sh`, `06_train_refine.sh`), cả 2 dùng chung 1
   thuật toán (vòng lặp bash thuần, so sánh số nguyên `10#$n`, không tách trường theo
   `_`/khoảng trắng). Toàn bộ chỗ khác dùng `sort`/`sorted()` trong repo (grep 15 kết
   quả) đều là sort TÊN FILE/chuỗi thông thường (`sorted(gt_dir.glob(...))`,
   `sorted(poses.keys())`, `sorted(sparse_root.iterdir())`...), KHÔNG áp lên đường dẫn
   `iteration_<N>` — không có bug class tương tự sót lại ở vị trí thứ 3. `03_render_
   test_poses.py`/`05_generate_error_mask.py` dùng
   `int(p.name.split("_")[-1])` (Python, không phải `sort -t_ -k2`) — an toàn theo
   thiết kế khác hẳn (`split("_")[-1]` lấy đúng field cuối cùng, không lệ thuộc số
   dấu `_` phía trước trong toàn đường dẫn). **Xác nhận: không còn nơi thứ 3 dính bug
   class này.**

### Phần C — Góc mới (theo đúng gợi ý cuối `MILESTONE_06`, cộng thêm các mục bài giao)

- **`apply_error_refine_patch.py`** — đọc TOÀN BỘ file (201 dòng, không skim): 6 patch
  (import numpy/PIL, thêm tham số `training()`, call site, argparse, `Scene(
  load_iteration=...)` + fix `spatial_lr_scale` + hàm `_error_mask()`, loss weighted
  L1) đều khớp đúng mô tả trong `docs/PORTED_KNOWLEDGE.md` mục 2/3 — công thức
  `Ll1 = (weight * |render-gt|).sum() / weight.expand_as(render).sum()` đúng, hằng số
  `_ERROR_MASK_SCALE = 1000.0` khớp `05_generate_error_mask.py`, marker
  `"error-refine"` (dùng để tự nhận diện đã vá) xuất hiện đúng trong đoạn code được
  chèn (dòng print), guard chặn vá đè 2 lần / chặn xung đột với antenna-focus đều hoạt
  động đúng logic (kiểm tra bằng đọc code, không suy đoán). Không tìm thấy vấn đề.
- **`07_package_submission.py`** — đọc TOÀN BỘ file (196 dòng): `target_filename()`,
  `check_scene()` (đối chiếu kích thước + đếm số file dư/thiếu), `encode_for_arcname()`
  (mã hoá lại đúng định dạng theo đuôi, không đổi tên suông), `build_zip()`/
  `verify_zip()` đều đúng logic đã mô tả. Đối chiếu `--jpeg_quality` mặc định của
  script (95) với giá trị THẬT truyền vào từ `kaggle_submission.ipynb` (`--jpeg_quality
  98`, khác mặc định của script nhưng là do notebook chủ động truyền tường minh, không
  phải bug — default 95 chỉ dùng khi gọi trực tiếp không truyền cờ). Không tìm thấy
  vấn đề.
- **Numeric formatting / truncation** — grep toàn bộ `int(`/`round(`/`float(` áp dụng
  cho số iteration, weight, percentile, kích thước ảnh trên khắp `pipeline/scripts/` +
  `pipeline/common/`: mọi chỗ `int()` áp lên số iteration đều đi sau bước validate
  regex (`^[0-9]+$`) hoặc trên chuỗi tách từ tên thư mục `iteration_<N>` (luôn là số
  nguyên do chính pipeline tạo ra, không phải input người dùng tự do) — không có
  `int()` áp lên 1 giá trị float có thể có phần thập phân ý nghĩa bị mất. Điểm đáng
  chú ý duy nhất: `pipeline/common/poses.py::read_test_poses()` dùng
  `int(float(r["width"]))` — cố ý chấp nhận cả `"1920"` lẫn `"1920.0"` trong CSV rồi ép
  về int, đúng ý đồ (width/height luôn là số nguyên theo ngữ nghĩa, không mất thông tin
  thật). Không tìm thấy bug loại "âm thầm cắt cụt số có nghĩa".
- **Default config Vòng 1** — đọc trực tiếp `pipeline/scripts/02_train_baseline.sh`
  dòng 79-84: `ANTIALIASING="${ANTIALIASING:-1}"` (mặc định BẬT), không có biến môi
  trường nào cho depth-prior/exposure-comp/antenna-focus (3 kỹ thuật này không tồn tại
  trong script — hardcode `false` khi ghi `pipeline_train_flags.json`) — khớp CHÍNH XÁC
  tuyên bố của `docs/00_MASTER_PLAN.md` mục 3.1 ("antialiasing BẬT, không depth-prior/
  antenna-focus/exposure-comp"). **Khớp, không lệch.**
- **`GDRIVE_URL`** — so sánh bằng script (không đọc mắt) cả 4 notebook: cùng 1 chuỗi
  `https://drive.google.com/file/d/178EL7jCSVD59q19SMpeOgnOfOIC66I_t/view?usp=drive_link`
  byte-for-byte ở cả 4 file. **Khớp.**
- **`REPO_URL`/`GIT_BRANCH`** — so sánh bằng script cả 4 notebook: cùng
  `https://github.com/ThongLuc2k3/BTS-Digital-Twin-MultiRound.git` / `"main"` byte-
  for-byte. Không còn tham chiếu `coordination/round1-status` hay tên repo tiền nhiệm
  nào — dòng comment trong `kaggle_submission.ipynb` nhắc "vd coordination/..." chỉ là
  VÍ DỤ MINH HOẠ chung chung trong văn bản giải thích (không phải giá trị biến thật),
  giá trị biến `GIT_BRANCH` vẫn là `"main"`. **Khớp, không có leftover.**
- **TODO/FIXME/XXX/"chưa test"/"giả định"** — grep toàn bộ `pipeline/` (`.py`, `.sh`,
  cả nội dung cell `.ipynb`): không có `TODO`/`FIXME`/`XXX` nào. Các chỗ có từ "giả
  định" (`05_generate_error_mask.py` — tương quan lỗi train/test chưa có bằng chứng
  thực nghiệm; `poses.py` — principal point ở giữa ảnh, `assert_centered_principal_
  point()` tự raise nếu sai; `poses.py::representative_intrinsics` — train/test dùng
  chung 1 camera vật lý) đều là giả định ĐÃ ĐƯỢC GHI RÕ + có cơ chế tự kiểm tra/tự báo
  lỗi đi kèm (không phải giả định âm thầm) — đã được `docs/PORTED_KNOWLEDGE.md` mục 3
  và các pass trước xác nhận là rủi ro đã biết, có kế hoạch giảm thiểu (đo Score
  holdout trước/sau mỗi vòng), không phải lỗ hổng mới. Không có gì cần flag thêm.
- **`README.md`** — đọc lại: vẫn ngắn gọn, chính xác (không liệt kê số lượng file
  milestone cụ thể nên không có nguy cơ lạc hậu theo số đếm). Không cần sửa.
- **`tests/README.md`** — đọc lại: nội dung pass #3 cập nhật vẫn khớp hiện trạng (không
  có test mới nào được thêm ở pass #3/#4 để cần cập nhật thêm).

### Phần D — Mô phỏng end-to-end MỚI, tập trung kịch bản "Score Vòng refine TỆ HƠN —
rollback/dừng ở vòng trước" (góc chưa pass nào làm — 3 pass trước đều mô phỏng kịch
bản "refine cải thiện/scene bị bỏ qua vì thiếu mask", chưa pass nào cố tình test
đường "đo được Score XẤU ĐI thì sao")

Dựng fixture MỚI hoàn toàn trong scratchpad riêng (không tái dùng scratch cũ):
`train.py` mock ghi `cfg_args`/`point_cloud.ply` giả + marker `[error-refine]`, fixture
`colmap/dense/{sparse/0,images}` cho scene `chair`.

1. `02_train_baseline.sh chair` (ITERATIONS=20, `python` alias sang `python3` vì máy
   dev chỉ có `python3`) — checkpoint Vòng 1 THẬT được tạo tại
   `gs_model/point_cloud/iteration_20/point_cloud.ply`, `pipeline_train_flags.json`
   đúng schema (`antialiasing: true`). Lưu `md5sum` checkpoint này TRƯỚC khi refine.
2. Tái tạo `dense/images/`, hand-craft `error_masks/manifest.json` (`iteration: 20`).
3. `06_train_refine.sh chair` (REFINE_ITERATIONS=5) — checkpoint Vòng 2 được tạo tại
   `iteration_25` (=20+5, đúng luỹ kế), `refine_history` được ghi thêm vào
   `pipeline_train_flags.json`.
4. **Xác nhận checkpoint Vòng 1 (`iteration_20/point_cloud.ply`) còn NGUYÊN BYTE**
   (so `md5sum` trước/sau refine — khớp 100%) — refine KHÔNG hề đụng/ghi đè checkpoint
   nguồn, chỉ tạo thư mục MỚI cạnh nó.
5. **Mô phỏng đúng quyết định "Score Vòng 2 TỆ HƠN Vòng 1, user chọn KHÔNG dùng"**
   (đúng cell "Bước 11" của `kaggle_round2_refine.ipynb`/`kaggle_round3_refine.ipynb` —
   in ra khuyến nghị "GIỮ checkpoint Vòng 1, DỪNG LẠI ở scene này" khi
   `after <= before`): dựng lại 1 "phiên submission" HOÀN TOÀN MỚI, giả lập việc tải
   LẠI checkpoint từ đúng link Drive Vòng 1 gốc (chưa từng bị vòng 2 đụng tới — vì
   theo thiết kế, việc "tải lên Drive" ở Bước 12 của notebook refine là 1 bước THỦ CÔNG
   tách biệt, chỉ làm nếu Score cải thiện; nếu Score tệ hơn thì user đơn giản là không
   làm bước đó, link Drive Vòng 1 cũ vẫn y nguyên) — dựng thư mục `gs_model` CHỈ chứa
   `iteration_20` (đúng như những gì thực sự nằm trên Drive nếu user không upload lại).
6. Chạy logic `find_latest_iteration()` (trích y hệt từ `03_render_test_poses.py`) trên
   thư mục "chỉ-có-Vòng-1" này: chọn ĐÚNG `iteration_20` — xác nhận `kaggle_submission.
   ipynb` (khi trỏ `CHECKPOINT_LINKS["chair"]["link"]` về đúng link Drive Vòng 1 cũ) sẽ
   render bằng ĐÚNG checkpoint Vòng 1, không có cơ chế nào trong code buộc phải dùng
   checkpoint mới nhất/vòng cao nhất đã từng chạy.

**Kết quả Phần D**: xác nhận bằng THỰC THI THẬT (không chỉ đọc code) rằng "dừng lại ở
Vòng N" là 1 kết quả hạng nhất (first-class), không có đoạn code nào trong
`06_train_refine.sh`/`03_render_test_poses.py`/`kaggle_submission.ipynb` tự động buộc
tiến lên vòng tiếp theo hoặc tự động dùng checkpoint "mới nhất từng train ra" thay vì
checkpoint người dùng THỰC SỰ chọn tải lên Drive. Quyết định "dùng vòng nào" hoàn toàn
nằm ở việc user chọn LINK DRIVE nào dán vào `CHECKPOINT_LINKS` — đúng thiết kế đã nêu ở
`docs/00_MASTER_PLAN.md` mục 3.3 và `PORTED_KNOWLEDGE.md`. Không phát hiện bug.

Dọn sạch toàn bộ thư mục scratch sau khi xong (`git status` xác nhận sạch trong suốt
quá trình — không có gì được tạo trong cây thư mục repo).

## Kết luận

**Không tìm thấy/sửa/flag bất kỳ vấn đề genuine nào ở pass này.** Đã cân nhắc kỹ 1 khác
biệt nhỏ (dòng `BTS_DATASET_ROOT = str(target)` vs `str(target.resolve())` giữa
`kaggle_submission.ipynb` và 3 notebook kia) và kết luận đây KHÔNG phải bug (không gây
sai lệch hành vi, chỉ khác text) — không đủ điều kiện để tính là "phát hiện" theo đúng
yêu cầu "không tạo phát hiện cosmetic giả" của nhiệm vụ.

## Giới hạn của pass này (ghi rõ, không giấu)

- Vẫn CHƯA chạy được train/render 3DGS thật trên GPU Kaggle — giới hạn không đổi qua
  mọi pass, chỉ giải quyết được khi có 1 lần chạy Kaggle thật.
- `00_make_holdout_split.py`/`01_run_colmap.py` vẫn chỉ được audit bằng đọc code (đọc
  toàn bộ 2 file lần này, không tìm thấy vấn đề) — chưa chạy thật với `pycolmap` +
  dữ liệu COLMAP thật cục bộ (gợi ý này đã có từ `MILESTONE_06`, vẫn chưa ai làm được
  vì tốn thời gian dựng fixture COLMAP thật; rủi ro thấp vì logic port gần như nguyên
  vẹn từ repo tiền nhiệm đã chạy thật 7/7 scene).

## Bước tiếp theo

1. Chạy **pass #5** (agent verify độc lập khác) — bộ đếm sạch liên tiếp hiện tại: 1/3.
   Cần pass #5 VÀ pass #6 đều sạch mới coi là xong (không phải chỉ cần 2 pass nữa —
   đúng 3 lần liên tiếp KHÔNG lỗi tính từ đầu, pass #4 này là lần đầu tiên trong chuỗi
   3 lần đó).
2. Gợi ý cho pass #5 (giảm trùng lặp phạm vi): các khu vực đã bị soi rất kỹ nhiều lần
   (schema `pipeline_train_flags.json`, va chạm thư mục iteration, đường dẫn dấu cách,
   `spatial_lr_scale`, tên file `eval_metrics`, `git submodule update`, checkpoint sort
   bug, symlink dataset, GPU-check, rollback/stop-at-round-N) nên giảm ưu tiên. Còn lại
   thực sự chưa test bằng dữ liệu thật cục bộ: `00_make_holdout_split.py`/
   `01_run_colmap.py` với `pycolmap` + fixture COLMAP thật (không phải chỉ đọc code).
   Cũng có thể thử build 1 bản `gaussian-splatting` KHÔNG-CUDA thật (như pass #1 đã làm
   1 lần) nhưng lần này thử áp CẢ patch `apply_error_refine_patch.py` một cách chạy
   thật hàm `training()` tới mức gọi `_error_mask()`/loss weighted (không chỉ verify
   patch áp được + `py_compile`) — đây là phần sâu nhất của `train.py` đã vá mà chưa
   pass nào thực thi qua, dù đã đọc code kỹ.

## Lịch sử

### 2026-07-18 — Verification pass #4 (agent kiểm tra độc lập, khác pass #1/#2/#3)
- Đọc đầy đủ STEP 0 (`00_MASTER_PLAN.md`, `PORTED_KNOWLEDGE.md` toàn bộ kể cả mục
  6b/6c/6d, 7 milestone log 00-06, `git log --oneline`, `git show` đầy đủ diff của cả 3
  commit verify trước).
- Phần A: chạy lại `tests/test_syntax_all.py` (15/15 + 4/4 PASS),
  `tests/test_07_package_submission.py` (21/21 PASS), `bash -n` 2 file `.sh` — sạch.
- Phần B: re-audit trực tiếp 3 fix của 3 pass trước bằng cách đọc code/diff thật (không
  chỉ tin milestone log) — symlink dataset (khớp, xác nhận đúng pattern cả 4 file, ghi
  nhận 1 khác biệt cosmetic không phải bug ở dòng `BTS_DATASET_ROOT`), GPU-check
  (byte-for-byte giống hệt cả 4 file, đúng vị trí trước Bước 2/4 tốn thời gian),
  `_latest_iteration_dir()` (grep toàn repo xác nhận không còn nơi thứ 3 dính bug class
  `sort -t_ -k2 -n`).
- Phần C: đọc toàn bộ `apply_error_refine_patch.py` + `07_package_submission.py` (lần
  đầu 1 pass đọc trọn vẹn, không skim) — không tìm thấy vấn đề; audit numeric
  formatting/truncation toàn `pipeline/scripts/`+`pipeline/common/` — không tìm thấy
  bug; xác nhận default config Vòng 1 khớp tuyên bố `00_MASTER_PLAN.md`; xác nhận
  `GDRIVE_URL`/`REPO_URL`/`GIT_BRANCH` byte-for-byte giống nhau cả 4 notebook, không có
  leftover tên nhánh/repo tiền nhiệm; grep TODO/FIXME/XXX/giả định — không có gì mới
  cần flag; `README.md`/`tests/README.md` vẫn chính xác.
- Phần D: dựng fake `GS_REPO` MỚI trong scratch riêng, chạy THẬT `02_train_baseline.sh`
  (Vòng 1) → `06_train_refine.sh` (Vòng 2) → xác nhận checkpoint Vòng 1 còn nguyên byte
  (md5sum khớp) sau refine → mô phỏng kịch bản "Score Vòng 2 TỆ HƠN, user chọn KHÔNG
  dùng" bằng cách dựng lại 1 phiên submission chỉ có checkpoint Vòng 1 (đúng những gì
  thực sự tồn tại trên Drive nếu user không upload Vòng 2) → xác nhận
  `find_latest_iteration()` chọn đúng Vòng 1, không có cơ chế nào ép buộc dùng vòng cao
  hơn. Không phát hiện bug — xác nhận "dừng ở Vòng N" là kết quả hạng nhất được hỗ trợ
  đầy đủ bởi tooling thật (không chỉ theo tài liệu).
- Không sửa bất kỳ file nào trong repo (không có bug thật để sửa). Dọn sạch toàn bộ
  scratch (`git status` sạch trong suốt quá trình).
- Ghi milestone log này. KHÔNG thêm mục mới vào `docs/PORTED_KNOWLEDGE.md` (không có
  bug/bài học mới để ghi — mục 6e không được tạo vì không cần thiết).
