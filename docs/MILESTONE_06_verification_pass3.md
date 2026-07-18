# Milestone 06 — Kiểm tra độc lập lần 3 (verification pass #3 / 3)

## Trạng thái hiện tại

**HOÀN TẤT pass #3.** Tìm ra + sửa **2 vấn đề thật mới** (khác hoàn toàn phạm vi 2 pass
trước — cả 2 đều ở khu vực chưa từng bị soi kỹ: notebook `kaggle_submission.ipynb` cell
dò-dataset và cell kiểm tra GPU dùng chung ở cả 4 notebook). **Vì có phát hiện thật, pass
này KHÔNG tính là "sạch" — bộ đếm 3-lần-liên-tiếp reset về 0.** Cần chạy tiếp pass #4, #5,
#6 (liên tiếp, không lỗi) mới coi là xong.

## Phạm vi đã làm

Đọc đầy đủ STEP 0: `docs/00_MASTER_PLAN.md`, `docs/PORTED_KNOWLEDGE.md` toàn bộ (kể cả
mục 6b/6c), `docs/MILESTONE_00` → `docs/MILESTONE_05_verification_pass2.md`, `git log
--oneline`, `git show 92b6623` + `git show 0571dbc` (đúng diff 2 commit verify trước) để
biết chính xác đã đổi gì, tránh dẫm lại đúng phạm vi cũ. Chủ động tìm góc **CHƯA** bị 2
pass trước soi kỹ, theo đúng gợi ý của nhiệm vụ.

### Phần A — Baseline: chạy lại toàn bộ test suite có sẵn
- `tests/test_syntax_all.py`: 15/15 `.py` PASS + 4/4 `.ipynb` PASS.
- `tests/test_07_package_submission.py`: 21/21 PASS.
- `bash -n` cho cả 2 file `.sh`: sạch.
- Chạy lại lần cuối SAU khi sửa (xem Phần C/D) — vẫn PASS 100%, không regression.

### Phần B — Audit backport-style bug (đúng loại pass #2 đã tìm)

1. **So `02_train_baseline.sh` với `06_train_refine.sh` từng đoạn logic trùng lặp**
   (disk-space check, `PYTORCH_CUDA_ALLOC_CONF`, ghi `pipeline_train_flags.json`,
   progress-poll loop, `_latest_iteration_dir()`): đọc từng cặp đối chiếu dòng — TẤT CẢ
   đều nhất quán/đúng ở cả 2 file (bao gồm cả fix của pass #2 đã backport đúng
   `_latest_iteration_dir()` sang `02_train_baseline.sh`). Không tìm thêm sai lệch nào.
2. **So cell-by-cell 4 notebook** (`kaggle_round1_baseline.ipynb`,
   `kaggle_round2_refine.ipynb`, `kaggle_round3_refine.ipynb`,
   `kaggle_submission.ipynb`) bằng script (dump toàn bộ source từng cell ra file, so
   từng cell tương ứng) — phát hiện **`kaggle_submission.ipynb` có 1 bản logic symlink
   dataset CŨ HƠN/kém an toàn hơn 3 notebook kia** (xem Bug #1 dưới) — đúng loại "1 file
   được fix/viết tốt hơn, file anh em có cùng logic bị bỏ sót" mà pass #2 đã cảnh báo
   trong bài học của mình.
3. **`03_render_test_poses.py` CLI/output contract đối chiếu với `04_eval_metrics.py`,
   `05_generate_error_mask.py`, `07_package_submission.py`**: đọc trực tiếp argparse
   thật (`--scene/--model_dir/--iteration/--out_dir/--poses_csv/--sh_degree/
   --antialiasing`), output mặc định `work/<scene>/renders/<stem>.png`. Khớp đúng
   `07_package_submission.py::build_zip()` (đọc `renders_root/<scene>/renders/`) và
   `04_eval_metrics.py` (đọc `--renders_subdir` mặc định `holdout_renders`, khớp
   `--out_dir` mà `kaggle_round2/3_refine.ipynb` truyền tường minh). **KHÔNG có
   mismatch.**

### Phần C — Khu vực chưa bị soi kỹ ở pass #1/#2

- **`pipeline/common/colmap_runner.py`, `alignment.py`, `poses.py`** — đọc toàn bộ (lần
  đầu 1 pass verify đọc kỹ cả 3 file này, 2 pass trước chỉ grep tên scene/13-scene).
  Không tìm thấy tham chiếu round-1-repo-cụ-thể/scene-list cứng nào khác ngoài 2 chỗ đã
  sửa ở pass #1. Logic Umeyama (`alignment.py`) và quaternion/FOV (`poses.py`) đúng, có
  chú thích đối chiếu source thật.
- **Quaternion order fix (`00_make_holdout_split.py`)**: xác nhận dòng
  `qx, qy, qz, qw = cfw.rotation.quat` unpack đúng thứ tự `[x,y,z,w]` của pycolmap rồi
  ghi lại đúng `qw,qx,qy,qz` vào CSV — khớp đúng bài học mục 1 `PORTED_KNOWLEDGE.md`.
- **`05_generate_error_mask.py` có đọc `points2D` không?** Đọc toàn bộ file: KHÔNG — chỉ
  đọc `images.bin/.txt` + `cameras.bin/.txt` (extrinsics/intrinsics qua
  `read_extrinsics_*`/`read_intrinsics_*`), không hề chạm `points3D`/`points2D`. Xác
  nhận ĐÚNG như giả định trong nhiệm vụ — không cần bước tự đo scale COLMAP.
- **Checklist "còn thiếu" cuối `docs/MILESTONE_03_submission_and_testing.md`** — đối
  chiếu từng mục với hiện trạng:
  1. `06_train_refine.sh` có ghi đúng schema `pipeline_train_flags.json` (4 key gốc +
     `refine_history`, dùng `setdefault` không xoá key cũ)? **CÓ, đã đúng** (xác nhận
     lại bằng đọc source + test thật ở Phần D).
  2. `kaggle_round2/3_refine.ipynb` có nhắc tải nguyên `gs_model/` lên Drive? **CÓ**
     (Bước 12 của cả 2 notebook).
  3. Test tích hợp CHẠY THẬT trên Kaggle — **VẪN CHƯA LÀM** (không có GPU cục bộ, giới
     hạn không đổi qua mọi pass).
  4. `apply_error_refine_patch.py` áp thật lên `train.py` gốc — **ĐÃ LÀM** ở pass #1
     Phần C (áp lên bản train.py thật tải từ commit pin, không phải mock).
  5. Re-chạy `test_syntax_all.py` sau khi merge đủ nhánh — **ĐÃ LÀM**, PASS từ milestone
     03 trở đi. Tất cả 5 mục trong checklist cũ đã được các pass sau giải quyết hoặc ghi
     nhận rõ giới hạn — không có mục nào "rơi qua khe nứt" bị quên.
- **`get_scene()` với tên scene sai chính tả**: `get_scene()` raise `ValueError` rõ ràng
  nếu tên không khớp `BTS_SCENES`/`GENERIC_SCENES`. Grep toàn bộ 5 script gọi hàm này —
  không có script nào `try/except` nuốt lỗi này, exception luôn nổi lên tới traceback +
  exit code khác 0 (fail loudly, đúng triết lý dự án).
- **GPU-check cell ở cả 4 notebook** — **TÌM THẤY VẤN ĐỀ THẬT** (xem Bug #2 dưới).
- **`README.md`/`tests/README.md`** — `README.md` vẫn chính xác (không đổi). `tests/README.md`
  mục "Test còn THIẾU" liệt kê nhiều việc **ĐÃ ĐƯỢC LÀM** ở milestone 01/02 và pass #1
  (mock `02_train_baseline.sh`/`06_train_refine.sh`, áp patch thật lên `train.py`) —
  tài liệu bị lạc hậu so thực tế dù không sai về mặt chức năng. Đã cập nhật (xem "Đã sửa").

### Phần D — Mô phỏng end-to-end MULTI-SCENE thật (góc chưa pass nào làm)

Cả pass #1 và pass #2 đều chỉ chạy chuỗi qua **1 scene duy nhất** (`chair`) xuyên suốt
Vòng 1→2→3. Pass này dựng lại **fake `GS_REPO` MỚI** (không tái dùng scratch cũ, đúng gợi
ý cuối `docs/MILESTONE_05_verification_pass2.md`) và chạy **2 scene cùng lúc trong 1 lệnh
gọi script thật**, đúng trọng tâm thiết kế "mỗi scene có thể dừng ở vòng khác nhau":

1. `train.py` mock (ghi `cfg_args`/checkpoint giả, hỗ trợ `MOCK_CRASH`) + fixture
   `colmap/dense/sparse/0`, `colmap/dense/images/` cho **2 scene `chair` + `bonsai`**
   cùng lúc.
2. `02_train_baseline.sh chair bonsai` (1 lệnh, 2 scene) — **CẢ 2 scene** train xong
   đúng, `pipeline_train_flags.json` đúng schema, `dense/images/` bị dọn đúng cho cả 2.
3. Tái tạo `dense/images/`, hand-craft `error_masks/manifest.json` (iteration=20,
   max_weight=6.0) **CHỈ cho `chair`** (cố ý KHÔNG tạo cho `bonsai`, mô phỏng đúng kịch
   bản "user quyết định chỉ refine 1 trong 2 scene").
4. `06_train_refine.sh chair bonsai` (1 lệnh, 2 scene) — xác nhận:
   - `chair`: refine thành công, checkpoint mới tại `iteration_25` (=20+5, đúng luỹ kế),
     `iteration_20` gốc còn nguyên nội dung, `pipeline_train_flags.json` có
     `refine_history` 1 phần tử.
   - `bonsai`: in `[BỎ QUA]` đúng lý do (thiếu `error_masks/manifest.json`), **KHÔNG**
     làm dừng cả lệnh (exit 0), `bonsai` vẫn giữ nguyên `iteration_20` (không đụng tới
     `colmap/dense/images/` của bonsai vì không refine).
5. Đối chiếu logic `find_latest_iteration()` (trích y hệt từ `03_render_test_poses.py`)
   trên 2 thư mục `gs_model` thật vừa tạo: `chair` → 25 (đúng, tự nhận checkpoint Vòng 2
   mới nhất), `bonsai` → 20 (đúng, tự nhận checkpoint Vòng 1, KHÔNG cần biết/khai báo
   "vòng nào" ở bất kỳ đâu).
6. Dựng fixture dataset (`BTS_DATASET_ROOT` trỏ tới `test/test_poses.csv` giả cho đủ
   7 scene) + `renders/` giả cho 7 scene, chạy **`07_package_submission.py` THẬT** (qua
   subprocess, không mock hàm) — đóng gói + `verify_zip()` tự chấm **OK** cho 1
   `submission.zip` chứa hỗn hợp: `chair` (thực chất đại diện checkpoint Vòng 2) +
   `bonsai` (Vòng 1) + 5 scene BTS còn lại — đúng kịch bản thật sẽ xảy ra lúc nộp bài.

**Kết quả Phần D: KHÔNG phát hiện bug mới** trong luồng multi-scene/mixed-round — xác
nhận cơ chế "mỗi scene độc lập, tự dò checkpoint mới nhất của chính nó, không cần biết
khái niệm 'vòng'" hoạt động đúng khi 2 script `.sh` nhận NHIỀU scene ở các trạng thái
khác nhau trong CÙNG 1 lệnh gọi (trước đây chỉ được xác nhận bằng đọc code
`for SCENE in "$@"`, chưa từng chạy thật với 2 scene khác trạng thái trong 1 lệnh).

## Vấn đề thật tìm ra + đã sửa

### Bug #1 — `kaggle_submission.ipynb` có bản logic symlink dataset CŨ/kém an toàn hơn 3 notebook kia (âm thầm giữ dataset CŨ nếu target đã là thư mục thật)

So cell-by-cell (Bước 4, cell dò+symlink thư mục dataset) của cả 4 notebook, phát hiện
`kaggle_round1_baseline.ipynb`/`kaggle_round2_refine.ipynb`/`kaggle_round3_refine.ipynb`
dùng logic:
```python
if target.is_symlink() or target.exists():
    if target.is_symlink():
        target.unlink()
    else:
        import shutil
        shutil.rmtree(target)
os.symlink(found.resolve(), target)
```
(xử lý ĐÚNG cả 2 trường hợp: symlink cũ VÀ thư mục thật còn sót lại), nhưng
`kaggle_submission.ipynb` (viết bởi agent milestone 03, không đối chiếu byte-for-byte với
3 notebook kia lúc viết) lại có bản CŨ HƠN:
```python
if target.exists() or target.is_symlink():
    target.unlink() if target.is_symlink() else None
if not target.exists():
    os.symlink(found.resolve(), target)
```
**Bug thật**: nếu `target` (`/kaggle/working/Dataset/VAI_NVS_DATA_ROUND2`) đã tồn tại
dưới dạng **thư mục thật (không phải symlink)** — biểu thức
`target.unlink() if target.is_symlink() else None` khi `is_symlink()=False` chỉ trả về
`None`, KHÔNG làm gì cả (không xoá thư mục cũ). Dòng sau `if not target.exists():` cũng
sai luôn vì thư mục cũ vẫn còn tồn tại → `os.symlink(...)` **KHÔNG BAO GIỜ được gọi**.
Kết quả: dataset MỚI vừa tải/giải nén bị **âm thầm bỏ qua hoàn toàn**, notebook tiếp tục
chạy với dữ liệu CŨ trong `target` mà KHÔNG có bất kỳ lỗi/cảnh báo nào — vi phạm đúng
triết lý "fail loudly, not silently misbehave" mà toàn bộ dự án đã tuân thủ.

**Verify bằng test thật** (không suy đoán): dựng `target` là 1 thư mục thật chứa file
đánh dấu `STALE_MARKER.txt`, `found` (nguồn mới) chứa `FRESH_MARKER.txt` — chạy đúng
đoạn code cũ của `kaggle_submission.ipynb`: kết quả `target` vẫn là thư mục thật, vẫn
chứa `STALE_MARKER.txt` (dữ liệu CŨ), symlink KHÔNG được tạo. Sau khi sửa (dùng logic
`if target.is_symlink(): unlink() elif target.exists(): rmtree()` rồi LUÔN
`os.symlink(...)`, giống hệt 3 notebook kia) — chạy lại: `target` trở thành symlink trỏ
đúng `found`, nội dung chỉ còn `FRESH_MARKER.txt`.

**Mức độ nghiêm trọng thực tế**: trong 1 phiên Kaggle mới hoàn toàn (trường hợp phổ
biến nhất), `target` chưa từng tồn tại nên nhánh lỗi không kích hoạt — bug chỉ lộ ra khi
người dùng **chạy lại** Bước 4 trong CÙNG phiên (vd retry tải dataset sau lỗi mạng, đổi
`GDRIVE_URL` để thử dataset khác) mà LẦN CHẠY TRƯỚC đã từng để lại thư mục thật (không
phải symlink) ở đúng vị trí đó — kịch bản hẹp nhưng có thể xảy ra thật khi debug trên
Kaggle, và hậu quả (đóng gói `submission.zip` bằng dataset SAI mà không biết) rất nặng vì
đây là notebook nộp bài cuối cùng.

**Đã sửa**: `pipeline/kaggle_submission.ipynb` cell dò+symlink dataset (Bước 4) — đổi
sang đúng logic robust của 3 notebook kia (`if is_symlink(): unlink() elif exists():
rmtree()`, rồi LUÔN gọi `os.symlink()` sau đó, không đặt trong `if not exists()` nữa).

### Bug #2 (thiết kế an toàn, không phải bug logic sai) — Cell kiểm tra GPU ở cả 4 notebook chỉ `print()` cảnh báo, KHÔNG dừng "Run All" khi thiếu GPU

Cả 4 notebook có CÙNG 1 cell (Bước 1) kiểm tra GPU:
```python
print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "KHÔNG có GPU — ...")
```
— chỉ `print()`, không `raise`/`assert`. Nếu người dùng bấm "Run All" mà quên bật
Accelerator GPU (Settings → Accelerator), dòng cảnh báo này dễ bị chìm giữa hàng trăm
dòng log của `!pip install` ngay cell kế tiếp — notebook tiếp tục "chạy" (tải dataset
nhiều GB, clone + build CUDA extension — build KHÔNG cần GPU device, chỉ cần toolchain,
nên có thể build "thành công" dù không có GPU) rồi mới crash ở lần gọi `.cuda()` ĐẦU TIÊN
bên trong `train.py`/`03_render_test_poses.py` — sau khi đã tốn nhiều phút/hàng chục phút
quota Kaggle. Đây đúng loại rủi ro "lãng phí thời gian nghiêm trọng dưới áp lực deadline"
mà `docs/00_MASTER_PLAN.md` mục 1 cảnh báo (deadline Vòng 1: 30/07/2026, giới hạn
5 lần nộp/ngày + không rõ giới hạn thời lượng notebook train).

Đây không phải bug tính toán sai (thông tin `torch.cuda.is_available()` vẫn đúng), nhưng
vi phạm đúng triết lý "fail loudly, not silently misbehave" đã áp dụng nhất quán ở mọi
chỗ khác trong dự án (vd `!lệnh` không tự dừng notebook → luôn có bước tự kiểm tra tường
minh, xem `docs/PORTED_KNOWLEDGE.md` mục 4) — cell GPU-check là ngoại lệ duy nhất còn sót
lại chưa áp dụng nguyên tắc này.

**Đã sửa**: cả 4 notebook — thêm `raise SystemExit(...)` ngay nếu
`not torch.cuda.is_available()`, thông báo rõ cách bật GPU (Settings → Accelerator → GPU
T4 x2/P100) và lý do vì sao phải dừng ngay thay vì chạy tiếp.

## Đã sửa thêm (tài liệu, không phải bug chức năng)

- `tests/README.md`: cập nhật mục "Test còn THIẾU" — tách rõ phần ĐÃ làm (mock
  `02_train_baseline.sh`/`06_train_refine.sh`, áp patch thật lên `train.py`, mô phỏng
  chuỗi multi-round bằng fake `GS_REPO`) khỏi phần THẬT SỰ còn thiếu (chỉ chạy được trên
  GPU Kaggle thật) — tránh người đọc sau tưởng nhầm các việc đã xong là chưa làm.

## Giới hạn của pass này (ghi rõ, không giấu)

- Phần D dùng `train.py` MOCK (không phải code thật không-CUDA như pass #1 đã làm) —
  đổi lại lấy được tốc độ để tập trung vào đúng trọng tâm multi-scene/mixed-round (2
  script `.sh` xử lý ĐÚNG khi nhận nhiều scene ở nhiều trạng thái khác nhau trong 1 lệnh)
  thay vì re-verify lại rasterizer/công thức render (đã verify kỹ ở pass #1). Không chạy
  lại `03_render_test_poses.py`/`05_generate_error_mask.py` thật (cần torch+rasterizer
  stub) — chỉ trích xuất+chạy lại đúng hàm `find_latest_iteration()` (copy y hệt logic
  nguồn) để verify riêng phần "tự dò checkpoint mới nhất không phân biệt vòng".
- Vẫn CHƯA chạy được train/render 3DGS thật trên GPU Kaggle — giới hạn không đổi qua mọi
  pass, chỉ giải quyết được khi có 1 lần chạy Kaggle thật.
- Không phát hiện thêm vấn đề nào trong `00_make_holdout_split.py`/`01_run_colmap.py`
  ngoài đọc lại code (không chạy thật với `pycolmap`) — giới hạn đã ghi từ milestone 01,
  chưa đổi.

## Bước tiếp theo

1. Chạy **pass #4** (agent verify độc lập khác) — vì pass #3 tìm ra vấn đề thật, bộ đếm
   reset về 0, cần 3 lần liên tiếp KHÔNG lỗi mới coi xong. Gợi ý cho pass #4: các khu vực
   đã bị soi kỹ nhiều lần (schema `pipeline_train_flags.json`, va chạm thư mục iteration,
   đường dẫn dấu cách, `spatial_lr_scale`, tên file `eval_metrics`, `git submodule
   update`, checkpoint sort bug, symlink dataset, GPU-check) có thể giảm ưu tiên; thử
   soi kỹ hơn `01_run_colmap.py` (script duy nhất trong `pipeline/scripts/` CHƯA từng
   được chạy thật cục bộ dù chỉ mock — luôn chỉ `py_compile`), hoặc thử chạy thật
   `00_make_holdout_split.py`/`01_run_colmap.py` với `pycolmap` cài vào venv riêng +
   dữ liệu COLMAP giả dựng thủ công (format text COLMAP chuẩn) để lần đầu verify được
   luồng thật của 2 script này thay vì chỉ đọc code.
2. Các hạng mục "CHƯA test được" liệt kê ở mọi milestone trước (train/render GPU thật
   trên Kaggle) vẫn còn nguyên, không nằm trong phạm vi verify cục bộ.

## Lịch sử

### 2026-07-18 — Verification pass #3 (agent kiểm tra độc lập, khác pass #1/#2)
- Đọc đầy đủ STEP 0 (`00_MASTER_PLAN.md`, `PORTED_KNOWLEDGE.md` toàn bộ kể cả mục 6b/6c,
  6 milestone log 00-05, `git log --oneline`, `git show 92b6623` + `git show 0571dbc`).
- Phần A: chạy lại `tests/test_syntax_all.py` (15/15 + 4/4 PASS),
  `tests/test_07_package_submission.py` (21/21 PASS), `bash -n` 2 file `.sh` — sạch.
- Phần B: audit backport-style bug — so `02_train_baseline.sh`/`06_train_refine.sh` từng
  đoạn logic trùng lặp (không tìm thêm sai lệch); so cell-by-cell cả 4 notebook bằng
  script (tìm ra Bug #1); đối chiếu CLI/output contract `03_render_test_poses.py` với 3
  script còn lại (khớp).
- Phần C: đọc toàn bộ `colmap_runner.py`/`alignment.py`/`poses.py` (lần đầu 1 pass đọc
  kỹ, không chỉ grep tên scene); xác nhận quaternion-order fix đúng; xác nhận
  `05_generate_error_mask.py` không đọc `points2D` (đúng giả định); đối chiếu 5 mục
  "pending" cuối `MILESTONE_03` với hiện trạng — cả 5 đều đã được giải quyết/ghi nhận rõ
  ở các pass sau, không mục nào bị bỏ sót; xác nhận `get_scene()` fail loudly ở mọi nơi
  gọi; phát hiện Bug #2 (GPU-check cell); phát hiện `tests/README.md` lạc hậu.
- Phần D: dựng fake `GS_REPO` MỚI (mock `train.py`), chạy THẬT `02_train_baseline.sh
  chair bonsai` (2 scene 1 lệnh) → hand-craft error mask CHỈ cho `chair` →
  `06_train_refine.sh chair bonsai` (2 scene 1 lệnh, xác nhận `chair` refine thành công
  lên `iteration_25` còn `bonsai` bị `[BỎ QUA]` sạch không chặn cả lệnh) → verify
  `find_latest_iteration()` chọn đúng checkpoint theo từng scene (chair=25, bonsai=20) →
  dựng fixture dataset+renders cho 7 scene → chạy THẬT `07_package_submission.py` đóng
  gói + `verify_zip()` OK cho zip mixed-round. Không phát hiện bug mới ở luồng này.
- Sửa `pipeline/kaggle_submission.ipynb` (Bug #1 — logic symlink dataset), sửa cả 4
  notebook (Bug #2 — GPU-check raise SystemExit), cập nhật `tests/README.md`.
- Verify lại: `bash -c` test trực tiếp logic symlink cũ (tái hiện bug) và mới (xác nhận
  hết), chạy lại toàn bộ test suite (`test_syntax_all.py`, `test_07_package_submission.py`)
  — vẫn PASS 100%, không regression.
- Ghi thêm mục "6d" vào `docs/PORTED_KNOWLEDGE.md` (chỉ ADD, không sửa/xoá mục có sẵn).
- Dọn sạch toàn bộ thư mục scratch tự tạo (`p3_e2e`, `pass3_symlink_test`, `verify_fix`,
  `*_cells.txt`) — `git status` xác nhận sạch, chỉ còn đúng các file đã sửa.
