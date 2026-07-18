# Milestone 05 — Kiểm tra độc lập lần 2 (verification pass #2 / 3)

## Trạng thái hiện tại

**HOÀN TẤT pass #2.** Tìm ra + sửa **1 bug thật mới** (khác hoàn toàn 2 bug pass #1 đã
tìm — góc nhìn khác, không lặp lại phạm vi đã kiểm tra). **Vì có bug thật, pass này
KHÔNG tính là "sạch" — bộ đếm 3-lần-liên-tiếp reset về 0.** Cần chạy tiếp pass #3 (agent
độc lập khác) và pass #4 sau đó nếu pass #3 cũng sạch — vẫn cần 3 lần liên tiếp KHÔNG
lỗi.

## Phạm vi đã làm

Đọc đầy đủ STEP 0 theo đúng yêu cầu: `docs/00_MASTER_PLAN.md`, `docs/PORTED_KNOWLEDGE.md`
(toàn bộ, bao gồm mục 6b), `docs/MILESTONE_00_setup.md` -> `docs/MILESTONE_04_verification_pass1.md`,
`git log --oneline`, `git show 92b6623` (đúng commit pass #1) để biết chính xác pass #1
đã đổi gì trước khi bắt đầu — không lặp lại đúng các dòng đó, chủ động tìm góc khác.

### Phần A — Chạy lại toàn bộ test suite có sẵn (baseline sanity check)
- `tests/test_syntax_all.py`: 15/15 `.py` PASS + 4/4 `.ipynb` PASS.
- `tests/test_07_package_submission.py`: 21/21 PASS.
- `bash -n` cho cả 2 file `.sh`: sạch.
- Chạy lại lần cuối SAU khi sửa bug (xem Phần C) — vẫn PASS 100%, không có regression.

### Phần B — Verify lại chính 2 fix của pass #1 (không chỉ tin theo commit message)

1. **`eval_metrics.txt` -> `.csv`**: đọc trực tiếp `pipeline/scripts/04_eval_metrics.py`
   (dòng `write_csv(renders_root / scene.name / "eval_metrics.csv", ...)`,
   `renders_root` mặc định = `pipeline_root / "work"` = `pipeline/work`). Grep TOÀN BỘ
   (không chỉ 2 chỗ pass #1 nêu) tham chiếu `eval_metrics` trong cả
   `kaggle_round2_refine.ipynb` và `kaggle_round3_refine.ipynb` — tổng cộng 6 tham chiếu
   mỗi file (đường dẫn ghi + đọc lại ở 2 cell, biến `before`/`after`) — TẤT CẢ đều dùng
   `.csv` và đường dẫn `/kaggle/working/pipeline/work/{SCENE}/eval_metrics.csv`, khớp
   CHÍNH XÁC (cả tên file lẫn thư mục) với nơi `04_eval_metrics.py` thật sự ghi ra.
   **Xác nhận fix đúng, không sót chỗ nào.**
2. **`git submodule update --init --recursive`**: dump + diff cell clone
   `gaussian-splatting` (Bước 2) của CẢ 4 notebook bằng script Python (không đọc bằng
   mắt để tránh bỏ sót) — cả 4 file (`kaggle_round1_baseline.ipynb`,
   `kaggle_round2_refine.ipynb`, `kaggle_round3_refine.ipynb`,
   `kaggle_submission.ipynb`) có nội dung cell **giống hệt nhau byte-for-byte**, đúng thứ
   tự `git checkout <commit pin>` RỒI MỚI `git submodule update --init --recursive`
   (không đảo ngược). **Xác nhận fix đúng vị trí, đúng thứ tự, nhất quán cả 4 notebook.**

### Phần C — Tìm góc mới (khác Phần B/C của pass #1)

Checklist đã làm, tất cả **KHÔNG phát hiện thêm vấn đề** trừ mục cuối:

- **Multi-scene invocation thật của `02_train_baseline.sh`/`06_train_refine.sh`**: đọc
  code (không chỉ đọc docstring) — cả 2 dùng `for SCENE in "$@"; do ... done`, thật sự
  lặp qua nhiều scene truyền vào dòng lệnh (không phải chỉ hỗ trợ trên giấy). Khớp đúng
  docstring "cách dùng" của cả 2 file.
- **`07_package_submission.py` + `kaggle_submission.ipynb` đối chiếu `docs/00_MASTER_PLAN.md`
  mục 1**: đọc trực tiếp `target_filename()` (`literal` mode giữ NGUYÊN `image_name` từ
  CSV, kể cả đuôi `.JPG`), `encode_for_arcname()` (mã hoá lại đúng định dạng theo đuôi
  thật, không đổi tên suông), `check_scene()`/`verify_zip()` (đối chiếu kích thước ảnh
  với cột `width`/`height` của `test_poses.csv`, đối chiếu đủ số ảnh = số pose). Đúng yêu
  cầu đề bài mục 1 (tên/đuôi/kích thước ảnh khớp `test_poses.csv`). Không phát hiện sai
  lệch.
- **Đường dẫn tương đối/`parents[N]`**: grep toàn bộ `parents[N]` trong `pipeline/` — tất
  cả dùng `Path(__file__).resolve()` (không dựa vào `cwd`), độ sâu đúng với vị trí file
  thật (`pipeline/scripts/*.py` -> `parents[1]` = `pipeline/`; `pipeline/common/scenes.py`
  -> `parents[2]` = gốc repo) — không phụ thuộc thư mục làm việc lúc gọi script, an toàn
  khi Kaggle chạy từ `/kaggle/working/pipeline/scripts/` hay bất kỳ đâu khác.
- **`int()`/`float()` trên dữ liệu parse từ file**: `06_train_refine.sh` đọc
  `error_masks/manifest.json` malformed -> Python heredoc raise exception -> exit code
  khác 0 -> bash `||` bắt được, dừng script rõ ràng (không im lặng sai). Thư mục
  `point_cloud/iteration_<N>` có hậu tố không phải số (file/thư mục rác) -> cả
  `_latest_iteration_dir()` (`06_train_refine.sh`, và nay cả `02_train_baseline.sh`) lẫn
  `find_latest_iteration()` (`03_render_test_poses.py`, `05_generate_error_mask.py`) đều
  validate bằng regex (`[[ "$n" =~ ^[0-9]+$ ]]`) hoặc để `int()` tự raise lỗi rõ ràng —
  đúng triết lý "fail loudly, not silently misbehave".
- **`!command {expr}` và `print(` thiếu tiền tố `f`**: quét lại toàn bộ 4 notebook bằng
  script (không chỉ đọc mắt) — mọi nội suy trong `!lệnh {...}` chỉ dùng tên biến đơn giản
  hoặc subscript đơn giản (`{os.environ['GS_REPO']}`); không tìm thấy `print(...)` nào
  thiếu tiền tố `f` dù có `{...}` bên trong. Không có bug mới loại này.
- **`pipeline/common/`** (`scenes.py`, `poses.py`, `colmap_runner.py`, `logging_utils.py`,
  `alignment.py`): đọc lại toàn bộ — không còn tham chiếu tên repo/số scene cũ nào khác
  ngoài 2 chỗ pass #1 đã sửa (đã xác nhận `colmap_runner.py`/`logging_utils.py` đúng "7
  scene" rồi). `scenes.py` dùng `BTS_DATASET_ROOT` env var hoặc tự suy `parents[2]` —
  không hardcode tên thư mục repo nào.
- **`.gitignore` + `git ls-files`**: `Dataset/`, `pipeline/work/`, `checkpoints/`,
  `gaussian-splatting/`, `*.zip` đều bị ignore đúng. `git ls-files` liệt kê 31 file — toàn
  bộ đều nhẹ (script/notebook/doc), không có file nặng/generated nào lọt vào git.
  `git status` sạch (chỉ có đúng 1 file mình sửa).
- **Cơ chế "vòng nào cũng nộp được, không phân biệt round"**: đọc trực tiếp
  `03_render_test_poses.py::find_latest_iteration()` — chỉ tự dò iteration LỚN NHẤT có
  sẵn trong `gs_model/point_cloud/`, KHÔNG có logic nào tham chiếu "round"/"vòng" ở bất
  kỳ đâu trong code. `kaggle_submission.ipynb` Bước 5 tải `gs_model/` nguyên vẹn của vòng
  đã CHỌN (field `"round"` trong `CHECKPOINT_LINKS` chỉ để ghi chú, không dùng trong logic
  — grep xác nhận biến `entry["round"]` chỉ xuất hiện trong 1 dòng `print()`), Bước 6 gọi
  `03_render_test_poses.py --scene {scene}` (không truyền `--iteration`, dùng mặc định tự
  dò) cho MỌI scene trong vòng lặp giống hệt nhau — xác nhận đúng thiết kế: scene A dùng
  checkpoint Vòng 1, scene B dùng checkpoint Vòng 3, cùng chạy qua 1 vòng lặp không phân
  biệt gì cả. **Xác nhận ĐÚNG mục tiêu thiết kế "không phân biệt round", không phải chỉ
  do tài liệu khẳng định suông.**

**Bug thật MỚI (khác pass #1) — `02_train_baseline.sh` chọn nhầm checkpoint trong
thông báo "[CỨU ĐƯỢC]" khi train crash giữa chừng:**

Dòng gốc (trước khi sửa):
```bash
LAST_CKPT=$(ls -d "$MODEL_DIR"/point_cloud/iteration_* 2>/dev/null | sort -t_ -k2 -n | tail -1 || true)
```
Đây CHÍNH XÁC là bug class mà milestone 02 đã tự phát hiện + fix ở `06_train_refine.sh`
(`_latest_iteration_dir()`, xem comment giải thích ngay trong file đó) — nhưng KHÔNG được
backport lại vào `02_train_baseline.sh`, nơi cùng pattern `sort -t_ -k2 -n` này vẫn tồn
tại nguyên vẹn từ trước milestone 02. Nguyên nhân: đường dẫn model thật
(`.../gs_model/point_cloud/iteration_15000`) chứa NHIỀU dấu `_` đứng TRƯỚC
"iteration_N" (chính "gs_model" và "point_cloud" đều có `_`), nên field thứ 2 tách theo
`_` (`-t_ -k2`) KHÔNG phải là số iteration — `sort -n` coi field không parse được là `0`,
giữ nguyên thứ tự lexical gốc của `ls -d` (glob liệt kê "iteration_15000" TRƯỚC
"iteration_7000" vì ký tự '1' < '7' theo thứ tự chữ cái) — `tail -1` chọn NHẦM
"iteration_7000", checkpoint CŨ HƠN/kém train hơn, thay vì "iteration_15000".

**Verify bằng lệnh `sort` thật** (không suy đoán): dựng 2 thư mục giả
`iteration_7000`/`iteration_15000` dưới 1 đường dẫn có dấu cách (mô phỏng đúng đường dẫn
dự án thật "Khóa Luận Tốt Nghiệp") — `sort -t_ -k2 -n | tail -1` trả về `iteration_7000`
(SAI, đáng lẽ phải là `iteration_15000`).

**Verify bằng chạy THẬT `02_train_baseline.sh`** với `train.py` giả (mock, hỗ trợ
`CRASH_AT`): `ITERATIONS=18500 CRASH_AT=18000` (đủ để cả 2 checkpoint 7000 và 15000 đều
được lưu trước khi crash ở 18000) — TRƯỚC khi sửa, thông báo `[CỨU ĐƯỢC]` sẽ trỏ sai vào
`iteration_7000`; SAU khi sửa, chạy lại xác nhận thông báo trỏ ĐÚNG
`iteration_15000`.

**Tác động**: đây CHỈ là dòng thông báo hiển thị cho người dùng khi train thất bại giữa
chừng (không dùng để quyết định logic gì khác trong script, không ảnh hưởng file/thư mục
thật nào) — nhưng có thể khiến người dùng dưới áp lực deadline (30/07/2026) tin nhầm và
cố ý dùng checkpoint `iteration_7000` (train ít hơn, chất lượng thấp hơn) để render/nộp
bài thay vì `iteration_15000` sẵn có tốt hơn, dù thông báo tự nhận là "checkpoint gần
nhất".

**Đã sửa**: `pipeline/scripts/02_train_baseline.sh` — thêm hàm `_latest_iteration_dir()`
(bash thuần, vòng lặp so sánh số nguyên, không tách trường theo `_`/khoảng trắng —
y hệt cách đã verify đúng ở `06_train_refine.sh`), thay dòng
`LAST_CKPT=$(ls -d ... | sort -t_ -k2 -n | tail -1 || true)` bằng
`LAST_CKPT="$(_latest_iteration_dir "$MODEL_DIR")"`. Verify lại: `bash -n` sạch, chạy lại
kịch bản crash-sau-2-checkpoint xác nhận thông báo đúng, chạy lại toàn bộ
`tests/test_syntax_all.py`/`tests/test_07_package_submission.py` xác nhận không có
regression.

Đã ghi thêm mục "6c" vào `docs/PORTED_KNOWLEDGE.md` (KHÔNG xoá/sửa mục nào có sẵn, chỉ
ADD) — kèm bài học: khi 1 bug class được fix ở 1 file, phải chủ động kiểm tra các file
khác có đoạn code TƯƠNG TỰ (cùng nguồn gốc/cùng tác giả viết cùng lúc) có dính cùng bug
đó chưa, không coi là "đã xử lý xong" chỉ vì đã sửa đúng 1 chỗ tìm thấy đầu tiên.

## Giới hạn của pass này (ghi rõ, không giấu)

- KHÔNG dựng lại toàn bộ mock end-to-end 1-GS_REPO-thống-nhất như pass #1 (Phần C/D của
  pass #1 đã làm kỹ, dùng code thật không-CUDA từ đúng commit pin) — thay vào đó tập
  trung đọc code trực tiếp + test có mục tiêu (targeted) cho từng câu hỏi cụ thể trong
  checklist, cộng 1 test thật (mock `train.py` + `02_train_baseline.sh` thật) riêng cho
  đúng bug tìm được. Cách này bao phủ ĐƯỢC các câu hỏi trọng tâm mà STEP D yêu cầu (multi-
  scene loop, round-agnostic CHECKPOINT_LINKS, đối chiếu format submission, path
  resolution, malformed input) bằng cách đọc source + test có mục tiêu, nhưng KHÔNG lặp
  lại được đầy đủ chuỗi Vòng1->Vòng2->Vòng3->render->package bằng 1 fake GS_REPO hoàn
  chỉnh dùng code thật không-CUDA như pass #1 đã làm — nếu muốn phủ lại chuỗi đó bằng 1
  bộ fixture MỚI hoàn toàn (không tái dùng scratch cũ), cần thêm 1 lượt riêng.
- Vẫn CHƯA chạy được train/render 3DGS thật trên GPU Kaggle (không có GPU cục bộ đủ
  toolchain CUDA) — giới hạn này giống hệt mọi milestone trước, không đổi.
- Không phát hiện thêm vấn đề nào trong `00_make_holdout_split.py`/`01_run_colmap.py`
  ngoài việc đọc lại code (không chạy thật với `pycolmap`/dữ liệu COLMAP thật cục bộ) —
  giới hạn đã ghi từ milestone 01, chưa thay đổi.

## Bước tiếp theo

1. Chạy **pass #3** (agent verify độc lập khác) — vì pass #2 tìm ra bug thật, bộ đếm
   reset về 0, cần 3 lần liên tiếp KHÔNG lỗi mới coi xong. Gợi ý cho pass #3: thử phủ lại
   đầy đủ chuỗi mock end-to-end (Vòng1->Vòng2->Vòng3->render->package) bằng 1 fake
   `GS_REPO` MỚI dựng từ đầu (không tái dùng scratch cũ) để tăng độ phủ so với pass này —
   pass #1 đã làm 1 lần, làm lại độc lập lần 2 có thể bắt được bug khác (khác góc nhìn
   assertion/mock).
2. Các hạng mục "CHƯA test được" liệt kê ở milestone 01/02/03/04 (train/render GPU thật
   trên Kaggle) vẫn còn nguyên, không nằm trong phạm vi verify cục bộ.

## Lịch sử

### 2026-07-18 — Verification pass #2 (agent kiểm tra độc lập, khác pass #1)
- Đọc đầy đủ STEP 0 (`00_MASTER_PLAN.md`, `PORTED_KNOWLEDGE.md` toàn bộ kể cả mục 6b, 5
  milestone log 00-04, `git log --oneline`, `git show 92b6623` để biết chính xác pass #1
  đã sửa gì).
- Phần A: chạy lại `tests/test_syntax_all.py` (15/15 + 4/4 PASS), `tests/test_07_package_submission.py`
  (21/21 PASS), `bash -n` 2 file `.sh` — sạch, làm baseline trước khi tìm bug mới.
- Phần B: re-verify độc lập 2 fix của pass #1 bằng cách đọc trực tiếp code nguồn
  (`04_eval_metrics.py::write_csv()`, diff byte-for-byte cell clone `gaussian-splatting`
  của cả 4 notebook) — xác nhận CẢ 2 fix đúng, không sót chỗ nào.
- Phần C: quét theo checklist các góc pass #1 chưa làm (multi-scene loop thật,
  round-agnostic CHECKPOINT_LINKS, đối chiếu format submission với đề bài, path
  resolution/`parents[N]`, malformed JSON/int(), `.gitignore`/`git ls-files`, quét lại
  toàn bộ `!lệnh {...}`/`print(f...)` bằng script) — tìm ra 1 bug thật mới: thông báo
  "[CỨU ĐƯỢC]" của `02_train_baseline.sh` chọn nhầm checkpoint nhỏ hơn khi có >=2
  checkpoint tồn tại lúc crash, do CÙNG bug class `sort -t_ -k2 -n` đã fix ở
  `06_train_refine.sh` (milestone 02) nhưng chưa backport.
- Verify bug bằng 2 cách: (1) lệnh `sort` thật trên thư mục giả có dấu cách trong đường
  dẫn, xác nhận chọn sai; (2) chạy `02_train_baseline.sh` thật với `train.py` giả
  (`ITERATIONS=18500 CRASH_AT=18000`, đủ để có cả 2 checkpoint 7000+15000 trước khi
  crash) — trước khi sửa chọn sai `iteration_7000`, sau khi sửa chọn đúng
  `iteration_15000`.
- Sửa `pipeline/scripts/02_train_baseline.sh`: thêm `_latest_iteration_dir()` (bash
  thuần, so sánh số nguyên, không tách trường theo `_`/khoảng trắng), thay chỗ dùng
  `sort -t_ -k2 -n` bằng hàm mới. Verify lại `bash -n` sạch + chạy lại toàn bộ test suite
  (`test_syntax_all.py`, `test_07_package_submission.py`) — vẫn PASS 100%, không
  regression.
- Ghi thêm mục "6c" vào `docs/PORTED_KNOWLEDGE.md` (chỉ ADD, không sửa/xoá mục có sẵn).
- Dọn sạch thư mục scratch tự tạo trong phiên này (`sorttest`, `rescuetest`, `verify2`)
  — `git status` xác nhận sạch, chỉ còn đúng 1 file sửa
  (`pipeline/scripts/02_train_baseline.sh`).
