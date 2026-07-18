# Milestone 10 — Kiểm tra độc lập lần 7 (verification pass #7 / N)

## Trạng thái hiện tại

**HOÀN TẤT pass #7 — TÌM THẤY + SỬA 1 BUG THẬT NGHIÊM TRỌNG.** Bộ đếm 3-lần-liên-tiếp-
không-lỗi hiện tại: **0/3** (reset, vì pass này có phát hiện thật cần sửa). Cần lại đủ
3 pass sạch liên tiếp kể từ pass tiếp theo.

## Phạm vi đã làm

Đọc đầy đủ STEP 0: `docs/00_MASTER_PLAN.md`, `docs/PORTED_KNOWLEDGE.md` toàn bộ (kể cả
mục 6b–6f), cả 9 milestone log trước (`MILESTONE_00` → `MILESTONE_09`), `git log
--oneline`, `git show 050dab2` (đúng diff pass #6) để biết chính xác phạm vi đã audit.

### Phần A — Baseline: test suite

- `tests/test_syntax_all.py`: 15/15 `.py` PASS + 4/4 `.ipynb` PASS.
- `tests/test_07_package_submission.py`: 21/21 PASS.
- `bash -n` cho `02_train_baseline.sh` + `06_train_refine.sh`: sạch.
- Chạy lại lần cuối SAU khi sửa (mục C) — vẫn PASS 100%, không regression.

### Phần B — Audit import/dependency exhaustive (theo đúng chỉ đạo nhiệm vụ, vì pass #6
tìm ra bug thiếu `opencv-python`)

Liệt kê TOÀN BỘ `import`/`from ... import` top-level của mọi `pipeline/scripts/*.py` +
`pipeline/common/*.py`, đối chiếu với dòng `!pip install` thật của cả 4 notebook:

- Thư viện bên thứ 3 cần cài: `pycolmap` (00_make_holdout_split.py, colmap_runner.py,
  logging_utils.py), `scikit-image`/`skimage` + `lpips` (04_eval_metrics.py), `cv2`/
  `opencv-python` (05_generate_error_mask.py). Cả 4 notebook đều
  `!pip install -q "pycolmap>=3.10" "scikit-image>=0.19" lpips plyfile tqdm
  opencv-python-headless "gdown>=6,<7"` — **KHỚP ĐỦ, nhất quán cả 4 file** (đã fix ở
  pass #6, re-verify lại đúng vẫn còn nguyên). **KHÔNG tìm thêm instance thứ 2** của bug
  class "script import X nhưng không notebook nào cài X".
- `torch`/`numpy`/`PIL` — có sẵn base image Kaggle, không cần cài thêm (đúng giả định an
  toàn theo yêu cầu nhiệm vụ).
- `plyfile`/`tqdm` được cài nhưng **KHÔNG được import trực tiếp** bởi bất kỳ script nào
  trong `pipeline/` (grep xác nhận 0 kết quả) — xác nhận đây là dependency của CHÍNH
  `train.py`/`scene/gaussian_model.py` (repo `graphdeco-inria/gaussian-splatting`
  ngoài, không phải code của ta), cài theo đúng nguyên tắc "superset" đã áp dụng sẵn cho
  `pycolmap`/`lpips` ở các notebook không trực tiếp gọi script cần chúng — không phải
  thiếu sót.
- `pipeline/common/alignment.py` chỉ dùng `numpy` — an toàn, và hiện KHÔNG được import
  bởi script nào trong repo này (utility "ngủ đông", đúng như pass #5 đã ghi nhận cho
  bug scale COLMAP — không phải vấn đề).

**Kết luận Phần B: KHÔNG tìm thêm bug thiếu dependency nào.**

### Phần C — Góc mới theo yêu cầu nhiệm vụ

1. **Milestone 00–03, tìm claim "sẽ verify sau"/"chưa kiểm tra, để sau" bị bỏ sót**:
   grep `sẽ...sau|để sau|chưa...kiểm tra|chưa...test|chưa...verify|CHƯA|còn thiếu` trên
   cả 4 file — mọi kết quả đều quy về CÙNG 1 nhóm hạng mục: "train/render 3DGS thật
   trên GPU Kaggle" (chưa làm được, không có GPU cục bộ đủ toolchain CUDA) — hạng mục
   này được các pass sau NHẤT QUÁN nhắc lại, KHÔNG bị quên/rơi qua khe nứt. Không tìm
   thấy claim "để sau" nào khác bị bỏ sót.
2. **`apply_error_refine_patch.py` có xung đột `--start_checkpoint`/
   `--checkpoint_iterations` với `--refine_from_iteration` không?** Đọc trực tiếp logic
   xây dựng lệnh `train.py` trong CẢ 2 file `.sh`:
   - `02_train_baseline.sh` (dòng gọi `train.py`): KHÔNG bao giờ truyền
     `--start_checkpoint`/`--checkpoint_iterations`/`--refine_from_iteration` — luôn
     train từ đầu (`first_iter=0`).
   - `06_train_refine.sh` (dòng gọi `train.py`): CHỈ truyền `--refine_from_iteration`
     + `--error_mask_dir` (2 cờ patch thêm) — KHÔNG BAO GIỜ truyền `--start_checkpoint`/
     `--checkpoint_iterations`.
   - `apply_error_refine_patch.py::PATCHES` — call site vá thêm đúng 2 tham số mới vào
     cuối lời gọi `training(...)`, không đụng tới `checkpoint`/`checkpoint_iterations`
     gốc (vẫn truyền `args.start_checkpoint`/`args.checkpoint_iterations` nguyên vẹn,
     luôn là `None`/rỗng vì không script nào set 2 cờ CLI đó).
   **Kết luận: KHÔNG có xung đột** — 2 cơ chế resume (cũ `--start_checkpoint` cần
   `.pth`, mới `--refine_from_iteration` dùng `.ply` qua `Scene(load_iteration=N)`)
   không bao giờ được dùng đồng thời trong repo này.
3. **Hardcoded iteration numbers**: grep `7000|15000|30000|18000|21000` toàn
   `pipeline/`. Chỉ xuất hiện ở `02_train_baseline.sh` (giá trị mặc định
   `ITERATIONS`/`SAVE_ITERATIONS`, tự động theo `$ITERATIONS`, không hardcode "iteration
   cuối") và `kaggle_round1_baseline.ipynb` (`ITERATIONS = "15000" if MODE=="holdout"
   else "30000"`, đúng thiết kế 2 mode). `03_render_test_poses.py`/
   `05_generate_error_mask.py::find_latest_iteration()` hoàn toàn ĐỘNG (tự dò iteration
   lớn nhất có sẵn bằng `glob("iteration_*")`, không hardcode con số nào) — checkpoint
   Vòng 2/3 ở số luỹ kế bất kỳ (18000, 21000, ...) đều được nhận diện đúng, không có
   code path nào giả định chỉ có 7000/15000/30000. **Không tìm thấy vấn đề.**
4. **Đọc lại toàn bộ `tests/test_syntax_all.py` + `tests/test_07_package_submission.py`**
   (không chỉ tin "21/21 PASS" như hộp đen) — xác nhận từng assertion đều có Ý NGHĨA
   THẬT (vd test #3 dựng zip giả đúng kiểu bug cũ "đổi tên .jpg giữ nguyên byte PNG" rồi
   assert `verify_zip()` PHẢI raise; test #5 dựng thiếu 1 ảnh + sai kích thước 1 ảnh rồi
   assert exit code != 0 VÀ log có nhắc đúng tên scene lỗi) — không có assertion nào
   "trivially true" (vd `assert True`, so sánh giá trị với chính nó). **Test suite đáng
   tin cậy.**
5. **Consistency Score holdout-estimated vs BTC-graded, rủi ro nộp nhầm checkpoint
   `MODE="holdout"`** — xem mục "Bug tìm thấy" dưới đây, đây chính là góc phát hiện ra
   bug nghiêm trọng nhất của pass này.

### Phần D — Mock end-to-end: checkpoint `MODE="final"` (Vòng 1) feed vào Vòng 2 refine

Theo đúng yêu cầu nhiệm vụ — thử kịch bản: checkpoint Vòng 1 train ở `MODE="final"`
(100% ảnh, 30000 iteration, đúng dạng checkpoint sẽ dùng để nộp bài thật) được đưa vào
`kaggle_round2_refine.ipynb`. `06_train_refine.sh`/2 notebook refine cần GT holdout để
đo Score trước/sau — kịch bản này lộ ra **chính xác bug đã tìm thấy** (xem dưới).

Đã verify bằng THỰC THI THẬT (không suy đoán):
- Trích xuất hàm `choose_holdout_names()` (thuần Python, không cần `pycolmap`) từ
  `00_make_holdout_split.py`, chạy với 200 tên ảnh giả: xác nhận **hoàn toàn
  deterministic** (2 lần gọi cùng scene/seed/frac cho ĐÚNG 1 tập holdout giống hệt
  nhau) — seed=42 cố định không phụ thuộc lịch sử phiên chạy trước.
- Mô phỏng "checkpoint MODE=final" (train 100% ảnh): tập holdout Vòng 2 tự dựng lại
  **CHỒNG LẤN 100%** với tập ảnh checkpoint đã train — **RÒ RỈ DỮ LIỆU HOÀN TOÀN**.
- Mô phỏng "checkpoint MODE=holdout" (train chỉ trên phần loại-trừ-holdout): tập
  holdout Vòng 2 tự dựng lại **CHỒNG LẤN 0%** với tập ảnh checkpoint đã train — SẠCH,
  đúng ý nghĩa.
- Trích xuất NGUYÊN VĂN đoạn code thật (không viết lại/diễn giải) trong cell "Bước 6"
  của `kaggle_round2_refine.ipynb`/`kaggle_round3_refine.ipynb` (sau khi đã sửa, xem
  mục "Đã sửa" dưới), chạy qua `exec()` với `pipeline_train_flags.json` giả có
  `train_mode="final"`/`"holdout"`/vắng mặt — xác nhận **CẢ 3 nhánh hoạt động đúng như
  thiết kế**: `"final"` → `raise SystemExit` (lỗi RÕ RÀNG, có tên scene + lý do + hướng
  khắc phục, KHÔNG phải crash mù mờ); `"holdout"` → im lặng cho qua (đúng, checkpoint
  hợp lệ); vắng mặt (checkpoint cũ chưa có field) → in `[CẢNH BÁO]` rõ ràng nhưng vẫn
  cho tiếp tục chạy (tương thích ngược, không chặn checkpoint hợp lệ nhưng thiếu field
  mới).
- Verify riêng `06_train_refine.sh`: field `train_mode` được BẢO TOÀN nguyên vẹn qua
  cơ chế `flags.setdefault(...)` khi ghi thêm `refine_history` sau mỗi vòng refine
  (test bằng đúng đoạn code Python heredoc trích từ script) — nghĩa là chuỗi Vòng
  1(holdout) → Vòng 2 → Vòng 3 vẫn giữ đúng `train_mode="holdout"` xuyên suốt, không bị
  mất dấu vết ở Vòng 3.

## Bug thật tìm ra + đã sửa

### Bug — Rò rỉ dữ liệu (data leakage) nếu dùng checkpoint `MODE="final"` làm input Vòng 2+, do 2 notebook TỰ MÂU THUẪN NHAU về yêu cầu MODE

**Mô tả đầy đủ, đối chiếu bằng chứng thật xem `docs/PORTED_KNOWLEDGE.md` mục 6g.** Tóm
tắt:

- `kaggle_round1_baseline.ipynb` (TRƯỚC khi sửa) khẳng định `MODE="final"` "là checkpoint
  Vòng 1 dùng làm điểm khởi đầu cho Vòng 2+", còn `MODE="holdout"` thì "KHÔNG dùng làm
  input Vòng 2+".
- `kaggle_round2_refine.ipynb`/`kaggle_round3_refine.ipynb` (không đổi từ milestone 02)
  lại khẳng định NGƯỢC LẠI: "Yêu cầu: checkpoint Vòng 1 đã train ở `MODE="holdout"`".
- **2 tài liệu mâu thuẫn trực tiếp** — không pass nào trong 6 pass trước đối chiếu
  đúng câu chữ "MODE nào làm input Vòng 2+" giữa 2 cặp notebook, dù đã đối chiếu rất kỹ
  nhiều thứ khác (schema JSON, CLI contract, tên file, GPU-check, symlink dataset...).
- Xác định bên nào đúng bằng đọc CODE + THỰC THI THẬT (không suy đoán): Vòng 2+ LUÔN tự
  dựng lại đúng 1 tập holdout CỐ ĐỊNH (seed=42) ở phiên Kaggle MỚI của nó để đo Score
  TRƯỚC/SAU — nếu checkpoint nạp vào đã train trên CHÍNH các ảnh đó (`MODE="final"`,
  100% ảnh) thì Score TRƯỚC bị rò rỉ dữ liệu hoàn toàn (đã verify: overlap 100%).
  `MODE="holdout"` mới là lựa chọn ĐÚNG (overlap 0%, đã verify).
- **Hậu quả thật**: nếu người dùng làm đúng theo hướng dẫn CŨ (dùng `"final"` làm input
  Vòng 2+), Score TRƯỚC bị thổi phồng giả tạo; refine 1 đợt ngắn (chỉ chỉnh trên phần
  không-holdout) rất có thể đo ra Score SAU thấp hơn Score TRƯỚC giả tạo đó — notebook
  kết luận SAI "KHÔNG cải thiện, GIỮ Vòng 1, DỪNG LẠI" — **âm thầm loại bỏ 1 cải tiến
  thật có ích**, không có lỗi/crash nào báo. Đây đúng loại lỗi nguy hiểm nhất theo tinh
  thần cốt lõi của `docs/00_MASTER_PLAN.md` mục 3.2 bước 4 ("KHÔNG tin bằng trực giác,
  luôn đo Score thật") — chính PHÉP ĐO bị hỏng, không phải thuật toán refine.

**Đã sửa (chỉ thêm, không đổi hành vi cũ khi field mới vắng mặt — tương thích ngược):**

1. `pipeline/scripts/02_train_baseline.sh` — thêm env var `TRAIN_MODE`, ghi field
   `"train_mode"` (string hoặc `null`) vào `pipeline_train_flags.json`.
2. `pipeline/kaggle_round1_baseline.ipynb` — Bước 5: `os.environ["TRAIN_MODE"] = MODE`
   trước khi gọi `02_train_baseline.sh`. Sửa NGÔN TỪ Bước 5/Bước 6 markdown: đảo lại
   đúng kỹ thuật (`"holdout"` PHẢI dùng làm input Vòng 2+, `"final"` chỉ dùng để nộp bài
   trực tiếp khi KHÔNG chạy thêm Vòng 2+).
3. `pipeline/kaggle_round2_refine.ipynb`/`kaggle_round3_refine.ipynb` — Bước 6 (cell tải
   checkpoint): thêm chặn cứng đọc `pipeline_train_flags.json["train_mode"]` — `"final"`
   → `raise SystemExit` rõ ràng; vắng mặt → `[CẢNH BÁO]` không chặn (tương thích ngược
   với checkpoint train trước khi có field này).
4. `docs/00_MASTER_PLAN.md` mục 3.2 bước 1 — thêm ghi chú bắt buộc.
5. `docs/PORTED_KNOWLEDGE.md` — thêm mục 6g đầy đủ chi tiết + bằng chứng.

**Verify sau sửa**: `bash -n`/`py_compile`/`nbformat.validate()` sạch; chạy thật
`02_train_baseline.sh` (mock `train.py`) qua `TRAIN_MODE=holdout` VÀ không set —
`pipeline_train_flags.json` ghi đúng `"train_mode": "holdout"` / `"train_mode": null`;
trích NGUYÊN VĂN đoạn code thật của cả 2 notebook Vòng 2/3, chạy qua `exec()` với 3 giá
trị `train_mode` — cả 3 nhánh đúng thiết kế (xem Phần D). Test suite
(`test_syntax_all.py` 15/15+4/4, `test_07_package_submission.py` 21/21) PASS 100% sau
sửa, không regression.

**Giới hạn còn lại (đánh đổi THIẾT KẾ, không phải bug, KHÔNG tự ý mở rộng sửa ở pass
này)**: bắt buộc dùng `MODE="holdout"` làm input Vòng 2+ nghĩa là checkpoint cuối cùng
sau khi refine LUÔN thiếu ~12.5% ảnh train + khởi đầu từ 15000 iteration (không phải
30000 của `"final"`) so với 1 checkpoint không-refine. Chưa có cơ chế tự động "nâng
cấp" 1 cấu hình refine đã validate-bằng-holdout lên bản `MODE="final"` 100% dữ liệu để
nộp bài — đây là hạn chế kiến trúc đã biết, cần quyết định của người dùng, ghi rõ ở
`docs/PORTED_KNOWLEDGE.md` mục 6g để không ai hiểu nhầm là bug chưa sửa.

## Giới hạn của pass này (ghi rõ, không giấu)

- Vẫn CHƯA chạy được train/render 3DGS thật trên GPU Kaggle — giới hạn không đổi qua
  mọi pass.
- `00_make_holdout_split.py`/`01_run_colmap.py` vẫn chỉ audit bằng đọc code + verify
  hàm `choose_holdout_names()` thuần Python tách rời (không cần `pycolmap`, không có
  sẵn cục bộ) — chưa chạy full script thật với `pycolmap` + dữ liệu COLMAP thật (gợi ý
  tồn đọng từ nhiều pass trước, vẫn chưa ai làm, rủi ro thấp vì port gần nguyên vẹn từ
  repo tiền nhiệm đã chạy thật 7/7 scene).
- Bug tìm được ở pass này là bug TÀI LIỆU + THIẾU CHẶN KỸ THUẬT (không phải lỗi cú
  pháp/crash) — mức độ nghiêm trọng cao vì ảnh hưởng trực tiếp tới độ tin cậy của cơ
  chế Score-gating cốt lõi, nhưng CHỈ lộ ra nếu người dùng thực sự làm theo đúng hướng
  dẫn CŨ (dùng `"final"` làm input Vòng 2+) — chưa có bằng chứng đã có ai làm vậy trên
  Kaggle thật (chưa chạy Kaggle thật ở bất kỳ pass nào), nhưng hướng dẫn CŨ hoàn toàn có
  thể dẫn người dùng thật đi đúng đường sai này.

## Bước tiếp theo

1. Chạy **pass #8** (agent verify độc lập khác) — bộ đếm sạch liên tiếp: 0/3.
2. Gợi ý cho pass #8: re-verify kỹ fix `train_mode` của pass này (đọc code trực tiếp,
   không chỉ tin milestone log) theo đúng thông lệ đã thiết lập. Các khu vực đã soi rất
   kỹ nhiều lần (schema `pipeline_train_flags.json` 4-key gốc, va chạm thư mục
   iteration, đường dẫn dấu cách, `spatial_lr_scale`, tên file `eval_metrics`, `git
   submodule update`, checkpoint sort bug, symlink dataset, GPU-check, rollback/
   stop-at-round-N, adversarial input, re-run stale checkpoint, `gdown --fuzzy`, thiếu
   `opencv-python`) nên giảm ưu tiên. Còn thực sự chưa test bằng dữ liệu thật cục bộ:
   `00_make_holdout_split.py`/`01_run_colmap.py` full script với `pycolmap` + fixture
   COLMAP thật (gợi ý tồn đọng nhiều pass, vẫn chưa ai làm).
3. Trước khi chạy Kaggle thật lần đầu: LUÔN dùng checkpoint `MODE="holdout"` làm input
   cho `kaggle_round2_refine.ipynb`/`kaggle_round3_refine.ipynb` (đã sửa tài liệu +
   thêm chặn cứng ở pass này, nhưng double-check bằng mắt khi thao tác thật trên Kaggle
   trước khi tải checkpoint lên Drive, đề phòng nhầm lẫn thao tác thủ công).

## Lịch sử

### 2026-07-18 — Verification pass #7 (agent kiểm tra độc lập, khác pass #1-6)
- Đọc đầy đủ STEP 0 (`00_MASTER_PLAN.md`, `PORTED_KNOWLEDGE.md` toàn bộ kể cả mục
  6b-6f, 10 milestone log 00-09, `git log --oneline`, `git show 050dab2`).
- Phần A: test suite baseline PASS 100%.
- Phần B: audit exhaustive import/dependency của mọi `pipeline/scripts/*.py` +
  `pipeline/common/*.py` đối chiếu `!pip install` — không tìm thêm bug thiếu dependency
  (fix pass #6 đã đủ + nhất quán).
- Phần C: rà milestone 00-03 tìm claim "để sau" bị bỏ sót (không có); audit xung đột
  `--start_checkpoint`/`--refine_from_iteration` (không có); grep hardcoded iteration
  numbers toàn repo (sạch, mọi nơi đều dynamic); đọc kỹ toàn bộ 2 file test xác nhận
  assertion có ý nghĩa thật; audit consistency Score holdout/final — **tìm ra bug thật
  nghiêm trọng nhất pass này**: `kaggle_round1_baseline.ipynb` và
  `kaggle_round2_refine.ipynb`/`kaggle_round3_refine.ipynb` tự mâu thuẫn nhau về MODE
  nào (`"holdout"` hay `"final"`) phải dùng làm input Vòng 2+ — xác định `"holdout"`
  mới đúng (verify bằng thực thi thật: overlap 100% ảnh nếu dùng `"final"` = rò rỉ dữ
  liệu hoàn toàn vào phép đo Score).
- Phần D: verify bằng thực thi thật hàm `choose_holdout_names()` (deterministic,
  overlap 100%/0% tuỳ MODE) + trích nguyên văn code thật của cell "Bước 6" (sau khi sửa)
  chạy qua `exec()` với 3 giá trị `train_mode` — xác nhận cả 3 nhánh (SystemExit rõ
  ràng/im lặng qua/cảnh báo không chặn) hoạt động đúng thiết kế.
- Sửa: `pipeline/scripts/02_train_baseline.sh` (thêm `TRAIN_MODE` + field
  `pipeline_train_flags.json`), `pipeline/kaggle_round1_baseline.ipynb` (export
  `TRAIN_MODE` + sửa ngôn từ Bước 5/6), `pipeline/kaggle_round2_refine.ipynb` +
  `pipeline/kaggle_round3_refine.ipynb` (chặn cứng `train_mode=="final"`),
  `docs/00_MASTER_PLAN.md` mục 3.2 (ghi chú bắt buộc), `docs/PORTED_KNOWLEDGE.md` mục
  6g (chi tiết đầy đủ).
- Verify lại toàn bộ SAU khi sửa: test suite PASS 100% không regression; mock thật
  `02_train_baseline.sh` xác nhận field ghi đúng cả 2 trường hợp có/không
  `TRAIN_MODE`; `exec()` trực tiếp code thật của cả 2 notebook Vòng 2/3 xác nhận 3
  nhánh xử lý đúng; verify `06_train_refine.sh` bảo toàn field `train_mode` qua
  `refine_history`.
- `git status` sạch trong suốt quá trình (không dùng `tempfile.TemporaryDirectory` để
  lại rác), chỉ còn đúng 6 file đã sửa thật trong repo.
