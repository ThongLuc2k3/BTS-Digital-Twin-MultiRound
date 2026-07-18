# Milestone 12 — Kiểm tra độc lập lần 9 (verification pass #9 / N)

## Trạng thái hiện tại

**HOÀN TẤT pass #9 — TÌM THẤY + SỬA 1 LỖI TÀI LIỆU THẬT** (tham chiếu chéo sai trong
`docs/00_MASTER_PLAN.md`, tồn tại từ commit khởi tạo repo, không pass nào trong 8 pass
trước phát hiện ra). Bộ đếm 3-lần-liên-tiếp-không-lỗi hiện tại: **0/3** (reset, vì pass
này có phát hiện thật cần sửa). Cần lại đủ 3 pass sạch liên tiếp kể từ pass tiếp theo.

Ngoài lỗi trên, pass này dành phần lớn thời gian làm 2 việc trọng tâm được giao thêm so
với 8 pass trước: (1) audit "còn hợp lý như 1 tổng thể không" (không chỉ đúng-sai từng
sự kiện) cho cả 4 notebook + `00_MASTER_PLAN.md` + `PORTED_KNOWLEDGE.md`, và (2) LẦN ĐẦU
TIÊN chạy `00_make_holdout_split.py`/`01_run_colmap.py` bằng **`pycolmap` cài thật + dữ
liệu COLMAP tổng hợp thật** (không phải mock/text thủ công) — hạng mục đã bị 5 pass liên
tiếp (04, 06, 07, 08, và ngầm định ở 09) liệt là "còn thiếu, chưa ai làm".

## Phạm vi đã làm

Đọc đầy đủ STEP 0: `docs/00_MASTER_PLAN.md`, `docs/PORTED_KNOWLEDGE.md` toàn bộ (kể cả
mục 6b–6h), cả 12 milestone log trước (`MILESTONE_00` → `MILESTONE_11`), `git log
--oneline`, `git show 1ff9341` (đúng diff pass #8, đọc từng dòng).

### Phần A — Baseline: test suite

- `python -m py_compile` toàn bộ 9 file `.py` (`pipeline/scripts/` + `pipeline/common/`):
  sạch.
- `bash -n` cho `02_train_baseline.sh` + `06_train_refine.sh`: sạch.
- `nbformat.validate()` cả 4 notebook: hợp lệ.
- `tests/test_syntax_all.py`: 15/15 `.py` + 4/4 `.ipynb` PASS.
- `tests/test_07_package_submission.py`: 21/21 PASS.
- Chạy lại lần cuối SAU khi sửa (mục "Đã sửa" dưới) — vẫn PASS 100%, không regression.

### Phần B — Verify lại fix của pass #8 bằng mock MỚI, thực thi thật (không tin lại report)

Dựng mock `GS_REPO`/`train.py` MỚI (không tái dùng scratch cũ), `python` symlink sang
`python3` (máy dev chỉ có `python3`, đúng ghi chú milestone trước), chạy thẳng
`pipeline/scripts/06_train_refine.sh` (bỏ qua notebook hoàn toàn) qua đúng **4 kịch bản**
mà pass #8 tuyên bố đã test, tự dựng lại từ đầu:

1. `pipeline_train_flags.json` có `"train_mode": "final"`, không set
   `ALLOW_FINAL_TRAIN_MODE` → **exit 1**, in `[LỖI]` đúng nội dung giải thích + hướng
   khắc phục, **xác nhận `train.py` KHÔNG hề được gọi** (không có thư mục
   `iteration_25` mới nào được tạo — kiểm tra trực tiếp bằng `ls`, không chỉ đọc log).
2. Giống trên nhưng thêm `ALLOW_FINAL_TRAIN_MODE=1` → **exit 0**, `train.py` ĐƯỢC gọi
   (checkpoint mới `iteration_25` xuất hiện đúng vị trí, đổi tên đúng theo số luỹ kế).
3. `pipeline_train_flags.json` KHÔNG có field `train_mode` (mô phỏng checkpoint train
   trước khi pass #7 thêm field này) → in `[CẢNH BÁO]`, **không chặn**, exit 0, `train.py`
   được gọi bình thường.
4. `train_mode` = `"holdout"` (trường hợp bình thường) → hoàn toàn im lặng (không dòng
   `[LỖI]`/`[CẢNH BÁO]` nào liên quan `train_mode`), exit 0, `train.py` được gọi.

**Kết quả: cả 4 kịch bản đúng chính xác như thiết kế + đúng như pass #8 báo cáo.** Điểm
xác nhận quan trọng nhất (yêu cầu cụ thể của nhiệm vụ): ở kịch bản 1, "nhánh chặn thật sự
`exit` TRƯỚC khi gọi `train.py`" — verify bằng cách kiểm tra trực tiếp filesystem (không
có checkpoint mới nào được tạo ra), không chỉ tin exit code khác 0 (exit code khác 0
CŨNG có thể xảy ra nếu `train.py` được gọi rồi crash — đã loại trừ khả năng này bằng
bằng chứng filesystem trực tiếp).

Cũng đọc lại nguyên văn cell 21 (`kaggle_round1_baseline.ipynb`) — xác nhận câu chủ đề
và đoạn "Quy trình đầy đủ cho MỖI scene" bên dưới **nhất quán với nhau** (cả 2 đều nói
`MODE="holdout"` là input Vòng 2+, `MODE="final"` không dùng làm input Vòng 2+) — đọc
như 1 người dùng mới đọc từ đầu tới cuối cell, không thấy chỗ nào tự mâu thuẫn nữa.

### Phần C — Đọc toàn bộ 4 notebook như 1 contributor mới (không diff lịch sử)

Dump nguyên văn tất cả cell (`kaggle_round1_baseline.ipynb` 22 cell,
`kaggle_round2_refine.ipynb`/`kaggle_round3_refine.ipynb` 29 cell mỗi file,
`kaggle_submission.ipynb` 21 cell) ra file, đọc tuần tự từ đầu tới cuối như người mới lần
đầu thấy repo (không tra cứu milestone log để biết "chỗ nào từng bị sửa"). Không tìm
thấy đoạn nào đọc lên **tự mâu thuẫn, trùng lặp gây khó hiểu, hoặc còn sót ngôn từ cũ**:

- 3 notebook train (`round1`/`round2`/`round3`) dùng chung gần như nguyên văn các cell
  hạ tầng (GPU-check, clone+build, lấy code, tải dataset, symlink) — đã diff cell-by-cell
  `round2` vs `round3`: MỌI khác biệt đều là thay số vòng hợp lý (`"Vòng 2"` ->
  `"Vòng 3"`, `eval_metrics_BEFORE_round2.csv` -> `..._round3.csv`, checkpoint nguồn
  `"Vòng 1"` -> `"Vòng 2"`) — không có chỗ nào quên đổi hoặc đổi sai.
- `kaggle_round1_baseline.ipynb` cell 21 (đã sửa ở pass #8): đọc lại xác nhận nhất quán,
  không còn đoạn nào lặp lại hướng dẫn cũ sai.
- `kaggle_submission.ipynb`: không có bất kỳ tham chiếu `train_mode` nào (đúng thiết kế —
  notebook này chấp nhận checkpoint bất kỳ vòng/MODE nào, người dùng tự chịu trách nhiệm
  qua field `"round"` chỉ để ghi chú) — xác nhận lại đúng kết luận "không over-correction"
  của pass #8.
- Không tìm thấy cell/đoạn văn nào là tàn dư từ round trước (13 scene, `HCM0249`,
  `public_set`/`private_set1`...) — đã dọn sạch từ pass #1, vẫn sạch.

**Kết luận Phần C: 4 notebook đọc mạch lạc, không có "cruft" tích luỹ qua 8 pass vá.**

### Phần D — `docs/00_MASTER_PLAN.md`: còn là "nguồn sự thật duy nhất" đúng nghĩa không?

Đọc lại toàn bộ file, đối chiếu từng câu với code/notebook hiện tại (không chỉ tin đã
đúng từ pass trước):

- Mục 3.2 (train_mode gating): khớp đúng code hiện tại (`02_train_baseline.sh` ghi
  `train_mode`, `06_train_refine.sh` + 2 notebook refine đều chặn `"final"`) — mô tả cơ
  chế Round 2/3 vẫn đúng thực tế sau khi `train_mode` gating được thêm ở pass #7/#8,
  KHÔNG bị lạc hậu.
- Mục 1 (đề bài): deadline, công thức Score, giới hạn nộp — đối chiếu lại với
  `04_eval_metrics.py::compute_score()`/`07_package_submission.py` — vẫn khớp (không có
  gì đổi từ pass #5 fact-check trước).

**Tìm thấy 1 lỗi thật — sai tham chiếu chéo nội bộ (xem "Bug tìm ra + đã sửa" dưới).**

### Phần E — `docs/PORTED_KNOWLEDGE.md`: có chỗ nào mục SAU đá mục TRƯỚC mà không ghi chú?

Đọc lại toàn bộ 1 lượt liền mạch, tập trung tìm chỗ 1 mục sau thay đổi quyết định của 1
mục trước mà không note rõ:

- Mục 6e (pass #5) kết luận "KHÔNG pin `gdown`" (bản mới nhất vẫn chạy đúng sau khi bỏ
  `--fuzzy`, pin bản cũ tạo rủi ro mới) — mục 6f (pass #6, NGAY SAU 6e) lại pin thành
  `"gdown>=6,<7"`. Đã đọc kỹ 2 đoạn: đây **KHÔNG phải mâu thuẫn ẩn** — `>=6,<7` không
  phải "pin bản cũ" (vẫn nhận mọi bản vá 6.x mới, chỉ chặn nhảy sang 7.x giả định trong
  tương lai có thể đổi API tiếp) nên vẫn đúng tinh thần lý do 6e đưa ra, và 6f nằm NGAY
  sau 6e trong thứ tự đọc tuần tự (không phải cách nhau nhiều mục khiến người đọc dễ bỏ
  lỡ) — không cần sửa gì, chỉ xác nhận đã kiểm tra kỹ theo đúng yêu cầu nhiệm vụ.
- Mục 1 (bug scale COLMAP) đã được mục 6e tự ghi chú là "kiến thức ngủ đông, không có
  code nào trong repo này cần áp dụng" — đúng mẫu hình "mục sau ghi chú lại mục trước"
  mà nhiệm vụ yêu cầu kiểm tra, đã có sẵn, không cần thêm.
- **Nhận xét cấu trúc (không sửa, vì nhiệm vụ yêu cầu giữ nguyên quy ước append-only,
  không tổ chức lại file)**: các mục "6b"–"6h" (bug MỚI tìm theo từng pass) nằm VẬT LÝ
  TRƯỚC mục "6. Triết lý test" (mục gốc, có từ bản đầu tiên của file, viết trước khi có
  bug nào được thêm) — không có mục "6a" nào. Một người đọc lần đầu quét heading theo thứ
  tự sẽ thấy 1,2,3,4,5,6b,6c,...,6h,6 — hơi gây khó hiểu (mục "6" xuất hiện SAU "6h" dù
  số nhỏ hơn, không có "6a"). Đây là hệ quả tự nhiên của quy ước "chỉ APPEND, không đổi
  số mục cũ để khỏi phải sửa lại mọi chỗ đã trích dẫn 'mục 6g'" — đã cân nhắc đổi số mục
  cuối thành "7" cho gọn nhưng **quyết định KHÔNG sửa** vì nhiệm vụ yêu cầu tường minh
  "không tổ chức lại file" — ghi nhận ở đây để pass sau biết đã cân nhắc, không phải bỏ
  sót.

### Phần F — LẦN ĐẦU chạy `00_make_holdout_split.py`/`01_run_colmap.py` bằng pycolmap thật + dữ liệu COLMAP thật (không mock)

Hạng mục này bị liệt là "còn thiếu" liên tục từ `MILESTONE_04` (pass #1) tới
`MILESTONE_11` (pass #8) — luôn chỉ audit bằng đọc code vì máy dev "chưa cài `pycolmap`".
Thử cài thật: `pip install pycolmap` (không venv nặng, chỉ `--target` 1 thư mục scratch)
— **cài thành công trong vài giây, ra bản `4.1.1`** (thoả `>=3.10`). Từ đây dựng được 1
bài test THẬT chưa ai làm:

1. Dùng `pycolmap.synthesize_dataset()` (API tổng hợp reconstruction có sẵn ngay trong
   `pycolmap`, sinh cameras/images/points3D với pose/intrinsics THẬT theo mô hình COLMAP
   chuẩn) tạo 1 scene giả `chair` — 24 ảnh, 1 camera, 200 điểm 3D — ghi ra
   `train/sparse/0/*.bin` bằng `rec.write()` (API pycolmap thật, không tự tay build
   struct nhị phân), sinh 24 file ảnh PNG giả (nội dung không quan trọng, chỉ cần tồn
   tại đúng tên) khớp tên trong reconstruction.
2. Set `BTS_DATASET_ROOT` trỏ vào scene giả, chạy THẬT (không mock)
   `python pipeline/scripts/00_make_holdout_split.py --scene chair` — **chạy thành công**,
   sinh `holdout_poses.csv` (3/24 ảnh holdout, đúng tỉ lệ 12.5% mặc định).
3. **Verify độc lập thứ tự quaternion** (đúng bug đã cảnh báo ở `PORTED_KNOWLEDGE.md`
   mục 1 — `pycolmap` trả `[x,y,z,w]`, CSV cần `qw,qx,qy,qz`): lấy trực tiếp
   `image.cam_from_world().rotation.quat` của 1 ảnh holdout từ chính đối tượng
   `Reconstruction` gốc (KHÔNG qua code của repo), so với dòng tương ứng trong
   `holdout_poses.csv` do `00_make_holdout_split.py` ghi ra — khớp CHÍNH XÁC theo đúng
   phép đảo `[x,y,z,w] -> (w,x,y,z)` (vd ảnh `camera000001_frame000023.png`: quat gốc
   `[0.32272617, -0.62647855, 0.0, 0.70948745]` -> CSV ghi
   `qw=0.70948745, qx=0.32272617, qy=-0.62647855, qz=0.0` — đúng). Đây là lần đầu fix
   quaternion-order được verify bằng **dữ liệu `pycolmap.Reconstruction` thật** (8 pass
   trước chỉ đọc code + tin lại kết quả đã verify ở repo tiền nhiệm).
4. Chạy tiếp THẬT (không mock) `python pipeline/scripts/01_run_colmap.py --scene chair
   --holdout` — script tự phát hiện `has_valid_provided_sparse()=True`, gọi
   `use_provided_sparse()` (dùng `pycolmap.undistort_images()`, KHÔNG cần binary `colmap`
   CLI riêng — xác nhận đúng docstring `colmap_runner.py` "chạy hoàn toàn bằng
   pycolmap") — **chạy xong sạch, exit 0**: 21/21 ảnh train (loại đúng 3 ảnh holdout)
   được undistort vào `colmap/dense/images/`, `colmap/dense/sparse/0/` sinh đúng, in
   đúng `[LƯU Ý]` liệt kê 3 ảnh bị loại (khớp `_find_missing_images()`/
   `deregister_frame()` mô tả trong docstring).
5. **Câu hỏi phụ nảy sinh trong lúc test (không có trong checklist trước đó, tự đặt ra
   khi thấy `pycolmap` 4.1.1 ghi thêm `rigs.bin`/`frames.bin` — file KHÔNG có trong định
   dạng COLMAP cổ điển 3 file mà `graphdeco-inria/gaussian-splatting` mong đợi)**: liệu
   `pycolmap.Reconstruction()` có ĐỌC ĐƯỢC 1 thư mục sparse CHỈ có 3 file cổ điển
   (`cameras.bin`/`images.bin`/`points3D.bin`, không có `rigs.bin`/`frames.bin` — đúng
   định dạng nhiều khả năng BTC dùng, vì `has_valid_provided_sparse()` trong
   `scenes.py` chỉ kiểm tra sự tồn tại của `cameras.bin`) hay không? Verify bằng thực thi
   thật: xoá `rigs.bin`/`frames.bin` khỏi 1 bản copy, `pycolmap.Reconstruction()` đọc lại
   — **vẫn đọc đúng đủ 24 ảnh/1 camera/200 điểm**, không lỗi (pycolmap tự suy ra rig
   ngầm định cho từng camera đơn khi thiếu file rig). Xác nhận **không có rủi ro tương
   thích ngược** giữa `pycolmap` bản mới và sparse định dạng cổ điển mà BTC nhiều khả
   năng cung cấp.

**Kết quả Phần F: KHÔNG tìm thấy bug nào trong `00_make_holdout_split.py`/
`01_run_colmap.py`/`colmap_runner.py`** — lần đầu tiên cả 2 script này được chạy thật
(không mock) với `pycolmap` thật + dữ liệu COLMAP thật, xác nhận đúng mọi tuyên bố đã ghi
trong `PORTED_KNOWLEDGE.md` mục 1 (thứ tự quaternion) và docstring `colmap_runner.py`
(undistort thuần `pycolmap`, tự loại ảnh thiếu file, tương thích sparse cổ điển). Dọn
sạch toàn bộ fixture/scratch sau khi test (`git status` xác nhận sạch trong suốt quá
trình, không rác nào lọt vào `pipeline/work/` của repo thật).

## Bug tìm ra + đã sửa

### Bug — `docs/00_MASTER_PLAN.md` mục 3.1 trỏ SAI tới "mục 6" của `PORTED_KNOWLEDGE.md` (phải là "mục 2")

**Mô tả:** Dòng "Cấu hình mặc định (đã đo thật ở repo cũ, xem mục 6 `PORTED_KNOWLEDGE.md`):
`--antialiasing` BẬT, không depth-prior, không antenna-focus, không exposure-comp — các
cờ đó đã đo KHÔNG cải thiện Score đo được trên holdout thật (HCM0421)..." trỏ tới **"mục
6"** của `PORTED_KNOWLEDGE.md` — nhưng mục 6 của file đó là **"Triết lý test"** (quy tắc
kiểm thử chung, không liên quan gì tới kết quả đo antialiasing/depth-prior/antenna-focus).
Nội dung THẬT sự chứa các kết quả đo này (Score `HCM0421` 0.644 vs 0.6616 cho
depth-prior, 0.6611 vs 0.6616 cho antenna-focus, lý do giữ `--antialiasing`...) nằm ở
**mục 2 ("Train")** của `PORTED_KNOWLEDGE.md`.

**Xác nhận đây là lỗi có từ ĐẦU, không phải do 1 pass sau làm lệch:** `git log -p
--follow -- docs/00_MASTER_PLAN.md` cho thấy dòng này xuất hiện y nguyên từ đúng commit
đầu tiên tạo ra file (`920519f`, "Khởi tạo repo") và **KHÔNG hề bị đổi bởi bất kỳ commit
nào sau đó** (8 pass verify trước đều đọc `00_MASTER_PLAN.md` — theo đúng yêu cầu STEP 0
của mỗi pass — nhưng không ai bắt được vì đọc trôi qua như 1 câu dẫn nguồn bình thường,
không có ai bấm vào kiểm tra đúng "mục 6" có thật sự chứa nội dung được nhắc tới hay
không).

**Tác động:** Thấp về mặt kỹ thuật (không ảnh hưởng code/hành vi pipeline nào — thuần
tham chiếu tài liệu), nhưng VI PHẠM đúng lời hứa ở đầu chính file này ("`docs/
00_MASTER_PLAN.md` là nguồn sự thật duy nhất... đọc file này TRƯỚC khi đọc bất kỳ
milestone log nào khác") — 1 người/agent mới đọc theo đúng chỉ dẫn, muốn xác minh lại
tuyên bố "`--antialiasing` BẬT... đã đo KHÔNG cải thiện" bằng cách mở đúng "mục 6" của
`PORTED_KNOWLEDGE.md` như được trỏ tới, sẽ thấy "Triết lý test" — hoàn toàn không liên
quan, có thể khiến họ nghi ngờ nhầm tuyên bố này chưa được kiểm chứng thật (trong khi
thực tế ĐÃ được đo thật, chỉ là trỏ sai mục).

**Đã sửa:** đổi `xem mục 6` → `xem mục 2` tại `docs/00_MASTER_PLAN.md` (dòng duy nhất
trỏ sai — đã grep toàn repo xác nhận không còn tham chiếu "mục 6" trần nào khác ngoài 1
dòng comment ĐÚNG trong `02_train_baseline.sh` (nói về `|| true` sau `grep`, đúng thuộc
mục 6 "Triết lý test" thật) và các tham chiếu "mục 6g"/"mục 6h" khác không liên quan.

**Verify sau sửa:** đọc lại `docs/PORTED_KNOWLEDGE.md` mục 2 xác nhận đúng nội dung
antialiasing/depth-prior/antenna-focus được nhắc tới khớp 100% với câu ở
`00_MASTER_PLAN.md`. `git diff` xác nhận CHỈ đổi đúng 1 từ ("6" -> "2"), không đụng gì
khác. Test suite chạy lại (`test_syntax_all.py` 15/15+4/4, `test_07_package_submission.py`
21/21) vẫn PASS 100% (thay đổi thuần văn bản `.md`, không ảnh hưởng code — chạy lại chỉ
để chắc chắn theo đúng thông lệ mọi pass trước).

## Giới hạn của pass này (ghi rõ, không giấu)

- Vẫn CHƯA chạy được train/render 3DGS thật trên GPU Kaggle — giới hạn không đổi qua mọi
  pass, chỉ giải quyết được khi có 1 lần chạy Kaggle thật.
- Phần F chỉ verify được `00_make_holdout_split.py`/`01_run_colmap.py` (2 script duy
  nhất KHÔNG cần CUDA/rasterizer, chỉ cần `pycolmap` thuần) — `03_render_test_poses.py`/
  `04_eval_metrics.py`/`05_generate_error_mask.py`/`apply_error_refine_patch.py`/train.py
  đã vá vẫn chỉ verify được qua mock rasterizer (như pass #1/#4 đã làm trước đây), không
  đổi so với trước — cần GPU CUDA thật, không có ở máy dev.
- Dữ liệu tổng hợp qua `pycolmap.synthesize_dataset()` là scene ĐƠN GIẢN (1 camera, 24
  ảnh, quỹ đạo tổng hợp) — không đại diện đầy đủ độ phức tạp của 7 scene thật (nhiều
  camera SIMPLE_RADIAL, méo ống kính, quỹ đạo bay drone thật, số điểm 3D lớn hơn nhiều)
  — vẫn là bước tiến so với "chỉ đọc code" nhưng chưa thay thế hoàn toàn 1 lần chạy với
  chính dataset thật của cuộc thi.
- Không tìm thêm phát hiện nào khác ngoài lỗi tham chiếu "mục 6" nêu trên — các khu vực
  đã bị soi rất kỹ nhiều lần qua 8 pass trước (schema `pipeline_train_flags.json`, va
  chạm thư mục iteration, đường dẫn dấu cách, `spatial_lr_scale`, tên file
  `eval_metrics`, `git submodule update`, checkpoint sort bug, symlink dataset,
  GPU-check, rollback/stop-at-round-N, adversarial input, re-run stale checkpoint,
  `gdown --fuzzy`, thiếu `opencv-python`, mâu thuẫn MODE Vòng 1/Vòng 2+, guard
  `06_train_refine.sh`) đều được re-verify (Phần B) và xác nhận vẫn đúng, không có gì
  mới.

## Bước tiếp theo

1. Chạy **pass #10** (agent verify độc lập khác) — bộ đếm sạch liên tiếp: 0/3.
2. Gợi ý cho pass #10: hạng mục "chưa test được cục bộ" LỚN NHẤT còn lại đã thu hẹp đáng
   kể sau pass này — chỉ còn phần cần CUDA thật (train.py đã vá chạy thật, rasterizer
   thật, render/Score chất lượng thật) — không có cách nào verify cục bộ tiếp mà không
   có GPU CUDA + toolchain đầy đủ. Có thể thử mở rộng Phần F của pass này: dùng
   `pycolmap.synthesize_dataset()` với cấu hình PHỨC TẠP hơn (nhiều camera, camera model
   SIMPLE_RADIAL có tham số méo khác 0, quỹ đạo lớn hơn) để tăng độ phủ so với scene đơn
   giản ở pass này — hoặc thử tải fixture COLMAP text thật nhỏ (nếu tìm được nguồn public
   phù hợp, KHÔNG được lấy từ dataset cuộc thi — vi phạm mục 10.1 đề bài) để so sánh.
3. Các khu vực đã soi rất kỹ nhiều lần (liệt kê ở mục "Giới hạn" trên) nên tiếp tục giảm
   ưu tiên trừ khi có thay đổi code liên quan.

## Lịch sử

### 2026-07-18 — Verification pass #9 (agent kiểm tra độc lập, khác pass #1-8)
- Đọc đầy đủ STEP 0 (`00_MASTER_PLAN.md`, `PORTED_KNOWLEDGE.md` toàn bộ kể cả mục
  6b-6h, 12 milestone log 00-11, `git log --oneline`, `git show 1ff9341`).
- Phần A: test suite baseline PASS 100% (`py_compile`/`bash -n`/`nbformat.validate()`/2
  bộ test).
- Phần B: dựng mock MỚI, thực thi thật 4 kịch bản train_mode guard của
  `06_train_refine.sh` (final-chặn/final+override/vắng mặt-cảnh báo/holdout-im lặng) —
  xác nhận đúng thiết kế + xác nhận nhánh chặn thật sự KHÔNG gọi `train.py` (bằng
  filesystem, không chỉ exit code); đọc lại cell 21 `kaggle_round1_baseline.ipynb` xác
  nhận nhất quán.
- Phần C: đọc toàn bộ 4 notebook (101 cell) như contributor mới — không tìm thấy cruft/
  mâu thuẫn/trùng lặp nào; diff cell-by-cell round2 vs round3 xác nhận mọi khác biệt hợp
  lý.
- Phần D: đối chiếu `00_MASTER_PLAN.md` với code hiện tại — mục 3.2 (train_mode gating)
  vẫn khớp thực tế; **phát hiện lỗi tham chiếu "mục 6" sai** ở mục 3.1.
- Phần E: đọc `PORTED_KNOWLEDGE.md` tìm chỗ mục sau đá mục trước không ghi chú — không
  tìm thấy mâu thuẫn ẩn thật sự (gdown pin 6e/6f đã tự nhất quán); ghi nhận (không sửa)
  1 điểm cấu trúc đánh số 6b-6h/6 hơi khó theo dõi, theo đúng yêu cầu giữ nguyên quy ước
  append-only của nhiệm vụ.
- Phần F: cài `pycolmap` thật (4.1.1) lần đầu ở 1 pass verify, dùng
  `pycolmap.synthesize_dataset()` dựng scene giả thật, chạy THẬT (không mock)
  `00_make_holdout_split.py` + `01_run_colmap.py` — xác nhận thứ tự quaternion đúng
  (verify độc lập bằng dữ liệu `Reconstruction` gốc), undistort + loại ảnh holdout đúng,
  và xác nhận `pycolmap` đọc được sparse cổ điển (thiếu `rigs.bin`/`frames.bin`) không
  lỗi — không tìm thấy bug.
- Sửa: `docs/00_MASTER_PLAN.md` (1 từ, "mục 6" -> "mục 2").
- Verify sau sửa: `git diff` xác nhận đúng 1 từ đổi; test suite PASS 100% không
  regression.
- Dọn sạch toàn bộ scratch/fixture (`git status` sạch trong suốt quá trình, kể cả sau
  khi chạy script thật ghi ra `pipeline/work/chair/` — đã xoá lại sau khi verify xong).
- Cập nhật `docs/PORTED_KNOWLEDGE.md` mục 6i (bug tìm được + xác nhận pycolmap thật lần
  đầu). Ghi milestone log này.
