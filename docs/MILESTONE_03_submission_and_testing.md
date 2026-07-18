# Milestone 03 — Đóng gói submission + bộ test cục bộ toàn repo

## Trạng thái hiện tại

**HOÀN TẤT phần việc của agent này** (port packaging + notebook submission + test cục
bộ cho toàn bộ code có thể test không cần GPU). **CÒN THIẾU test tích hợp thật trên
Kaggle** (bắt buộc phải chạy GPU, không mock được — xem mục "Bước tiếp theo").

Đã tạo/port:
- `pipeline/scripts/07_package_submission.py` — port nguyên vẹn logic từ
  `06_package_submission.py` của repo tiền nhiệm (`BTS-Digital-Twin`, đã kiểm chứng
  thật), chỉ đổi số thứ tự script (07, theo phân công `docs/00_MASTER_PLAN.md` mục 5
  — 00-04 thuộc agent Round-1, 05-06 thuộc agent refine, 07 là script cuối pipeline).
  Giữ nguyên toàn bộ cơ chế đã kiểm chứng: `--filename_mode literal` (mặc định, giữ
  đúng tên/đuôi `image_name` trong `test_poses.csv`), mã hoá lại nội dung đúng định
  dạng theo đuôi thật (`encode_for_arcname()`), `check_scene()` + `verify_zip()` kiểm
  tra đủ scene/ảnh/đúng kích thước TRƯỚC và SAU khi nén, `--check_only` để kiểm tra
  lại 1 zip có sẵn không cần nén lại.
- `pipeline/kaggle_submission.ipynb` — port + cập nhật từ `kaggle_submission.ipynb`
  của repo tiền nhiệm (đã đọc bản MỚI NHẤT, có fix bug tải thiếu `cfg_args`/
  `pipeline_train_flags.json`, KHÔNG phải bản cũ có bug). Thay đổi so bản gốc:
  - `REPO_URL` trỏ sang `ThongLuc2k3/BTS-Digital-Twin-MultiRound`, thêm biến
    `GIT_BRANCH` (mặc định `"main"`) — nhất quán với cách `kaggle_round1_baseline.ipynb`
    của agent Round-1 đã làm (đối chiếu thật, xem mục "Đối chiếu chéo" bên dưới).
  - Gọi `03_render_test_poses.py` (không phải `04_render_test_poses.py` như repo
    tiền nhiệm — đổi số theo cách đánh số mới của repo này) và
    `07_package_submission.py` (không phải `06_...`).
  - `CHECKPOINT_LINKS` đổi cấu trúc từ `{scene: link}` sang
    `{scene: {"round": N, "link": ...}}` — field `"round"` CHỈ để ghi chú/theo dõi
    (vòng nào đã được CHỌN cho scene đó, in ra log lúc tải) — không ảnh hưởng logic,
    vì việc chọn vòng 1/2/3 cho mỗi scene là quyết định của con người (so Score
    holdout giữa các vòng), không phải thứ notebook này tự quyết định (đúng yêu cầu
    trong đề bài giao việc).
  - Thêm cell kiểm tra sớm (ngay sau bước symlink code) rằng
    `pipeline/scripts/03_render_test_poses.py` đã tồn tại trong bản clone — báo
    CẢNH BÁO rõ ràng (không phải lỗi mù mờ ở Bước 6) nếu agent Round-1 chưa đẩy file
    này lên nhánh đang dùng.
  - Thêm bước tự kiểm tra sau mỗi lần gọi `03_render_test_poses.py` (Bước 6): xác
    nhận thư mục `renders/` thực sự có ảnh PNG trước khi sang scene kế — vì `!lệnh`
    thất bại KHÔNG tự dừng notebook (xem `docs/PORTED_KNOWLEDGE.md` mục 4).
  - `git checkout 54c035f7834b564019656c3e3fcc3646292f727d` thêm vào Bước 2 (clone
    gaussian-splatting) để PIN đúng commit đã xác nhận có `--antialiasing` — bản gốc
    của repo tiền nhiệm chỉ `git clone --recursive` không pin commit tường minh
    trong notebook (chỉ ghi trong comment); thêm dòng lệnh tường minh cho chắc, nhất
    quán với `02_train_baseline.sh` của agent Round-1 (đã thấy pin commit trong
    docstring script đó).
- `tests/` (thư mục mới, theo yêu cầu bài giao việc):
  - `tests/README.md` — hướng dẫn chạy, danh sách test hiện có, danh sách test còn
    thiếu (đồng bộ với mục "Bước tiếp theo" ở file này).
  - `tests/test_syntax_all.py` — quét TOÀN repo (không chỉ file của agent này):
    `py_compile` mọi `.py`, `nbformat.validate()` mọi `.ipynb`. Tự bỏ qua (không
    FAIL) nếu số file ít hơn kỳ vọng cuối cùng — để không chặn lẫn nhau giữa 3 agent
    chạy song song.
  - `tests/test_07_package_submission.py` — test riêng cho 2 file agent này sở hữu:
    unit test thuần Python (`target_filename()`, `check_scene()`), **test regression
    trực tiếp cho đúng bug thật đã tìm+sửa ở repo tiền nhiệm** (đổi tên `.jpg` mà giữ
    nguyên byte PNG — dựng 1 zip giả đúng kiểu bug cũ, xác nhận `verify_zip()` của
    chính script PHẢI raise lỗi), test end-to-end mock (dataset giả 7 scene + renders
    giả qua `BTS_DATASET_ROOT`/`--renders_root`, chạy script thật qua subprocess,
    kiểm tra `.jpg`/`.JPG`/`.png` trong zip output đều có nội dung ĐÚNG định dạng
    thật — không phải đổi tên suông), và test nhánh lỗi (thiếu ảnh, sai kích thước,
    `--check_only` trên zip hỏng).

**Kết quả chạy test lúc viết file này**: `tests/test_syntax_all.py` — 15/15 file
`.py` + 2/2 file `.ipynb` PASS (bao gồm cả file của agent Round-1 và agent refine đã
đẩy lên tới thời điểm này). `tests/test_07_package_submission.py` — 21/21 PASS.

## Đối chiếu chéo với file của 2 agent kia (đã làm được tới đâu)

Tại thời điểm agent này hoàn thành, agent Round-1 và agent refine ĐÃ đẩy các file sau
lên (không còn trống như lúc agent này bắt đầu đọc `docs/00_MASTER_PLAN.md`):
`pipeline/scripts/01_run_colmap.py`, `02_train_baseline.sh`, `03_render_test_poses.py`,
`04_eval_metrics.py`, `05_generate_error_mask.py`, `apply_error_refine_patch.py`,
`pipeline/kaggle_round1_baseline.ipynb`. Đã tranh thủ đọc + đối chiếu (KHÔNG sửa file
của 2 agent kia, chỉ đọc để kiểm tra tương thích):

- **`03_render_test_poses.py`** (agent Round-1): CLI đúng như giả định lúc viết
  `kaggle_submission.ipynb` — `--scene`/`--model_dir`/`--iteration`/`--out_dir`/
  `--poses_csv`/`--sh_degree`/`--antialiasing {auto,on,off}`. Output
  `pipeline/work/<scene>/renders/<stem>.png` — khớp đúng chỗ
  `07_package_submission.py` đọc vào (`renders_root/<scene>/renders/`). **KHỚP, không
  cần sửa gì.**
- **`pipeline_train_flags.json`** (do `02_train_baseline.sh` ghi, `03_render_test_poses.py`
  đọc): schema thật `{"antialiasing": bool, "depth_prior": bool, "exposure_comp": bool,
  "antenna_focus": bool}` (4 key luôn có mặt, kể cả 3 key mà Vòng 1 luôn ghi `false`
  vì không hỗ trợ). `07_package_submission.py`/`kaggle_submission.ipynb` của agent
  này KHÔNG tự đọc trực tiếp file này (chỉ assert nó TỒN TẠI lúc tải checkpoint từ
  Drive ở Bước 5 — việc đọc/diễn giải nội dung là việc của `03_render_test_poses.py`,
  đã đúng theo `docs/PORTED_KNOWLEDGE.md` mục 2). **KHÔNG có xung đột schema.**
- **`kaggle_round1_baseline.ipynb`** (agent Round-1): dùng đúng cùng `REPO_URL`
  (`ThongLuc2k3/BTS-Digital-Twin-MultiRound`), `GIT_BRANCH="main"`, `GDRIVE_URL`
  (`.../178EL7jCSVD59q19SMpeOgnOfOIC66I_t/...`) y hệt bản agent này đã dùng ở
  `kaggle_submission.ipynb` — copy đúng theo hướng dẫn ở `docs/PORTED_KNOWLEDGE.md`
  mục 1. Cách dò thư mục pipeline/dataset trong zip cũng viết y hệt logic (đối chiếu
  từng dòng, khớp 100%). **NHẤT QUÁN, không cần sửa.**
- **`apply_error_refine_patch.py`** (agent refine): chỉ vá `train.py`, không tự ghi
  `pipeline_train_flags.json` — file ghi flags này (thuộc về `06_train_refine.sh`,
  theo phân công ở `docs/00_MASTER_PLAN.md`) **CHƯA tồn tại** tại thời điểm agent này
  hoàn thành (xem mục "Bước tiếp theo" — đây là phần CHƯA kiểm chứng được).

## Bước tiếp theo — checklist tích hợp còn lại (cho coordinator hoặc agent verify sau)

Liệt kê tường minh để không ai phải đoán lại từ đầu khi merge 3 nhánh:

1. **Chờ `06_train_refine.sh` (agent refine) tồn tại**, rồi kiểm tra:
   - Nó có ghi lại `pipeline_train_flags.json` sau mỗi vòng refine không, và có giữ
     ĐÚNG schema 4-key như `02_train_baseline.sh` đã ghi (`antialiasing`/
     `depth_prior`/`exposure_comp`/`antenna_focus`) không — nếu đổi schema (thêm/bớt
     key, đổi tên key) mà không cập nhật `read_pipeline_train_flags()` trong
     `03_render_test_poses.py`, checkpoint vòng 2/3 sẽ bị đọc sai flags khi render
     cho submission. **Việc sửa (nếu có xung đột) thuộc phạm vi file của agent Round-1
     (`03_render_test_poses.py`) hoặc agent refine (`06_train_refine.sh`), KHÔNG phải
     file của agent này — chỉ báo lại, không tự sửa.**
   - Checkpoint sau refine có vẫn nằm ở cấu trúc `gs_model/point_cloud/iteration_<N>/
     point_cloud.ply` + `cfg_args` + `pipeline_train_flags.json` cùng cấp
     `gs_model/` không (đúng cấu trúc mà Bước 5 của `kaggle_submission.ipynb` giả
     định khi assert sau khi tải từ Drive) — nếu refine đổi cấu trúc thư mục output,
     `kaggle_submission.ipynb` cần cập nhật lại đường dò `candidates = [p.parent for
     p in raw_dl_dir.rglob("cfg_args")]`.
2. **Chờ `kaggle_round2_refine.ipynb`/`kaggle_round3_refine.ipynb`** (agent refine) —
   kiểm tra chúng có nhắc đúng hướng dẫn "tải NGUYÊN thư mục `gs_model/` lên Drive sau
   khi refine xong" giống `kaggle_round1_baseline.ipynb` đã làm không (đây chính là
   input mà `kaggle_submission.ipynb` Bước 5 kỳ vọng — thiếu hướng dẫn này ở notebook
   refine có thể khiến người dùng lại tải thiếu file, tái lặp bug đã sửa).
3. **Test tích hợp CHẠY THẬT trên Kaggle (bắt buộc GPU, chưa làm được — không có GPU
   cục bộ)**:
   - Chạy hết `kaggle_round1_baseline.ipynb` (`MODE="final"`) cho ít nhất 1 scene nhỏ
     (khuyến nghị `chair` hoặc `bonsai`, nhẹ nhất) → tải `gs_model/` lên Drive → dán
     link vào `CHECKPOINT_LINKS` của `kaggle_submission.ipynb` → chạy hết
     `kaggle_submission.ipynb` → xác nhận `submission.zip` sinh ra đúng, mở thử vài
     ảnh bằng mắt.
   - Lặp lại đường vòng 2 (`kaggle_round2_refine.ipynb` → checkpoint mới → dán lại
     link Drive khác vào `CHECKPOINT_LINKS` cho ĐÚNG SCENE đó → chạy lại
     `kaggle_submission.ipynb`) để xác nhận cơ chế "mỗi scene 1 vòng khác nhau" hoạt
     động đúng khi trộn lẫn (vd `chair` dùng checkpoint vòng 2, các scene khác vẫn
     vòng 1) — đây là kịch bản THẬT sẽ xảy ra lúc nộp bài thật, chưa test được vì
     `kaggle_round2_refine.ipynb` chưa tồn tại lúc viết milestone này.
   - Đo dung lượng `submission.zip` thật (7 scene đầy đủ, không phải mock nhỏ) để xác
     nhận lại ngưỡng `--jpeg_quality 98` vẫn an toàn dưới 350MB với dataset/độ phân
     giải THẬT của round này (mock test chỉ dùng ảnh giả rất nhỏ, không đại diện cho
     dung lượng thật).
4. **Chưa test được**: `apply_error_refine_patch.py` áp thật lên 1 bản clone
   `gaussian-splatting` sạch (nằm ngoài phạm vi file agent này sở hữu — nên KHÔNG tự
   viết test cho file này, chỉ ghi chú lại đây để coordinator biết đã kiểm tra hay
   chưa; xem milestone log của agent refine, nếu có, để biết agent đó đã tự test
   chưa).
5. Sau khi cả 3 nhánh merge vào `main`, chạy lại `tests/test_syntax_all.py` — kỳ vọng
   lúc đó phải thấy ĐỦ các file: `06_train_refine.sh`, `kaggle_round2_refine.ipynb`,
   `kaggle_round3_refine.ipynb` (hiện chưa thấy — script tự in "LƯU Ý" chứ không
   FAIL khi thiếu, để không chặn agent khác, nhưng coordinator cần tự đối chiếu số
   file quét được với danh sách kỳ vọng đầy đủ ở `docs/00_MASTER_PLAN.md` mục 5).

## Lịch sử

### 2026-07-18 — Đóng gói submission + bộ test cục bộ

- Đọc `docs/00_MASTER_PLAN.md`, `docs/PORTED_KNOWLEDGE.md` (đặc biệt mục 4 cạm bẫy
  Jupyter/Kaggle + mục 5 quy tắc đóng gói submission), skim `pipeline/common/scenes.py`.
- Đọc file NGUỒN thật (không suy đoán) ở repo tiền nhiệm: `pipeline/kaggle_submission.ipynb`
  (bản đã fix bug tải thiếu `cfg_args`/`pipeline_train_flags.json`, xác nhận qua nội
  dung cell thật, không phải bản cũ), `pipeline/scripts/06_package_submission.py`,
  `pipeline/scripts/04_render_test_poses.py` (để hiểu CLI contract mà
  `03_render_test_poses.py` của repo này phải tương thích).
- Port `pipeline/scripts/07_package_submission.py` — giữ nguyên logic đã kiểm chứng,
  chỉ đổi số thứ tự + cập nhật comment cho khớp bối cảnh multi-round.
- Viết mới `pipeline/kaggle_submission.ipynb` (dùng `nbformat` build trực tiếp, không
  copy tay JSON) — cấu trúc `CHECKPOINT_LINKS` hỗ trợ ghi chú vòng đã chọn/scene,
  thêm các bước tự-kiểm-tra tường minh (script tồn tại, renders/ có ảnh) theo đúng
  cạm bẫy `!lệnh` không tự dừng notebook đã ghi ở `PORTED_KNOWLEDGE.md` mục 4.
- Trong lúc làm, phát hiện agent Round-1 + agent refine đã đẩy nhiều file lên
  (`01_run_colmap.py`, `02_train_baseline.sh`, `03_render_test_poses.py`,
  `04_eval_metrics.py`, `05_generate_error_mask.py`, `apply_error_refine_patch.py`,
  `kaggle_round1_baseline.ipynb`) — tranh thủ đọc + đối chiếu chéo (xem mục "Đối
  chiếu chéo" ở trên), xác nhận KHỚP, không cần sửa file của agent nào khác.
- Viết `tests/` (README + `test_syntax_all.py` + `test_07_package_submission.py`),
  chạy thật: `test_syntax_all.py` 15/15 `.py` + 2/2 `.ipynb` PASS,
  `test_07_package_submission.py` 21/21 PASS (bao gồm test regression trực tiếp cho
  bug "đổi tên .jpg giữ nguyên byte PNG" — xác nhận `verify_zip()` bắt được lỗi này).
- Ghi milestone log này, liệt kê tường minh checklist tích hợp còn lại (mục "Bước
  tiếp theo") cho coordinator/agent verify sau khi merge đủ 3 nhánh.
