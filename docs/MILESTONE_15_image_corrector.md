# Milestone 15 — Model 2: mạng sửa lỗi pixel 2D (nhánh `feature/nn-image-corrector`)

> File này chỉ áp dụng cho nhánh git `feature/nn-image-corrector` — KHÔNG nằm trên
> `main`. Đây là 1 kiến trúc THỬ NGHIỆM, tách biệt hoàn toàn với pipeline đã verify kỹ
> trên `main` (11 pass verification, xem `docs/MILESTONE_04..14_verification_passN.md`).
> `main` không bị đụng tới bởi bất kỳ file nào ở milestone này.

## Trạng thái hiện tại

**HOÀN TẤT phần code + test cục bộ (không GPU).** Chưa chạy thật trên Kaggle (cần GPU
+ dataset thật — xem "Bước tiếp theo").

## Bối cảnh — vì sao có nhánh này

Ý tưởng gốc của người dùng (khác với kiến trúc trên `main`): thay vì train tiếp CÙNG 1
bộ Gaussian 3D với loss có trọng số theo lỗi (`06_train_refine.sh`, Vòng 2/3 trên
`main`), người dùng muốn:

1. Train 3D Gaussian Splatting (**Model 1**) trên **100% ảnh train** (không giữ lại
   holdout).
2. Lấy ảnh Model 1 render tại pose train + ảnh GT thật, đưa vào **Model 2** — 1 mạng
   neural 2D **hoàn toàn riêng biệt**, học sửa/làm nét ảnh render dựa trên so sánh đó.
3. Áp Model 2 lên ảnh Model 1 render tại pose test thật, trước khi đóng gói nộp bài.
4. Muốn tinh chỉnh thêm ("lần 2"), chỉ cần chạy lại đúng file/cell đó.
5. Chấp nhận KHÔNG có Score khách quan (không holdout) — chỉ kiểm tra bằng mắt.

Do đây là thay đổi kiến trúc lớn, khác biệt rõ với nguyên tắc "luôn đo Score holdout
trước/sau" đã verify kỹ trên `main`, quyết định tách hẳn sang nhánh riêng thay vì sửa
`main` — giữ nguyên bản đã verify làm phương án nộp bài an toàn, nhánh này là phương án
thử nghiệm song song.

## Thiết kế

### File mới (strictly additive — không sửa file nào đã có trên `main`)

- `pipeline/common/corrector_model.py` — `ResidualCorrectorNet` (fully-convolutional,
  KHÔNG pooling, residual predict-and-add, tail conv khởi tạo 0 để an toàn khi
  chưa/train hỏng), SSIM khả vi thuần `torch`, `save_checkpoint()`/`load_checkpoint()`.
- `pipeline/scripts/08_build_corrector_dataset.py` — render toàn bộ pose train bằng
  checkpoint Model 1, lưu cặp `(render, GT)` ra `pipeline/work/<scene>/corrector_dataset/`
  (tự đủ, không phụ thuộc `colmap/dense/images/` còn tồn tại hay không). Cần GS_REPO +
  GPU CUDA (giống `05_generate_error_mask.py`).
- `pipeline/scripts/09_train_corrector.py` — train Model 2, thuần `torch` (KHÔNG cần
  GS_REPO/CUDA rasterizer nữa). `--steps` LUÔN là số bước THÊM của lần chạy này (không
  phải luỹ kế) — số bước thật lưu trong field `"step"` của checkpoint. Guard
  resume/overwrite giống triết lý `06_train_refine.sh`: có checkpoint sẵn mà không
  truyền `--resume`/`--overwrite` thì báo lỗi rõ, không âm thầm ghi đè.
- `pipeline/scripts/10_apply_corrector.py` — áp Model 2 lên render Model 1 (output của
  `03_render_test_poses.py`, KHÔNG sửa script đó) bằng suy luận kiểu tiled (cắt ô, chạy
  mạng từng ô, ghép lại có trộn mượt biên) để tránh OOM ảnh lớn. Ghi ra
  `pipeline/work_corrected/<scene>/renders/` — đúng layout `07_package_submission.py`
  đã hỗ trợ sẵn qua `--renders_root` (KHÔNG sửa script đó).
- `pipeline/kaggle_pixel_corrector.ipynb` — notebook Kaggle mới, dùng lại nguyên vẹn
  `kaggle_round1_baseline.ipynb` để train Model 1 (khuyến nghị `MODE="final"`), rồi mới
  chạy các bước Model 2 (8→9→10→11→12 QC bằng mắt→13 lưu Drive).
- `tests/test_corrector_pipeline.py` — 32 test cục bộ (không GPU): kiến trúc mạng, an
  toàn zero-init, checkpoint round-trip, SSIM, toán học ghép ô (partition of unity),
  VÀ chạy THẬT (subprocess, CPU) `09_train_corrector.py` với dataset giả để xác nhận cơ
  chế resume/overwrite hoạt động đúng — không chỉ đọc code suy đoán.

### Vì sao KHÔNG dùng U-Net/pooling, KHÔNG BatchNorm

Render và GT đã align pixel-for-pixel tuyệt đối (cùng pose camera) — tác vụ sửa lỗi cục
bộ (làm nét cạnh, bù chi tiết mảnh) không cần receptive field lớn/bottleneck ngữ nghĩa
mà pooling mang lại, và pooling+upsample có nguy cơ sinh checkerboard artifact — ngược
mục tiêu "ảnh nét". Giữ fully-convolutional nghĩa là CÙNG 1 trọng số áp được cho ảnh
kích thước bất kỳ (train patch cố định, suy luận ảnh test kích thước thật). BatchNorm bị
bỏ vì đi ngược mục tiêu giữ đúng giá trị màu tuyệt đối trong phục hồi ảnh, và dữ liệu
mỗi scene quá ít (vài trăm ảnh) để ước lượng batch statistics ổn định.

## Rủi ro — đọc trước khi dùng để nộp bài thật

1. **Rủi ro tổng quát hoá (lớn nhất).** Model 2 chỉ thấy pose TRAIN lúc học, không có
   nhận thức 3D/pose nào — có thể học "sửa" theo kiểu đặc thù góc nhìn train rồi áp sai
   ở pose test thật khác hẳn. KHÔNG có holdout để tự động phát hiện việc này (quyết định
   có chủ đích của người dùng) — chỉ có Bước 12 (xem bằng mắt) đứng giữa rủi ro này và
   ảnh nộp bài. Nên xem kỹ ảnh trước/sau cho MỖI scene trước khi quyết định dùng
   `work_corrected/` hay giữ nguyên render Model 1 gốc.
2. **Rủi ro học theo nhiễu/artifact nén JPEG** của ảnh GT thay vì chi tiết cảnh thật —
   nếu train quá lâu, ảnh "sau" có thể trông nhiễu/giả tạo hơn là "nét" hơn. Mitigation
   đã có trong thiết kế: ngân sách bước mặc định vừa phải (4000 lần đầu), tail zero-init
   (xuống cấp về "không sửa gì" thay vì hỏng), khuyến nghị so sánh bằng mắt tại nhiều mốc
   step khác nhau thay vì tin tuyệt đối vào loss giảm.
3. **Tuân thủ chống gian lận (đề bài mục 10):** Model 2 là 1 mạng train tự động, áp
   ĐỀU cho mọi pixel/mọi ảnh qua `10_apply_corrector.py` — không có can thiệp tay từng
   ảnh/pixel nào. Quyết định con người DUY NHẤT được phép: chọn dùng hay không dùng
   Model 2 cho MỖI SCENE (qua có/không chạy Bước 11-13 cho scene đó) — cùng loại quyết
   định "chọn vòng nào" đã chấp nhận ở `kaggle_submission.ipynb` trên `main`.
4. Ngân sách thời gian: sinh dataset (Bước 8) ~ ngang `03_render_test_poses.py` (vài
   phút/scene). Train Model 2 (~0.5M tham số, patch 256px, vài nghìn bước) là vài phút,
   không đáng kể so với train Model 1 (15-30 nghìn iteration).

## Bước tiếp theo (bắt buộc trước khi dùng để nộp bài thật)

1. **Chạy thật trên Kaggle** (chưa làm được — không có GPU cục bộ): train Model 1
   `MODE="final"` 1 scene nhẹ (chair/bonsai) → chạy hết `kaggle_pixel_corrector.ipynb`
   → xem ảnh trước/sau Bước 12 bằng mắt thật.
2. Nếu ảnh "sau" rõ ràng tốt hơn cho ít nhất vài scene: lặp lại cho 7 scene, đóng gói
   thử `submission.zip` từ `pipeline/work_corrected`, đối chiếu dung lượng/định dạng
   giống hệt kiểm tra đã làm trên `main` (`07_package_submission.py` không đổi nên logic
   kiểm tra vẫn y hệt).
3. Cân nhắc: giữ `main` (đã verify 11 pass) làm phương án nộp bài AN TOÀN mặc định; chỉ
   thay bằng `work_corrected/` cho SCENE NÀO kiểm chứng bằng mắt thấy tốt hơn thật sự.
4. KHÔNG merge nhánh này vào `main` cho tới khi có bằng chứng thị giác thật (không chỉ
   suy đoán từ code) rằng Model 2 cải thiện chất lượng — nếu không cải thiện/tệ hơn, có
   thể bỏ hẳn nhánh này mà không ảnh hưởng gì tới `main`.

## Lịch sử

### 2026-07-20 — Tạo nhánh + code + test cục bộ

- Xác nhận qua 2 câu hỏi làm rõ với người dùng: (a) "2 model riêng biệt" nghĩa là 1 mạng
  neural thứ 2 THẬT SỰ khác Model 1 (không phải chỉ 2 checkpoint của cùng Model 1), (b)
  đổi sang train Model 1 100% dữ liệu, chấp nhận không có Score khách quan.
- Research kiến trúc hiện có (`apply_error_refine_patch.py`, `04_eval_metrics.py`,
  `05_generate_error_mask.py`, notebook Kaggle) qua 2 agent (Explore + Plan) trước khi
  viết code — xác nhận zero code neural network độc lập tồn tại trước đó, torch có sẵn
  trên Kaggle base image (không cần pip install thêm), và các script `03`/`07` đã đủ
  linh hoạt (`--renders_root`/`--poses_csv`) để KHÔNG cần sửa gì cho nhánh này.
- Viết 4 file mới (`corrector_model.py`, `08/09/10_*.py`) + 1 notebook mới
  (`kaggle_pixel_corrector.ipynb`, build bằng `nbformat`, 32 cell) + 1 file test mới
  (`tests/test_corrector_pipeline.py`, 32 test).
- Chạy thật (không chỉ đọc code): cài `torch` CPU vào venv riêng, chạy toàn bộ 32 test —
  PASS hết, bao gồm chạy THẬT `09_train_corrector.py` qua subprocess (dataset giả, CPU)
  để xác nhận cơ chế `--resume`/`--overwrite` hoạt động đúng (cộng dồn step đúng, reset
  đúng, chặn re-run không cờ đúng).
- Chạy lại `tests/test_syntax_all.py` (20/20 `.py` + 5/5 `.ipynb` PASS) và
  `tests/test_07_package_submission.py` (21/21 PASS) — xác nhận KHÔNG có regression
  trên các file kế thừa từ `main`.
- **CHƯA test được** (cần GPU + GS_REPO thật, ghi rõ để không ai quên): chức năng thật
  của `08_build_corrector_dataset.py` (render + lưu cặp ảnh đúng), và toàn bộ pipeline
  chạy thật trên Kaggle — xem "Bước tiếp theo".
