# Milestone 15 — Model 2: mạng sửa lỗi pixel 2D (nhánh `feature/nn-image-corrector`)

> File này chỉ áp dụng cho nhánh git `feature/nn-image-corrector` — KHÔNG nằm trên
> `main`. Đây là 1 kiến trúc THỬ NGHIỆM, tách biệt hoàn toàn với pipeline đã verify kỹ
> trên `main` (11 pass verification, xem `docs/MILESTONE_04..14_verification_passN.md`).
> `main` không bị đụng tới bởi bất kỳ file nào ở milestone này.

## Trạng thái hiện tại

**HOÀN TẤT phần code + test cục bộ (không GPU), kể cả nhánh gate (tự học vùng cần sửa).**
Chưa chạy thật trên Kaggle (cần GPU + dataset thật — xem "Bước tiếp theo"). Đang chờ
người dùng cung cấp vị trí checkpoint Round 1 thật đã có sẵn (xem "Bước tiếp theo" mục 0).

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
  KHÔNG pooling, residual predict-and-add, tail_residual khởi tạo 0 để an toàn khi
  chưa/train hỏng), SSIM khả vi thuần `torch`, `save_checkpoint()`/`load_checkpoint()`.
  **Nhánh gate** (cập nhật 2026-07-20, theo yêu cầu người dùng "Model 2 phải tự suy luận
  vùng cần sửa"): thêm đầu ra phụ `tail_gate` (qua sigmoid, [0,1]) làm hệ số nhân lên
  residual — `output = clamp(input + gate*residual, 0, 1)`. Gate KHÔNG được giám sát
  trực tiếp (không có nhãn "vùng nhiễu" nào ở test time) — tự nổi lên từ việc tối ưu
  loss tái tạo cộng 1 số hạng phạt thưa nhẹ `mean(gate)` (`--gate_sparsity_weight`, mặc
  định 0.01, xem `09_train_corrector.py`): không có phạt này, gate=1 khắp ảnh cũng tối
  ưu loss tái tạo ngang/tốt hơn, mạng sẽ không tự học khoanh vùng dù thừa khả năng biểu
  diễn. `forward()` trả về ảnh đã sửa; `forward_with_gate()` trả về thêm (gate,
  residual) — dùng lúc train (tính phạt) và lúc suy luận (xuất heatmap QC, xem
  `10_apply_corrector.py`).
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
  đã hỗ trợ sẵn qua `--renders_root` (KHÔNG sửa script đó). Mặc định (`--save_gate_map`,
  bật sẵn) còn xuất thêm heatmap gate `pipeline/work_corrected/<scene>/gate_maps/` —
  TRẮNG = Model 2 tự tin sửa mạnh, ĐEN = gần như giữ nguyên render gốc — cho phép người
  dùng KIỂM TRA BẰNG MẮT vùng mà mạng tự chọn xử lý (đúng ý người dùng: "chỉ cần kiểm
  tra vùng kết quả render nhiễu để xử lý"), không phải hộp đen hoàn toàn.
- `pipeline/kaggle_pixel_corrector.ipynb` — notebook Kaggle mới, dùng lại nguyên vẹn
  `kaggle_round1_baseline.ipynb` để train Model 1 (khuyến nghị `MODE="final"`), rồi mới
  chạy các bước Model 2 (8→9→10→11→12 QC bằng mắt: trước/gate heatmap/sau→13 lưu Drive).
- `tests/test_corrector_pipeline.py` — 42 test cục bộ (không GPU): kiến trúc mạng (kể cả
  nhánh gate — shape/range/an toàn), checkpoint round-trip, SSIM, toán học ghép ô
  (partition of unity, kể cả ghép gate map), VÀ chạy THẬT (subprocess, CPU)
  `09_train_corrector.py` với dataset giả để xác nhận cơ chế resume/overwrite/gate-loss
  hoạt động đúng — không chỉ đọc code suy đoán.
- `pipeline/kaggle_model2_validation_hcm0031.ipynb` — **notebook kiểm chứng riêng, KHÔNG
  phải luồng nộp bài** (xem mục "Notebook kiểm chứng" bên dưới). Dùng checkpoint +
  dataset scene `hcm0031` (Round 1 cũ, người dùng cung cấp) để render + tính Score THẬT
  (PSNR/SSIM/LPIPS/Score đúng công thức BTC) TRƯỚC và SAU khi áp Model 2, so sánh trực
  tiếp — trả lời đúng câu hỏi "Model 2 giúp tăng bao nhiêu điểm" bằng số liệu thật thay
  vì chỉ QC bằng mắt.

### Notebook kiểm chứng (`kaggle_model2_validation_hcm0031.ipynb`) — vì sao tách riêng

Scene `hcm0031` KHÔNG thuộc registry `pipeline/common/scenes.py` (chỉ có 7 scene Round
2) — cố tình KHÔNG đăng ký thêm vào đó để tránh nhầm lẫn với luồng nộp bài thật. Thay vào
đó, notebook này **import `04_eval_metrics.py`/`08_build_corrector_dataset.py`/
`09_train_corrector.py`/`10_apply_corrector.py` làm THƯ VIỆN** qua `importlib`
(`spec_from_file_location` + `exec_module`, giống hệt kỹ thuật `tests/test_corrector_
pipeline.py` đã dùng) — lấy đúng các hàm THUẦN (không phụ thuộc `Scene`/`get_scene()`):
`build_minicam`/`load_train_poses`/`read_cfg_args` (từ 08), `CorrectorPatchDataset`
(từ 09), `apply_tiled` (từ 10), `eval_scene`/`compute_score`/`print_stats` (từ 04 — dùng
`SimpleNamespace(name="hcm0031")` thay `Scene` thật, vì `eval_scene()` chỉ đọc
`scene.name` để in log, không cần object `Scene` đầy đủ). KHÔNG gọi `main()` của các
script đó (tránh `get_scene("hcm0031")` raise lỗi) — tự viết vòng lặp orchestration
(render, train, so Score) ngay trong notebook, tái dùng ĐÚNG logic đã test thay vì viết
lại từ đầu.

Đã dry-run CPU thật (không GPU) toàn bộ phần code KHÔNG cần GS_REPO/CUDA (Bước 6/8/9/10
— tính Score, train loop, apply_tiled, so sánh) bằng dữ liệu giả — chạy sạch, không lỗi.
Phần Bước 5/7 (render 3DGS thật qua GS_REPO) KHÔNG test được cục bộ (cần CUDA rasterizer
thật) — CHƯA chạy thật, cần Kaggle.

### Vì sao KHÔNG dùng U-Net/pooling, KHÔNG BatchNorm

Render và GT đã align pixel-for-pixel tuyệt đối (cùng pose camera) — tác vụ sửa lỗi cục
bộ (làm nét cạnh, bù chi tiết mảnh) không cần receptive field lớn/bottleneck ngữ nghĩa
mà pooling mang lại, và pooling+upsample có nguy cơ sinh checkerboard artifact — ngược
mục tiêu "ảnh nét". Giữ fully-convolutional nghĩa là CÙNG 1 trọng số áp được cho ảnh
kích thước bất kỳ (train patch cố định, suy luận ảnh test kích thước thật). BatchNorm bị
bỏ vì đi ngược mục tiêu giữ đúng giá trị màu tuyệt đối trong phục hồi ảnh, và dữ liệu
mỗi scene quá ít (vài trăm ảnh) để ước lượng batch statistics ổn định.

## Bug thật tìm được bằng train thật trên Kaggle (không phải suy đoán) — đã sửa

**Gate sập về đúng 0.0000 chỉ sau ~1400 bước** (log thật:
`[1400/4000] loss=0.02062 l1=0.02062 gate_mean=0.0000`), dù `--gate_sparsity_weight`
mặc định chỉ 0.01 (nhỏ) — corrector trở thành no-op vĩnh viễn (không sửa gì cả), mất hết
tác dụng "tự học vùng cần sửa" mà toàn bộ nhánh gate được thiết kế ra để làm.

Nguyên nhân (đối chiếu công thức, verify lại bằng train thật cục bộ trên dữ liệu tổng
hợp có cấu trúc): lúc khởi tạo `residual=0` MỌI NƠI (zero-init có chủ đích, xem trên) —
`d(loss tái tạo)/d(gate) = d(loss)/d(output) * residual = 0` ở bước đầu, tức loss tái tạo
KHÔNG cho gate gradient nào để "bênh vực" việc giữ gate cao. Trong khi đó
`d(phạt thưa)/d(gate) = gate_sparsity_weight` LUÔN LUÔN có gradient thật kéo gate xuống
0, không phụ thuộc gì vào residual. Nếu gate khởi tạo ngẫu nhiên quanh sigmoid(0)=0.5
(mặc định Kaiming trên bias gần 0), nó bị kéo dần về 0 — và vì
`d(loss)/d(residual) = d(loss)/d(output) * gate`, gate càng nhỏ thì residual CÀNG khó
học được gì có ích, vòng lặp tự củng cố, cả gate lẫn residual cùng "chết".

**Sửa:** ép `tail_gate.bias` khởi tạo dương (4.0, `sigmoid(4.0)≈0.982`) thay vì mặc định
PyTorch — gate khởi đầu gần 1 thay vì gần 0.5. Thuộc tính an toàn "output=input lúc khởi
tạo" GIỮ NGUYÊN (chỉ phụ thuộc `residual=0`, không phụ thuộc giá trị gate — `gate*0=0`
bất kể gate bằng bao nhiêu). Nhưng giờ `d(loss)/d(residual) ≈ 0.98 * gradient thật` NGAY
TỪ BƯỚC ĐẦU — residual có cơ hội học sửa lỗi có ích TRƯỚC khi phạt thưa đủ đòn bẩy bóp
gate xuống ở vùng KHÔNG cần sửa.

Verify lại bằng train thật (không mock) trên dữ liệu tổng hợp có lỗi cục bộ ở 1 góc ảnh
(phần còn lại render=GT y hệt): sau 800 bước, `gate_mean≈0.04` (KHÔNG sập về 0), và gate
tự học phân biệt ĐÚNG — `gate_corner≈0.37-0.39` (vùng có lỗi) vs `gate_rest≈0.0000-0.0016`
(vùng không lỗi) — đúng hành vi thiết kế. Thêm 3 test hồi quy vào
`tests/test_corrector_pipeline.py` (`test_gate_does_not_collapse` + check bias dương) —
42 -> 45 test, tất cả PASS.

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
5. **`--gate_sparsity_weight` cần tinh chỉnh bằng mắt, không có giá trị "đúng" cố định.**
   Quá cao -> mạng "sợ" bật gate, gần như không sửa gì (heatmap Bước 12 toàn đen, ảnh
   "sau" gần như y hệt "trước"). Quá thấp/0 -> mất tác dụng khoanh vùng, gate~1 khắp ảnh
   (heatmap toàn trắng, quay lại đúng như bản KHÔNG có gate). Xem heatmap Bước 12 trước
   khi quyết định giữ nguyên mặc định (0.01) hay chỉnh lại.

## Bước tiếp theo (bắt buộc trước khi dùng để nộp bài thật)

0. **[XONG — đã xác định]** Người dùng cung cấp checkpoint thật tại
   `Round1/gs_model/` (local, KHÔNG commit — xem `.gitignore`) — nhưng đây là checkpoint
   scene **`hcm0031`**, thuộc **dataset Round 1 CŨ đã bị BTC huỷ**
   (`Dataset/VAI_NVS_DATA/phase1/public_set/`, 5 scene: `hcm0031, HCM0181, HCM0193,
   HCM0204, hcm0034` — KHÁC HẲN 7 scene Round 2 hiện tại), KHÔNG phải checkpoint Round 2
   thật. Đã xác nhận qua `cfg_args` (`source_path=".../work/hcm0031/..."`) + đối chiếu
   `Dataset/VAI_NVS_DATA/phase1/public_set/hcm0031/`. Checkpoint hợp lệ (30000 iteration,
   4.9 triệu Gaussian, `plyfile` đọc được, 62 property/vertex đúng sh_degree=3) — dùng để
   **kiểm chứng** pipeline Model 2 bằng Score THẬT (dataset này CÓ ảnh GT test thật,
   Round 2 thì KHÔNG), không dùng để nộp bài.
1. **Chạy `kaggle_model2_validation_hcm0031.ipynb` trên Kaggle** (notebook mới, xem mục
   "Notebook kiểm chứng" bên dưới) — chưa chạy được ở đây vì máy cục bộ có GPU (GTX 1650,
   CUDA 12.8 qua WSL2) nhưng THIẾU `nvcc`/CUDA toolkit để build
   `diff-gaussian-rasterization` + RAM quá thấp lúc kiểm tra (295MB trống) — người dùng
   đã chọn KHÔNG cài CUDA toolkit cục bộ (rủi ro treo máy), nên phải chạy trên Kaggle.
2. Sau khi có Score TRƯỚC/SAU thật từ notebook kiểm chứng (`hcm0031`): nếu Model 2 tăng
   Score rõ ràng, đó là bằng chứng thật (không phải suy đoán) để tự tin áp dụng
   `kaggle_pixel_corrector.ipynb` cho 7 scene Round 2 thật (train Model 1 `MODE="final"`
   mới cho từng scene Round 2, vì checkpoint `hcm0031` KHÔNG dùng được cho Round 2). Nếu
   Score KHÔNG tăng/giảm trên `hcm0031`, cân nhắc dừng nhánh này trước khi tốn công cho
   cả 7 scene thật.
3. Lặp lại `kaggle_pixel_corrector.ipynb` cho từng scene Round 2, đóng gói thử
   `submission.zip` từ `pipeline/work_corrected`, đối chiếu dung lượng/định dạng giống
   hệt kiểm tra đã làm trên `main` (`07_package_submission.py` không đổi nên logic kiểm
   tra vẫn y hệt).
4. Cân nhắc: giữ `main` (đã verify 11 pass) làm phương án nộp bài AN TOÀN mặc định; chỉ
   thay bằng `work_corrected/` cho SCENE NÀO kiểm chứng thấy tốt hơn thật sự (bằng Score
   nếu có thể, hoặc bằng mắt như thiết kế gốc).
5. KHÔNG merge nhánh này vào `main` cho tới khi có bằng chứng thật (Score hoặc thị giác,
   không chỉ suy đoán từ code) rằng Model 2 cải thiện chất lượng — nếu không cải thiện/tệ
   hơn, có thể bỏ hẳn nhánh này mà không ảnh hưởng gì tới `main`.

### 2026-07-20 — Notebook kiểm chứng bằng dữ liệu thật (`hcm0031`)

- Người dùng cung cấp checkpoint thật tại `Round1/gs_model/` (local) — kiểm tra phát
  hiện đây là scene `hcm0031` thuộc **dataset Round 1 CŨ đã bị BTC huỷ**
  (`Dataset/VAI_NVS_DATA/phase1/public_set/`), KHÔNG phải 1 trong 7 scene Round 2. Xác
  nhận qua `cfg_args` (`source_path` chứa `hcm0031`) đối chiếu với dataset local sẵn có.
  Điểm hay: dataset Round 1 cũ này CÓ ảnh GT test thật (`test/images/`, 50 ảnh) — Round 2
  KHÔNG có GT cho bất kỳ scene nào — nên đây là cơ hội DUY NHẤT hiện có để đo Score
  khách quan thật (không phải ước lượng qua holdout tự tạo).
- Phát hiện máy cục bộ CÓ GPU thật qua WSL2 (GTX 1650, driver CUDA 12.8) nhưng THIẾU
  `nvcc` (CUDA toolkit) để build `diff-gaussian-rasterization`/`simple-knn`, và RAM lúc
  đó chỉ còn 295MB trống, swap gần đầy — hỏi người dùng trước khi cài gì nặng (rủi ro
  treo máy), người dùng chọn KHÔNG cài, chỉ kiểm tra nhẹ + chuẩn bị notebook chạy trên
  Kaggle. Kiểm tra nhẹ cục bộ (không CUDA): cài `plyfile`, đọc `point_cloud.ply` xác nhận
  4.948.598 Gaussian, 62 property/vertex (đúng sh_degree=3, không hỏng file); đọc
  `cameras.json` (200 camera, khớp 200 ảnh train), `cfg_args` (baseline thuần, KHÔNG
  antialiasing/depth/exposure đặc biệt — đối chiếu với notebook train gốc
  `Round1/bts-digital-twin-public.ipynb` xác nhận không dùng cờ nào đặc biệt).
- Viết `pipeline/kaggle_model2_validation_hcm0031.ipynb` (23 cell) — render pose test
  bằng checkpoint thật → tính Score THẬT (trước) → render pose train → train Model 2 →
  áp lên render test → tính Score THẬT (sau) → so sánh + hiển thị ảnh mẫu (trước/gate
  heatmap/sau/GT). Tái dùng `04_eval_metrics.py`/`08_build_corrector_dataset.py`/
  `09_train_corrector.py`/`10_apply_corrector.py` làm THƯ VIỆN qua `importlib` (không
  gọi `main()`, tránh vướng `get_scene("hcm0031")` chưa đăng ký) — xem mục "Notebook
  kiểm chứng" ở trên.
- Dry-run CPU thật (không GPU, dữ liệu giả) toàn bộ phần code KHÔNG cần GS_REPO/CUDA
  (Bước 6/8/9/10 của notebook) — chạy sạch, không lỗi, xác nhận glue code (viết trực
  tiếp trong notebook, chưa từng chạy) không có bug cú pháp/API trước khi người dùng tốn
  GPU Kaggle thật. Phần render 3DGS thật (Bước 5/7) CHƯA test được (cần CUDA thật).
- Thêm `Round1/` vào `.gitignore` (checkpoint 1.2GB + dataset 280MB, không commit, giống
  `checkpoints/` đã ignore trước đó). Chạy lại `tests/test_syntax_all.py` — 20/20 `.py` +
  7/7 `.ipynb` PASS (bao gồm notebook mới) — không regression.
- **CHƯA làm**: chạy thật `kaggle_model2_validation_hcm0031.ipynb` trên Kaggle (cần
  người dùng điền 2 link Drive — checkpoint + dataset `hcm0031` — vào Bước 4) để có
  Score TRƯỚC/SAU thật.

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

### 2026-07-20 — Thêm nhánh gate (tự học vùng cần sửa)

- Người dùng làm rõ qua 2 câu hỏi: (a) "tự suy luận/kiểm tra vùng nhiễu" nghĩa là Model 2
  phải TỰ HỌC dự đoán vùng cần sửa (thêm nhánh confidence/gate map trong kiến trúc, học
  cùng lúc với train, KHÔNG phải heuristic cổ điển cố định), (b) người dùng ĐÃ CÓ
  checkpoint/kết quả thật từ 1 lần chạy `kaggle_round1_baseline.ipynb` trên Kaggle trước
  đó (chưa cung cấp vị trí cụ thể — xem "Bước tiếp theo" mục 0).
- Sửa `ResidualCorrectorNet`: thêm `tail_gate` (sigmoid, 1 kênh) làm hệ số nhân lên
  residual, `forward_with_gate()` trả về (output, gate, residual). Thuộc tính an toàn
  (zero-init -> output=input) giữ nguyên vì chỉ phụ thuộc `tail_residual` zero-init,
  không phụ thuộc gate.
- Sửa `09_train_corrector.py`: thêm `--gate_sparsity_weight` (mặc định 0.01), loss =
  recon_loss + weight*mean(gate) — cơ chế DUY NHẤT khiến gate tự học khoanh vùng (không
  có phạt này, gate=1 khắp ảnh cũng tối ưu loss ngang/tốt hơn). `loss_history` giờ có
  thêm field `gate_mean` để theo dõi qua các lần train.
- Sửa `10_apply_corrector.py`: `apply_tiled()` nhận `return_gate=True`, ghép/trộn gate
  map bằng ĐÚNG cơ chế trộn ảnh chính (dùng chung `ramp_weight`). Model không có
  `forward_with_gate` (tương thích ngược, vd model test) tự coi như gate=1. Thêm
  `--save_gate_map` (mặc định BẬT) xuất heatmap PNG grayscale ra `gate_maps/`.
- Sửa notebook Bước 12: hiển thị 3 cột trước/gate-heatmap/sau thay vì 2 cột.
- Thêm 10 test mới (32 -> 42 test), chạy thật (không chỉ đọc code): shape/range nhánh
  gate, thuộc tính an toàn giữ nguyên, ghép ô gate map bằng model giả gate cố định (xác
  nhận blend đúng cơ chế), model không có gate API vẫn chạy được (fallback), subprocess
  thật xác nhận `loss_history` có `gate_mean` và `--gate_sparsity_weight` được nhận cờ
  đúng. Toàn bộ 42/42 PASS + `test_syntax_all.py`/`test_07_package_submission.py` vẫn
  xanh (không regression).
- **CHƯA làm** (cần input từ người dùng): xác nhận vị trí checkpoint Round 1 thật để test
  bằng dữ liệu thật thay vì dataset giả cục bộ — xem "Bước tiếp theo" mục 0.
