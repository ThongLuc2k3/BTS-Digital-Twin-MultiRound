# Kiến thức port từ repo `BTS-Digital-Twin` — KHÔNG được lặp lại các bug này

> Toàn bộ mục dưới đây đã được TÌM RA THẬT (bằng dữ liệu/test thật, không suy đoán) và
> SỬA THẬT ở repo `BTS-Digital-Twin` (cùng thư mục cha, xem `WORKLOG.md` ở đó để đọc
> chi tiết đầy đủ + bằng chứng nếu cần đối chiếu). Khi viết code mới ở repo này, PHẢI
> áp dụng ngay các fix này từ đầu, không viết code "ngây thơ" rồi tự dính lại bug cũ.

## 1. Dataset

- 7 scene: `HCM0421`, `HCM0539`, `HCM0540`, `HCM0644`, `HCM0674` (domain BTS) +
  `bonsai`, `chair` (domain generic). Link Google Drive dataset thật xem trong
  `kaggle_private.ipynb` của repo cũ (`GDRIVE_URL`) — copy sang khi cần.
- **Bug scale COLMAP (NGHIÊM TRỌNG, ảnh hưởng MỌI script tự đọc `points2D.xy`)**:
  `points2D[i].xy` trong sparse COLMAP (CẢ sparse gốc BTC cấp lẫn sparse tự sinh lại
  qua `image_undistorter`) được lưu ở **ĐỘ PHÂN GIẢI KHÁC** với
  `camera.width/height/fx/fy/cx/cy` (đã đúng theo ảnh `train/images/` thật). Lệch theo
  hệ số cố định mỗi scene, đo thật: `HCM0421`=3.9042x, `chair`=1.4999x, `bonsai`=1.0000x
  (không lệch). **KHÔNG hardcode các số này** — luôn tự đo runtime bằng cách so
  `point2d.xy` với `image.project_point(point3D.xyz)` (dùng camera intrinsics đã đúng),
  lấy median tỉ lệ trên nhiều điểm của 1 ảnh có nhiều quan sát nhất. Bất kỳ script nào
  đọc `points2D.xy` trực tiếp (dựng mask từ khung pixel, tìm patch tham chiếu...) PHẢI
  qua bước tự đo scale này trước.
- `image_undistorter` của COLMAP viết lại ĐÚNG `camera.width/height` (khớp ảnh
  `train/images/`) nhưng KHÔNG viết lại `points2D.xy` theo scale mới — đây là NGUỒN GỐC
  bug trên, không phải lỗi ngẫu nhiên.
- File zip dataset khi giải nén KHÔNG đảm bảo có đúng 1 lớp thư mục bọc ngoài tên cố
  định — tự dò thư mục chứa `>= 4/7` tên scene mong đợi trực tiếp bên trong (không bắt
  buộc đúng tên "VAI_NVS_DATA_ROUND2" hay tương tự), loại trừ `__MACOSX/` (rác do Mac
  nén zip, tự nhân bản cấu trúc thư mục thật nhưng chỉ chứa file rác `._<tên>`).
- Holdout tự tạo (vì test thật không có GT): `pycolmap.Rigid3d.rotation.quat` trả về
  **[x,y,z,w]**, KHÁC quy ước `qw,qx,qy,qz` dùng trong `test_poses.csv`/toàn bộ code —
  phải tự đảo thứ tự khi ghi CSV pose, nếu không mọi pose sẽ SAI HOÀN TOÀN mà không có
  lỗi báo rõ (chỉ lộ ra khi so ảnh render lệch góc).

## 2. Train (`graphdeco-inria/gaussian-splatting`, commit pin `54c035f7834b564019656c3e3fcc3646292f727d`)

- `cfg_args` (train.py tự ghi) CHỈ chứa `ModelParams` (`sh_degree`, `resolution`,
  `train_test_exp`, ...) — **KHÔNG BAO GIỜ chứa `antialiasing`** (field của
  `PipelineParams`), dù lúc train có bật `--antialiasing` hay không. Nếu render tự
  "đoán" antialiasing từ `cfg_args` sẽ LUÔN ra `False` — SAI hoàn toàn nếu lúc train đã
  bật, làm méo PSNR/SSIM/LPIPS mà KHÔNG có lỗi báo (đây là bug ~10 điểm từng xảy ra
  thật ở vòng thi trước). **Fix bắt buộc**: script train tự ghi thêm 1 file riêng
  (`pipeline_train_flags.json`: `antialiasing`/`depth_prior`/`exposure_comp`/...) ngay
  cạnh `cfg_args`, và MỌI script render sau này phải đọc file này (không phải đoán từ
  `cfg_args`) để biết chính xác cấu hình thật đã train.
- `densify_until_iter` mặc định CỐ ĐỊNH `15000` trong `OptimizationParams`, KHÔNG tự co
  giãn theo `--iterations`. Ý nghĩa quan trọng: nếu `--iterations` == `densify_until_iter`
  (vd chạy holdout 15000 iter), densify chạy suốt KHÔNG BAO GIỜ ổn định trước khi kết
  thúc → rủi ro CUDA OOM cao nhất ở CUỐI quá trình train (Gaussian tích luỹ liên tục).
  Nếu `--iterations` > `densify_until_iter` (vd final 30000), nửa sau train KHÔNG sinh
  thêm Gaussian — ổn định hơn. Muốn tắt hẳn densify (đúng ý "chỉ tinh chỉnh, không sinh
  thêm Gaussian" cho các vòng refine) — set `--densify_until_iter 0` tường minh.
- CUDA OOM đã gặp thật khi bật `DEPTH_PRIOR`/`--depths` trên GPU Kaggle (T4, ~14.56GiB)
  — giảm được bằng `--sh_degree 2` (mặc định 3) + `--densify_grad_threshold 0.0004`
  (mặc định 0.0002). NHƯNG: đo thật trên holdout (`HCM0421`, công bằng cùng 15000 iter)
  cho thấy depth-prior (dù đã fix OOM) vẫn cho Score THẤP HƠN baseline không depth-prior
  (0.644 vs 0.6616, -1.7 điểm) — **không đáng dùng cho scene BTS**, dù kỹ thuật chạy
  được. Bài học: chạy được (không OOM) KHÔNG đồng nghĩa có lợi — luôn đo Score thật.
- Antenna-focus (loss masking theo 1 khung 3D cố định, patch `train.py` thêm
  `--antenna_weights_json`) đã test thật (`HCM0421` holdout) — Score gần như hoà với
  baseline (0.6611 vs 0.6616, trong biên độ nhiễu tự nhiên giữa 2 lần train cùng cấu
  hình ~0.0004) — KHÔNG cải thiện Score TỔNG đo được (dù có thể nét hơn cục bộ ở vùng
  đánh dấu, không đủ kéo Score trung bình toàn ảnh). Cơ chế patch (weighted L1 loss)
  vẫn hữu ích — được tái sử dụng cho error-guided refine (mục 3 dưới), chỉ khác NGUỒN
  mask (đo tự động thay vì khung cố định do người chọn).
- `Scene(dataset, gaussians, load_iteration=N)` là cơ chế CHÍNH THỨC có sẵn của repo để
  nạp `.ply` có sẵn (KHÔNG cần file `.pth` optimizer state, khác `--start_checkpoint`)
  — nhưng đi qua nhánh này thì `create_from_pcd()` (nơi DUY NHẤT set
  `gaussians.spatial_lr_scale`) KHÔNG được gọi, `spatial_lr_scale` giữ nguyên mặc định
  `0`. Vì learning rate vị trí = `position_lr_init * spatial_lr_scale`, không tự sửa sẽ
  làm Gaussian **ĐỨNG YÊN HOÀN TOÀN** suốt quá trình train tiếp theo mà KHÔNG báo lỗi
  gì. Fix bắt buộc: sau `Scene(..., load_iteration=N)`, gán lại
  `gaussians.spatial_lr_scale = scene.cameras_extent` (đúng giá trị
  `Scene.__init__` tự tính cho nhánh train-from-scratch bình thường) TRƯỚC khi gọi
  `gaussians.training_setup(opt)`.
- Công thức weighted L1 loss (dùng chung cho MỌI cơ chế loss-masking, đã verify đúng):
  `Ll1 = (weight * |render - gt|).sum() / weight.expand_as(render).sum()` — `weight`
  là tensor `(1, H, W)`, giá trị `1.0` = không đổi, `> 1.0` = ưu tiên vùng đó.

## 3. Error-guided refine (cơ chế train nhiều vòng — CỐT LÕI của repo này)

Ý tưởng: render lại pose TRAIN (có GT thật, khác test/holdout), đo lỗi pixel thật, sinh
mask trọng số, train tiếp ngắn ưu tiên vùng lỗi cao. Toàn bộ cơ chế đã build + verify
cục bộ (không GPU) ở repo cũ ngày 2026-07-18 — port nguyên vẹn 2 file:
`apply_error_refine_patch.py` (vá `train.py`) + `12_generate_error_mask.py` (sinh mask).

- Công thức mask: `err = mean(|render-GT|, kênh màu)` → Gaussian blur (radius mặc định
  4px, tránh mask vụn theo từng pixel JPEG noise) → percentile hoá:
  `weight = 1 + (MAX_WEIGHT-1) * clip((err-p50)/(p95-p50), 0, 1)` (vùng lỗi trung vị trở
  xuống không đổi, top 5% lỗi được boost tới `MAX_WEIGHT`, mặc định 6x).
- Lưu mask dạng 16-bit PNG: `pixel = round(weight * 1000)` — hằng số `1000.0` PHẢI
  giống hệt ở cả script sinh mask (dùng `cv2.imwrite`) lẫn patch đọc mask lúc train
  (dùng `PIL.Image.open`) — đã verify 2 thư viện tương thích, không lệch giá trị
  round-trip.
- Ảnh GT dùng để so sánh PHẢI là ảnh ĐÃ undistort chính xác pixel-for-pixel
  (`colmap/dense/images/`), KHÔNG chấp nhận bản xấp xỉ resize như cách làm tạm cho
  sanity-check thông thường — sai số resize sẽ lẫn vào chính error map muốn đo, làm mất
  tác dụng của kỹ thuật này. `dense/images/` bị script train tự xoá sau khi train xong
  (dọn đĩa) — PHẢI chạy lại bước undistort COLMAP (deterministic, seed cố định) để tái
  tạo trước khi sinh mask.
- 2 patch (antenna-focus VÀ error-refine) đụng CÙNG 1 điểm vá trong `train.py` ("Loss")
  — KHÔNG dùng chung được trên cùng 1 checkpoint/lần chạy. Mỗi patch tự kiểm tra + báo
  lỗi rõ nếu phát hiện file đã bị patch kia vá trước, không âm thầm chạy sai.
- **Giả định CHƯA có bằng chứng thực nghiệm xác nhận mức độ đúng**: lỗi đo trên ảnh
  TRAIN có tương quan với lỗi ở pose TEST/HOLDOUT lân cận hay không — đây là lý do bắt
  buộc mỗi vòng refine phải tự đo Score holdout TRƯỚC/SAU, CHỈ giữ vòng đó nếu Score
  tăng thật, không suy đoán/tin trực giác.

## 4. Notebook Kaggle — cạm bẫy Jupyter/IPython

- `!command` (shell magic) khi thất bại (exit code != 0) **KHÔNG raise Python
  exception** — không tự dừng "Run All". Nếu 1 cell chỉ nên chạy CÓ ĐIỀU KIỆN (vd chỉ
  chạy nếu cell trước thất bại), PHẢI tự kiểm tra bằng code Python tường minh (vd
  `if not os.path.exists(checkpoint_path):`), KHÔNG dựa vào giả định "cell trước lỗi
  thì cell sau sẽ không chạy tới" — đã gặp bug thật: 1 cell backup tự chạy tiếp dù cell
  train chính đã xong, gây lỗi (không hỏng dữ liệu nhưng gây nhiễu log).
- Nội suy biến trong dòng `!lệnh {biểu_thức}` — chỉ dùng biểu thức ĐƠN GIẢN (tên biến,
  hoặc truy cập dict/attr đơn giản như `{os.environ['X']}`, đã verify chạy thật OK).
  TRÁNH biểu thức phức tạp (join, ternary lồng bên trong `{}`) — chưa có tiền lệ chạy
  thật trong dự án, rủi ro không đáng — tính trước ra biến đơn giản rồi mới nội suy.
- Multi-line `!lệnh ... \` (tiếp dòng bằng backslash) BÊN TRONG khối `if:` thụt lề — đã
  verify chạy thật OK nhiều lần (dùng cho `04_render_test_poses.py`), an toàn dùng lại.
- Tải checkpoint từ Google Drive để dùng ở notebook khác: PHẢI tải NGUYÊN thư mục
  `gs_model/` (`gdown --fuzzy --folder`), KHÔNG chỉ file `.ply` đơn — thiếu `cfg_args`/
  `pipeline_train_flags.json` sẽ làm render tự đoán SAI `antialiasing` (xem mục 2) mà
  không báo lỗi. Sau khi tải, PHẢI assert cứng có đủ `cfg_args` +
  `pipeline_train_flags.json` + ít nhất 1 `point_cloud.ply` trước khi dùng tiếp — không
  tin cấu trúc thư mục Drive mù quáng.

## 5. Đóng gói submission

- Tên file trong zip PHẢI giữ NGUYÊN `image_name` từ `test_poses.csv` (kể cả đuôi
  `.JPG` gốc nếu có) — KHÔNG đổi đuôi. Nội dung file phải MÃ HOÁ LẠI đúng định dạng
  theo đuôi đó (đuôi `.jpg` → JPEG thật, không phải đổi tên suông giữ nguyên byte PNG —
  từng gây file nặng gấp 4-8 lần, vượt hạn mức dung lượng nộp bài).
- `PSNR_max` trong công thức Score KHÔNG được đề bài công bố số cụ thể — repo cũ dùng
  `50.0` làm ước lượng tham khảo (ghi rõ đây là ước lượng, không phải số chính thức từ
  BTC) khi tự đo Score holdout để so sánh tương đối giữa các cấu hình.

## 6b. Bug MỚI tìm ra ở CHÍNH repo này (không phải từ repo tiền nhiệm)

Tìm ở verification pass #1 (`docs/MILESTONE_04_verification_pass1.md`), khi audit chéo
giữa các file do các agent khác nhau viết (không ai tự kiểm tra được, vì mỗi agent chỉ
sở hữu 1 phần):

- **Tên file kết quả `04_eval_metrics.py` (`eval_metrics.csv`) bị notebook Vòng 2/3 gõ
  nhầm thành `eval_metrics.txt`** khi đọc lại để so Score trước/sau — 2 file khác nhau
  do 2 agent khác nhau viết (agent Round-1 viết `04_eval_metrics.py`, agent refine viết
  `kaggle_round2_refine.ipynb`/`kaggle_round3_refine.ipynb`), không ai đối chiếu tên
  file thật. Bài học: khi 1 file ĐỌC lại output của 1 file KHÁC do agent khác viết, PHẢI
  đối chiếu tên file THẬT trong code nguồn (không suy đoán/gõ theo trí nhớ đuôi
  `.txt`/`.csv`).
- **Thiếu `git submodule update --init --recursive` sau `git checkout <commit pin>`** ở
  `kaggle_submission.ipynb` (3 notebook kia đều có dòng này, đúng theo comment trong
  `02_train_baseline.sh`, nhưng agent viết `kaggle_submission.ipynb` port từ 1 bản cũ
  hơn chưa có dòng này). Bài học: khi có N notebook làm CÙNG 1 bước cài đặt (ở đây là
  Bước 2 — clone + build gaussian-splatting), PHẢI đối chiếu cả N bản với nhau (diff),
  không chỉ tự tin bản mình port đúng.

## 6c. Bug MỚI tìm ra ở verification pass #2 (`docs/MILESTONE_05_verification_pass2.md`)

- **`02_train_baseline.sh` — thông báo "[CỨU ĐƯỢC]" (rescue message) khi train crash
  giữa chừng chọn NHẦM checkpoint NHỎ HƠN** (vd báo `iteration_7000` trong khi
  `iteration_15000` mới là checkpoint lớn nhất/mới nhất còn sống sót). Nguyên nhân: dòng
  `LAST_CKPT=$(ls -d "$MODEL_DIR"/point_cloud/iteration_* | sort -t_ -k2 -n | tail -1)`
  dùng đúng bug class đã được milestone 02 (`docs/MILESTONE_02_refine_pipeline.md`) tự
  phát hiện + fix ở `06_train_refine.sh` (`_latest_iteration_dir()`) — nhưng KHÔNG được
  backport lại vào `02_train_baseline.sh`, script gốc/sinh ra thông báo cứu hộ tương tự.
  Lý do bug thật (đã verify bằng lệnh `sort` thật): đường dẫn model đầy đủ chứa NHIỀU
  dấu `_` đứng TRƯỚC "iteration_N" (`gs_model`, `point_cloud` đều có `_`), nên field số 2
  (`-k2`, tách theo `_`) KHÔNG phải là số iteration — `sort -n` coi field đó là `0`
  (không parse được số), sort giữ nguyên thứ tự lexical gốc của `ls -d` (glob liệt kê
  "iteration_15000" TRƯỚC "iteration_7000" vì ký tự '1' < '7'), nên `tail -1` chọn nhầm
  "iteration_7000" — checkpoint CŨ HƠN, kém train hơn. Đây CHỈ là bug ở dòng thông báo
  (không dùng để quyết định logic nào khác trong script), nhưng có thể khiến người dùng
  dưới áp lực deadline tin nhầm và dùng checkpoint kém hơn để render/nộp bài. Đã verify
  bằng test thật (`train.py` giả, `ITERATIONS=18500`, `CRASH_AT=18000` — đã qua cả 2
  checkpoint 7000 và 15000): trước khi sửa, thông báo trỏ sai vào `iteration_7000`; sau
  khi sửa (thêm hàm `_latest_iteration_dir()` bash thuần, so số nguyên, không tách
  trường theo `_`/khoảng trắng — y hệt cách `06_train_refine.sh` đã làm), thông báo trỏ
  đúng `iteration_15000`. **Đã sửa**: `pipeline/scripts/02_train_baseline.sh` — thêm
  `_latest_iteration_dir()`, thay `LAST_CKPT=$(ls -d ... | sort -t_ -k2 -n | tail -1)`
  bằng `LAST_CKPT="$(_latest_iteration_dir "$MODEL_DIR")"`.
  Bài học: khi 1 bug class được phát hiện + fix ở 1 file, PHẢI tự hỏi "file nào khác
  trong repo có ĐOẠN CODE TƯƠNG TỰ (copy/port từ cùng 1 nguồn, hoặc cùng tác giả viết
  cùng lúc) có thể dính CÙNG bug này chưa được kiểm tra?" — không chỉ coi bug đã "xử lý
  xong" sau khi fix đúng 1 chỗ tìm thấy nó đầu tiên.
- Đã re-verify (không tìm thêm bug mới): 2 fix của pass #1 (`eval_metrics.txt`->`.csv`
  ở notebook Vòng 2/3, `git submodule update` ở `kaggle_submission.ipynb`) — đối chiếu
  trực tiếp `04_eval_metrics.py::write_csv()` (ghi `renders_root/<scene>/eval_metrics.csv`,
  `renders_root` mặc định = `pipeline/work`) khớp CHÍNH XÁC đường dẫn
  `/kaggle/working/pipeline/work/{SCENE}/eval_metrics.csv` mà cả 2 notebook Vòng 2/3 đọc
  lại (không chỉ tên file — cả thư mục cũng khớp); và diff byte-for-byte cell clone
  `gaussian-splatting` của cả 4 notebook xác nhận thứ tự `git checkout <pin>` rồi mới
  `git submodule update --init --recursive` giống hệt nhau ở cả 4 file.

## 6. Triết lý test — áp dụng cho MỌI code mới ở repo này

- Không có GPU cục bộ (Kaggle mới có GPU) — TOÀN BỘ phần train/render thật CHỈ verify
  được trên Kaggle. Nhưng vẫn phải test cục bộ MỌI THỨ có thể test được trước khi tin
  code chạy đúng trên Kaggle thật (tiết kiệm GPU quota, tránh lặp lại vòng chờ):
  - Cú pháp: `python -m py_compile` mọi file `.py`, `nbformat.validate()` mọi `.ipynb`.
  - Patch `train.py`: áp thật lên 1 bản clone `gaussian-splatting` sạch (không cần
    CUDA để clone/áp patch — chỉ cần train/render thật mới cần GPU), verify
    `py_compile` sau vá.
  - Logic thuần Python/numpy (công thức, parse pose, round-trip file): test bằng dữ
    liệu giả lập HOẶC dữ liệu COLMAP thật đã có sẵn cục bộ (không cần pycolmap có sẵn —
    cài vào venv riêng nếu cần, xem cách làm ở repo cũ).
  - Shell script: test bằng `train.py` GIẢ (mock, in tiến độ giả + ghi file `.ply` giả)
    để verify đúng luồng gọi lệnh/xử lý lỗi/resume — đã bắt được bug thật theo cách này
    (vd lỗi `set -e` không bắt được exit code trong process substitution).
  - LUÔN thử cả kịch bản THÀNH CÔNG lẫn kịch bản LỖI (input sai, crash giữa chừng) —
    bug thường nằm ở nhánh xử lý lỗi, không phải nhánh chính.
