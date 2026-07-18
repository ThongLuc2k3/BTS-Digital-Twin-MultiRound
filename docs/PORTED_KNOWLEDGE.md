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

## 6d. Bug MỚI tìm ra ở verification pass #3 (`docs/MILESTONE_06_verification_pass3.md`)

- **`kaggle_submission.ipynb` có bản logic symlink dataset CŨ/kém an toàn hơn 3 notebook
  kia** — bug class "sibling file không được backport fix" (đúng loại pass #2 đã tìm,
  nhưng ở vị trí khác hoàn toàn: notebook thay vì shell script). 3 notebook
  `kaggle_round1_baseline.ipynb`/`kaggle_round2_refine.ipynb`/`kaggle_round3_refine.ipynb`
  xử lý ĐÚNG cả 2 trường hợp target đã tồn tại (symlink cũ → `unlink()`; thư mục THẬT
  còn sót lại → `shutil.rmtree()`) rồi LUÔN tạo symlink mới. `kaggle_submission.ipynb`
  (viết độc lập bởi agent milestone 03, không đối chiếu byte-for-byte với 3 notebook
  kia) có bản: `target.unlink() if target.is_symlink() else None` rồi
  `if not target.exists(): os.symlink(...)` — nếu `target` là THƯ MỤC THẬT (không phải
  symlink) còn sót từ 1 lần chạy trước, biểu thức ternary không làm gì cả (`None`), thư
  mục cũ không bị xoá, `target.exists()` vẫn `True` sau đó nên `os.symlink()` KHÔNG BAO
  GIỜ được gọi — dataset MỚI vừa tải bị ÂM THẦM bỏ qua, notebook tiếp tục chạy với dữ
  liệu CŨ mà không có lỗi/cảnh báo nào. Trigger hẹp (chỉ lộ khi user chạy lại Bước 4
  trong CÙNG phiên Kaggle, vd retry sau lỗi mạng hoặc đổi `GDRIVE_URL`) nhưng hậu quả
  nặng (đóng gói `submission.zip` sai dataset ở đúng notebook nộp bài cuối cùng, không
  báo lỗi). Verify bằng test thật (dựng thư mục target giả có file đánh dấu STALE,
  chạy đúng code cũ → xác nhận symlink không được tạo, dữ liệu cũ còn nguyên; chạy code
  đã sửa → xác nhận symlink trỏ đúng dataset mới). **Đã sửa**: đổi
  `kaggle_submission.ipynb` sang đúng logic robust của 3 notebook kia.
- **Cell kiểm tra GPU ở cả 4 notebook chỉ `print()` cảnh báo, không dừng "Run All"**:
  vi phạm triết lý "fail loudly" đã áp dụng nhất quán ở mọi nơi khác trong dự án — nếu
  người dùng quên bật Accelerator GPU, cell này chỉ in 1 dòng cảnh báo dễ chìm giữa log
  `!pip install` ngay sau, notebook tiếp tục "chạy" (tải dataset, build extension —
  build KHÔNG cần GPU device nên có thể "thành công" giả) rồi mới crash muộn ở lần gọi
  `.cuda()` đầu tiên, sau khi đã tốn nhiều phút quota Kaggle — rủi ro lãng phí thời gian
  nghiêm trọng dưới deadline gấp (`docs/00_MASTER_PLAN.md` mục 1). **Đã sửa**: cả 4
  notebook — thêm `raise SystemExit(...)` ngay nếu `not torch.cuda.is_available()`.
- Bài học (khác góc pass #2, cùng tinh thần): khi audit "backport-style bug", đừng chỉ
  giới hạn ở các file `.sh`/`.py` — notebook Jupyter cũng có cell boilerplate lặp lại
  giữa nhiều file, và dễ bị bỏ sót hơn vì không có công cụ diff tự nhiên như `git diff`
  trên source thuần (phải tự dump `cell.source` ra rồi so bằng script, không đọc bằng
  mắt). Cũng: một cell có vẻ "chỉ là cảnh báo UX" (GPU check) vẫn có thể là 1 dạng bug
  thật theo đúng triết lý "fail loudly" mà dự án đã tự đặt ra ở những chỗ khác — audit
  nhất quán triết lý, không chỉ audit tính đúng-sai toán học/logic.
- Đã re-verify các fix của pass #1/#2 (đọc lại `_latest_iteration_dir()` ở cả 2 file
  `.sh`, đọc lại schema `pipeline_train_flags.json`, đối chiếu CLI contract
  `03_render_test_poses.py` với 3 script còn lại) — không tìm thêm sai lệch nào.

## 6e. Bug MỚI tìm ra ở verification pass #5 (`docs/MILESTONE_08_verification_pass5.md`)

Pass này đổi hướng: không re-audit các fix cũ nữa (4 pass trước đã làm kỹ), mà chủ động
**thử phá** bằng input/kịch bản chưa ai test (re-run, input rỗng/hỏng, dependency latest
thật) — tìm ra **3 bug thật MỚI**, cả 3 đều verify bằng THỰC THI THẬT (không suy đoán):

1. **`02_train_baseline.sh` KHÔNG chặn re-run đè lên checkpoint SỐ CŨ khác cấu hình** —
   script không `--start_checkpoint` nên MỖI lần chạy train lại từ đầu; nếu re-run cùng
   scene với `ITERATIONS` NHỎ HƠN lần trước (hoặc `ANTIALIASING` khác), các checkpoint
   `iteration_<N>` SỐ LỚN của lần trước CÒN SÓT LẠI nguyên vẹn (train.py chỉ ghi/ghi đè
   đúng `SAVE_ITERATIONS` của lần NÀY) trong khi `pipeline_train_flags.json` bị ghi đè
   theo cấu hình MỚI ở cuối script — `03_render_test_poses.py::find_latest_iteration()`
   mặc định chọn iteration SỐ LỚN NHẤT (checkpoint SÓT LẠI, cấu hình CŨ) nhưng đọc
   antialiasing từ flags MỚI, làm méo hoàn toàn PSNR/SSIM/LPIPS mà KHÔNG báo lỗi — đúng
   loại bug đã cảnh báo ở mục 2 nhưng do nguyên nhân MỚI (checkpoint sót lại từ re-run,
   không phải thiếu file). Trigger THẬT (không phải giả định): chính notebook
   `kaggle_round1_baseline.ipynb` KHUYẾN NGHỊ tường minh ở Bước 6 quy trình "chạy vài
   version MODE=holdout (15000 iter) rồi 1 version MODE=final (30000 iter)" — nếu làm
   NGƯỢC LẠI (final trước, holdout sau, trong cùng phiên tương tác không restart kernel
   — vd thử lại A/B `ANTIALIASING` như chính usage-comment của script gợi ý), checkpoint
   `iteration_30000` (train 100% ảnh, dùng để nộp bài) CÒN SÓT LẠI và bị
   `03_render_test_poses.py` dùng NHẦM để tính Score holdout — vừa sai antialiasing vừa
   RÒ RỈ DỮ LIỆU (model đã thấy ảnh holdout lúc train `final`), làm Score đo được KHÔNG
   CÒN Ý NGHĨA mà không có cảnh báo nào. Verify bằng mock thật: train
   `ANTIALIASING=1 ITERATIONS=15000` (tạo `iteration_7000`+`15000`, antialiasing=true),
   re-run `ANTIALIASING=0 ITERATIONS=7000` (chỉ ghi đè `iteration_7000`) — xác nhận
   `iteration_15000` còn nguyên nội dung `antialiasing=true` trong khi
   `pipeline_train_flags.json` đã đổi thành `antialiasing:false`. **Đã sửa**: thêm guard
   ngay đầu vòng lặp scene trong `02_train_baseline.sh` — nếu
   `$MODEL_DIR/point_cloud` đã có checkpoint từ trước, in `[BỎ QUA]` (không phá batch
   nhiều scene, giống cách xử lý thiếu `sparse/0`) + hướng dẫn rõ xoá thủ công hoặc set
   `CLEAN_MODEL_DIR=1` để tự xoá trước khi train lại — verify lại bằng mock: default
   block đúng (không đụng checkpoint cũ), `CLEAN_MODEL_DIR=1` xoá sạch + train lại đúng,
   batch nhiều scene (1 scene bị chặn) không ảnh hưởng scene khác vẫn train bình thường.
2. **`05_generate_error_mask.py` KHÔNG dọn `error_masks/` cũ trước khi ghi mask mới** —
   chỉ ghi ĐÈ đúng ảnh nằm trong lần chạy NÀY (theo tên file trùng stem); nếu lần trước
   có PHẠM VI khác (vd `--n_images` debug lấy mẫu 1 phần, hoặc lần trước CRASH giữa
   chừng trước khi xử lý hết ảnh — `manifest.json` chỉ ghi ở CUỐI hàm nên crash giữa
   chừng để lại mask MỚI trộn với mask CŨ mà KHÔNG cập nhật manifest), các mask `.png`
   CŨ (sinh với `--max_weight`/`--blur_radius`/iteration KHÁC) CÒN SÓT LẠI. Nguy hiểm vì
   đã đọc trực tiếp `apply_error_refine_patch.py::_error_mask()` xác nhận: hàm đó tra
   mask theo ĐÚNG TÊN FILE (`stem + ".png"`) cho MỌI ảnh train, KHÔNG đối chiếu với
   `manifest.json` để biết ảnh nào thuộc lần sinh mask nào — 1 mask CŨ tồn tại vẫn được
   dùng ÂM THẦM, trộn lẫn trọng số của 2 cấu hình khác nhau vào cùng 1 lượt refine mà
   không có cảnh báo gì (khác việc chỉ "gây nhiễu khi xem tay" như suy đoán ban đầu —
   ảnh hưởng THẬT tới train). Verify bằng test logic thật (dựng thư mục giả 3 mask +
   manifest CŨ, chạy đúng đoạn code dọn dẹp, xác nhận mask KHÔNG nằm trong lần chạy mới
   bị xoá sạch, mask CÓ trong lần chạy mới giữ nội dung mới). **Đã sửa**: thêm bước dọn
   `*.png` + `manifest.json` cũ trong `out_dir` ngay đầu `main()` (trước khi xử lý ảnh
   nào) — đảm bảo `error_masks/` LUÔN phản ánh ĐÚNG VÀ CHỈ đúng 1 lần chạy gần nhất; nếu
   crash giữa chừng, thư mục chỉ có 1 phần mask MỚI + KHÔNG CÓ `manifest.json` (thay vì
   mask MỚI + manifest.json CŨ) — `06_train_refine.sh` tự phát hiện thiếu manifest và
   `[BỎ QUA]` rõ ràng thay vì âm thầm dùng nhầm manifest cũ.
3. **`gdown --fuzzy` KHÔNG CÒN TỒN TẠI ở gdown ≥ 6.0.0 (bản mới nhất PyPI hiện tại,
   6.1.0)** — đây là bug NGHIÊM TRỌNG NHẤT tìm được ở pass này: cả 4 notebook luôn
   `!pip install -q ... gdown` KHÔNG PIN version, nên MỖI LẦN chạy fresh Kaggle session
   sẽ tự động lấy bản mới nhất. Verify bằng THỰC THI THẬT (không suy đoán, đúng yêu cầu
   nhiệm vụ): cài `gdown` mới nhất (6.1.0) trong venv sạch, chạy đúng lệnh notebook dùng
   (`gdown --fuzzy "<url>" -O ...` và `gdown --fuzzy --folder "<url>" -O ...`) — CẢ 2
   đều lỗi CLI NGAY LẬP TỨC: `gdown: error: unrecognized arguments: --fuzzy` (exit 2,
   trước khi kịp tải bất kỳ thứ gì). Đối chiếu ngược `gdown==5.2.2` xác nhận `--fuzzy`
   THẬT SỰ tồn tại ở bản cũ (`(file only) extract Google Drive's file ID`) — bị
   `gdown==6.0.0` xoá hẳn (không phải lỗi gõ nhầm ở đây, không phải version cache lạ).
   Tin TỐT xác nhận cùng lúc bằng thực thi thật: `gdown.parse_url.parse_url()` (hàm lõi
   dùng để suy ra file ID từ URL) ở bản 6.1.0 vẫn tự nhận diện ĐÚNG file ID từ URL dạng
   share-link đầy đủ (`.../file/d/<id>/view?usp=...`) MÀ KHÔNG CẦN cờ `--fuzzy` nào —
   hành vi "fuzzy" đã trở thành MẶC ĐỊNH LUÔN BẬT ở bản mới, cờ bị xoá vì hết cần thiết
   (không phải vì tính năng bị xoá). Tức là: pipeline vẫn tải được dữ liệu bình thường
   nếu bỏ hẳn cờ `--fuzzy` — không cần pin version cũ (rủi ro version cũ hết được
   support/security patch), chỉ cần bỏ cờ đã lỗi thời. Đã verify lại bằng thực thi thật
   sau khi sửa: `gdown --json "<GDRIVE_URL thật>"` (không `--fuzzy`) trả về đúng
   `"path": "VAI_NVS_DATA_ROUND2.zip"` — xác nhận vẫn resolve đúng file thật, không phải
   chỉ hết lỗi cú pháp suông. **Đã sửa**: bỏ `--fuzzy` khỏi TẤT CẢ lệnh `gdown` (dataset +
   checkpoint) ở cả 4 notebook (`kaggle_round1_baseline.ipynb`,
   `kaggle_round2_refine.ipynb`, `kaggle_round3_refine.ipynb`, `kaggle_submission.ipynb`)
   + 1 dòng comment/markdown còn nhắc `--fuzzy` trong `kaggle_submission.ipynb`.
   Bài học: cờ CLI/API của 1 tool bên thứ 3 KHÔNG cố định qua thời gian dù pipeline
   không đổi gì — `!pip install -q ... <tool>` KHÔNG pin version nghĩa là hành vi có thể
   đổi ÂM THẦM giữa lần verify (viết code) và lần chạy thật (Kaggle session sau này),
   không phải rủi ro lý thuyết suông — đã xảy ra thật với đúng gdown, đúng flag dự án
   đang dùng, ngay tại thời điểm viết milestone log này. Nên coi bất kỳ dependency nào
   cài KHÔNG PIN version trong notebook là rủi ro cần re-verify định kỳ (không chỉ 1 lần
   lúc viết), đặc biệt trước mỗi lần chạy Kaggle thật quan trọng gần deadline.

Ngoài 3 bug trên, đã fact-check 3 tuyên bố tài liệu load-bearing khác đối chiếu trực
tiếp code hiện tại (không chỉ tin milestone log cũ):
- Công thức `Score = 0.4*(1-LPIPS) + 0.3*SSIM + 0.3*PSNR_norm` +
  `PSNR_max` mặc định `50.0` — đối chiếu `04_eval_metrics.py::compute_score()` +
  `--psnr_max` default: khớp CHÍNH XÁC. Đúng.
- Commit pin `54c035f7834b564019656c3e3fcc3646292f727d` — grep byte-for-byte TOÀN repo
  (`.sh`, `.py`, `.ipynb`, `docs/`): khớp nhau ở MỌI nơi xuất hiện. Đúng, không có
  leftover/lệch.
- Cảnh báo "bug scale COLMAP" (`points2D.xy` lệch resolution, mục 1) — grep TOÀN
  `pipeline/*.py` cho `points2D`/`.xy`/`project_point`: KHÔNG có kết quả nào — hiện
  KHÔNG có script nào trong repo NÀY thực sự đọc `points2D.xy` trực tiếp (khác repo tiền
  nhiệm, nơi antenna-focus dùng nó để dựng mask khung 3D — kỹ thuật này đã bị loại khỏi
  Vòng 1 của repo này, xem mục 2). Không phải tuyên bố SAI (vẫn đúng như kiến thức
  phòng ngừa), nhưng hiện là kiến thức "ngủ đông" — không có code nào đang thực sự cần
  áp dụng nó. Ghi rõ ở đây để pass sau không tưởng nhầm đây là bug đang active cần fix.

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
