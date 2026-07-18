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

## 6f. Bug MỚI tìm ra ở verification pass #6 (`docs/MILESTONE_09_verification_pass6.md`)

Pass này bị ngắt giữa chừng do hết session quota (không phải phát hiện lỗi khiến dừng)
— agent gốc kịp pin `gdown` thành `"gdown>=6,<7"` (phòng gdown đổi API tiếp trước
deadline, không chỉ vá đúng phiên bản hiện tại). Điều phối viên hoàn thiện phần còn lại:

- Verify thật bằng `gdown --help` (bản 6.1.0): `--fuzzy` không tồn tại ở CẢ 2 chế độ
  (file đơn lẫn `--folder`) — xác nhận fix của pass #5 đã ĐỦ, không cần sửa thêm cho
  `--folder`.
- Thêm version floor còn thiếu cho `pycolmap` (`>=3.10`, khớp `requirements.txt` đã
  kiểm chứng của repo tiền nhiệm — API `pycolmap` từng đổi giữa các bản).
- **Bug thật độc lập tìm thêm**: `05_generate_error_mask.py` dùng `import cv2` nhưng
  KHÔNG notebook nào cài `opencv-python` — verify bằng cách đọc thật dòng `!pip install`
  ở cả 4 notebook (không có) + test `import cv2` trong môi trường sạch
  (`ModuleNotFoundError`). Không giả định Kaggle có cài sẵn hay không (dù nhiều khả
  năng CÓ) — đã thêm tường minh `opencv-python-headless` vào cả 4 notebook.
- Fresh clone thật `graphdeco-inria/gaussian-splatting` hôm nay (không dùng lại clone
  cũ), áp `apply_error_refine_patch.py` — sạch. Xác nhận pin theo COMMIT HASH (khác pin
  theo version pip) miễn nhiễm hoàn toàn với upstream đổi code — không cần lo hướng
  rủi ro này thêm.
- Fact-check claim "timeout 600 giây" (Đề_bài.md) — kết luận hợp lý: nhiều khả năng là
  hạ tầng chấm điểm phía BTC (sau khi nộp `submission.zip` qua portal), không áp dụng
  cho notebook train/render của thí sinh. `docs/00_MASTER_PLAN.md` đã ghi đúng mức thận
  trọng cần thiết, không cần sửa code.

## 6g. Bug MỚI tìm ra ở verification pass #7 (`docs/MILESTONE_10_verification_pass7.md`)

Pass này audit toàn bộ import/dependency (không tìm thêm gì — cả 4 notebook đã nhất
quán cài đủ `opencv-python-headless`/`pycolmap>=3.10` từ pass #6), re-audit các fix cũ
(sạch), rồi đào sâu 1 góc CHƯA pass nào làm: đối chiếu YÊU CẦU (không chỉ CODE) giữa
`kaggle_round1_baseline.ipynb` (MODE=`"holdout"`/`"final"`) và
`kaggle_round2_refine.ipynb`/`kaggle_round3_refine.ipynb` (yêu cầu checkpoint Vòng
trước phải ở MODE nào) — tìm ra **1 bug thật NGHIÊM TRỌNG, dạng "tài liệu tự mâu thuẫn
+ thiếu chặn kỹ thuật", làm hỏng chính cơ chế an toàn cốt lõi của kiến trúc multi-round**
(mục "chỉ giữ vòng nếu Score tăng THẬT", `docs/00_MASTER_PLAN.md` mục 3.2 bước 4).

**Mô tả bug — rò rỉ dữ liệu (data leakage) nếu dùng checkpoint `MODE="final"` làm input
Vòng 2+:**

- `kaggle_round1_baseline.ipynb` Bước 5 (markdown, trước khi sửa) ghi: `MODE="final"`
  "**Đây là checkpoint Vòng 1 dùng làm điểm khởi đầu cho Vòng 2+**" — và Bước 6 ghi:
  nếu `MODE="holdout"` thì "**KHÔNG dùng làm input Vòng 2+**". Tức là tài liệu Vòng 1
  khẳng định: dùng `"final"` (100% ảnh train, 30000 iteration) làm input Vòng 2+, KHÔNG
  dùng `"holdout"`.
- NHƯNG `kaggle_round2_refine.ipynb`/`kaggle_round3_refine.ipynb` (cell đầu, markdown,
  không đổi từ lúc viết ở milestone 02) lại ghi NGƯỢC LẠI: "**Yêu cầu: checkpoint Vòng 1
  đã train ở `MODE="holdout"`** (cần GT holdout để tự đo Score)".
- **2 tài liệu MÂU THUẪN TRỰC TIẾP nhau** — không ai trong 6 pass trước đối chiếu chéo
  đúng câu chữ "MODE nào làm input Vòng 2+" giữa 2 cặp notebook (dù đã đối chiếu rất kỹ
  nhiều thứ khác: schema JSON, CLI contract, tên file...).
- **Xác định bên nào đúng bằng cách đọc CODE THẬT (không suy đoán)**: `kaggle_round2_
  refine.ipynb` Bước 7 (cell tái tạo `colmap/dense/`) LUÔN gọi
  `00_make_holdout_split.py` (nếu `holdout/` chưa tồn tại trong phiên Kaggle hiện tại —
  luôn đúng vì Vòng 2+ chạy ở phiên Kaggle MỚI, tách biệt hoàn toàn khỏi phiên Vòng 1)
  rồi `01_run_colmap.py --holdout` — tạo lại **ĐÚNG 1 tập holdout cố định** (
  `00_make_holdout_split.py` dùng `seed=42` cố định + thuật toán chọn index cách đều
  theo tên đã sort — xác nhận đọc trực tiếp `choose_holdout_names()`: hoàn toàn
  deterministic, KHÔNG phụ thuộc gì vào lịch sử phiên trước). Bước 8 (đo Score TRƯỚC)
  render checkpoint VỪA NẠP trên đúng `holdout_poses.csv` này để tính Score baseline.
  **Nếu checkpoint vừa nạp là `MODE="final"`** (train trên 100% ảnh, bao gồm CHÍNH các
  ảnh mà `00_make_holdout_split.py` sẽ chọn làm holdout ở phiên Vòng 2 này — vì cùng
  scene + cùng seed=42 luôn cho cùng 1 tập ảnh) **thì model ĐÃ "thấy" các ảnh holdout đó
  lúc train `final`** — Score TRƯỚC đo được là **RÒ RỈ DỮ LIỆU** (đo trên ảnh model đã
  học thuộc, không phải đo generalization thật). Ngược lại, checkpoint `MODE="holdout"`
  (train CHỈ trên phần loại-trừ-holdout, cùng seed=42 nên cùng đúng tập ảnh) chưa từng
  "thấy" các ảnh holdout — Score TRƯỚC đo được SẠCH, đúng ý nghĩa.
- **Hậu quả THẬT (không phải lý thuyết suông)**: nếu 1 người dùng làm đúng theo hướng
  dẫn CŨ của `kaggle_round1_baseline.ipynb` (dùng `"final"` làm input Vòng 2+, đúng như
  văn bản khuyến nghị lúc đó), Score TRƯỚC sẽ bị thổi phồng giả tạo (model đã học thuộc
  ảnh "test" nội bộ) — refine 1 đợt NGẮN (chỉ chỉnh trên phần ảnh KHÔNG-holdout, mask lỗi
  đo trên ảnh train) rất có thể làm Score SAU đo thấp hơn Score TRƯỚC giả tạo đó (dù
  chất lượng generalization THẬT có thể đã tăng) → notebook kết luận SAI "KHÔNG cải
  thiện, GIỮ checkpoint Vòng 1, DỪNG LẠI" — **âm thầm loại bỏ 1 cải tiến thật có ích**,
  không có lỗi/crash nào báo hiệu có gì bất thường. Đây đúng loại lỗi nguy hiểm nhất
  theo tinh thần `docs/00_MASTER_PLAN.md` mục 3.2 bước 4 ("KHÔNG tin bằng trực giác,
  luôn đo Score thật") — chính phép ĐO lại là thứ bị hỏng, không phải thuật toán refine.
- **KHÔNG phải lỗi crash/exception** — trước khi sửa, không có bất kỳ cơ chế nào (code
  hay cảnh báo) phát hiện việc nạp nhầm checkpoint `MODE="final"` vào Vòng 2+; mọi thứ
  "chạy được" bình thường, chỉ có Ý NGHĨA của con số Score bị hỏng âm thầm.

**Đã sửa (thay đổi tối thiểu, chỉ thêm — không đổi hành vi cũ khi field mới vắng mặt):**

1. `pipeline/scripts/02_train_baseline.sh` — thêm biến môi trường `TRAIN_MODE`
   (`"holdout"`/`"final"`/để trống), ghi thêm field `"train_mode"` vào
   `pipeline_train_flags.json` (giá trị JSON string hoặc `null` nếu không truyền) —
   THUẦN GHI CHÚ, không ảnh hưởng logic train nào (giống cách 3 field
   `depth_prior`/`exposure_comp`/`antenna_focus` cũ đã làm).
2. `pipeline/kaggle_round1_baseline.ipynb` — Bước 5 (cell train): thêm dòng
   `os.environ["TRAIN_MODE"] = MODE` trước khi gọi `02_train_baseline.sh`. Sửa lại
   NGÔN TỪ Bước 5/Bước 6 (markdown) cho ĐÚNG kỹ thuật: `MODE="holdout"` PHẢI dùng làm
   input Vòng 2+ nếu định refine tiếp (KHÔNG phải `"final"` như bản cũ ghi sai);
   `MODE="final"` chỉ dùng để NỘP BÀI TRỰC TIẾP nếu KHÔNG chạy thêm Vòng 2+.
3. `pipeline/kaggle_round2_refine.ipynb`/`kaggle_round3_refine.ipynb` — Bước 6 (cell tải
   checkpoint): thêm chặn cứng đọc lại `pipeline_train_flags.json["train_mode"]` — nếu
   `"final"`, `raise SystemExit` với thông báo rõ nguyên nhân + hướng khắc phục (dùng
   đúng checkpoint `MODE="holdout"`); nếu field vắng mặt (checkpoint train TRƯỚC khi có
   field này, hoặc train trực tiếp CLI không qua notebook), in `[CẢNH BÁO]` rõ ràng
   (không chặn, để giữ tương thích ngược) thay vì im lặng hoàn toàn.
4. `docs/00_MASTER_PLAN.md` mục 3.2 bước 1 — thêm ghi chú BẮT BUỘC checkpoint input
   Vòng 2+ phải là `MODE="holdout"`.

**Verify**: `bash -n`/`py_compile`/`nbformat.validate()` sạch sau sửa; chạy thật
`02_train_baseline.sh` với `train.py` giả qua cả 2 trường hợp `TRAIN_MODE=holdout` và
không set `TRAIN_MODE` — xác nhận `pipeline_train_flags.json` ghi đúng
`"train_mode": "holdout"` / `"train_mode": null` tương ứng; test logic 3 nhánh chặn
(`"final"` → raise, `None` → cảnh báo tiếp tục, `"holdout"` → im lặng qua) bằng đoạn
code trích y hệt từ notebook. Test suite `tests/test_syntax_all.py` (15/15 + 4/4) +
`tests/test_07_package_submission.py` (21/21) vẫn PASS 100%, không regression.

**Giới hạn còn lại (ghi rõ, không giấu — đây là đánh đổi THIẾT KẾ, không phải bug, chưa
sửa vì ngoài phạm vi "fix an toàn" của 1 pass verify):** kiến trúc hiện tại buộc: muốn
Vòng 2+ (refine) có phép đo Score hợp lệ, checkpoint carry-forward xuyên suốt MỌI vòng
sau đó vĩnh viễn chỉ dựa trên ~87.5% ảnh train (phần loại-trừ-holdout) + khởi đầu từ
15000 iteration (không phải 30000 của `"final"`) — chưa có cơ chế tự động "nâng cấp" 1
cấu hình refine đã validate-bằng-holdout lên bản `MODE="final"` 100% dữ liệu để nộp bài
(vd train lại `"final"` từ đầu rồi áp lại đúng số `REFINE_ITERATIONS`/`MAX_ERROR_WEIGHT`
đã biết là có lợi, KHÔNG cần đo lại Score). Đây là hạn chế đã biết của kiến trúc, cần
quyết định của người dùng (đánh đổi ít dữ liệu hơn nhưng có validate được, hay nhiều dữ
liệu hơn nhưng không refine/không tự validate) — không tự ý mở rộng sửa thêm ở pass này.

Ngoài bug trên, đã audit exhaustive **toàn bộ import/dependency** của mọi
`pipeline/scripts/*.py` + `pipeline/common/*.py` đối chiếu với `!pip install` của cả 4
notebook (theo đúng yêu cầu — bug class `cv2`/`opencv-python` pass #6 tìm được có lặp
lại chỗ khác không): `pycolmap`/`scikit-image`/`lpips`/`cv2` đều đã được cài đủ và nhất
quán ở cả 4 notebook (`opencv-python-headless`, `pycolmap>=3.10` từ fix pass #6) —
KHÔNG tìm thêm instance thứ 2 của bug class này. `plyfile`/`tqdm` được cài nhưng KHÔNG
được import trực tiếp bởi bất kỳ script nào trong repo này — xác nhận đây là dependency
của CHÍNH `train.py`/`scene/gaussian_model.py` (repo `graphdeco-inria/gaussian-splatting`
ngoài), cài đúng theo nguyên tắc "superset" đã áp dụng sẵn — không phải thiếu sót.

## 6h. Bug MỚI tìm ra ở verification pass #8 (`docs/MILESTONE_11_verification_pass8.md`)

Pass này đào sâu re-audit fix `train_mode` của pass #7 (mục 6g) từ góc "guard có đủ lớp
phòng thủ không, hay chỉ có 1 lớp dễ bị bypass" + đọc lại toàn bộ tài liệu 1 lượt liền
mạch tìm chỗ 2 pass khác nhau "đá nhau" — tìm ra **2 vấn đề thật**:

1. **`pipeline/scripts/06_train_refine.sh` không tự kiểm tra `train_mode` — chặn cứng
   pass #7 thêm CHỈ tồn tại ở cell Python "Bước 6" của notebook** (chạy 1 LẦN lúc tải
   checkpoint từ Drive), không có ở bất kỳ đâu trong chính script `.sh` (dù script này
   đã đọc `pipeline_train_flags.json` ngay trong nó để lấy `antialiasing`). Verify bằng
   thực thi thật: gọi thẳng `06_train_refine.sh` (bỏ qua notebook hoàn toàn) trên 1
   checkpoint có `"train_mode": "final"` — script chạy xong, exit 0, không cảnh báo gì.
   Kịch bản thật: chạy lại 1 cell riêng lẻ ngoài thứ tự trên Kaggle (không bắt buộc
   "Run All"), dùng terminal Kaggle gọi thẳng script, hoặc copy dòng lệnh bash ra cell
   khác để debug mà quên chạy lại cell "Bước 6" (guard) trước đó cùng phiên — guard ở
   notebook không có cơ hội chạy, rò rỉ dữ liệu (mục 6g) vẫn xảy ra ÂM THẦM. Đáng chú ý:
   comment do CHÍNH pass #7 viết trong `02_train_baseline.sh` (biến `TRAIN_MODE`) đã
   tuyên bố "Vòng 2+ (`06_train_refine.sh`/`kaggle_round2_refine.ipynb`) tự phát hiện +
   CHẶN CỨNG" — tuyên bố SAI ở phần `06_train_refine.sh` tại thời điểm viết (chỉ đúng
   cho notebook), pass #7 overstate implementation thật. **Đã sửa**: thêm đọc
   `train_mode` vào đúng đoạn Python heredoc đã đọc `antialiasing`/`sh_degree` trong
   `06_train_refine.sh`, lặp lại đúng 3 nhánh xử lý y hệt cell Python của notebook
   (`"final"` → `[LỖI]` + `exit 1`, trừ khi `ALLOW_FINAL_TRAIN_MODE=1` — escape hatch
   theo đúng pattern có sẵn trong repo như `CLEAN_MODEL_DIR=1`; vắng mặt → `[CẢNH BÁO]`
   không chặn; `"holdout"`/khác → im lặng qua). Verify bằng thực thi thật 4 kịch bản
   (final-chặn/final+override/vắng mặt-cảnh báo/holdout-im lặng) — cả 4 đúng thiết kế.
2. **`kaggle_round1_baseline.ipynb` cell 21 (Bước 6) tự mâu thuẫn giữa câu chủ đề (pass
   #7 đã sửa đúng) và đoạn "Quy trình đầy đủ cho MỖI scene" ngay bên dưới trong CÙNG 1
   cell (pass #7 KHÔNG chạm tới)** — `git show 99ed04d` xác nhận pass #7 chỉ sửa câu chủ
   đề đầu cell (từ "Chỉ làm bước này khi MODE='final'." SAI thành có thêm điều kiện
   "...VÀ bạn KHÔNG định chạy thêm Vòng 2+" + giải thích đúng holdout mới là input Vòng
   2+), nhưng đoạn "Quy trình đầy đủ" 4 dòng ngay dưới đó vẫn giữ NGUYÊN VĂN CŨ: "Chạy 1
   version cuối với `MODE="final"` — đây là checkpoint Vòng 1 chính thức, làm **input
   cho Vòng 2+**..." — **chính xác hướng dẫn nguy hiểm mà toàn bộ fix pass #7 nhằm loại
   bỏ**, vẫn còn nguyên trong cùng 1 cell với đoạn văn vừa sửa đúng ngay phía trên. Hậu
   quả kỹ thuật đã được giảm nhẹ SẴN bởi Bug #1 (guard ở cell 16 notebook Vòng 2+ vẫn
   chặn được nếu người dùng làm theo hướng dẫn sai này) nhưng vẫn tốn thời gian/quota
   Kaggle oan uổng (train 30000 iteration `MODE="final"` rồi mới bị chặn) — đúng loại
   "tài liệu tự mâu thuẫn gây nhầm lẫn người dùng thật" cần audit theo tinh thần dự án.
   **Đã sửa**: viết lại cell 21 — câu chủ đề thành 2 gạch đầu dòng tách rõ mục đích
   TỪNG `MODE` (đều "làm bước này", chỉ khác lý do/đích đến), sửa "Quy trình đầy đủ"
   bước 2 cho khớp (`MODE="final"` — KHÔNG dùng làm input Vòng 2+, chỉ nộp bài trực
   tiếp) + thêm 1 dòng ở bước 1 nhắc tải checkpoint `holdout` lên Drive nếu định chạy
   Vòng 2+. Grep lại toàn bộ 4 notebook + `docs/*.md` sau khi sửa — không còn instance
   thứ 3 nào của claim sai này.

Bài học: 1 fix "chặn cứng" chỉ đặt ở ĐÚNG 1 lớp (notebook) trong khi logic thật thực thi
ở 1 lớp KHÁC (shell script được gọi từ notebook nhưng cũng gọi được độc lập) là chưa đủ
— nếu lớp ngoài (notebook) bị bypass (thứ tự cell không tuyến tính, chạy lại 1 phần, gọi
trực tiếp), lớp trong không có gì tự bảo vệ. Cũng: khi 1 fix chỉ sửa ĐÚNG phần văn bản bị
trích dẫn/nêu tên trong report (ở đây là "câu chủ đề đầu cell"), phải tự hỏi "còn đoạn
văn bản NÀO KHÁC trong CÙNG file/cell nói cùng 1 điều, có thể đã bị bỏ sót không" — không
chỉ tin đã sửa xong sau khi sửa đúng đoạn được trích dẫn đầu tiên tìm thấy.

## 6i. Bug MỚI tìm ra + xác nhận thật (không mock) ở verification pass #9 (`docs/MILESTONE_12_verification_pass9.md`)

- **`docs/00_MASTER_PLAN.md` mục 3.1 trỏ SAI "mục 6" (Triết lý test) thay vì "mục 2"
  (Train) của chính file này** khi dẫn nguồn cho tuyên bố "`--antialiasing` BẬT, không
  depth-prior/antenna-focus/exposure-comp — đã đo KHÔNG cải thiện Score". Nội dung đo
  đạc thật (Score `HCM0421` depth-prior 0.644 vs 0.6616, antenna-focus 0.6611 vs 0.6616)
  nằm ở mục 2, không phải mục 6 (mục 6 là quy tắc kiểm thử chung, không liên quan). Lỗi
  có từ commit khởi tạo repo (`920519f`), không đổi qua bất kỳ commit nào sau đó — 8 pass
  verify trước đều đọc `00_MASTER_PLAN.md` theo yêu cầu STEP 0 nhưng không ai bấm vào
  kiểm tra "mục 6" có thật chứa nội dung được trỏ tới hay không. Không ảnh hưởng code/
  hành vi (thuần tham chiếu tài liệu) nhưng vi phạm đúng lời hứa "nguồn sự thật duy nhất"
  ở đầu chính file đó — người đọc theo đúng chỉ dẫn sẽ mở nhầm mục, có thể nghi ngờ nhầm
  tuyên bố chưa được kiểm chứng. **Đã sửa**: đổi thành "mục 2".
- **Lần đầu chạy THẬT (không mock) `00_make_holdout_split.py`/`01_run_colmap.py` bằng
  `pycolmap` cài thật (bản `4.1.1`, cài trong vài giây, không cần venv nặng) + dữ liệu
  COLMAP tổng hợp thật qua `pycolmap.synthesize_dataset()`** — hạng mục bị liệt "còn
  thiếu, cần `pycolmap` cài thật" liên tục từ pass #1 (`MILESTONE_04`) tới pass #8
  (`MILESTONE_11`), 8 pass liền không ai thử cài. Kết quả: **KHÔNG tìm thấy bug** — xác
  nhận lại bằng dữ liệu `pycolmap.Reconstruction` thật (không phải chỉ đọc code):
  - Thứ tự đảo quaternion `[x,y,z,w] -> qw,qx,qy,qz` trong `00_make_holdout_split.py`
    đúng chính xác (so trực tiếp `image.cam_from_world().rotation.quat` gốc với dòng CSV
    tương ứng — khớp).
  - `01_run_colmap.py` (qua `common/colmap_runner.py::use_provided_sparse()`) chạy sạch,
    dùng thuần `pycolmap.undistort_images()` (đúng docstring "không cần binary `colmap`
    CLI riêng"), tự loại đúng ảnh holdout thiếu file khỏi reconstruction trước khi
    undistort (`_find_missing_images()`/`deregister_frame()` hoạt động đúng).
  - **Câu hỏi phụ tự đặt ra khi thấy `pycolmap` 4.1.1 ghi thêm `rigs.bin`/`frames.bin`**
    (định dạng rig mới, KHÔNG có trong sparse COLMAP cổ điển 3 file mà
    `graphdeco-inria/gaussian-splatting` mong đợi, và nhiều khả năng BTC cũng cung cấp
    dạng cổ điển vì `has_valid_provided_sparse()` chỉ kiểm tra `cameras.bin`): verify
    bằng thực thi thật — xoá `rigs.bin`/`frames.bin` khỏi 1 bản copy, `pycolmap.
    Reconstruction()` vẫn đọc đúng đủ ảnh/camera/điểm 3D, không lỗi. Xác nhận KHÔNG có
    rủi ro tương thích ngược giữa `pycolmap` bản mới và sparse định dạng cổ điển.
  - Giới hạn còn lại: scene tổng hợp qua `synthesize_dataset()` đơn giản hơn nhiều so
    với 7 scene thật (1 camera, quỹ đạo tổng hợp, không méo ống kính) — vẫn chưa thay
    thế hoàn toàn 1 lần chạy với chính dataset thật của cuộc thi, nhưng là bước tiến so
    với "chỉ đọc code" đã lặp lại ở 8 pass trước.

## 6j. Kiến thức MỚI xác nhận thật ở verification pass #10 (`docs/MILESTONE_13_verification_pass10.md`) — KHÔNG phải bug

Pass này mở rộng real-data coverage của pass #9 sang `05_generate_error_mask.py`
(phần không cần CUDA) và `03_render_test_poses.py`→`07_package_submission.py`, cả
2 luồng chạy thật bằng `pycolmap.synthesize_dataset()` + fake `GS_REPO` (code thật
không-CUDA từ commit pin + stub tối thiểu rasterizer/`GaussianModel`) — **KHÔNG tìm
thấy bug nào** trong cả 2 luồng (percentile/weight mask math + 16-bit PNG round-trip
đúng với ảnh thật có cấu trúc không gian biết trước; `check_scene()` bắt đúng lệch
1 pixel với `PIL.Image` thật). `pipeline/common/alignment.py` cũng lần đầu được chạy
thật (không chỉ đọc) — xác nhận đúng số học (Umeyama khôi phục đúng scale/rotation/
translation từ dữ liệu tổng hợp có đáp án biết trước, sai số ~1e-5), vẫn là dead
code (0 import trong repo Round 2 hiện tại).

**Phát hiện phụ về hành vi `pycolmap` (KHÔNG phải bug, chỉ là kiến thức mới cần
biết để không hoảng khi gặp)**: `pycolmap.undistort_images()` (bản 4.1.1) có 1 "fast
path" — nếu camera `SIMPLE_RADIAL` có tham số méo `k` đúng **bằng 0.0 tuyệt đối**
(bit-exact), hàm chỉ COPY ảnh nguyên vẹn và **GIỮ NGUYÊN model `SIMPLE_RADIAL`**
(không convert sang `PINHOLE` như hành vi thông thường/như docstring
`colmap_runner.py` mô tả). Verify bằng thực thi thật quét `k ∈ {0.0, 0.00001, 0.0001,
0.001, 0.01}`: CHỈ đúng `k=0.0` tuyệt đối mới giữ nguyên `SIMPLE_RADIAL`, mọi giá trị
khác (kể cả `0.00001`) đều convert đúng sang `PINHOLE` như bình thường. Nếu tình
huống này xảy ra thật, `05_generate_error_mask.py::load_train_poses()` (chỉ chấp
nhận `SIMPLE_PINHOLE`/`PINHOLE`) sẽ `raise ValueError` rõ ràng (fail loudly, không
phải lỗi âm thầm) — và đối chiếu trực tiếp `scene/dataset_readers.py` gốc của
`graphdeco-inria/gaussian-splatting` (dòng ~88-98, `assert False, "Colmap camera
model not handled: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras)
supported!"`) xác nhận **chính `train.py` chính thống cũng crash y hệt** nếu gặp
tình huống này — không phải rủi ro riêng của script tự viết trong repo này. Về khả
năng xảy ra thật: `k` là kết quả bundle-adjustment (tối ưu số thực liên tục) từ ảnh
chụp thật — xác suất hội tụ về đúng bit `0.0` tuyệt đối với ảnh drone/camera thật là
gần như 0 (khác dữ liệu tổng hợp nơi có thể cố ý đặt `k=0.0`, đây chính xác là lỗi
mà pass này tự vấp phải lúc đầu khi dựng fixture, trước khi nhận ra và sửa
`camera_params` test cho khớp). **Kết luận: không cần sửa code gì** — ghi lại ở đây
để pass sau không tưởng nhầm đây là bug đang active cần vá.

## 6. Triết lý test — áp dụng cho MỌI code mới ở repo này

> **Ghi chú đánh số (thêm ở verification pass #9, xem `docs/MILESTONE_12_verification_
> pass9.md` Phần E):** mục "6" này (triết lý test) là mục GỐC từ bản đầu tiên của file,
> viết trước khi có bug nào được thêm — các mục "6b"–"6i" ở TRÊN (vật lý đứng trước mục
> "6" này) là bug MỚI tìm theo từng verification pass, thêm theo đúng thứ tự thời gian
> (quy ước append-only, không đổi số mục cũ để khỏi phải sửa lại mọi chỗ đã trích dẫn
> "mục 6g"/"mục 6h"...). Đã cân nhắc đổi số mục này thành "7" cho gọn thứ tự đọc nhưng
> QUYẾT ĐỊNH KHÔNG đổi ở pass #9 (yêu cầu nhiệm vụ: không tổ chức lại file này) — ghi chú
> lại đây để pass đọc sau không thấy khó hiểu vì "6" xuất hiện sau "6h"/"6i" mà không có
> "6a".

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
