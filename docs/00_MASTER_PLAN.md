# Master Plan — BTS Digital Twin, pipeline train nhiều vòng (multi-round refine)

> File này là nguồn sự thật duy nhất về kiến trúc + lý do thiết kế. Đọc file này
> TRƯỚC khi đọc bất kỳ milestone log nào khác. Nếu bị ngắt ngang giữa chừng, đọc lại
> file này + `docs/MILESTONE_*.md` mới nhất để biết đang ở đâu, KHÔNG đoán.

## 0. Bối cảnh — vì sao có repo này

Repo `BTS-Digital-Twin` (repo gốc, cùng tác giả) đã làm gần xong pipeline train 3DGS
chuẩn (1 lần train/scene). User đề xuất ý tưởng mới: **train nhiều vòng liên tiếp**
(vòng 1 ra baseline, vòng 2 tinh chỉnh, vòng 3 tinh chỉnh lần cuối), mỗi vòng là 1
notebook Kaggle riêng chạy tay tuần tự. User yêu cầu tạo **repo mới hoàn toàn** để làm
sạch kiến trúc (không kế thừa code cũ chồng chất nhiều thử nghiệm), nhưng **KHÔNG** yêu
cầu viết lại engine 3D Gaussian Splatting từ đầu — vẫn dùng
`graphdeco-inria/gaussian-splatting` làm engine train/render (baseline chính thức BTC
gợi ý, xem mục 1 dưới).

**Quyết định quan trọng đã thống nhất với user:** không xây từ số 0 tuyệt đối — PORT
lại các thành phần đã kiểm chứng thật (không phải đoán) từ `BTS-Digital-Twin`, đặc biệt
cơ chế "error-guided refine" (mục 3.2) đã build + test xong ngày 2026-07-18 — vì đúng
là cơ chế train-tinh-chỉnh-lặp-lại mà user mô tả. Xem `docs/PORTED_KNOWLEDGE.md` để
biết chính xác bug nào đã tìm ra + sửa ở repo cũ, KHÔNG được lặp lại ở repo này.

## 1. Đề bài — tóm tắt bắt buộc phải bám sát

Nguồn: `/home/thongluc/Khóa Luận Tốt Nghiệp/BTS Digital Twin/Đề_bài.md` (đọc lại bản
gốc nếu nghi ngờ, đây chỉ là tóm tắt).

- **Cuộc thi:** Viettel AI Race 2026, Bài 1 — BTS Digital Twin (Novel View Synthesis).
  3 vòng: Vòng 1 Sơ loại (02/07–**30/07/2026**), Vòng 2 Sơ khảo (17–19/08/2026), Vòng 3
  Chung kết (09–10/09/2026). **Đang làm cho Vòng 1 — DEADLINE 30/07/2026** (~12 ngày kể
  từ lúc viết file này, 2026-07-18).
- **Input mỗi scene:** `train/images/` + `train/sparse/0/{cameras,images,points3D}.bin`
  (COLMAP dựng sẵn, BTC cấp) + `test/test_poses.csv`
  (`image_name,qw,qx,qy,qz,tx,ty,tz,fx,fy,cx,cy,width,height`).
- **Output:** ảnh RGB đúng kích thước/tên cho mọi pose trong `test_poses.csv`, đóng gói
  `submission.zip` (cấu trúc `scene_xxx/000x.png`).
- **Điểm:** `Score = 0.4*(1-LPIPS) + 0.3*SSIM + 0.3*PSNR_norm`,
  `PSNR_norm = clamp(PSNR/PSNR_max, 0, 1)` — **PSNR_max không được BTC công bố cụ thể ở
  đề, phải tự ước lượng/tham khảo** (repo cũ dùng 50.0, xem `docs/PORTED_KNOWLEDGE.md`).
  Điểm bảng xếp hạng = **trung bình Score các scene** — thiếu/thừa scene so GT thì KHÔNG
  được tính điểm (bắt buộc nộp ĐỦ, ĐÚNG tên mọi scene/ảnh).
- **Baseline BTC gợi ý:** `graphdeco-inria/gaussian-splatting` — xác nhận hướng đi hiện
  tại (dùng đúng lib này) là đúng baseline chính thức, không phải tự chọn liều.
- **Chống gian lận (mục 10, BẮT BUỘC tuân thủ):**
  - Không dùng dữ liệu ngoài đề (không tự thu thập thêm ảnh/3D của scene đó).
  - Không truy xuất/suy đoán ground-truth test.
  - **Phải tái lập được kết quả** nếu BTC yêu cầu (mã nguồn train/infer, config, danh
    sách thư viện+phiên bản, checkpoint, training log) — đây là lý do bắt buộc phải
    ghi milestone log đầy đủ + code sạch, không phải chỉ để phòng ngắt kết nối.
  - **Không sửa tay ảnh output** — toàn bộ phải sinh tự động bởi model. Train nhiều
    vòng/tinh chỉnh loss theo vùng lỗi đo được **VẪN HỢP LỆ** (vẫn là thuật toán tự
    động, không phải sửa tay từng ảnh/pose) — đã cân nhắc kỹ trước khi chọn hướng này.
  - Giới hạn nộp: 5 lần/ngày, timeout 600s/job (không rõ áp dụng notebook train hay chỉ
    lúc chấm — giả định thận trọng: pipeline sinh ảnh test cho 1 scene nên xong trong
    vài phút, không phải vài giờ như train).

## 2. Dataset hiện có

7 scene: 5 scene BTS (`HCM0421`, `HCM0539`, `HCM0540`, `HCM0644`, `HCM0674`) + 2 scene
tổng quát (`bonsai`, `chair`) — xem link Google Drive + cấu trúc chi tiết ở
`docs/PORTED_KNOWLEDGE.md` mục dataset. Ảnh train KHÔNG đồng nhất scale với sparse gốc ở
vài scene (xem bug scale mục đó) — PHẢI tự đo scale runtime, không hardcode.

## 3. Kiến trúc — train nhiều vòng (multi-round)

### 3.1. Vòng 1 — Baseline

Train 3DGS chuẩn 1 lần/scene bằng `graphdeco-inria/gaussian-splatting` (commit pin
`54c035f7834b564019656c3e3fcc3646292f727d` — bản đã xác nhận có `--antialiasing`
(mip-splatting tích hợp sẵn), `--depths` (depth reg), `--train_test_exp` (exposure)).
Cấu hình mặc định (đã đo thật ở repo cũ, xem mục 6 `PORTED_KNOWLEDGE.md`):
`--antialiasing` BẬT, không depth-prior, không antenna-focus, không exposure-comp —
các cờ đó đã đo KHÔNG cải thiện Score đo được trên holdout thật (HCM0421), giữ baseline
đơn giản là lựa chọn tốt nhất đã kiểm chứng.

Output: `pipeline/work/<scene>/gs_model/` (`cfg_args`, `pipeline_train_flags.json`,
`point_cloud/iteration_<N>/point_cloud.ply`).

### 3.2. Vòng 2, 3, ... — Error-guided refine (lặp lại N lần)

Cơ chế (đã build + verify cục bộ ở repo cũ ngày 2026-07-18, port nguyên vẹn — xem
`docs/PORTED_KNOWLEDGE.md` mục "error-guided refine" để biết đúng cạm bẫy đã tránh):

1. Nạp checkpoint vòng trước (`.ply`, KHÔNG cần `.pth` — dùng
   `Scene(..., load_iteration=N)` của repo gốc, kèm fix `spatial_lr_scale` bắt buộc).
2. Render lại CHÍNH pose ảnh train, so với GT thật (luôn có, khác test) → đo lỗi
   pixel → sinh mask trọng số theo percentile (vùng lỗi cao được ưu tiên loss).
3. Vá `train.py` (`apply_error_refine_patch.py`) để nhận mask, train tiếp NGẮN
   (`--densify_until_iter 0` — không sinh thêm Gaussian, chỉ tinh chỉnh cái có sẵn).
4. Tự đo Score holdout TRƯỚC/SAU mỗi vòng — **CHỈ giữ lại vòng đó nếu Score tăng**,
   không tin bằng trực giác (bài học từ repo cũ: depth-prior/antenna-focus đo thật
   không tăng dù trực giác nghĩ sẽ tăng).

Mỗi vòng = 1 notebook riêng (`kaggle_round1_baseline.ipynb`,
`kaggle_round2_refine.ipynb`, `kaggle_round3_refine.ipynb`, ...) — user tự chạy tay
tuần tự trên Kaggle, tải checkpoint vòng N lên Drive, dán link cho vòng N+1.

### 3.3. Đóng gói & nộp

`kaggle_submission.ipynb` — tải checkpoint vòng cuối cùng (đã chọn) của cả 7 scene,
render `test_poses.csv` thật, đóng gói `submission.zip` đúng format mục 1.

## 4. Kỷ luật tài liệu hoá (bắt buộc, theo đúng yêu cầu tái lập của đề bài mục 10.3)

- `docs/00_MASTER_PLAN.md` (file này) — kiến trúc + lý do, ít đổi.
- `docs/PORTED_KNOWLEDGE.md` — danh sách bug/bài học đã tìm ra ở repo cũ, KHÔNG được
  lặp lại. Cập nhật nếu tìm thêm bug MỚI ở chính repo này.
- `docs/MILESTONE_<số>_<tên>.md` — 1 file/hạng mục công việc lớn (vd
  `MILESTONE_01_port_common.md`, `MILESTONE_02_round1_pipeline.md`,
  `MILESTONE_03_refine_pipeline.md`, `MILESTONE_04_submission_and_testing.md`). Mỗi
  file PHẢI có mục "Trạng thái hiện tại" + "Bước tiếp theo" luôn cập nhật, và "Lịch sử"
  ghi timeline (giống quy tắc `WORKLOG.md` ở repo cũ) — để bất kỳ ai (người hoặc agent
  khác) đọc vào giữa chừng đều biết chính xác đã làm gì, đang làm gì, làm tiếp gì.
- `README.md` — tổng quan ngắn cho người ngoài, trỏ vào `docs/00_MASTER_PLAN.md`.

## 5. Phân công (2026-07-18, khởi tạo)

- Agent (tôi, điều phối): tạo skeleton repo, port `pipeline/common/` (nền tảng dùng
  chung), viết plan này + `PORTED_KNOWLEDGE.md`.
- Sub-agent A: port + refactor pipeline Vòng 1 (baseline) — scripts + notebook.
- Sub-agent B: port + tổng quát hoá pipeline Vòng 2+ (error-guided refine, phải chạy
  lặp lại được N vòng, không hardcode "vòng 2" và "vòng 3" là 2 code path khác nhau) —
  scripts + 2 notebook mẫu (round2, round3).
- Sub-agent C: port submission packaging + xây bộ test cục bộ (mock, giống cách repo
  cũ đã làm — test cú pháp/patch/logic không cần GPU) cho TOÀN BỘ script mới.

Xem chi tiết từng agent ở milestone log tương ứng.
