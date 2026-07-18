# Milestone 02 — Pipeline Vòng 2+ (error-guided refine, lặp lại nhiều vòng)

## Trạng thái hiện tại
**HOÀN TẤT.** Toàn bộ cơ chế "error-guided refine" đã port + tổng quát hoá thành 1
script dùng lại được cho MỌI vòng refine (không phải 2 code path riêng cho vòng 2/3),
cộng 2 notebook Kaggle (`kaggle_round2_refine.ipynb`, `kaggle_round3_refine.ipynb`),
đã test kỹ cục bộ (mock, vì không có GPU) và tìm+sửa được **2 bug thật mới** (không có
ở bản gốc `kaggle_error_refine.ipynb` của repo tiền nhiệm — xem "Lịch sử").

## Bước tiếp theo
Không còn việc bắt buộc cho milestone này. Việc còn lại (không chặn):
- Chưa test thật trên GPU (chỉ mock cục bộ) — cần 1 lần chạy Kaggle thật.
- Đối chiếu chéo với milestone 01 (Round 1)/03 (submission) đã làm ở vòng kiểm tra
  chéo cấp điều phối (xem `docs/MILESTONE_00_setup.md`/lịch sử commit sau milestone
  này) — không cần lặp lại ở đây trừ khi phát hiện thêm gì mới.

## File đã tạo
- `pipeline/scripts/apply_error_refine_patch.py` — vá `train.py`: thêm
  `--refine_from_iteration`/`--error_mask_dir`, tự sửa `spatial_lr_scale` (bug quan
  trọng nếu không sửa: Gaussian đứng yên hoàn toàn suốt refine mà không báo lỗi — xem
  `docs/PORTED_KNOWLEDGE.md` mục 2), hàm `_error_mask()` đọc mask 16-bit PNG. KHÔNG
  dùng chung được với antenna-focus patch (đụng cùng điểm vá "Loss") — tự chặn nếu
  phát hiện `train.py` đã vá antenna-focus trước.
- `pipeline/scripts/05_generate_error_mask.py` — render lại pose TRAIN, so GT thật,
  sinh mask 16-bit PNG theo percentile lỗi (giống hệt công thức đã verify ở repo tiền
  nhiệm, đối chiếu `diff` xác nhận không lệch 1 dòng nào ở phần công thức cốt lõi).
- `pipeline/scripts/06_train_refine.sh` — **MỚI HOÀN TOÀN** (không có ở repo tiền
  nhiệm, nơi cơ chế này lần đầu viết dạng Python inline trong notebook). Tự dò
  checkpoint nguồn mới nhất, đối chiếu `error_masks/manifest.json` (chặn cứng nếu
  iteration lệch — mask đo lỗi ở trạng thái Gaussian khác sẽ sai mục đích), tự đọc lại
  `sh_degree`/`antialiasing` từ checkpoint nguồn (không bắt gõ tay, tránh lệch cấu
  hình), chạy `train.py` đã vá, rồi **đổi tên thư mục output thành số iteration LUỸ KẾ
  thật** (xem "Quyết định thiết kế quan trọng" trong chính file — bug tiềm ẩn tự phát
  hiện: `train.py` không `--start_checkpoint` luôn đếm lại từ 0 và lưu tại
  `iteration_<REFINE_ITERATIONS>`, nên 2 vòng refine liên tiếp dùng cùng
  `REFINE_ITERATIONS` mặc định sẽ ghi ra thư mục TÊN GIỐNG HỆT NHAU nếu không đổi tên
  — vi phạm yêu cầu tái lập kết quả, Đề bài mục 10.3).
- `pipeline/kaggle_round2_refine.ipynb`, `pipeline/kaggle_round3_refine.ipynb` — cùng
  1 cơ chế, chỉ khác chú thích "vòng mấy tinh chỉnh vòng mấy". Tự đo Score holdout
  TRƯỚC/SAU, in rõ kết luận nên giữ hay bỏ kết quả vòng đó.
- File này.

## Bug thật tìm ra + sửa (khi test cục bộ bằng mock, KHÔNG suy đoán)

1. **`_latest_iteration_dir()` vỡ vì đường dẫn dự án có dấu cách** ("Khóa Luận Tốt
   Nghiệp") — bản gốc ghép `"$n $d"` bằng `printf` rồi lấy lại qua
   `sort | tail -1 | awk '{print $2}'`; vì `awk` tách trường theo khoảng trắng, nếu
   CHÍNH `$d` (đường dẫn) chứa dấu cách thì `awk '{print $2}'` chỉ in ra 1 TỪ đầu tiên
   của đường dẫn, cắt cụt phần còn lại — ra đường dẫn RÁC mà không có lỗi cú pháp nào,
   chỉ lộ ra khi file theo đường dẫn cụt đó không tồn tại (`[BỎ QUA] ... không thấy
   .../iteration_/home/thongluc/Khóa/point_cloud.ply`). Verify bằng test thật (không
   mock phần này — dùng chính đường dẫn dự án thật). Sửa: bỏ hẳn pipe/sort/awk trên
   chuỗi có thể chứa dấu cách, thay bằng vòng lặp bash thuần + so sánh số nguyên
   (`(( 10#$n > best_n ))`), không tách trường theo khoảng trắng ở đâu cả.
2. **Thiếu chặn va chạm thư mục output thô** — phát hiện khi thiết kế lại test (dùng
   `REFINE_ITERATIONS` trùng số với `CKPT_ITERATION` để test cố ý): nếu
   `point_cloud/iteration_$REFINE_ITERATIONS` đã tồn tại TRƯỚC khi `train.py` chạy (vd
   trùng số với 1 checkpoint có sẵn, hoặc sót lại từ lần chạy dở dang), `train.py` sẽ
   ghi đè/pha trộn nó mà không biết/không báo lỗi — bug im lặng, có thể MẤT checkpoint
   thật. Đã thêm chặn cứng TRƯỚC khi gọi `train.py`, không tự động ghi đè.

## Test đã chạy (mock, không có GPU — theo đúng triết lý `docs/PORTED_KNOWLEDGE.md` mục 6)

- `python -m py_compile` + `bash -n` toàn bộ file mới — sạch.
- `nbformat.validate()` cả 2 notebook — hợp lệ (sau khi sửa thêm 1 lỗi nhỏ: thiếu tiền
  tố `f` ở 1 câu `print(...)` khiến `{TARGET_ITERATION}` in ra chữ thay vì giá trị —
  tự phát hiện khi rà lại code trước khi test, đã sửa cả 2 notebook).
- `apply_error_refine_patch.py` áp thật lên bản clone sạch
  `graphdeco-inria/gaussian-splatting` (commit đã pin) — 6/6 chỗ vá thành công,
  `python -m py_compile` sau vá không lỗi cú pháp.
- `06_train_refine.sh` test bằng `train.py` giả (in tiến độ, ghi `.ply`/`cfg_args` giả,
  hỗ trợ biến `MOCK_CRASH=<iteration>` để mô phỏng crash giữa chừng) qua 6 kịch bản:
  happy path không va chạm (verify CẢ 2 thư mục `iteration_100` gốc và `iteration_150`
  mới đều còn nguyên), happy path CÓ va chạm cố ý (verify chặn đúng, verify checkpoint
  nguồn KHÔNG bị đụng vào — đọc lại nội dung file `.ply` xác nhận không đổi), crash
  giữa chừng (verify thông báo cứu hộ + exit code đúng), thiếu `error_masks/
  manifest.json` (verify tự bỏ qua scene đó, không dừng cả loop), `manifest.json` lệch
  iteration (verify chặn cứng đúng thông báo), `GS_REPO` chưa vá (verify chặn cứng
  đúng thông báo). Dọn dẹp toàn bộ thư mục test tạm sau khi xong (`pipeline/work/
  _test_refine/`, không để lại trong repo).
- Đối chiếu `05_generate_error_mask.py` bằng `diff` trực tiếp với công thức gốc ở repo
  tiền nhiệm (`pipeline/scripts/12_generate_error_mask.py`) — khớp 100%, không lệch.

## Lịch sử

### 2026-07-18 — Sub-agent build phần lớn, bị ngắt giữa chừng do hết session quota
- Sub-agent (chạy song song với milestone 01/03) đọc `docs/00_MASTER_PLAN.md` +
  `docs/PORTED_KNOWLEDGE.md`, port `apply_error_refine_patch.py` +
  `05_generate_error_mask.py` từ repo tiền nhiệm, **tự thiết kế mới hoàn toàn**
  `06_train_refine.sh` (không có bản gốc để port — cơ chế cũ chỉ tồn tại dạng Python
  inline trong notebook) với chất lượng kỹ thuật cao (tự phát hiện + tự sửa vấn đề đặt
  tên thư mục trùng lặp giữa các vòng refine liên tiếp, việc mà bản thiết kế gốc CHƯA
  giải quyết). Đã commit các file `.py`/`.sh` (chưa commit notebook/milestone doc) thì
  bị dừng giữa chừng do hết hạn mức phiên làm việc (session limit, không phải lỗi kỹ
  thuật) — để lại 3 file `apply_error_refine_patch.py`/`05_generate_error_mask.py`/
  `06_train_refine.sh` ở trạng thái untracked (chưa `git add`), thiếu 2 notebook +
  milestone doc.
- Điều phối viên (tôi) tiếp tục: verify lại 3 file sub-agent để lại (patch áp sạch lên
  clone thật, `py_compile` sạch, đối chiếu công thức mask khớp bản gốc), viết 2
  notebook `kaggle_round2_refine.ipynb`/`kaggle_round3_refine.ipynb` (tái dùng
  boilerplate đã proven từ `kaggle_round1_baseline.ipynb`), rồi tự test toàn bộ
  `06_train_refine.sh` bằng mock — tìm ra **2 bug thật mới** nêu trên (bug đường dẫn có
  dấu cách đặc biệt nghiêm trọng vì đúng ngay tên thư mục dự án thật), sửa xong, test
  lại xác nhận cả 2 đều hết.
