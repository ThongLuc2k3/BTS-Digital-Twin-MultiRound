# Test cục bộ — không dùng pytest

Đúng triết lý test đã ghi ở `docs/PORTED_KNOWLEDGE.md` mục 6: repo này **không có
GPU cục bộ** (Kaggle mới có GPU), nên toàn bộ phần train/render thật CHỈ verify được
trên Kaggle. Ở đây chỉ test những gì test được KHÔNG CẦN GPU: cú pháp, logic thuần
Python/numpy, patch `train.py`, và mock end-to-end bằng dữ liệu giả — để bắt lỗi sớm,
tiết kiệm quota GPU Kaggle thay vì phát hiện lỗi giữa chừng lúc chạy thật.

Không dùng framework pytest — mỗi file `test_*.py` trong thư mục này là 1 **script độc
lập**, tự chạy được bằng `python3`, tự in `[PASS]`/`[FAIL]` từng bước, và tự trả về mã
thoát khác 0 nếu có bất kỳ test nào fail (để dùng được trong CI/script kiểm tra tự
động sau này nếu cần).

## Chạy toàn bộ

```bash
cd "BTS-Digital-Twin-MultiRound"
python3 tests/test_syntax_all.py            # cú pháp: mọi .py (py_compile) + mọi .ipynb (nbformat)
python3 tests/test_07_package_submission.py # logic + mock end-to-end cho 07_package_submission.py + kaggle_submission.ipynb
```

Yêu cầu cài đặt tối thiểu (không cần CUDA/pycolmap để chạy 2 file trên):
```bash
pip install nbformat pillow numpy
```

## Danh sách file test hiện có

| File | Phạm vi | Cần GPU? | Ghi chú |
|---|---|---|---|
| `test_syntax_all.py` | Toàn repo: `py_compile` mọi `.py`, `nbformat.validate()` mọi `.ipynb` | Không | Tự bỏ qua nếu 1 agent khác chưa đẩy hết file — báo "LƯU Ý" chứ không FAIL, để không chặn lẫn nhau khi 3 agent cùng làm song song |
| `test_07_package_submission.py` | `pipeline/scripts/07_package_submission.py` + `pipeline/kaggle_submission.ipynb` (2 file do agent packaging/testing sở hữu) | Không | Mock toàn bộ dataset (`test_poses.csv` giả) + renders (PNG giả) qua biến môi trường `BTS_DATASET_ROOT` + `--renders_root`, chạy script thật qua `subprocess`. Có test regression riêng cho bug thật đã tìm+sửa ở repo tiền nhiệm (đổi tên `.jpg` mà không mã hoá lại nội dung — xem `docs/PORTED_KNOWLEDGE.md` mục 5) |

## Test đã bổ sung sau khi merge đủ các nhánh agent (không còn nằm trong `tests/`)

Các mục dưới đây từng nằm trong danh sách "còn THIẾU" lúc `docs/MILESTONE_03_submission_and_testing.md`
được viết (khi 3 nhánh agent còn làm song song) — đã được các milestone/verification pass
SAU ĐÓ tự thực hiện bằng mock/test có mục tiêu, NHƯNG không phải dưới dạng file
`tests/test_*.py` cố định (không chạy lại tự động qua 2 lệnh ở mục "Chạy toàn bộ" phía
trên) — chỉ có kết quả ghi lại trong milestone log tương ứng:

- Test patch `train.py` thật cho `apply_error_refine_patch.py` — áp thật lên bản clone
  `graphdeco-inria/gaussian-splatting` tại đúng commit pin, `py_compile` sạch sau vá
  (`docs/MILESTONE_02_refine_pipeline.md`, re-verify độc lập ở
  `docs/MILESTONE_04_verification_pass1.md` Phần C).
- Test mock end-to-end cho `02_train_baseline.sh`/`06_train_refine.sh` bằng `train.py`
  giả (mock, hỗ trợ `CRASH_AT`/`MOCK_CRASH`) — `docs/MILESTONE_01_round1_baseline.md`,
  `docs/MILESTONE_02_refine_pipeline.md`; test THẬT với 2 scene trong 1 lệnh (multi-scene
  invocation) ở `docs/MILESTONE_06_verification_pass3.md`.
- Mô phỏng chuỗi Vòng1→Vòng2→Vòng3→render→package bằng 1 fake `GS_REPO` dùng code thật
  không-CUDA từ đúng commit pin (`docs/MILESTONE_04_verification_pass1.md` Phần C) —
  KHÔNG thay thế được 1 lần chạy GPU Kaggle thật (xem mục dưới).

## Test còn THIẾU (chỉ chạy được trên Kaggle thật, có GPU)

- Test tích hợp CHẠY THẬT `03_render_test_poses.py`/`train.py` trên GPU CUDA thật (chất
  lượng render/Score thật) — chỉ verify được bằng cách đọc code + mock rasterizer cục bộ
  cho tới khi có dịp chạy thật trên Kaggle.
- Test full "Run All" cả 4 notebook (`kaggle_round1_baseline.ipynb`,
  `kaggle_round2_refine.ipynb`, `kaggle_round3_refine.ipynb`,
  `kaggle_submission.ipynb`) nối tiếp nhau bằng checkpoint thật — chỉ làm được trên
  Kaggle thật, không mock được vì cần GPU + thời gian train thật.

## Ước lượng "test PASS nghĩa là gì" (tránh hiểu lầm)

`test_07_package_submission.py` PASS nghĩa là: logic đóng gói/kiểm tra
(`07_package_submission.py`) và luồng gọi lệnh trong notebook đúng khi INPUT đã đúng
định dạng (renders PNG đúng kích thước, `test_poses.csv` đúng schema). Nó **KHÔNG**
verify chất lượng ảnh render (PSNR/SSIM/LPIPS thật) hay việc `train.py`/
`03_render_test_poses.py` chạy đúng trên GPU thật — 2 việc đó chỉ verify được trên
Kaggle (xem `docs/00_MASTER_PLAN.md` mục "Kỷ luật tài liệu hoá" + `PORTED_KNOWLEDGE.md`
mục 6).
