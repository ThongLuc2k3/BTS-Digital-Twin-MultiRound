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

## Test còn THIẾU (cần làm sau khi merge đủ 3 nhánh agent)

Xem chi tiết đầy đủ + lý do ở `docs/MILESTONE_03_submission_and_testing.md` mục
"Bước tiếp theo". Tóm tắt nhanh:

- Test patch `train.py` thật cho `apply_error_refine_patch.py` (thuộc agent refine,
  không phải phạm vi agent này — kiểm tra xem agent đó đã tự test chưa).
- Test mock end-to-end cho `02_train_baseline.sh`/`06_train_refine.sh` bằng `train.py`
  giả (mock) — thuộc agent Round-1/refine.
- Test tích hợp CHẠY THẬT `03_render_test_poses.py` (cần GPU, không test được cục
  bộ) — chỉ verify được bằng cách đọc code (đã làm, xem milestone doc) cho tới khi có
  dịp chạy thật trên Kaggle.
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
