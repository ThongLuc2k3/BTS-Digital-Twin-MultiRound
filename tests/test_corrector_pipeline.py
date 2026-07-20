#!/usr/bin/env python3
"""Test cục bộ (không cần GPU/GS_REPO) cho nhánh Model 2 — mạng sửa lỗi pixel 2D
HOÀN TOÀN RIÊNG BIỆT với Model 1 (3D Gaussian Splatting), xem
`docs/MILESTONE_15_image_corrector.md`: `pipeline/common/corrector_model.py`,
`pipeline/scripts/09_train_corrector.py`, `pipeline/scripts/10_apply_corrector.py`.

`08_build_corrector_dataset.py` KHÔNG test được chức năng ở đây (giống hệt
`05_generate_error_mask.py` gốc) — cần GS_REPO thật (clone
graphdeco-inria/gaussian-splatting đã build) + GPU CUDA để render; chỉ test được
py_compile + guard thiếu GS_REPO. Test chức năng thật (render ra cặp ảnh đúng) phải làm
trên máy có GPU (xem "Bước tiếp theo" trong milestone doc) — không suy đoán thay.

Cách chạy:
    python3 tests/test_corrector_pipeline.py

Test bao gồm:
  1. Cú pháp: py_compile 4 file mới + nbformat.validate() notebook mới.
  2. ResidualCorrectorNet: output giữ nguyên shape input (mọi H,W, kể cả không chia hết
     cho gì — xác nhận đúng tính chất fully-convolutional), và tại lúc khởi tạo (tail
     conv zero-init CÓ CHỦ ĐÍCH) output == input y hệt (residual = 0) — xác nhận đúng
     thuộc tính "an toàn" đã thiết kế (corrector chưa train/train hỏng không tự sinh
     nhiễu, chỉ trả về nguyên render Model 1).
  3. Checkpoint save/load round-trip (state_dict khớp lại đúng), và load_checkpoint()
     PHẢI báo lỗi rõ nếu kiến trúc (num_blocks/channels) không khớp model đang tạo.
  4. ssim(): ssim(x,x) ~= 1.0, khác ảnh thì < 1.0, sai shape phải raise.
  5. Ghép ô (tiled inference) ở 10_apply_corrector.py: với 1 "model" identity (trả về
     nguyên input), apply_tiled() PHẢI tái tạo lại ĐÚNG ảnh gốc (xác nhận không có pixel
     bị bỏ sót/tính trùng — bài test coverage, KHÔNG chứng minh trộn biên mượt vì mọi
     tile trả về đúng giá trị gốc bất kể trọng số). Trộn biên mượt được test riêng bằng
     thuộc tính toán học của ramp_weight() (tổng trọng số 2 ô liền kề ~= 1 ở đúng vùng
     chồng lấn danh nghĩa — partition of unity).
  6. 09_train_corrector.py chạy THẬT (subprocess, CPU, dataset giả rất nhỏ): train lần
     đầu tạo checkpoint; chạy lại KHÔNG cờ phải bị chặn (exit != 0); --resume phải tăng
     step lũy kế đúng; --overwrite phải reset về step nhỏ ban đầu (không cộng dồn).
  7. 08_build_corrector_dataset.py: guard thiếu biến môi trường GS_REPO báo lỗi rõ.
"""
import importlib.util
import json
import os
import py_compile
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
CORRECTOR_MODEL_PATH = REPO_ROOT / "pipeline" / "common" / "corrector_model.py"
SCRIPT_08 = REPO_ROOT / "pipeline" / "scripts" / "08_build_corrector_dataset.py"
SCRIPT_09 = REPO_ROOT / "pipeline" / "scripts" / "09_train_corrector.py"
SCRIPT_10 = REPO_ROOT / "pipeline" / "scripts" / "10_apply_corrector.py"
NOTEBOOK_PATH = REPO_ROOT / "pipeline" / "kaggle_pixel_corrector.ipynb"

_failures: list[str] = []
_n_pass = 0


def check(name: str, cond: bool, detail: str = ""):
    global _n_pass
    if cond:
        _n_pass += 1
        print(f"  [PASS] {name}")
    else:
        _failures.append(name)
        print(f"  [FAIL] {name} {('-- ' + detail) if detail else ''}")


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_syntax():
    print("== 1. Cú pháp ==")
    for p in (CORRECTOR_MODEL_PATH, SCRIPT_08, SCRIPT_09, SCRIPT_10):
        try:
            py_compile.compile(str(p), doraise=True)
            check(f"py_compile {p.name}", True)
        except py_compile.PyCompileError as e:
            check(f"py_compile {p.name}", False, str(e))
    try:
        import nbformat
        nb = nbformat.read(str(NOTEBOOK_PATH), as_version=4)
        nbformat.validate(nb)
        check("nbformat.validate kaggle_pixel_corrector.ipynb", True)
    except Exception as e:
        check("nbformat.validate kaggle_pixel_corrector.ipynb", False, str(e))


def test_network_architecture():
    print("== 2. ResidualCorrectorNet ==")
    mod = load_module(CORRECTOR_MODEL_PATH, "corrector_model_under_test")
    model = mod.build_corrector(num_blocks=3, channels=8)
    model.eval()

    for h, w in [(37, 53), (64, 64), (17, 129)]:
        x = torch.rand(1, 3, h, w)
        with torch.no_grad():
            y = model(x)
        check(f"output shape khớp input ({h}x{w})", tuple(y.shape) == tuple(x.shape),
              f"{tuple(y.shape)} != {tuple(x.shape)}")

    x = torch.rand(2, 3, 32, 32).clamp(0, 1)
    with torch.no_grad():
        y = model(x)
    check("tail zero-init -> output == input lúc khởi tạo (residual=0)",
          bool(torch.allclose(y, x, atol=1e-6)), f"max diff={float((y - x).abs().max())}")

    check("receptive_field_px tính đúng (1 + num_blocks*2)",
          model.receptive_field_px == 1 + 3 * 2)


def test_checkpoint_roundtrip():
    print("== 3. Checkpoint save/load ==")
    mod = load_module(CORRECTOR_MODEL_PATH, "corrector_model_under_test2")
    model = mod.build_corrector(num_blocks=2, channels=4)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    with tempfile.TemporaryDirectory() as td:
        ckpt = Path(td) / "corrector.pt"
        mod.save_checkpoint(ckpt, model, opt, step=123,
                             train_args={"patch_size": 16}, dataset_source_iteration=30000,
                             loss_history=[{"step": 100, "loss": 0.5}])
        check("checkpoint file được ghi ra", ckpt.exists())

        model2 = mod.build_corrector(num_blocks=2, channels=4)
        opt2 = torch.optim.Adam(model2.parameters(), lr=1e-3)
        payload = mod.load_checkpoint(ckpt, model2, opt2)
        check("step load lại đúng", payload["step"] == 123)
        check("dataset_source_iteration load lại đúng", payload["dataset_source_iteration"] == 30000)
        same = all(torch.equal(p1, p2) for p1, p2 in zip(model.parameters(), model2.parameters()))
        check("state_dict load lại khớp y hệt model gốc", same)

        model_wrong = mod.build_corrector(num_blocks=5, channels=4)
        try:
            mod.load_checkpoint(ckpt, model_wrong)
            check("load_checkpoint báo lỗi khi kiến trúc lệch (num_blocks)", False, "không raise")
        except ValueError:
            check("load_checkpoint báo lỗi khi kiến trúc lệch (num_blocks)", True)


def test_ssim():
    print("== 4. ssim() ==")
    mod = load_module(CORRECTOR_MODEL_PATH, "corrector_model_under_test3")
    torch.manual_seed(0)
    x = torch.rand(1, 3, 64, 64)
    s_same = float(mod.ssim(x, x))
    check("ssim(x,x) ~= 1.0", abs(s_same - 1.0) < 1e-4, f"got {s_same}")

    y = torch.rand(1, 3, 64, 64)
    s_diff = float(mod.ssim(x, y))
    check("ssim(x, y ngẫu nhiên khác) < ssim(x,x)", s_diff < s_same, f"{s_diff} vs {s_same}")

    try:
        mod.ssim(x, torch.rand(1, 3, 32, 32))
        check("ssim() báo lỗi khi shape lệch nhau", False, "không raise")
    except ValueError:
        check("ssim() báo lỗi khi shape lệch nhau", True)


class _IdentityModel(nn.Module):
    def forward(self, x):
        return x


def test_tiled_reconstruction():
    print("== 5. Ghép ô (tiled inference), 10_apply_corrector.py ==")
    mod = load_module(SCRIPT_10, "apply_corrector_under_test")

    rng = np.random.RandomState(0)
    img = rng.rand(97, 141, 3).astype(np.float32)
    identity = _IdentityModel()
    identity.eval()

    out = mod.apply_tiled(identity, img, tile_size=32, overlap=8, device="cpu")
    check("apply_tiled với model identity tái tạo đúng ảnh gốc (coverage, không thiếu pixel)",
          bool(np.allclose(out, img, atol=1e-5)),
          f"max diff={float(np.abs(out - img).max())}")

    starts = mod.compute_tile_starts(100, 32, 8)
    check("compute_tile_starts phủ hết từ 0 tới cuối", starts[0] == 0 and starts[-1] == 100 - 32)
    check("compute_tile_starts: ảnh nhỏ hơn tile -> 1 ô duy nhất",
          mod.compute_tile_starts(20, 32, 8) == [0])

    w_edge = mod.ramp_weight(32, 8, has_prev=False, has_next=True)
    check("ramp_weight: biên ẢNH THẬT (không có ô trước) giữ trọng số 1 ở đầu",
          float(w_edge[0]) == 1.0)
    w_mid = mod.ramp_weight(32, 8, has_prev=True, has_next=True)
    check("ramp_weight: có ô kề 2 bên -> ramp về 0 ở cả 2 đầu",
          float(w_mid[0]) == 0.0 and float(w_mid[-1]) == 0.0)

    overlap = 8
    w_a = mod.ramp_weight(32, overlap, has_prev=False, has_next=True)   # ô đầu
    w_b = mod.ramp_weight(32, overlap, has_prev=True, has_next=False)   # ô kề, cuối
    sum_overlap = w_a[-overlap:] + w_b[:overlap]
    check("ramp_weight: tổng trọng số 2 ô liền kề ~= 1 ở đúng vùng chồng lấn danh nghĩa "
          "(partition of unity, trường hợp overlap chuẩn)",
          bool(np.allclose(sum_overlap, 1.0, atol=1e-6)), f"{sum_overlap}")


def _make_fake_corrector_dataset(dataset_dir: Path, n_pairs: int = 3, size: int = 48):
    render_dir, gt_dir = dataset_dir / "render", dataset_dir / "gt"
    render_dir.mkdir(parents=True, exist_ok=True)
    gt_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.RandomState(1)
    for i in range(n_pairs):
        stem = f"img_{i:03d}"
        render_arr = rng.randint(0, 255, size=(size, size, 3)).astype(np.uint8)
        gt_arr = np.clip(render_arr.astype(int) + rng.randint(-10, 10, render_arr.shape), 0, 255).astype(np.uint8)
        Image.fromarray(render_arr).save(render_dir / f"{stem}.png")
        Image.fromarray(gt_arr).save(gt_dir / f"{stem}.png")
    manifest = {"scene": "chair", "source_iteration": 30000, "n_pairs": n_pairs}
    (dataset_dir / "manifest.json").write_text(json.dumps(manifest))


def run_09(args, extra_env=None):
    env = os.environ.copy()
    if extra_env:
        env.update(extra_env)
    return subprocess.run([sys.executable, str(SCRIPT_09)] + args, capture_output=True, text=True, env=env)


def test_train_resume_guard():
    print("== 6. 09_train_corrector.py (subprocess CPU thật, dataset giả rất nhỏ) ==")
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        dataset_dir = td / "corrector_dataset"
        out_dir = td / "corrector_model"
        _make_fake_corrector_dataset(dataset_dir, n_pairs=3, size=48)

        common_args = ["--scene", "chair", "--dataset_dir", str(dataset_dir), "--out_dir", str(out_dir),
                        "--patch_size", "16", "--batch_size", "2", "--num_blocks", "2", "--channels", "4",
                        "--num_workers", "0", "--log_every", "5", "--save_every", "5"]

        r1 = run_09(common_args + ["--steps", "5"])
        check("train lần đầu (không cờ) chạy thành công", r1.returncode == 0, (r1.stdout + r1.stderr)[-800:])
        ckpt = out_dir / "corrector.pt"
        check("checkpoint được tạo ra", ckpt.exists())
        if not ckpt.exists():
            return  # không thể test tiếp nếu bước đầu đã fail

        r2 = run_09(common_args + ["--steps", "5"])
        check("chạy lại KHÔNG --resume/--overwrite phải bị chặn (exit != 0)", r2.returncode != 0)

        payload_before = torch.load(ckpt, map_location="cpu", weights_only=False)
        r3 = run_09(common_args + ["--steps", "5", "--resume"])
        check("--resume chạy thành công", r3.returncode == 0, (r3.stdout + r3.stderr)[-800:])
        payload_after = torch.load(ckpt, map_location="cpu", weights_only=False)
        check("--resume cộng dồn đúng step (base+5)",
              payload_after["step"] == payload_before["step"] + 5,
              f"{payload_before['step']} -> {payload_after['step']}")

        r4 = run_09(common_args + ["--steps", "3", "--overwrite"])
        check("--overwrite chạy thành công", r4.returncode == 0, (r4.stdout + r4.stderr)[-800:])
        payload_reset = torch.load(ckpt, map_location="cpu", weights_only=False)
        check("--overwrite reset step về đúng số bước mới (3, không cộng dồn)",
              payload_reset["step"] == 3, f"got {payload_reset['step']}")


def test_08_gs_repo_guard():
    print("== 7. 08_build_corrector_dataset.py: guard thiếu GS_REPO ==")
    env = os.environ.copy()
    env.pop("GS_REPO", None)
    r = subprocess.run([sys.executable, str(SCRIPT_08), "--scene", "chair"],
                        capture_output=True, text=True, env=env)
    check("thiếu GS_REPO -> exit != 0 kèm thông báo rõ",
          r.returncode != 0 and "GS_REPO" in (r.stdout + r.stderr))


def main():
    for p in (CORRECTOR_MODEL_PATH, SCRIPT_08, SCRIPT_09, SCRIPT_10, NOTEBOOK_PATH):
        if not p.exists():
            print(f"[LỖI] Không tìm thấy {p}.")
            return 1

    test_syntax()
    test_network_architecture()
    test_checkpoint_roundtrip()
    test_ssim()
    test_tiled_reconstruction()
    test_train_resume_guard()
    test_08_gs_repo_guard()

    print()
    print(f"===== {_n_pass} PASS, {len(_failures)} FAIL =====")
    if _failures:
        print("Test FAIL:")
        for f in _failures:
            print(f"  - {f}")
        return 1
    print("TẤT CẢ TEST PASS.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
