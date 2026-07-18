#!/usr/bin/env python3
"""Sinh mask trọng số lỗi (error weight mask) cho cơ chế "tinh chỉnh có hướng dẫn bởi
lỗi" (error-guided refine — CỐT LÕI của repo này, xem `docs/00_MASTER_PLAN.md` mục 3.2
+ `docs/PORTED_KNOWLEDGE.md` mục 3): render lại CHÍNH các pose ảnh TRAIN bằng checkpoint
đã có, so với ảnh GT thật (có sẵn 100% cho mọi ảnh train, khác test/holdout), đo vùng
nào còn sai nhiều ("pixel nhiễu") rồi sinh mask 16-bit PNG dùng để tăng trọng số loss ở
đúng vùng đó khi train tiếp bằng `apply_error_refine_patch.py` + `train.py
--error_mask_dir` (`06_train_refine.sh` gọi cả 2 script này theo đúng thứ tự).

QUAN TRỌNG — chỉ đo được lỗi trên ảnh TRAIN (có GT thật), KHÔNG đo trực tiếp được trên
test/holdout (đề bài không cấp GT cho test thật). Giả định ngầm: vùng model tái tạo kém
trên ảnh train (chi tiết mảnh, motion blur, vùng khuất nhiều) nhiều khả năng cũng kém
tương tự ở pose lân cận trong test/holdout — CHƯA CÓ BẰNG CHỨNG THỰC NGHIỆM xác nhận
giả định này đúng mức nào, đây là lý do bắt buộc mỗi vòng refine (xem
`kaggle_round2_refine.ipynb`/`kaggle_round3_refine.ipynb`) phải tự đo Score holdout
trước/sau đợt tinh chỉnh trước khi tin dùng cho bản nộp thật.

Công thức mask (percentile-based, KHÔNG boost đều toàn ảnh — chỉ vùng lỗi THỰC SỰ cao):
  1. err(x,y) = trung bình |render - GT| trên 3 kênh màu, mỗi ảnh train.
  2. Làm mượt err bằng Gaussian blur (mặc định radius=4px) — lỗi thật (chi tiết mảnh,
     motion blur) thường lan trên 1 vùng liền mạch, không phải nhiễu ảnh JPEG rời rạc
     từng pixel; blur giúp mask không bị vụn/gián đoạn.
  3. weight = 1 + (MAX_WEIGHT-1) * clip((err - p_lo) / (p_hi - p_lo), 0, 1)
     — p_lo/p_hi là percentile của chính err trong ẢNH ĐÓ (mặc định p50/p95): pixel ở
     mức lỗi trung vị trở xuống -> weight=1 (không đổi), pixel ở top 5% lỗi -> weight=
     MAX_WEIGHT, chuyển tiếp mượt ở giữa.
  4. Lưu weight dạng 16-bit PNG: pixel_value = round(weight * 1000) — PHẢI dùng đúng
     hằng số 1000.0 này (`_ERROR_MASK_SCALE`, khớp hệt hằng số trong
     `apply_error_refine_patch.py`), 2 nơi KHÔNG được lệch nhau.

Cách dùng (cần GPU CUDA + GS_REPO — CHƯA cần vá bằng apply_error_refine_patch.py để
sinh mask, chỉ cần load được checkpoint; vá là bước SAU, lúc train tiếp):
    export GS_REPO=/path/to/gaussian-splatting
    python 05_generate_error_mask.py --scene HCM0421

Yêu cầu: đã có `pipeline/work/<scene>/colmap/dense/{images/,sparse/0/}` (chạy
`01_run_colmap.py --scene <scene> [--holdout]` trước — LƯU Ý nếu `03_train_3dgs.sh`
(hoặc `06_train_refine.sh` của vòng trước) đã tự dọn `dense/images/` (dọn đĩa mặc định)
thì phải chạy lại `01_run_colmap.py` để tái tạo ảnh đã undistort TRƯỚC khi chạy script
này — script này CẦN ảnh chính xác pixel-for-pixel, KHÔNG chấp nhận ảnh gốc chưa
undistort xấp xỉ như cách làm tạm cho sanity-check thông thường vì sai số ~1% do resize
sẽ lẫn vào chính error map muốn đo).

Output:
    pipeline/work/<scene>/error_masks/<stem>.png   (16-bit, 1 kênh, cùng tên ảnh gốc)
    pipeline/work/<scene>/error_masks/manifest.json (thông số đã dùng + lỗi trung bình
        trước tinh chỉnh, để `06_train_refine.sh` đối chiếu iteration/max_weight và để
        đối chiếu lại sau khi refine xong xem lỗi có giảm không)
"""
import argparse
import json
import sys
from argparse import Namespace
from pathlib import Path

import cv2
import numpy as np
import os
import torch
from PIL import Image as PILImage

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.scenes import get_scene
from common.poses import TestPose, pose_to_R_T_fov, assert_centered_principal_point

GS_REPO = os.environ.get("GS_REPO")
if not GS_REPO or not (Path(GS_REPO) / "train.py").exists():
    raise SystemExit(
        "Chưa set biến môi trường GS_REPO hoặc đường dẫn sai.\n"
        "  export GS_REPO=/path/to/gaussian-splatting\n"
        "(thư mục clone --recursive https://github.com/graphdeco-inria/gaussian-splatting)"
    )
sys.path.insert(0, GS_REPO)

from scene.cameras import MiniCam                              # noqa: E402
from scene.gaussian_model import GaussianModel                  # noqa: E402
from scene.colmap_loader import (                                # noqa: E402
    read_extrinsics_binary, read_intrinsics_binary,
    read_extrinsics_text, read_intrinsics_text,
)
from gaussian_renderer import render                             # noqa: E402
from utils.graphics_utils import getWorld2View2, getProjectionMatrix  # noqa: E402

_ERROR_MASK_SCALE = 1000.0  # PHẢI khớp _ERROR_MASK_SCALE trong apply_error_refine_patch.py


class _PipelineParamsStub:
    """Y hệt _PipelineParamsStub của 04_render_test_poses.py — render() chỉ đọc đúng
    4 field này."""
    convert_SHs_python = False
    compute_cov3D_python = False
    debug = False
    antialiasing = False


def read_cfg_args(model_dir: Path) -> dict:
    cfg_path = model_dir / "cfg_args"
    if not cfg_path.exists():
        return {}
    try:
        ns = eval(cfg_path.read_text(), {"Namespace": Namespace})
        return vars(ns)
    except Exception as e:
        print(f"[CẢNH BÁO] Không đọc/parse được {cfg_path}: {e} — dùng giá trị mặc định/CLI.")
        return {}


def read_pipeline_train_flags(model_dir: Path) -> dict:
    p = model_dir / "pipeline_train_flags.json"
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text())
    except Exception as e:
        print(f"[CẢNH BÁO] Không đọc/parse được {p}: {e} — bỏ qua.")
        return {}


def build_minicam(pose: TestPose, znear: float = 0.01, zfar: float = 100.0) -> MiniCam:
    """Y hệt 04_render_test_poses.py::build_minicam()."""
    R, T, FovX, FovY = pose_to_R_T_fov(pose)
    world_view_transform = torch.tensor(getWorld2View2(R, T)).transpose(0, 1).float().cuda()
    projection_matrix = getProjectionMatrix(
        znear=znear, zfar=zfar, fovX=FovX, fovY=FovY
    ).transpose(0, 1).float().cuda()
    full_proj_transform = (
        world_view_transform.unsqueeze(0).bmm(projection_matrix.unsqueeze(0))
    ).squeeze(0)
    return MiniCam(pose.width, pose.height, FovY, FovX, znear, zfar,
                    world_view_transform, full_proj_transform)


def find_latest_iteration(model_dir: Path) -> int:
    pc_dir = model_dir / "point_cloud"
    iters = [int(p.name.split("_")[-1]) for p in pc_dir.glob("iteration_*") if p.is_dir()]
    if not iters:
        raise FileNotFoundError(f"Không tìm thấy checkpoint nào trong {pc_dir}")
    return max(iters)


def load_train_poses(sparse_dir: Path) -> dict[str, TestPose]:
    """Y hệt 04_render_test_poses.py::load_train_poses() (đọc pose từ COLMAP sparse
    thay vì từ test_poses.csv)."""
    bin_images, bin_cameras = sparse_dir / "images.bin", sparse_dir / "cameras.bin"
    txt_images, txt_cameras = sparse_dir / "images.txt", sparse_dir / "cameras.txt"
    if bin_images.exists() and bin_cameras.exists():
        images = read_extrinsics_binary(str(bin_images))
        cameras = read_intrinsics_binary(str(bin_cameras))
    elif txt_images.exists() and txt_cameras.exists():
        images = read_extrinsics_text(str(txt_images))
        cameras = read_intrinsics_text(str(txt_cameras))
    else:
        raise FileNotFoundError(f"Không tìm thấy images.bin/.txt + cameras.bin/.txt trong {sparse_dir}")

    poses: dict[str, TestPose] = {}
    for img in images.values():
        cam = cameras[img.camera_id]
        if cam.model == "SIMPLE_PINHOLE":
            f, cx, cy = cam.params[:3]
            fx = fy = f
        elif cam.model == "PINHOLE":
            fx, fy, cx, cy = cam.params[:4]
        else:
            raise ValueError(f"{img.name}: camera model '{cam.model}' không được hỗ trợ.")
        poses[img.name] = TestPose(
            image_name=img.name,
            qvec=np.array(img.qvec, dtype=np.float64),
            tvec=np.array(img.tvec, dtype=np.float64),
            fx=float(fx), fy=float(fy), cx=float(cx), cy=float(cy),
            width=int(cam.width), height=int(cam.height),
        )
    return poses


def load_img01(path: Path) -> np.ndarray:
    return np.asarray(PILImage.open(path).convert("RGB")).astype(np.float32) / 255.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True)
    ap.add_argument("--model_dir", default=None, help="Mặc định pipeline/work/<scene>/gs_model")
    ap.add_argument("--iteration", type=int, default=-1, help="-1 = iteration lớn nhất có sẵn")
    ap.add_argument("--sparse_dir", default=None, help="Mặc định pipeline/work/<scene>/colmap/dense/sparse/0")
    ap.add_argument("--images_dir", default=None, help="Mặc định pipeline/work/<scene>/colmap/dense/images (ảnh ĐÃ undistort — bắt buộc chính xác, không dùng ảnh gốc)")
    ap.add_argument("--out_dir", default=None, help="Mặc định pipeline/work/<scene>/error_masks")
    ap.add_argument("--max_weight", type=float, default=6.0, help="Trọng số tối đa ở vùng lỗi cao nhất (mặc định 6x)")
    ap.add_argument("--error_percentile_lo", type=float, default=50.0, help="Percentile lỗi coi là 'bình thường' (weight=1)")
    ap.add_argument("--error_percentile_hi", type=float, default=95.0, help="Percentile lỗi coi là 'tệ nhất' (weight=max_weight)")
    ap.add_argument("--blur_radius", type=int, default=4, help="Bán kính Gaussian blur (px) làm mượt error map trước khi tính weight, 0 = tắt")
    ap.add_argument("--n_images", type=int, default=0, help="0 = dùng hết ảnh train, >0 = lấy mẫu đều n ảnh (debug nhanh)")
    ap.add_argument("--white_background", action="store_true")
    args = ap.parse_args()

    if args.error_percentile_hi <= args.error_percentile_lo:
        raise SystemExit("--error_percentile_hi phải > --error_percentile_lo")

    scene = get_scene(args.scene)
    pipeline_root = Path(__file__).resolve().parents[1]
    model_dir = Path(args.model_dir) if args.model_dir else pipeline_root / "work" / scene.name / "gs_model"
    sparse_dir = Path(args.sparse_dir) if args.sparse_dir else (
        pipeline_root / "work" / scene.name / "colmap" / "dense" / "sparse" / "0"
    )
    images_dir = Path(args.images_dir) if args.images_dir else (
        pipeline_root / "work" / scene.name / "colmap" / "dense" / "images"
    )
    out_dir = Path(args.out_dir) if args.out_dir else pipeline_root / "work" / scene.name / "error_masks"

    if not images_dir.exists():
        raise SystemExit(
            f"Không thấy {images_dir} (ảnh đã undistort) — nếu bước train trước đó đã tự dọn thư mục "
            f"này (dọn đĩa mặc định), chạy lại `01_run_colmap.py --scene {scene.name}"
            f"{' --holdout' if 'holdout' in str(model_dir) else ''}` để tái tạo TRƯỚC khi chạy script này."
        )

    iteration = args.iteration if args.iteration > 0 else find_latest_iteration(model_dir)
    ply_path = model_dir / "point_cloud" / f"iteration_{iteration}" / "point_cloud.ply"
    if not ply_path.exists():
        raise SystemExit(f"Không thấy {ply_path}.")

    cfg = read_cfg_args(model_dir)
    train_flags = read_pipeline_train_flags(model_dir)
    sh_degree = cfg.get("sh_degree", 3)
    if "antialiasing" in train_flags:
        antialiasing = bool(train_flags["antialiasing"])
    else:
        antialiasing = False
        print("  [CẢNH BÁO NGHIÊM TRỌNG] Không có pipeline_train_flags.json — giả định antialiasing=False, "
              "CÓ THỂ SAI (xem docs/PORTED_KNOWLEDGE.md mục 2).")
    print(f"  Dùng sh_degree={sh_degree}, antialiasing={antialiasing}, iteration={iteration}")

    gaussians = GaussianModel(sh_degree)
    gaussians.load_ply(str(ply_path))
    bg_color = [1, 1, 1] if args.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    pipe = _PipelineParamsStub()
    pipe.antialiasing = antialiasing

    all_poses = load_train_poses(sparse_dir)
    names_sorted = sorted(all_poses.keys())
    if not names_sorted:
        raise SystemExit(f"Không đọc được pose train nào từ {sparse_dir}")
    if args.n_images > 0:
        step = max(1, len(names_sorted) // args.n_images)
        names_sorted = names_sorted[::step][: args.n_images]

    out_dir.mkdir(parents=True, exist_ok=True)
    mean_errors = []
    n_saved, n_skipped = 0, 0
    print(f"===== {scene.name}: sinh error mask cho {len(names_sorted)} ảnh train =====")
    for name in names_sorted:
        gt_path = images_dir / name
        if not gt_path.exists():
            print(f"  [BỎ QUA] {name}: không thấy {gt_path}.")
            n_skipped += 1
            continue
        pose = all_poses[name]
        assert_centered_principal_point(pose)
        cam = build_minicam(pose)
        with torch.no_grad():
            out = render(cam, gaussians, pipe, background)
        pred = out["render"].clamp(0, 1).detach().cpu().numpy().transpose(1, 2, 0)
        gt = load_img01(gt_path)
        if gt.shape != pred.shape:
            print(f"  [BỎ QUA] {name}: kích thước khác nhau GT={gt.shape[:2]} render={pred.shape[:2]} "
                  f"(cả 2 đáng lẽ cùng size — kiểm tra lại images_dir có đúng ảnh đã undistort không).")
            n_skipped += 1
            continue

        err = np.abs(gt - pred).mean(axis=2).astype(np.float32)
        if args.blur_radius > 0:
            k = args.blur_radius * 2 + 1
            err = cv2.GaussianBlur(err, (k, k), 0)

        p_lo, p_hi = np.percentile(err, [args.error_percentile_lo, args.error_percentile_hi])
        if p_hi - p_lo < 1e-8:
            p_hi = p_lo + 1e-8
        weight = 1.0 + (args.max_weight - 1.0) * np.clip((err - p_lo) / (p_hi - p_lo), 0.0, 1.0)
        weight_u16 = np.round(weight * _ERROR_MASK_SCALE).astype(np.uint16)

        stem = Path(name).stem
        cv2.imwrite(str(out_dir / f"{stem}.png"), weight_u16)
        mean_errors.append(float(err.mean()))
        n_saved += 1
        if n_saved % 20 == 0:
            print(f"  [{n_saved}/{len(names_sorted)}] {name}: mean_err={err.mean():.4f} "
                  f"p{args.error_percentile_lo:.0f}={p_lo:.4f} p{args.error_percentile_hi:.0f}={p_hi:.4f}")

    if n_saved == 0:
        raise SystemExit("Không sinh được mask nào — xem lại images_dir/sparse_dir.")

    manifest = {
        "scene": scene.name,
        "iteration": iteration,
        "n_images": n_saved,
        "n_skipped": n_skipped,
        "max_weight": args.max_weight,
        "error_percentile_lo": args.error_percentile_lo,
        "error_percentile_hi": args.error_percentile_hi,
        "blur_radius": args.blur_radius,
        "error_mask_scale": _ERROR_MASK_SCALE,
        "mean_error_before_refine": float(np.mean(mean_errors)),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"\n-> Đã sinh {n_saved} mask (bỏ qua {n_skipped}) tại {out_dir}")
    print(f"-> Lỗi trung bình (|render-GT|, trước tinh chỉnh) = {np.mean(mean_errors):.4f}")
    print(f"-> manifest: {out_dir / 'manifest.json'}")


if __name__ == "__main__":
    main()
