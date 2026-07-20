#!/usr/bin/env python3
"""Sinh tập dữ liệu cặp (render, GT) từ checkpoint Model 1 (3D Gaussian Splatting) cho
Model 2 (`pipeline/common/corrector_model.py::ResidualCorrectorNet`) — mạng sửa lỗi
pixel HOÀN TOÀN RIÊNG BIỆT với Model 1, xem docstring `corrector_model.py` để biết vì
sao đây KHÔNG phải cùng cơ chế với "error-guided refine" (Vòng 2/3).

Render lại CHÍNH các pose ảnh TRAIN bằng checkpoint Model 1 đã có (khuyến nghị: checkpoint
train trên 100% dữ liệu, `MODE="final"` của `kaggle_round1_baseline.ipynb`), lưu cặp
(ảnh render, ảnh GT thật) ra đĩa — CHỈ CẦN CHẠY 1 LẦN cho mỗi checkpoint Model 1 (không
render lại mỗi epoch lúc train Model 2 — 09_train_corrector.py chỉ đọc PNG từ đĩa, thuần
`torch`, không cần GS_REPO/CUDA rasterizer nữa).

Vì sao COPY hẳn ảnh GT ra `out_dir/gt/` thay vì trỏ tham chiếu tới
`colmap/dense/images/`: thư mục đó bị `02_train_baseline.sh`/`06_train_refine.sh` tự XOÁ
sau khi train xong (dọn đĩa mặc định) — copy ra đây làm dataset TỰ ĐỦ (self-contained),
không phụ thuộc dense/images/ còn tồn tại hay không, và không cần GS_REPO/COLMAP nữa cho
các bước sau (09/10).

Ảnh GT dùng so sánh PHẢI là ảnh ĐÃ undistort chính xác pixel-for-pixel
(`colmap/dense/images/`, do `01_run_colmap.py` sinh ra) — giống hệt yêu cầu của
`05_generate_error_mask.py`, lý do y hệt: sai số resize/xấp xỉ sẽ lẫn vào chính tín hiệu
học của Model 2.

Cách dùng (cần GPU CUDA + GS_REPO, giống 05_generate_error_mask.py):
    export GS_REPO=/path/to/gaussian-splatting
    python 08_build_corrector_dataset.py --scene HCM0421

Output:
    pipeline/work/<scene>/corrector_dataset/render/<stem>.png   (ảnh Model 1 render tại pose train)
    pipeline/work/<scene>/corrector_dataset/gt/<stem>.png       (ảnh GT thật, copy PNG lossless)
    pipeline/work/<scene>/corrector_dataset/manifest.json       (thông số đã dùng, để 09/10 đối chiếu)
"""
import argparse
import json
import sys
from argparse import Namespace
from pathlib import Path

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


class _PipelineParamsStub:
    """Y hệt _PipelineParamsStub của 03_render_test_poses.py/05_generate_error_mask.py —
    render() chỉ đọc đúng 4 field này."""
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
    """Y hệt 03_render_test_poses.py/05_generate_error_mask.py::build_minicam()."""
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
    """Y hệt 05_generate_error_mask.py::load_train_poses() (đọc pose từ COLMAP sparse
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True)
    ap.add_argument("--model_dir", default=None, help="Mặc định pipeline/work/<scene>/gs_model")
    ap.add_argument("--iteration", type=int, default=-1, help="-1 = iteration lớn nhất có sẵn")
    ap.add_argument("--sparse_dir", default=None, help="Mặc định pipeline/work/<scene>/colmap/dense/sparse/0")
    ap.add_argument("--images_dir", default=None,
                     help="Mặc định pipeline/work/<scene>/colmap/dense/images (ảnh ĐÃ undistort — "
                          "bắt buộc chính xác, không dùng ảnh gốc train/images/ chưa undistort)")
    ap.add_argument("--out_dir", default=None, help="Mặc định pipeline/work/<scene>/corrector_dataset")
    ap.add_argument("--n_images", type=int, default=0, help="0 = dùng hết ảnh train, >0 = lấy mẫu đều n ảnh (debug nhanh)")
    ap.add_argument("--white_background", action="store_true")
    ap.add_argument("--force", action="store_true",
                     help="Sinh lại dataset dù manifest.json hiện có đã khớp checkpoint/tham số này "
                          "(mặc định: bỏ qua nếu đã khớp, tiết kiệm thời gian render lại khi chạy lại "
                          "notebook nhiều lần).")
    args = ap.parse_args()

    scene = get_scene(args.scene)
    pipeline_root = Path(__file__).resolve().parents[1]
    model_dir = Path(args.model_dir) if args.model_dir else pipeline_root / "work" / scene.name / "gs_model"
    sparse_dir = Path(args.sparse_dir) if args.sparse_dir else (
        pipeline_root / "work" / scene.name / "colmap" / "dense" / "sparse" / "0"
    )
    images_dir = Path(args.images_dir) if args.images_dir else (
        pipeline_root / "work" / scene.name / "colmap" / "dense" / "images"
    )
    out_dir = Path(args.out_dir) if args.out_dir else pipeline_root / "work" / scene.name / "corrector_dataset"

    if not images_dir.exists():
        raise SystemExit(
            f"Không thấy {images_dir} (ảnh đã undistort) — nếu bước train trước đó đã tự dọn thư mục "
            f"này (dọn đĩa mặc định), chạy lại `01_run_colmap.py --scene {scene.name}` để tái tạo "
            f"TRƯỚC khi chạy script này."
        )

    iteration = args.iteration if args.iteration > 0 else find_latest_iteration(model_dir)
    ply_path = model_dir / "point_cloud" / f"iteration_{iteration}" / "point_cloud.ply"
    if not ply_path.exists():
        raise SystemExit(f"Không thấy {ply_path}.")

    manifest_path = out_dir / "manifest.json"
    if manifest_path.exists() and not args.force:
        try:
            old_manifest = json.loads(manifest_path.read_text())
        except Exception:
            old_manifest = {}
        if old_manifest.get("source_iteration") == iteration and old_manifest.get("scene") == scene.name:
            print(f"[đã có, bỏ qua] {out_dir} đã khớp checkpoint iteration={iteration} — "
                  f"dùng --force nếu muốn sinh lại.")
            return
        print(f"  {manifest_path} lệch checkpoint hiện tại (manifest cũ: iteration="
              f"{old_manifest.get('source_iteration')}, hiện tại: {iteration}) — dọn dẹp + sinh lại.")

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

    render_dir = out_dir / "render"
    gt_dir = out_dir / "gt"
    for d in (render_dir, gt_dir):
        if d.exists():
            for f in d.glob("*.png"):
                f.unlink()
        d.mkdir(parents=True, exist_ok=True)

    n_saved, n_skipped = 0, 0
    image_sizes: set[tuple[int, int]] = set()
    print(f"===== {scene.name}: sinh {len(names_sorted)} cặp (render, GT) cho Model 2 =====")
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
        pred_u8 = (pred * 255.0).round().astype(np.uint8)

        with PILImage.open(gt_path) as gt_im:
            gt_im = gt_im.convert("RGB")
            if gt_im.size != (pose.width, pose.height):
                print(f"  [BỎ QUA] {name}: kích thước GT {gt_im.size} != pose ({pose.width},{pose.height}) "
                      f"— kiểm tra lại images_dir có đúng ảnh đã undistort không.")
                n_skipped += 1
                continue
            stem = Path(name).stem
            gt_im.save(gt_dir / f"{stem}.png", format="PNG")

        PILImage.fromarray(pred_u8).save(render_dir / f"{stem}.png", format="PNG")
        image_sizes.add((pose.width, pose.height))
        n_saved += 1
        if n_saved % 20 == 0:
            print(f"  [{n_saved}/{len(names_sorted)}] {name}")

    if n_saved == 0:
        raise SystemExit("Không sinh được cặp (render, GT) nào — xem lại images_dir/sparse_dir.")

    from datetime import datetime, timezone
    manifest = {
        "scene": scene.name,
        "source_iteration": iteration,
        "sh_degree": sh_degree,
        "antialiasing": antialiasing,
        "n_pairs": n_saved,
        "n_skipped": n_skipped,
        "image_sizes": sorted(list(image_sizes)),
        "built_at_utc": datetime.now(timezone.utc).isoformat(),
        "args": vars(args),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"\n-> Đã sinh {n_saved} cặp (bỏ qua {n_skipped}) tại {out_dir}")
    print(f"-> manifest: {manifest_path}")


if __name__ == "__main__":
    main()
