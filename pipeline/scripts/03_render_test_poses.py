#!/usr/bin/env python3
"""Render ảnh RGB tại từng pose trong test_poses.csv (hoặc holdout_poses.csv),
dùng model 3DGS đã train bằng repo GỐC graphdeco-inria/gaussian-splatting.

KHÔNG dùng render.py có sẵn của repo đó — nó chỉ render lại đúng train/test split
COLMAP nội bộ (cần ảnh GT có sẵn trên đĩa để load Scene). Ở đây ta cần render tại
pose TÙY Ý lấy từ CSV (test thật không có ảnh GT), nên tự dựng camera (MiniCam)
trực tiếp, dùng đúng công thức toán mà Camera/MiniCam của repo dùng (đã đối chiếu
source thật ở repo tiền nhiệm: scene/cameras.py, utils/graphics_utils.py,
gaussian_renderer/__init__.py — không suy đoán).

Yêu cầu trước khi chạy:
  export GS_REPO=/path/to/gaussian-splatting   # đã clone --recursive + cài xong
                                                # diff_gaussian_rasterization (cần CUDA)
  python 01_run_colmap.py --scene <scene> ...   # đã có dense/sparse
  bash 02_train_baseline.sh <scene>             # đã có gs_model/point_cloud/...

Output: pipeline/work/<scene>/renders/<stem>.png (LUÔN là PNG thật, kể cả nếu
image_name gốc trong CSV có đuôi .JPG). Việc đặt tên file CUỐI CÙNG khi đóng gói
zip nộp bài (giữ đuôi .JPG hay đổi .png) do script đóng gói submission (agent khác)
quyết định.

Hướng đi Mip-Splatting: script tự đọc `sh_degree` từ `cfg_args` mà train.py ghi lại
trong model_dir, và tự đọc `antialiasing` từ `pipeline_train_flags.json` mà
02_train_baseline.sh tự ghi thêm sau khi train xong — để biết chính xác cấu hình
lúc train, tránh trường hợp train bật --antialiasing nhưng render quên bật lại
(rasterizer sẽ chạy nhưng kết quả không nhất quán, không hề báo lỗi). ĐÃ TỪNG
XẢY RA ĐÚNG LỖI NÀY ở repo tiền nhiệm: cfg_args của repo gốc chỉ lưu ModelParams,
KHÔNG lưu `antialiasing` (field của PipelineParams) — nên trước khi có
pipeline_train_flags.json, script này luôn âm thầm render với antialiasing=False dù
train đã bật, làm méo hoàn toàn PSNR/SSIM/LPIPS (xem read_cfg_args()/
read_pipeline_train_flags() bên dưới, và docs/PORTED_KNOWLEDGE.md mục 2). Model
train bằng bản script thiếu pipeline_train_flags.json sẽ được cảnh báo rõ thay vì
âm thầm sai — cần re-train hoặc dùng --antialiasing on/off thủ công đúng với giá
trị thật đã dùng lúc train. Chỉ dùng --antialiasing on/off để ép thủ công khi thật
sự cần so sánh A/B.
"""
import argparse
import json
import os
import sys
from argparse import Namespace
from pathlib import Path

import numpy as np
import torch
from PIL import Image as PILImage

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.scenes import get_scene
from common.poses import read_test_poses, pose_to_R_T_fov, assert_centered_principal_point
from common.logging_utils import FileLog

GS_REPO = os.environ.get("GS_REPO")
if not GS_REPO or not (Path(GS_REPO) / "train.py").exists():
    raise SystemExit(
        "Chưa set biến môi trường GS_REPO hoặc đường dẫn sai.\n"
        "  export GS_REPO=/path/to/gaussian-splatting\n"
        "(thư mục clone --recursive https://github.com/graphdeco-inria/gaussian-splatting)"
    )
sys.path.insert(0, GS_REPO)

from scene.cameras import MiniCam                # noqa: E402
from scene.gaussian_model import GaussianModel    # noqa: E402
from gaussian_renderer import render              # noqa: E402
from utils.graphics_utils import getWorld2View2, getProjectionMatrix  # noqa: E402


class _PipelineParamsStub:
    """Thay cho arguments.PipelineParams — render() chỉ đọc đúng 4 field này."""
    convert_SHs_python = False
    compute_cov3D_python = False
    debug = False
    antialiasing = False


def read_cfg_args(model_dir: Path) -> dict:
    """Đọc file cfg_args mà train.py tự ghi (Namespace(...) dạng str, xem
    train.py::prepare_output_and_logger) để tự phát hiện sh_degree ĐÚNG như lúc
    train. LƯU Ý (đã đối chiếu trực tiếp source thật tại commit đã pin,
    54c035f7834b564019656c3e3fcc3646292f727d): cfg_args CHỈ chứa ModelParams
    (train.py gọi `prepare_output_and_logger(dataset)` với
    `dataset = lp.extract(args)`), nên `sh_degree`/`train_test_exp`/`depths` có
    trong file này, nhưng `antialiasing` (field của PipelineParams) THÌ KHÔNG
    BAO GIỜ có mặt — dù lúc train có bật --antialiasing hay không, cfg.get(
    'antialiasing') luôn ra None. Dùng read_pipeline_train_flags() bên dưới để
    lấy đúng giá trị antialiasing thật đã dùng lúc train."""
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
    """Đọc pipeline_train_flags.json do 02_train_baseline.sh tự ghi sau khi train
    xong (chứa antialiasing/depth_prior/exposure_comp/antenna_focus THẬT đã
    dùng — ở Vòng 1 chỉ antialiasing có thể true, 3 key còn lại luôn false vì
    02_train_baseline.sh không hỗ trợ) — nguồn đáng tin cậy duy nhất cho
    `antialiasing`, vì cfg_args của repo gốc không lưu field này (xem
    read_cfg_args()). Model train TRƯỚC khi có file này (checkpoint cũ) sẽ
    không có — main() sẽ cảnh báo rõ thay vì âm thầm coi như antialiasing=False."""
    p = model_dir / "pipeline_train_flags.json"
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text())
    except Exception as e:
        print(f"[CẢNH BÁO] Không đọc/parse được {p}: {e} — bỏ qua.")
        return {}


def build_minicam(pose, znear: float = 0.01, zfar: float = 100.0) -> MiniCam:
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True)
    ap.add_argument("--model_dir", default=None, help="Mặc định pipeline/work/<scene>/gs_model")
    ap.add_argument("--iteration", type=int, default=-1, help="-1 = iteration lớn nhất có sẵn")
    ap.add_argument("--out_dir", default=None, help="Mặc định pipeline/work/<scene>/renders")
    ap.add_argument("--poses_csv", default=None,
                     help="Mặc định test_poses.csv thật của scene. Trỏ sang "
                          "pipeline/work/<scene>/holdout/holdout_poses.csv để render holdout tự chấm "
                          "điểm (xem 00_make_holdout_split.py) — cùng schema nên dùng chung code này.")
    ap.add_argument("--white_background", action="store_true")
    ap.add_argument("--sh_degree", type=int, default=None,
                     help="Mặc định: tự đọc từ cfg_args (đúng giá trị lúc train). "
                          "Chỉ tự set nếu model không có cfg_args (checkpoint cũ) — khi đó mặc định 3.")
    ap.add_argument("--antialiasing", choices=["auto", "on", "off"], default="auto",
                     help="Mặc định 'auto': tự đọc từ pipeline_train_flags.json — PHẢI khớp giá trị "
                          "lúc train (xem 02_train_baseline.sh biến ANTIALIASING). Chỉ ép 'on'/'off' thủ "
                          "công nếu chắc chắn biết mình đang làm gì.")
    args = ap.parse_args()

    scene = get_scene(args.scene)
    pipeline_root = Path(__file__).resolve().parents[1]
    model_dir = Path(args.model_dir) if args.model_dir else pipeline_root / "work" / scene.name / "gs_model"
    out_dir = Path(args.out_dir) if args.out_dir else pipeline_root / "work" / scene.name / "renders"
    out_dir.mkdir(parents=True, exist_ok=True)

    iteration = args.iteration if args.iteration > 0 else find_latest_iteration(model_dir)
    ply_path = model_dir / "point_cloud" / f"iteration_{iteration}" / "point_cloud.ply"

    cfg = read_cfg_args(model_dir)
    train_flags = read_pipeline_train_flags(model_dir)
    if cfg:
        print(f"  cfg_args đọc được: sh_degree={cfg.get('sh_degree')}")
    else:
        print("  [CẢNH BÁO] Không có cfg_args trong model_dir (checkpoint train trước khi pipeline hỗ trợ "
              "tự phát hiện) — dùng mặc định sh_degree=3 trừ khi chỉ định --sh_degree.")

    sh_degree = args.sh_degree if args.sh_degree is not None else cfg.get("sh_degree", 3)
    if args.antialiasing == "auto":
        if "antialiasing" in train_flags:
            antialiasing = bool(train_flags["antialiasing"])
            print(f"  antialiasing đọc từ pipeline_train_flags.json (giá trị THẬT lúc train): {antialiasing}")
        else:
            antialiasing = False
            print(
                "  [CẢNH BÁO NGHIÊM TRỌNG] Không có pipeline_train_flags.json trong model_dir — model này train "
                "bằng bản 02_train_baseline.sh CŨ hoặc bị xoá nhầm (xem docstring read_cfg_args()). KHÔNG THỂ tự "
                "biết chắc lúc train có bật --antialiasing hay không -> đang mặc định antialiasing=False, CÓ THỂ "
                "SAI và làm méo hoàn toàn PSNR/SSIM/LPIPS (train/render lệch antialiasing) mà không báo lỗi gì "
                "khác. Nếu bạn train scene này với ANTIALIASING=1 (mặc định của pipeline), hãy chạy lại với "
                "--antialiasing on để ép đúng giá trị, rồi chạy lại 04_eval_metrics.py."
            )
    else:
        antialiasing = args.antialiasing == "on"

    gaussians = GaussianModel(sh_degree)
    gaussians.load_ply(str(ply_path))

    bg_color = [1, 1, 1] if args.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    pipe = _PipelineParamsStub()
    pipe.antialiasing = antialiasing
    print(f"  Render với sh_degree={sh_degree}, antialiasing={antialiasing}")

    poses_csv = Path(args.poses_csv) if args.poses_csv else scene.test_poses_csv
    poses = read_test_poses(poses_csv)
    log_path = out_dir.parent / "03_render_test_poses.log"
    log = FileLog(log_path)
    print(f"===== {scene.name}: render {len(poses)} pose từ {poses_csv.name} "
          f"(iteration {iteration}) -> {out_dir} =====")
    log.write(f"Model: {ply_path}")
    log.write(f"Poses CSV: {poses_csv}")

    for i, pose in enumerate(poses):
        assert_centered_principal_point(pose)
        cam = build_minicam(pose)
        with torch.no_grad():
            out = render(cam, gaussians, pipe, background)
        img = out["render"].clamp(0, 1).detach().cpu().numpy()  # (3,H,W)
        img = (np.transpose(img, (1, 2, 0)) * 255.0).round().astype(np.uint8)
        pil_img = PILImage.fromarray(img)
        if pil_img.size != (pose.width, pose.height):
            raise RuntimeError(
                f"{pose.image_name}: render ra {pil_img.size}, khác yêu cầu "
                f"({pose.width},{pose.height}) — kiểm tra lại MiniCam/getProjectionMatrix."
            )
        stem = Path(pose.image_name).stem
        out_path = out_dir / f"{stem}.png"
        pil_img.save(out_path, format="PNG")
        log.write(f"[{i + 1}/{len(poses)}] {pose.image_name} -> {out_path.name}")

    log.write(f"Xong. {len(poses)} ảnh PNG tại {out_dir}")
    log.close()
    print(f"-> Xong {len(poses)} ảnh. Log chi tiết từng ảnh: {log_path}")


if __name__ == "__main__":
    main()
