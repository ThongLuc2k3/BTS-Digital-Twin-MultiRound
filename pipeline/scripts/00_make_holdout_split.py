#!/usr/bin/env python3
"""Tạo tập holdout nội bộ từ ảnh train — dùng để tự chấm PSNR/SSIM/LPIPS/Score khi
scene KHÔNG có ảnh ground-truth test thật (toàn bộ 7 scene của dataset hiện tại đều
vậy — xem `docs/00_MASTER_PLAN.md` mục 1 "Input mỗi scene").

Cách chọn holdout: sort tên ảnh train (tên file phản ánh thứ tự chụp/bay), rồi lấy
index CÁCH ĐỀU nhau (np.linspace) trong danh sách đã sort — để tập holdout trải đều
khắp quỹ đạo bay thay vì dồn vào 1 đoạn liên tiếp (giữ đa dạng góc nhìn). Seed cố
định để reproducible (bắt buộc theo mục 10.3 đề bài — phải tái lập được kết quả).

Pose/intrinsics của ảnh holdout lấy trực tiếp từ sparse COLMAP có sẵn
(`scene.provided_sparse_dir`, do BTC cấp) bằng pycolmap — CHÍNH XÁC, không phải suy
diễn — rồi ghi ra `holdout_poses.csv` ĐÚNG SCHEMA với `test_poses.csv`
(qw,qx,qy,qz,tx,ty,tz,fx,fy,cx,cy,width,height, xem TestPose/read_test_poses ở
common/poses.py) để tái dùng được toàn bộ code render/eval hiện có.

QUAN TRỌNG — đã verify thực nghiệm ở repo tiền nhiệm (so khớp ma trận xoay
`Rotation3d.matrix()` với `qvec2rotmat` của common/poses.py, sai lệch 0.0):
`pycolmap.Rigid3d.rotation.quat` trả về theo thứ tự [x,y,z,w] (scalar-last, kiểu
Eigen), KHÁC với quy ước qw,qx,qy,qz (scalar-first) mà COLMAP file format /
test_poses.csv / common/poses.py dùng — bắt buộc phải đảo thứ tự khi ghi ra CSV, nếu
không thì mọi pose holdout sẽ bị lật sai hoàn toàn (không lỗi rõ ràng, chỉ render ra
ảnh sai góc — rất khó phát hiện). Xem `docs/PORTED_KNOWLEDGE.md` mục 1.

Sparse do BTC cấp có thể được build từ tập ảnh gốc LỚN HƠN tập đóng gói trong
`train/images/` — nên số ảnh có pose trong sparse có thể > số file thật trên đĩa.
Script này CHỈ chọn holdout trong giao của 2 tập (ảnh vừa có pose vừa có file thật),
bỏ qua phần dư trong sparse.

Cơ chế tái sử dụng cho bước train (không cần sửa colmap_runner.py): thư mục
`train_images/` sinh ra ở đây chỉ chứa symlink tới ảnh KHÔNG bị chọn làm holdout.
Khi truyền thư mục này làm `images_dir` cho `colmap_runner.use_provided_sparse()`,
cơ chế `_find_missing_images`/`_undistort_and_fix_layout` đã có sẵn trong
`common/colmap_runner.py` sẽ tự phát hiện các ảnh holdout là "có pose trong sparse
nhưng thiếu file trên đĩa" và tự động deregister khỏi reconstruction trước khi
train — hoàn toàn không cần thêm code lọc sparse riêng.

Cách dùng:
    python 00_make_holdout_split.py --scene chair
    python 00_make_holdout_split.py --all
    python 00_make_holdout_split.py --scene bonsai --holdout_frac 0.15 --overwrite
"""
import argparse
import random
import sys
from pathlib import Path

import numpy as np
import pycolmap

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.scenes import Scene, all_scenes, get_scene

CSV_HEADER = ["image_name", "qw", "qx", "qy", "qz", "tx", "ty", "tz",
              "fx", "fy", "cx", "cy", "width", "height"]


def load_registered_poses(scene: Scene) -> dict[str, dict]:
    """Đọc sparse có sẵn -> {tên ảnh: pose+intrinsics}, CHỈ cho ảnh có file thật
    trong train/images/ (xem docstring đầu file về sparse lớn hơn tập ảnh đóng gói)."""
    rec = pycolmap.Reconstruction(str(scene.provided_sparse_dir))
    disk_names = {p.name for p in scene.train_images_dir.iterdir()}

    poses: dict[str, dict] = {}
    for image in rec.images.values():
        if image.name not in disk_names:
            continue
        cam = rec.cameras[image.camera_id]
        cfw = image.cam_from_world()
        qx, qy, qz, qw = cfw.rotation.quat  # pycolmap: [x,y,z,w] -> đảo sang wxyz bên dưới
        tx, ty, tz = cfw.translation
        poses[image.name] = dict(
            qw=qw, qx=qx, qy=qy, qz=qz, tx=tx, ty=ty, tz=tz,
            fx=cam.focal_length_x, fy=cam.focal_length_y,
            cx=cam.principal_point_x, cy=cam.principal_point_y,
            width=cam.width, height=cam.height,
        )
    return poses


def choose_holdout_names(names_sorted: list[str], frac: float, seed: int) -> list[str]:
    n = len(names_sorted)
    k = max(1, round(n * frac))
    # index cách đều nhau trong danh sách đã sort theo tên (~thứ tự bay/chụp) để
    # holdout trải khắp quỹ đạo, không dồn cục — seed chỉ dùng để jitter nhẹ thứ
    # tự bắt đầu, không ảnh hưởng tính "trải đều".
    rng = random.Random(seed)
    offset = rng.randrange(0, max(1, n // k)) if k > 0 else 0
    idxs = np.linspace(offset, n - 1, k).round().astype(int)
    idxs = sorted(set(int(i) for i in idxs))
    return [names_sorted[i] for i in idxs]


def make_split(scene: Scene, frac: float, seed: int, overwrite: bool) -> None:
    pipeline_root = Path(__file__).resolve().parents[1]
    out_dir = pipeline_root / "work" / scene.name / "holdout"
    if out_dir.exists() and not overwrite:
        raise SystemExit(
            f"{out_dir} đã tồn tại — dùng --overwrite nếu thật sự muốn tạo lại holdout "
            f"(cẩn thận: đổi holdout giữa chừng làm các lần đo trước/sau không còn so "
            f"sánh được với nhau)."
        )

    poses = load_registered_poses(scene)
    names_sorted = sorted(poses.keys())
    if not names_sorted:
        raise SystemExit(f"{scene.name}: không tìm thấy ảnh nào vừa có pose vừa có file thật.")

    holdout_names = choose_holdout_names(names_sorted, frac, seed)
    holdout_set = set(holdout_names)
    train_names = [n for n in names_sorted if n not in holdout_set]

    train_images_dir = out_dir / "train_images"
    holdout_gt_dir = out_dir / "holdout_gt"
    for d in (train_images_dir, holdout_gt_dir):
        if d.exists():
            for f in d.iterdir():
                f.unlink()
        d.mkdir(parents=True, exist_ok=True)

    for name in train_names:
        (train_images_dir / name).symlink_to((scene.train_images_dir / name).resolve())
    for name in holdout_names:
        (holdout_gt_dir / name).symlink_to((scene.train_images_dir / name).resolve())

    csv_path = out_dir / "holdout_poses.csv"
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write(",".join(CSV_HEADER) + "\n")
        for name in holdout_names:
            p = poses[name]
            f.write(",".join(str(p[c]) if c != "image_name" else name for c in CSV_HEADER) + "\n")

    manifest_path = out_dir / "manifest.txt"
    with open(manifest_path, "w", encoding="utf-8") as f:
        f.write(f"scene={scene.name}\nseed={seed}\nholdout_frac={frac}\n")
        f.write(f"n_total={len(names_sorted)}\nn_holdout={len(holdout_names)}\nn_train={len(train_names)}\n")
        f.write("holdout_images=\n")
        for name in holdout_names:
            f.write(f"  {name}\n")

    print(f"{scene.name}: {len(names_sorted)} ảnh -> {len(train_names)} train / "
          f"{len(holdout_names)} holdout ({len(holdout_names) / len(names_sorted):.1%}) "
          f"-> {out_dir}")


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--scene", help="1 scene cụ thể, vd chair")
    g.add_argument("--all", action="store_true", help="Toàn bộ 7 scene")
    ap.add_argument("--holdout_frac", type=float, default=0.125)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    scenes = [get_scene(args.scene)] if args.scene else all_scenes()
    for scene in scenes:
        make_split(scene, args.holdout_frac, args.seed, args.overwrite)


if __name__ == "__main__":
    main()
