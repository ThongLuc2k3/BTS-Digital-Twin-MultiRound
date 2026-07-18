#!/usr/bin/env python3
"""Chuẩn bị dữ liệu COLMAP (sparse + ảnh undistort PINHOLE) cho 1 scene hoặc
toàn bộ scene, sẵn sàng đưa vào train 3D Gaussian Splatting.

Dataset có sparse hợp lệ ở cả 7/7 scene — mặc định script này DÙNG THẲNG sparse
có sẵn (chỉ undistort, rất nhanh), KHÔNG tự chạy lại COLMAP. Chỉ tự chạy COLMAP khi:
  - Scene không có sparse hợp lệ (`has_valid_provided_sparse()` = False), hoặc
  - Người dùng ép buộc qua --force_own_colmap (vd để đối chiếu/nghi ngờ dữ liệu).

Ví dụ:
    python 01_run_colmap.py --scene HCM0421
    python 01_run_colmap.py --all --domain bts
    python 01_run_colmap.py --scene chair --force_own_colmap --matching exhaustive

Output: pipeline/work/<scene>/colmap/dense/{images/, sparse/0/}
        -> đây chính là thư mục để đưa vào 02_train_baseline.sh.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.scenes import Scene, all_scenes, get_scene
from common.colmap_runner import run_colmap_scene, use_provided_sparse
from common.poses import representative_intrinsics

WORK_ROOT = Path(__file__).resolve().parents[1] / "work"


def _report_missing_images(scene: Scene, result: dict) -> None:
    """Chỉ in SỐ LƯỢNG ra console (tránh spam khi thiếu hàng trăm ảnh) — đây là
    đặc điểm bình thường của dữ liệu (sparse có thể dựng từ tập ảnh gốc lớn hơn
    train/images/ đã phát hành), KHÔNG phải lỗi COLMAP hay lỗi tải dữ liệu, nên
    tách riêng khỏi cảnh báo "tỉ lệ đăng ký thấp". Tên file cụ thể đã có sẵn
    trong log_path (ghi bởi common/colmap_runner.py), không cần in lại ở đây."""
    missing = result.get("missing_images") or []
    if missing:
        print(f"[LƯU Ý] {scene.name}: {len(missing)} ảnh có pose trong sparse nhưng không có file "
              f"trong train/images/ (đã tự loại ra, không phải lỗi) — danh sách tên file: {result['log_path']}")


def process_scene(scene: Scene, matching: str, camera_model: str, use_prior: bool,
                   overwrite: bool, force_own_colmap: bool, images_dir: Path | None = None):
    """images_dir: override ảnh input (mặc định scene.train_images_dir) — dùng khi train
    ở chế độ holdout-eval (xem 00_make_holdout_split.py), truyền
    pipeline/work/<scene>/holdout/train_images (chỉ symlink phần ảnh KHÔNG bị holdout).
    Cơ chế `_find_missing_images` có sẵn trong colmap_runner.py sẽ tự loại các ảnh
    holdout khỏi sparse (coi như "thiếu file trên đĩa"), không cần code lọc riêng."""
    images_dir = images_dir if images_dir is not None else scene.train_images_dir
    workdir = WORK_ROOT / scene.name / "colmap"
    log_path = WORK_ROOT / scene.name / "01_run_colmap.log"

    if scene.has_valid_provided_sparse() and not force_own_colmap:
        print(f"===== {scene.name} ({scene.domain}) — dùng sparse có sẵn (bỏ qua tự chạy COLMAP) =====")
        result = use_provided_sparse(
            images_dir=images_dir,
            sparse_dir=scene.provided_sparse_dir,
            workdir=workdir,
            log_path=log_path,
        )
        n_total = len(list(images_dir.glob("*")))
        print(f"-> {result['num_reg_images']}/{n_total} ảnh, {result['num_points3D']} điểm 3D (từ sparse có sẵn). "
              f"Dense: {result['dense_dir']} | Log: {result['log_path']}")
        _report_missing_images(scene, result)
        return result

    camera_params_prior = None
    if use_prior and scene.test_poses_csv.exists():
        fx, cx, cy, width, height = representative_intrinsics(scene.test_poses_csv)
        if camera_model == "SIMPLE_RADIAL":
            camera_params_prior = f"{fx},{cx},{cy},0.0"
        elif camera_model == "PINHOLE":
            camera_params_prior = f"{fx},{fx},{cx},{cy}"

    print(f"===== {scene.name} ({scene.domain}) — tự chạy COLMAP ({matching}, {camera_model}) =====")
    result = run_colmap_scene(
        images_dir=images_dir,
        workdir=workdir,
        matching=matching,
        camera_model=camera_model,
        camera_params_prior=camera_params_prior,
        overwrite=overwrite,
        log_path=log_path,
    )
    n_total = len(list(images_dir.glob("*")))
    print(f"-> {result['num_reg_images']}/{n_total} ảnh đăng ký, {result['num_points3D']} điểm 3D. "
          f"Dense: {result['dense_dir']} | Log chi tiết: {result['log_path']}")
    _report_missing_images(scene, result)
    if result["num_reg_images"] < 0.8 * n_total:
        print(f"[CẢNH BÁO] {scene.name}: tỉ lệ đăng ký ảnh thấp (<80%) — cân nhắc thử "
              f"--matching exhaustive hoặc kiểm tra lại ảnh scene này.")
    return result


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--scene", help="Tên 1 scene, vd HCM0421, chair, bonsai")
    g.add_argument("--all", action="store_true", help="Chạy toàn bộ scene")
    ap.add_argument("--domain", choices=["bts", "generic"], default=None,
                     help="Chỉ dùng với --all, giới hạn scene BTS hoặc scene tổng quát (bonsai/chair)")
    ap.add_argument("--force_own_colmap", action="store_true",
                     help="Ép tự chạy COLMAP dù scene đã có sparse hợp lệ (vd để đối chiếu/nghi ngờ dữ liệu)")
    ap.add_argument("--matching", default="sequential", choices=["sequential", "exhaustive"],
                     help="Chỉ áp dụng khi thực sự tự chạy COLMAP")
    ap.add_argument("--camera_model", default="SIMPLE_RADIAL",
                     help="SIMPLE_RADIAL (khớp dữ liệu gốc, có méo nhẹ) hoặc PINHOLE — chỉ áp dụng khi tự chạy COLMAP")
    ap.add_argument("--no_prior", action="store_true",
                     help="Không dùng fx/cx/cy từ test_poses.csv làm prior, để COLMAP tự đoán từ EXIF/heuristic")
    ap.add_argument("--overwrite", action="store_true", help="Chạy lại từ đầu, xoá database.db cũ")
    ap.add_argument("--holdout", action="store_true",
                     help="Train ở chế độ holdout-eval (xem 00_make_holdout_split.py): dùng "
                          "pipeline/work/<scene>/holdout/train_images (đã tạo bằng "
                          "00_make_holdout_split.py) thay vì scene.train_images_dir đầy đủ — để đo "
                          "Score trên holdout trước khi retrain lần cuối trên 100% ảnh.")
    args = ap.parse_args()

    if args.scene:
        scenes = [get_scene(args.scene)]
    else:
        scenes = all_scenes(args.domain)

    for scene in scenes:
        images_dir = None
        if args.holdout:
            images_dir = WORK_ROOT / scene.name / "holdout" / "train_images"
            if not images_dir.exists():
                raise SystemExit(
                    f"{scene.name}: --holdout nhưng chưa có {images_dir} — chạy "
                    f"00_make_holdout_split.py --scene {scene.name} trước."
                )
        process_scene(scene, args.matching, args.camera_model, not args.no_prior,
                      args.overwrite, args.force_own_colmap, images_dir=images_dir)


if __name__ == "__main__":
    main()
