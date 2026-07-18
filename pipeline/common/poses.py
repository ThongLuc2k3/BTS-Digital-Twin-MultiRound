"""Đọc test_poses.csv và chuyển 1 dòng pose sang định dạng mà
graphdeco-inria/gaussian-splatting cần (R, T, FovX, FovY).

Quy ước quaternion/translation: đúng như COLMAP images.bin — (qw,qx,qy,qz,tx,ty,tz)
biểu diễn phép biến đổi WORLD -> CAMERA (X_cam = Rmat(qvec) @ X_world + tvec).
Đây cũng chính xác là quy ước mà scene/dataset_readers.py::readColmapCameras của
graphdeco-inria/gaussian-splatting dùng để build CameraInfo:
    R = transpose(qvec2rotmat(qvec))   (transpose vì Camera lưu R theo cột, xem
                                         utils/graphics_utils.py::getWorld2View2)
    T = tvec  (giữ nguyên)
Nguồn đối chiếu: https://github.com/graphdeco-inria/gaussian-splatting
  scene/dataset_readers.py (readColmapCameras), scene/colmap_loader.py (qvec2rotmat),
  utils/graphics_utils.py (getWorld2View2, getProjectionMatrix, focal2fov)
Đã fetch trực tiếp source các file trên để đối chiếu byte-for-byte trước khi viết file này.
"""
import csv
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class TestPose:
    image_name: str
    qvec: np.ndarray  # (4,) qw,qx,qy,qz — world->camera rotation
    tvec: np.ndarray  # (3,) world->camera translation
    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int


def qvec2rotmat(qvec: np.ndarray) -> np.ndarray:
    """Y hệt scene/colmap_loader.py::qvec2rotmat của gaussian-splatting / COLMAP."""
    qw, qx, qy, qz = qvec
    return np.array([
        [1 - 2 * qy**2 - 2 * qz**2,     2 * qx * qy - 2 * qw * qz,       2 * qz * qx + 2 * qw * qy],
        [2 * qx * qy + 2 * qw * qz,     1 - 2 * qx**2 - 2 * qz**2,       2 * qy * qz - 2 * qw * qx],
        [2 * qz * qx - 2 * qw * qy,     2 * qy * qz + 2 * qw * qx,       1 - 2 * qx**2 - 2 * qy**2],
    ])


def focal2fov(focal: float, pixels: float) -> float:
    """Y hệt utils/graphics_utils.py::focal2fov."""
    return 2 * math.atan(pixels / (2 * focal))


def read_test_poses(csv_path: Path) -> list[TestPose]:
    rows: list[TestPose] = []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(TestPose(
                image_name=r["image_name"],
                qvec=np.array([float(r["qw"]), float(r["qx"]), float(r["qy"]), float(r["qz"])]),
                tvec=np.array([float(r["tx"]), float(r["ty"]), float(r["tz"])]),
                fx=float(r["fx"]), fy=float(r["fy"]),
                cx=float(r["cx"]), cy=float(r["cy"]),
                width=int(float(r["width"])), height=int(float(r["height"])),
            ))
    return rows


def pose_to_R_T_fov(pose: TestPose) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Trả về (R, T, FovX, FovY) đúng convention Camera(...) của gaussian-splatting.

    Giả định cx == width/2 và cy == height/2 (đã kiểm tra đúng với mọi pose trong
    dataset thi hiện tại) vì getProjectionMatrix gốc của repo không nhận cx,cy lệch
    tâm. Nếu dữ liệu sau này có nguyên tắc khác, cảnh báo sẽ được in ra (xem hàm
    assert_centered_principal_point) — không âm thầm bỏ qua.
    """
    Rmat = qvec2rotmat(pose.qvec)
    R = Rmat.transpose()
    T = pose.tvec.copy()
    FovY = focal2fov(pose.fy, pose.height)
    FovX = focal2fov(pose.fx, pose.width)
    return R, T, FovX, FovY


def assert_centered_principal_point(pose: TestPose, atol: float = 1.0) -> None:
    cx_expected = pose.width / 2.0
    cy_expected = pose.height / 2.0
    if abs(pose.cx - cx_expected) > atol or abs(pose.cy - cy_expected) > atol:
        raise ValueError(
            f"{pose.image_name}: cx,cy ({pose.cx},{pose.cy}) lệch tâm ảnh "
            f"({cx_expected},{cy_expected}) quá {atol}px — script render hiện tại "
            f"dùng công thức FOV giả định principal point ở giữa ảnh, cần sửa lại "
            f"getProjectionMatrix nếu muốn hỗ trợ cx,cy lệch tâm."
        )


def representative_intrinsics(csv_path: Path) -> tuple[float, float, float, int, int]:
    """Lấy fx, cx, cy, width, height từ dòng đầu tiên của test_poses.csv — dùng làm
    prior camera_params khi tự chạy COLMAP trên train/images/ của CÙNG scene đó
    (giả định train và test dùng chung 1 camera vật lý trong cùng 1 chuyến bay).
    """
    poses = read_test_poses(csv_path)
    if not poses:
        raise ValueError(f"{csv_path} rỗng")
    p0 = poses[0]
    fxs = {round(p.fx, 3) for p in poses}
    if len(fxs) > 1:
        print(f"[CẢNH BÁO] {csv_path}: fx không đồng nhất giữa các pose ({fxs}) — "
              f"dùng giá trị của pose đầu tiên làm prior, kiểm tra lại nếu nghi ngờ.")
    return p0.fx, p0.cx, p0.cy, p0.width, p0.height
