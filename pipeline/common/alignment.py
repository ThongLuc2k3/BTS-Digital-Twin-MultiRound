"""Umeyama similarity-transform alignment — dùng cho Phase 0 (kiểm định hệ toạ độ).

Thuật toán chuẩn (Umeyama, 1991): tìm scale s, rotation R (3x3), translation t
tối thiểu hoá sum ||dst_i - (s * R @ src_i + t)||^2, cho 2 tập điểm tương ứng
src, dst (Nx3, N>=3, đã khớp theo thứ tự/tên).
"""
import numpy as np


def umeyama_alignment(src: np.ndarray, dst: np.ndarray):
    """src, dst: (N,3) camera centers tương ứng (đã match theo tên ảnh).

    Trả về (s, R, t, residuals, aligned):
      s: scale (float)
      R: rotation 3x3
      t: translation (3,)
      residuals: khoảng cách Euclid từng điểm sau khi align (N,)
      aligned: src đã biến đổi bằng s,R,t (N,3)
    """
    assert src.shape == dst.shape and src.shape[1] == 3
    n = src.shape[0]
    if n < 3:
        raise ValueError("Cần tối thiểu 3 điểm tương ứng để ước lượng Sim3.")

    mu_src = src.mean(axis=0)
    mu_dst = dst.mean(axis=0)
    src_c = src - mu_src
    dst_c = dst - mu_dst

    cov = (dst_c.T @ src_c) / n
    U, D, Vt = np.linalg.svd(cov)
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[-1, -1] = -1.0
    R = U @ S @ Vt

    var_src = (src_c ** 2).sum() / n
    s = float(np.trace(np.diag(D) @ S) / var_src) if var_src > 1e-12 else 1.0
    t = mu_dst - s * (R @ mu_src)

    aligned = (s * (R @ src.T).T) + t
    residuals = np.linalg.norm(aligned - dst, axis=1)
    return s, R, t, residuals, aligned


def raw_residuals(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """Residual KHÔNG align gì cả (identity transform) — dùng để kiểm tra xem
    2 hệ toạ độ có tự nhiên trùng khớp hay không (trường hợp tốt nhất)."""
    return np.linalg.norm(src - dst, axis=1)
