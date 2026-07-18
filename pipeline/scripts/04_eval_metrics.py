#!/usr/bin/env python3
"""Tự chấm PSNR/SSIM/LPIPS trên holdout tự tạo (KHÔNG scene nào có ảnh GT test
thật — xem `docs/00_MASTER_PLAN.md` mục 1 "Input mỗi scene"). Đây là cách DUY NHẤT
hợp lệ để tự chấm trước khi nộp: giữ lại ~10-15% ảnh train làm holdout (xem
00_make_holdout_split.py), so ảnh render tại pose holdout với ảnh train thật đó.

Cách dùng:
    python 00_make_holdout_split.py --scene HCM0421          # 1 lần, tạo holdout
    python 03_render_test_poses.py --scene HCM0421 \\
        --poses_csv pipeline/work/HCM0421/holdout/holdout_poses.csv \\
        --out_dir pipeline/work/HCM0421/holdout_renders
    python 04_eval_metrics.py --scene HCM0421
    python 04_eval_metrics.py --all
    python 04_eval_metrics.py --all --psnr_max 25   # tự chọn PSNR_max khác

Yêu cầu đã chạy 00_make_holdout_split.py + 03_render_test_poses.py (--poses_csv trỏ
sang holdout_poses.csv) cho scene đó trước — renders nằm ở
pipeline/work/<scene>/holdout_renders/<stem>.png (mặc định), GT nằm ở
pipeline/work/<scene>/holdout/holdout_gt/ (do 00_make_holdout_split.py tạo).

Ngoài PSNR/SSIM/LPIPS riêng lẻ, còn tính điểm tổng hợp Score theo ĐÚNG công thức
chính thức của BTC (xem `docs/00_MASTER_PLAN.md` mục 1 "Điểm"):

    Score = 0.4 * (1 - LPIPS) + 0.3 * SSIM + 0.3 * PSNR_norm
    PSNR_norm = clamp(PSNR / PSNR_max, 0, 1)

QUAN TRỌNG: đề bài KHÔNG công bố giá trị PSNR_max cụ thể. Mặc định ở đây là 50.0 —
ước lượng tham khảo dùng ở repo tiền nhiệm (`docs/PORTED_KNOWLEDGE.md` mục 5), KHÔNG
phải số chính thức từ BTC. --psnr_max vẫn cho tự đổi nếu muốn thử giá trị khác.
Script tự in thêm bảng Score ở vài giá trị PSNR_max khác để thấy độ nhạy.

CÁCH GỘP ĐIỂM NHIỀU SCENE — đúng nguyên văn đề bài: "Điểm trên bảng xếp hạng là điểm
trung bình của toàn bộ các scene" — tức TRUNG BÌNH CỘNG của Score từng scene (mỗi
scene 1 Score, rồi lấy trung bình N scene đó), KHÔNG PHẢI gộp hết ảnh của mọi scene
lại tính chung 1 lần. 2 cách này cho kết quả KHÁC NHAU nếu số ảnh test giữa các scene
không bằng nhau (scene ít ảnh sẽ bị lép vế nếu gộp ảnh thay vì gộp theo scene). Script
tính đúng: Score từng scene trước, rồi trung bình cộng các Score đó ở cuối.
"""
import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.scenes import get_scene, all_scenes, Scene

try:
    import lpips
    _HAS_LPIPS = True
except ImportError:
    _HAS_LPIPS = False


def load_img01(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB")).astype(np.float32) / 255.0


def eval_scene(scene: Scene, renders_dir: Path, gt_dir: Path, lpips_fn) -> list[tuple]:
    rows = []
    gt_paths = sorted(gt_dir.glob("*"))
    for gt_path in gt_paths:
        stem = gt_path.stem
        render_path = renders_dir / f"{stem}.png"
        if not render_path.exists():
            print(f"  [THIẾU] {stem}: không có render tại {render_path}")
            continue
        gt = load_img01(gt_path)
        pred = load_img01(render_path)
        if gt.shape != pred.shape:
            raise ValueError(f"{stem}: kích thước khác nhau GT={gt.shape[:2]} pred={pred.shape[:2]}")

        psnr_v = peak_signal_noise_ratio(gt, pred, data_range=1.0)
        ssim_v = structural_similarity(gt, pred, data_range=1.0, channel_axis=2)

        lpips_v = float("nan")
        if lpips_fn is not None:
            t_gt = torch.from_numpy(gt).permute(2, 0, 1).unsqueeze(0) * 2 - 1
            t_pred = torch.from_numpy(pred).permute(2, 0, 1).unsqueeze(0) * 2 - 1
            if torch.cuda.is_available():
                t_gt, t_pred = t_gt.cuda(), t_pred.cuda()
            with torch.no_grad():
                lpips_v = float(lpips_fn(t_gt, t_pred).item())

        rows.append((stem, psnr_v, ssim_v, lpips_v))

    n_missing = len(gt_paths) - len(rows)
    if n_missing:
        print(f"  [CẢNH BÁO] {scene.name}: thiếu {n_missing}/{len(gt_paths)} render — "
              f"điểm trung bình dưới đây KHÔNG đại diện cho toàn bộ scene.")
    return rows


def compute_score(psnr_v: float, ssim_v: float, lpips_v: float, psnr_max: float) -> float:
    """Đúng công thức Score của đề bài. Nếu thiếu LPIPS (chưa cài package), coi
    như 0 cho phần (1 - LPIPS) — KHÔNG đại diện đúng điểm thật, chỉ để không crash."""
    lpips_term = 0.0 if np.isnan(lpips_v) else (1.0 - lpips_v)
    psnr_norm = min(max(psnr_v / psnr_max, 0.0), 1.0)
    return 0.4 * lpips_term + 0.3 * ssim_v + 0.3 * psnr_norm


def write_csv(csv_path: Path, rows: list[tuple], psnr_max: float) -> None:
    """Ghi điểm từng ảnh test ra CSV — dùng để vẽ biểu đồ/so sánh ảnh trong notebook,
    vì console chỉ in mean/min/max."""
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["image", "psnr", "ssim", "lpips", "score"])
        for stem, psnr_v, ssim_v, lpips_v in rows:
            score = compute_score(psnr_v, ssim_v, lpips_v, psnr_max)
            writer.writerow([stem, psnr_v, ssim_v, lpips_v, score])


def print_stats(name: str, rows: list[tuple], psnr_max: float) -> float:
    """Trả về Score của riêng `name` (1 scene) — dùng để main() tự trung bình cộng
    theo scene ở cuối, đúng "điểm trung bình của toàn bộ các scene", KHÔNG gộp ảnh
    của nhiều scene lại tính chung."""
    arr = np.array([[r[1], r[2], r[3]] for r in rows])
    print(f"\n=== {name}: {len(rows)} ảnh ===")
    print(f"  PSNR  mean={arr[:, 0].mean():.3f}  min={arr[:, 0].min():.3f}")
    print(f"  SSIM  mean={arr[:, 1].mean():.4f}  min={arr[:, 1].min():.4f}")
    if not np.isnan(arr[:, 2]).all():
        print(f"  LPIPS mean={arr[:, 2].mean():.4f}  max={arr[:, 2].max():.4f}")

    scores = [compute_score(r[1], r[2], r[3], psnr_max) for r in rows]
    scene_score = float(np.mean(scores))
    print(f"  Score mean={scene_score:.4f}  min={np.min(scores):.4f}  "
          f"(công thức BTC, PSNR_max={psnr_max} — ước lượng, xem docstring đầu file)")

    print("  Độ nhạy Score theo PSNR_max khác (để tham khảo, không phải điểm chính thức):")
    for candidate in (20.0, 25.0, 30.0, 35.0, 40.0, 50.0):
        alt_scores = [compute_score(r[1], r[2], r[3], candidate) for r in rows]
        marker = " <- đang dùng" if candidate == psnr_max else ""
        print(f"    PSNR_max={candidate:5.1f} -> Score mean={np.mean(alt_scores):.4f}{marker}")

    return scene_score


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--scene", help="1 scene cụ thể, vd chair")
    g.add_argument("--all", action="store_true", help="Toàn bộ 7 scene")
    ap.add_argument("--renders_root", default=None, help="Mặc định pipeline/work")
    ap.add_argument("--renders_subdir", default="holdout_renders",
                     help="Thư mục con trong work/<scene>/ chứa ảnh render holdout "
                          "(mặc định 'holdout_renders' — khớp --out_dir gợi ý ở "
                          "03_render_test_poses.py)")
    ap.add_argument("--no_lpips", action="store_true", help="Bỏ qua LPIPS (nếu chưa cài package lpips)")
    ap.add_argument("--psnr_max", type=float, default=50.0,
                     help="Ngưỡng chuẩn hoá PSNR cho công thức Score (mặc định 50.0 — "
                          "ước lượng tham khảo, xem docstring đầu file)")
    args = ap.parse_args()

    scenes = [get_scene(args.scene)] if args.scene else all_scenes()
    pipeline_root = Path(__file__).resolve().parents[1]
    renders_root = Path(args.renders_root) if args.renders_root else pipeline_root / "work"

    lpips_fn = None
    if not args.no_lpips:
        if not _HAS_LPIPS:
            print("[CẢNH BÁO] Chưa cài package `lpips` (pip install lpips) — bỏ qua LPIPS, chỉ tính PSNR/SSIM.")
        else:
            lpips_fn = lpips.LPIPS(net="alex")
            if torch.cuda.is_available():
                lpips_fn = lpips_fn.cuda()

    scene_scores = []
    for scene in scenes:
        gt_dir = renders_root / scene.name / "holdout" / "holdout_gt"
        if not gt_dir.exists():
            print(f"[BỎ QUA] {scene.name}: chưa có holdout (thiếu {gt_dir}) — chạy "
                  f"00_make_holdout_split.py --scene {scene.name} trước.")
            continue
        renders_dir = renders_root / scene.name / args.renders_subdir
        if not renders_dir.exists():
            print(f"[BỎ QUA] {scene.name}: chưa render holdout (thiếu {renders_dir}) — chạy "
                  f"03_render_test_poses.py --scene {scene.name} --poses_csv "
                  f".../holdout/holdout_poses.csv --out_dir {renders_dir} trước.")
            continue
        rows = eval_scene(scene, renders_dir, gt_dir, lpips_fn)
        if rows:
            scene_score = print_stats(scene.name, rows, args.psnr_max)
            write_csv(renders_root / scene.name / "eval_metrics.csv", rows, args.psnr_max)
            scene_scores.append((scene.name, scene_score))

    if scene_scores:
        leaderboard_est = float(np.mean([s for _, s in scene_scores]))
        print(f"\n=== ƯỚC LƯỢNG ĐIỂM BẢNG XẾP HẠNG ({len(scene_scores)} scene) ===")
        print("  Đúng cách BTC gộp điểm (\"điểm trung bình của toàn bộ các scene\") — trung "
              "bình cộng Score TỪNG SCENE, không gộp ảnh của các scene lại tính chung.")
        for name, s in scene_scores:
            print(f"    {name}: {s:.4f}")
        print(f"  Score ước lượng (x100 cho dễ so leaderboard) = {leaderboard_est * 100:.4f}  "
              f"(PSNR_max={args.psnr_max} — ước lượng, KHÔNG phải điểm BTC chấm thật)")


if __name__ == "__main__":
    main()
