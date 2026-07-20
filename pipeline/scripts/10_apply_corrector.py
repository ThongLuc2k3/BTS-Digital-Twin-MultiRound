#!/usr/bin/env python3
"""Áp Model 2 (`ResidualCorrectorNet` đã train bởi `09_train_corrector.py`) lên ảnh
render THẬT của Model 1 tại pose `test_poses.csv` (output của `03_render_test_poses.py`,
KHÔNG SỬA script đó) — bước CUỐI trước khi đóng gói submission.

Chạy suy luận (inference) kiểu TILED (cắt ảnh thành các ô vuông có chồng lấn biên,
chạy mạng từng ô, ghép lại có trộn mượt ở vùng chồng lấn) thay vì đưa nguyên ảnh vào
mạng 1 lần — ảnh drone gốc có thể rất lớn, tiled tránh OOM trên GPU chia sẻ của Kaggle.
Mạng fully-convolutional (không pooling/resize, xem `corrector_model.py`) nên an toàn
khi chạy trên ô kích thước khác lúc train (patch cố định) — chỉ cần overlap >= receptive
field của mạng (`model.receptive_field_px`) để không bị thiếu ngữ cảnh ở biên ô.

Cách dùng:
    python 10_apply_corrector.py --scene HCM0421

Input:  pipeline/work/<scene>/renders/<stem>.png       (output CÓ SẴN của 03_render_test_poses.py)
        pipeline/work/<scene>/corrector_model/corrector.pt  (checkpoint Model 2, nếu có)
Output: pipeline/work_corrected/<scene>/renders/<stem>.png
        pipeline/work_corrected/<scene>/gate_maps/<stem>.png   (heatmap gate — TRẮNG = Model 2
            tự chọn sửa mạnh, ĐEN = gần như giữ nguyên render gốc; --no_save_gate_map để tắt)
        pipeline/work_corrected/<scene>/10_apply_corrector.log

Đóng gói submission từ đây: dùng
    python 07_package_submission.py --renders_root pipeline/work_corrected --out submission.zip
(KHÔNG SỬA 07_package_submission.py — script đó vốn đã có sẵn `--renders_root` để trỏ
sang thư mục render khác thay vì `pipeline/work` mặc định).

Scene CHƯA có corrector (chưa chạy 08/09 cho scene đó): mặc định COPY NGUYÊN render
Model 1 sang `work_corrected/` không sửa gì (để `07_package_submission.py --renders_root
pipeline/work_corrected` vẫn đóng gói được ĐỦ 7 scene dù chỉ 1 vài scene có corrector) —
dùng `--strict` nếu muốn báo lỗi cứng thay vì âm thầm copy-through.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image as PILImage

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.scenes import get_scene
from common.poses import read_test_poses
from common.logging_utils import FileLog
from common.corrector_model import build_corrector


def compute_tile_starts(total: int, tile: int, overlap: int) -> list[int]:
    """Vị trí bắt đầu (0-based) của mỗi ô dọc 1 trục, kích thước ô CỐ ĐỊNH = tile
    (tile đã được clamp <= total trước khi gọi hàm này), ô cuối LUÔN chạm đúng biên
    total (không tràn, không thiếu) — nếu total <= tile, chỉ 1 ô duy nhất bao hết."""
    if total <= tile:
        return [0]
    stride = max(1, tile - overlap)
    starts = list(range(0, total - tile + 1, stride))
    if starts[-1] != total - tile:
        starts.append(total - tile)
    return starts


def ramp_weight(length: int, overlap: int, has_prev: bool, has_next: bool) -> np.ndarray:
    """Trọng số trộn biên (feather) cho 1 ô theo 1 trục — chỉ ramp 0->1 ở phía CÓ ô kề
    (has_prev/has_next), để pixel ở đúng biên ẢNH THẬT (không có ô kề) giữ trọng số=1,
    không bị mờ/tối do "trộn với hư không"."""
    w = np.ones(length, dtype=np.float32)
    if has_prev and overlap > 0:
        n = min(overlap, length)
        w[:n] = np.minimum(w[:n], np.linspace(0.0, 1.0, n, dtype=np.float32))
    if has_next and overlap > 0:
        n = min(overlap, length)
        w[-n:] = np.minimum(w[-n:], np.linspace(1.0, 0.0, n, dtype=np.float32))
    return w


def apply_tiled(model, img01: np.ndarray, tile_size: int, overlap: int, device: str,
                 return_gate: bool = False):
    """img01: (H,W,3) float32 [0,1]. Trả về ảnh đã sửa, cùng shape.

    Nếu `return_gate=True`, trả về thêm (H,W,1) gate map đã ghép/trộn y hệt cách ghép
    ảnh chính — dùng để xuất heatmap QC bằng mắt (xem `main()`), cho thấy Model 2 TỰ
    CHỌN sửa mạnh (gate~1) ở vùng nào, gần như không sửa (gate~0) ở vùng nào (đúng yêu
    cầu "tự suy luận vùng cần sửa" thay vì sửa đều toàn ảnh — xem `corrector_model.py`).
    Model không có `forward_with_gate` (vd model giả dùng trong test) sẽ được coi như
    "luôn sửa" (gate=1 khắp nơi) để tương thích ngược, không bắt buộc mọi model truyền
    vào đây phải có nhánh gate."""
    H, W, _ = img01.shape
    th, tw = min(tile_size, H), min(tile_size, W)
    y_starts = compute_tile_starts(H, th, overlap)
    x_starts = compute_tile_starts(W, tw, overlap)

    acc = np.zeros((H, W, 3), dtype=np.float32)
    gate_acc = np.zeros((H, W, 1), dtype=np.float32)
    wsum = np.zeros((H, W, 1), dtype=np.float32)
    has_gate_api = hasattr(model, "forward_with_gate")

    with torch.no_grad():
        for yi, y0 in enumerate(y_starts):
            for xi, x0 in enumerate(x_starts):
                tile = img01[y0:y0 + th, x0:x0 + tw]
                t = torch.from_numpy(tile).permute(2, 0, 1).unsqueeze(0).to(device)
                if has_gate_api:
                    out_t, gate_t, _residual_t = model.forward_with_gate(t)
                else:
                    out_t = model(t)
                    gate_t = torch.ones_like(out_t[:, :1])
                out = out_t.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
                gate_np = gate_t.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()

                wy = ramp_weight(th, overlap, has_prev=yi > 0, has_next=yi < len(y_starts) - 1)
                wx = ramp_weight(tw, overlap, has_prev=xi > 0, has_next=xi < len(x_starts) - 1)
                w2d = (wy[:, None] * wx[None, :])[:, :, None]

                acc[y0:y0 + th, x0:x0 + tw] += out * w2d
                gate_acc[y0:y0 + th, x0:x0 + tw] += gate_np * w2d
                wsum[y0:y0 + th, x0:x0 + tw] += w2d

    if np.any(wsum <= 0):
        raise RuntimeError("apply_tiled: có pixel không được ô nào phủ tới — lỗi logic ghép ô.")
    corrected = np.clip(acc / wsum, 0.0, 1.0)
    if return_gate:
        return corrected, np.clip(gate_acc / wsum, 0.0, 1.0)
    return corrected


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True)
    ap.add_argument("--corrector_checkpoint", default=None,
                     help="Mặc định pipeline/work/<scene>/corrector_model/corrector.pt")
    ap.add_argument("--renders_in_dir", default=None, help="Mặc định pipeline/work/<scene>/renders")
    ap.add_argument("--out_dir", default=None, help="Mặc định pipeline/work_corrected/<scene>/renders")
    ap.add_argument("--tile_size", type=int, default=512)
    ap.add_argument("--tile_overlap", type=int, default=32)
    ap.add_argument("--strict", action="store_true",
                     help="Báo lỗi cứng nếu thiếu checkpoint corrector, thay vì copy-through render gốc.")
    ap.add_argument("--save_gate_map", dest="save_gate_map", action="store_true", default=True,
                     help="Lưu thêm heatmap gate (vùng Model 2 TỰ CHỌN sửa mạnh/nhẹ) để QC bằng mắt "
                          "(mặc định BẬT — chi phí rẻ, xem Bước 12 kaggle_pixel_corrector.ipynb).")
    ap.add_argument("--no_save_gate_map", dest="save_gate_map", action="store_false")
    args = ap.parse_args()

    scene = get_scene(args.scene)
    pipeline_root = Path(__file__).resolve().parents[1]
    ckpt_path = Path(args.corrector_checkpoint) if args.corrector_checkpoint else (
        pipeline_root / "work" / scene.name / "corrector_model" / "corrector.pt"
    )
    renders_in_dir = Path(args.renders_in_dir) if args.renders_in_dir else pipeline_root / "work" / scene.name / "renders"
    out_dir = Path(args.out_dir) if args.out_dir else pipeline_root / "work_corrected" / scene.name / "renders"
    out_dir.mkdir(parents=True, exist_ok=True)
    gate_dir = out_dir.parent / "gate_maps"
    log = FileLog(out_dir.parent / "10_apply_corrector.log")

    if not renders_in_dir.exists():
        raise SystemExit(
            f"Không thấy {renders_in_dir} — chạy 03_render_test_poses.py --scene {scene.name} trước."
        )
    expected = {Path(p.image_name).stem: (p.width, p.height) for p in read_test_poses(scene.test_poses_csv)}

    if not ckpt_path.exists():
        if args.strict:
            raise SystemExit(f"--strict: không thấy corrector checkpoint tại {ckpt_path}.")
        print(f"[bỏ qua sửa lỗi] {scene.name}: chưa có corrector ({ckpt_path}) — copy nguyên render "
              f"Model 1 sang {out_dir} không sửa gì.")
        n_copied = 0
        for stem in sorted(expected):
            src = renders_in_dir / f"{stem}.png"
            if not src.exists():
                raise SystemExit(f"{scene.name}: thiếu render {src} (chạy 03_render_test_poses.py trước).")
            with PILImage.open(src) as im:
                im.convert("RGB").save(out_dir / f"{stem}.png", format="PNG")
            n_copied += 1
        log.write(f"copy-through (không có corrector): {n_copied} ảnh.")
        log.close()
        print(f"-> Đã copy {n_copied} ảnh (không sửa) sang {out_dir}")
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    payload = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = build_corrector(num_blocks=payload["num_blocks"], channels=payload["channels"]).to(device)
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    print(f"  Corrector step={payload.get('step')}, num_blocks={payload['num_blocks']}, "
          f"channels={payload['channels']}, receptive_field={model.receptive_field_px}px")
    if args.tile_overlap < model.receptive_field_px:
        print(f"  [CẢNH BÁO] --tile_overlap={args.tile_overlap} < receptive_field mạng "
              f"({model.receptive_field_px}px) — có thể lộ seam nhẹ ở biên ô, nên tăng "
              f"--tile_overlap >= {model.receptive_field_px}.")

    log.write(f"scene={scene.name} checkpoint={ckpt_path} step={payload.get('step')} "
              f"tile_size={args.tile_size} tile_overlap={args.tile_overlap} save_gate_map={args.save_gate_map}")
    if args.save_gate_map:
        gate_dir.mkdir(parents=True, exist_ok=True)

    n_done = 0
    gate_means = []
    for stem, (exp_w, exp_h) in sorted(expected.items()):
        src = renders_in_dir / f"{stem}.png"
        if not src.exists():
            raise SystemExit(f"{scene.name}: thiếu render {src} (chạy 03_render_test_poses.py trước).")
        with PILImage.open(src) as im:
            im = im.convert("RGB")
            if im.size != (exp_w, exp_h):
                raise SystemExit(
                    f"{scene.name}/{stem}: render {im.size} != kích thước yêu cầu ({exp_w},{exp_h}) "
                    f"trong test_poses.csv."
                )
            img01 = np.asarray(im, dtype=np.float32) / 255.0

        if args.save_gate_map:
            corrected, gate_map = apply_tiled(model, img01, args.tile_size, args.tile_overlap, device,
                                               return_gate=True)
            gate_u8 = (gate_map[:, :, 0] * 255.0).round().astype(np.uint8)
            PILImage.fromarray(gate_u8, mode="L").save(gate_dir / f"{stem}.png", format="PNG")
            gate_means.append(float(gate_map.mean()))
        else:
            corrected = apply_tiled(model, img01, args.tile_size, args.tile_overlap, device)

        out_u8 = (corrected * 255.0).round().astype(np.uint8)
        out_im = PILImage.fromarray(out_u8)
        if out_im.size != (exp_w, exp_h):
            raise RuntimeError(f"{scene.name}/{stem}: output {out_im.size} != input ({exp_w},{exp_h}) — "
                                f"lỗi logic tiled inference (mạng phải giữ nguyên kích thước).")
        out_im.save(out_dir / f"{stem}.png", format="PNG")
        n_done += 1
        if n_done % 20 == 0:
            print(f"  [{n_done}/{len(expected)}] {stem}")
        log.write(f"[{n_done}/{len(expected)}] {stem}.png")

    log.write(f"Xong. {n_done} ảnh đã sửa tại {out_dir}")
    log.close()
    print(f"-> Xong {n_done} ảnh (đã sửa bằng Model 2) tại {out_dir}")
    if args.save_gate_map and gate_means:
        print(f"-> gate_mean trung bình toàn scene: {sum(gate_means) / len(gate_means):.4f} "
              f"(heatmap từng ảnh tại {gate_dir}, trắng = sửa mạnh, đen = gần như không sửa)")
    print(f"-> Đóng gói: python 07_package_submission.py --renders_root {out_dir.parent.parent} --out submission.zip")


if __name__ == "__main__":
    main()
