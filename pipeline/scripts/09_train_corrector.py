#!/usr/bin/env python3
"""Train Model 2 (`ResidualCorrectorNet`, xem `pipeline/common/corrector_model.py`) —
mạng sửa lỗi pixel 2D, HOÀN TOÀN RIÊNG BIỆT với Model 1 (3D Gaussian Splatting). Chỉ
cần `torch` thuần (KHÔNG cần GS_REPO/CUDA rasterizer/COLMAP nữa) — đọc cặp (render, GT)
đã cache sẵn bởi `08_build_corrector_dataset.py`.

Model 2 TỰ HỌC vùng cần sửa (nhánh "gate" trong `ResidualCorrectorNet.forward_with_gate()`)
thay vì sửa đều toàn ảnh — không cần nhãn/mask vùng lỗi nào (không có ở test time), chỉ
cần phạt thưa nhẹ lên `mean(gate)` (`--gate_sparsity_weight`) cộng vào loss tái tạo, để
mạng chỉ "được lợi" khi bật sửa mạnh (gate~1) ở đúng vùng residual thực sự giúp giảm
loss. Xem docstring nhánh gate ở `pipeline/common/corrector_model.py` để hiểu đầy đủ.

KHÔNG có holdout/validation set khách quan (quyết định có chủ đích của người dùng repo
này — xem `docs/MILESTONE_15_image_corrector.md` mục rủi ro): mọi cặp (render, GT) đều
dùng để train. Muốn kiểm tra chất lượng, chỉ có thể xem bằng mắt ảnh sau khi áp
`10_apply_corrector.py` lên render test thật — KHÔNG có Score định lượng khách quan cho
nhánh này, khác hẳn cơ chế Vòng 2/3 (`06_train_refine.sh`) vốn luôn tự đo Score holdout.

"Train tiếp lần 2" (đúng ý người dùng: "muốn tinh chỉnh render lần 2 chỉ cần chạy lại
file lần 2"): gọi lại CHÍNH script này với `--resume --steps <N thêm>` — KHÔNG giống cơ
chế đổi tên thư mục iteration luỹ kế của `06_train_refine.sh` (cơ chế đó tồn tại vì
`gaussian-splatting/train.py` không cho ta kiểm soát, luôn đếm lại iteration từ 0). Ở
đây ta tự viết vòng lặp train nên `--steps` LUÔN là "số bước THÊM của lần chạy này" —
số bước LUỸ KẾ thật lưu trong chính field "step" của checkpoint `corrector.pt`, không
cần đổi tên thư mục/file gì cả.

Cách dùng:
    python 09_train_corrector.py --scene HCM0421 --steps 4000        # train lần đầu
    python 09_train_corrector.py --scene HCM0421 --resume --steps 2000   # train tiếp (lần 2, 3, ...)
    python 09_train_corrector.py --scene HCM0421 --overwrite --steps 4000  # huỷ, train lại từ đầu

Output:
    pipeline/work/<scene>/corrector_model/corrector.pt       (checkpoint model+optimizer+lịch sử loss)
    pipeline/work/<scene>/corrector_model/corrector_train.log
"""
import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image as PILImage
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.scenes import get_scene
from common.logging_utils import FileLog
from common.corrector_model import build_corrector, save_checkpoint, load_checkpoint, ssim


class CorrectorPatchDataset(Dataset):
    """Lấy patch vuông NGẪU NHIÊN, ĐỒNG BỘ giữa render và GT (cùng toạ độ crop, cùng
    lật ngang/dọc nếu có) — lật KHÔNG làm hỏng tín hiệu học ở đây vì mạng không có giả
    định về hướng/ngữ nghĩa cảnh (thuần hàm sửa pixel cục bộ), giúp tăng đa dạng dữ liệu
    khi mỗi scene chỉ có vài trăm ảnh. `__len__` chỉ định nghĩa độ dài 1 "epoch" cho
    DataLoader — thực chất lấy mẫu vô hạn (mỗi lần gọi là 1 crop ngẫu nhiên mới)."""

    def __init__(self, dataset_dir: Path, patch_size: int, patches_per_image: int = 50, seed: int = 42):
        self.render_dir = dataset_dir / "render"
        self.gt_dir = dataset_dir / "gt"
        self.stems = sorted(p.stem for p in self.render_dir.glob("*.png"))
        if not self.stems:
            raise SystemExit(
                f"Không tìm thấy ảnh nào trong {self.render_dir} — chạy "
                f"08_build_corrector_dataset.py --scene <scene> trước."
            )
        self.patch_size = patch_size
        self.patches_per_image = patches_per_image
        self._rng = random.Random(seed)

    def __len__(self) -> int:
        return len(self.stems) * self.patches_per_image

    def __getitem__(self, idx: int):
        stem = self.stems[idx % len(self.stems)]
        render_im = PILImage.open(self.render_dir / f"{stem}.png").convert("RGB")
        gt_im = PILImage.open(self.gt_dir / f"{stem}.png").convert("RGB")
        if render_im.size != gt_im.size:
            raise ValueError(f"{stem}: kích thước render {render_im.size} != GT {gt_im.size}.")
        w, h = render_im.size
        p = self.patch_size
        if w < p or h < p:
            raise ValueError(
                f"{stem}: ảnh {w}x{h} nhỏ hơn --patch_size={p} — giảm --patch_size xuống <= "
                f"{min(w, h)}."
            )
        x0 = self._rng.randint(0, w - p)
        y0 = self._rng.randint(0, h - p)
        render_patch = render_im.crop((x0, y0, x0 + p, y0 + p))
        gt_patch = gt_im.crop((x0, y0, x0 + p, y0 + p))
        if self._rng.random() < 0.5:
            render_patch = render_patch.transpose(PILImage.FLIP_LEFT_RIGHT)
            gt_patch = gt_patch.transpose(PILImage.FLIP_LEFT_RIGHT)
        if self._rng.random() < 0.5:
            render_patch = render_patch.transpose(PILImage.FLIP_TOP_BOTTOM)
            gt_patch = gt_patch.transpose(PILImage.FLIP_TOP_BOTTOM)
        render_t = torch.from_numpy(np.asarray(render_patch, dtype=np.float32) / 255.0).permute(2, 0, 1)
        gt_t = torch.from_numpy(np.asarray(gt_patch, dtype=np.float32) / 255.0).permute(2, 0, 1)
        return render_t, gt_t


def _endless_batches(loader: DataLoader):
    while True:
        for batch in loader:
            yield batch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True)
    ap.add_argument("--dataset_dir", default=None, help="Mặc định pipeline/work/<scene>/corrector_dataset")
    ap.add_argument("--out_dir", default=None, help="Mặc định pipeline/work/<scene>/corrector_model")
    ap.add_argument("--steps", type=int, default=4000, help="Số bước train THÊM của lần chạy này (không phải luỹ kế)")
    ap.add_argument("--patch_size", type=int, default=256)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--loss", choices=["l1", "l1_ssim"], default="l1")
    ap.add_argument("--ssim_weight", type=float, default=0.2, help="Chỉ dùng nếu --loss l1_ssim")
    ap.add_argument("--gate_sparsity_weight", type=float, default=0.01,
                     help="Trọng số phạt thưa lên gate map (mean(gate)) — buộc mạng TỰ HỌC chỉ bật "
                          "gate=1 (sửa mạnh) ở vùng thực sự cần, thay vì sửa đều toàn ảnh (gate=1 khắp "
                          "nơi cũng làm giảm loss tái tạo nếu không bị phạt). 0 = tắt phạt (mạng có thể "
                          "tự do sửa toàn ảnh nếu thấy có lợi cho loss).")
    ap.add_argument("--num_blocks", type=int, default=6)
    ap.add_argument("--channels", type=int, default=64)
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--log_every", type=int, default=100)
    ap.add_argument("--save_every", type=int, default=500, help="Lưu checkpoint giữa chừng mỗi N bước (phòng crash)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--resume", action="store_true", help="Nạp checkpoint hiện có, train tiếp thêm --steps bước")
    ap.add_argument("--overwrite", action="store_true", help="Xoá checkpoint hiện có, train lại từ đầu")
    ap.add_argument("--allow_dataset_mismatch", action="store_true",
                     help="Bỏ qua kiểm tra checkpoint corrector đang resume có khớp đúng dataset "
                          "(iteration Model 1) hiện tại hay không — CHỈ dùng nếu chắc chắn hiểu rủi ro.")
    args = ap.parse_args()

    if args.resume and args.overwrite:
        raise SystemExit("--resume và --overwrite loại trừ nhau — chỉ chọn 1.")

    scene = get_scene(args.scene)
    pipeline_root = Path(__file__).resolve().parents[1]
    dataset_dir = Path(args.dataset_dir) if args.dataset_dir else pipeline_root / "work" / scene.name / "corrector_dataset"
    out_dir = Path(args.out_dir) if args.out_dir else pipeline_root / "work" / scene.name / "corrector_model"
    ckpt_path = out_dir / "corrector.pt"
    log_path = out_dir / "corrector_train.log"

    manifest_path = dataset_dir / "manifest.json"
    if not manifest_path.exists():
        raise SystemExit(
            f"Không thấy {manifest_path} — chạy 08_build_corrector_dataset.py --scene {scene.name} trước."
        )
    manifest = json.loads(manifest_path.read_text())
    dataset_source_iteration = manifest["source_iteration"]

    if ckpt_path.exists():
        if args.overwrite:
            ckpt_path.unlink()
            if log_path.exists():
                log_path.unlink()
            print(f"[dọn dẹp] Đã xoá checkpoint/log cũ tại {out_dir} — train lại từ đầu.")
        elif not args.resume:
            raise SystemExit(
                f"Đã có checkpoint tại {ckpt_path} — dùng --resume để train tiếp (thêm --steps bước "
                f"nữa), hoặc --overwrite để huỷ và train lại từ đầu."
            )
    elif args.resume:
        raise SystemExit(f"--resume nhưng không thấy checkpoint nào tại {ckpt_path} — bỏ --resume để train lần đầu.")

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("[CẢNH BÁO] Không có GPU CUDA — train trên CPU sẽ RẤT chậm, chỉ nên dùng để test nhanh "
              "với --steps nhỏ.")

    model = build_corrector(num_blocks=args.num_blocks, channels=args.channels).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    step_start = 0
    loss_history: list[dict] = []
    if args.resume:
        payload = load_checkpoint(ckpt_path, model, optimizer, map_location=device)
        prev_source_iteration = payload.get("dataset_source_iteration")
        if prev_source_iteration != dataset_source_iteration and not args.allow_dataset_mismatch:
            raise SystemExit(
                f"Checkpoint {ckpt_path} được train trên dataset ứng với checkpoint Model 1 "
                f"iteration={prev_source_iteration}, nhưng corrector_dataset hiện tại ứng với "
                f"iteration={dataset_source_iteration} (Model 1 có vẻ đã được train lại/khác) — "
                f"train tiếp trên dữ liệu không khớp sẽ làm hỏng corrector đang có. Chạy lại "
                f"08_build_corrector_dataset.py cho đúng checkpoint Model 1 muốn dùng, hoặc dùng "
                f"--allow_dataset_mismatch nếu CHẮC CHẮN cố ý (rủi ro tự chịu)."
            )
        prev_args = payload.get("train_args", {})
        for key in ("patch_size", "loss", "num_blocks", "channels"):
            if key in prev_args and str(prev_args[key]) != str(getattr(args, key)):
                print(f"  [CẢNH BÁO] --{key}={getattr(args, key)} khác giá trị lần train trước "
                      f"({prev_args[key]}) — vẫn tiếp tục, nhưng kết quả có thể không nhất quán.")
        step_start = payload["step"]
        loss_history = payload.get("loss_history", [])
        print(f"  Resume từ checkpoint step={step_start} ({ckpt_path}).")

    dataset = CorrectorPatchDataset(dataset_dir, patch_size=args.patch_size, seed=args.seed)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                         num_workers=args.num_workers, drop_last=True)
    batches = _endless_batches(loader)

    l1_loss_fn = nn.L1Loss()
    log = FileLog(log_path)
    log.write(f"scene={scene.name} dataset_source_iteration={dataset_source_iteration} "
              f"step_start={step_start} steps_this_run={args.steps} loss={args.loss} "
              f"patch_size={args.patch_size} batch_size={args.batch_size} lr={args.lr}")
    print(f"===== {scene.name}: train Model 2 (corrector) — step {step_start} -> "
          f"{step_start + args.steps} (loss={args.loss}) =====")

    model.train()
    target_step = step_start + args.steps
    step = step_start
    while step < target_step:
        render_batch, gt_batch = next(batches)
        render_batch = render_batch.to(device)
        gt_batch = gt_batch.to(device)

        output, gate, _residual = model.forward_with_gate(render_batch)
        l1 = l1_loss_fn(output, gt_batch)
        if args.loss == "l1_ssim":
            ssim_val = ssim(output, gt_batch)
            recon_loss = (1.0 - args.ssim_weight) * l1 + args.ssim_weight * (1.0 - ssim_val)
        else:
            recon_loss = l1
        gate_mean = gate.mean()
        # Phạt thưa CÓ CHỦ ĐÍCH lên gate — không có phạt này, gate=1 khắp ảnh cũng tối
        # ưu recon_loss ngang hoặc tốt hơn (không có gì "phạt" việc sửa lan tràn), mạng
        # sẽ không tự học ra khoanh vùng dù thừa khả năng biểu diễn. Đây là cơ chế DUY
        # NHẤT khiến mạng "tự suy luận vùng cần sửa" thay vì sửa đều toàn ảnh — xem
        # docstring nhánh gate ở corrector_model.py.
        loss = recon_loss + args.gate_sparsity_weight * gate_mean

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        step += 1

        if step % args.log_every == 0 or step == target_step:
            entry = {"step": step, "loss": float(loss.item()), "l1": float(l1.item()),
                     "gate_mean": float(gate_mean.item())}
            loss_history.append(entry)
            msg = (f"[{step}/{target_step}] loss={loss.item():.5f} l1={l1.item():.5f} "
                   f"gate_mean={gate_mean.item():.4f}")
            print(f"  {msg}")
            log.write(msg)

        if step % args.save_every == 0 or step == target_step:
            save_checkpoint(
                ckpt_path, model, optimizer, step,
                train_args={"patch_size": args.patch_size, "batch_size": args.batch_size, "lr": args.lr,
                            "loss": args.loss, "ssim_weight": args.ssim_weight,
                            "gate_sparsity_weight": args.gate_sparsity_weight,
                            "num_blocks": args.num_blocks, "channels": args.channels, "seed": args.seed},
                dataset_source_iteration=dataset_source_iteration,
                loss_history=loss_history,
            )

    log.write(f"Xong. checkpoint cuối tại step={step}: {ckpt_path}")
    log.close()
    print(f"-> Xong. Checkpoint Model 2: {ckpt_path} (step={step})")
    print(f"-> Muốn train tiếp: python 09_train_corrector.py --scene {scene.name} --resume --steps <N>")


if __name__ == "__main__":
    main()
