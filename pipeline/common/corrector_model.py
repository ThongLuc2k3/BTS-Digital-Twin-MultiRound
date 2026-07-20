"""Model 2 — mạng sửa lỗi pixel (image-domain corrector) chạy SAU 3D Gaussian Splatting
(Model 1). Đây là kiến trúc HOÀN TOÀN KHÁC với cơ chế "error-guided refine" (Vòng 2/3,
`06_train_refine.sh`/`apply_error_refine_patch.py`) — cơ chế đó train TIẾP cùng 1 bộ
Gaussian 3D (không tạo model mới); còn ở đây là 1 mạng neural 2D THẬT SỰ riêng biệt,
nhận ảnh RGB render bởi Model 1 làm input, học cách sửa/làm nét dựa trên so sánh với
ảnh train GT thật — không đụng gì tới hình học/Gaussian của Model 1.

Dùng chung module này ở CẢ `09_train_corrector.py` lẫn `10_apply_corrector.py` (không
định nghĩa lại kiến trúc ở mỗi file) — nếu 2 nơi tự định nghĩa `nn.Module` riêng mà lỡ
lệch nhau dù chỉ 1 tham số (vd `num_blocks`), `load_state_dict` có thể silently load sai
lệch (nếu shape trùng hợp khớp) hoặc crash mù mờ — đúng loại lỗi "sai âm thầm" mà toàn bộ
repo này đã nhiều lần trả giá để tránh (xem `docs/PORTED_KNOWLEDGE.md`).

Kiến trúc: fully-convolutional, KHÔNG pooling/downsampling, residual predict-and-add.
Lý do KHÔNG dùng U-Net/pooling: render và GT đã align pixel-for-pixel tuyệt đối (cùng
pose camera, cùng độ phân giải — render() và GT undistort ra đúng cùng lưới pixel), tác
vụ sửa lỗi ở đây là cục bộ (làm nét cạnh, bù chi tiết mảnh mà splat density chưa đủ dày,
sửa alias nhẹ) — không cần receptive field lớn/bottleneck ngữ nghĩa mà pooling+upsample
mang lại, và pooling+upsample còn có nguy cơ sinh checkerboard artifact, ngược hẳn mục
tiêu "ảnh nét". Giữ fully-convolutional (mọi conv `padding=same`, không resize) nghĩa là
CÙNG 1 bộ trọng số áp được cho ẢNH KÍCH THƯỚC BẤT KỲ (quan trọng: train bằng patch cố
định nhưng suy luận (inference) trên ảnh test kích thước thật, xem `10_apply_corrector.py`).

KHÔNG dùng BatchNorm: chuẩn hoá theo batch đi ngược lại mục tiêu giữ đúng giá trị màu
pixel tuyệt đối trong bài toán phục hồi ảnh (image restoration) — cùng lựa chọn đã được
xác nhận trong tài liệu tham khảo (vd EDSR bỏ hẳn BN so với SRResNet), càng đúng hơn khi
dữ liệu train ở đây rất ít đa dạng batch (chỉ vài trăm ảnh/scene).

Lớp tail_residual (Conv2d cuối nhánh residual) được khởi tạo weight/bias = 0 — tại bước
0 (và nếu train bị cấu hình sai/lr quá lớn/dữ liệu hỏng), residual ~ 0 nên output ~ chính
render gốc của Model 1 — nghĩa là 1 corrector chưa train tốt sẽ "xuống cấp nhẹ nhàng" về
đúng ảnh Model 1 gốc, KHÔNG BAO GIỜ tự sinh nhiễu/ảnh vỡ ngay từ đầu — cùng triết lý
"không bao giờ âm thầm sai mà không báo" của toàn repo, ở đây thực hiện bằng kiến trúc
thay vì bằng guard code. Thuộc tính này giữ nguyên dù có thêm nhánh gate bên dưới, vì
`gate * 0 = 0` bất kể gate là bao nhiêu.

NHÁNH GATE (bản đồ độ tin cậy/mức sửa, tự học — theo yêu cầu người dùng: "Model 2 phải
tự suy luận vùng cần sửa" thay vì sửa đều toàn ảnh): thêm 1 đầu ra phụ `tail_gate` (1
kênh, qua sigmoid -> [0,1]) dùng làm HỆ SỐ NHÂN lên residual trước khi cộng vào input:
`output = clamp(input + gate * residual, 0, 1)`. Mạng KHÔNG được giám sát trực tiếp
gate (không có "nhãn vùng nhiễu" nào ở test time — làm gì có GT để biết trước) — gate tự
nổi lên hoàn toàn từ việc tối ưu loss tái tạo (L1/SSIM) kết hợp 1 số hạng phạt thưa nhẹ
trên gate (xem `--gate_sparsity_weight` ở `09_train_corrector.py`): mạng chỉ "được lợi"
khi bật gate=1 ở đúng vùng mà residual thực sự giúp giảm loss nhiều hơn cái giá phải trả
của số hạng phạt — tự nhiên học ra "chỉ sửa vùng cần sửa" mà không cần bất kỳ nhãn/mask
thủ công nào. `forward()` (dùng ở suy luận thường) chỉ trả về ảnh đã sửa; dùng
`forward_with_gate()` khi cần cả gate map (lúc train để tính phạt thưa, hoặc lúc suy
luận muốn xuất heatmap QC bằng mắt — xem `10_apply_corrector.py`).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


class _ResBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.act(self.conv1(x))
        out = self.conv2(out)
        return x + out


class ResidualCorrectorNet(nn.Module):
    """input/output: (B,3,H,W) float32 trong [0,1]. H,W BẤT KỲ (fully-convolutional).

    output = clamp(input + gate * residual, 0, 1), với gate,residual đều suy ra từ
    cùng 1 trunk feature (stem + num_blocks resblock) — xem `forward_with_gate()`.
    """

    def __init__(self, num_blocks: int = 6, channels: int = 64) -> None:
        super().__init__()
        self.num_blocks = num_blocks
        self.channels = channels
        self.stem = nn.Conv2d(3, channels, kernel_size=3, padding=1)
        self.stem_act = nn.ReLU(inplace=True)
        self.blocks = nn.ModuleList([_ResBlock(channels) for _ in range(num_blocks)])
        self.tail_residual = nn.Conv2d(channels, 3, kernel_size=3, padding=1)
        self.tail_gate = nn.Conv2d(channels, 1, kernel_size=3, padding=1)
        # Khởi tạo 0 CÓ CHỦ ĐÍCH (xem docstring đầu file) — KHÔNG phải khởi tạo mặc định
        # của PyTorch (Kaiming uniform khác 0), phải set tay sau khi tạo layer.
        nn.init.zeros_(self.tail_residual.weight)
        nn.init.zeros_(self.tail_residual.bias)

        # BUG THẬT tìm được bằng train thật (không phải suy đoán) — gate bị "sập" về
        # ~0.0000 chỉ sau ~1400 bước, dù --gate_sparsity_weight mặc định chỉ 0.01 (nhỏ).
        # Nguyên nhân: lúc khởi tạo residual=0 (zero-init ở trên) MỌI NƠI, nên
        # d(recon_loss)/d(gate) = d(recon_loss)/d(output) * residual = 0 — loss tái tạo
        # KHÔNG cho gate bất kỳ gradient nào ở bước đầu, trong khi d(gate_sparsity_loss)/
        # d(gate) = gate_sparsity_weight LUÔN LUÔN có gradient thật kéo gate xuống 0. Nếu
        # gate khởi tạo ngẫu nhiên quanh sigmoid(0)=0.5 (mặc định Kaiming trên bias~0), nó
        # bị kéo dần về 0 mà KHÔNG có lực đối trọng nào — và vì output = input + gate*
        # residual, gate càng nhỏ thì d(recon_loss)/d(residual) = d(recon_loss)/d(output)*
        # gate CŨNG càng nhỏ theo, khiến residual càng khó học được gì có ích -> vòng lặp
        # tự củng cố, gate + residual cùng "chết" về 0 (corrector = no-op vĩnh viễn, mất
        # hết tác dụng "tự học vùng cần sửa").
        #
        # Sửa: ép bias tail_gate dương đủ lớn (sigmoid(4.0)~=0.982) để gate KHỞI ĐẦU gần 1
        # — output lúc khởi tạo VẪN = input y hệt (vì residual=0 bất kể gate bằng bao
        # nhiêu, thuộc tính an toàn không đổi), nhưng giờ d(recon_loss)/d(residual) =
        # d(recon_loss)/d(output)*gate ~= 0.98*(gradient thật) NGAY TỪ BƯỚC ĐẦU — residual
        # có cơ hội học ra sửa lỗi có ích TRƯỚC khi phạt thưa có đủ đòn bẩy để bóp gate
        # xuống ở những vùng KHÔNG cần sửa. Chỉ cần set bias (không set weight — muốn
        # gate vẫn phụ thuộc feature cục bộ để phân biệt vùng, không phải hằng số tuyệt
        # đối), khởi tạo mặc định Kaiming của weight vẫn giữ nguyên.
        nn.init.constant_(self.tail_gate.bias, 4.0)

    def _features(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.stem_act(self.stem(x))
        for block in self.blocks:
            feat = block(feat)
        return feat

    def forward_with_gate(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Trả về (output, gate, residual) — dùng lúc train (tính phạt thưa lên gate,
        xem `09_train_corrector.py`) hoặc lúc suy luận muốn xuất heatmap QC (xem
        `10_apply_corrector.py`). gate: (B,1,H,W) trong [0,1] (sigmoid) — 1 = "tin cậy
        cao, sửa mạnh", 0 = "để nguyên render gốc"."""
        feat = self._features(x)
        residual = self.tail_residual(feat)
        gate = torch.sigmoid(self.tail_gate(feat))
        output = torch.clamp(x + gate * residual, 0.0, 1.0)
        return output, gate, residual

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output, _gate, _residual = self.forward_with_gate(x)
        return output

    @property
    def receptive_field_px(self) -> int:
        """Bán kính receptive field 1 phía (dùng để chọn tile_overlap ở
        `10_apply_corrector.py` — overlap PHẢI >= giá trị này để tránh seam do thiếu
        context ở biên tile). stem (1) + mỗi resblock 2 conv 3x3 (mỗi conv +1px bán kính)
        -> tổng = 1 + num_blocks*2 (bỏ qua tail vì nó chỉ tổng hợp feature tại đúng vị trí
        pixel, không mở rộng thêm receptive field không gian)."""
        return 1 + self.num_blocks * 2


def build_corrector(num_blocks: int = 6, channels: int = 64) -> ResidualCorrectorNet:
    return ResidualCorrectorNet(num_blocks=num_blocks, channels=channels)


def _gaussian_window(window_size: int, sigma: float, device, dtype) -> torch.Tensor:
    coords = torch.arange(window_size, dtype=dtype, device=device) - window_size // 2
    g = torch.exp(-(coords**2) / (2 * sigma**2))
    g = g / g.sum()
    window_2d = g.unsqueeze(0) * g.unsqueeze(1)
    return window_2d


def ssim(img1: torch.Tensor, img2: torch.Tensor, window_size: int = 11) -> torch.Tensor:
    """SSIM khả vi (differentiable), thuần `torch` — KHÔNG dùng
    `skimage.metrics.structural_similarity` (numpy-only, không backprop được, đã dùng ở
    `04_eval_metrics.py` cho mục đích ĐO Score cuối, khác mục đích LOSS lúc train ở đây).
    Công thức chuẩn Wang et al. 2004, dynamic range L=1.0 (ảnh input đã chuẩn hoá [0,1]).
    Trả về SSIM trung bình toàn ảnh, giá trị trong [-1, 1] (1 = giống hệt nhau)."""
    if img1.shape != img2.shape:
        raise ValueError(f"ssim: shape lệch nhau {img1.shape} != {img2.shape}")
    channels = img1.shape[1]
    window_2d = _gaussian_window(window_size, sigma=1.5, device=img1.device, dtype=img1.dtype)
    window = window_2d.expand(channels, 1, window_size, window_size).contiguous()
    pad = window_size // 2

    mu1 = F.conv2d(img1, window, padding=pad, groups=channels)
    mu2 = F.conv2d(img2, window, padding=pad, groups=channels)
    mu1_sq, mu2_sq, mu1_mu2 = mu1 * mu1, mu2 * mu2, mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=pad, groups=channels) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=pad, groups=channels) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=pad, groups=channels) - mu1_mu2

    L = 1.0
    c1, c2 = (0.01 * L) ** 2, (0.03 * L) ** 2
    ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / (
        (mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2)
    )
    return ssim_map.mean()


def save_checkpoint(
    path: Path,
    model: ResidualCorrectorNet,
    optimizer: torch.optim.Optimizer,
    step: int,
    train_args: dict[str, Any],
    dataset_source_iteration: int,
    loss_history: list[dict[str, Any]],
) -> None:
    from datetime import datetime, timezone

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    created_at = train_args.get("_created_at_utc") or datetime.now(timezone.utc).isoformat()
    payload = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "step": step,
        "num_blocks": model.num_blocks,
        "channels": model.channels,
        "train_args": train_args,
        "dataset_source_iteration": dataset_source_iteration,
        "loss_history": loss_history,
        "created_at_utc": created_at,
        "last_updated_utc": datetime.now(timezone.utc).isoformat(),
    }
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp_path)
    tmp_path.replace(path)  # ghi nguyên tử — tránh checkpoint hỏng nếu crash giữa lúc ghi


def load_checkpoint(
    path: Path,
    model: ResidualCorrectorNet,
    optimizer: torch.optim.Optimizer | None = None,
    map_location: str = "cpu",
) -> dict[str, Any]:
    payload = torch.load(Path(path), map_location=map_location, weights_only=False)
    if payload.get("num_blocks") != model.num_blocks or payload.get("channels") != model.channels:
        raise ValueError(
            f"{path}: checkpoint được train với num_blocks={payload.get('num_blocks')}, "
            f"channels={payload.get('channels')} nhưng model hiện tại là "
            f"num_blocks={model.num_blocks}, channels={model.channels} — kiến trúc lệch nhau, "
            "load_state_dict() sẽ sai. Dùng đúng --num_blocks/--channels lúc tạo model để khớp "
            "checkpoint, hoặc --overwrite nếu cố ý muốn đổi kiến trúc (sẽ train lại từ đầu)."
        )
    model.load_state_dict(payload["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in payload:
        optimizer.load_state_dict(payload["optimizer_state_dict"])
    return payload
