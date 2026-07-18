#!/usr/bin/env python3
"""Vá `train.py` của repo GỐC graphdeco-inria/gaussian-splatting để hỗ trợ cơ chế
"tinh chỉnh có hướng dẫn bởi lỗi" (error-guided refine — CỐT LÕI của repo này, xem
`docs/00_MASTER_PLAN.md` mục 3.2 + `docs/PORTED_KNOWLEDGE.md` mục 3): nạp lại 1
checkpoint ĐÃ TRAIN (chỉ cần `point_cloud.ply`, KHÔNG cần `.pth` optimizer state) rồi
train tiếp 1 đợt NGẮN, ưu tiên loss vào đúng vùng pixel còn lỗi cao (đo được bằng cách
render lại chính ảnh train + so ảnh GT thật — xem `05_generate_error_mask.py`, chạy
TRƯỚC script này để sinh mask; `06_train_refine.sh` gọi cả 2 script này theo đúng thứ
tự + tự dò cấu hình checkpoint gốc).

Cơ chế này PORT NGUYÊN VẸN từ repo `BTS-Digital-Twin` (đã build + verify cục bộ ngày
2026-07-18, xem `docs/PORTED_KNOWLEDGE.md`) — dùng LẠI được cho MỌI vòng refine (Vòng
2 tinh chỉnh Vòng 1, Vòng 3 tinh chỉnh Vòng 2, ...), không phải 1 patch riêng cho mỗi
vòng — vì bản chất chỉ là "nạp .ply có sẵn + train tiếp", không quan tâm .ply đó tới
từ vòng nào.

Ý TƯỞNG KỸ THUẬT (đã đối chiếu trực tiếp source thật, không suy đoán — xem
scene/__init__.py::Scene.__init__ tại commit đã pin
54c035f7834b564019656c3e3fcc3646292f727d):
  - `Scene(dataset, gaussians, load_iteration=N)` là cơ chế CÓ SẴN, CHÍNH THỨC của repo
    (bình thường chỉ dùng để RENDER/inference, KHÔNG dùng để train tiếp) — gọi thẳng
    `gaussians.load_ply(".../iteration_N/point_cloud.ply")`, KHÔNG cần file `.pth`
    (khác hẳn `--start_checkpoint`, vốn cần `.pth` do `--checkpoint_iterations` sinh ra
    — checkpoint thường có từ `03_train_3dgs.sh` KHÔNG có file này, chỉ có `.ply`).
  - CẠM BẪY đã tự phát hiện + tự vá: khi đi qua nhánh `load_iteration` này,
    `create_from_pcd()` (nơi DUY NHẤT set `gaussians.spatial_lr_scale`) KHÔNG được gọi
    — `spatial_lr_scale` giữ nguyên mặc định `0` từ `__init__`. Vì learning rate vị trí
    Gaussian = `position_lr_init * spatial_lr_scale`, nếu không tự gán lại, vị trí
    Gaussian sẽ ĐỨNG YÊN 100% suốt đợt tinh chỉnh (learning rate = 0), coi như không
    train gì cả dù không báo lỗi. Đã tự vá: gán lại
    `gaussians.spatial_lr_scale = scene.cameras_extent` — ĐÚNG giá trị
    `Scene.__init__` tự truyền cho `create_from_pcd()` ở nhánh train-from-scratch bình
    thường (`self.cameras_extent = scene_info.nerf_normalization["radius"]`).
  - Densify: khuyến nghị dùng `--densify_until_iter 0` (cờ CÓ SẴN của repo, không cần
    vá) khi chạy đợt tinh chỉnh — không muốn sinh thêm Gaussian mới, chỉ tối ưu lại vị
    trí/màu/opacity Gaussian đã có ở vùng lỗi cao (`06_train_refine.sh` tự set cờ này).
  - `--iterations M` (M = ngân sách tinh chỉnh, vd 3000) là số iteration TUYỆT ĐỐI của
    RIÊNG đợt train.py này (vì không truyền `--start_checkpoint .pth`, `first_iter=0`
    trong `train.py::training()` — đối chiếu trực tiếp source, không suy đoán) — nghĩa
    là mỗi đợt refine chạy 1 lịch learning-rate MỚI từ 0..M, KHÔNG cộng dồn với số
    iteration của checkpoint gốc. `train.py` tự lưu kết quả vào
    `point_cloud/iteration_M/`. `06_train_refine.sh` tự đổi tên thư mục này thành
    `point_cloud/iteration_<CKPT_ITERATION + M>/` NGAY SAU khi train xong — để số
    iteration trong tên thư mục phản ánh ĐÚNG luỹ kế số vòng đã tinh chỉnh (tránh 2
    vòng refine liên tiếp cùng vô tình ghi ra CÙNG 1 tên thư mục `iteration_3000` mà
    không phân biệt được vòng nào — xem comment trong `06_train_refine.sh`).

Cách dùng:
    python apply_error_refine_patch.py --gs_repo /kaggle/working/gaussian-splatting

Sau khi vá, train tiếp bằng (KHÔNG cần gọi tay — `06_train_refine.sh` làm việc này):
    python train.py -s <dense_dir> -m <model_dir> \\
        --refine_from_iteration <N>            # N = iteration của checkpoint .ply có sẵn
        --error_mask_dir <thư mục mask>          # do 05_generate_error_mask.py sinh
        --iterations <M>                         # ngân sách tinh chỉnh (vd 3000), NHỎ hơn
                                                  # nhiều so với train gốc — đây là fine-tune,
                                                  # không phải train lại từ đầu
        --densify_until_iter 0                   # không sinh thêm Gaussian mới
        --save_iterations <M> --test_iterations <M> \\
        <các cờ khác GIỐNG HỆT lúc train checkpoint gốc — antialiasing/sh_degree/...>

An toàn: mỗi đoạn vá đối chiếu ĐÚNG 1 lần với text gốc — nếu không khớp (repo gốc đổi
code, hoặc file đã bị vá trước đó bởi patch khác) sẽ báo lỗi rõ ràng và dừng lại, KHÔNG
âm thầm bỏ qua.

CHƯA TEST TRÊN GPU THẬT (không có GPU cục bộ) — đã verify: patch áp sạch lên bản clone
thật + `python -m py_compile` sau vá không lỗi cú pháp + đối chiếu logic
`spatial_lr_scale`/`load_iteration`/`first_iter` trực tiếp với source thật (không suy
đoán). Vẫn cần 1 lần chạy GPU thật (Kaggle) để xác nhận đợt tinh chỉnh thực sự cải
thiện Score — xem `kaggle_round2_refine.ipynb`/`kaggle_round3_refine.ipynb`, đo Score
TRƯỚC/SAU tự động, không tin bằng trực giác.
"""
import argparse
from pathlib import Path

PATCHES: list[tuple[str, str, str]] = [
    (
        "import numpy + PIL cho doc error mask",
        "import os\nimport torch\nfrom random import randint\n",
        "import os\nimport torch\nimport numpy as np\nfrom PIL import Image as _ErrMaskImage\nfrom random import randint\n",
    ),
    (
        "training(): them 2 tham so refine_from_iteration/error_mask_dir",
        "def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from):",
        "def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from, refine_from_iteration=None, error_mask_dir=None):",
    ),
    (
        "call site: truyen refine_from_iteration/error_mask_dir",
        "    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from)",
        "    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from, args.refine_from_iteration, args.error_mask_dir)",
    ),
    (
        "argparse: --refine_from_iteration/--error_mask_dir",
        '    parser.add_argument("--start_checkpoint", type=str, default = None)\n'
        '    args = parser.parse_args(sys.argv[1:])',
        '    parser.add_argument("--start_checkpoint", type=str, default = None)\n'
        '    parser.add_argument("--refine_from_iteration", type=int, default=None,\n'
        '                        help="Nap point_cloud/iteration_<N>/point_cloud.ply co san (KHONG can .pth) roi train tiep")\n'
        '    parser.add_argument("--error_mask_dir", type=str, default=None,\n'
        '                        help="Thu muc mask 16-bit PNG do 05_generate_error_mask.py sinh, uu tien loss vao vung loi cao")\n'
        '    args = parser.parse_args(sys.argv[1:])',
    ),
    (
        "Scene load_iteration + fix spatial_lr_scale + ham doc error mask",
        "    gaussians = GaussianModel(dataset.sh_degree, opt.optimizer_type)\n"
        "    scene = Scene(dataset, gaussians)\n"
        "    gaussians.training_setup(opt)",
        "    gaussians = GaussianModel(dataset.sh_degree, opt.optimizer_type)\n"
        "    scene = Scene(dataset, gaussians, load_iteration=refine_from_iteration)\n"
        "    if refine_from_iteration:\n"
        "        # xem docstring apply_error_refine_patch.py: Scene(load_iteration=...) di qua\n"
        "        # gaussians.load_ply(), KHONG qua create_from_pcd() -> spatial_lr_scale KHONG\n"
        "        # duoc tu dong set, phai gan lai dung gia tri Scene.__init__ da tinh.\n"
        "        gaussians.spatial_lr_scale = scene.cameras_extent\n"
        "        print(f\"[error-refine] Nap Gaussian tu iteration {refine_from_iteration} (.ply), \"\n"
        "              f\"spatial_lr_scale={scene.cameras_extent:.6f}\")\n"
        "    gaussians.training_setup(opt)\n"
        "\n"
        "    _error_mask_cache = {}\n"
        "    _ERROR_MASK_SCALE = 1000.0  # DUNG HANG SO nay o ca 05_generate_error_mask.py\n"
        "\n"
        "    def _error_mask(cam):\n"
        "        if not error_mask_dir:\n"
        "            return None\n"
        "        if cam.image_name in _error_mask_cache:\n"
        "            return _error_mask_cache[cam.image_name]\n"
        "        stem = os.path.splitext(cam.image_name)[0]\n"
        "        mask_path = os.path.join(error_mask_dir, stem + \".png\")\n"
        "        if not os.path.exists(mask_path):\n"
        "            _error_mask_cache[cam.image_name] = None\n"
        "            return None\n"
        "        arr = np.array(_ErrMaskImage.open(mask_path)).astype(np.float32)\n"
        "        weight = arr / _ERROR_MASK_SCALE\n"
        "        t = torch.from_numpy(weight).unsqueeze(0).cuda()\n"
        "        if t.shape[1] != cam.image_height or t.shape[2] != cam.image_width:\n"
        "            t = torch.nn.functional.interpolate(\n"
        "                t.unsqueeze(0), size=(cam.image_height, cam.image_width), mode=\"nearest\"\n"
        "            ).squeeze(0)\n"
        "        _error_mask_cache[cam.image_name] = t\n"
        "        return t",
    ),
    (
        "loss: weighted L1 theo error mask",
        "        # Loss\n"
        "        gt_image = viewpoint_cam.original_image.cuda()\n"
        "        Ll1 = l1_loss(image, gt_image)\n"
        "        if FUSED_SSIM_AVAILABLE:",
        "        # Loss\n"
        "        gt_image = viewpoint_cam.original_image.cuda()\n"
        "        _err_w = _error_mask(viewpoint_cam)\n"
        "        if _err_w is not None:\n"
        "            Ll1 = (_err_w * torch.abs(image - gt_image)).sum() / _err_w.expand_as(image).sum()\n"
        "        else:\n"
        "            Ll1 = l1_loss(image, gt_image)\n"
        "        if FUSED_SSIM_AVAILABLE:",
    ),
]


def apply_patches(train_py: Path) -> None:
    content = train_py.read_text(encoding="utf-8")

    if "error-refine" in content:
        print(f"-> {train_py} đã được vá error-refine từ trước, bỏ qua (không vá lại 2 lần).")
        return
    if "antenna-focus" in content:
        raise SystemExit(
            f"[LỖI] {train_py} có vẻ đã bị vá bởi 1 patch khác đụng cùng điểm 'Loss' (đánh dấu "
            f"'antenna-focus' tìm thấy trong file) — 2 patch vá cùng 1 chỗ KHÔNG dùng chung được "
            f"trên 1 file. Dùng 1 bản clone $GS_REPO SẠCH riêng (chưa vá gì) cho error-refine."
        )

    for label, old, new in PATCHES:
        n = content.count(old)
        if n != 1:
            raise SystemExit(
                f"[LỖI] Vá '{label}' thất bại: tìm thấy {n} lần đoạn text cần thay (cần đúng 1).\n"
                f"Khả năng cao graphdeco-inria/gaussian-splatting đã đổi code kể từ lúc viết patch này "
                f"— cần đối chiếu lại {train_py} thủ công, KHÔNG bỏ qua bước này."
            )
        content = content.replace(old, new, 1)

    train_py.write_text(content, encoding="utf-8")
    print(f"-> Đã vá xong {train_py} ({len(PATCHES)} chỗ) — chỉ có tác dụng khi train.py chạy kèm "
          f"--refine_from_iteration/--error_mask_dir, không có 2 cờ đó thì hành vi y hệt bản gốc.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gs_repo", required=True, help="Thư mục đã clone --recursive graphdeco-inria/gaussian-splatting (SẠCH, chưa vá patch nào khác)")
    args = ap.parse_args()

    train_py = Path(args.gs_repo) / "train.py"
    if not train_py.exists():
        raise SystemExit(f"Không thấy {train_py} — kiểm tra lại --gs_repo.")
    apply_patches(train_py)


if __name__ == "__main__":
    main()
