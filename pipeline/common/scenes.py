"""Liệt kê scene trong Dataset/VAI_NVS_DATA_ROUND2 và đường dẫn con của từng scene.

Round 2 (BTC đã bỏ round 1, thi lại từ đầu bằng dataset mới) đổi cấu trúc so với
round 1: KHÔNG còn chia public_set/private_set1 — chỉ có 1 danh sách phẳng 7 scene,
và KHÔNG scene nào có ảnh ground-truth thật (`test/` chỉ có test_poses.csv). Vì vậy
lớp Scene ở đây không còn field `split`/`gt_test_images_dir` như bản round 1 — muốn
tự chấm điểm phải tự tạo holdout split từ chính ảnh train (xem
`pipeline/scripts/00_make_holdout_split.py`), không lấy từ đây.

5 scene có tiền tố HCM là ảnh drone trạm BTS thật (domain "bts") — các kỹ thuật đặc
thù cấu trúc BTS (antenna-region-focus...) chỉ nên áp dụng cho nhóm này. `bonsai` và
`chair` là scene tổng quát, không phải BTS (domain "generic") — LƯU Ý: `bonsai` trùng
tên scene chuẩn của bộ dataset học thuật Mip-NeRF 360, nhưng đây là ảnh BTC tự cấp
riêng cho cuộc thi; tuyệt đối không tải/dùng bản Mip-NeRF360 gốc từ nguồn ngoài (vi
phạm mục 10.1 Đề bài.md — cấm dữ liệu ngoài chứa cùng đối tượng/scene).

Đường dẫn dataset ưu tiên lấy từ biến môi trường BTS_DATASET_ROOT nếu có — cần thiết
khi code và dataset KHÔNG nằm chung 1 thư mục gốc (vd trên Kaggle: code lấy từ git
clone, dataset tải riêng, 2 nơi khác gốc nhau nên không thể suy ra bằng "đi lên N cấp
từ vị trí file code" như lúc chạy local).
"""
import os
from dataclasses import dataclass
from pathlib import Path

_env_root = os.environ.get("BTS_DATASET_ROOT")
if _env_root:
    DATASET_ROOT = Path(_env_root)
else:
    REPO_ROOT = Path(__file__).resolve().parents[2]
    DATASET_ROOT = REPO_ROOT / "Dataset" / "VAI_NVS_DATA_ROUND2"

BTS_SCENES = ["HCM0421", "HCM0539", "HCM0540", "HCM0644", "HCM0674"]
GENERIC_SCENES = ["bonsai", "chair"]


@dataclass
class Scene:
    name: str
    domain: str  # "bts" | "generic"
    root: Path

    @property
    def train_images_dir(self) -> Path:
        return self.root / "train" / "images"

    @property
    def provided_sparse_dir(self) -> Path:
        """sparse/0 do BTC cung cấp sẵn — đã xác nhận hợp lệ ở 7/7 scene round 2."""
        return self.root / "train" / "sparse" / "0"

    @property
    def test_poses_csv(self) -> Path:
        return self.root / "test" / "test_poses.csv"

    def has_valid_provided_sparse(self) -> bool:
        d = self.provided_sparse_dir
        if not d.exists():
            return False
        cams = d / "cameras.bin"
        return cams.exists() and cams.stat().st_size > 0


def get_scene(name: str) -> Scene:
    if name in BTS_SCENES:
        return Scene(name=name, domain="bts", root=DATASET_ROOT / name)
    if name in GENERIC_SCENES:
        return Scene(name=name, domain="generic", root=DATASET_ROOT / name)
    raise ValueError(
        f"Không rõ scene '{name}'. BTS: {BTS_SCENES}. Generic: {GENERIC_SCENES}."
    )


def all_scenes(domain: str | None = None) -> list[Scene]:
    """domain=None -> toàn bộ 7 scene; domain="bts"/"generic" -> chỉ tập tương ứng."""
    names: list[str] = []
    if domain in (None, "bts"):
        names += BTS_SCENES
    if domain in (None, "generic"):
        names += GENERIC_SCENES
    return [get_scene(n) for n in names]
