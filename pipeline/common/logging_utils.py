"""Ghi log chi tiết ra file riêng theo scene, console chỉ hiện dòng tóm tắt.

Lý do: chạy hàng loạt 13 scene (COLMAP + train 3DGS) sinh ra rất nhiều dòng log
chi tiết (từng bước COLMAP, log nội bộ pycolmap, tqdm progress bar của train.py
qua 30.000 iteration...) — không cần thiết phải hiện hết trên console/notebook,
chỉ cần 1-2 dòng tóm tắt mỗi scene. Muốn xem lại chi tiết thì mở file log tương ứng.
"""
import datetime
from pathlib import Path


class FileLog:
    """log.write(msg): CHỈ ghi vào file, KHÔNG in ra console.

    Dòng tóm tắt quan trọng thì cứ `print()` bình thường ở nơi gọi — object này
    chỉ dùng cho các dòng chi tiết/trung gian muốn giữ lại để tra cứu sau.
    """

    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = open(self.path, "a", encoding="utf-8")

    def write(self, msg: str) -> None:
        ts = datetime.datetime.now().strftime("%H:%M:%S")
        self._fh.write(f"[{ts}] {msg}\n")
        self._fh.flush()

    def close(self) -> None:
        self._fh.close()

    def __enter__(self) -> "FileLog":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def quiet_pycolmap(log_dir: Path | None = None) -> None:
    """Tắt bớt log nội bộ rất dài dòng của pycolmap (feature extraction từng
    ảnh, từng bước bundle adjustment...) — chỉ còn WARNING/ERROR thoát ra
    console. Gọi 1 lần trước khi dùng pycolmap.

    Mức log của glog (thư viện log C++ mà pycolmap dùng bên dưới):
    0=INFO (rất nhiều), 1=WARNING, 2=ERROR, 3=FATAL.

    Nếu truyền `log_dir`: log đầy đủ (mức INFO) vẫn được GHI RA FILE trong thư
    mục đó (glog tự đặt tên file kiểu INFO.<timestamp>...), chỉ không hiện ra
    console/notebook nữa — dùng khi cần tra cứu chi tiết về sau.
    """
    try:
        import pycolmap
        pycolmap.logging.minloglevel = 2
        if log_dir is not None:
            log_dir.mkdir(parents=True, exist_ok=True)
            pycolmap.logging.log_dir = str(log_dir)
            pycolmap.logging.alsologtostderr = False
    except Exception:
        pass
