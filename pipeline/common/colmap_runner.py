"""Chạy COLMAP (SfM) hoàn toàn bằng pycolmap — không cần cài `colmap` CLI riêng.

Các hàm/enums dùng ở đây (pycolmap.ImageReaderOptions, pycolmap.CameraMode,
pycolmap.extract_features, match_sequential/match_exhaustive, incremental_mapping,
undistort_images, pycolmap.Reconstruction) đã được đối chiếu trực tiếp với source
C++ / pybind hiện tại của COLMAP (repo colmap/colmap, nhánh main, thư mục
src/pycolmap/pipeline/*.cc và src/pycolmap/scene/*.cc) để đảm bảo đúng tên tham số,
KHÔNG suy đoán từ trí nhớ.

Phát hiện quan trọng khi soi dữ liệu thật (xem Dataset/README.md mục 4): sparse
gốc của BTC cho scene HCM0249 dùng camera model SIMPLE_RADIAL (model_id=2, 4 tham
số f,cx,cy,k — giải mã trực tiếp từ cameras.bin, không suy đoán), khớp với việc
mọi hàng test_poses.csv luôn có fx==fy (SIMPLE_RADIAL chỉ có 1 focal length dùng
chung cho cả 2 trục). Vì vậy mặc định script này cũng dùng SIMPLE_RADIAL khi tự
chạy COLMAP trên ảnh train (ảnh gốc CÓ thể có méo ống kính nhẹ), rồi undistort
sang PINHOLE sạch (không méo) trước khi đưa vào 3D Gaussian Splatting — đúng quy
trình chuẩn mà chính script convert.py của graphdeco-inria/gaussian-splatting dùng
(feature_extractor -> matcher -> mapper -> image_undistorter).
"""
import shutil
from pathlib import Path

import pycolmap

from common.logging_utils import FileLog, quiet_pycolmap


def _find_missing_images(rec: "pycolmap.Reconstruction", images_dir: Path) -> tuple[list[str], list[str]]:
    """So khớp danh sách ảnh mà sparse THAM CHIẾU (có pose trong images.bin) với
    ảnh THỰC SỰ có trong images_dir trên đĩa.

    Đây KHÔNG phải trường hợp hiếm/lỗi lạ — sparse do BTC tạo có thể dựng từ tập
    ảnh gốc lớn hơn rồi mới cắt bớt khi đóng gói train/images/, nên 1 vài ảnh có
    pose trong sparse nhưng không tồn tại trên đĩa là chuyện bình thường, không
    phải bug. Quan trọng là phải biết CHÍNH XÁC tên file nào thiếu (không chỉ
    đếm số lượng lệch) để không nhầm với lỗi thật (vd thiếu quá nhiều/thiếu hết).

    Trả về (valid_names, missing_names) — valid_names dùng để truyền cho
    pycolmap.undistort_images(image_names=...), tránh nó tự crash khi gặp ảnh
    không tồn tại.
    """
    all_names = sorted(image.name for image in rec.images.values())
    valid_names = [n for n in all_names if (images_dir / n).exists()]
    missing_names = sorted(set(all_names) - set(valid_names))
    return valid_names, missing_names


def _undistort_and_fix_layout(sparse_dir: Path, images_dir: Path, dense_dir: Path,
                               rec: "pycolmap.Reconstruction", log: FileLog,
                               workdir: Path) -> list[str]:
    """undistort_images ghi thẳng vào <dense_dir>/sparse/*.bin (không có "0/"),
    trong khi graphdeco-inria/gaussian-splatting cần <source>/sparse/0/*.bin —
    dùng chung cho cả 2 đường (tự chạy COLMAP / dùng thẳng sparse có sẵn).

    QUAN TRỌNG: tham số `image_names` của pycolmap.undistort_images chỉ quyết
    định ảnh nào được COPY PIXEL sang dense/images/ — nó KHÔNG tự xoá ảnh thiếu
    khỏi model sparse output (dense/sparse/0/images.bin vẫn có thể còn tham
    chiếu ảnh đó), khiến train.py vẫn cố mở file và crash muộn hơn. Nên ở đây
    phải XOÁ HẲN ảnh thiếu khỏi reconstruction (deregister_frame) rồi ghi ra
    1 bản sparse đã lọc sạch, dùng bản đó làm input cho undistort_images.

    Trả về danh sách tên ảnh bị thiếu (rỗng nếu đủ) để caller báo cáo lại.
    """
    valid_names, missing_names = _find_missing_images(rec, images_dir)
    input_sparse_dir = sparse_dir

    if missing_names:
        log.write(f"[CẢNH BÁO] {len(missing_names)}/{len(valid_names) + len(missing_names)} ảnh "
                  f"có pose trong sparse nhưng KHÔNG có file trong {images_dir} — xoá khỏi "
                  f"reconstruction trước khi undistort, không phải lỗi COLMAP: {missing_names}")
        missing_set = set(missing_names)
        for image in list(rec.images.values()):
            if image.name in missing_set:
                rec.deregister_frame(image.frame_id)
        input_sparse_dir = workdir / "_sparse_filtered"
        input_sparse_dir.mkdir(parents=True, exist_ok=True)
        rec.write_binary(input_sparse_dir)

    # Dọn sạch dense_dir trước khi ghi — nếu chạy lại trên cùng workdir với tập
    # ảnh khác (vd chuyển từ chế độ holdout-eval sang final retrain 100% ảnh),
    # pycolmap.undistort_images ghi đè lên state cũ có thể crash khó hiểu (đã
    # gặp thật: "Uncaught exceptions in thread pool destructor" khi dense_dir cũ
    # còn sparse/images ứng với tập ảnh thiếu khác) thay vì báo lỗi rõ ràng.
    if dense_dir.exists():
        shutil.rmtree(dense_dir)

    pycolmap.undistort_images(
        output_path=dense_dir,
        input_path=input_sparse_dir,
        image_path=images_dir,
        output_type="COLMAP",
    )
    flat_sparse = dense_dir / "sparse"
    nested_sparse = flat_sparse / "0"
    if flat_sparse.exists() and not nested_sparse.exists():
        tmp = dense_dir / "_sparse_tmp"
        flat_sparse.rename(tmp)
        tmp_nested = dense_dir / "sparse"
        tmp_nested.mkdir(parents=True)
        tmp.rename(tmp_nested / "0")

    # Tự kiểm tra lại: đảm bảo KHÔNG còn ảnh nào trong sparse cuối cùng mà thiếu
    # file thật — thà báo lỗi rõ ràng ngay tại đây còn hơn để train.py crash sau
    # khi đã tốn thời gian train.
    final_rec = pycolmap.Reconstruction(dense_dir / "sparse" / "0")
    dense_images_dir = dense_dir / "images"
    still_missing = sorted(
        image.name for image in final_rec.images.values()
        if not (dense_images_dir / image.name).exists()
    )
    if still_missing:
        raise RuntimeError(
            f"Sau khi undistort, sparse cuối cùng ({dense_dir}/sparse/0) vẫn còn "
            f"{len(still_missing)} ảnh thiếu file thật trong {dense_images_dir}: "
            f"{still_missing}. Cần báo lỗi này thay vì để train.py crash — kiểm tra "
            f"lại thủ công."
        )

    return missing_names


def use_provided_sparse(images_dir: Path, sparse_dir: Path, workdir: Path,
                         log_path: Path | None = None) -> dict:
    """Dùng THẲNG sparse đã có sẵn (do BTC cung cấp) — KHÔNG tự chạy COLMAP.

    Dataset có sparse/0/ hợp lệ ở cả 13/13 scene, nên bước feature extraction +
    matching + incremental mapping (vốn tốn thời gian và dễ lỗi nhất) không còn
    cần thiết cho hầu hết trường hợp — chỉ còn bước undistort (rất nhanh, vài
    giây tới vài chục giây) để chuyển sang PINHOLE sạch trước khi đưa vào 3DGS.

    log_path: đường dẫn file log (mặc định workdir/colmap.log nếu không truyền) —
    truyền vào để đặt tên log trùng với tên script gọi hàm này, dễ tra cứu hơn.

    Trả về dict cùng format với run_colmap_scene(), thêm "used_provided_sparse": True.
    "num_reg_images" đã TRỪ ĐI số ảnh thiếu file thật (xem "missing_images" —
    danh sách tên ảnh có pose trong sparse nhưng không có file trên đĩa, bị tự
    động loại ra khi undistort, không phải lỗi).
    """
    workdir.mkdir(parents=True, exist_ok=True)
    log_path = log_path if log_path is not None else workdir / "colmap.log"
    log = FileLog(log_path)
    quiet_pycolmap(log_dir=workdir / "pycolmap_internal_logs")

    rec = pycolmap.Reconstruction(sparse_dir)
    num_points3D = rec.num_points3D()
    log.write(f"Dùng sparse có sẵn: {sparse_dir} "
              f"({rec.num_reg_images()} ảnh, {num_points3D} điểm) — bỏ qua bước tự chạy COLMAP.")

    dense_dir = workdir / "dense"
    log.write(f"Undistort ảnh + camera model -> PINHOLE sạch tại {dense_dir} ...")
    # Chốt số liệu TRƯỚC khi gọi hàm dưới đây — nó sẽ sửa trực tiếp lên `rec`
    # (xoá ảnh thiếu qua deregister_frame), nên rec.num_reg_images() sau lời gọi
    # đã tự giảm sẵn rồi, không được trừ thêm lần nữa.
    num_reg_images_before = rec.num_reg_images()
    missing_images = _undistort_and_fix_layout(sparse_dir, images_dir, dense_dir, rec, log, workdir)
    num_reg_images = num_reg_images_before - len(missing_images)

    log.write(f"Xong. num_reg_images={num_reg_images} num_points3D={num_points3D}")
    log.close()

    return {
        "sparse_dir": sparse_dir,
        "dense_dir": dense_dir,
        "num_reg_images": num_reg_images,
        "num_points3D": num_points3D,
        "log_path": log_path,
        "used_provided_sparse": True,
        "missing_images": missing_images,
    }


def run_colmap_scene(
    images_dir: Path,
    workdir: Path,
    matching: str = "sequential",
    camera_model: str = "SIMPLE_RADIAL",
    camera_params_prior: str | None = None,
    overwrite: bool = False,
    log_path: Path | None = None,
) -> dict:
    """Chạy toàn bộ pipeline SfM cho 1 scene.

    Các bước trung gian ([1/4]...[4/4]) chỉ ghi vào file log (không in ra
    console) — xem "log_path" trong dict trả về nếu cần tra cứu chi tiết.

    log_path: đường dẫn file log (mặc định workdir/colmap.log nếu không truyền) —
    truyền vào để đặt tên log trùng với tên script gọi hàm này, dễ tra cứu hơn.

    Trả về dict {
        "sparse_dir": Path,      # workdir/sparse/<best_idx>  (định dạng COLMAP gốc, có thể méo)
        "dense_dir": Path,       # workdir/dense              (images/ + sparse/0/, đã undistort PINHOLE)
        "num_reg_images": int,   # đã trừ đi ảnh có pose nhưng thiếu file thật (xem "missing_images")
        "num_points3D": int,
        "log_path": Path,
        "missing_images": list[str],  # tên ảnh có pose trong sparse nhưng không có file trên đĩa
    }
    """
    workdir.mkdir(parents=True, exist_ok=True)
    log_path = log_path if log_path is not None else workdir / "colmap.log"
    log = FileLog(log_path)
    quiet_pycolmap(log_dir=workdir / "pycolmap_internal_logs")

    database_path = workdir / "database.db"
    if overwrite and database_path.exists():
        database_path.unlink()

    reader_options = pycolmap.ImageReaderOptions()
    reader_options.camera_model = camera_model
    if camera_params_prior:
        reader_options.camera_params = camera_params_prior

    if not database_path.exists():
        log.write(f"[1/4] Feature extraction ({images_dir.name}, model={camera_model}) ...")
        pycolmap.extract_features(
            database_path=database_path,
            image_path=images_dir,
            camera_mode=pycolmap.CameraMode.SINGLE,
            reader_options=reader_options,
        )
    else:
        log.write("[1/4] Bỏ qua feature extraction (database.db đã tồn tại).")

    log.write(f"[2/4] Feature matching ({matching}) ...")
    if matching == "exhaustive":
        pycolmap.match_exhaustive(database_path)
    else:
        pycolmap.match_sequential(database_path)

    sparse_root = workdir / "sparse"
    sparse_root.mkdir(exist_ok=True)
    if any(sparse_root.iterdir()) and not overwrite:
        log.write("[3/4] Bỏ qua mapping (sparse/ đã có kết quả).")
        recs = {
            int(p.name): pycolmap.Reconstruction(p)
            for p in sorted(sparse_root.iterdir()) if p.is_dir() and (p / "cameras.bin").exists()
        }
    else:
        log.write("[3/4] Incremental mapping (SfM + bundle adjustment) ...")
        recs = pycolmap.incremental_mapping(
            database_path=database_path,
            image_path=images_dir,
            output_path=sparse_root,
        )

    if not recs:
        log.close()
        raise RuntimeError(
            f"COLMAP không tạo được reconstruction nào cho {images_dir}. "
            f"Thử lại với matching='exhaustive' hoặc kiểm tra chất lượng/độ chồng lấn ảnh. "
            f"Chi tiết: {log_path}"
        )

    best_idx = max(recs, key=lambda i: recs[i].num_reg_images())
    best_rec = recs[best_idx]
    if len(recs) > 1:
        sizes = {i: r.num_reg_images() for i, r in recs.items()}
        # Quan trọng — vẫn in ra console (không chỉ ghi log), vì ảnh hưởng trực
        # tiếp tới độ đầy đủ của scene.
        print(f"[CẢNH BÁO] {images_dir.parent.parent.name}: COLMAP tách ra {len(recs)} model rời rạc: {sizes}. "
              f"Dùng model {best_idx} (nhiều ảnh nhất: {best_rec.num_reg_images()}/{len(list(images_dir.iterdir()))}). "
              f"Các ảnh ở model khác sẽ KHÔNG có trong sparse cuối cùng.")
        log.write(f"[CẢNH BÁO] tách {len(recs)} model rời rạc: {sizes}")

    sparse_dir = sparse_root / str(best_idx)
    num_points3D = best_rec.num_points3D()

    dense_dir = workdir / "dense"
    log.write(f"[4/4] Undistort ảnh + camera model -> PINHOLE sạch tại {dense_dir} ...")
    # Chốt số liệu TRƯỚC khi gọi hàm dưới đây — nó sẽ sửa trực tiếp lên `best_rec`
    # (xoá ảnh thiếu qua deregister_frame), nên num_reg_images() sau đó đã tự
    # giảm sẵn rồi, không được trừ thêm lần nữa.
    num_reg_images_before = best_rec.num_reg_images()
    missing_images = _undistort_and_fix_layout(sparse_dir, images_dir, dense_dir, best_rec, log, workdir)
    num_reg_images = num_reg_images_before - len(missing_images)

    log.write(f"Xong. num_reg_images={num_reg_images} num_points3D={num_points3D}")
    log.close()

    return {
        "sparse_dir": sparse_dir,
        "dense_dir": dense_dir,
        "num_reg_images": num_reg_images,
        "num_points3D": num_points3D,
        "log_path": log_path,
        "used_provided_sparse": False,
        "missing_images": missing_images,
    }
