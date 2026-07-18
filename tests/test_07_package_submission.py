#!/usr/bin/env python3
"""Test cục bộ (không cần GPU/mạng) cho `pipeline/scripts/07_package_submission.py`
+ `pipeline/kaggle_submission.ipynb` — mock toàn bộ dataset/render bằng dữ liệu giả,
theo đúng triết lý test ở `docs/PORTED_KNOWLEDGE.md` mục 6 (không dùng pytest, script
độc lập, in PASS/FAIL rõ ràng, tự có mã thoát khác 0 nếu có test fail).

Cách chạy:
    python3 tests/test_07_package_submission.py

Các test bao gồm:
  1. Cú pháp: py_compile 07_package_submission.py + nbformat.validate() notebook.
  2. Unit test thuần Python cho target_filename()/check_scene() (import trực tiếp
     module script, không qua subprocess).
  3. TEST REGRESSION cho đúng bug thật đã tìm+sửa ở repo tiền nhiệm (xem
     docs/PORTED_KNOWLEDGE.md mục 5): dựng 1 zip "giả bug" (đổi tên .jpg nhưng giữ
     nguyên bytes PNG, không mã hoá lại) rồi xác nhận verify_zip() của chính script
     PHẢI phát hiện + báo lỗi (SystemExit) — nếu ai đó vô tình quay lại cách làm cũ
     ("chỉ đổi tên, giữ nguyên byte"), test này phải FAIL để cảnh báo ngay.
  4. Test end-to-end: fabricate 1 dataset giả (7 scene, test_poses.csv giả) + 1 thư
     mục renders/ giả (PNG thật, kích thước đúng theo pose) qua biến môi trường
     BTS_DATASET_ROOT + --renders_root, chạy `07_package_submission.py` như CLI thật
     (subprocess), rồi kiểm tra submission.zip sinh ra: đủ scene, đúng ảnh, ảnh .jpg
     trong CSV thật sự là JPEG thật (không phải PNG đổi tên).
  5. Test nhánh lỗi: thiếu 1 ảnh render, sai kích thước 1 ảnh, --check_only trên zip
     thiếu ảnh — xác nhận script dừng với exit code != 0 và in lỗi rõ ràng (không
     phải crash mù mờ / không phải âm thầm bỏ qua).
"""
import importlib.util
import io
import os
import py_compile
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "pipeline" / "scripts" / "07_package_submission.py"
NOTEBOOK_PATH = REPO_ROOT / "pipeline" / "kaggle_submission.ipynb"

BTS_SCENES = ["HCM0421", "HCM0539", "HCM0540", "HCM0644", "HCM0674"]
GENERIC_SCENES = ["bonsai", "chair"]
ALL_SCENES = BTS_SCENES + GENERIC_SCENES

CSV_HEADER = ["image_name", "qw", "qx", "qy", "qz", "tx", "ty", "tz",
              "fx", "fy", "cx", "cy", "width", "height"]

_failures: list[str] = []
_n_pass = 0


def check(name: str, cond: bool, detail: str = ""):
    global _n_pass
    if cond:
        _n_pass += 1
        print(f"  [PASS] {name}")
    else:
        _failures.append(name)
        print(f"  [FAIL] {name} {('-- ' + detail) if detail else ''}")


def load_script_module():
    """Import 07_package_submission.py như 1 module Python bình thường (không qua
    subprocess) để unit-test trực tiếp target_filename()/check_scene()/... — module
    này tự thêm pipeline/ vào sys.path và import common.scenes/common.poses ở cấp
    module, nên cứ import là chạy được (không cần BTS_DATASET_ROOT hợp lệ tại thời
    điểm import, common/scenes.py chỉ dựng Path object, không đọc đĩa)."""
    spec = importlib.util.spec_from_file_location("pkg_submission_under_test", SCRIPT_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def make_fake_pose_row(image_name: str, width: int, height: int) -> dict:
    return dict(
        image_name=image_name,
        qw=1.0, qx=0.0, qy=0.0, qz=0.0,
        tx=0.0, ty=0.0, tz=0.0,
        fx=float(width), fy=float(width),
        cx=width / 2.0, cy=height / 2.0,
        width=width, height=height,
    )


def write_csv(csv_path: Path, rows: list[dict]):
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="") as f:
        f.write(",".join(CSV_HEADER) + "\n")
        for r in rows:
            f.write(",".join(str(r[c]) for c in CSV_HEADER) + "\n")


def make_fake_png(path: Path, width: int, height: int, color=(255, 0, 0)):
    """Ảnh giả thật sự có nội dung (không phải file rỗng) — dùng noise nhẹ trên nền
    màu để PNG/JPEG khi mã hoá lại không trùng byte tình cờ, giúp phân biệt rõ 2
    định dạng lúc test."""
    arr = np.zeros((height, width, 3), dtype=np.uint8)
    arr[:, :] = color
    rng = np.random.RandomState(0)
    noise = rng.randint(0, 40, size=arr.shape, dtype=np.uint8)
    arr = np.clip(arr.astype(int) + noise, 0, 255).astype(np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr).save(path, format="PNG")


def build_fake_dataset(dataset_root: Path, sizes: dict[str, tuple[int, int]],
                        scene_image_exts: dict[str, list[str]]) -> dict[str, list[dict]]:
    """Tạo test_poses.csv giả cho toàn bộ 7 scene (giữ đúng schema common/poses.py).
    scene_image_exts[scene] = danh sách đuôi ảnh (mỗi phần tử = 1 pose) — cho phép
    trộn .png/.jpg/.JPG để test filename_mode=literal với nhiều đuôi khác nhau."""
    scene_rows: dict[str, list[dict]] = {}
    for scene in ALL_SCENES:
        w, h = sizes[scene]
        exts = scene_image_exts[scene]
        rows = [make_fake_pose_row(f"{scene}_{i:04d}{ext}", w, h) for i, ext in enumerate(exts)]
        write_csv(dataset_root / scene / "test" / "test_poses.csv", rows)
        scene_rows[scene] = rows
    return scene_rows


def build_fake_renders(renders_root: Path, scene_rows: dict[str, list[dict]],
                        sizes: dict[str, tuple[int, int]], skip: set = frozenset(),
                        wrong_size: set = frozenset()):
    """Sinh renders_root/<scene>/renders/<stem>.png cho mọi pose, TRỪ (scene, image_name)
    có trong `skip` (mô phỏng thiếu render) hoặc `wrong_size` (mô phỏng render sai kích
    thước, đúng loại bug mà check_scene() phải bắt được)."""
    for scene, rows in scene_rows.items():
        w, h = sizes[scene]
        out_dir = renders_root / scene / "renders"
        for r in rows:
            key = (scene, r["image_name"])
            if key in skip:
                continue
            stem = Path(r["image_name"]).stem
            rw, rh = (w // 2, h // 2) if key in wrong_size else (w, h)
            make_fake_png(out_dir / f"{stem}.png", rw, rh)


def run_script(args: list[str], env_extra: dict) -> subprocess.CompletedProcess:
    env = os.environ.copy()
    env.update(env_extra)
    return subprocess.run(
        [sys.executable, str(SCRIPT_PATH)] + args,
        capture_output=True, text=True, env=env,
    )


def test_syntax():
    print("== 1. Cú pháp (py_compile + nbformat.validate) ==")
    try:
        py_compile.compile(str(SCRIPT_PATH), doraise=True)
        check("py_compile 07_package_submission.py", True)
    except py_compile.PyCompileError as e:
        check("py_compile 07_package_submission.py", False, str(e))

    try:
        import nbformat
        nb = nbformat.read(str(NOTEBOOK_PATH), as_version=4)
        nbformat.validate(nb)
        check("nbformat.validate kaggle_submission.ipynb", True)
    except Exception as e:
        check("nbformat.validate kaggle_submission.ipynb", False, str(e))


def test_unit_functions():
    print("== 2. Unit test target_filename()/check_scene() ==")
    mod = load_script_module()

    check("target_filename literal giữ nguyên .JPG",
          mod.target_filename("DJI_0001.JPG", "literal") == "DJI_0001.JPG")
    check("target_filename png_ext đổi đuôi",
          mod.target_filename("DJI_0001.JPG", "png_ext") == "DJI_0001.png")
    try:
        mod.target_filename("x.png", "bogus_mode")
        check("target_filename báo lỗi mode không hợp lệ", False, "không raise")
    except ValueError:
        check("target_filename báo lỗi mode không hợp lệ", True)

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        renders_dir = td / "renders"
        renders_dir.mkdir()
        make_fake_png(renders_dir / "a.png", 64, 48)

        Scene = mod.Scene
        csv_path = td / "test_poses.csv"
        write_csv(csv_path, [make_fake_pose_row("a.jpg", 64, 48), make_fake_pose_row("b.jpg", 64, 48)])
        scene = Scene(name="fake", domain="generic", root=td)
        # monkeypatch test_poses_csv property target: Scene.test_poses_csv suy ra từ root,
        # nên đặt csv đúng chỗ root/test/test_poses.csv
        (td / "test").mkdir(exist_ok=True)
        shutil.move(str(csv_path), str(td / "test" / "test_poses.csv"))

        errors = mod.check_scene(scene, renders_dir)
        check("check_scene() phát hiện thiếu render 'b'",
              any("b.jpg" in e and "THIẾU" in e for e in errors), str(errors))
        check("check_scene() phát hiện lệch số lượng file",
              any("số file render" in e for e in errors), str(errors))


def test_regression_bad_bytes_caught():
    print("== 3. Regression: verify_zip() PHẢI bắt được bug 'đổi tên, giữ nguyên byte PNG' ==")
    mod = load_script_module()
    Scene = mod.Scene

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        (td / "test").mkdir(parents=True)
        write_csv(td / "test" / "test_poses.csv", [make_fake_pose_row("img.jpg", 32, 24)])
        scene = Scene(name="fakebug", domain="generic", root=td)

        # Dựng 1 zip GIẢ theo ĐÚNG kiểu bug cũ: arcname .jpg nhưng bytes bên trong
        # là PNG thật (chưa mã hoá lại) -- đây chính là dạng lỗi "tăng dung lượng
        # 4-8 lần" đã xảy ra thật.
        raw_png = io.BytesIO()
        arr = np.zeros((24, 32, 3), dtype=np.uint8)
        Image.fromarray(arr).save(raw_png, format="PNG")

        bad_zip = td / "bad.zip"
        with zipfile.ZipFile(bad_zip, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("fakebug/img.jpg", raw_png.getvalue())

        try:
            mod.verify_zip(bad_zip, [scene])
            check("verify_zip() raise SystemExit trên zip có bug 'đổi tên suông'", False,
                  "không raise -- NGHIÊM TRỌNG: bug cũ có thể tái diễn mà không bị phát hiện")
        except SystemExit:
            check("verify_zip() raise SystemExit trên zip có bug 'đổi tên suông'", True)


def test_end_to_end_happy_path():
    print("== 4. End-to-end: dataset + renders giả -> submission.zip hợp lệ ==")
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        dataset_root = td / "Dataset" / "VAI_NVS_DATA_ROUND2"
        renders_root = td / "work"
        out_zip = td / "submission.zip"

        sizes = {s: (64, 48) for s in ALL_SCENES}
        # Trộn đuôi .jpg/.JPG/.png để test filename_mode=literal xử lý đủ trường hợp,
        # đúng thực tế test_poses.csv thật có thể có đuôi hoa/thường lẫn lộn.
        scene_image_exts = {
            "HCM0421": [".JPG", ".JPG", ".png"],
            "HCM0539": [".jpg"],
            "HCM0540": [".jpg"],
            "HCM0644": [".jpg"],
            "HCM0674": [".jpg"],
            "bonsai": [".png", ".png"],
            "chair": [".jpg", ".png"],
        }
        scene_rows = build_fake_dataset(dataset_root, sizes, scene_image_exts)
        build_fake_renders(renders_root, scene_rows, sizes)

        env = {"BTS_DATASET_ROOT": str(dataset_root)}
        result = run_script(
            ["--out", str(out_zip), "--renders_root", str(renders_root),
             "--filename_mode", "literal", "--jpeg_quality", "95"],
            env,
        )
        check("script thoát mã 0 (happy path)", result.returncode == 0,
              f"stdout={result.stdout}\nstderr={result.stderr}")
        check("submission.zip được tạo ra", out_zip.exists())

        if out_zip.exists():
            with zipfile.ZipFile(out_zip) as zf:
                names = set(zf.namelist())
                n_expected = sum(len(rows) for rows in scene_rows.values())
                check("đủ số ảnh trong zip", len(names) == n_expected,
                      f"expected={n_expected} actual={len(names)}")

                # Kiểm tra CHÍNH bug đã sửa: ảnh .JPG trong CSV -> bytes zip PHẢI là
                # JPEG thật (không phải PNG đổi tên).
                jpg_entries = [n for n in names if n.lower().endswith((".jpg", ".jpeg"))]
                check("có ít nhất 1 ảnh .jpg trong bộ test", len(jpg_entries) > 0)
                all_jpeg_real = True
                for n in jpg_entries:
                    with Image.open(io.BytesIO(zf.read(n))) as im:
                        if im.format != "JPEG":
                            all_jpeg_real = False
                check("MỌI file .jpg/.JPG trong zip có nội dung JPEG thật (không phải PNG đổi tên)",
                      all_jpeg_real)

                png_entries = [n for n in names if n.lower().endswith(".png")]
                all_png_real = True
                for n in png_entries:
                    with Image.open(io.BytesIO(zf.read(n))) as im:
                        if im.format != "PNG":
                            all_png_real = False
                check("mọi file .png trong zip có nội dung PNG thật", all_png_real)

        # --check_only trên chính zip vừa tạo (đường dẫn dataset vẫn cần
        # BTS_DATASET_ROOT vì --check_only vẫn cần đọc lại test_poses.csv để đối chiếu).
        result_check = run_script(["--check_only", str(out_zip)], env)
        check("--check_only báo OK trên zip hợp lệ", result_check.returncode == 0,
              f"stdout={result_check.stdout}\nstderr={result_check.stderr}")


def test_error_paths():
    print("== 5. Nhánh lỗi: thiếu ảnh / sai kích thước / check_only trên zip thiếu ==")
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        dataset_root = td / "Dataset" / "VAI_NVS_DATA_ROUND2"
        renders_root = td / "work"
        out_zip = td / "submission.zip"

        sizes = {s: (64, 48) for s in ALL_SCENES}
        scene_image_exts = {s: [".jpg", ".jpg"] for s in ALL_SCENES}
        scene_rows = build_fake_dataset(dataset_root, sizes, scene_image_exts)

        # Thiếu render ảnh đầu tiên của HCM0421, sai kích thước ảnh đầu của chair.
        missing_key = ("HCM0421", scene_rows["HCM0421"][0]["image_name"])
        wrong_size_key = ("chair", scene_rows["chair"][0]["image_name"])
        build_fake_renders(renders_root, scene_rows, sizes,
                            skip={missing_key}, wrong_size={wrong_size_key})

        env = {"BTS_DATASET_ROOT": str(dataset_root)}
        result = run_script(
            ["--out", str(out_zip), "--renders_root", str(renders_root)], env,
        )
        check("script thoát mã != 0 khi thiếu/lệch kích thước ảnh render", result.returncode != 0)
        check("báo lỗi có nhắc scene HCM0421 (thiếu render)", "HCM0421" in result.stdout)
        check("báo lỗi có nhắc scene chair (sai kích thước)", "chair" in result.stdout)
        check("KHÔNG tạo submission.zip khi có lỗi", not out_zip.exists())

        # --check_only trên 1 zip build thủ công thiếu hẳn 1 scene.
        bad_zip = td / "bad_missing_scene.zip"
        with zipfile.ZipFile(bad_zip, "w", zipfile.ZIP_DEFLATED) as zf:
            for scene in ALL_SCENES:
                if scene == "bonsai":
                    continue  # cố tình thiếu nguyên scene "bonsai"
                for r in scene_rows[scene]:
                    buf = io.BytesIO()
                    Image.fromarray(np.zeros((48, 64, 3), dtype=np.uint8)).convert("RGB").save(
                        buf, format="JPEG", quality=95)
                    zf.writestr(f"{scene}/{r['image_name']}", buf.getvalue())
        result_check = run_script(["--check_only", str(bad_zip)], env)
        check("--check_only phát hiện thiếu nguyên 1 scene", result_check.returncode != 0)
        check("báo lỗi --check_only có nhắc 'bonsai'", "bonsai" in result_check.stdout)


def main():
    if not SCRIPT_PATH.exists():
        print(f"[LỖI] Không tìm thấy {SCRIPT_PATH} — chạy test này sau khi 07_package_submission.py tồn tại.")
        return 1
    if not NOTEBOOK_PATH.exists():
        print(f"[LỖI] Không tìm thấy {NOTEBOOK_PATH}.")
        return 1

    test_syntax()
    test_unit_functions()
    test_regression_bad_bytes_caught()
    test_end_to_end_happy_path()
    test_error_paths()

    print()
    print(f"===== {_n_pass} PASS, {len(_failures)} FAIL =====")
    if _failures:
        print("Test FAIL:")
        for f in _failures:
            print(f"  - {f}")
        return 1
    print("TẤT CẢ TEST PASS.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
