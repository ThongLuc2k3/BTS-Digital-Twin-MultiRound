#!/usr/bin/env python3
"""Quét TOÀN BỘ repo, kiểm tra cú pháp mọi file `.py` (py_compile) và mọi file
`.ipynb` (nbformat.validate) — bước rẻ nhất/nhanh nhất trong triết lý test ở
`docs/PORTED_KNOWLEDGE.md` mục 6, nên chạy TRƯỚC mọi test khác, kể cả khi các agent
khác (Round-1 baseline, Round-2/3 refine) chưa xong việc — script này tự bỏ qua nếu
gặp ít file hơn dự kiến, KHÔNG coi là lỗi (report rõ số file quét được).

Cách chạy:
    python3 tests/test_syntax_all.py

Loại trừ: `.git/`, `pipeline/work/` (output sinh ra, gitignore), bất kỳ thư mục
`gaussian-splatting/` nào lỡ bị clone cục bộ (repo ngoài, không phải code của ta),
`__pycache__/`, `.ipynb_checkpoints/`.
"""
import py_compile
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

EXCLUDE_DIR_NAMES = {".git", "pipeline/work", "gaussian-splatting", "__pycache__",
                     ".ipynb_checkpoints", "Dataset", ".venv", "venv"}


def is_excluded(p: Path) -> bool:
    parts = set(p.relative_to(REPO_ROOT).parts)
    return bool(parts & EXCLUDE_DIR_NAMES)


def main():
    py_files = sorted(p for p in REPO_ROOT.rglob("*.py") if not is_excluded(p))
    nb_files = sorted(p for p in REPO_ROOT.rglob("*.ipynb") if not is_excluded(p))

    failures = []

    print(f"== py_compile: {len(py_files)} file .py ==")
    for p in py_files:
        rel = p.relative_to(REPO_ROOT)
        try:
            py_compile.compile(str(p), doraise=True)
            print(f"  [PASS] {rel}")
        except py_compile.PyCompileError as e:
            print(f"  [FAIL] {rel} -- {e}")
            failures.append(str(rel))

    print(f"\n== nbformat.validate: {len(nb_files)} file .ipynb ==")
    try:
        import nbformat
    except ImportError:
        print("  [LỖI] Chưa cài nbformat -- `pip install nbformat` rồi chạy lại.")
        return 1
    for p in nb_files:
        rel = p.relative_to(REPO_ROOT)
        try:
            nb = nbformat.read(str(p), as_version=4)
            nbformat.validate(nb)
            print(f"  [PASS] {rel} ({len(nb['cells'])} cell)")
        except Exception as e:
            print(f"  [FAIL] {rel} -- {e}")
            failures.append(str(rel))

    print()
    if failures:
        print(f"===== {len(failures)} FILE LỖI =====")
        for f in failures:
            print(f"  - {f}")
        return 1
    print(f"===== TẤT CẢ {len(py_files)} file .py + {len(nb_files)} file .ipynb hợp lệ. =====")
    if len(py_files) < 6 or len(nb_files) < 1:
        print("[LƯU Ý] Số file quét được còn ít hơn kỳ vọng cuối cùng (7 scene x nhiều script + "
              "3-4 notebook) -- BÌNH THƯỜNG nếu các agent Round-1/Round-2+3 chưa đẩy hết file lên. "
              "Chạy lại script này sau khi merge đủ 3 nhánh để có bức tranh đầy đủ.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
