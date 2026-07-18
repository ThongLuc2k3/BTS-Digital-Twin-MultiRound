#!/usr/bin/env bash
# Train tiếp (fine-tune NGẮN) 1 checkpoint 3DGS ĐÃ CÓ SẴN bằng cơ chế "error-guided
# refine" — CỐT LÕI của repo này (xem docs/00_MASTER_PLAN.md mục 3.2 +
# docs/PORTED_KNOWLEDGE.md mục 3). CÙNG 1 script này dùng lại được cho MỌI vòng refine
# (Vòng 2 tinh chỉnh checkpoint Vòng 1, Vòng 3 tinh chỉnh checkpoint Vòng 2, ...) — không
# có "vòng 2" và "vòng 3" là 2 code path khác nhau, chỉ khác THAM SỐ (checkpoint nguồn +
# ngân sách iteration của vòng đó).
#
# Wrap quanh `$GS_REPO/train.py` ĐÃ VÁ bằng `apply_error_refine_patch.py` (script này tự
# kiểm tra, báo lỗi rõ nếu chưa vá) với 2 cờ mới `--refine_from_iteration`/
# `--error_mask_dir` + cờ có sẵn `--densify_until_iter 0` (không sinh thêm Gaussian mới,
# chỉ tối ưu lại cái có sẵn ở vùng lỗi cao).
#
# Cài đặt 1 lần + set GS_REPO: giống hệt 03_train_3dgs.sh (xem comment ở đó).
# Yêu cầu TRƯỚC khi chạy script này (mỗi scene):
#   1. Có checkpoint nguồn:  pipeline/work/<scene>/gs_model/point_cloud/iteration_<N>/point_cloud.ply
#      + cfg_args + pipeline_train_flags.json (ghi bởi 03_train_3dgs.sh hoặc chính
#      06_train_refine.sh của 1 vòng refine trước đó).
#   2. `colmap/dense/{images/,sparse/0/}` đã tái tạo (01_run_colmap.py --scene <scene>
#      [--holdout]) — train.py CẦN ảnh đã undistort để đọc; nếu vòng trước đã dọn đĩa
#      (mặc định), phải chạy lại 01_run_colmap.py TRƯỚC khi chạy script này.
#   3. Đã sinh error mask: `python 05_generate_error_mask.py --scene <scene> --iteration <N>
#      --max_weight <W>` — script này ĐỌC LẠI iteration/max_weight đã dùng từ
#      error_masks/manifest.json để đối chiếu, KHÔNG tự đoán lại.
#   4. `$GS_REPO/train.py` đã vá: `python apply_error_refine_patch.py --gs_repo "$GS_REPO"`.
#
# SH_DEGREE/ANTIALIASING KHÔNG phải tự gõ tay — script tự đọc từ cfg_args/
# pipeline_train_flags.json của checkpoint nguồn (đúng cấu hình đã train ra checkpoint đó,
# xem docs/PORTED_KNOWLEDGE.md mục 2 vì sao antialiasing PHẢI đọc từ file riêng, không
# suy đoán từ cfg_args). Có thể ép đè bằng biến môi trường SH_DEGREE/ANTIALIASING nhưng
# KHÔNG khuyến khích (dễ lệch với checkpoint nguồn, in cảnh báo rõ nếu làm vậy).
#
# Cách dùng:
#   export GS_REPO=/path/to/gaussian-splatting   # đã vá bằng apply_error_refine_patch.py
#   ./06_train_refine.sh HCM0421                          # CKPT_ITERATION tự dò (lớn nhất có sẵn)
#   CKPT_ITERATION=15000 REFINE_ITERATIONS=3000 \
#     ./06_train_refine.sh HCM0421                        # chỉ định rõ checkpoint nguồn
#   MAX_ERROR_WEIGHT=6.0 ./06_train_refine.sh HCM0421 chair   # nhiều scene 1 lệnh (xem LƯU Ý dưới)
#   PROGRESS_INTERVAL=30 ./06_train_refine.sh HCM0421     # in tiến độ mỗi 30s thay vì 60s mặc định
#
# LƯU Ý nhiều scene 1 lệnh: nếu đặt CKPT_ITERATION tường minh, giá trị đó áp dụng CHO CẢ
# mọi scene trong lệnh (thường không đúng ý nếu các scene ở iteration khác nhau) — để
# trống (mặc định) cho mỗi scene tự dò iteration lớn nhất của CHÍNH nó, hoặc chạy riêng
# từng scene 1 lệnh nếu cần chỉ định khác nhau.
#
# Biến môi trường:
#   CKPT_ITERATION     iteration của checkpoint NGUỒN cần nạp (point_cloud/iteration_<N>).
#                       Để trống (mặc định) = tự dò iteration LỚN NHẤT có sẵn trong
#                       gs_model/point_cloud/ của scene đó.
#   REFINE_ITERATIONS  ngân sách tinh chỉnh THÊM của vòng NÀY (số iteration TUYỆT ĐỐI mà
#                       chính lần chạy train.py này thực hiện — KHÔNG cộng dồn với
#                       CKPT_ITERATION bên trong train.py, xem docstring
#                       apply_error_refine_patch.py). Mặc định 3000.
#   MAX_ERROR_WEIGHT   chỉ dùng để ĐỐI CHIẾU với error_masks/manifest.json (script KHÔNG
#                       tự truyền giá trị này cho train.py — trọng số đã "nướng" sẵn vào
#                       từng pixel mask PNG lúc 05_generate_error_mask.py chạy). Mặc định
#                       6.0 — PHẢI khớp giá trị --max_weight đã dùng lúc sinh mask, nếu
#                       không script BÁO LỖI dừng lại (tránh refine bằng mask sinh với
#                       cấu hình khác ý định mà không biết).
#   DENSIFY_UNTIL_ITER  mặc định 0 (không sinh thêm Gaussian mới — đúng tinh thần "chỉ
#                       tinh chỉnh"). Đổi giá trị này CHỈ nên làm khi thử nghiệm có chủ
#                       đích, không phải mặc định.
#   CLEANUP_DENSE_IMAGES  mặc định 1 — xoá colmap/dense/images/ sau khi train xong (dọn
#                       đĩa, giống 03_train_3dgs.sh). Set 0 nếu cần giữ lại debug.
#
# Output: point_cloud MỚI trong CHÍNH gs_model/ đã nạp vào (không tạo thư mục model mới)
# — tại point_cloud/iteration_<CKPT_ITERATION + REFINE_ITERATIONS>/point_cloud.ply.
#
# QUYẾT ĐỊNH THIẾT KẾ QUAN TRỌNG (khác notebook cũ kaggle_error_refine.ipynb — nơi cơ chế
# này lần đầu được viết dạng Python inline, CHƯA giải quyết vấn đề này): train.py (không
# --start_checkpoint .pth) luôn bắt đầu đếm lại từ iteration 0 cho MỖI lần chạy refine
# (first_iter=0, đối chiếu trực tiếp source train.py::training(), không suy đoán) và tự
# lưu checkpoint tại point_cloud/iteration_<REFINE_ITERATIONS>/ — nghĩa là 2 vòng refine
# LIÊN TIẾP dùng CÙNG REFINE_ITERATIONS mặc định (vd cả round2 và round3 đều 3000) sẽ CÙNG
# ghi ra thư mục TÊN GIỐNG HỆT NHAU "iteration_3000", không phân biệt được checkpoint đó
# thuộc vòng nào chỉ nhìn tên thư mục — vi phạm yêu cầu tái lập được kết quả (Đề bài mục
# 10.3). Script này ĐỔI TÊN thư mục output ngay sau khi train xong thành
# "iteration_<CKPT_ITERATION + REFINE_ITERATIONS>" (số iteration LUỸ KẾ thật) để giải
# quyết việc này — đồng thời giúp vòng SAU tự dò "iteration lớn nhất có sẵn" ra ĐÚNG
# checkpoint mới nhất (không bị nhầm lẫn với checkpoint cũ cùng tên).

set -euo pipefail

if [[ -z "${GS_REPO:-}" ]]; then
  echo "Lỗi: chưa set biến môi trường GS_REPO (đường dẫn tới repo graphdeco-inria/gaussian-splatting đã clone)." >&2
  exit 1
fi
if [[ ! -f "$GS_REPO/train.py" ]]; then
  echo "Lỗi: không thấy $GS_REPO/train.py — kiểm tra lại GS_REPO." >&2
  exit 1
fi
if ! grep -q "error-refine" "$GS_REPO/train.py"; then
  echo "Lỗi: $GS_REPO/train.py CHƯA được vá error-refine — chạy trước:" >&2
  echo "  python apply_error_refine_patch.py --gs_repo \"$GS_REPO\"" >&2
  exit 1
fi

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PIPELINE_DIR="$(dirname "$SCRIPT_DIR")"
REFINE_ITERATIONS="${REFINE_ITERATIONS:-3000}"
MAX_ERROR_WEIGHT="${MAX_ERROR_WEIGHT:-6.0}"
DENSIFY_UNTIL_ITER="${DENSIFY_UNTIL_ITER:-0}"
CLEANUP_DENSE_IMAGES="${CLEANUP_DENSE_IMAGES:-1}"

if [[ $# -eq 0 ]]; then
  echo "Cách dùng: $0 <scene1> [scene2 ...]" >&2
  exit 1
fi

# Tự dò thư mục point_cloud/iteration_<N> có N LỚN NHẤT trong 1 MODEL_DIR.
# KHÔNG dùng `sort -t_ -k2 -n` (cách 03_train_3dgs.sh dùng cho dòng thông báo cứu hộ) —
# đã tự phát hiện bug thật khi test cục bộ: `-t_` cắt trường theo MỌI dấu "_" trong CẢ
# ĐƯỜNG DẪN, không chỉ trong tên thư mục "iteration_N". Đường dẫn thật luôn có sẵn nhiều
# dấu "_" khác đứng TRƯỚC "iteration_N" (vd ".../gs_model/point_cloud/iteration_15000")
# nên trường số 2 (-k2) không phải là số iteration, sort numeric trên nó vô nghĩa, hàm
# có thể chọn NHẦM thư mục không phải iteration lớn nhất mà không báo lỗi gì — nguy hiểm
# vì đây là input trực tiếp cho --refine_from_iteration.
#
# BUG THẬT THỨ 2 tự phát hiện khi test cục bộ (chính trong repo này, đường dẫn thư mục
# dự án có dấu cách "Khóa Luận Tốt Nghiệp"): bản đầu tiên của hàm này ghép "$n $d" bằng
# printf rồi lấy lại qua `sort | tail -1 | awk '{print $2}'` — awk tách trường theo
# KHOẢNG TRẮNG, nên nếu chính đường dẫn `$d` (biến vào $2 trở đi) chứa dấu cách, `awk
# '{print $2}'` chỉ in ra ĐÚNG 1 TỪ đầu tiên của đường dẫn, cắt cụt phần còn lại — ra
# đường dẫn rác mà KHÔNG báo lỗi cú pháp gì (chỉ lộ ra khi file/thư mục theo đường dẫn
# cụt đó không tồn tại). Sửa hẳn: không dùng pipe/sort/awk trên chuỗi có thể chứa dấu
# cách nữa — so sánh số nguyên bằng vòng lặp bash thuần, không tách trường theo khoảng
# trắng ở đâu cả.
_latest_iteration_dir() {
  local model_dir="$1" d n best_n=-1 best_d=""
  for d in "$model_dir"/point_cloud/iteration_*/; do
    [[ -d "$d" ]] || continue
    d="${d%/}"
    n="${d##*/iteration_}"
    [[ "$n" =~ ^[0-9]+$ ]] || continue
    if (( 10#$n > best_n )); then
      best_n=$((10#$n))
      best_d="$d"
    fi
  done
  [[ -n "$best_d" ]] && printf '%s\n' "$best_d"
}

for SCENE in "$@"; do
  SOURCE_DIR="$PIPELINE_DIR/work/$SCENE/colmap/dense"
  MODEL_DIR="$PIPELINE_DIR/work/$SCENE/gs_model"
  ERROR_MASK_DIR="$PIPELINE_DIR/work/$SCENE/error_masks"
  LOG_FILE="$PIPELINE_DIR/work/$SCENE/06_train_refine.log"

  if [[ ! -d "$SOURCE_DIR/sparse/0" ]]; then
    echo "[BỎ QUA] $SCENE: chưa thấy $SOURCE_DIR/sparse/0 — chạy 01_run_colmap.py --scene $SCENE trước." >&2
    continue
  fi
  if [[ ! -d "$SOURCE_DIR/images" ]]; then
    echo "[BỎ QUA] $SCENE: chưa thấy $SOURCE_DIR/images (ảnh đã undistort, cần cho train.py) — chạy lại" >&2
    echo "          01_run_colmap.py --scene $SCENE trước (bước train/refine trước có thể đã tự dọn thư mục này)." >&2
    continue
  fi
  if [[ ! -d "$MODEL_DIR/point_cloud" ]]; then
    echo "[BỎ QUA] $SCENE: chưa thấy $MODEL_DIR/point_cloud — chưa có checkpoint nào để refine." >&2
    continue
  fi
  if [[ ! -f "$ERROR_MASK_DIR/manifest.json" ]]; then
    echo "[BỎ QUA] $SCENE: chưa thấy $ERROR_MASK_DIR/manifest.json — chạy trước:" >&2
    echo "          python 05_generate_error_mask.py --scene $SCENE" >&2
    continue
  fi

  # Xác định CKPT_ITERATION: dùng biến môi trường nếu có, không thì tự dò iteration
  # LỚN NHẤT có sẵn bằng _latest_iteration_dir() (xem comment định nghĩa hàm ở trên vì
  # sao KHÔNG dùng `sort -t_ -k2 -n` như 03_train_3dgs.sh).
  if [[ -n "${CKPT_ITERATION:-}" ]]; then
    THIS_CKPT_ITERATION="$CKPT_ITERATION"
  else
    LATEST_CKPT_DIR="$(_latest_iteration_dir "$MODEL_DIR")"
    if [[ -z "$LATEST_CKPT_DIR" ]]; then
      echo "[BỎ QUA] $SCENE: không tự dò được checkpoint nào trong $MODEL_DIR/point_cloud." >&2
      continue
    fi
    THIS_CKPT_ITERATION="${LATEST_CKPT_DIR##*_}"
    echo "  [$SCENE] CKPT_ITERATION không set — tự dò iteration lớn nhất có sẵn: $THIS_CKPT_ITERATION"
  fi

  PLY_PATH="$MODEL_DIR/point_cloud/iteration_$THIS_CKPT_ITERATION/point_cloud.ply"
  if [[ ! -f "$PLY_PATH" ]]; then
    echo "[BỎ QUA] $SCENE: không thấy $PLY_PATH." >&2
    continue
  fi

  # Đối chiếu error_masks/manifest.json — mask PHẢI được sinh từ ĐÚNG checkpoint sắp nạp,
  # không thì mask đo lỗi ở 1 trạng thái Gaussian khác, làm sai lệch mục đích refine.
  # max_weight lệch chỉ cảnh báo (không sai lệch trạng thái Gaussian, chỉ khác cường độ
  # boost) — nhưng iteration lệch là lỗi CỨNG, dừng lại luôn.
  MANIFEST_CHECK="$(python3 - "$ERROR_MASK_DIR/manifest.json" "$THIS_CKPT_ITERATION" "$MAX_ERROR_WEIGHT" <<'PYEOF'
import json
import sys

manifest_path, ckpt_iter, max_weight = sys.argv[1], int(sys.argv[2]), float(sys.argv[3])
m = json.loads(open(manifest_path).read())
errors = []
if int(m.get("iteration", -1)) != ckpt_iter:
    errors.append(
        f"manifest.json ghi iteration={m.get('iteration')} nhưng checkpoint sắp nạp là "
        f"iteration={ckpt_iter} — mask KHÔNG khớp checkpoint này, chạy lại "
        f"05_generate_error_mask.py --iteration {ckpt_iter} trước."
    )
mw = float(m.get("max_weight", -1))
if abs(mw - max_weight) > 1e-6:
    print(
        f"WARN max_weight trong manifest.json ({mw}) khác MAX_ERROR_WEIGHT truyền vào ({max_weight}) "
        f"— mask vẫn dùng được (trọng số đã nướng sẵn vào pixel) nhưng KHÔNG khớp ý định, kiểm tra lại.",
        file=sys.stderr,
    )
if errors:
    print("ERROR " + " | ".join(errors))
    sys.exit(1)
print("OK")
PYEOF
  )" || { echo "[LỖI] $SCENE: đối chiếu error_masks/manifest.json thất bại: $MANIFEST_CHECK" >&2; exit 1; }
  if [[ "$MANIFEST_CHECK" != "OK" ]]; then
    echo "[LỖI] $SCENE: $MANIFEST_CHECK" >&2
    exit 1
  fi

  # Tự đọc sh_degree (cfg_args) + antialiasing (pipeline_train_flags.json, xem
  # docs/PORTED_KNOWLEDGE.md mục 2 vì sao KHÔNG được suy đoán từ cfg_args) của checkpoint
  # NGUỒN — refine PHẢI dùng lại ĐÚNG cấu hình đã train ra checkpoint đó.
  CFG_INFO="$(python3 - "$MODEL_DIR" <<'PYEOF'
import sys
import json
from argparse import Namespace
from pathlib import Path

model_dir = Path(sys.argv[1])
sh_degree = 3
cfg_path = model_dir / "cfg_args"
if cfg_path.exists():
    try:
        ns = eval(cfg_path.read_text(), {"Namespace": Namespace})
        sh_degree = getattr(ns, "sh_degree", 3)
    except Exception as e:
        print(f"WARN không đọc được cfg_args: {e}", file=sys.stderr)

antialiasing = False
flags_path = model_dir / "pipeline_train_flags.json"
have_flags = flags_path.exists()
if have_flags:
    try:
        flags = json.loads(flags_path.read_text())
        antialiasing = bool(flags.get("antialiasing", False))
    except Exception as e:
        print(f"WARN không đọc được pipeline_train_flags.json: {e}", file=sys.stderr)
        have_flags = False

print(f"SH_DEGREE_DETECTED={sh_degree}")
print(f"ANTIALIASING_DETECTED={'1' if antialiasing else '0'}")
print(f"HAVE_FLAGS={'1' if have_flags else '0'}")
PYEOF
  )"
  eval "$CFG_INFO"

  if [[ "$HAVE_FLAGS" != "1" ]]; then
    echo "  [$SCENE] [CẢNH BÁO NGHIÊM TRỌNG] Không có pipeline_train_flags.json ở $MODEL_DIR — giả định" >&2
    echo "           antialiasing=false, CÓ THỂ SAI (xem docs/PORTED_KNOWLEDGE.md mục 2)." >&2
  fi

  THIS_SH_DEGREE="${SH_DEGREE:-$SH_DEGREE_DETECTED}"
  THIS_ANTIALIASING="${ANTIALIASING:-$ANTIALIASING_DETECTED}"
  if [[ -n "${SH_DEGREE:-}" && "$SH_DEGREE" != "$SH_DEGREE_DETECTED" ]]; then
    echo "  [$SCENE] [CẢNH BÁO] SH_DEGREE ép đè ($SH_DEGREE) khác giá trị tự dò từ checkpoint nguồn ($SH_DEGREE_DETECTED)." >&2
  fi
  if [[ -n "${ANTIALIASING:-}" && "$ANTIALIASING" != "$ANTIALIASING_DETECTED" ]]; then
    echo "  [$SCENE] [CẢNH BÁO] ANTIALIASING ép đè ($ANTIALIASING) khác giá trị tự dò từ checkpoint nguồn ($ANTIALIASING_DETECTED)." >&2
  fi

  MIP_ARGS=()
  if [[ "$THIS_ANTIALIASING" == "1" ]]; then
    MIP_ARGS+=(--antialiasing)
  fi

  # Cảnh báo sớm nếu đĩa sắp hết, giống 03_train_3dgs.sh.
  AVAIL_KB=$(df -Pk "$PIPELINE_DIR" | tail -1 | awk '{print $4}')
  AVAIL_GB=$((AVAIL_KB / 1024 / 1024))
  echo "[$SCENE] Đĩa còn trống: ${AVAIL_GB}GB"
  if [[ "$AVAIL_GB" -lt 5 ]]; then
    echo "[LỖI] Đĩa còn dưới 5GB trước khi refine $SCENE — dừng lại để tránh hỏng notebook giữa chừng." >&2
    exit 1
  fi

  TARGET_ITERATION=$((THIS_CKPT_ITERATION + REFINE_ITERATIONS))

  # train.py (không --start_checkpoint) LUÔN lưu output thô tại
  # point_cloud/iteration_$REFINE_ITERATIONS/ trước khi script này đổi tên (xem giải
  # thích đầu file). Nếu thư mục đó đã tồn tại TRƯỚC KHI train.py chạy (vd
  # REFINE_ITERATIONS trùng số với 1 checkpoint có sẵn — CKPT_ITERATION của chính vòng
  # này, hoặc còn sót lại từ 1 lần chạy dở dang trước), train.py sẽ GHI ĐÈ/pha trộn nó
  # mà không báo lỗi gì (train.py không biết/không quan tâm thư mục đó có ý nghĩa gì) —
  # rồi bước đổi tên phía sau tưởng đó là output mới toanh của chính lần chạy này. Chặn
  # sớm tại đây để không bao giờ dính bug ghi đè checkpoint có sẵn mà không hay biết.
  RAW_OUT_DIR="$MODEL_DIR/point_cloud/iteration_$REFINE_ITERATIONS"
  if [[ -d "$RAW_OUT_DIR" ]]; then
    echo "[LỖI] $SCENE: $RAW_OUT_DIR đã tồn tại TRƯỚC khi chạy train.py — REFINE_ITERATIONS" >&2
    echo "      ($REFINE_ITERATIONS) trùng số với 1 checkpoint có sẵn (hoặc còn sót từ lần chạy dở" >&2
    echo "      dang trước). Đổi REFINE_ITERATIONS khác, hoặc xoá/đổi tên $RAW_OUT_DIR nếu chắc chắn" >&2
    echo "      đó là rác — KHÔNG tự động ghi đè để tránh mất checkpoint thật." >&2
    exit 1
  fi

  echo "===== Error-refine: $SCENE (nạp iteration $THIS_CKPT_ITERATION, tinh chỉnh thêm $REFINE_ITERATIONS" \
       "iteration -> lưu tại iteration_$TARGET_ITERATION, sh_degree=$THIS_SH_DEGREE," \
       "antialiasing=$THIS_ANTIALIASING, densify_until_iter=$DENSIFY_UNTIL_ITER) — log: $LOG_FILE ====="

  # Chạy nền + poll tiến độ định kỳ, giống 03_train_3dgs.sh (tránh spam log dài trên
  # notebook nhưng vẫn biết đang chạy tới đâu).
  python "$GS_REPO/train.py" \
    -s "$SOURCE_DIR" \
    -m "$MODEL_DIR" \
    --refine_from_iteration "$THIS_CKPT_ITERATION" \
    --error_mask_dir "$ERROR_MASK_DIR" \
    --iterations "$REFINE_ITERATIONS" \
    --densify_until_iter "$DENSIFY_UNTIL_ITER" \
    --save_iterations "$REFINE_ITERATIONS" \
    --test_iterations "$REFINE_ITERATIONS" \
    --sh_degree "$THIS_SH_DEGREE" \
    "${MIP_ARGS[@]}" \
    > "$LOG_FILE" 2>&1 &
  TRAIN_PID=$!

  while kill -0 "$TRAIN_PID" 2>/dev/null; do
    sleep "${PROGRESS_INTERVAL:-60}"
    LAST_PROGRESS=$(grep -oE "[0-9]+/${REFINE_ITERATIONS}" "$LOG_FILE" 2>/dev/null | tail -1 || true)
    if [[ -n "$LAST_PROGRESS" ]]; then
      echo "  [$SCENE] tiến độ refine: $LAST_PROGRESS iterations"
    fi
  done

  set +e
  wait "$TRAIN_PID"
  STATUS=$?
  set -e

  if [[ $STATUS -ne 0 ]]; then
    echo "[LỖI] Refine thất bại cho $SCENE (exit $STATUS) — 50 dòng cuối log:" >&2
    tail -n 50 "$LOG_FILE" >&2
    echo "[CỨU ĐƯỢC] Checkpoint NGUỒN (trước refine) vẫn còn nguyên vẹn tại:" >&2
    echo "           $PLY_PATH (dùng tạm để render/nộp nếu cần, refine chưa ghi đè checkpoint gốc)." >&2
    LAST_NEW_CKPT="$(_latest_iteration_dir "$MODEL_DIR")"
    if [[ -n "$LAST_NEW_CKPT" && "$LAST_NEW_CKPT" != "$MODEL_DIR/point_cloud/iteration_$THIS_CKPT_ITERATION" ]]; then
      echo "[CỨU ĐƯỢC] Ngoài ra có checkpoint mới hơn (có thể dở dang) tại: $LAST_NEW_CKPT" >&2
    fi
    exit $STATUS
  fi

  # train.py (không --start_checkpoint) luôn đếm lại từ 0 -> lưu tại
  # point_cloud/iteration_<REFINE_ITERATIONS>/ (xem docstring apply_error_refine_patch.py).
  # Đổi tên thành iteration LUỸ KẾ thật để không đụng tên với vòng refine khác cùng
  # REFINE_ITERATIONS mặc định (xem giải thích đầu file này).
  RAW_OUT_DIR="$MODEL_DIR/point_cloud/iteration_$REFINE_ITERATIONS"
  TARGET_OUT_DIR="$MODEL_DIR/point_cloud/iteration_$TARGET_ITERATION"
  if [[ ! -d "$RAW_OUT_DIR" ]]; then
    echo "[LỖI] $SCENE: train.py báo thành công nhưng không thấy $RAW_OUT_DIR — kiểm tra lại log $LOG_FILE." >&2
    exit 1
  fi
  if [[ "$RAW_OUT_DIR" != "$TARGET_OUT_DIR" ]]; then
    if [[ -d "$TARGET_OUT_DIR" ]]; then
      echo "[LỖI] $SCENE: đích $TARGET_OUT_DIR đã tồn tại từ trước (checkpoint vòng refine khác?) —" >&2
      echo "      không tự ghi đè, kiểm tra tay rồi xoá/đổi tên nếu chắc chắn muốn ghi đè." >&2
      exit 1
    fi
    mv "$RAW_OUT_DIR" "$TARGET_OUT_DIR"
    echo "  -> Đổi tên $RAW_OUT_DIR -> $TARGET_OUT_DIR (số iteration luỹ kế thật)."
  fi
  echo "-> Xong $SCENE. Model đã refine: $TARGET_OUT_DIR/point_cloud.ply"

  # Ghi lại lịch sử refine vào pipeline_train_flags.json (giữ nguyên antialiasing/
  # depth_prior/exposure_comp/antenna_focus gốc — checkpoint nguồn dùng cấu hình gì thì
  # bản refine vẫn dùng đúng cấu hình đó, chỉ CỘNG THÊM 1 mục lịch sử refine, không mất
  # dấu vết các vòng trước — phục vụ yêu cầu tái lập kết quả, Đề bài mục 10.3).
  python3 - "$MODEL_DIR/pipeline_train_flags.json" "$THIS_CKPT_ITERATION" "$REFINE_ITERATIONS" \
    "$TARGET_ITERATION" "$MAX_ERROR_WEIGHT" "$DENSIFY_UNTIL_ITER" <<'PYEOF'
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

flags_path = Path(sys.argv[1])
refine_from_iteration = int(sys.argv[2])
refine_iterations = int(sys.argv[3])
output_iteration = int(sys.argv[4])
max_error_weight = float(sys.argv[5])
densify_until_iter = int(sys.argv[6])

flags = {}
if flags_path.exists():
    try:
        flags = json.loads(flags_path.read_text())
    except Exception:
        flags = {}

flags.setdefault("antialiasing", False)
flags.setdefault("depth_prior", False)
flags.setdefault("exposure_comp", False)
flags.setdefault("antenna_focus", False)
history = flags.get("refine_history", [])
history.append({
    "refine_from_iteration": refine_from_iteration,
    "refine_iterations": refine_iterations,
    "output_iteration": output_iteration,
    "max_error_weight": max_error_weight,
    "densify_until_iter": densify_until_iter,
    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
})
flags["refine_history"] = history

flags_path.write_text(json.dumps(flags, indent=2), encoding="utf-8")
print(f"  -> Đã cập nhật {flags_path} (refine_history: {len(history)} vòng).")
PYEOF

  if [[ "$CLEANUP_DENSE_IMAGES" == "1" && -d "$SOURCE_DIR/images" ]]; then
    FREED_KB=$(du -sk "$SOURCE_DIR/images" 2>/dev/null | awk '{print $1}')
    rm -rf "$SOURCE_DIR/images"
    echo "  [dọn đĩa] Đã xoá $SOURCE_DIR/images (~$((FREED_KB / 1024))MB, không cần cho render/eval/package) — giữ lại sparse/0."
  fi
done
