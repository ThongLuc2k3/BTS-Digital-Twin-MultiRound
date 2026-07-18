#!/usr/bin/env bash
# Train 3D Gaussian Splatting — Vòng 1 (baseline), 1 lần/scene, dùng repo GỐC
# graphdeco-inria/gaussian-splatting (không tự viết lại trainer — quá nhiều chi
# tiết dễ sai: densification, adaptive density control, SH coefficients...).
#
# Đây là script CỦA VÒNG 1 — chỉ giữ lại cấu hình đã đo THẬT là có lợi
# (`docs/PORTED_KNOWLEDGE.md` mục 2): antialiasing (mip-splatting, BẬT mặc định).
# KHÔNG hỗ trợ antenna-focus/depth-prior/exposure-comp ở đây — cả 3 đã đo thật ở
# repo tiền nhiệm trên holdout (HCM0421) và KHÔNG cải thiện Score tổng, nên bị
# loại khỏi baseline để giữ Vòng 1 đơn giản, nhanh, đáng tin cậy (xem
# `docs/00_MASTER_PLAN.md` mục 3.1). Muốn thử nghiệm thêm thì làm ở Vòng 2+.
#
# Bản gaussian-splatting đã pin (từ bản cập nhật 10/2024) đã TÍCH HỢP SẴN đúng
# "EWA Filter" của Mip-Splatting làm cờ `--antialiasing` (đối chiếu trực tiếp
# README.md + gaussian_renderer/__init__.py của repo thật, không suy đoán) —
# KHÔNG cần clone riêng autonomousvision/mip-splatting hay đổi rasterizer.
#
# Cài đặt 1 lần (máy có GPU CUDA, chạy trước khi dùng script này):
#   git clone --recursive https://github.com/graphdeco-inria/gaussian-splatting.git
#   cd gaussian-splatting
#   git checkout 54c035f7834b564019656c3e3fcc3646292f727d   # PIN commit đã xác nhận có antialiasing
#   git submodule update --init --recursive                  # re-sync submodule đúng theo commit vừa checkout
#   conda env create --file environment.yml   # hoặc tự pip install theo requirements.txt của repo
#   conda activate gaussian_splatting
#
# Set biến môi trường GS_REPO trỏ tới thư mục clone ở trên trước khi chạy script này:
#   export GS_REPO=/path/to/gaussian-splatting
#
# Cách dùng:
#   ./02_train_baseline.sh HCM0421                 # train 1 scene (mặc định: antialiasing BẬT)
#   ./02_train_baseline.sh HCM0421 HCM0539 chair   # train nhiều scene liên tiếp
#   ITERATIONS=15000 ./02_train_baseline.sh HCM0421   # đổi số iteration (mặc định 30000 của repo)
#   PROGRESS_INTERVAL=30 ./02_train_baseline.sh HCM0421  # in tiến độ mỗi 30s thay vì 60s mặc định
#   ANTIALIASING=0 ./02_train_baseline.sh HCM0421     # tắt antialiasing để A/B so với bản có (mặc định BẬT)
#
# Nếu bị "CUDA out of memory" (hay gặp với scene nhiều chi tiết mảnh — dây cáp,
# khung thép BTS — vì số Gaussian sinh ra qua densify tăng rất nhanh), thử lần
# lượt theo thứ tự (mỗi lần giảm 1 mức, không cần giảm hết cùng lúc):
#   1) Không cần làm gì — script đã tự set PYTORCH_CUDA_ALLOC_CONF để giảm phân
#      mảnh bộ nhớ (đúng như gợi ý trong thông báo lỗi gốc của PyTorch).
#   2) SH_DEGREE=2 ./02_train_baseline.sh HCM0421          (giảm dữ liệu màu/Gaussian, ảnh hưởng chất lượng ít)
#   3) DENSIFY_GRAD_THRESHOLD=0.0004 ./02_train_baseline.sh HCM0421   (hạn chế sinh thêm Gaussian, mặc định repo 0.0002)
#   4) RESOLUTION=2 ./02_train_baseline.sh HCM0421          (train ở nửa độ phân giải, giảm mạnh nhất nhưng ảnh hưởng chi tiết)
#   Có thể kết hợp nhiều biến cùng lúc, vd: SH_DEGREE=2 DENSIFY_GRAD_THRESHOLD=0.0004 ./02_train_baseline.sh HCM0421
#
# Input mong đợi: pipeline/work/<scene>/colmap/dense/{images/,sparse/0/}
#                 (do 01_run_colmap.py tạo ra)
# Output: pipeline/work/<scene>/gs_model/point_cloud/iteration_<N>/point_cloud.ply
#         (có checkpoint giữa chừng ở 7000/15000 — nếu train bị crash muộn hơn,
#          vẫn dùng được model ở checkpoint gần nhất thay vì mất trắng)
#
# Dọn đĩa giữa các scene: sau khi train xong 1 scene, script tự xoá
# colmap/dense/images/ của scene đó (bản ảnh full-res undistort chỉ lúc train
# cần — 03_render_test_poses.py/04_eval_metrics.py không đụng tới, chỉ cần
# point_cloud.ply). Quan trọng khi chạy nhiều scene trong 1 lệnh: nếu không xoá,
# dữ liệu dense của các scene TRƯỚC vẫn nằm nguyên trên đĩa cộng dồn tới khi hết
# dung lượng giữa chừng (đã từng gặp "OSError: No space left on device" ở repo
# tiền nhiệm). Set CLEANUP_DENSE_IMAGES=0 nếu muốn giữ lại để debug COLMAP sau này.

set -euo pipefail

if [[ -z "${GS_REPO:-}" ]]; then
  echo "Lỗi: chưa set biến môi trường GS_REPO (đường dẫn tới repo graphdeco-inria/gaussian-splatting đã clone)." >&2
  exit 1
fi
if [[ ! -f "$GS_REPO/train.py" ]]; then
  echo "Lỗi: không thấy $GS_REPO/train.py — kiểm tra lại GS_REPO." >&2
  exit 1
fi

# Giảm lỗi CUDA OOM do phân mảnh bộ nhớ (khuyến nghị chính thức của PyTorch khi
# gặp "reserved but unallocated memory is large") — không đánh đổi chất lượng,
# nên bật mặc định luôn, không cần người dùng tự nhớ set.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PIPELINE_DIR="$(dirname "$SCRIPT_DIR")"
ITERATIONS="${ITERATIONS:-30000}"
SH_DEGREE="${SH_DEGREE:-3}"
DENSIFY_GRAD_THRESHOLD="${DENSIFY_GRAD_THRESHOLD:-0.0002}"
RESOLUTION="${RESOLUTION:--1}"
CLEANUP_DENSE_IMAGES="${CLEANUP_DENSE_IMAGES:-1}"
ANTIALIASING="${ANTIALIASING:-1}"

if [[ $# -eq 0 ]]; then
  echo "Cách dùng: $0 <scene1> [scene2 ...]" >&2
  exit 1
fi

if [[ "$ANTIALIASING" == "1" ]] && ! grep -q "antialiasing" "$GS_REPO/arguments/__init__.py" 2>/dev/null; then
  echo "Lỗi: ANTIALIASING=1 nhưng \$GS_REPO có vẻ là bản clone CŨ (trước 10/2024, chưa có cờ" >&2
  echo "  --antialiasing). Checkout đúng commit đã pin (xem comment đầu file này) rồi thử lại," >&2
  echo "  hoặc set ANTIALIASING=0 nếu cố ý muốn train bản không chống alias để so sánh." >&2
  exit 1
fi

# Checkpoint giữa chừng ở 7000/15000 (nếu ITERATIONS đủ lớn) để không mất trắng
# nếu crash muộn hơn (vd OOM ở densify) — trước đây chỉ lưu đúng lúc kết thúc.
SAVE_ITERATIONS=()
for v in 7000 15000 "$ITERATIONS"; do
  if [[ "$v" -le "$ITERATIONS" ]]; then
    SAVE_ITERATIONS+=("$v")
  fi
done
SAVE_ITERATIONS=($(printf "%s\n" "${SAVE_ITERATIONS[@]}" | awk '!seen[$0]++'))

# Tự dò thư mục point_cloud/iteration_<N> có N LỚN NHẤT trong 1 MODEL_DIR — dùng cho
# thông báo "[CỨU ĐƯỢC]" khi train thất bại giữa chừng (bên dưới). KHÔNG dùng
# `ls | sort -t_ -k2 -n | tail -1`: đã tự phát hiện bug thật (xem
# `pipeline/scripts/06_train_refine.sh` — cùng bug class đã fix ở đó) — đường dẫn thật chứa NHIỀU
# dấu "_" khác đứng TRƯỚC "iteration_N" (vd "gs_model", "point_cloud"), nên trường số 2
# (-k2) không phải là số iteration; kết quả `sort -n` coi field đó là 0 (không phải
# số), sort giữ nguyên thứ tự lexical của `ls` (vd "iteration_15000" đứng TRƯỚC
# "iteration_7000" vì '1' < '7'), khiến `tail -1` chọn NHẦM checkpoint NHỎ HƠN
# (7000) thay vì lớn hơn thật sự (15000) — sai lệch thông báo cứu hộ, có thể khiến
# người dùng dùng nhầm checkpoint kém train hơn. Sửa bằng vòng lặp bash thuần, so sánh
# số nguyên, không tách trường theo dấu "_"/khoảng trắng ở đâu cả (đường dẫn dự án thật
# còn chứa dấu cách "Khóa Luận Tốt Nghiệp" nữa, awk/sort trên đó cũng không an toàn).
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
  LOG_FILE="$PIPELINE_DIR/work/$SCENE/02_train_baseline.log"

  if [[ ! -d "$SOURCE_DIR/sparse/0" ]]; then
    echo "[BỎ QUA] $SCENE: chưa thấy $SOURCE_DIR/sparse/0 — chạy 01_run_colmap.py --scene $SCENE trước." >&2
    continue
  fi

  # Chặn re-run âm thầm đè lên checkpoint SỐ CŨ khác cấu hình (đã tự phát hiện + verify
  # bằng test thật ở verification pass #5, KHÔNG suy đoán): script này không
  # --start_checkpoint nên MỖI lần chạy train lại từ đầu (iteration 0); nếu MODEL_DIR đã
  # có sẵn point_cloud/ từ 1 lần chạy TRƯỚC (vd đã chạy MODE=holdout rồi thử lại A/B với
  # ANTIALIASING khác, hoặc lần trước bị crash giữa chừng) và lần này ITERATIONS NHỎ HƠN
  # lần trước, các thư mục iteration_<N> SỐ LỚN của lần trước sẽ CÒN SÓT LẠI nguyên vẹn
  # (train.py chỉ ghi/ghi đè đúng các SAVE_ITERATIONS của lần chạy NÀY) trong khi
  # pipeline_train_flags.json bị ghi đè theo cấu hình MỚI ở cuối script — kết quả:
  # 03_render_test_poses.py::find_latest_iteration() mặc định chọn iteration SỐ LỚN NHẤT
  # (checkpoint SÓT LẠI, cấu hình CŨ) nhưng đọc antialiasing từ pipeline_train_flags.json
  # (đã là cấu hình MỚI) -> lệch antialiasing giữa checkpoint thật và flags đọc được, làm
  # méo hoàn toàn PSNR/SSIM/LPIPS mà KHÔNG có lỗi báo (đúng loại bug đã cảnh báo ở
  # docs/PORTED_KNOWLEDGE.md mục 2, nhưng do nguyên nhân MỚI: checkpoint sót lại từ
  # re-run, không phải thiếu file). Đã verify bằng mock thật: train ANTIALIASING=1
  # ITERATIONS=15000 (tạo iteration_7000+15000, antialiasing=true) rồi re-run
  # ANTIALIASING=0 ITERATIONS=7000 (chỉ ghi đè iteration_7000) — xác nhận iteration_15000
  # còn nguyên NỘI DUNG antialiasing=true trong khi pipeline_train_flags.json đã đổi
  # thành antialiasing:false. KHÔNG tự động xoá/ghi đè — báo lỗi rõ, để user tự quyết.
  if [[ -d "$MODEL_DIR/point_cloud" ]] && \
     [[ -n "$(find "$MODEL_DIR/point_cloud" -mindepth 1 -maxdepth 1 -print -quit 2>/dev/null)" ]]; then
    echo "[BỎ QUA] $SCENE: $MODEL_DIR/point_cloud đã có checkpoint từ (các) lần chạy TRƯỚC." >&2
    echo "         Script này luôn train lại từ đầu (không --start_checkpoint) — nếu ITERATIONS lần" >&2
    echo "         này khác lần trước (hoặc ANTIALIASING/SH_DEGREE đổi), các checkpoint iteration_<N>" >&2
    echo "         SỐ LỚN của lần trước có thể CÒN SÓT LẠI với cấu hình CŨ trong khi" >&2
    echo "         pipeline_train_flags.json bị ghi đè theo cấu hình MỚI, khiến bước render sau" >&2
    echo "         này âm thầm chọn NHẦM checkpoint sai cấu hình (xem chi tiết trong comment code)." >&2
    echo "         Xoá $MODEL_DIR/point_cloud (hoặc cả $MODEL_DIR) rồi chạy lại nếu CỐ Ý muốn train" >&2
    echo "         lại từ đầu, hoặc set CLEAN_MODEL_DIR=1 để script tự xoá trước khi train $SCENE." >&2
    if [[ "${CLEAN_MODEL_DIR:-0}" == "1" ]]; then
      echo "  [$SCENE] CLEAN_MODEL_DIR=1 — tự xoá $MODEL_DIR/point_cloud trước khi train lại từ đầu."
      rm -rf "$MODEL_DIR/point_cloud"
    else
      continue
    fi
  fi

  # Cảnh báo sớm nếu đĩa sắp hết TRƯỚC KHI đâm đầu vào train (có thể mất vài
  # tiếng) — tốt hơn là để nó chết giữa chừng lúc lưu checkpoint. Ngưỡng 5GB là
  # ước lượng an toàn (1 checkpoint point_cloud.ply có thể nặng cỡ vài trăm MB
  # tới hơn 1GB tuỳ số Gaussian sau densify).
  AVAIL_KB=$(df -Pk "$PIPELINE_DIR" | tail -1 | awk '{print $4}')
  AVAIL_GB=$((AVAIL_KB / 1024 / 1024))
  echo "[$SCENE] Đĩa còn trống: ${AVAIL_GB}GB"
  if [[ "$AVAIL_GB" -lt 5 ]]; then
    echo "[LỖI] Đĩa còn dưới 5GB trước khi train $SCENE — dừng lại để tránh hỏng notebook giữa chừng." >&2
    echo "       Dọn bớt (vd rm -rf pipeline/work/<scene cũ>/colmap/dense/images) rồi chạy lại." >&2
    exit 1
  fi

  MIP_ARGS=()
  if [[ "$ANTIALIASING" == "1" ]]; then
    MIP_ARGS+=(--antialiasing)
  fi

  # train.py in progress bar (tqdm) qua hàng chục nghìn iteration — rất dài nếu
  # hiện hết ra console/notebook, nên vẫn redirect toàn bộ ra file log. Nhưng
  # chạy nền (&) rồi định kỳ lấy đúng số "hiện tại/ITERATIONS" cuối cùng trong
  # log để in 1 dòng gọn ra console — biết đang chạy tới đâu mà không bị spam.
  # Đổi tần suất bằng PROGRESS_INTERVAL=<giây> (mặc định 60s).
  echo "===== Train 3DGS (Vòng 1 - baseline): $SCENE ($ITERATIONS iterations, sh_degree=$SH_DEGREE, densify_grad_threshold=$DENSIFY_GRAD_THRESHOLD, antialiasing=$ANTIALIASING) — log: $LOG_FILE ====="
  python "$GS_REPO/train.py" \
    -s "$SOURCE_DIR" \
    -m "$MODEL_DIR" \
    --iterations "$ITERATIONS" \
    --save_iterations "${SAVE_ITERATIONS[@]}" \
    --test_iterations "$ITERATIONS" \
    --sh_degree "$SH_DEGREE" \
    --densify_grad_threshold "$DENSIFY_GRAD_THRESHOLD" \
    --resolution "$RESOLUTION" \
    "${MIP_ARGS[@]}" \
    > "$LOG_FILE" 2>&1 &
  # Không dùng --eval: ta muốn dùng TOÀN BỘ ảnh input (train/images/ đầy đủ, hoặc
  # phần không-holdout ở chế độ MODE=holdout) để train (không giữ lại phần nào
  # làm test nội bộ của repo) — việc tự đánh giá chất lượng làm riêng bằng
  # 03_render_test_poses.py + 04_eval_metrics.py trên holdout tự tạo.
  TRAIN_PID=$!

  while kill -0 "$TRAIN_PID" 2>/dev/null; do
    sleep "${PROGRESS_INTERVAL:-60}"
    # `|| true` BẮT BUỘC: script này bật `set -euo pipefail` ở đầu, nên nếu
    # grep không tìm thấy match (exit code 1, hay gặp ở những vòng poll đầu khi
    # log chưa in dòng tiến độ nào) thì `pipefail` sẽ khiến cả pipeline
    # `grep | tail -1` trả về exit != 0 -> `set -e` giết chết luôn cả script
    # ngay giữa lúc train đang chạy nền, dù train.py không hề lỗi gì (đã gặp
    # bug thật này ở repo tiền nhiệm, xem docs/PORTED_KNOWLEDGE.md mục 6).
    # `|| true` chặn đúng chỗ này, không ảnh hưởng gì tới việc bắt lỗi thật của
    # tiến trình train (STATUS ở dưới vẫn kiểm tra riêng qua `wait "$TRAIN_PID"`).
    LAST_PROGRESS=$(grep -oE "[0-9]+/${ITERATIONS}" "$LOG_FILE" 2>/dev/null | tail -1 || true)
    if [[ -n "$LAST_PROGRESS" ]]; then
      echo "  [$SCENE] tiến độ: $LAST_PROGRESS iterations"
    fi
  done

  set +e
  wait "$TRAIN_PID"
  STATUS=$?
  set -e

  if [[ $STATUS -ne 0 ]]; then
    echo "[LỖI] Train thất bại cho $SCENE (exit $STATUS) — 50 dòng cuối log:" >&2
    tail -n 50 "$LOG_FILE" >&2
    LAST_CKPT="$(_latest_iteration_dir "$MODEL_DIR")"
    if [[ -n "$LAST_CKPT" ]]; then
      echo "[CỨU ĐƯỢC] Vẫn còn checkpoint gần nhất tại: $LAST_CKPT (dùng tạm để render nếu cần)." >&2
    fi
    exit $STATUS
  fi
  echo "-> Xong $SCENE. Model: $MODEL_DIR/point_cloud/iteration_$ITERATIONS/point_cloud.ply"

  # `antialiasing` là field của PipelineParams trong repo Inria gốc, còn cfg_args
  # mà train.py tự ghi CHỈ chứa ModelParams (xem train.py::training(), dòng
  # `tb_writer = prepare_output_and_logger(dataset)` với dataset=lp.extract(args))
  # -> cfg_args KHÔNG BAO GIỜ có field antialiasing, dù có bật --antialiasing hay
  # không (đã đối chiếu trực tiếp source thật ở repo tiền nhiệm). Nếu render tự
  # "đoán" antialiasing từ cfg_args sẽ LUÔN ra False — SAI hoàn toàn nếu lúc train
  # đã bật, làm méo PSNR/SSIM/LPIPS mà KHÔNG có lỗi báo (xem
  # docs/PORTED_KNOWLEDGE.md mục 2). Ghi lại đúng giá trị thật đã dùng lúc train
  # ra 1 file riêng để 03_render_test_poses.py đọc lại — GIỮ NGUYÊN schema cũ
  # (depth_prior/exposure_comp/antenna_focus luôn false vì script này không hỗ
  # trợ) để mọi code downstream đọc các key đó không bị lỗi thiếu key.
  cat > "$MODEL_DIR/pipeline_train_flags.json" <<EOF
{"antialiasing": $( [[ "$ANTIALIASING" == "1" ]] && echo true || echo false ), "depth_prior": false, "exposure_comp": false, "antenna_focus": false}
EOF

  if [[ "$CLEANUP_DENSE_IMAGES" == "1" && -d "$SOURCE_DIR/images" ]]; then
    FREED_KB=$(du -sk "$SOURCE_DIR/images" 2>/dev/null | awk '{print $1}')
    rm -rf "$SOURCE_DIR/images"
    echo "  [dọn đĩa] Đã xoá $SOURCE_DIR/images (~$((FREED_KB / 1024))MB, không cần cho render/eval) — giữ lại sparse/0."
  fi
done
