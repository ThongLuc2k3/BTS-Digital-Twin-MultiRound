# Milestone 08 — Kiểm tra độc lập lần 5 (verification pass #5 / N)

## Trạng thái hiện tại

**HOÀN TẤT pass #5 — TÌM THẤY + SỬA 3 BUG THẬT MỚI.** Pass #4 là lần sạch đầu tiên (bộ
đếm 1/3). Pass này đổi hướng theo đúng chỉ đạo nhiệm vụ (không lặp lại phạm vi 4 pass
trước — audit các fix cũ), mà chủ động **thử phá bằng input/kịch bản chưa ai test**
(re-run cùng scene, file input rỗng/hỏng, dependency phiên bản MỚI NHẤT thật) — tìm ra
3 bug thật, cả 3 đều verify bằng THỰC THI THẬT (không suy đoán). **Vì có bug thật, bộ
đếm 3-lần-liên-tiếp-không-lỗi RESET VỀ 0/3.** Cần lại đủ 3 pass sạch liên tiếp kể từ
pass tiếp theo.

## Phạm vi đã làm

Đọc đầy đủ STEP 0: `docs/00_MASTER_PLAN.md`, `docs/PORTED_KNOWLEDGE.md` toàn bộ (kể cả
mục 6b/6c/6d), cả 8 milestone log trước (`MILESTONE_00` → `MILESTONE_07`), `git log
--oneline`, `git show` đầy đủ diff của pass #4 (`dbbf92a`) để biết chính xác phạm vi
ĐàQ được audit kỹ (giảm trùng lặp, tập trung vào góc mới).

### Phần A — Baseline: test suite hiện có

- `tests/test_syntax_all.py`: 15/15 `.py` PASS + 4/4 `.ipynb` PASS.
- `tests/test_07_package_submission.py`: 21/21 PASS.
- `bash -n` cho `02_train_baseline.sh` + `06_train_refine.sh`: sạch.
- Chạy lại lần cuối SAU khi sửa xong 3 bug (mục C) — vẫn PASS 100%, không có
  regression.

### Phần B — Adversarial input testing (chưa pass nào làm theo đúng tinh thần này)

1. **`error_masks/manifest.json` rỗng (0 byte)** — test thật bằng
   `06_train_refine.sh` trên fixture giả (checkpoint + manifest 0 byte): script crash
   với `json.decoder.JSONDecodeError` (Python traceback đầy đủ, KHÔNG bị nuốt) + in
   thêm dòng `[LỖI] <scene>: đối chiếu error_masks/manifest.json thất bại: ` + **exit
   code 1** (không phải giả-thành-công, không treo). Đúng nhánh "acceptable" theo tiêu
   chí nhiệm vụ đặt ra — **không phải bug**, chỉ ghi nhận đã verify thật (không suy
   đoán).
2. **`gdown --fuzzy --folder` chạy thật với URL bogus** (dùng `gdown==5.2.2`, bản CÓ hỗ
   trợ cú pháp này để cô lập đúng câu hỏi "link chết thì sao", tách khỏi bug #3 ở mục
   C) — quan sát thật: gdown in "Retrieving folder contents" → "Failed to retrieve
   folder contents" → **exit code 0** (im lặng "thành công"), thư mục output RỖNG
   HOÀN TOÀN. Đối chiếu `kaggle_submission.ipynb` Bước 5: dòng
   `candidates = [p.parent for p in raw_dl_dir.rglob("cfg_args")]` +
   `assert candidates, f"...KHÔNG tìm thấy file 'cfg_args'..."` **KHÔNG dựa vào exit
   code của gdown** — tự kiểm tra trực tiếp nội dung thư mục tải về, nên VẪN bắt được
   lỗi này đúng như thiết kế, dù gdown tự nó không báo lỗi qua exit code. **Không phải
   bug** — cơ chế phòng vệ hiện có đã đủ cho đúng kịch bản "link chết/bogus" này.
3. **`gdown --fuzzy` (cú pháp cả 4 notebook đang dùng) chạy thật với gdown mới nhất
   PyPI (6.1.0)** — xem mục C hạng mục 3, đây là phát hiện LỚN NHẤT của pass này.
4. **2 scene trong 1 lệnh, scene ĐẦU crash giữa chừng** — test thật bằng mock cho CẢ
   `02_train_baseline.sh` (`MOCK_CRASH`, 2 scene `sceneA`/`sceneB`) LẪN
   `06_train_refine.sh` (`MOCK_CRASH`, 2 scene `refA`/`refB`): cả 2 script đều
   **ABORT TOÀN BỘ lệnh** khi scene đầu crash thật (train.py exit != 0) — scene thứ 2
   KHÔNG được thử. Đối chiếu với nhánh "[BỎ QUA]" (thiếu precondition dữ liệu — sparse
   thiếu, manifest thiếu...) vốn `continue` sang scene kế: 2 hành vi khác nhau NHƯNG
   **nhất quán có chủ đích** giữa 2 script (không phải ngẫu nhiên) — lỗi
   PRECONDITION DỮ LIỆU (scene-specific, an toàn bỏ qua) dùng `continue`; lỗi
   TRAIN.PY THẬT CRASH hoặc TÀI NGUYÊN HỆ THỐNG (disk, GS_REPO sai — có khả năng ảnh
   hưởng CẢ scene sau) dùng `exit`. **Không phải bug** — verify hành vi thật khớp đúng
   design pattern nhất quán đã dùng ở cả 2 script.
5. **`get_scene()` raise với SCENE sai** — trace toàn bộ: KHÔNG có bash `for SCENE in
   ...` loop nào gọi `get_scene()` trực tiếp (2 script `.sh` duy nhất có loop nhiều
   scene — `02_train_baseline.sh`/`06_train_refine.sh` — hoàn toàn không import
   `common.scenes`, chỉ thao tác đường dẫn bash thuần). 5 script `.py` gọi
   `get_scene(args.scene)` đều là lệnh `!python ... --scene {SCENE}` ĐƠN LẺ (1 scene/
   lần gọi, đúng thiết kế "1 scene/version" của notebook) — ValueError raise ra
   traceback đầy đủ (hiển thị trong cell output, không bị nuốt) nhưng không tự dừng
   "Run All" (đúng cạm bẫy ĐÃ ghi ở `PORTED_KNOWLEDGE.md` mục 4, không phải phát hiện
   mới). Các bước sau (04_train_baseline.sh check `sparse/0`, `find_latest_iteration`
   check `point_cloud/`...) đều tự in `[BỎ QUA]`/raise rõ ràng khi thiếu dữ liệu do
   bước trước fail — không có bước nào "giả vờ thành công". **Không phải bug mới**,
   chỉ xác nhận lại hành vi đã biết.

### Phần C — 3 bug thật tìm ra + sửa (verify bằng thực thi thật, không suy đoán)

**1. `02_train_baseline.sh` không chặn re-run đè checkpoint SỐ CŨ khác cấu hình.**
Dựng mock `train.py` (bash thuần + Python argparse giả, ghi `.ply`/`cfg_args` giả theo
đúng `--save_iterations`) trong scratch riêng. Kịch bản: RUN 1
`ANTIALIASING=1 ITERATIONS=15000` (tạo `iteration_7000` + `iteration_15000`, cả 2
`antialiasing=true`, thành công) → RUN 2 (re-run CÙNG scene, KHÔNG dọn `gs_model/`)
`ANTIALIASING=0 ITERATIONS=7000` (chỉ tạo/ghi đè `iteration_7000`) → **xác nhận thật**:
`iteration_15000/point_cloud.ply` CÒN NGUYÊN nội dung `antialiasing=True` (checkpoint
CŨ, KHÔNG bị đụng) trong khi `pipeline_train_flags.json` đã bị ghi đè thành
`antialiasing:false` (từ RUN 2). `03_render_test_poses.py::find_latest_iteration()`
mặc định chọn SỐ LỚN NHẤT (`15000`, checkpoint CŨ) — kết hợp với flags MỚI gây lệch
antialiasing giữa checkpoint thật và cấu hình đọc được, y hệt bug "méo PSNR/SSIM/LPIPS
không báo lỗi" đã cảnh báo ở `PORTED_KNOWLEDGE.md` mục 2, nhưng do nguyên nhân MỚI.

Trigger THẬT (không phải kịch bản giả): `kaggle_round1_baseline.ipynb` Bước 6 khuyến
nghị tường minh "chạy vài version `MODE=holdout` (15000 iter) RỒI 1 version
`MODE=final` (30000 iter)" cho CÙNG 1 scene — nếu làm theo thứ tự NGƯỢC hoặc thử lại
`ANTIALIASING` khác trong CÙNG phiên tương tác (không restart kernel giữa 2 lần —
đúng workflow A/B mà chính usage-comment của `02_train_baseline.sh` gợi ý:
`ANTIALIASING=0 ./02_train_baseline.sh HCM0421 # tắt antialiasing để A/B so với bản
có`), checkpoint `iteration_30000` (train 100% ảnh, dùng để nộp bài) có thể CÒN SÓT
LẠI và bị dùng NHẦM để tính Score holdout — vừa sai antialiasing vừa RÒ RỈ DỮ LIỆU
(model final đã "thấy" ảnh holdout lúc train), làm Score đo được mất hết ý nghĩa mà
không có cảnh báo nào.

**Đã sửa**: thêm guard đầu vòng lặp scene trong `02_train_baseline.sh` (ngay sau check
`sparse/0`) — nếu `$MODEL_DIR/point_cloud` đã có checkpoint từ trước, in `[BỎ QUA]`
(không phá batch nhiều scene — nhất quán với cách xử lý thiếu `sparse/0`) + giải thích
rõ rủi ro + hướng dẫn xoá thủ công hoặc set `CLEAN_MODEL_DIR=1` để script tự xoá trước
khi train lại. Verify lại bằng mock: RUN 2 (default, không set `CLEAN_MODEL_DIR`) bị
chặn đúng, `iteration_15000` + `pipeline_train_flags.json` giữ nguyên KHÔNG đổi (không
có mismatch nào xảy ra); RUN 3 (`CLEAN_MODEL_DIR=1`) tự xoá `point_cloud/` rồi train
lại sạch, chỉ còn đúng checkpoint mới + flags khớp. Batch 2 scene (`chair` bị chặn +
`freshscene` mới) xác nhận scene bị chặn KHÔNG ảnh hưởng scene khác vẫn train bình
thường (đúng convention `continue`, không `exit`).

**2. `05_generate_error_mask.py` không dọn `error_masks/` cũ trước khi ghi mask mới.**
Đọc trực tiếp `apply_error_refine_patch.py::_error_mask()` (hàm đọc mask lúc train
refine): tra mask theo ĐÚNG TÊN FILE (`stem + ".png"`) cho MỌI ảnh train, **không hề
đối chiếu với `manifest.json`** để biết ảnh nào thuộc lần sinh mask nào — chỉ cần file
tồn tại là dùng, bất kể là mask mới hay mask sót lại từ lần chạy trước với
`--max_weight`/`--blur_radius`/iteration KHÁC. Vì `05_generate_error_mask.py` trước
đây chỉ `out_dir.mkdir(exist_ok=True)` rồi ghi đè đúng ảnh nằm trong lần chạy NÀY
(không xoá gì trước), 2 kịch bản để lại mask cũ SÓT LẠI THẬT:
  - `--n_images` debug (lấy mẫu 1 phần) chạy SAU 1 lần full-sweep trước đó — chỉ
    N ảnh mẫu được ghi đè, các ảnh còn lại giữ nguyên mask CŨ.
  - Crash giữa chừng (`manifest.json` chỉ ghi ở CUỐI hàm `main()`, SAU khi xử lý hết
    ảnh) — nếu crash ở ảnh thứ K/N, out_dir có K mask MỚI trộn N-K mask CŨ, nhưng
    `manifest.json` (nếu tồn tại) vẫn là bản CŨ từ lần chạy trước, không phản ánh
    đúng trạng thái thật của thư mục.

Verify logic dọn dẹp bằng test thật (dựng 3 mask + manifest giả trong scratch, chạy
đúng đoạn code đã thêm, xác nhận: mask KHÔNG nằm trong lần chạy mới bị xoá sạch, mask
CÓ trong lần chạy mới giữ đúng nội dung mới, không sót gì).

**Đã sửa**: thêm bước dọn `*.png` + `manifest.json` cũ (nếu có) trong `out_dir` NGAY
ĐẦU `main()`, TRƯỚC khi xử lý ảnh nào — đảm bảo `error_masks/` LUÔN phản ánh ĐÚNG VÀ
CHỈ đúng 1 lần chạy gần nhất. Tác dụng phụ có lợi: nếu crash giữa chừng sau khi sửa,
thư mục chỉ còn 1 phần mask MỚI + KHÔNG CÓ `manifest.json` (thay vì mask MỚI trộn CŨ +
manifest CŨ) — `06_train_refine.sh` tự phát hiện thiếu `manifest.json` và `[BỎ QUA]`
rõ ràng thay vì âm thầm dùng nhầm manifest cũ không khớp thực trạng thư mục.

**3. `gdown --fuzzy` không còn tồn tại ở gdown ≥ 6.0.0 (bản mới nhất PyPI hiện tại,
6.1.0) — bug NGHIÊM TRỌNG NHẤT tìm được ở pass này.** Cả 4 notebook luôn
`!pip install -q ... gdown` KHÔNG PIN version — mỗi phiên Kaggle mới sẽ tự lấy bản mới
nhất tại thời điểm chạy. Verify bằng THỰC THI THẬT (đúng yêu cầu nhiệm vụ, không suy
đoán): cài `gdown==6.1.0` (xác nhận đúng là bản LATEST trên PyPI qua
`pip index versions gdown` lúc viết milestone này) trong venv sạch, chạy đúng 2 dạng
lệnh notebook đang dùng:
```
gdown --fuzzy "<url>" -O test.zip                       # dataset (4 notebook)
gdown --fuzzy --folder "<url>" -O <dir>                  # checkpoint (3 notebook)
```
CẢ 2 đều lỗi CLI NGAY LẬP TỨC: `gdown: error: unrecognized arguments: --fuzzy` (exit
2), TRƯỚC KHI kịp tải bất kỳ byte nào — không phải lỗi mạng/link, mà lỗi cú pháp dòng
lệnh xảy ra với BẤT KỲ URL nào (kể cả URL đúng 100%). Đối chiếu ngược `gdown==5.2.2`
xác nhận `--fuzzy` THẬT SỰ từng tồn tại (help text: `(file only) extract Google
Drive's file ID`) — bị xoá hẳn ở `gdown==6.0.0` (đã cài thử cả 2 bản, so `--help`
trực tiếp, không suy đoán từ changelog).

Tin TỐT xác nhận CÙNG LÚC bằng thực thi thật: gọi trực tiếp
`gdown.parse_url.parse_url(url)` (hàm lõi suy ra file ID từ URL, dùng nội bộ bởi CLI)
ở bản 6.1.0 — vẫn tự nhận diện ĐÚNG file ID từ URL dạng share-link đầy đủ
(`.../file/d/<id>/view?usp=drive_link`) MÀ KHÔNG CẦN tham số `fuzzy` nào (chữ ký hàm
đã đổi thành `parse_url(url: str) -> tuple[str | None, bool]`, không còn tham số
`fuzzy` — hành vi "fuzzy" trở thành MẶC ĐỊNH LUÔN BẬT, cờ bị xoá vì hết cần thiết chứ
KHÔNG phải vì tính năng URL-parsing bị xoá). Verify lại bằng thực thi thật SAU khi sửa
(bỏ hẳn `--fuzzy`): `gdown --json "<GDRIVE_URL thật của dự án>"` (không tải, chỉ liệt
kê metadata — an toàn, không tốn băng thông/dung lượng) trả về đúng
`"path": "VAI_NVS_DATA_ROUND2.zip"` — xác nhận VẪN resolve đúng file dataset thật của
dự án, không phải chỉ hết lỗi cú pháp suông mà còn hoạt động đúng chức năng.

**Đã sửa**: bỏ `--fuzzy` khỏi TẤT CẢ lệnh `gdown` (dataset lẫn checkpoint) ở cả 4
notebook (`kaggle_round1_baseline.ipynb`, `kaggle_round2_refine.ipynb`,
`kaggle_round3_refine.ipynb`, `kaggle_submission.ipynb`) bằng script sửa trực tiếp qua
`nbformat` (không copy tay JSON) + 1 dòng markdown/comment còn nhắc `--fuzzy` trong
`kaggle_submission.ipynb`. `grep -rn fuzzy pipeline/*.ipynb` sau khi sửa: không còn kết
quả nào. `nbformat.validate()` lại cả 4 file: vẫn PASS.

### Phần D — Fact-check 3 tuyên bố tài liệu load-bearing (đối chiếu trực tiếp code
HIỆN TẠI, không chỉ tin milestone log cũ)

1. **Công thức Score + `PSNR_max`** — đối chiếu `04_eval_metrics.py::compute_score()`:
   `return 0.4 * lpips_term + 0.3 * ssim_v + 0.3 * psnr_norm` với
   `psnr_norm = min(max(psnr_v / psnr_max, 0.0), 1.0)`, `--psnr_max` default `50.0` —
   khớp CHÍNH XÁC tuyên bố `docs/00_MASTER_PLAN.md` mục 1 +
   `docs/PORTED_KNOWLEDGE.md` mục 5. **Đúng.**
2. **Commit pin `54c035f7834b564019656c3e3fcc3646292f727d`** — grep byte-for-byte
   TOÀN repo (`.sh`, `.py`, `.ipynb`, `docs/`): khớp nhau ở MỌI nơi xuất hiện (2 script
   `.py`/`.sh`, cả 4 notebook, nhiều file `docs/`). **Đúng, không có leftover/lệch.**
3. **Cảnh báo "bug scale COLMAP" (`points2D.xy` lệch resolution với camera
   intrinsics), `docs/PORTED_KNOWLEDGE.md` mục 1** — grep TOÀN `pipeline/*.py` cho
   `points2D`/`.xy`/`project_point`: **KHÔNG có kết quả nào.** Hiện KHÔNG có script nào
   trong repo NÀY thực sự đọc `points2D.xy` trực tiếp — khác repo tiền nhiệm, nơi
   antenna-focus dùng nó để dựng mask khung 3D cố định (kỹ thuật này đã bị loại khỏi
   Vòng 1 của repo này, xem `docs/00_MASTER_PLAN.md` mục 3.1, vì đo không cải thiện
   Score). Không phải tuyên bố SAI (vẫn đúng như kiến thức phòng ngừa cho tương lai nếu
   ai đó thêm lại antenna-focus hoặc code đọc points2D), nhưng hiện là kiến thức "ngủ
   đông" — không có code active nào trong repo này cần áp dụng nó ngay bây giờ. Ghi rõ
   ở đây (và ở `docs/PORTED_KNOWLEDGE.md` mục 6e) để pass sau không tưởng nhầm đây là
   bug đang active cần fix trong code hiện tại.

## Kết luận

**Tìm thấy + sửa 3 bug thật mới** (2 bug "stale state do re-run không dọn dẹp" —
`02_train_baseline.sh` checkpoint cũ, `05_generate_error_mask.py` mask cũ — cùng 1 bug
class "không dọn state cũ trước khi ghi state mới", nhưng ở 2 vị trí độc lập, không ai
audit theo góc "re-run" trước pass này; + 1 bug "dependency latest thật đã đổi API,
phá cú pháp lệnh `gdown` đang dùng ở CẢ 4 notebook" — nghiêm trọng nhất vì ảnh hưởng
NGAY BÂY GIỜ, không cần trigger đặc biệt gì, chỉ cần `pip install` mới nhất bình
thường). **Vì có bug thật, bộ đếm 3-lần-liên-tiếp-không-lỗi RESET VỀ 0/3** (không phải
1/3 hay giữ nguyên — pass #4 cũ không còn tính, phải bắt đầu lại đủ 3 pass sạch liên
tiếp KỂ TỪ pass tiếp theo).

## Giới hạn của pass này (ghi rõ, không giấu)

- Vẫn CHƯA chạy được train/render 3DGS thật trên GPU Kaggle — giới hạn không đổi qua
  mọi pass, chỉ giải quyết được khi có 1 lần chạy Kaggle thật (bug gdown tìm được ở
  pass này CHỈ lộ ra được nhờ test bằng gdown thật ngoài Kaggle — không dùng mock/giả
  lập, một dữ liệu tham khảo rằng 1 số bug KHÔNG mock được, phải test thành phần thật
  dù không có GPU).
- `00_make_holdout_split.py`/`01_run_colmap.py` vẫn chỉ được audit bằng đọc code (chưa
  chạy thật với `pycolmap` + dữ liệu COLMAP thật cục bộ — gợi ý này tồn tại từ
  `MILESTONE_06`/`MILESTONE_07`, vẫn chưa ai làm, rủi ro thấp vì port gần nguyên vẹn từ
  repo tiền nhiệm đã chạy thật 7/7 scene).
- Chưa kiểm tra gdown version PIN CỤ THỂ có nên thêm vào `pip install` hay không (đã cân
  nhắc: KHÔNG pin, vì bản mới nhất vẫn hoạt động đúng sau khi bỏ `--fuzzy`, pin version
  cũ tạo rủi ro MỚI — hết được support/security patch — không đáng đánh đổi cho 1 vấn đề
  đã tự hết bằng cách bỏ đúng 1 cờ lỗi thời). Rủi ro còn lại (chấp nhận được, ghi rõ):
  nếu gdown phiên bản TƯƠNG LAI (>6.1.0) tiếp tục đổi API theo hướng khác (vd đổi cách
  parse URL, đổi cách xử lý `--folder`), lệnh hiện tại (không `--fuzzy`) có thể lại gãy
  — không có cách nào phòng ngừa tuyệt đối dependency bên thứ 3 không pin version, chỉ
  có thể khuyến nghị re-verify gần ngày chạy Kaggle thật quan trọng (đã ghi vào bước
  tiếp theo).

## Bước tiếp theo

1. Chạy lại pass verify tiếp theo — bộ đếm sạch liên tiếp: **0/3** (reset vì pass này
   có bug thật). Cần lại đủ 3 pass sạch LIÊN TIẾP mới coi là xong.
2. **Trước khi chạy Kaggle thật lần đầu** (dù pass verify có đạt 3/3 hay chưa): chạy lại
   nhanh `!gdown --json "<GDRIVE_URL>"` (không `--fuzzy`, không tải, chỉ liệt kê) ở đúng
   Kaggle session để xác nhận KHÔNG có thay đổi API gdown nào mới kể từ lúc viết
   milestone này (đề phòng dependency drift tiếp tục xảy ra giữa lúc viết code và lúc
   chạy thật). Đây LÀ rủi ro thật đã xảy ra 1 lần (bug #3 mục C), không phải cẩn thận
   thừa.
3. Gợi ý cho pass sau (giảm trùng lặp phạm vi): các khu vực đã bị soi rất kỹ nhiều lần
   (schema `pipeline_train_flags.json`, va chạm thư mục iteration, đường dẫn dấu cách,
   `spatial_lr_scale`, tên file `eval_metrics`, `git submodule update`, checkpoint sort
   bug, symlink dataset, GPU-check, rollback/stop-at-round-N, adversarial input đã test
   ở Phần B pass này, 3 bug ở Phần C pass này) nên giảm ưu tiên — nhưng PHẢI re-verify
   nhanh cả 3 fix MỚI của pass này (đọc code trực tiếp, không chỉ tin milestone log này)
   theo đúng thông lệ đã thiết lập từ pass #2-4. Còn lại thực sự chưa test bằng dữ liệu
   thật cục bộ: `00_make_holdout_split.py`/`01_run_colmap.py` với `pycolmap` + fixture
   COLMAP thật (gợi ý tồn đọng từ nhiều pass trước, vẫn chưa ai làm).

## Lịch sử

### 2026-07-18 — Verification pass #5 (agent kiểm tra độc lập, khác pass #1-4)
- Đọc đầy đủ STEP 0 (`00_MASTER_PLAN.md`, `PORTED_KNOWLEDGE.md` toàn bộ kể cả mục
  6b/6c/6d, 8 milestone log 00-07, `git log --oneline`, `git show` đầy đủ diff pass #4).
- Phần A: test suite baseline PASS 100% (15/15 `.py` + 4/4 `.ipynb` + 21/21 package
  test + `bash -n` 2 file `.sh`).
- Phần B: adversarial testing — `manifest.json` 0 byte (JSONDecodeError + exit 1, chấp
  nhận được, không phải bug); `gdown --fuzzy --folder` (gdown 5.2.2) với URL bogus (exit
  0 + thư mục rỗng, nhưng `assert candidates` của notebook vẫn bắt được — không phải
  bug); 2 scene 1 lệnh, scene đầu crash (`02_train_baseline.sh`/`06_train_refine.sh` cả
  2 đều ABORT toàn bộ, xác nhận nhất quán có chủ đích — không phải bug); trace
  `get_scene()` exception qua notebook (đúng cạm bẫy đã biết mục 4, không phải phát
  hiện mới).
- Phần C: tìm + sửa 3 bug thật — (1) `02_train_baseline.sh` không chặn re-run đè
  checkpoint số cũ khác cấu hình (verify bằng mock thật, thêm guard `[BỎ QUA]` +
  `CLEAN_MODEL_DIR=1`); (2) `05_generate_error_mask.py` không dọn `error_masks/` cũ
  (verify bằng test logic thật, thêm bước dọn dẹp đầu `main()`); (3) `gdown --fuzzy` bị
  xoá khỏi gdown ≥6.0.0 — bug nghiêm trọng nhất, verify bằng cài + chạy gdown 6.1.0
  THẬT trong venv sạch, đối chiếu ngược 5.2.2, xác nhận fix bằng `--json` thật với
  GDRIVE_URL thật của dự án — sửa cả 4 notebook (bỏ `--fuzzy` khỏi mọi lệnh `gdown`).
- Phần D: fact-check 3 tuyên bố — công thức Score/PSNR_max (đúng), commit pin (đúng,
  không leftover), cảnh báo bug scale COLMAP mục 1 PORTED_KNOWLEDGE.md (đúng nhưng hiện
  "ngủ đông" — không có code nào trong repo này đọc `points2D.xy`, ghi rõ để tránh hiểu
  nhầm ở pass sau).
- Test lại toàn bộ SAU khi sửa: `tests/test_syntax_all.py` 15/15 + 4/4 PASS,
  `tests/test_07_package_submission.py` 21/21 PASS, `bash -n` sạch cả 2 file `.sh` —
  không có regression từ 3 fix.
- Dọn sạch toàn bộ scratch (`git status` sạch trong suốt quá trình, chỉ còn đúng 6 file
  đã sửa thật trong repo — không rác nào sót lại từ venv/mock test).
- Cập nhật `docs/PORTED_KNOWLEDGE.md` mục 6e (3 bug mới + 3 fact-check). Ghi milestone
  log này.
