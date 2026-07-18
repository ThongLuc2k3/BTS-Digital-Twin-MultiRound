# Milestone 11 — Kiểm tra độc lập lần 8 (verification pass #8 / N)

## Trạng thái hiện tại

**HOÀN TẤT pass #8 — TÌM THẤY + SỬA 2 VẤN ĐỀ THẬT** (1 lỗ hổng kỹ thuật thật —
`06_train_refine.sh` không tự chặn `train_mode="final"`, chỉ notebook mới chặn; 1 lỗi
tài liệu tự mâu thuẫn thật — `kaggle_round1_baseline.ipynb` Bước 6 tự đá nhau giữa 2
đoạn trong CÙNG 1 cell). Bộ đếm 3-lần-liên-tiếp-không-lỗi hiện tại: **0/3** (reset, vì
pass này có phát hiện thật cần sửa). Cần lại đủ 3 pass sạch liên tiếp kể từ pass tiếp
theo.

## Phạm vi đã làm

Đọc đầy đủ STEP 0: `docs/00_MASTER_PLAN.md`, `docs/PORTED_KNOWLEDGE.md` toàn bộ (kể cả
mục 6b–6g), cả 10 milestone log trước (`MILESTONE_00` → `MILESTONE_10`, đọc trọn vẹn
`MILESTONE_00`/`MILESTONE_07`/`MILESTONE_10`, các mục còn lại đã có tóm tắt đầy đủ +
chính xác trong `PORTED_KNOWLEDGE.md` nên đối chiếu chéo thay vì đọc lại nguyên văn),
`git log --oneline`, `git show 99ed04d` (đúng diff pass #7, đọc từng dòng đã đổi, không
chỉ tin commit message).

### Phần A — Baseline: test suite

- `tests/test_syntax_all.py`: 15/15 `.py` PASS + 4/4 `.ipynb` PASS.
- `tests/test_07_package_submission.py`: 21/21 PASS.
- `bash -n` cho `02_train_baseline.sh` + `06_train_refine.sh`: sạch.
- Chạy lại lần cuối SAU khi sửa (mục "Đã sửa" dưới) — vẫn PASS 100%, không regression.

### Phần B — Re-audit sâu fix `train_mode` của pass #7 (đọc CODE THẬT, không suy đoán)

Nhiệm vụ giao 3 câu hỏi cụ thể, trả lời từng câu bằng đọc code + thực thi thật:

1. **`06_train_refine.sh` có TỰ kiểm tra `train_mode` không, hay chỉ notebook mới
   kiểm tra?** Đọc toàn bộ `06_train_refine.sh` (TRƯỚC khi sửa) + grep `train_mode`
   trên toàn `pipeline/scripts/*.sh` + `*.py`: **chỉ có đúng 1 nơi ghi field này**
   (`02_train_baseline.sh`), và **hoàn toàn KHÔNG có nơi nào trong `06_train_refine.sh`
   đọc lại field này** — script chỉ đọc `antialiasing`/`sh_degree` từ
   `pipeline_train_flags.json`/`cfg_args`, không đụng tới `train_mode`. Chặn cứng CHỈ
   tồn tại ở cell Python "Bước 6" của `kaggle_round2_refine.ipynb`/
   `kaggle_round3_refine.ipynb` (chạy 1 LẦN lúc tải checkpoint từ Drive). **Xác nhận
   bằng thực thi thật** (mock `GS_REPO`/`train.py`, scene `finalscene` có
   `pipeline_train_flags.json` ghi `"train_mode": "final"`): gọi thẳng
   `bash 06_train_refine.sh finalscene` (bỏ qua hoàn toàn notebook, đúng kịch bản
   "copy-paste lệnh bash vào 1 cell/terminal riêng, không theo đúng thứ tự Bước 6") —
   **script chạy xong bình thường, exit code 0, KHÔNG có bất kỳ cảnh báo/lỗi nào** về
   `train_mode`. Đây là 1 lỗ hổng thật: nếu 1 checkpoint `MODE="final"` đã có sẵn trong
   `$MODEL_DIR` (vd copy tay, hoặc chạy lại cell "Bước 6" của notebook nhưng bỏ qua vì
   tưởng đã tải rồi, hoặc gọi script này ở 1 phiên Kaggle/terminal riêng không qua
   notebook) thì kịch bản rò rỉ dữ liệu mô tả ở `docs/PORTED_KNOWLEDGE.md` mục 6g **vẫn
   xảy ra ÂM THẦM**, không có gì chặn. Đáng chú ý: comment trong `02_train_baseline.sh`
   (do CHÍNH pass #7 viết, dòng "TRAIN_MODE:...") đã tuyên bố "Vòng 2+
   (`06_train_refine.sh`/`kaggle_round2_refine.ipynb`) tự phát hiện + CHẶN CỨNG" — tuyên
   bố này SAI ở phần `06_train_refine.sh` (chỉ đúng cho notebook) tại thời điểm viết,
   pass #7 đã overstate những gì thực sự implement. **Đã sửa** (xem mục "Đã sửa" dưới)
   — giờ tuyên bố đó đã ĐÚNG THẬT.
2. **Field `train_mode` có được bảo toàn đúng qua NHIỀU vòng refine không (`06_train_
   refine.sh`'s JSON-merge logic khi ghi `refine_history`)?** Đọc trực tiếp đoạn Python
   heredoc cuối `06_train_refine.sh` (dòng ghi lại `pipeline_train_flags.json`): code
   `flags = json.loads(flags_path.read_text())` nạp NGUYÊN VẸN toàn bộ dict cũ (bao gồm
   `train_mode` nếu có), sau đó chỉ gọi `flags.setdefault(...)` cho 4 key CŨ
   (`antialiasing`/`depth_prior`/`exposure_comp`/`antenna_focus`) — **KHÔNG có
   `setdefault`/ghi đè nào cho `train_mode`**, nên giá trị cũ (nếu có) giữ nguyên
   100%, giá trị vắng mặt (nếu checkpoint gốc thiếu field) vẫn vắng mặt sau refine (đúng
   ý — không tự "bịa" ra giá trị). **Verify bằng thực thi thật** (không chỉ đọc code):
   train Vòng 1 mock với `TRAIN_MODE=holdout` → `pipeline_train_flags.json` có
   `"train_mode": "holdout"` → chạy `06_train_refine.sh` (Vòng 2 refine) → xác nhận
   `pipeline_train_flags.json` SAU refine vẫn `"train_mode": "holdout"` (kèm
   `refine_history` mới) — checkpoint Vòng 2 (output của lần refine này) sẵn sàng cho
   Vòng 3 tự kiểm tra đúng. **Xác nhận: propagation hoạt động đúng như pass #7 tuyên
   bố, không có gì bị rơi rớt.**
3. **`kaggle_submission.ipynb` có bị "over-correction" (chặn nhầm checkpoint
   `MODE="final"`) không?** Grep `train_mode` trên cả 4 notebook: **chỉ xuất hiện ở
   `kaggle_round1_baseline.ipynb` (cell ghi field) + `kaggle_round2_refine.ipynb`/
   `kaggle_round3_refine.ipynb` (cell 16, đọc + chặn)** — `kaggle_submission.ipynb`
   **hoàn toàn KHÔNG có bất kỳ tham chiếu `train_mode` nào**, đúng như mong đợi (notebook
   này nhận BẤT KỲ checkpoint nào — vòng 1/2/3, mode nào cũng được — người dùng tự quyết
   định qua field `"round"` trong `CHECKPOINT_LINKS`, chỉ để ghi chú không ảnh hưởng
   logic). **Xác nhận: không có over-correction.**

### Phần C — Đọc lại TOÀN BỘ tài liệu (12 file `docs/`) 1 lượt liền mạch, tìm chỗ 2 pass
khác nhau "đá nhau"

- **Đối chiếu 3 fix cũ (symlink dataset — pass #3, GPU-check — pass #3, pip-install
  pin — pass #6) còn nguyên vẹn + nhất quán ở cả 4 notebook SAU khi pass #7 sửa 3
  notebook (round1/round2/round3)**: dump nguyên văn cell liên quan bằng script (không
  đọc mắt) — GPU-check (`raise SystemExit` nếu `not torch.cuda.is_available()`) có mặt
  ở cả 4 file; `!pip install -q "pycolmap>=3.10" ... "gdown>=6,<7"` giống hệt byte-for-
  byte ở cả 4 file; symlink dataset dùng đúng pattern
  `is_symlink()->unlink() / exists()->rmtree()` rồi luôn `os.symlink()` ở cả 4 file.
  **Cả 3 fix cũ vẫn nguyên vẹn, KHÔNG bị pass #7 vô tình đụng/xoá.**
- **Đọc lại nguyên văn NGUYÊN CELL 21 (`Bước 6 — Lấy checkpoint Vòng 1`) của
  `kaggle_round1_baseline.ipynb`** — đây chính là cell pass #7 đã sửa (đối chiếu
  `git show 99ed04d` xác nhận đúng 2 đoạn trong cell này bị đổi: bullet mô tả `MODE` ở
  cell 13, và câu chủ đề đầu cell 21) — **PHÁT HIỆN BUG THẬT**: câu chủ đề đầu cell 21
  (pass #7 đã sửa) và đoạn "Quy trình đầy đủ cho MỖI scene" (pass #7 KHÔNG đụng tới,
  nằm NGAY BÊN DƯỚI trong CÙNG 1 cell) **tự mâu thuẫn nhau**. Xem chi tiết mục "Bug tìm
  ra + đã sửa" dưới.
- **Grep toàn bộ 4 notebook** cho pattern `final.*input`/`input.*final` (không phân biệt
  hoa thường) để tìm instance THỨ 3 của claim sai — chỉ còn đúng 2 kết quả: dòng comment
  đúng ở `kaggle_round1_baseline.ipynb` (giải thích mục đích field `TRAIN_MODE`) và
  chính đoạn văn đã sửa ở cell 21 (mục "Đã sửa" dưới). **Không còn instance thứ 3 nào
  sai sót.**
- **Grep `docs/00_MASTER_PLAN.md`/`docs/PORTED_KNOWLEDGE.md`** cho cùng pattern: mọi kết
  quả đều là MÔ TẢ LỊCH SỬ bug đã tìm+sửa ở pass #7 (đúng ngữ cảnh "trước đây sai thế
  nào"), không có hướng dẫn LIVE nào còn sai. `00_MASTER_PLAN.md` mục 3.2 bước 1
  (`**BẮT BUỘC là checkpoint Vòng 1 train ở `MODE="holdout"`, KHÔNG phải `"final"`**`)
  — đúng, khớp code hiện tại. **Không tìm thêm vấn đề.**

### Phần D — Đọc narrative toàn bộ 4 notebook, cell-by-cell, như 1 user thật

Đọc TOÀN BỘ nội dung (không skim) cả 22 cell `kaggle_round1_baseline.ipynb`, 29 cell
`kaggle_round2_refine.ipynb`, 29 cell `kaggle_round3_refine.ipynb` (đối chiếu byte-for-
byte các cell trùng với round2 — chỉ khác đúng 2 chỗ text "Vòng 1"->"Vòng 2" hợp lý,
không phải bug), 21 cell `kaggle_submission.ipynb` — đóng vai người dùng thật đọc theo
đúng thứ tự, làm theo hướng dẫn markdown literal.

- **Tìm thấy đúng 1 chỗ đọc-vào-sẽ-nhầm** (mô tả ở mục "Bug tìm ra" dưới) — cell 21 của
  `kaggle_round1_baseline.ipynb`.
- Còn lại: không tìm thêm chỗ mâu thuẫn/mơ hồ nào khác. Các cảnh báo BANNER (cell 14,
  in đậm SCENE/MODE/ANTIALIASING trước khi chạy các cell tốn thời gian), thứ tự
  Bước 1(GPU check)->Bước 2(clone+build, tốn phút)->Bước 4(tải dataset, tốn phút) đúng
  logic chặn sớm, các assert rõ ràng (`CHECKPOINT_DRIVE_LINK`/`GDRIVE_URL` rỗng) đều
  nhất quán và dễ hiểu với người không chuyên.

## Bug tìm ra + đã sửa

### Bug #1 (lỗ hổng kỹ thuật) — `06_train_refine.sh` không tự chặn checkpoint `train_mode="final"`, chỉ notebook mới chặn

**Mô tả:** Guard chặn cứng `train_mode="final"` (pass #7 thêm) CHỈ tồn tại ở cell Python
"Bước 6" của 2 notebook refine — không có ở bất kỳ đâu trong
`pipeline/scripts/06_train_refine.sh`, dù script này đọc `pipeline_train_flags.json`
ngay trong chính nó (để lấy `antialiasing`) và hoàn toàn có đủ thông tin để tự kiểm tra
thêm `train_mode`. Verify bằng thực thi thật (mock `GS_REPO`): gọi thẳng
`06_train_refine.sh` trên 1 checkpoint có `"train_mode": "final"` (bỏ qua notebook hoàn
toàn) — script chạy xong bình thường, exit 0, không cảnh báo gì. Kịch bản THẬT có thể
xảy ra: người dùng chạy lại 1 cell riêng lẻ ngoài thứ tự (Kaggle cho phép chạy từng
cell tuỳ ý, không bắt buộc "Run All" từ đầu), dùng terminal Kaggle gọi thẳng script,
hoặc copy dòng lệnh bash ra 1 cell khác để debug/thử nghiệm mà quên chạy lại cell "Bước
6" (guard) trước đó trong cùng phiên.

**Đã sửa:** `pipeline/scripts/06_train_refine.sh` — thêm đọc `train_mode` vào cùng đoạn
Python heredoc đã đọc `antialiasing`/`sh_degree` (in ra biến bash
`TRAIN_MODE_DETECTED`), rồi lặp lại ĐÚNG 3 nhánh xử lý y hệt cell Python của notebook
ngay sau khi `eval "$CFG_INFO"`:
- `train_mode == "final"` → in `[LỖI]` rõ ràng (giải thích đúng nguyên nhân + hướng khắc
  phục) + `exit 1`, TRỪ KHI người dùng cố ý set `ALLOW_FINAL_TRAIN_MODE=1` (escape hatch
  theo đúng pattern đã có sẵn trong repo, vd `CLEAN_MODEL_DIR=1` ở `02_train_baseline.sh`).
- `train_mode` vắng mặt (`None`/key không tồn tại) → in `[CẢNH BÁO]`, KHÔNG chặn (tương
  thích ngược với checkpoint train trước khi field này tồn tại).
- `train_mode == "holdout"` (hoặc bất kỳ giá trị nào khác `"final"`) → im lặng, tiếp tục
  bình thường.

**Verify sau sửa** (thực thi thật, mock `GS_REPO`, 4 kịch bản):
1. `train_mode="final"`, không set `ALLOW_FINAL_TRAIN_MODE` → `[LỖI]` + exit 1 (đúng).
2. `train_mode="final"` + `ALLOW_FINAL_TRAIN_MODE=1` → chạy xong bình thường, exit 0
   (đúng, escape hatch hoạt động).
3. `train_mode` vắng mặt hoàn toàn (mô phỏng checkpoint train bởi bản `02_train_
   baseline.sh` TRƯỚC pass #7, dùng schema 4-key cũ không có `train_mode`) → in
   `[CẢNH BÁO]`, chạy xong bình thường, exit 0 (đúng, không chặn checkpoint hợp lệ cũ).
4. `train_mode="holdout"` (trường hợp bình thường) → không có dòng `[LỖI]`/`[CẢNH BÁO]`
   nào liên quan `train_mode`, chạy xong sạch, exit 0 (đúng).

`bash -n`/`py_compile`/`nbformat.validate()` sạch sau sửa; `tests/test_syntax_all.py`
(15/15+4/4) + `tests/test_07_package_submission.py` (21/21) PASS 100%, không regression
(file `.py`/`.ipynb` không đổi ở phần này — chỉ `.sh` đổi, không nằm trong phạm vi 2 bộ
test đó nhưng chạy lại để chắc chắn không có hiệu ứng phụ ngoài ý muốn).

### Bug #2 (tài liệu tự mâu thuẫn) — `kaggle_round1_baseline.ipynb` Bước 6 (cell 21) đá nhau giữa câu chủ đề (đã sửa ở pass #7) và đoạn "Quy trình đầy đủ" (pass #7 KHÔNG đụng tới)

**Mô tả:** Pass #7 sửa câu chủ đề đầu cell 21 từ *"Chỉ làm bước này khi `MODE = "final"`."*
(bản CŨ, SAI) thành *"Chỉ làm bước này khi `MODE = "final"` VÀ bạn KHÔNG định chạy thêm
Vòng 2+ cho scene này."* — rồi giải thích ĐÚNG ngay sau đó: checkpoint `MODE="holdout"`
mới là thứ PHẢI tải lên Drive nếu định chạy Vòng 2+. Đến đây văn bản đã đúng kỹ thuật.

NHƯNG đoạn **"Quy trình đầy đủ cho MỖI scene"** ngay bên dưới, TRONG CÙNG CELL đó
(`git show 99ed04d` xác nhận pass #7 KHÔNG chạm vào 4 dòng này), vẫn giữ nguyên văn bản
CŨ, SAI:

> 2. Chạy 1 version cuối với `MODE="final"` — đây là checkpoint Vòng 1 chính thức, làm
>    **input cho Vòng 2+** (`kaggle_round2_refine.ipynb`) hoặc dùng để nộp bài trực tiếp
>    nếu không chạy thêm vòng nào.

Đây **chính xác là hướng dẫn nguy hiểm mà toàn bộ fix của pass #7 nhằm loại bỏ** — nếu 1
người dùng thật đọc "Quy trình đầy đủ" (đọc như 1 tóm tắt quy trình dứt khoát, dễ đọc
nhảy tới đoạn này bỏ qua phần giải thích dài phía trên) và làm ĐÚNG theo đó (chạy
`MODE="final"` rồi tải checkpoint đó lên Drive, dán vào `kaggle_round2_refine.ipynb`),
Bug #1 ở trên (trước khi sửa) + guard ở cell 16 của notebook Vòng 2+ vẫn sẽ CHẶN được
họ (raise `SystemExit` rõ ràng) — nên hậu quả kỹ thuật thực tế được giảm nhẹ nhờ lớp
chặn cứng đã có, nhưng đây vẫn là 1 lỗi tài liệu THẬT (tự mâu thuẫn ngay trong 1 cell,
đúng loại "gây nhầm lẫn cho người dùng thật" mà nhiệm vụ yêu cầu tìm) — người dùng sẽ bị
`SystemExit` chặn lại giữa chừng (sau khi đã tốn thời gian/quota train `MODE="final"`
30000 iteration — ĐẮT hơn nhiều `MODE="holdout"` 15000 iteration) rồi mới biết mình làm
sai, thay vì được hướng dẫn đúng ngay từ đầu.

**Đã sửa:** Viết lại cell 21 (`kaggle_round1_baseline.ipynb`) — thay câu chủ đề bằng 2
gạch đầu dòng tách rõ mục đích của TỪNG `MODE` (both đều "làm bước này", chỉ khác lý
do/đích đến), và sửa lại bước 2 của "Quy trình đầy đủ" cho khớp: `MODE="final"` —
checkpoint chất lượng cao nhất, **KHÔNG dùng làm input Vòng 2+**, chỉ dùng nộp bài trực
tiếp. Đồng thời thêm 1 dòng ở bước 1 nhắc tải checkpoint `holdout` lên Drive nếu định
chạy Vòng 2+ (để 2 bước trong "Quy trình đầy đủ" tự đủ nghĩa, không cần nhảy ngược lên
đọc lại đoạn giải thích phía trên). Grep lại toàn bộ 4 notebook + `docs/*.md` sau khi
sửa — xác nhận không còn instance thứ 3 nào của claim sai này.

**Verify sau sửa**: `nbformat.validate()` sạch; `git diff` xác nhận CHỈ đúng phần cell
21 bị đổi (19 dòng thêm/12 dòng xoá), không đụng cell nào khác, không đổi `metadata`/
`outputs`/`execution_count` của bất kỳ cell nào (dump lại toàn bộ 22 cell bằng script,
so với bản gốc trước sửa — chỉ cell 21 khác). Test suite PASS 100% sau sửa.

## Giới hạn của pass này (ghi rõ, không giấu)

- Vẫn CHƯA chạy được train/render 3DGS thật trên GPU Kaggle — giới hạn không đổi qua
  mọi pass.
- `00_make_holdout_split.py`/`01_run_colmap.py` vẫn chỉ audit bằng đọc code (không đọc
  lại toàn bộ 2 file lần này — đã đọc trọn vẹn ở pass #4/#7, không có gì đổi ở 2 file
  này từ đó) — chưa chạy full script thật với `pycolmap` + dữ liệu COLMAP thật (gợi ý
  tồn đọng từ nhiều pass, vẫn chưa ai làm — rủi ro thấp, port gần nguyên vẹn từ repo
  tiền nhiệm đã chạy thật 7/7 scene).
- Bug #2 (tài liệu) có hậu quả kỹ thuật đã được giảm nhẹ SẴN bởi lớp chặn cứng ở cell 16
  (dù đúng, người dùng theo hướng dẫn sai sẽ bị `SystemExit` chặn lại, không mất dữ liệu/
  không rò rỉ Score) — nhưng vẫn tốn thời gian/quota Kaggle oan uổng (train 30000
  iteration `MODE="final"` rồi mới biết dùng sai) nếu không sửa. Đã sửa triệt để ở pass
  này, không còn để lại cho pass sau.
- Chưa thử nghiệm kịch bản "user chạy lại `!bash .../06_train_refine.sh {SCENE}` (cell
  24 của `kaggle_round2_refine.ipynb`) TRONG notebook thật (Kaggle), KHÔNG chạy lại cell
  16 trước đó nhưng biến `SCENE` vẫn còn trong bộ nhớ kernel từ lần chạy trước" — đây là
  đúng kịch bản thực tế nhất mà Bug #1 nhắm tới (không chỉ giả định lý thuyết), nhưng
  chỉ verify được bằng mock cục bộ (gọi thẳng `06_train_refine.sh` từ shell, tương đương
  hành vi) — chưa thử trên Kaggle Jupyter kernel thật (cần GPU/phiên Kaggle thật).

## Bước tiếp theo

1. Chạy **pass #9** (agent verify độc lập khác) — bộ đếm sạch liên tiếp: 0/3.
2. Gợi ý cho pass #9: re-verify kỹ 2 fix của pass này (đọc code trực tiếp, không chỉ tin
   milestone log) — đặc biệt `ALLOW_FINAL_TRAIN_MODE=1` escape hatch mới thêm ở
   `06_train_refine.sh` (đúng behavior, không bị dùng nhầm ở đâu khác trong repo — hiện
   KHÔNG có notebook nào set biến này, đúng ý đồ, chỉ dành cho người dùng cố ý override
   tay). Các khu vực đã soi rất kỹ nhiều lần (schema `pipeline_train_flags.json` gốc,
   va chạm thư mục iteration, đường dẫn dấu cách, `spatial_lr_scale`, tên file
   `eval_metrics`, `git submodule update`, checkpoint sort bug, symlink dataset,
   GPU-check, rollback/stop-at-round-N, adversarial input, re-run stale checkpoint,
   `gdown --fuzzy`, thiếu `opencv-python`, mâu thuẫn MODE Vòng 1/Vòng 2+) nên giảm ưu
   tiên. Còn thực sự chưa test bằng dữ liệu thật cục bộ: `00_make_holdout_split.py`/
   `01_run_colmap.py` full script với `pycolmap` + fixture COLMAP thật (gợi ý tồn đọng
   nhiều pass, vẫn chưa ai làm).
3. Trước khi chạy Kaggle thật: đọc lại cell 21 MỚI của `kaggle_round1_baseline.ipynb`
   (đã viết lại rõ ràng hơn ở pass này) trước khi quyết định tải checkpoint nào lên
   Drive cho từng scene.

## Lịch sử

### 2026-07-18 — Verification pass #8 (agent kiểm tra độc lập, khác pass #1-7)
- Đọc đầy đủ STEP 0 (`00_MASTER_PLAN.md`, `PORTED_KNOWLEDGE.md` toàn bộ kể cả mục
  6b-6g, 11 milestone log 00-10 — đọc trọn vẹn 00/07/10, đối chiếu chéo phần còn lại,
  `git log --oneline`, `git show 99ed04d` đọc từng dòng diff thật).
- Phần A: test suite baseline PASS 100%.
- Phần B: re-audit sâu 3 câu hỏi cụ thể về fix `train_mode` của pass #7 — xác nhận
  propagation qua `refine_history` ĐÚNG (test thật), xác nhận `kaggle_submission.ipynb`
  KHÔNG bị over-correction (đúng, không có guard nào ở đó) — **tìm ra Bug #1**:
  `06_train_refine.sh` không tự chặn `train_mode="final"`, chỉ notebook mới chặn (verify
  bằng thực thi thật: gọi thẳng script trên checkpoint `final`, chạy xong exit 0 không
  cảnh báo gì).
- Phần C: đọc lại toàn bộ 12 file `docs/` 1 lượt liền mạch — xác nhận 3 fix cũ (symlink,
  GPU-check, pip-install pin) còn nguyên vẹn ở cả 4 notebook sau khi pass #7 sửa 3
  notebook — **tìm ra Bug #2**: cell 21 (`Bước 6`) của `kaggle_round1_baseline.ipynb` tự
  mâu thuẫn giữa câu chủ đề (pass #7 đã sửa đúng) và đoạn "Quy trình đầy đủ" ngay bên
  dưới (pass #7 không chạm tới, vẫn giữ hướng dẫn SAI dùng `MODE="final"` làm input
  Vòng 2+).
- Phần D: đọc narrative toàn bộ 4 notebook (101 cell) như 1 user thật — chỉ tìm thêm
  đúng 1 chỗ (Bug #2), không có gì khác.
- Sửa: `pipeline/scripts/06_train_refine.sh` (thêm guard `train_mode` + escape hatch
  `ALLOW_FINAL_TRAIN_MODE=1`), `pipeline/kaggle_round1_baseline.ipynb` (viết lại cell 21
  cho nhất quán, không còn tự mâu thuẫn).
- Verify sau sửa: 4 kịch bản thực thi thật cho `06_train_refine.sh`
  (final-chặn/final+override/vắng mặt-cảnh báo/holdout-im lặng) đều đúng thiết kế; diff
  notebook xác nhận chỉ đổi đúng cell 21, không đụng cell/metadata nào khác; test suite
  (`test_syntax_all.py` 15/15+4/4, `test_07_package_submission.py` 21/21) PASS 100% sau
  sửa, không regression.
- `git status` sạch trong suốt quá trình (dọn scratch riêng sau khi test xong), chỉ còn
  đúng các file thật đã sửa trong repo.
