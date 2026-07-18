# Vòng lặp Verification — quy trình + prompt mẫu để tái sử dụng

> File này ghi lại CHÍNH XÁC quy trình đang chạy để kiểm tra toàn bộ repo tới khi đạt
> chất lượng tốt nhất, theo yêu cầu của user: *"tạo AI agent mới mỗi khi sắp hết công
> việc để kiểm tra và chỉnh sửa, không dừng lại cho đến khi có 3 AI agent liên tục xác
> nhận không có lỗi gì cả sau nhiều lần mô phỏng chạy"*. Dùng file này để: (1) hiểu tại
> sao có nhiều `docs/MILESTONE_XX_verification_passN.md`, (2) tiếp tục đúng quy trình
> nếu bị ngắt giữa chừng, (3) tái sử dụng prompt mẫu cho các đợt kiểm tra khác sau này
> (kể cả ở repo khác) nếu user muốn.

## Quy tắc chung

- **Điều kiện dừng**: cần **3 lần verification pass LIÊN TIẾP** đều báo "PASS" (không
  tìm thấy/sửa bất kỳ vấn đề gì). Nếu 1 pass bất kỳ tìm thấy VÀ SỬA dù chỉ 1 vấn đề nhỏ
  → tính là FAIL → bộ đếm reset về 0 → phải chạy lại pass tiếp theo từ đầu chuỗi 3.
- Mỗi pass là **1 agent MỚI HOÀN TOÀN** (không phải agent trước tiếp tục) — đảm bảo góc
  nhìn độc lập, không bị "quen mắt" bỏ sót thứ agent trước đã quen thuộc.
- Mỗi pass PHẢI đọc đủ: `docs/00_MASTER_PLAN.md`, TOÀN BỘ `docs/PORTED_KNOWLEDGE.md`
  (kể cả các mục phụ đã thêm bởi pass trước), TOÀN BỘ `docs/MILESTONE_*.md` theo đúng
  thứ tự số, và `git show` commit của pass ngay trước đó — không được bỏ qua bước đọc
  này dù có vẻ tốn thời gian, vì đây là cách duy nhất pass sau biết pass trước đã kiểm
  tra góc nào rồi để tự tìm góc KHÁC (đã chứng minh hiệu quả: pass càng về sau càng
  phải "đi lệch hướng" mới tìm ra vấn đề mới).
- Mỗi pass PHẢI tự chạy mô phỏng thật (mock, vì không có GPU cục bộ) — không chỉ đọc
  code bằng mắt. Toàn bộ bug thật tìm được trong quá trình này đều tìm ra bằng cách
  THỰC THI THẬT (mock train.py, patch thật lên clone `gaussian-splatting` thật, cài
  thật gói pip mới nhất để verify hành vi thật...), KHÔNG phải suy đoán từ đọc code.
- Mỗi pass được phép + được khuyến khích COMMIT trực tiếp (không cần hỏi lại) — đây là
  quy trình đã được user cấp phép tự chủ hoàn toàn cho repo này. Agent KHÔNG được PUSH
  — điều phối viên (Claude phiên chính) review rồi mới push, làm điểm kiểm soát cuối.
- Đặt tên file milestone theo thứ tự số tăng dần: `docs/MILESTONE_<NN>_verification_
  pass<N>.md`. Nếu tìm thấy bug mới thật sự (không phải chỉ re-verify lại cái cũ), thêm
  1 mục con MỚI vào cuối `docs/PORTED_KNOWLEDGE.md` (không sửa/xoá mục cũ — chỉ nối
  thêm, theo đúng convention `6b`, `6c`, `6d`, ... đã dùng).

## Lịch sử bộ đếm (tính tới lúc viết file này)

| Pass | Kết quả | Tìm thấy gì |
|---|---|---|
| #1 | FAIL | 4 vấn đề (2 chức năng, 2 tài liệu tàn dư) |
| #2 | FAIL | 1 bug (bug class sort/awk chưa backport) |
| #3 | FAIL | 2 bug (symlink dataset cũ, GPU-check không chặn) |
| #4 | **PASS** | (không tìm thấy gì — 1/3) |
| #5 | FAIL | 3 bug (re-run đè checkpoint, mask cũ không dọn, `gdown --fuzzy` bị xoá) |
| #6 | FAIL | thiếu `opencv-python`, pin thêm `gdown`/`pycolmap` |
| #7 | FAIL | rò rỉ dữ liệu nếu dùng nhầm checkpoint `MODE=final` làm input refine |
| #8 | FAIL | guard `train_mode` bị bypass được ở cấp shell script, 1 cell tự mâu thuẫn |
| #9+ | *(xem `docs/MILESTONE_*.md` mới nhất để biết tiếp)* | |

Bộ đếm reset về 0 sau MỖI lần FAIL — tính tới pass gần nhất đã chạy, xem file
milestone số cao nhất trong `docs/` để biết trạng thái THẬT hiện tại (bảng trên có thể
đã lạc hậu nếu có pass mới chạy sau khi viết file này — không sửa bảng cũ, chỉ đọc
tiếp `docs/MILESTONE_*.md` mới hơn để cập nhật).

## Prompt mẫu cho 1 verification pass (điều phối viên copy + điền số/lịch sử mới nhất)

```
You are doing an independent verification pass on a repository at:
  <đường dẫn repo>
<1-2 câu mô tả dự án>. The user requires 3 CONSECUTIVE verification passes with zero
findings. History: <liệt kê PASS/FAIL từng pass trước + tóm tắt 1 dòng mỗi pass tìm
được gì>. Counter is currently <N>/3. You are pass #<N+1>.

STEP 0 — REQUIRED READING (in full):
1. docs/00_MASTER_PLAN.md
2. docs/PORTED_KNOWLEDGE.md in full, all sections through §<chữ cái mới nhất>.
3. Every docs/MILESTONE_*.md in numeric order (00 through <số mới nhất>).
4. git log --oneline and git show on the immediately-prior pass's commit.

YOUR JOB:
A. Baseline: full existing test suite (tests/, py_compile, bash -n, nbformat.validate())
   — must be green.
B. Verify the PRIOR pass's own fixes are correct and complete, from a angle it didn't
   cover itself (don't just trust its commit message — re-derive/re-execute).
C. New angles not yet covered by ANY prior pass — <điều phối viên tự nghĩ 3-5 hướng cụ
   thể dựa trên những gì CHƯA ai làm, xem bảng lịch sử + milestone log gần nhất để biết
   chưa ai chạm vào đâu>.
D. Fresh end-to-end mock simulation, from scratch (không dùng lại thư mục scratch của
   pass trước) — biến tấu 1 khía cạnh CHƯA ai test (multi-scene thật, kịch bản lỗi cụ
   thể, đọc notebook như người dùng thật lần đầu...).
E. Fix what you can safely fix directly. Leave design-judgment items unfixed, clearly
   reported — do not manufacture findings just to have something to report.

When done: create docs/MILESTONE_<NN>_verification_pass<N>.md (Vietnamese, matching
format of docs/MILESTONE_00_setup.md). Append to docs/PORTED_KNOWLEDGE.md only if
genuinely new (§<chữ cái tiếp theo> if needed — NEVER edit/remove earlier sections).
Commit (Vietnamese message, already authorized for this repo's verification workflow
— no need to ask first). Do NOT push (coordinator pushes after review).

End with an explicit PASS/FAIL verdict: FAIL if you found/fixed/flagged anything at
all, however minor. Only declare PASS if, after genuinely trying hard across the new
angles above, you find nothing.
```

## Ghi chú vận hành cho điều phối viên

- Sau MỖI pass (dù PASS hay FAIL): `git log --oneline -3` + `git status --short` để
  xác nhận commit đã có, rồi `git push origin main` — review nhanh nội dung agent báo
  cáo trước khi push (đối chiếu tinh thần "SECURITY WARNING" nếu có — phần lớn là báo
  động giả do agent thực hiện đúng việc được giao trong 1 repo đã được cấp phép tự chủ,
  nhưng vẫn nên liếc qua diff/commit message 1 lượt trước khi đẩy lên remote).
- Nếu 1 agent bị NGẮT GIỮA CHỪNG do hết session quota (không phải lỗi kỹ thuật) — kiểm
  tra `git status --short` xem để lại gì (đã commit hay chỉ untracked/modified), tự
  hoàn thiện phần còn dang dở (đọc kỹ log cuối cùng của agent để biết nó đang làm tới
  đâu), rồi mới tính pass đó là PASS/FAIL dựa trên tổng thể đã làm — không tự động tính
  là FAIL chỉ vì bị ngắt, cũng không bỏ qua nếu phần dang dở đó thật sự có vấn đề.
- Không cần chờ user xác nhận giữa các pass — quy trình này đã được cấp phép chạy liên
  tục tới khi đạt 3 lần sạch liên tiếp.
