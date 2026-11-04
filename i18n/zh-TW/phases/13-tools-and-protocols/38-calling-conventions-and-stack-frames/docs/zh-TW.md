# 呼叫慣例與 Stack Frame：System V、Windows x64 與 AArch64 (Calling Conventions and Stack Frames: System V, Windows x64, and AArch64)

> 反組譯顯示函式讀取了哪些暫存器。只有 ABI 才說明這些暫存器的意義，而 ABI 是由 object header 選擇的，所以這堂課先讀 header，然後對照把每個暫存器放在那裡的規則來檢查它。

**Type:** Build
**Languages:** Python, C, x86-64 and AArch64 assembly
**Prerequisites:** Phase 13 · 32 (ELF from scratch), Phase 13 · 33 (disassembly and control flow)
**Time:** ~140 分鐘

## 學習目標 (Learning Objectives)

- 從 object header 挑出呼叫慣例：ELF x86-64 是 System V psABI，COFF AMD64 是 Microsoft x64，ELF AArch64 是 AAPCS64，Mach-O arm64 是 Apple 的變體。
- 對每種 ABI，逐位元組解碼一個函式的 prologue，並把每個被儲存的暫存器、return address 與 frame pointer，放在它們距離進入點 stack pointer 的 offset。
- 從函式在寫入之前先讀取的暫存器，推論參數暫存器；從進入點 stack pointer 之上的 load，推論 stack 引數的 slot。
- 區分 System V 各自獨立的暫存器序列、Microsoft x64 以位置決定的 slot，以及 AAPCS64 的 NGRN 與 NSRN 計數器。
- 對照原始碼簽章、規格中的規則，以及編譯器自己在 `.pdata`、`.xdata` 與 `.eh_frame` 中的 unwind 資料，檢查每個推論。
- 比較 Ghidra、IDA、Binary Ninja、radare2、rizin 與 angr 如何命名並還原相同的事實，並記錄它們的輸出在哪裡與原始碼不符。

## 問題 (The Problem)

你為 Linux x86-64 以 `-O2` 編譯 `convention.c`，並向 radare2 6.2.4 詢問 `eight_args` 的引數。它回答七個：`rdi`、`rsi`、`rdx`、`r8`、`r9`、`rsp+0x8` 和 `rsp+0x10`。C 原始碼有八個參數。位元組只在 `leaq (%rdx,%rcx), %rax` 中讀取了一次 `%rcx`，而工具的變數還原沒有把只出現在位址裡面的暫存器算進去。

接著你在 rizin 0.9.1 中開啟同一個檔案的 Windows build。它回報呼叫慣例是 `amd64`，並找到 `r8`、`rcx`、`rdx` 以及三個 stack slot。這個 object 是 COFF，所以呼叫慣例是 Microsoft x64，第四個暫存器引數是 `r9`，並且有四個 stack slot。兩個工具在同一份原始碼中漏掉不同的暫存器。

清單是指令的序列。呼叫慣例是存在於清單之外的合約：哪些暫存器攜帶引數、callee 必須保留哪些、return address 與 stack 引數在哪裡，以及 caller 必須在 stack pointer 之上留下什麼。有四份文件定義了這裡重要的合約。System V AMD64 psABI（Version 1.0，build 日期 2025 年 3 月 12 日）的 3.2.1 到 3.2.3 節。Microsoft x64 呼叫慣例、stack 使用，以及 prolog 與 epilog 頁面（頁面日期從 2018-12-17 到 2025-03-19）。Arm AAPCS64 release 2025Q4（2026 年 1 月 23 日）的 6.1 到 6.9 節。Apple 的 arm64 平台筆記。這堂課閱讀這些規則、把它們套用到以三種方式編譯的一個 C 檔案上，並建構分析器，把清單變成經過檢查的參數位置、被儲存的暫存器與 frame offset 清單。

```figure
re-38-three-frames
```

## 一個 header 選出規則書 (One Header Picks the Rulebook)

`convention.c` 有十二個函式：`leaf_add`、`eight_args`、`mixed`、`by_value`、`big_ret`、`sum_varargs`、`keeps`，以及五個呼叫相同形狀的 `extern` 函式的 caller。Struct 使用 `long long`，因為 `long` 在 MSVC 目標上是 4 位元組，在其他目標上是 8 位元組，而大小改變的 struct 會改變呼叫慣例。Apple clang 17.0.0 在這台機器上編譯它七次：`-target x86_64-unknown-linux-gnu`、`x86_64-pc-windows-msvc` 和 `aarch64-unknown-linux-gnu` 各在 `-O0` 與 `-O2`，以及 `arm64-apple-macos` 的 `-O2`。沒有安裝 Linux 或 Windows 的 header，所以檔案自己宣告需要的東西，只 include clang 自己的 `stdarg.h` 和 `stddef.h`。

`code/main.py` 中的 `describe_object()` 讀取 4 位元組的 magic number 與一個 machine 欄位，並說出合約：

| Header | 欄位 | 值 | 規則書 |
|---|---|---|---|
| `7f 45 4c 46` ELF | offset 0x12 的 `e_machine` | 62 | System V AMD64 psABI |
| `64 86` COFF | offset 0 的 `Machine` | 0x8664 | Microsoft x64 |
| `7f 45 4c 46` ELF | `e_machine` | 183 | AAPCS64 |
| `cf fa ed fe` Mach-O 64 | offset 4 的 `cputype` | 0x0100000c | 帶有 Apple 差異的 AAPCS64 |

這三份合約在接下來表格的幾乎每一列都不同。每個值都引用自標頭列所指名的文件，而且每一個都由測試對照 fixture 檢查。

| 規則 | System V psABI | Microsoft x64 | AAPCS64 |
|---|---|---|---|
| 整數引數 | `%rdi %rsi %rdx %rcx %r8 %r9`，下一個空的暫存器 | `RCX RDX R8 R9`，依位置 | `x0` 到 `x7`，依 NGRN |
| 浮點引數 | `%xmm0` 到 `%xmm7`，下一個空的暫存器 | `XMM0` 到 `XMM3`，依位置 | `v0` 到 `v7`，依 NSRN |
| 整數結果 | `%rax`，然後 `%rdx` | `RAX` | `x0`，然後 `x1` |
| 浮點結果 | `%xmm0`，然後 `%xmm1` | `XMM0` | `v0` |
| 暫存器放不下的結果 | 位址「如同它是第一個引數」放在 `%rdi`，於 `%rax` 回傳 | 指標放在 `RCX`，引數向右移，於 `RAX` 回傳 | 位址放在 `x8`，`x0` 到 `x7` 不動 |
| Callee-saved | `%rbp %rbx %r12` 到 `%r15` | `RBX RBP RDI RSI RSP R12` 到 `R15`、`XMM6` 到 `XMM15` | `r19` 到 `r29`、`SP`、`v8` 到 `v15` 的低 64 位元 |
| 進入點時第一個 stack 引數 | `8(%rsp)` | `40(%rsp)`，位於 32 位元組 shadow store 之上 | `0(sp)` |
| Stack pointer 之下 | 128 位元組 red zone | 「視為 volatile」 | 沒有 red zone（Apple：128 位元組） |
| Call 時的對齊 | 進入點時 `(%rsp + 8)` 為 16 的倍數 | 在 prolog 與 epilog 之外 16 位元組對齊 | `SP mod 16 = 0` |
| Variadic 標記 | `%al` 持有所使用向量暫存器數量的上界 | 浮點值複製到整數暫存器 | 暫存器中沒有；Apple 使用 8 位元組 stack slot |

## System V：兩個獨立的暫存器序列與 red zone (System V: Two Independent Register Sequences and a Red Zone)

psABI 3.2.3 把每個引數的每個 eightbyte 分類為 INTEGER、SSE 或 MEMORY，然後從兩份互不影響的清單指派暫存器。「INTEGER」取「序列 %rdi、%rsi、%rdx、%rcx、%r8 和 %r9 中的下一個可用暫存器」。SSE 取 `%xmm0` 到 `%xmm7`。「如果引數的任何 eightbyte 沒有可用暫存器，整個引數就透過 stack 傳遞」，而 memory 引數「以反向（由右至左）的順序 push 到 stack 上」。

`-O2` 的 `eight_args(a, b, c, d, e, f, g, h)` 是 27 位元組：

```text
      10: 48 01 fe                     	addq	%rdi, %rsi
      13: 48 8d 04 0a                  	leaq	(%rdx,%rcx), %rax
      17: 48 01 f0                     	addq	%rsi, %rax
      1a: 4c 01 c0                     	addq	%r8, %rax
      1d: 4c 01 c8                     	addq	%r9, %rax
      20: 48 03 44 24 08               	addq	0x8(%rsp), %rax
      25: 48 03 44 24 10               	addq	0x10(%rsp), %rax
      2a: c3                           	retq
```

六個暫存器，然後是 `8(%rsp)` 和 `16(%rsp)` 的兩個 load。psABI 的 Figure 3.3 在以 `%rbp` 為 frame base 時，把「memory argument eightbyte 0」放在 `16(%rbp)`，這在進入點、任何 push 之前就是 `8(%rsp)`。`g` 在 8，`h` 在 16。

`mixed(int i, double d, long long l, float f)` 顯示兩份清單。Int 取 `%edi` 和 `%rsi`，也就是第一個和第二個 INTEGER 暫存器。Double 和 float 取 `%xmm0` 和 `%xmm1`，也就是第一個和第二個 SSE 暫存器。第一條指令是 `cvtsi2sd %edi, %xmm2`，而 `%xmm2` 是函式在讀取之前就先寫入的暫存變數暫存器。把 `cvtsi2sd` 的目的地當作讀取的分析器，會捏造出第三個浮點引數。radare2 正是這麼做的，並把 `mixed` 回報為 `xmm0`、`xmm1`、`xmm2` 中的 `(double arg1, double arg2, double arg3)`。

`by_value(struct pair p)` 把 16 位元組的 struct 作為兩個 INTEGER eightbyte，放在 `%rdi` 和 `%rsi` 傳入，並以同樣方式回傳：`leaq 0x1(%rdi), %rax; leaq -0x1(%rsi), %rdx; retq`。`big_ret(long long seed)` 回傳 32 位元組，屬於 MEMORY 類別，而 psABI 說「caller 提供回傳值的空間，並把該儲存空間的位址如同它是第一個引數一樣，傳在 %rdi」。`seed` 移到 `%rsi`，而函式以 `movq %rdi, %rax` 開頭，因為「回傳時 %rax 將包含 caller 在 %rdi 傳入的位址」。

Red zone 是 System V 的另一個特徵。psABI 3.2.2：「%rsp 所指位置之外的 128 位元組區域被視為保留，不得被 signal 或 interrupt handler 修改。」在 `-O0`，`eight_args` 是 `pushq %rbp; movq %rsp, %rbp`，然後是六個寫入 `-0x8(%rbp)` 到 `-0x30(%rbp)` 的 store，沒有 `subq`。這些 store 位於 stack pointer 之下 48 位元組。`-O2` 版的 `sum_varargs` 用 `subq $0x58, %rsp` 配置 88 位元組，然後把暫存器儲存區寫在 `-0x58(%rsp)` 往下到 `-0x80(%rsp)`：stack pointer 之下的 128 位元組，也就是整個 red zone。用 `-mno-red-zone` 建構同一個檔案，`leaf_add` 會多出 `pushq %rax` 與 `addq $0x8, %rsp`，以在 stack pointer 之上存放它 8 位元組的區域變數。

Variadic call 增加一個暫存器。「%al 用作隱藏的引數，指定所使用的向量暫存器數量」，這是 0 到 8 的上界。`call_varargs` 在 `-O2` 用 `xorl %eax, %eax` 設定它，在 `-O0` 用 `movb $0x0, %al`。`sum_varargs` 先讀取它：`testb %al, %al; je` 在沒有使用向量暫存器時，略過八個 `movaps %xmmN` 儲存。

## Microsoft x64：位置、shadow store，以及沒有 red zone (Microsoft x64: Positions, a Shadow Store, and No Red Zone)

Microsoft 的規則是依位置的。「整數引數透過暫存器 RCX、RDX、R8 和 R9 傳遞。浮點引數透過 XMM0L、XMM1L、XMM2L 和 XMM3L 傳遞。」以及「前四個參數中的任何浮點與倍精度引數，依位置透過 XMM0 - XMM3 傳遞。」同樣的 `mixed` 抵達時是 `%ecx`、`%xmm1`、`%r8`、`%xmm3`：slot 0 是 int，所以是 `RCX`；slot 1 是 double，所以是 `XMM1`；slot 3 是 float，所以是 `XMM3`。`XMM0` 和 `XMM2` 保持空白。

Caller 也必須留出空間。「Caller 必須永遠配置足夠的空間來儲存四個暫存器參數，即使 callee 沒有接受那麼多個。」這些頁面稱之為 shadow store 或 home address，而這四個 slot「實質上由被呼叫的函式擁有」。進入點時，`0(%rsp)` 是 return address，`8(%rsp)` 到 `39(%rsp)` 是四個 home，而第一個 stack 引數在 `40(%rsp)`。`-O2` 的 `eight_args` 讀取 `0x28(%rsp)`、`0x30(%rsp)`、`0x38(%rsp)`、`0x40(%rsp)`：40、48、56、64。

Caller 端顯示了算術。`-O2` 的 `call8` 是 `subq $0x48, %rsp`，然後在 `movl $1, %ecx` 等指令之前，把 5、6、7、8 存到 `0x20(%rsp)` 到 `0x38(%rsp)`。72 位元組是 32（shadow store）、32（四個 stack 引數），以及讓 return address 位於 16 位元組邊界的 8。`call_varargs` 只有暫存器引數，卻仍然配置 `subq $0x28, %rsp`：32 給 shadow store，8 給對齊。`keeps` push 三個 nonvolatile 暫存器（24 位元組），然後 `subq $0x20, %rsp`，讓 `ext` 在 return address 之下找到它的 32 位元組 shadow store。

Struct 遵循自己的規則。「任何放不進 8 位元組、或不是 1、2、4 或 8 位元組的引數，都必須以參照傳遞。」以及「16 位元組的引數以參照傳遞。」因此 `by_value` 在 `%rcx` 的隱藏結果指標之後，於 `%rdx` 收到指向副本的指標：「caller 必須配置回傳值的記憶體，並把指標作為第一個引數傳入。其餘的引數接著向右移一個引數。」`call_pair` 用 `movaps` 在 `0x20(%rsp)` 建構副本、用 `leaq 0x20(%rsp), %rdx` 傳遞它、用 `leaq 0x38(%rsp), %rcx` 傳遞結果緩衝區，並從 `0x40(%rsp)` 讀回 `.b`。

沒有 red zone。「目前 RSP 位址之外的所有記憶體都被視為 volatile：OS 或 debugger 可能在使用者 debug session 期間覆寫這塊記憶體，或是 interrupt handler 也可能。」`-O0` 的 `leaf_add` 需要 8 位元組的區域變數，並在進入點以 `pushq %rax`、在 `retq` 之前以 `popq %rcx` 取得它們，這是兩條單一位元組的指令，取代了 `subq` 與 `addq`。Stack 使用頁面定義了可以這麼做的 leaf：「leaf 函式是不需要函式表項目的函式。它不能改變任何 nonvolatile 暫存器，包括 RSP」。

Variadic callee 會把它們的暫存器存到 home。「對於 vararg 或沒有 prototype 的函式，任何浮點值都必須複製到對應的 general-purpose 暫存器。」`-O2` 的 `sum_varargs` 在一次 `pushq %rax` 之後，把 `%rdx`、`%r8`、`%r9` 存到 `0x18(%rsp)`、`0x20(%rsp)`、`0x28(%rsp)`，也就是進入點 offset 16、24、32：第二、第三、第四個暫存器參數的 home。它的 `va_list` 是 `leaq 0x18(%rsp), %rax`，即 `%rdx` 的 home，也就是第一個沒有名稱的引數。

## AAPCS64，以及 Apple 改變了什麼 (AAPCS64, and What Apple Changes)

AAPCS64 6.8.2 走過一份帶有兩個計數器的規則清單。C.9：整數引數「被複製到 x[NGRN] 的最低有效位元」，而 NGRN 遞增。C.1 對浮點引數用 NSRN 做同樣的事。C.12：最多 `8 - NGRN` 個 double-word 的複合型別「被複製到連續的 general-purpose 暫存器」。C.13 在一個引數去往 stack 之後，設定「NGRN 被設為 8」，所以之後不會再回到暫存器。`eight_args` 使用 `x0` 到 `x7`，從不碰 stack。`mixed` 是 `w0`、`d0`、`x1`、`s1`。`by_value` 進出都是 `x0` 和 `x1`。

6.9 說，原本會在暫存器中傳遞的結果「在相同的暫存器中回傳」，否則「記憶體區塊的位址必須在 x8 中作為額外的引數傳給函式」。`big_ret` 寫入 `str x0, [x8]`、`stur q0, [x8, #0x8]`、`str x9, [x8, #0x18]`，並從不寫入 `x0`。`call_big` 設定 `mov x8, sp`，而 `seed` 留在 `x0`，因為 6.9 補充說「callee 沒有義務保留儲存在 x8 的值」，而 `x8` 不是引數暫存器之一。

6.1.1：「子程式呼叫必須保留暫存器 r19-r29 和 SP 的內容。」6.4.5.2：在公開介面上「SP mod 16 = 0」。Stack 是 full-descending，而且「不允許任何 thread 存取（讀或寫）」其下方的 inactive 區域，所以 Linux 的 `-O0` `leaf_add` 為 8 位元組的區域變數送出 `sub sp, sp, #0x10`，而 System V 用的是 red zone。

6.4.6 定義了 frame record：「stack 上由兩個 64 位元值組成的 frame record ……位址最低的 double-word 應指向前一個 frame record，而位址最高的 double-word 應包含進入目前函式時 LR 中傳入的值。」以及「frame record 在 stack frame 中的位置沒有被指定。」clang 在這一個檔案中顯示了兩種位置。Linux 上 `-O2` 的 `keeps` 以 `stp x29, x30, [sp, #-0x30]!; mov x29, sp` 把 record 放在底部。同一個函式在 `-O0` 則以 `sub sp, sp, #0x20; stp x29, x30, [sp, #0x10]; add x29, sp, #0x10` 把它放在頂部。

Apple 的 arm64 筆記保留暫存器與對齊，並改變 frame、red zone 與 variadic call。「Frame pointer 暫存器 (x29) 必須永遠指向有效的 frame record。有些函式，例如 leaf 函式或 tail call，可以選擇不在這份清單中建立項目。」Apple 的 build 的 `keeps` 先儲存 `x22, x21, x20, x19`，並把 record 放在頂部：`stp x29, x30, [sp, #0x20]; add x29, sp, #0x20`。「各平台保留暫存器 x18。請勿使用這個暫存器。」「ARM64 red zone 由 stack pointer 正下方的 128 位元組組成」，而 clang 對 `-O0` 的 leaf 仍然送出 `sub sp, sp, #0x10`，所以編譯器預設不使用它。Variadic 引數被指派到「適當數量的 8 位元組 stack slot」，而「型別 va_list 是 char* 的別名」。Apple 上的 `call_varargs` 把 10、20、30 存到 `[sp]`、`[sp, #8]`、`[sp, #0x10]`，並只傳 `w0`；Linux 的 build 則傳 `w1`、`w2`、`w3`，並以 `b extva` 做 tail call。Apple 的 `sum_varargs` 讀取 `add x8, sp, #0x10`（caller 的 stack pointer）作為它的 `va_list`；Linux 的 `sum_varargs` 把 `x1` 到 `x7` 與 `q0` 到 `q7` 存進 224 位元組的 frame，並用一條 `mov x8, #-0x38; movk x8, #0xff80, lsl #32` 把結構欄位 `__gr_offs` 初始化為 -56，`__vr_offs` 初始化為 -128。

Symbol 也不同。Mach-O 在每個 C symbol 前加上 `_`，所以清單顯示 `_keeps`，而第一個函式出現在私有標籤 `ltmp0` 之下。分析器透過 symbol table 重新命名標籤。

## 手動解碼一個 prologue，三次 (Decode One Prologue by Hand, Three Times)

`keeps(int n)` 在迴圈中呼叫 `ext(i)`，並跨 call 保留 `n`、`i` 和 `acc`，所以每種 ABI 都必須儲存暫存器。針對每個目標，閱讀 `-O2` build 的前幾個位元組。

System V，4 位元組：

1. `55` 是 `push r64`，暫存器在低三個位元：`50 + 5`，`%rbp`。深度 8，`%rbp` 儲存在進入點 offset -8。
2. `41 56`：`41` 是設定 `B` 的 REX，`56` 是 `50 + 6`。REX.B 把 6 擴充為 14：位於 -16 的 `pushq %r14`。
3. `53` 是 `50 + 3`：位於 -24 的 `pushq %rbx`。

三次 push 是 24 位元組。Return address 加 8，而 32 是 16 的倍數，所以迴圈內的 `callq ext` 不需要 `subq` 就是對齊的。沒有 `movq %rsp, %rbp`，所以這個 build 沒有 frame pointer。本文把 `%rbp` 當作持有 `acc` 的 callee-saved 暫存器使用（`xorl %ebp, %ebp`、`addl %eax, %ebp`）。Epilogue 是鏡像：`5b 41 5e 5d c3`。

Microsoft x64，7 位元組：

1. `56` 是位於 -8 的 `pushq %rsi`。`57` 是位於 -16 的 `pushq %rdi`。`53` 是位於 -24 的 `pushq %rbx`。三者在這裡都是 nonvolatile，而 `RDI` 和 `RSI` 在 System V 下是引數暫存器。
2. `48 83 ec 20`：REX.W、帶 8 位元 immediate 的 opcode `83 /5`，ModRM `ec` 是 `11 101 100`：`subq $0x20, %rsp`。深度 56。

56 加上 return address 是 64，已對齊。那 32 位元組就是 `ext` 應得的 shadow store。Epilog 頁面只允許「一個 add RSP,constant 或 lea RSP,constant[FPReg]，後面接零個或多個 8 位元組暫存器 pop，以及一個 return 或 jmp」，而位元組是 `48 83 c4 20 5b 5f 5e c3`。

AArch64，四個 word 共 16 位元組，little-endian：

1. `a9bd7bfd` 是帶 pre-index 的 `stp`。位元 31:30 `10` 選擇 64 位元暫存器，位元 25:23 `011` 選擇 pre-index，位元 22 為 0 表示 store。位元 21:15 的 imm7 是 `1111010`，即 -6，以 8 為 scale：-48。位元 14:10 的 Rt2 是 30，位元 9:5 的 Rn 是 31（`sp`），位元 4:0 的 Rt 是 29。`stp x29, x30, [sp, #-0x30]!`：stack pointer 下降 48，`x29` 進入 -48，`x30` 進入 -40。
2. `f9000bf5` 是帶無號 12 位元 offset、以 8 為 scale 的 `str`：位元 21:10 的 imm12 是 2，Rn 是 `sp`，Rt 是 21。`str x21, [sp, #0x10]`，進入點 offset -32。
3. `910003fd` 是 `add` immediate：imm12 為 0，Rn 是 `sp`，Rd 是 29。`mov x29, sp`：frame pointer 指向位於 -48 的 record。
4. `a9024ff4` 是帶有號 offset 的 `stp`：imm7 是 4，scale 後為 32。Rt2 是 19，Rt 是 20。`stp x20, x19, [sp, #0x20]`：`x20` 在 -16，`x19` 在 -8。

Frame 是 48 位元組：`x29` 在 0、`x30` 在 8、`x21` 在 16、24 處有 8 位元組 padding、`x20` 在 32、`x19` 在 40。如 6.4.6 所要求，record 最低的 word 指向前一個 record，而下一個 word 持有 LR。`cmp w0, #0x1` 位於第一個與第二個 store 之間；排程器把它移進了 prologue，而分析器容許在 stack 操作之間有比較或算術指令。

## 編譯器把相同的事實寫了兩次 (The Compiler Wrote the Same Facts Twice)

Prologue 解析是一個推導。編譯器為 unwinder 送出第二個推導，而兩者必須一致。

在 Windows 上，第二個推導是 `.pdata` 和 `.xdata`。在不加 `-fno-asynchronous-unwind-tables` 的情況下建構 `-O2` object，`llvm-objdump -h` 顯示 0x34 位元組的 `.xdata` 與 0x48 位元組的 `.pdata`。0x48 是 72，也就是六個 12 位元組的 `RUNTIME_FUNCTION` 項目，每個都有三個 `IMAGE_REL_AMD64_ADDR32NB` relocation。這六個是 `sum_varargs`、`keeps`、`call8`、`call_varargs`、`call_pair` 和 `call_big`。其他六個函式是 leaf，而例外處理頁面說「如果搜尋沒有找到函式表項目，就假設該程式碼是 leaf 函式的一部分，而 RSP 直接指向 return pointer。」

`keeps` 的 `.xdata` 位元組是 `01 07 04 00 07 32 03 30 02 70 01 60`。`UNWIND_INFO` 以 3 位元的 Version 與 5 位元的 Flags 開始：version 1，沒有 flag。SizeOfProlog 是 7，CountOfCodes 是 4，FrameRegister 和 FrameOffset 是 0。每個 `UNWIND_CODE` 是 prolog 中的 offset 加上 opcode 與 info 的一個位元組。`07 32`：在 offset 7，op 2 是 `UWOP_ALLOC_SMALL`，info 3 表示 `3 * 8 + 8 = 32` 位元組。`03 30`：op 0 是 `UWOP_PUSH_NONVOL`，暫存器 3 是 `rbx`。`02 70`：`rdi`。`01 60`：`rsi`。這些 code 以執行的相反順序儲存，並指名與位元組 `56 57 53 48 83 ec 20` 相同的三次 push 與相同的 32 位元組配置。`sum_varargs` 的項目是 `01 01 01 00 01 02 00 00`：1 位元組的 prolog 加一個 code，8 位元組的 `UWOP_ALLOC_SMALL`。這就是 unwinder 看待 `pushq %rax` 的方式：一次配置，因為 `%rax` 是 volatile，沒有任何東西被儲存。

在 Linux 上，第二個推導是 `.eh_frame`，由 psABI 4.2.4 定義，並由 DWARF 5 的 6.4 節編碼。在可重定位的 object 上執行 `llvm-dwarfdump --eh-frame`，會為每個函式印出一個 FDE，並解析 `R_X86_64_PC32` relocation，所以範圍可用。對 `keeps`：

```text
  DW_CFA_def_cfa_offset: +16
  DW_CFA_def_cfa_offset: +24
  DW_CFA_def_cfa_offset: +32
  DW_CFA_offset: RBX -32
  DW_CFA_offset: R14 -24
  DW_CFA_offset: RBP -16
```

CFA 是 call 之前的 stack pointer，比進入點的 stack pointer 高 8，所以 `RBP -16` 是進入點 offset -8，`R14 -24` 是 -16，`RBX -32` 是 -24，而最深的 CFA offset 32 是 24 位元組的 frame 加上 return address。`keeps` 的 AArch64 FDE 說 `DW_CFA_def_cfa: W29 +48`，搭配 `W29 -48`、`W30 -40`、`W21 -32`、`W20 -16`、`W19 -8`。在那裡 CFA 就是進入點的 stack pointer 本身，所以 offset 就是不需轉換的 frame offset，而當 frame record 存活時，CFA 是由 `x29` 定義的。`eh_frame_cross_check()` 對兩個 object 的全部 12 個 FDE 執行這個比較。有一個項目需要小心：System V 下的 `call8` 在本文中途 push 兩個外送引數，所以它最深的 CFA offset 是 32，而它的 prologue frame 是 8，因此檢查比較的是本文到達的最深 stack pointer 24，再加 8。

## 工具怎麼稱呼這些 (What the Tools Call This)

這個領域的每個工具都有一個呼叫慣例的資料結構，以及一個為函式指派呼叫慣例的 pass。名稱不同，缺口也不同。閉源工具是根據它們的文件描述；開源工具則在 fixture 上執行，輸出擷取在 object 旁邊。

**Ghidra 12.1.4（Apache-2.0）。** 規則存在於與 SLEIGH 語言並列的編譯器規格檔 `.cspec` 中。`x86-64-gcc.cspec` 宣告了名為 `__stdcall` 的 `<default_proto>`，「衍生自 System V Application Binary Interface AMD64 Architecture Processor Supplement April 2016」，帶有 `XMM0_Qa` 到 `XMM7_Qa` 與 `RDI, RSI, RDX, RCX, R8, R9` 的 `<pentry>` 清單、位於 offset 8 的 stack 項目、針對最多 16 位元組型別的 `join_dual_class` 規則、輸出 `XMM0_Qa, XMM1_Qa, RAX, RDX`，以及其餘情況的 `hidden_return`。`x86-64-win.cspec` 宣告了 `__fastcall`，其四個 `<group>` 元素把 `XMM0_Qa` 與 `RCX`、`XMM1_Qa` 與 `RDX` 等配對，這就是依位置的規則；一個位於 offset 40 的 stack 項目；針對 3、5、6 或 7 位元組 struct 的 `convert_to_ptr`，註解為「只有大小為 8、16、32、64 位元的 struct 才能打包進暫存器」；列出 `RBX RBP RDI RSI RSP R12` 到 `R15` 與 `XMM6` 到 `XMM15` 的 `unaffected`；以及 stack 8 到 39 的 `localrange`，也就是 shadow store。`AARCH64.cspec` 宣告了 `__cdecl`，其第一個項目是 `storage="hiddenret"` 的 `x8`，然後是 `q0` 到 `q7` 和 `x0` 到 `x7`，針對 17 位元組以上 struct 的 `convert_to_ptr`，以及包含 `x8` 的 `killedbycall`，註解為「間接結果位置暫存器，不反映在 pentry 清單中」。反編譯器的最後階段名為「Establish Function's Prototype」；analyzer 是「Call Convention ID」和「Decompiler Parameter ID」；程式選項「Prototype Evaluation」挑選預設模型。

**IDA Pro 與 IDA Free（Hex-Rays，商業；目前版本 9.4，build 9.4.260714）。** 函式型別是透過 SetType 設定的 C 宣告，而「IDA 會使用型別資訊，以函式引數的資訊為反組譯加上註解。」沒有任何標準呼叫慣例會產生的位置，以 IDA 自己的慣例書寫：`int __usercall func@<ebx>(int x, int y@<esi>)` 把結果放在 `EBX`，把 `y` 放在 `ESI`，而 `@<^12.4>` 以 offset 與大小指名 stack 位置。`__userpurge` 相同，但由 callee 清理 stack。同一頁列出用於被破壞集合的 `__spoils`，以及用於這堂課手動還原的隱含引數的 `__hidden`、`__return_ptr` 和 `__struct_ptr`。反編譯器的「Decompile as call」動作接受單一指令的 `__usercall` prototype。IDA Free 只支援 x86 與 x86-64。

**Binary Ninja 6.0「Krypton」（Vector 35，商業，附 Free 版，可反編譯 x86、x86_64、ARMv7 與 AArch64）。** `Platform`「包含與 binary 執行環境有關的所有資訊，主要是所使用的呼叫慣例、作業系統與架構」，清單包括 `linux-x86_64`、`windows-x86_64`、`linux-aarch64` 和 `mac-aarch64`，每一個都有 `default_calling_convention`。`CallingConvention` 有 `int_arg_regs`、`float_arg_regs`、`callee_saved_regs`、`caller_saved_regs`、`int_return_reg`、`high_int_return_reg`、`float_return_reg`，以及兩個直接編碼這堂課差異的旗標：`arg_regs_share_index`，「為 True 時，第 N 個引數不論型別，都消耗整數與浮點暫存器清單的第 N 個 slot」，以及 `stack_reserved_for_arg_regs`，「Windows x64 呼叫慣例使用的 shadow/home 空間」。`eligible_for_heuristics`、`required_arg_regs` 和 `required_clobbered_regs` 驅動它的啟發式慣例偵測。平台型別是 `types/platform/` 下的 C header，每個平台名稱一個。

**radare2 6.2.4（LGPL-3.0，此處已執行）。** `afcl` 列出 architecture plugin 認得的慣例：x86-64 是 `amd64 amd64syscall dlang golang golang_abi0 ms p9 reg swift`，AArch64 是 `arm64 dlang golang p9 reg swift`。`afc` 回報目前函式的慣例，`afcr` 印出 `cc.amd64.arg0: rdi` 到 `arg5: r9` 以及 `ret0: rax`，而 profile 是名為 `cc-ARCH-BITS.sdb` 的 SDB 檔案。radare2 為 ELF object 挑了 `amd64`，為 COFF object 挑了 `ms`，這是正確的區分。`afv` 列出引數與變數，而 `afs` 印出簽章。在 Linux 的 `-O2` object 上，`afv` 為 `eight_args` 找到七個引數，漏掉 `rcx`；在 Windows object 上，它找到 `rcx, rdx, r8` 與四個 stack slot，漏掉 `r9`。兩個漏掉的暫存器都只出現在 `lea` 內。對 `mixed`，它在 Linux 上回報 `double arg1 @ xmm0, arg2 @ xmm1, arg3 @ xmm2`，在 Windows 上完全沒有引數。在 AArch64 上，它為 `eight_args` 找到 `x0` 到 `x7`，而對 `big_ret`，它列出 `x0` 與 `double arg2 @ d1`，而函式實際讀取的是 `x8`，並把常數載入 `q1`。每個 `afs` 簽章都以 `void` 開頭，因為不推論回傳型別。在 `-O0`，每個引數都存進 frame，`afv` 找到 `eight_args` 的全部八個引數，其中 stack 的那一對是 `rbp+0x10` 和 `rbp+0x18`。

**rizin 0.9.1（LGPL-3.0，此處已執行）與 Cutter 2.5.0。** `afcl` 列出 `amd64 amd64syscall ms reg swift`。`afc` 對 COFF object 回傳 `amd64`，所以 Microsoft profile 存在，卻沒有從格式選出。`afi` 為 Linux 上的 `eight_args` 列出七個引數，同樣沒有 `rcx`，而在 Windows 上是 `r8, rcx, rdx` 加三個 stack slot，沒有 `r9`，也沒有 `0x40` 的 slot。`afv` 不存在；列出的指令是 `afvl`。Cutter 是 rizin 之上的圖形化前端。

**angr 10.0.3（BSD-2，此處已執行）。** 先 `CFGFast`，再 `CompleteCallingConventions(recover_variables=True)`，依 binary 格式將 `function.calling_convention` 設為 `SimCCSystemVAMD64`、`SimCCMicrosoftAMD64` 或 `SimCCAArch64`，並將 `function.prototype` 設為簽章。它在三個目標上都還原了 `eight_args` 的全部八個引數；在 x86-64 上還原了 `big_ret` 的隱藏指標，為 `(BOT*, long long) -> BOT*`；把 Windows `by_value` 的以參照傳遞 struct 還原為 `(BOT*, BOT*) -> BOT*`；並把 System V 的 variadic `sum_varargs` 還原為 `(long long, ...)`。它的缺口與 radare2 不同：`mixed` 在 System V 和 AArch64 上回來的是 `(long long, long long, double, float)`，暫存器正確但順序錯誤，因為整數清單印在浮點清單之前；AArch64 的 `big_ret` 回來的是 `(long long) -> double`，所以 `x8` 慣例被漏掉，而向量 store 被讀成結果；AArch64 的 `sum_varargs` 回來有 16 個參數，也就是整個暫存器儲存區。

**其他。** Hopper 6.6.0 是適用於 macOS 與 Linux 的商業 disassembler 與反編譯器；這堂課沒有執行它的慣例處理，也沒有文件記載。lldb 1703 與 gdb 以 `frame info` 和 `register read` 顯示即時的 frame，這是課程 42 對相同 offset 的動態視圖。Compiler Explorer 從原始碼並排顯示三份清單，是在你於位元組中尋找規則之前，確認某條規則最快的方式。

| 概念 | Ghidra | IDA | Binary Ninja | radare2 / rizin | angr | 這堂課 |
|---|---|---|---|---|---|---|
| 慣例物件 | `.cspec` 中的 `prototype` | 函式型別中的呼叫慣例關鍵字 | `Platform` 上的 `CallingConvention` | `afc` 名稱、`cc-ARCH-BITS.sdb` | `SimCC` 子類別 | `expected_locations(abi, name)` |
| 每個目標的預設值 | 每個 cspec 的 `<default_proto>` | 編譯器設定 | `Platform.default_calling_convention` | `afc=`、`anal.cc` | 依格式 | `describe_object()` |
| 引數暫存器 | `<pentry>` 清單，位置式用 `<group>` | `__usercall` 中的 `@<reg>` | `int_arg_regs`、`float_arg_regs`、`arg_regs_share_index` | `cc.NAME.argN` | `ARG_REGS`、`FP_ARG_REGS` | `INT_ARG_REGS`、`FLOAT_ARG_REGS` |
| Shadow store | `localrange` stack 8 到 39 | 隱含 | `stack_reserved_for_arg_regs` | `cc.ms.argn=stack` | `STACKARG_SP_DIFF` | `STACK_ARG_BASE[WIN64] = 40` |
| 隱藏的結果指標 | `hidden_return`、`storage="hiddenret"` | `__return_ptr`、`__hidden` | 對 `int_return_reg` 而言過大的回傳型別 | 未建模 | `BOT*` 第一個引數 | `HIDDEN_RETURN_REG` |
| Callee-saved | `<unaffected>` | `__spoils` 列出相反的集合 | `callee_saved_regs` | profile | `CALLEE_CLEANUP` 與 arch | `CALLEE_SAVED` |
| 為函式指派 | 「Call Convention ID」analyzer、「Establish Function's Prototype」 | 使用者或型別函式庫 | 以 `required_arg_regs` 做啟發式偵測 | `afc`、`afva`、`aaft` | `CallingConventionAnalysis` | 先讀後寫加上規則書 |

## 動手做 (Build It)

執行 demo：

```bash
cd phases/13-tools-and-protocols/38-calling-conventions-and-stack-frames
python3 code/main.py
```

輸出是一份 JSON 報告。對七個 object 中的每一個，它指出格式、machine 與 ABI，而對十二個函式中的每一個，它給出 prologue 位元組、帶進入點 offset 的被儲存暫存器、frame 大小、frame pointer、參數暫存器、stack 引數 offset、spill、回傳暫存器或隱藏指標、call 深度，以及 caller 的外送暫存器與 store。每個函式都以它的檢查結束：規則、來自簽章的預期值、來自位元組的觀察值，以及兩者是否一致。報告以兩項 unwind 交叉檢查與工具擷取作結。在 84 個函式分析中，執行了 489 項檢查，而 489 項全部一致。

預期的形狀：

```json
{
  "objects": {
    "convention-x86_64-windows-O2": {
      "format": "COFF", "machine": "AMD64", "abi": "Microsoft x64", "checks": "73/73 agree",
      "functions": {
        "keeps": {
          "prologue": ["56  pushq %rsi", "57  pushq %rdi", "53  pushq %rbx", "48 83 ec 20  subq $0x20, %rsp"],
          "frameBytes": 56,
          "savedRegisters": [{"register": "rsi", "entryOffset": -8}, {"register": "rdi", "entryOffset": -16}, {"register": "rbx", "entryOffset": -24}],
          "parameterRegisters": ["rcx"],
          "returnRegisters": ["rax"],
          "calls": [{"callee": "ext", "depth": 56}],
          "checks": [{"rule": "Microsoft x64: the caller must always allocate ...", "expected": "at least 32 bytes ...", "observed": [56], "agree": true}, "..."]
        }
      }
    }
  },
  "sehCrossCheck": ["..."],
  "ehFrameCrossCheck": {"...": "..."}
}
```

依此順序閱讀 `code/main.py`：

1. `describe_object()` 讀取 ELF、COFF 或 Mach-O 的 header 與 symbol table，並回傳 ABI 與函式清單。
2. `parse_listing()` 把 `llvm-objdump -d -r` 的文字切成函式、保留每個 call 與跳轉的 relocation 目標，並用正確語法的標記移除註解：AT&T 用 `#`，AArch64 用 `//` 和 `;`，因為在 AArch64 中 `#` 引入 immediate。
3. `effect_x86()` 與 `effect_arm()` 把一條指令變成讀取、位址讀取、寫入、load、store 與 stack pointer 的 delta。AT&T 的最後一個 operand 是目的地；AArch64 的第一個 operand 是目的地，store、比較與分支除外。`lea` 讀取位址而不載入任何東西。`mov`、`cvtsi2sd`、`setcc` 與 zero idiom `xor r, r` 的目的地只是寫入。Call 寫入整個 volatile 集合。Pre-indexed 的 store 在存取之前移動 `sp`，而 post-indexed 的 load 在之後移動。
4. `analyze_function()` 走訪本文一次，直到第一個 `ret` 或 tail jump。它追蹤 stack pointer 在進入值之下的深度，所以記憶體 operand `disp(%rsp)` 位於進入點 offset `disp - depth`，而 `disp(%rbp)` 位於 `frame_pointer + disp`。它記錄在寫入之前就被讀取的引數類別暫存器、位於或高於 ABI 第一個 stack 引數 offset 的 load、把引數暫存器存進 frame 的 store、位於目前 stack pointer 之下的存取，以及每次 call 的深度。
5. `_return_registers()` 在結果暫存器於離開之前最後一次存取是寫入時，保留它。作為資料 operand 的讀取會殺掉候選；作為位址 base 的讀取則不會，這讓 Microsoft x64 下 `big_ret` 中的 `rax` 得以保留，在那裡 `movq %rcx, %rax` 後面接著透過 `(%rax)` 的 store。
6. `expected_locations()` 把三份規則書套用到 C 簽章：帶 eightbyte 類別的兩份 System V 清單、帶以參照傳遞 struct 的 Microsoft 位置式 slot，以及對大於 16 位元組的結果使用 `x8` 的 AAPCS64 計數器。
7. `run_checks()` 比較兩者，並加入 frame 規則：被寫入的 callee-saved 暫存器必須被儲存、在每次 call 時 `(rsp + 8) % 16 == 0` 或 `sp % 16 == 0`、每次 Microsoft call 都有 32 位元組的 shadow、在 Microsoft 與 AAPCS64 下不得存取 stack pointer 之下、在 System V 下最多 128 位元組，以及鏡射 prologue、且 stack pointer 回到進入值的 epilogue。
8. `seh_cross_check()` 與 `eh_frame_cross_check()` 讀取不加 `-fno-asynchronous-unwind-tables` 所建構的 object 的 unwind 資料，並將它與 prologue 解析比較。

還有另外三個指令。`python3 code/main.py analyze convention-x86_64-windows-O2 keeps` 印出一個函式。`python3 code/main.py object FILE` 印出 header 的判定。`python3 code/main.py seh` 印出每個有 unwind 資料的函式的已解碼 `UNWIND_INFO`。

## 實際使用 (Use It)

依此順序，在你建構的新 object 上使用分析器。

1. 執行 `python3 code/main.py object FILE.o` 並寫下它指出的 ABI。如果 header 與你的建構指令不一致，就停下來；其中一個是錯的。
2. 在 object 旁邊擷取 `llvm-objdump -d -r FILE.o`，並執行分析。先閱讀 prologue 位元組與被儲存的暫存器，再做別的事，並把 frame 大小與 push 和 `sub` 比較。
3. 對照你的原始碼簽章，閱讀參數暫存器與 stack offset。簽章無法解釋的暫存器，是隱藏指標、variadic 標記，或是被分析器誤認為引數的暫存變數暫存器。最後一種情況是 effect 表中的 bug。請用針對該指令的一般性規則修正它。
4. 閱讀不一致的檢查（如果有的話）。每一項都指名它所依據的規則，所以先檢查規則，再檢查位元組，最後才檢查分析器。
5. 當 object 有 unwind 資料時，執行交叉檢查。兩個互相一致的推導，是這堂課能產生的最強證據。
6. 當安裝了其他工具時，用確切的指令擷取它的輸出並比較。一致的地方，你有三個推導。不同的地方，你有一個具體的暫存器和一條具體的指令可以看，那就是課程 35 的發現。

## 互動實驗 (Interactive Lab)

### 實驗 A：一條指令，三種效果

從 `code` 目錄開啟 Python：

```bash
cd phases/13-tools-and-protocols/38-calling-conventions-and-stack-frames/code
python3 -q
```

執行：

```python
from main import *
effect(Instruction(0, "48 8d 04 0a", "leaq", "(%rdx,%rcx), %rax"), SYSV)
effect(Instruction(0, "f2 0f 2a d7", "cvtsi2sd", "%edi, %xmm2"), SYSV)
effect(Instruction(0, "a9bd7bfd", "stp", "x29, x30, [sp, #-0x30]!"), AAPCS64)
```

在每次呼叫之前，先寫下你預期的讀取、位址讀取、寫入與 stack delta。然後用 `ldp` 把 `[sp, #-0x30]!` 改成 `[sp], #0x30`，並看 delta 從 `sp_delta` 移到 `post_sp_delta`。

### 實驗 B：keeps 的 frame

```python
info, listing = load_fixture("convention-aarch64-linux-O2")
keeps = analyze_function("keeps", listing["keeps"], info.abi)
keeps.saved
keeps.frame_pointer
[c.to_dict() for c in frame_cells(keeps)]
```

把 offset 與上面手動解碼的結果比較。然後載入 `convention-arm64-macos-O2` 並執行相同的三行；record 移到 -16，有四個暫存器位於它之下。再載入 `convention-x86_64-linux-O2-fp`，並閱讀 `keeps` 的 `saved`：frame pointer 回來了，一個唯一的工作是讓 call 時 `(rsp + 8)` 成為 16 的倍數的 `pushq %rax` 也回來了。

### 實驗 C：mixed 的先讀後寫

```python
info, listing = load_fixture("convention-x86_64-linux-O2")
mixed = analyze_function("mixed", listing["mixed"], info.abi)
mixed.param_registers
mixed.return_registers
expected_locations(SYSV, "mixed")["parameters"]
expected_locations(WIN64, "mixed")["parameters"]
```

System V 的清單是 `rdi, xmm0, rsi, xmm1`，而 Microsoft 的清單是 `rcx, xmm1, r8, xmm3`。現在載入 Windows object，再次執行前三行。然後開啟 Linux object 的 radare2 擷取，找到 `[afv mixed]`：三個 double。找出解釋第三個的那條指令。

### 實驗 D：unwinder 的第二意見

```python
for row in seh_cross_check():
    print(row["function"], row["sizeOfProlog"], row["unwindPushes"], row["unwindAllocBytes"], row["agree"])
for row in eh_frame_cross_check("convention-x86_64-linux-O2-eh"):
    print(row["function"], row["cfiSaved"], row["cfiMaxOffset"], row["deepestStack"], row["agree"])
```

在 SEH 表中找到 `sum_varargs` 的 `pushq %rax`，並說明為什麼 unwind code 將它記錄為配置。然後在 `.eh_frame` 表中找到 `call8`，並說明那個 32。

## 實作練習 (Practice Lab)

新增第四個目標與第四份規則書。

需求：

- 用 `-target x86_64-apple-macos -O2` 編譯 `convention.c` 並擷取它的清單。擴充 `describe_object()`，讓它讀取 Mach-O 的 `cputype` 0x01000007，並把它對應到 Apple 在 x86-64 上遵循的 System V psABI，附帶 `_` symbol 前綴。
- 檢查每個函式都與 System V 的檢查一致。記錄你在 prologue 中發現的那一項差異：Darwin 在每個最佳化層級都保留 frame pointer。
- 新增 Linux x86-64 object 的 `-fstack-protector-strong` build。擴充 prologue 掃描，讓來自 `%fs:0x28` 的 canary load 與區域變數之下的 store 被記錄為 `canary` 格子，而 `ret` 之前的比較則被記錄為守護 epilogue 的檢查。
- 新增測試，當 canary 的 store 被算成 spill 的引數時就失敗。
- 在執行分析器之前，先手動畫出 `with_canary` 風格程式碼的 frame，並把這張圖與測試放在一起。

成功的意思是，`python3 code/main.py` 回報新的 object，且每項檢查都一致，而受保護函式的 frame 格子顯示 canary 位於區域變數與被儲存的暫存器之間。

## 課程成品 (Lesson Artifact)

這堂課包含 `outputs/skill-calling-conventions-and-stack-frames.md`。用它把你建構的 object、它的清單與 C 簽章，變成參數位置、frame 版面、回傳暫存器，以及對每個推論所依據的規則和編譯器 unwind 資料的檢查。它要求先給出 header 的 ABI 判定，禁止把 `cvtsi2sd` 或 `xor r, r` 的目的地讀成讀取，並以第一個不一致作結。

## 驗證 (Verify It)

執行 demo 與確定性的測試套件：

```bash
cd phases/13-tools-and-protocols/38-calling-conventions-and-stack-frames
python3 code/main.py
python3 -m unittest discover -s code/tests -v
```

驗證應該證明：

- 七個 object 的 ELF、COFF 與 Mach-O header，指出 System V、Microsoft x64、AAPCS64 與 Apple arm64
- 每個 object 的 symbol table 列出相同的十二個函式，其中 Mach-O 的 `_` 前綴與 `ltmp0` 標籤已被解析
- 清單 parser 將 `ext` 保留為 call 目標，並保持 AArch64 的 immediate（例如 `#-0x30`）完整
- `leaq (%rdx,%rcx), %rax` 讀取兩個位址暫存器而不載入任何東西；`xorl %eax, %eax` 和 `cvtsi2sd` 是只寫的；pre-indexed 的 `stp` 在 store 之前移動 `sp`，而 post-indexed 的 `ldp` 在 load 之後移動
- `-O2` 的 `keeps` 在 System V 上是 `55 41 56 53`，帶 24 位元組 frame 且沒有 frame pointer；在 Microsoft x64 上是 `56 57 53 48 83 ec 20`，帶 56 位元組 frame；在 AArch64 上第一個是 `a9bd7bfd`，record 位於 -48
- Apple 的 build 把 record 放在 -16，位於四個被儲存的暫存器之上
- 每個函式在 Linux 的 `-O0` 有 frame pointer，在 `-O2` 沒有，而 `-fno-omit-frame-pointer` 會連同一次對齊的 push 一起把它帶回來
- `-O0` 的 leaf 在 System V 下使用 8 位元組的 red zone，在 Microsoft x64 下使用 `pushq %rax`，在 AAPCS64 下使用 `sub sp, sp, #0x10`；`-mno-red-zone` 移除 red zone 的使用
- 每個 epilogue 都鏡射它的 prologue，而 stack pointer 回到進入值
- `eight_args` 在兩個最佳化層級，讀取六個暫存器與 slot 8 和 16、四個暫存器與 slot 40 到 64，或 `x0` 到 `x7`
- `mixed` 在 System V 下是 `rdi, xmm0, rsi, xmm1`，在 Microsoft x64 下是 `rcx, xmm1, r8, xmm3`，兩者的結果都是 `xmm0`
- 16 位元組的 struct 在 System V 與 AAPCS64 下透過兩個暫存器傳遞，在 Microsoft x64 下則在隱藏指標之後以參照傳遞
- 32 位元組的 struct 透過 `rdi`、`rcx` 或 `x8` 回傳
- variadic callee 讀取 `%al`、把 `rdx, r8, r9` 存到 16、24、32、儲存 `x1` 到 `x7` 與 `q0` 到 `q7`，或在 Apple 上讀取 caller 的 stack pointer
- caller 設定預期的暫存器、把 7 和 8 存到 0 和 8，或把 5 到 8 存到 32 到 56、為 variadic call 將 `%al` 歸零，並透過 `rdi`、`rcx` 或 `x8` 傳遞結果緩衝區位址
- 全部 489 項檢查都與它們的規則一致
- 六個 `UNWIND_INFO` 項目與 prologue 解析相符，而 leaf 沒有
- 24 個 FDE 與被儲存暫存器的 offset 及最深的 stack pointer 相符
- 工具擷取記錄了 radare2 的 `amd64` 與 `ms`、rizin 在 COFF object 上的 `amd64`、漏掉的 `rcx`、三個 double，以及 angr 的 prototype
- 內嵌在網站模組中的圖表資料，等於分析器的輸出

## Production 的失敗模式 (Production Failure Modes)

| 失敗 | 薄弱的工具回報什麼 | 分析必須證明什麼 |
|---|---|---|
| COFF object 被讀成 System V | `rdi` 和 `rsi` 被當作函式從未讀取的引數 | Header 寫 0x8664；慣例是 Microsoft x64 |
| 暫存器只在 `lea` 內被讀取 | 八個引數的函式回報七個引數 | Base 與 index 暫存器是讀取 |
| `cvtsi2sd` 的目的地被算成讀取 | `xmm2` 中出現第三個 double 引數 | 對 `xmm2` 的第一次寫入是寫入；`mixed` 有兩個 SSE 引數 |
| 位置式 slot 被讀成序列 | Windows 上把 `xmm0` 當作 double 引數 | Slot 1 是 `XMM1`，slot 3 是 `XMM3` |
| `push rbp` 被當成 frame pointer | 在把 `rbp` 當累加器使用的 `-O2` 函式中出現 frame pointer | 只有 push 之後的 `mov %rsp, %rbp` 才建立 frame pointer |
| Red zone 從進入點 stack pointer 量起 | 「stack pointer 之下」216 位元組 | 在 88 位元組的 `sub` 之後，目前 stack pointer 之下 128 位元組 |
| Shadow store 被當作區域變數 | Stack 引數在 8 到 32 | Microsoft 的 stack 引數從 40 開始 |
| `lea` 被當成 load | Variadic callee 中位於 `0x60(%rsp)` 的 stack 引數 | `lea` 計算 `overflow_arg_area`；它不載入任何東西 |
| 隱藏指標被讀成一般引數 | `big_ret(void *, long long)` | 指標被透過它存入，並複製到 `rax`、`rax`，或停留在 `x8` |
| 最深的 CFA 與 prologue frame 比較 | 對 push 外送引數的 caller 回報不一致 | 與本文到達的最深 stack pointer 比較 |
| 從 ISA 斷言慣例 | 對 COFF object 斷言 `amd64` | 格式選擇規則書，然後 ISA 選擇暫存器 |
| 信任單一個工具的 prototype | 三個 double，或十六個 variadic 參數 | 原始碼簽章或第二個推導必須一致 |

## 營運規則 (Operational Rule)

在第一條指令之前先讀 header，並說出規則書。把每個被儲存的暫存器與每個引數，放在距離進入點 stack pointer 的 offset，並對照把它放在那裡的規則，以及編譯器為同一個 frame 寫下的 unwind 資料來檢查。在位元組或原始碼確認之前，把工具的 prototype 視為主張。

## 延伸閱讀 (Further Reading)

- [System V Application Binary Interface, AMD64 Architecture Processor Supplement, Version 1.0 (sections 3.2.1 to 3.2.3, 4.2.4)](https://gitlab.com/x86-psABIs/x86-64-ABI)
- [Microsoft x64 calling convention](https://learn.microsoft.com/en-us/cpp/build/x64-calling-convention)
- [Microsoft x64 stack usage](https://learn.microsoft.com/en-us/cpp/build/stack-usage)
- [Microsoft x64 prolog and epilog](https://learn.microsoft.com/en-us/cpp/build/prolog-and-epilog)
- [Microsoft x64 exception handling: RUNTIME_FUNCTION, UNWIND_INFO, UNWIND_CODE](https://learn.microsoft.com/en-us/cpp/build/exception-handling-x64)
- [Arm Procedure Call Standard for the Arm 64-bit Architecture (AAPCS64), release 2025Q4](https://github.com/ARM-software/abi-aa/blob/main/aapcs64/aapcs64.rst)
- [Apple: Writing ARM64 code for Apple platforms](https://developer.apple.com/documentation/xcode/writing-arm64-code-for-apple-platforms)
- [DWARF Debugging Information Format Version 5, section 6.4 Call Frame Information](https://dwarfstd.org/doc/DWARF5.pdf)
- [Clang driver frame-pointer defaults, `useFramePointerForTargetByDefault` in `CommonArgs.cpp`](https://github.com/llvm/llvm-project/blob/main/clang/lib/Driver/ToolChains/CommonArgs.cpp)
- [Clang command line reference: `-fomit-frame-pointer`, `-momit-leaf-frame-pointer`](https://clang.llvm.org/docs/ClangCommandLineReference.html)
- [Fedora 38 change: frame pointers by default](https://fedoraproject.org/wiki/Changes/fno-omit-frame-pointer)
- [Ubuntu 24.04: performance engineering with frame pointers by default](https://ubuntu.com/blog/ubuntu-performance-engineering-with-frame-pointers-by-default)
- [Ghidra compiler specification: `x86-64-gcc.cspec`](https://github.com/NationalSecurityAgency/ghidra/blob/master/Ghidra/Processors/x86/data/languages/x86-64-gcc.cspec)
- [Ghidra compiler specification: `x86-64-win.cspec`](https://github.com/NationalSecurityAgency/ghidra/blob/master/Ghidra/Processors/x86/data/languages/x86-64-win.cspec)
- [Ghidra compiler specification: `AARCH64.cspec`](https://github.com/NationalSecurityAgency/ghidra/blob/master/Ghidra/Processors/AARCH64/data/languages/AARCH64.cspec)
- [Hex-Rays: SetType and the `__usercall` convention](https://docs.hex-rays.com/ida-actions/settype)
- [Hex-Rays: Decompile as call](https://docs.hex-rays.com/ida-actions/hx_userdefinedcall)
- [Binary Ninja API 6.0: `callingconvention` module](https://api.binary.ninja/binaryninja.callingconvention-module.html)
- [Binary Ninja API 6.0: `platform` module](https://api.binary.ninja/binaryninja.platform-module.html)
- [Binary Ninja user documentation: platform types](https://docs.binary.ninja/guide/types/platformtypes.html)
- [The Official Radare2 Book: calling conventions](https://book.rada.re/analysis/calling_conventions.html)
- [The Official Radare2 Book: code analysis](https://book.rada.re/analysis/code_analysis.html)
- [angr documentation: decompiler analysis passes, including calling convention recovery](https://docs.angr.io/en/latest/analyses/decompiler.html)
- [radare2 6.2.4 release](https://github.com/radareorg/radare2/releases/tag/6.2.4)
- [rizin 0.9.1 release](https://github.com/rizinorg/rizin/releases/tag/v0.9.1)
