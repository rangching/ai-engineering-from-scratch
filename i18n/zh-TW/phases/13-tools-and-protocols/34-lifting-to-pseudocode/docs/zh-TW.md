# Lift 成 Pseudocode：從零開始的 IR、SSA 與結構化 (Lifting to Pseudocode: IR, SSA, and Structuring From Scratch)

> 反編譯器的輸出是一項帶有隱藏推導的主張。當你自己建構這個推導時，輸出的每一行都成為可以檢查的東西。

**Type:** Build
**Languages:** Python, C
**Prerequisites:** Phase 13 · 32 (ELF from scratch), Phase 13 · 33 (disassembly and control flow)
**Time:** ~150 分鐘

## 學習目標 (Learning Objectives)

- 把已解碼的 x86-64 指令 lift 成小型 IR，其中明確表示 flag、load、store、branch、call 與 return。
- 把 frame slot 變成區域變數，並把 System V 的引數暫存器變成參數。
- 在 dominance frontier 上以 phi 函式建構最小 SSA，然後做替換、簡化與移除 dead code。
- 利用 post-dominator 與 back edge，從 control-flow graph 還原 `if`、帶 `else` 的 `if`、`while` 與 `do while`。
- 把乘以 `0x66666667` 說明為除以 10，並證明該恆等式。
- 逐行閱讀 Ghidra 對相同函式產生的 pseudocode，並為每個差異指出背後的指令或猜測。

## 問題 (The Problem)

你向 rea 要 `ledger-x86_64-O1.o` 中 `percent_change` 的 pseudocode。Ghidra 12.1.4 在大約十秒內回答：

```c
undefined1  [16] percent_change(int param_1,int param_2,ulong param_3)

{
  long lVar1;
  undefined1 auVar2 [16];
  undefined1 auVar3 [16];
  
  if (param_1 != 0) {
    lVar1 = (long)((param_2 - param_1) * 100);
    auVar3._0_8_ = lVar1 / (long)param_1 & 0xffffffff;
    auVar3._8_8_ = lVar1 % (long)param_1 & 0xffffffff;
    return auVar3;
  }
  auVar2._8_8_ = 0;
  auVar2._0_8_ = param_3;
  return auVar2 << 0x40;
}
```

原始碼只有四行。它接收兩個 `int` 值並回傳一個 `int`。Ghidra 回報 16 位元組的回傳值，以及原始碼中沒有的第三個參數。這十六個位元組的機器碼是正確的，而且 Ghidra 把每一個都正確解碼了。簽章是一項猜測，而文字中沒有任何地方標記它是猜測。

rea 把這段文字作為 `procedure_pseudo_code` 結果回傳，並附上一項限制。該限制寫著「Pseudocode 是 Ghidra 反編譯器的輸出，不是原始碼，也與 Hopper 的輸出在文字上不等價」。`compare_functions` tool 把 pseudocode 當作文字行來做 diff。它自己的維度紀錄說「Assembly 是不透明的 provider 文字」，而 relocation 正規化不可用。rea 無法看進文字裡面，只讀文字的 agent 也一樣。

rea 的 DX-Ball 紀錄顯示了代價。第一個反編譯器結果是呼叫 `__ftol()` 的 `FUN_00406400(void)`。指令視圖顯示了 stack 輸入與 x87 表達式。Agent 之所以能繼續，只是因為兩個視圖不一致。

這堂課建構反編譯器所隱藏的推導。你從課程 33 解碼的 56 條指令開始，把它們 lift 成 IR、重新命名成 SSA、簡化並結構化。然後你把自己的 pseudocode，與 Ghidra 對 `percent_change`、`grade` 和 `total` 的版本並排放在一起，並說明每一個差異。測試會用 clang 編譯你的輸出，並在一組輸入格上，與原始碼比較。

```figure
lifting-stages-percent-change
```

## Fixture 與 Oracle (The Fixture and the Oracle)

`code/fixtures/ledger.c` 是課程這個部分共用的 fixture：

```c
int percent_change(int before, int after) {
    if (before == 0) return 0;
    return (after - before) * 100 / before;
}

int grade(int score) {
    switch (score / 10) {
    case 10:
    case 9: return 4;
    case 8: return 3;
    case 7: return 2;
    case 6: return 1;
    default: return 0;
    }
}

int total(const int *values, int count) {
    int sum = 0;
    for (int i = 0; i < count; i++) sum += values[i];
    return sum;
}
```

`grade_label` 也在檔案與 object 中，demo 也會 lift 它。上面三個函式是這堂課測試的。

這個 object 是用 `clang -target x86_64-unknown-linux-gnu -O1 -fno-asynchronous-unwind-tables -c ledger.c` 建構的。`-O0` 的 object 放在旁邊，供 stack-slot 那一節使用。`code/fixtures/README.md` 列出兩個指令、每個 object 的 SHA-256，以及產生 Ghidra fixture 的 rea 指令。在固定的 toolchain 上，建構結果可逐位元組重現。

Decoder 的 oracle 是 `llvm-objdump -d -r`。兩個 object 的文字都以 `ledger-x86_64-O1.objdump.txt` 與 `ledger-x86_64-O0.objdump.txt` 提交。測試會解析這些檔案，並依位址、位元組與 AT&T 文字，比較每一條指令。這涵蓋 `-O1` 的全部 56 條指令，以及 `-O0` 的每一條指令。當 `llvm-objdump` 安裝在固定的路徑時，第二個測試會即時執行它，並再比較一次。

這是 oracle 印出的 `percent_change`：

```text
0000000000000000 <percent_change>:
       0: 85 ff                        	testl	%edi, %edi
       2: 74 09                        	je	0xd <percent_change+0xd>
       4: 29 fe                        	subl	%edi, %esi
       6: 6b c6 64                     	imull	$0x64, %esi, %eax
       9: 99                           	cltd
       a: f7 ff                        	idivl	%edi
       c: c3                           	retq
       d: 31 c0                        	xorl	%eax, %eax
       f: c3                           	retq
```

## 階段 1：從位元組到指令 (Stage 1: Bytes to Instructions)

課程 33 建構了 decoder。這堂課保留一份小的副本，讓程式碼能獨立執行。這份副本只涵蓋 clang 在 `-O0` 與 `-O1` 為 `ledger.c` 送出的 opcode。對任何其他東西，它會丟出帶有位址與 opcode 位元組的 `DecodeError`。

Decoder 遵循 Intel SDM Volume 2 2.1 節的位元組順序。Prefix 在最前面，然後是 opcode、ModR/M、SIB、displacement，最後是 immediate。2.2.1 節加入 REX prefix，是從 `0x40` 到 `0x4F` 的一個位元組，必須直接位於 opcode 之前。REX.W 選擇 64 位元 operand，REX.R 擴充 `reg` 欄位，REX.X 擴充 SIB index，REX.B 擴充 `r/m` 或 SIB base。2.1.3 節定義三個 ModR/M 欄位 `mod`、`reg` 與 `r/m`，以及三個 SIB 欄位 `scale`、`index` 與 `base`。在 64 位元模式下，`mod = 00` 搭配 `r/m = 101` 表示相對於下一條指令的 32 位元 displacement（2.2.1.6 節）。

| Opcode 位元組 | 形式 | Fixture 範例 |
|---|---|---|
| `01 29 31 39 85 89` | `op Ev, Gv` | `29 fe` `subl %edi, %esi` |
| `03 2b 3b 8b 8d` | `op Gv, Ev` | `03 04 97` `addl (%rdi,%rdx,4), %eax` |
| `63` 搭配 REX.W | `movsxd Gv, Ev` | `48 63 c7` `movslq %edi, %rax` |
| `69` 和 `6b` | `imul Gv, Ev, imm` | `48 69 c0 67 66 66 66` |
| `83 /0 /5 /7` | `add sub cmp Ev, imm8` | `83 c1 fa` `addl $-0x6, %ecx` |
| `c1 /5 /7` | `shr sar Ev, imm8` | `48 c1 f9 22` `sarq $0x22, %rcx` |
| `f7 /7` | `idiv Ev` | `f7 ff` `idivl %edi` |
| `ff /0 /4` | `inc Ev`、`jmp Ev` | `48 ff c2` `incq %rdx` |
| `70` 到 `7f`、`eb`、`e8` | `jcc rel8`、`jmp rel8`、`call rel32` | `74 09` `je 0xd` |
| `99`、`c3`、`90`、`0f 1f` | `cltd`、`retq`、`nop` | `66 0f 1f 44 00 00` `nopw (%rax,%rax)` |

`grade+0x23` 的 `leaq (%rip), %rcx` 是七個位元組，displacement 為零。Object 在 offset `0x36`（也就是 displacement 欄位）帶有 relocation `R_X86_64_PC32.rodata+0x10`。Lifter 在階段 3 解析它。

## 階段 2：Block 與 Dominator (Stage 2: Blocks and Dominators)

Basic block 是一串 statement，在頂端有一個進入點，在底端有一個離開點。那是 Allen（1970）的定義，而每篇反編譯器論文都沿用它。`build_blocks()` 收集 leader：函式進入點、函式內的每個跳轉目標，以及每個跳轉或 return 之後的位址。每個 block 從它的最後一條指令取得後繼者。條件跳轉有兩個後繼者，`jmp` 有一個。`ret` 沒有，而以一般指令結尾的 block 會 fallthrough 到下一個 leader。

`-O1` 的 `total` 有五個 block：

| Block | 指令 | 後繼者 |
|---|---|---|
| `0x40` | `testl %esi, %esi` 然後 `jle 0x5c` | `0x5c`、`0x44` |
| `0x44` | `movl %esi, %ecx`、`xorl %edx, %edx`、`xorl %eax, %eax`、`nopw` | `0x50` |
| `0x50` | `addl (%rdi,%rdx,4), %eax`、`incq %rdx`、`cmpq %rdx, %rcx`、`jne 0x50` | `0x50`、`0x5b` |
| `0x5b` | `retq` | 離開 |
| `0x5c` | `xorl %eax, %eax` 然後 `retq` | 離開 |

`immediate_dominators()` 是 Cooper、Harvey 與 Kennedy（2001）的迭代演算法。它依反向 postorder 走訪 block，並將已處理前驅的 dominator 取交集，直到沒有任何改變。在這麼小的圖上，它兩趟就能完成。對 `total` 而言，它說 `0x44` dominate `0x50`。它也說 `0x50` 透過 back edge dominate 它自己，這就是階段 8 找出迴圈的方式。

同一個函式再執行兩次。`dominance_frontiers()` 在階段 5 的 SSA 中使用它。`post_dominators()` 在反轉的圖上加上虛擬離開節點來執行它，以在階段 8 找出 join。

## 階段 3：Lift 成小型 IR (Stage 3: Lifting to a Small IR)

IR 有六種 statement：`Assign`、`Store`、`Branch`、`Jump`、`IndirectJump` 和 `Return`。表達式是常數、變數、帶寬度的二元運算、cast（`sext`、`zext`、`trunc`）以及 load。另外還有五種形狀，只是為了之後被簡化掉而存在：`Flags`、`Cond`、`Concat`、`SignFill` 和 `Symbolic`。

暫存器是以其 64 位元家族命名的變數。對 `%eax` 的寫入，是對 `rax` 寬度 32 的 `Assign`，而對 `%eax` 的讀取，是對 `rax` 寬度 32 的讀取。在 64 位元模式下，32 位元的寫入會 zero-extend 到完整暫存器，所以之後對相同值的 64 位元讀取是 `zext`。Lifter 不知道到達的定義，所以它記錄讀取寬度，並讓 SSA 在階段 5 插入 cast。

| 指令 | IR statement |
|---|---|
| `subl %edi, %esi` | `flags = cmp(esi, edi)` 然後 `esi = esi - edi` |
| `testl %edi, %edi` | `flags = test(edi, edi)` |
| `je 0xd` | `if je(flags) goto 0xd else goto 0x4` |
| `xorl %eax, %eax` | `eax = 0` |
| `cltd` | `edx = signfill(eax)` |
| `idivl %edi` | `dividend = edx:eax`、`eax = dividend / edi`、`edx = dividend % edi` |
| `movslq %edi, %rax` | `rax = sext64(edi)` |
| `sarq $0x22, %rcx` | `rcx = rcx >>s 34` |
| `shrq $0x3f, %rax` | `rax = rax >>u 63` |
| `addl (%rdi,%rdx,4), %eax` | `flags = add(eax, load32(rdi + rdx * 4))` 然後 `eax = eax + load32(rdi + rdx * 4)` |
| 帶 relocation 的 `leaq (%rip), %rcx` | `rcx = &.rodata+0x14` |
| `pushq %rbp`、`movq %rsp, %rbp`、`subq $0x10, %rsp`、`popq %rbp` | 無，frame 簿記 |
| `retq` | `return rax` |

### Flag 是一個變數

每一條設定 condition code 的指令，都寫入一個名為 `flags` 的變數，而每個條件跳轉都讀取它。`cmp a, b` 寫入 `cmp(a, b)`，`test a, b` 寫入 `test(a, b)`，而 `sub` 則以副作用的方式寫入相同的 `cmp` 形狀。`jcc` 變成 `Branch(Cond(cc, flags), target, fallthrough)`。當兩者在階段 6 相遇時，condition code 決定比較運算子：

| Code | 在 `cmp(a, b)` 之後 | 在 `test(a, a)` 之後 |
|---|---|---|
| `je` | `a == b` | `a == 0` |
| `jne` | `a != b` | `a != 0` |
| `jl`、`jge`、`jle`、`jg` | 有號的 `<`、`>=`、`<=`、`>` | 與 0 的有號比較 |
| `jb`、`jae`、`jbe`、`ja` | 無號的 `<`、`>=`、`<=`、`>` | 與 0 的無號比較 |

`xorl %eax, %eax` 直接被 lift 成 `eax = 0`。它所設定的 flag 被寫成 `add` 風格的 flag，而 fixture 中沒有任何跳轉讀取它們。`xor` 慣用寫法在 `grade` 中很重要，Ghidra 在那裡把它弄丟了。

### cltd 與 idiv

`cltd` 用 `%eax` 的符號位元填滿 `%edx`。`idivl %edi` 以 `%edi` 除 64 位元的 `%edx:%eax` 配對，把商放進 `%eax`，把餘數放進 `%edx`。Lifter 先把這一對寫進名為 `dividend` 的暫存變數。餘數 statement 接著讀取舊的 `%eax`，而不是新的商。在 SSA 中，高半部顯示為低半部的 `signfill`。簡化器把這一對化簡為低半部，而除法變成 32 位元的 `/`。

### -O0 的 stack slot

在 `-O0`，clang 把每個區域變數保留在 frame 中。`percent_change` 把 `%edi` 存到 `-0x8(%rbp)`，把 `%esi` 存到 `-0xc(%rbp)`，然後在每次使用時讀回來。Lifter 注意 `pushq %rbp` 後面接著 `movq %rsp, %rbp`。在這個 prologue 之後，base 為 `%rbp`、沒有 index 且有 displacement 的記憶體 operand，會變成名為 `local_<hex>` 的變數。`-0x8(%rbp)` 是 `local_8`，`-0xc(%rbp)` 是 `local_c`。

Frame 指令本身不產生 IR，對 `%rsp` 的 `subq` 和 `addq` 也不會。任何其他記憶體 operand 仍是 `Load` 或 `Store`。

### Relocation

`grade` 中的 `leaq (%rip), %rcx` 在檔案中的 displacement 是零。位於 offset `0x36` 的 relocation 寫著 `R_X86_64_PC32.rodata+0x10`。Linker 會把 `S + A - P` 寫進該欄位。`S` 是 `.rodata` 的位址，`A` 是 addend `0x10`，`P` 是欄位的位址。指令接著計算 `next_ip + disp = (P + 4) + S + A - P = S + A + 4`。

Lifter 在沒有 linker 的情況下套用相同的算術。目標 offset 是 addend 加上從欄位到指令結尾的距離，`0x10 + 4 = 0x14`。IR 寫成 `rcx = &.rodata+0x14`，而那裡的表格持有五個 `int` 值 `1, 2, 3, 4, 4`。

## 階段 4：來自呼叫慣例的參數 (Stage 4: Parameters From the Calling Convention)

System V AMD64 psABI 的 3.2.3 節，依序以 `%rdi`、`%rsi`、`%rdx`、`%rcx`、`%r8`、`%r9` 傳遞 INTEGER 類別的引數。結果透過 `%rax` 然後 `%rdx` 回傳。3.2.1 節說 `%rbp`、`%rbx` 以及 `%r12` 到 `%r15` 屬於 caller，且必須被保留。其他一切都屬於 callee。

`collect_params()` 套用一條規則：在函式內先被讀取、之後才被寫入的引數暫存器，是參數。最高的這種暫存器決定數量，所以讀取 `%rsi` 而從不讀 `%rdi` 的函式，仍然有兩個參數。`percent_change` 讀取 `%edi` 和 `%esi`，所以它有 `param_1` 和 `param_2`。`grade` 只讀取 `%edi`。`total` 以 64 位元 base 讀取 `%rdi`，並以 32 位元 count 讀取 `%esi`。

現在 Ghidra 的 `param_3` 有了解釋。`idivl` 寫入 `%rdx`，而 psABI 允許在 `%rax:%rdx` 中回傳 128 位元的 INTEGER。在 object file 中，沒有 caller 可以說明回傳值有多寬。Ghidra 把兩個暫存器都保留為回傳值。

在零的路徑上，`xorl %eax, %eax` 只寫入 `%eax`。Ghidra 在那裡把 `%rdx` 讀成函式的輸入，而該輸入就變成了 `param_3`。指令是對的。簽章則是對指令留白部分的猜測。

## 階段 5：SSA，每個定義一個名稱 (Stage 5: SSA, One Name per Definition)

Lift 之後，`grade` 寫入 `rax` 四次、寫入 `rcx` 三次。Cytron、Ferrante、Rosen、Wegman 與 Zadeck（1991）定義了 static single assignment 形式。每個變數只被指派一次，而位於 join point 的 phi 函式，會從到達那裡的定義中選擇。他們的建構有兩個步驟。5.1 節用 dominance frontier 放置 phi 函式，5.2 節則以走訪 dominator 樹的方式重新命名每個定義與使用。`insert_phis()` 和 `rename()` 遵循這兩節。

`total` 的 SSA 清單顯示這個建構產生了什麼：

```text
block 0x44 -> 0x50
  ecx_1 = esi_0
  edx_1 = 0
  eax_1 = 0
block 0x50 -> 0x50, 0x5b
  eax_2 = phi(0x44: eax_1, 0x50: eax_3)
  rdx_2 = phi(0x44: edx_1, 0x50: rdx_3)
  flags_2 = phi(0x44: flags_1, 0x50: flags_5)
  flags_3 = add(eax_2, load32(rdi_0 + rdx_2 * 4))
  eax_3 = eax_2 + load32(rdi_0 + rdx_2 * 4)
  flags_4 = inc(rdx_2, 1)
  rdx_3 = rdx_2 + 1
  flags_5 = cmp(zext64(ecx_1), rdx_3)
  if jne(flags_5) goto 0x50 else goto 0x5b
```

暫存器的版本 0 是它在進入點的值。`esi_0` 和 `rdi_0` 是參數。迴圈 header `0x50` 有兩個前驅，也就是設定 block 與它自己，所以每個在迴圈中有定義的變數，都在那裡得到一個 phi。`flags_2` 的 phi 一開始就是 dead 的。最小 SSA 之所以放置它，是因為 `flags` 在兩個前驅中都有定義，而階段 6 把它移除，因為沒有任何東西讀取它。

重新命名的步驟也會修正寬度。`edx_1 = 0` 是 32 位元的定義，而 `rdx_3 = rdx_2 + 1` 是 64 位元，所以 phi `rdx_2` 取較寬的寬度。`cmpq %rdx, %rcx` 中對 `%rcx` 的讀取是 64 位元，而它的定義 `ecx_1 = esi_0` 是 32 位元，所以這個使用變成 `zext64(ecx_1)`。最終 pseudocode 中的每個 cast 都來自這一條規則。

「lite」的部分在於離開 SSA 的方式。`coalesce_webs()` 把每個 phi 與它的引數連成一個變數（稱為 web），並給這個 web 一個名稱。這對編譯器為這些函式送出的形狀有效。它可能在 parallel copy 上失敗，也就是一個 join 上的兩個 phi 互換值，而這堂課不處理那種情況。Cytron 的第 7 節描述了經由著色的一般性離開方式。

## 階段 6：簡化、替換、移除 dead code (Stage 6: Simplify, Substitute, Remove Dead Code)

`optimize()` 重複三趟，直到沒有任何改變。簡化器依這些規則由下往上改寫表達式：

| 之前 | 之後 |
|---|---|
| `Cond(je, cmp(a, b))` | `a == b`，其他九個 code 也一樣 |
| `Cond(jle, test(a, a))` | `a <= 0` |
| `x + (-6)` | `x - 6` |
| `trunc32(zext64(x))`，其中 `x` 是 32 位元 | `x` |
| 64 位元 operand 上的 `trunc32(a + b)` | `trunc32(a) + trunc32(b)` |
| `(signfill(x): x) / d` | 寬度 32 的 `x / d` |
| `load32(base + idx * 4)` | `base[idx]` |
| `trunc32(p >>s 34) + trunc32(p >>u 63)`，其中 `p = sext64(x) * 0x66666667` | `x / 10` |

Forward substitution 把定義移到它唯一的使用處。規則有三個條件。使用處在同一個 block 內。使用處是 statement 而不是 phi。當定義包含 load 時，定義與使用之間沒有 store 或 call。常數或 section 位址可以替換到任何地方，除了 phi，因為 phi 引數必須保持為 web 的變數。

Dead code 的移除，會刪除目的地沒有使用、且表達式沒有 call 的 assignment 或 phi。這些 pass 的順序很重要。在 `percent_change` 中，餘數 `edx_2 = dividend_1 % edi_0` 是 dead 的，所以它先被移除。這讓 `edx_1 = signfill(eax_1)` 只剩一個使用，因此它被替換進 `dividend_1 = edx_1:eax_1`。

接著配對規則把 `signfill(eax_1):eax_1` 化簡為 `eax_1`，而除法變成 `eax_1 / edi_0`。現在 `eax_1 = esi_1 * 100` 只有一個使用，`esi_1 = esi_0 - edi_0` 也只有一個使用，兩者都移進 return statement。`flags` 的定義已被替換進分支條件並消失。四條指令變成一行：

```c
return (param_2 - param_1) * 100 / param_1;
```

### 0x66666667 的乘法

`grade` 把分數除以 10，而 clang 沒有送出 `idiv`。它送出 `imulq $0x66666667, %rax, %rax`，然後對一份副本做 `sarq $0x22, %rcx`，再做 `shrq $0x3f, %rax`，最後把兩者相加。Granlund 與 Montgomery 的〈Division by Invariant Integers using Multiplication〉（PLDI 1994）描述了這個方法。編譯器挑選乘數 `m` 與 shift `s`。對範圍內的每個 `x`，`(x * m) >> s` 必須等於 `x / d`。

這裡 `m = 0x66666667 = 1717986919` 且 `s = 34`。算術驗證通過：`2^34 / 10 = 1717986918.4`，其 ceiling 是 `1717986919`。誤差是 `m * 10 - 2^34 = 6`。要讓恆等式對每個 32 位元 `x` 都成立，誤差乘以 `2^31` 必須小於 `2^34`，所以誤差必須最多為 8。六不超過八，所以 `(x * m) >> 34` 對每個非負的 `int` 都等於 `floor(x / 10)`。

對負的 `x`，floor 往遠離零的方向捨入，而 C 往零的方向捨入。`shrq $0x3f` 取出乘積的符號位元，當 `x` 為負時它恰好是 1。加上它就能修正捨入。

`magic_divisor(m, s)` 測試一個候選配對。它把 `d` 計算為 `2^s` 除以 `m` 的整數商，並嘗試 `d` 與 `d + 1`。只有當 `m` 等於 `2^s / d` 的 ceiling 且誤差上限成立時，它才接受某個 `d`。相同的乘數搭配 `s = 33` 得到 `d = 5`，這是正確的，因為 `0x66666667` 也是 `2^33 / 5` 的 ceiling。測試以 2,406 個值（包括 `int` 範圍的兩端），對照 `c_div(x, 10)` 檢查這個恆等式。

開啟辨識後，`grade` 讀作 `ecx_1 = param_1 / 10 - 6;`。關閉辨識後，demo 印出原始形式：

```c
rax_1 = (long)param_1 * 0x66666667;
ecx_1 = (int)(rax_1 >> 34) + (int)((unsigned long)rax_1 >> 63) - 6;
```

兩者都能編譯，也都算出相同的值。保留這個開關。當你閱讀來自你沒研究過的編譯器的程式碼時，原始形式才是誠實的。

## 階段 7：從使用推導型別 (Stage 7: Types From Use)

機器碼有寬度，沒有型別。`infer_types()` 把三種證據變成 C 型別。定義寬度為 32 位元的 web 是 `int`，64 位元的 web 是 `long`。作為 4 位元組元素的元素存取 `base[idx]` 的 base 的 web，是 `int *`。指標型別沿著 copy 與 phi 擴散，所以 `-O0` 的 `local_8 = param_1` 讓兩者都成為 `int *`。回傳型別是 `return` statement 中值的寬度。

`total` 的結果是 `int total(int *param_1, int param_2)`。Ghidra 寫成 `int total(long param_1,uint param_2)`。兩者都與指令一致。Lifter 把有 scale 的 load 當作指標的證據。Ghidra 保留暫存器寬度，並把存取寫成 `*(int *)(param_1 + uVar2 * 4)`。兩者都看不到 `const`，也都分不出 `int *` 與 `unsigned *`。

Lee、Avgerinos 與 Brumley（TIE，NDSS 2011）把型別還原構想為在 lattice 上的限制求解。結果是上界與下界，而不是單一個猜測。Noonan、Loginov 與 Cok（Retypd，PLDI 2016）加入了 polymorphism 與 recursion。他們回報，從編譯器留下的慣用寫法中，`const` 指標標註有 98% 的 recall。這堂課使用這條研究路線中最便宜的兩條規則，並指出它們看不到的東西。

無號算術保留在表達式中，而不在變數型別中。`cmpl $0x4, %ecx` 之後的 `ja` 印成 `(unsigned int)ecx_1 > 4`，而 `shrq` 印成 `(unsigned long)rax_1 >> 63`。變數保持有號，讓讀者看到暫存器的值，以及指令將它視為無號的那一個地方。

## 階段 8：從圖進行結構化 (Stage 8: Structuring From the Graph)

Cifuentes（1994）的第 6 章，把圖的區域與高階語言的形狀做比對。先是迴圈（6.6.1），再是雙向條件式（6.6.2），然後是 n 向條件式（6.6.3）。Yakdan、Eschweiler、Gerhards-Padilla 與 Smith（NDSS 2015）走了另一條路。他們推導出每個節點執行的條件，並從這些條件建構結構，不使用固定的形狀。他們的動機是一項量測。Hex-Rays 在一個樣本上，於 49,514 行中產生了 1,571 個 `goto` statement，每 32 行一個，而他們的結構化在 1,530 個 coreutils 函式上一個都沒有產生。

`structure()` 是第一個家族中的小成員。它需要階段 2 的兩項事實。以 `Branch` 結尾的 block 的 immediate post-dominator，是該條件式的 join。Back edge 是指向 dominate 其來源的 block 的 edge。自然迴圈是在不經過 header 的情況下能到達 back edge 來源的 block 集合。

對於 join 為 `j` 的 `Branch`：

- 如果 `j` 是跳轉目標，則送出 `if (!cond) { fallthrough region }` 並繼續於 `j`。
- 如果 `j` 是 fallthrough，則送出 `if (cond) { target region }` 並繼續於 `j`。
- 如果 `j` 是函式離開，則兩個分支都 return。把較小的分支送出為 `if (cond) {... }`，另一個分支放在它之後，沒有 `else`。這就是 `percent_change` 和 `total` 中的 early return。
- 否則送出 `if (cond) { target } else { fallthrough }` 並繼續於 `j`。這是 `-O0` 的 `percent_change`，兩個分支都儲存 `local_4`，並在 epilogue 相遇。

對於迴圈 header：

- 如果 header 以 `Branch` 結尾，且有一個後繼者在迴圈之外，則送出 `while (cond) { body }`。`-O0` 的 `total` 是這種形狀：header 比較 `local_14` 與 `local_c`，並在 `jge` 時離開。
- 如果 header 同時也是帶有 back edge 的 block，或是 latch 負責離開，則送出 `do { body } while (cond);`。`-O1` 的 `total` 是這種形狀：block `0x50` 做加法、遞增、比較，並跳回自己。

`IndirectJump` 以沒有後繼者的方式結束一個 block。`-O0` 的 `grade` 是真正的 jump table，透過位於 `.rodata+0x0` 的五個相對 offset 做 `jmpq *%rax`，而課程 33 解析了那些目標。在這裡，lifter 改為印出一則註記。位於 `0x76`、`0x7f`、`0x88` 和 `0x91` 的四個 block 只能透過 indirect jump 到達，且沒有被 lift。實作練習會加入 `switch`。

### -O1 對 switch 做了什麼

在 `-O1`，clang 沒有為 `grade` 保留 jump table。它計算 `score / 10 - 6`，並把結果當作無號數與 4 比較。接著它從位於 `.rodata+0x14` 的五個項目 `int` 表格載入回傳值。無號比較是一條指令，同時涵蓋 `score / 10 < 6` 與 `score / 10 > 10`。負的差值是很大的無號值，所以它同樣無法通過檢查。

表格為商 6 到 10 持有 `1, 2, 3, 4, 4`。帶有五個 `return` statement 的 `switch`，變成了 `if` 加上陣列讀取。Lifter 恰好送出這個，並依其 section offset 把表格命名為 `lookup_14`。

表格的長度是啟發式的。`table_values()` 從表格起點，一直讀到任何 `.text` relocation 所參照的下一個 `.rodata` offset，或 section 的結尾。在這個 object 中，下一個被參照的 offset 是 `0x28`，也就是 `grade_label` 的表格，所以 `lookup_14` 有 `(0x28 - 0x14) / 4 = 5` 個項目。與 4 的無號比較也同意這點。Ghidra 同樣不知道長度。它寫成 `*(uint *)(&DAT_001000b4 + (ulong)uVar2 * 4)`，也就是位元組位址加上 offset，並把那五個值留給讀者自己去查。

## 與 Ghidra 並排比較 (Side by Side With Ghidra)

Demo 印出每一對。Ghidra 的文字來自 `code/fixtures/ghidra/*.json`，也就是 rea 在 2026-10-14 回傳的 Evidence 紀錄。指令是 `npx -y rea-agents@6.5.0 decompile ledger-x86_64-O1.o <function> --provider ghidra --format json`。每筆紀錄都帶有 `confidence: observed` 與 `authority: shipped-artifact`。文字是被觀察到的。文字內的主張是推導出來的，而紀錄沒有說明是哪些。

### percent_change

```text
lifter                                         | ghidra
-----------------------------------------------+-----------------------------------------------
int percent_change(int param_1, int param_2)   | undefined1  [16] percent_change(int param_1,int param_2,ulong param_3)
{                                              | {
  if (param_1 == 0) {                          |   long lVar1;
    return 0;                                  |   undefined1 auVar2 [16];
  }                                            |   undefined1 auVar3 [16];
  return (param_2 - param_1) * 100 / param_1;  |   if (param_1 != 0) {
}                                              |     lVar1 = (long)((param_2 - param_1) * 100);
                                               |     auVar3._0_8_ = lVar1 / (long)param_1 & 0xffffffff;
                                               |     auVar3._8_8_ = lVar1 % (long)param_1 & 0xffffffff;
                                               |     return auVar3;
                                               |   }
                                               |   auVar2._8_8_ = 0;
                                               |   auVar2._0_8_ = param_3;
                                               |   return auVar2 << 0x40;
                                               | }
```

| 差異 | 指令或猜測 |
|---|---|
| 回傳型別 `undefined1 [16]` 對 `int` | 猜測。`idivl` 寫入 `%rax` 和 `%rdx`，而沒有 caller 決定回傳寬度。兩個暫存器都被保留。 |
| `param_3` | 從第一項推得的猜測。在零的路徑上只有 `%eax` 被寫入，所以 `%rdx` 被讀成輸入。 |
| `lVar1 / (long)param_1 & 0xffffffff` 對 `/ param_1` | 指令。`cltd` 加 `idivl` 是對符號擴充的 32 位元值做 64 位元除法，再截斷成 32 位元。兩段文字都描述了它。我們的用 `signfill` 規則把這一對化簡了。 |
| 保留 `lVar1 % (long)param_1` | 由回傳型別推得。`%rdx` 中的餘數在我們的分析中是 dead 的，在 Ghidra 的分析中是 alive 的。 |
| `if (param_1 != 0)`，零的情況在它之後 | 指令順序。`je 0xd` 跳到零的 return。Ghidra 把計算的分支放前面，我們把較小的分支放前面。兩者都正確。 |
| `return auVar2 << 0x40` | 猜測。16 位元組的值，低半部是 `param_3`，被 shift 到高半部，是反編譯器表達「`%rax = 0` 而 `%rdx` 未知」的方式。 |

`code/fixtures/ghidra/ledger-x86_64-O1.ghidra-repaired.c` 中修復後的版本只編輯這個函式，測試則在 81 組輸入配對上，將它與原始碼比較。

### grade

```text
lifter                                         | ghidra
-----------------------------------------------+-----------------------------------------------
static const int lookup_14[5] = {1, 2, 3, 4, 4}; | uint grade(int param_1)
int grade(int param_1)                         | {
{                                              |   uint uVar1;
  int eax_1;                                   |   uint uVar2;
  int ecx_1;                                   |   
                                               |   uVar2 = param_1 / 10 - 6;
  ecx_1 = param_1 / 10 - 6;                    |   uVar1 = -(param_1 >> 0x1f) ^ (uint)((ulong)((long)param_1 * 0x66666667) >> 0x3f);
  eax_1 = 0;                                   |   if (uVar2 < 5) {
  if ((unsigned int)ecx_1 <= 4) {              |     uVar1 = *(uint *)(&DAT_001000b4 + (ulong)uVar2 * 4);
    eax_1 = lookup_14[(unsigned long)(unsigned int)ecx_1]; |   }
  }                                            |   return uVar1;
  return eax_1;                                | }
}                                              |
```

| 差異 | 指令或猜測 |
|---|---|
| 兩者都是 `param_1 / 10 - 6` | 指令。兩者都辨識了 `0x66666667` 慣用寫法。 |
| `uVar1 = -(param_1 >> 0x1f) ^ (uint)(... >> 0x3f)` 對 `eax_1 = 0` | 指令 `grade+0x1a` 的 `xorl %eax, %eax`。該暫存器是零。Ghidra 把 `xor` 表達成 `param_1` 的符號與乘積符號位元的互斥或，這是同一個位元的兩種寫法，而且沒有互相抵消。對每個 `int`，值都是 0，測試在 141 個分數上證明了這點。信任這段文字的讀者，會相信預設值取決於分數。 |
| `uVar2 < 5` 對 `(unsigned int)ecx_1 <= 4` | 指令 `cmpl $0x4, %ecx` 然後 `ja`。相同的條件。Ghidra 把 `uVar2` 型別設為 `uint`，我們在使用處做 cast。 |
| `*(uint *)(&DAT_001000b4 + (ulong)uVar2 * 4)` 對 `lookup_14[...]` | 指令加 relocation。Ghidra 載入 object 時，`.rodata` 位於 `0x001000a0`，所以 `0x001000b4` 是 `.rodata + 0x14`。它套用了 relocation，並保留位元組位址。我們為表格命名並讀取它的五個值。 |
| `uint` 回傳對 `int` | 兩邊都是猜測。指令只說了 32 位元。 |

### total

```text
lifter                                         | ghidra
-----------------------------------------------+-----------------------------------------------
int total(int *param_1, int param_2)           | int total(long param_1,uint param_2)
{                                              | {
  int eax_1;                                   |   int iVar1;
  int ecx_1;                                   |   ulong uVar2;
  long rdx_1;                                  |   
                                               |   if (0 < (int)param_2) {
  if (param_2 <= 0) {                          |     uVar2 = 0;
    return 0;                                  |     iVar1 = 0;
  }                                            |     do {
  ecx_1 = param_2;                             |       iVar1 = iVar1 + *(int *)(param_1 + uVar2 * 4);
  rdx_1 = 0;                                   |       uVar2 = uVar2 + 1;
  eax_1 = 0;                                   |     } while (param_2 != uVar2);
  do {                                         |     return iVar1;
    eax_1 = eax_1 + param_1[rdx_1];            |   }
    rdx_1 = rdx_1 + 1;                         |   return 0;
  } while ((unsigned long)(unsigned int)ecx_1 != rdx_1); | }
  return eax_1;                                |
}                                              |
```

| 差異 | 指令或猜測 |
|---|---|
| `int *param_1` 對 `long param_1` | 我們這邊從 `addl (%rdi,%rdx,4), %eax` 推論。Ghidra 保留了寬度。 |
| `uint param_2` 與 `(int)param_2` 對 `int param_2` | Ghidra 選擇無號，因為 `cmpq %rdx, %rcx` 比較的是 zero-extend 的 count。我們保留 `int` 並在比較處做 cast。 |
| 兩者都是 `do { } while` | 指令 `jne 0x50`，從 block 到它自己。編譯器旋轉了迴圈。`-O0` 的 object 保留 `while`。 |
| 保留 `ecx_1 = param_2` 對 `param_2 != uVar2` | 我們的替換規則停留在一個 block 內。複製 `movl %esi, %ecx` 在設定 block 中，而比較在迴圈中，所以變數保留。Ghidra 做了傳播。 |
| `(unsigned long)(unsigned int)ecx_1` | 指令。`movl %esi, %ecx` 做 zero-extend，而 64 位元比較讀取完整暫存器。 |

行為測試原封不動地編譯 Ghidra 對 `grade` 和 `total` 的文字。它加上兩行 `typedef`，以及從 `.rodata+0x14` 的二十個位元組得到的 `DAT_001000b4` 定義。兩個函式在輸入格上都與原始碼相符。

## 研究怎麼說閱讀 pseudocode (What the Research Says About Reading Pseudocode)

LLM4Decompile（EMNLP 2024）量測到單獨使用 Ghidra，在 HumanEval-Decompile 上的 re-executability 為 20.12%。大約五分之一的函式能重新編譯，並通過 benchmark 附帶的斷言。用他們 6.7B 的模型改善 Ghidra 的輸出，把它提高到 52.74%。DecompileBench（Findings of ACL 2025）對 130 個程式中的 23,400 個函式，執行了六個工業級反編譯器與六種 LLM 方法。LLM 的輸出可讀性較好，但功能正確性低了 52.2%。

Liu、Raff 與 Micinski（arXiv 2609.05370，2026 年 9 月）解釋了這個落差。他們使用 300 個 GitHub 函式庫函式與 287 個以 CVE 為基礎的函式。最強的改善模型把 Ghidra 的 build 成功率從 75% 提高到 90%，而行為符合率卻從 74% 降到 62%。通過 benchmark 中每項測試的候選，在 fuzz 輸入上仍然有分歧。這在八個系統中佔 4.9% 的案例，而在某一個系統中高達 13%。多達十分之一已揭露的漏洞，在改善後的程式碼中消失了。

他們追溯到的原因很簡單。被捏造出來的欄位、型別、callee 與 guard，取代了反編譯器留在文字中看得見的未知。

這就是 lifter 印出 `(unsigned long)(unsigned int)ecx_1`，而不是更好看的名稱的原因。對於未解析的跳轉，它印出一則註記，而不是 `switch`；它印出 `lookup_14`，而不是模型會捏造的標籤。每一行都有你可以在 `code/main.py` 中讀到的推導。Ghidra 的 `undefined1 [16]` 很醜，但誠實。改善模型乾淨的 `int percent_change(int, int)` 是出於錯誤的理由而正確。同一個模型在下一個函式上，會產生乾淨但錯誤的簽章。

## 動手做 (Build It)

執行 demo：

```bash
cd phases/13-tools-and-protocols/34-lifting-to-pseudocode
python3 code/main.py
```

Demo 解碼 `-O1` object 的全部四個函式，並將每條指令與已提交的 `llvm-objdump` 文字比較。對 `percent_change`、`grade` 和 `total`，它印出 SSA 之前的 IR、帶有 phi 函式的 SSA 清單，以及與 Ghidra 文字並排的 pseudocode。接著它印出關閉除法辨識的 `grade`、`-O0` 版本的 `percent_change`、`total` 和 `grade_label`，以及摘要：

```json
{
  "functions": ["grade", "grade_label", "percent_change", "total"],
  "instructions": 56,
  "blocks": {"percent_change": 3, "grade": 3, "total": 5, "grade_label": 4},
  "phis": {"percent_change": 0, "grade": 1, "total": 2, "grade_label": 1},
  "objdumpDifferences": 0
}
```

依此順序閱讀 `code/main.py`：

1. `read_object()` 讀取 ELF64 可重定位 object 的 section header、symbol table 與 `.rela` section。
2. `decode_one()` 與 `render_att()` 解碼一條指令，並以 `llvm-objdump` 的方式印出它。
3. `build_blocks()`、`immediate_dominators()` 與 `dominance_frontiers()` 建構圖的事實。
4. `lift_insn()` 把一條指令變成 IR statement，並解析 relocation。
5. `insert_phis()` 與 `rename()` 建構 SSA 並插入寬度 cast。
6. `make_simplifier()`、`substitute_once()` 與 `remove_dead_code()` 在 `optimize()` 內執行。
7. `collect_params()`、`coalesce_webs()` 與 `infer_types()` 為 web 命名並設定型別。
8. 搭配 `post_dominators()` 與 `natural_loops()` 的 `structure()` 送出 C 的主體。
9. `lift_function()` 把各階段串在一起，而 `lift_all()` 帶著 callee 簽章 lift 整個 object。

## 實際使用 (Use It)

要從你自己的 object lift 函式，請用 `clang -target x86_64-unknown-linux-gnu -c` 建構它，並呼叫 `lift_all(read_object(path))`。當 decoder 丟出 `DecodeError` 時，從訊息中讀取 opcode 位元組，並在 SDM 的 opcode map 中找到它。然後在檔案開頭的表格中新增一個項目。當 lifter 對某個 mnemonic 丟出例外時，在 `lift_insn()` 中新增一條規則。當 structurer 送出 `goto` 時，用 `IrPrinter` 印出圖，並判斷缺少哪條形狀規則。每次新增都很小，而每一項都由既有的測試，對照 `llvm-objdump` 或 clang 來檢查。

當安裝了 Ghidra 時，對同一個函式執行 rea，並把兩段文字並排。在相信任一段文字之前，先填寫差異表。背後有指令的差異是呈現上的選擇。背後沒有指令的差異是猜測，而這個猜測可能在任何一邊。

## 互動實驗 (Interactive Lab)

### 實驗 A：一個函式經過五個階段

從 `code` 目錄開啟 Python：

```bash
cd phases/13-tools-and-protocols/34-lifting-to-pseudocode/code
python3 -q
```

執行：

```python
from main import *
obj = read_object(FIXTURES / "ledger-x86_64-O1.o")
lifted = lift_function(obj, "percent_change")
for insn in lifted.decoded: print(f"0x{insn.address:x}: {insn.raw.hex(' ')}  {insn.att()}")
print(lifted.ir_text)
print(lifted.ssa_text)
print(lifted.pseudocode)
```

跟隨 `%esi`，從 `subl %edi, %esi` 到 `esi_1 = esi_0 - edi_0` 再到 `(param_2 - param_1)`。位元組 `fe` 是 `subl` 的 ModR/M，其 `mod = 11`、`reg = edi`、`r/m = esi`。

### 實驗 B：除法的慣用寫法

```python
print(lift_function(obj, "grade", recognize_division=False).pseudocode)
print(magic_divisor(0x66666667, 34), magic_divisor(0x66666667, 33), magic_divisor(0x66666667, 31))
```

第一行印出原始的乘法與 shift。第二行印出 `10 5 None`。算出為什麼相同的乘數在少一次 shift 時除以 5，以及為什麼 shift 31 無法通過誤差上限。

### 實驗 C：新的 object 讓 decoder 停下來

以 `-O2` 建構 fixture 並 lift 它：

```bash
clang -target x86_64-unknown-linux-gnu -O2 -fno-asynchronous-unwind-tables -c fixtures/ledger.c -o "$TMPDIR/ledger-O2.o"
python3 -c 'from main import *; from pathlib import Path; import os; lift_all(read_object(Path(os.environ["TMPDIR"]) / "ledger-O2.o"))'
```

Decoder 在 `0x56` 停下來，訊息為 `opcode 81 is outside the fixture subset`。Opcode `81` 是帶 32 位元 immediate 的 group 1 形式。該指令是 `andl $0x7ffffff8, %edx`，clang 把它用在 `total` 的向量化迴圈中。把它加在 `decode_one()` 中 `0x83` 旁邊，重新執行，並記下下一個讓你停下來的 opcode。子集一次增加一個項目，而每個項目都對照 `llvm-objdump` 檢查。

### 實驗 D：Ghidra 在 grade 中的預設值

```python
print(ghidra_pseudocode("grade"))
for score in (-25, 0, 59, 60, 100, 115):
    sign = -(score >> 31)
    product_sign = ((score * 0x66666667) & ((1 << 64) - 1)) >> 63
    print(score, sign ^ product_sign)
```

每一列都印出 0。Ghidra 的 `uVar1` 表達式是正確的，但難以閱讀，而指令 `xorl %eax, %eax` 只用兩個位元組就說了。

## 實作練習 (Practice Lab)

為 `-O0` 的 `grade` 加入 `switch` 還原。

需求：

- 讀取 `ledger-x86_64-O0.o` 的 `.rela.rodata` 中的五個 relocation，每個都是 `R_X86_64_PC32.text+offset`，並從 `.rodata+0x0` 的表格計算出五個 case 目標。
- 在 IR 中加入 `Switch` terminator，帶有 index 表達式、目標清單，以及來自表格載入之前 `ja` 的 default。
- 讓 switch block 每個目標有一個後繼者，使 dominator、post-dominator 與 SSA 繼續運作。
- 送出 `switch (local_10) { case 0:... }`，每個目標一個 `case`，並在 join 之後放共用的 epilogue。
- 對任何你無法界定其表格的 indirect jump，保留註記。
- 用 clang 編譯結果，並在測試中的分數格上與原始碼比較。

成功的意思是，`-O0` 的 `grade` 與 `-O1` 的 `grade` 在輸入格上產生相同的輸出，有一個 `switch` 且沒有 `goto`。

## 評分實驗 (Graded Lab)

關於這堂課 fixture 的五個問題。用你建構的程式碼找出每個答案，然後寫進 `code/lab_answers.json`。數字寫成數字或像 `"0x40"` 這樣的十六進位字串，清單寫成 JSON 清單。

1. `t1`：呼叫慣例在 -O1 為 percent_change 提供的參數數量。
2. `t2`：-O1 的 total 的 SSA 形式中的 phi 函式數量。
3. `t3`：magic_divisor 對 0x66666667 搭配 shift 34 找到的除數。
4. `t4`：-O1 的 percent_change pseudocode 的最後一個 statement，以一行文字表示。
5. `t5`：-O1 的 total 的 param_1 還原後的 C 型別。

檢查你的答案：

```bash
cd phases/13-tools-and-protocols/34-lifting-to-pseudocode
python3 code/lab.py
```

檢查器會依據 `code/main.py` 在已 commit 的 fixture 上計算每個預期值。它為每個任務印出 `PASS`、`FAIL` 或 `TODO`，以及一個分數，只有當五個全部通過時才以 0 結束。`python3 code/lab.py my-answers.json` 檢查另一個檔案。`code/tests/test_lab.py` 證明檢查器接受正確的答案，並拒絕錯誤的答案。

### 看它執行

[Inside a Decompiler](https://aiengineeringfromscratch.com/inside-a-decompiler?opt=O0) 這台機器把這堂課當作它的 lift press、type pin、structuring jig 與 printer 來執行，並附上 `code/main.py` 印出的 IR、SSA 與 pseudocode。這個連結開啟 `-O0` 的 object：三個 stack slot 變成區域變數，而 jig 為 if/else 閉合兩個鉗口。

## 成品 (Artifact)

這堂課會產出 `outputs/skill-lifting-to-pseudocode.md`。用它 lift 一個函式，使每一行都有推導，並對照反編譯器的文字填寫差異表。這份成品要求參考反組譯、relocation、呼叫慣例、編譯檢查、行為比較，以及每個不同行的判定。

## 驗證 (Verify It)

執行 demo 與確定性的測試套件：

```bash
cd phases/13-tools-and-protocols/34-lifting-to-pseudocode
python3 code/main.py
python3 -m unittest discover -s code/tests -v
```

測試套件有 25 個測試。驗證證明：

- decoder 在兩個 object 的每條指令上，都與已提交的 `llvm-objdump` 文字一致，而且在安裝時也與即時的 `llvm-objdump` 一致
- REX、ModR/M、SIB 以及 `leaq (%rip), %rcx` 的 displacement 欄位 offset，都如 SDM 所描述的方式解碼
- 子集之外的 opcode 會丟出 `DecodeError`，而不是猜測
- ELF reader 列出四個函式與三個 `.text` relocation
- `total` 有五個 block，`0x44` dominate `0x50`，而 `0x50` 是以自己作為主體的迴圈 header
- `%rdi` 和 `%rsi` 變成 `param_1` 和 `param_2`，而 `total` 中的 `param_1` 是 `int *`
- 搭配 `je`、`jle`、`ja` 和 `jge` 的 `test` 與 `cmp`，變成 `==`、`<=`、無號 `>` 和 `<`
- SSA 在 `total` 的迴圈 header 放置兩個 live 的 phi 函式，而在 `percent_change` 中沒有
- `percent_change`、`grade` 和 `total` 的 pseudocode 等於預期的文字
- `grade` 在辨識開啟時讀作 `param_1 / 10 - 6`，在辨識關閉時是原始乘法
- `0x66666667` 搭配 shift 34 除以 10，搭配 shift 33 除以 5，而恆等式在輸入格上成立
- `-O0` 的 stack slot 變成 `local_4`、`local_8`、`local_c`，而 frame pointer 從不出現
- `.rodata+0x14` 的表格透過 relocation 讀作 `1, 2, 3, 4, 4`
- `-O0` 的 `grade` 中的 indirect jump 會被回報，並附上它四個未 lift 的 block
- Ghidra fixture 包含 16 位元組的回傳、`param_3`、未簡化的 `xor`，以及 `long param_1`
- 送出的六個單元用 `clang -std=c11 -Wall -Werror` 針對 `x86_64-unknown-linux-gnu` 編譯
- lift 後的函式、修復後的 Ghidra 文字與原始碼，在輸入格上印出相同的 232 行

## Production 的失敗模式 (Production Failure Modes)

| 失敗 | 薄弱的解讀回報什麼 | 推導必須顯示什麼 |
|---|---|---|
| 反編譯器加寬回傳值 | 「回傳 16 位元組的 struct」 | `idivl` 寫入 `%rdx`，而沒有 caller 決定寬度 |
| 反編譯器新增參數 | 「接受三個引數」 | 該暫存器只在一條路徑上被讀取，因為回傳寬度是猜的 |
| 慣用寫法沒有被辨識 | 「乘以一個 magic 常數」 | 乘數是 `2^s / d` 的 ceiling，且誤差上限成立 |
| 慣用寫法被辨識，但 `xor` 遺失 | 「預設值取決於輸入」 | `xorl %eax, %eax` 寫入零，同一個位元的兩種寫法互相抵消 |
| 指標被型別設為整數 | 「接受一個 long」 | 來自暫存器、有 scale 的 4 位元組 load 是指標的證據 |
| 表格長度未知 | 「讀過表格的結尾」 | 下一個被參照的 offset 或 section 結尾界定了它，且範圍檢查一致 |
| 迴圈形狀被誤讀 | 「while 迴圈」 | header 沒有離開，latch 跳回，所以是 `do while` |
| Indirect jump 被猜測 | 「五個 case 的 switch」 | 目標來自 `.rela.rodata`，否則該跳轉被回報為未解析 |
| 跨越 store 的替換 | 「一個表達式」 | 被載入的值移過 store 或 call 會改變意義 |
| 改善模型編輯文字 | 「能編譯，看起來也對」 | build 成功率上升，行為符合率下降，被捏造的 guard 取代了看得見的未知 |
| Decoder 猜測未知的 opcode | 「反組譯看起來沒問題」 | 帶有位址與位元組的 `DecodeError`，然後是一個對照 `llvm-objdump` 檢查過的表格項目 |

## 營運規則 (Operational Rule)

把 pseudocode 視為你尚未見過的推導。用你能讀懂的 lifter，從指令重現它。把兩段文字並排，並為每個不同的行，寫下背後的指令或猜測。如果有原始碼，就編譯兩者並對照原始碼執行。沒有指令支持的行，無論讀起來多乾淨，仍然是猜測。

## 延伸閱讀 (Further Reading)

- [Intel SDM Volume 2, chapter 2, Instruction Format](https://www.intel.com/content/www/us/en/developer/articles/technical/intel-sdm.html)
- [System V AMD64 psABI, section 3.2, Function Calling Sequence](https://gitlab.com/x86-psABIs/x86-64-ABI)
- [Cytron et al., Efficiently Computing Static Single Assignment Form and the Control Dependence Graph, TOPLAS 1991](https://doi.org/10.1145/115372.115320)
- [Cooper, Harvey, Kennedy, A Simple, Fast Dominance Algorithm, 2001](https://www.cs.tufts.edu/~nr/cs257/archive/keith-cooper/dom14.pdf)
- [Cifuentes, Reverse Compilation Techniques, PhD thesis, QUT 1994](https://eprints.qut.edu.au/36820/)
- [Yakdan et al., No More Gotos, NDSS 2015](https://www.ndss-symposium.org/ndss2015/ndss-2015-programme/no-more-gotos-decompilation-using-pattern-independent-control-flow-structuring-and-semantics/)
- [Granlund, Montgomery, Division by Invariant Integers using Multiplication, PLDI 1994](https://doi.org/10.1145/178243.178249)
- [Lee, Avgerinos, Brumley, TIE, NDSS 2011](https://www.ndss-symposium.org/ndss2011/tie-principled-reverse-engineering-of-types-in-binary-programs/)
- [Noonan, Loginov, Cok, Polymorphic Type Inference for Machine Code, PLDI 2016](https://arxiv.org/abs/1603.05495)
- [LLM4Decompile, EMNLP 2024](https://arxiv.org/abs/2403.05286)
- [DecompileBench, Findings of ACL 2025](https://aclanthology.org/2025.findings-acl.1194/)
- [Liu, Raff, Micinski, When LLM Decompilers Recompile More and Preserve Less, 2026](https://arxiv.org/abs/2609.05370)
- [rea 6.5.0 Ghidra provider notes](https://github.com/morluto/rea/blob/rea-agents-6.5.0/docs/ghidra-provider.md)
- 這堂課所描述的 rea 6.5.0 原始碼，位於 tag `3b79af2`：`src/domain/functionComparisonDimensions.ts`、`website/evidence/dx-ball-sound-pan.md`。
