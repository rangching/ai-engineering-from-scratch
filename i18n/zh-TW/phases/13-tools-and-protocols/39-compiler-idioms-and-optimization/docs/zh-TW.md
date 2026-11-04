# 編譯器慣用寫法：最佳化對你的程式碼做了什麼，以及如何讀回來 (Compiler Idioms: What Optimization Does to Your Code and How to Read It Back)

> 最佳化編譯器會用更便宜的指令形狀取代原始碼的結構。每種形狀都有推導，而這堂課建構的辨識器會把推導反向執行，並對照原始碼檢查。

**Type:** Build
**Languages:** Python, C, x86-64 assembly, AArch64 assembly
**Prerequisites:** Phase 13 · 33 (disassembly and control flow), Phase 13 · 34 (lifting to pseudocode), Phase 13 · 38 (calling conventions)
**Time:** ~150 分鐘

## 學習目標 (Learning Objectives)

- 用 Granlund-Montgomery 的上界，為任何常數除法推導 magic number 與 shift，並從 clang 送出的常數還原除數。
- 說出 LLVM 降低 `switch` 的五種方式，並從 object 讀取項目數量、表格位置與項目。
- 把向量化的迴圈讀成 lane、accumulator、每次迭代的元素數、trip-count mask、reduction 與餘數迴圈。
- 辨識 tail call、被 inline 的 callee、無分支的 select，以及變成 `memset` 或 `memcpy` 的迴圈。
- 在 x86-64 與 AArch64 上，從 `-O0` 到 `-O3`，為每個慣用寫法回報其證據指令與解碼後的原始碼形式。
- 將結果與 Ghidra、angr、radare2、rizin 和 lldb 對相同位元組顯示的內容比較。

## 問題 (The Problem)

你以 `-O2` 編譯 `idioms.c`，並問 agent `div7u` 做什麼。Agent 讀到 `imulq $0x24924925, %rax, %rax`，並回答該函式乘以 613566757、shift，然後相加。每個字都是真的。該函式計算的是 `x / 7`。

同一個 object 持有完全沒有跳轉的七個 case 的 `switch`、是 jump table 的五個 case 的 `switch`、是 `sete` 加 `cmovne` 的兩個 case 的 `switch`，以及一個對 `n` 個整數的迴圈，它在兩個 `xmm` 暫存器中每次迭代處理八個整數，然後在第二個迴圈中完成其餘的部分。在 `-O0`，同樣的原始碼產生了 `divl`、七個項目的 jump table，以及一個純量迴圈。在 `-O1`，七個 case 的 switch 變成值的表格，而向量迴圈還不存在。

這些形狀就是慣用寫法。編譯器根據你能在它原始碼中讀到的規則，為每個原始碼結構送出每一種形狀，而每一種都可以被解碼回它的來源。解碼是算術，算術可以被檢查。課程 34 證明了其中之一，也就是乘以 `0x66666667`。這堂課為它們全部建構辨識器，在我們自己程式碼的 18 個 build 上執行，並對照 C 原始碼驗證全部 373 份報告。

```figure
re-39-idioms-by-level
```

## 慣用寫法是帶有推導的形狀 (An Idiom Is a Shape With a Derivation)

慣用寫法是編譯器為某個原始碼結構送出的指令序列。三件事讓它可被辨識。它有固定的形狀，operand 帶有推導所預測的常數，而且形狀不同於程式設計師手寫的東西。

`code/main.py` 中的辨識器處理 `llvm-objdump -d -r` 的文字，課程 33 將它確立為指令邊界的 oracle。它把每個函式解析成帶有位址、位元組、mnemonic、operand，以及隨後的 relocation 行的指令。它只為兩件事讀取 object file：存放值表格的 `.rodata` 位元組，以及連結前存放 jump-table 目標的 `.rela.rodata` 項目。

每個偵測都是一筆 `Idiom` 紀錄，帶有 `kind`、解碼後的 `source` 形式、`evidence` 位址清單、`detail` dictionary 以及 `verified` 旗標。該旗標來自 C 原始碼。`parse_source()` 讀取 `idioms.c`，找到每個函式，並擷取 `/` 與 `%` 之後的除數、`*` 之後的乘數、`switch` 的 case 及其值、`? :` 運算子、`for` 迴圈以及 callee。除數在函式的除法清單中時，除法慣用寫法就是 verified。查詢表的項目符合 case 值（相差一個常數倍）時，就是 verified。Callee 在原始碼中被呼叫時，tail call 就是 verified。Demo 報告以 `unverified` 清單作結，而測試套件斷言，所有 18 個 build 的這份清單都是空的。

解碼需要名稱。辨識器對每個函式的 basic block 依清單順序執行小型符號求值器，並在課程 38 的 System V 與 AAPCS64 慣例的引數暫存器中，放入來自原始碼的參數名稱。Stack slot 以其 operand 文字追蹤，所以 `-O0` 的程式碼能讀穿它的 spill。在每個 block 進入點，前驅的狀態會被合併，而在它們不同的地方，先前某個慣用寫法指派的名稱勝過常數。這就是 `digest` 中的 `cmovll %edx, %eax` 被讀回 `sum % 10 < 5 ? sum % 10 : 5` 的方式：迴圈慣用寫法把累加器命名為 `sum`，除法慣用寫法把它的商命名為 `sum / 10`，而 modulo 慣用寫法把它的結果命名為 `sum % 10`。原始碼寫的是 `rem > 5 ? 5 : rem`。兩個表達式都給出 `rem` 和 5 中較小的那個，而且它們在 `rem == 5` 時一致。

## 常數除法 (Division by a Constant)

Granlund 與 Montgomery 的〈Division by Invariant Integers using Multiplication〉（PLDI 1994），用一次乘法與一次 shift 取代 `n / d`。挑選 shift `s` 與乘數 `m = ceil(2^s / d)`。當誤差 `e = m * d - 2^s` 夠小時，`floor(n * m / 2^s)` 對範圍內的每個 `n` 都等於 `floor(n / d)`。對無號 32 位元 `n`，上界是 `e <= 2^(s - 32)`。對 `|n| < 2^31` 的有號 `n`，上界是 `e <= 2^(s - 31)`，並加上符號修正：當 `n` 為負時加 1，因為 floor 對負的商往錯誤的方向捨入。

`main.py` 中的 `magic_numbers(d, bits, signed)` 從 32 開始往上搜尋 `s`，並回傳第一組滿足上界的配對。那個搜尋重現了 clang 在 fixture 中送出的每一個常數。

### 手算無號 10

`2^35 / 10 = 3435973836.8`。Ceiling 是 `3435973837 = 0xCCCCCCCD`。誤差是 `0xCCCCCCCD * 10 - 2^35 = 2`，而在 `s = 35` 的上界是 `2^3 = 8`。在 `s = 34` 時，誤差會是 6，而上界是 4，所以 35 是最小的 shift。用 `n = 123` 檢查：`123 * 3435973837 = 422624781951`，而 `422624781951 >> 35 = 12`。

clang 在 x86-64 的 `-O2` 為 `div10u` 送出的正是這個：

```text
       0: 89 f9                        	movl	%edi, %ecx
       2: b8 cd cc cc cc               	movl	$0xcccccccd, %eax
       7: 48 0f af c1                  	imulq	%rcx, %rax
       b: 48 c1 e8 23                  	shrq	$0x23, %rax
```

`0x23` 是 35。`movl %edi, %ecx` 把 32 位元引數 zero-extend 進 64 位元暫存器，這是無號的形式。辨識器從餵給雙 operand `imulq` 的 `movl $imm` 讀取乘數、讀取 shift，並呼叫 `divisor_from_magic(0xCCCCCCCD, 35)`。該函式嘗試 `2^35 // m = 10` 與 11，並接受 10，因為 `magic_numbers(10)` 回傳相同的乘數與 shift。

在 AArch64 上，同一個函式是 `mov w8, #0xcccd; movk w8, #0xcccc, lsl #16; umull x8, w0, w8; lsr x0, x8, #35`。常數以兩個 16 位元的半部抵達，乘法是 `umull`，而 shift 以十進位書寫。辨識器組合那兩個半部，並回報相同的除數。

### 手算無號 7，附加法修正 (Unsigned 7 by hand, with the add fix-up)

`2^35 / 7 = 4908534052.57`。Ceiling 是 `4908534053 = 0x124924925`，需要 33 位元。誤差是 3，上界是 8，所以 shift 是 35，沒有更小的 shift 可行。32 位元的乘法放不下這個乘數，所以編譯器乘以低 32 位元 `0x24924925`，並以固定的序列把被除數加回去：

```text
      32: 48 69 c0 25 49 92 24         	imulq	$0x24924925, %rax, %rax
      39: 48 c1 e8 20                  	shrq	$0x20, %rax
      3d: 29 c7                        	subl	%eax, %edi
      3f: d1 ef                        	shrl	%edi
      41: 01 f8                        	addl	%edi, %eax
      43: c1 e8 02                     	shrl	$0x2, %eax
```

把它讀作 `t = (n * 0x24924925) >> 32`，然後 `q = (((n - t) >> 1) + t) >> 2`。總 shift 是 `32 + 1 + 2 = 35`。以 `n = 100`：`t = 14`、`(100 - 14) >> 1 = 43`、`43 + 14 = 57`、`57 >> 2 = 14`。辨識器偵測 32 位元 shift 之後的 `sub`、`shr`、`add`、`shr` 尾端，把 `2^32` 加回乘數，並還原出 7。`evaluate_division()` 執行這個確切的序列，而測試對照 32 位元邊緣值的 `n // 7` 檢查。AArch64 的形式是 `sub w9, w0, w8; add w8, w8, w9, lsr #1; lsr w0, w8, #2`，少一條指令，因為帶 shift 的 add 是一條指令。

### 手算有號 3 (Signed 3 by hand)

`2^32 / 3 = 1431655765.33`，ceiling `1431655766 = 0x55555556`，誤差 2，而在 `s = 32` 時有號的上界是 `2^1 = 2`。上界剛好成立而沒有餘裕，所以 shift 是 32，clang 使用 `shrq $0x20`。以 `n = -7`：乘積是 `-10021590362`，算術 shift 32 得到 `-3`，符號位元已設定，所以修正加 1，結果是 `-2`，也就是 C 中的 `-7 / 3`。有號的 fixture `div10s` 和 `div3s` 都把修正寫成對乘積副本的 `shrq $0x3f`，後面接 `addl`。

### 常數不是什麼

乘數本身不指名除數。`0x66666667` 搭配 shift 34 是 10。相同的乘數搭配 shift 33 是 5，如課程 34 所示。因此辨識器永遠把乘數與總 shift 配對，而 `clangMatchesAlgorithm` 欄位記錄該配對就是搜尋所產生的那一組。2 的冪永遠不使用乘法。`magic_numbers()` 拒絕它們，因為編譯器改為送出帶符號偏移的 shift，而下一節的 `mod8s` 顯示了那個形狀。

### 慣用寫法屬於指令選擇器 (The idiom belongs to the instruction selector)

在 `-O0`，`div10u` 是 `movl $0xa, %ecx; xorl %edx, %edx; divl %ecx`。`div10s` 是 `cltd; idivl %ecx`。在同一個 object 中，`digest` 用 `imulq $0x66666667` 把它的迴圈總和除以 10。一個旗標解釋了差異。Fixture `idioms-x86_64-nofastisel-O0.o` 是用 `-O0 -mllvm -fast-isel=false` 建構的，其中每個除法都是乘法，包括帶 `movl $0xcccccccd, %ecx; imulq %rcx, %rax; shrq $0x23` 的 `div10u`。硬體除法是 `-O0` 時 fast instruction selector 的選擇。乘法是 SelectionDAG 的降低結果，也就是 LLVM 的 `UnsignedDivisionByConstantInfo` 和 `SignedDivisionByConstantInfo` 所產生的。持有除法的 `digest` block 也持有一個 `switch`，而那個 block 走了 SelectionDAG。辨識器兩種形式都回報：`hardware idivl` 與 `magic multiply`，而圖表顯示混合的 `-O0` 函式。

## Modulo 跟隨商 (Modulo Follows the Quotient)

`x % d` 是 `x - (x / d) * d`。編譯器用上面的慣用寫法計算商、乘回去，然後相減。在 x86-64 上，乘回 10 是 `addl %ecx, %ecx; leal (%rcx,%rcx,4), %ecx`，也就是 `2 * q * 5`。在 AArch64 上它是一條指令 `msub w0, w8, w9, w0`，計算 `w0 - w8 * w9`，其中 `w9 = 10`。辨識器透過複製追蹤商暫存器，透過 `add`、`lea`、`shl` 和 `imul` 追蹤累積的因子，並在因子等於除數時接受 `sub`。

當同一個商也餵給另一個表達式時，兩個慣用寫法都留在報告中。在 `-O1`，編譯器在 `digest` 中計算一次 `sum / 10`，並把它同時用於 `switch` 的 index 與 modulo。在 x86-64 上，它甚至把商結合兩次：modulo 用 `leal (%rax,%rsi), %edi`，switch 用 `addl %esi, %eax`。辨識器回報一次除法與一次 modulo，並把商標記為共用。在這裡丟失除法的報告，會讓 `switch` 的 index 變得無法閱讀。

2 的冪的有號 modulo 有自己的形狀，因為商從不被計算：

```text
      92: 8d 48 07                     	leal	0x7(%rax), %ecx
      95: 85 ff                        	testl	%edi, %edi
      97: 0f 49 cf                     	cmovnsl	%edi, %ecx
      9a: 83 e1 f8                     	andl	$-0x8, %ecx
      9d: 29 c8                        	subl	%ecx, %eax
```

對負的 `x`，程式碼在遮蔽之前先偏移 7，讓截斷如 C 所要求的往零捨入。被遮蔽的值是 `x` 捨入到 8 的倍數，而減法留下 `x % 8`。AArch64 的做法是 `negs w8, w0; and w9, w0, #0x7; and w8, w8, #0x7; csneg w0, w9, w8, mi`：遮蔽 `x` 和 `-x` 兩者，然後依符號挑選並取負。Ghidra 為此命名的規則是 `RuleSignMod2nOpt`，簡述為「Convert INT_SREM forms: `(V + (sign >> (64-n)) & (2^n-1)) - (sign >> (64-n)) => V s% 2^n`」。

在 `-O0`，AArch64 把 modulo 寫成 `sdiv w9, w8, w10; mul w9, w9, w10; subs w0, w8, w9`。辨識器也接受那個形式，並回報 `hardware sdiv + mul + subs`。

## 降低 switch 的五種方式 (Five Ways to Lower a switch)

LLVM 在兩個地方決定 `switch` 的形狀。`SimplifyCFG::simplifySwitchLookup` 在 `-O1` 以上執行，並在 switch 的所有 case 都產生常數、至少有三個 case、且目標能建構查詢表時，把它變成值的表格。它的表格種類是 `SingleValueKind`、`LinearMapKind`、`BitMapKind` 和 `ArrayKind`。`-O0` 的管線從不執行這個 pass。其餘的 switch 到達指令選擇中的 `SwitchLowering::findJumpTables`，當 cluster 至少有 `min-jump-table-entries` 個項目（預設為 4）且夠稠密時，它會建構 jump table：`NumCases * 100 >= Range * MinDensity`，其中一般函式的 `jump-table-density` 為 10，`-Os` 下為 40。剩下的會變成對 case 常數的二分搜尋，由 `splitWorkItem` 選擇 pivot。Fixture 為每種結果各持有一個函式。

| 函式 | 原始碼 | `-O0` x86-64 | `-O2` x86-64 | `-O2` AArch64 |
|---|---|---|---|---|
| `linear_switch` | case 0 到 6 回傳 10 到 70 | jump table，7 個項目 | 線性映射 `10*k + 10`，default -1 | `madd` 和 `csinv` |
| `table_switch` | case 0 到 6 回傳 10 到 60，然後 71 | jump table，7 個項目 | 位於 `.rodata+0x28` 的查詢表 | 位於 `.rodata+0x0` 的查詢表 |
| `call_switch` | case 0 到 4 呼叫五個指標之一 | jump table，5 個項目 | jump table，5 個項目 | 比較樹，pivot 1 |
| `sparse_switch` | case 1、100、10000、1000000、100000000 | 比較鏈 | 比較樹，pivot 9999 | 比較樹，pivot 9999 |
| `two_cases` | case 3 和 4 回傳 7 和 9 | 比較鏈 | select 鏈 `k != 3 ? (k == 4)*9 : 7` | 兩個 `csel` |

### 查詢表，從 object 讀取

`-O2` 的 `table_switch`：

```text
      d0: b8 ff ff ff ff               	movl	$0xffffffff, %eax
      d5: 83 ff 06                     	cmpl	$0x6, %edi
      d8: 77 0c                        	ja	0xe6 <table_switch+0x16>
      da: 89 f8                        	movl	%edi, %eax
      dc: 48 8d 0d 00 00 00 00         	leaq	(%rip), %rcx
		00000000000000df:  R_X86_64_PC32	.rodata+0x24
      e3: 8b 04 81                     	movl	(%rcx,%rax,4), %eax
      e6: c3                           	retq
```

邊界檢查 `cmpl $6; ja` 表示有 7 個有效項目。Relocation addend `0x24` 加上 RIP 相對欄位的 4 個位元組，把表格放在 `.rodata+0x28`，這是課程 33 推導的規則。那裡的位元組是 `0a 00 00 00 14 00 00 00 ...`，所以項目是 `[10, 20, 30, 40, 50, 60, 71]`，並且等於原始碼的 case 值。組譯器在 `.s` 檔案中把這個表格命名為 `.Lswitch.table.table_switch`，並附上註解 `@switch.table.table_switch`。在 object 中，只剩下 relocation 與位元組。沒有跳轉，所以 control-flow graph 有三個 block。

AArch64 的形式以 `adrp x8` 和 `add x8, x8, #0x0` 組出表格位址，各自帶有指向 `.rodata` 的 relocation，並以 `ldr w0, [x8, w0, uxtw #2]` 載入。辨識器從 `add` 的 relocation 讀取 addend。

### 線性映射，完全沒有表格

`linear_switch` 對 0 到 6 的 `k` 回傳 `10 * (k + 1)`。`simplifySwitchLookup` 偵測仿射關係並送出算術：

```text
      b0: 83 ff 07                     	cmpl	$0x7, %edi
      b3: 8d 04 bf                     	leal	(%rdi,%rdi,4), %eax
      b6: 8d 4c 00 0a                  	leal	0xa(%rax,%rax), %ecx
      ba: b8 ff ff ff ff               	movl	$0xffffffff, %eax
      bf: 0f 42 c1                     	cmovbl	%ecx, %eax
```

求值器把 `leal (%rdi,%rdi,4)` 讀成 `5*k`，把 `leal 0xa(%rax,%rax)` 讀成 `10*k + 10`，並把 `cmpl $7` 之後的 `cmovb` 讀成 `k < 7 ? 10*k + 10 : -1`。辨識器接著對每個原始碼 case 檢查該公式，並將慣用寫法標記為 verified。AArch64 用 `madd w8, w0, w8, w9` 和 `csinv w0, w8, wzr, lo` 到達相同的表達式，其中搭配 `wzr` 的 `csinv` 在 default 路徑上產生 `-1`。

### Jump table，從 relocation 讀取

`-O2` 的 `call_switch` 是課程 33 在 `-O0` 的 `grade` 中還原的形狀：`cmpl $4; ja`、指向 `.rodata-0x4` 的 `leaq`、4 位元組項目的 `movslq`、`addq` 和 `jmpq *%rax`。五個項目是 `.rela.rodata` 中的 relocation，而 `target = addend - 4k` 給出 `0x10c, 0x107, 0x10e, 0x110, 0x109`，五個不同的 block，各自透過函式指標做 tail-jump。`dispatch` fixture 把這個擴大到 16 個 case：在 x86-64 上是 16 個 relocation 項目，而在 AArch64 上是 16 個單一位元組的壓縮表格，用 `ldrb w11, [x9, x8]` 讀取，用 `add x10, x10, x11, lsl #2` 以 4 為 scale，並以第一個 case block 的 `adr` 為基底。辨識器從位元組表格解碼出 `[0x24, 0x74, 0x54, ...]`，並回報 `compressed: true`。

### 同一個 switch 在四個層級

`digest` 持有一個對 `sum / 10` 的 switch，case 6 到 10，值為 1、2、4、8、8，然後把結果乘以 16。在 x86-64 上，它在 `-O0` 是 jump table，在 `-O1` 是項目為 `[16, 32, 64, 128, 128]` 的查詢表，在 `-O2` 和 `-O3` 又是 jump table，其四個 case block 儲存 `0x10`、`0x20`、`0x40`、`0x80`。從 `-O1` 往上，乘以 16 已移進 case 內。在 `-O0`，case 儲存 1、2、4 和 8，而乘法留在 switch 之後。對於 `-O1` 的查詢表，辨識器將項目與原始碼的值比較，找到固定的比率，並回報 `foldedFactor: 16`。在 AArch64 上，`-O2` 的形式是 pivot 為 8 的比較樹、針對 6、7 和 8 的相等測試，以及以兩條指令涵蓋 case 9 和 10 的範圍測試 `sub w9, w9, #9; cmp w9, #2; b.hs`。辨識器回報 `rangeTests: [[9, 10]]`。

有兩個旗標會移動這些界線。`-fno-jump-tables` 在 x86-64 上，把 `table_switch`、`call_switch` 和 `digest` 的 switch 變成比較樹，其中七個 case 的函式 pivot 是 2 和 4。在 AArch64 上，`-mllvm -min-jump-table-entries=4` 讓 `call_switch` 保持為比較樹，所以在那個目標上，項目數量並不是間接 call 的五個 case switch 的決定性輸入。Fixture `idioms-arm64-jt4-O2.o` 記錄了那次執行。

GCC 把兩個決定都放在 `tree-switch-conversion.cc` 中，其類別為 `jump_table_cluster`、`bit_test_cluster`、`switch_decision_tree` 和 `switch_conversion`。它的 `-ftree-switch-conversion` 在 `-O2` 開啟，`-fjump-tables` 和 `-fbit-tests` 在 `-O1` 開啟，而 jump table 的門檻來自 `case-values-threshold`，其預設值是機器的值。這台機器沒有安裝 GCC，所以這些名稱來自 GCC 原始碼與手冊，而 Compiler Explorer 是你檢查它們的地方。

## 迴圈：向量化、展開，或被取代 (Loops: Vectorized, Unrolled, or Replaced)

LLVM 的迴圈向量化器在 `-O2`、`-O3` 和 `-Os` 開啟，在 `-O0` 和 `-O1` 關閉，這條規則可追溯到 2014 年 1 月的 LLVM 3.4。`-O2` 的 `sum_n` 顯示了這個形狀的每個部分：

```text
     1f6: 83 fe 07                     	cmpl	$0x7, %esi
     1f9: 77 09                        	ja	0x204 <sum_n+0x14>
     206: 81 e2 f8 ff ff 7f            	andl	$0x7ffffff8, %edx
     230: f3 0f 6f 14 37               	movdqu	(%rdi,%rsi), %xmm2
     235: 66 0f fe c2                  	paddd	%xmm2, %xmm0
     239: f3 0f 6f 54 37 10            	movdqu	0x10(%rdi,%rsi), %xmm2
     23f: 66 0f fe ca                  	paddd	%xmm2, %xmm1
     243: 48 83 c6 20                  	addq	$0x20, %rsi
     24a: 75 e4                        	jne	0x230 <sum_n+0x40>
     24c: 66 0f fe c8                  	paddd	%xmm0, %xmm1
     250: 66 0f 70 c1 ee               	pshufd	$0xee, %xmm1, %xmm0
     262: 66 0f 7e c8                  	movd	%xmm1, %eax
     270: 03 04 97                     	addl	(%rdi,%rdx,4), %eax
     279: 75 f5                        	jne	0x270 <sum_n+0x80>
```

Guard `cmpl $7; ja` 把 trip count 小於 8 的情況送到純量迴圈。Mask `andl $0x7ffffff8` 把 trip count 向下捨入為 8 的倍數。本文每次迭代把 32 位元組載入兩個 `xmm` accumulator，每個有四個 32 位元的 lane，所以向量化因子是 4，interleave 因子是 2。迴圈之後，`paddd`、兩個 `pshufd`、再兩個 `paddd`，以及 `movd`，把八個 lane 化簡為一個整數。位於 `0x270` 的迴圈是最後 0 到 7 個元素的純量餘數。辨識器從 back edge 找到迴圈、看到 SIMD 暫存器、從 `paddd` 讀取 lane 數、計算不同 accumulator 的數量、把步長 `0x20` 除以 4，並回報 `4 lanes x 2 accumulators = 8 elements per iteration`，其 mask、guard、reduction 與餘數迴圈各為欄位。AArch64 以 `ldp q2, q3`、兩個 `add v.4s`、`addv s0, v0.4s` 和 `fmov w8, s0` 做同樣的事。

固定的 trip count 會移除迴圈。`sum4` 是一個 `movdqu`、reduction 和 `movd`，沒有 back edge。`poly8` 無法向量化，因為每次迭代都依賴前一個 accumulator，所以編譯器把它展開八次：從 `%rdi` 在 offset 0 到 `0x1c` 的八個 load，每一個後面接著 shift 5 與減法，計算 `acc * 31`。辨識器計算從同一個 base 開始、offset 連續的 load，並回報 `loop unrolled by 8, no back edge`。在 AArch64 上，相同的 load 以四個 `ldp` 配對出現。

用 `-fno-vectorize -fno-slp-vectorize` 關掉向量化器，`sum_n` 就變成展開 8 次的純量迴圈，有八個 `addl k(%rdi,%rdx,4), %eax` 和 `addq $8`，外加一個餘數迴圈。`sum4` 變成四次加法。辨識器對第一個回報帶有 `copies: 8` 和 `indexStep: 8` 的 `unrolled loop`，對第二個回報 `copies: 4`。

課程 33 在 `-O2` 的 `total` 中找到兩條 back edge，並稱之為兩個迴圈。這堂課為它們命名：同一個原始碼迴圈的一個向量主體與一個純量餘數。只有當 back edge 的目標能再次到達它時，它才是迴圈。`is_loop()` 檢查這點，因為 AArch64 上的 `digest` 包含指向共用 default block 的向後分支，這些不是迴圈，而辨識器的第一版把它們算成了向量迴圈。

### 變成函式庫呼叫的迴圈

`LoopIdiomRecognize` 把向每個位元組存入零的迴圈，從 `-O1` 往上變成對 `memset` 的呼叫。`-O2` 的 `fill` 是 `testl %esi, %esi; jle; movl %esi, %edx; xorl %esi, %esi; jmp memset@PLT`。這個呼叫是帶有 relocation `R_X86_64_PLT32 memset-0x4` 的 tail jump，而函式的 `retq` 只出現在空的路徑上。辨識器回報 `store loop recognized as memset(...)`，並對照原始碼迴圈驗證它，因為原始碼從未提到 `memset`。

`copy16` 以常數 16 呼叫 `memcpy`，而 x86 backend 將它 inline 展開為 `movups (%rsi), %xmm0; movups %xmm0, (%rdi)`，在其每個 `memcpy` 最多 8 次 store 的限制之內。在 `-O0`，同樣的呼叫是兩對 8 位元組的 `movq`。使用 `-fno-builtin` 時，call 保持為 call，也就是帶有 `movl $0x10, %edx` 的 `jmp memcpy@PLT`，而 `fill` 保留它的迴圈，向量化器接著把它變成每次迭代兩個 16 位元組 `movups` store 的迴圈。辨識器對那個 build 回報 `vectorized store loop: 2 x 16-byte stores = 32 bytes per iteration`。

## 消失或變成跳轉的 call (Calls That Disappear or Become Jumps)

`tail_caller` 回傳 `tail_target(x + 1)`。在 `-O0`，它是 `addl $0x1, %edi; callq tail_target@PLT; addq $0x10, %rsp; popq %rbp; retq`。在 `-O2`，它是 `incl %edi; jmp tail_target@PLT`，沒有別的。Caller 的 frame 消失了，而 callee 回到 caller 的 caller。在 AArch64 上，跳轉是帶有 `R_AARCH64_JUMP26` 的 `b tail_target`。辨識器把帶有 call relocation 的 `jmp` 或 `b` 視為 tail call。課程 33 的 recursive descent 必須把它視為函式離開，因為目標在函式之外。

`twice_sum` 呼叫 `leaf_add`。在 `-O0`，call 存在，`callq leaf_add@PLT`。從 `-O1` 開始，`leaf_add` 的本文 `leal (%rdi,%rsi), %eax` 出現在 `twice_sum` 內，而不再有 call。在 `-O0`，clang 把每個函式標記為 `optnone` 和 `noinline`，並只執行 always-inliner。辨識器從原始碼得知 `twice_sum` 呼叫 `leaf_add`，找不到對它的 call 或 tail jump，然後在 caller 中尋找 callee 本文的形狀：去掉目的地的 mnemonic 與來源 operand，所以 AArch64 caller 中的 `add w8, w1, w0` 符合 callee 中的 `add w0, w1, w0`。一次符合就是證據。

`dispatch` 結合兩者：一個 16 項目的 jump table，每個 case 都是 `movl %esi, %edi; jmp fN@PLT`。辨識器回報一個 switch 與 16 個 tail call，全部對照原始碼中的 16 個 call 驗證。

## 變成資料搬移的分支 (Branches That Became Data Moves)

`clamp_max` 是 `movl %esi, %eax; cmpl %esi, %edi; cmovgl %edi, %eax`。求值器把 `%edi` 命名為 `a`，把 `%esi` 命名為 `b`，記錄比較，並把 `cmovg` 讀成 `a > b ? a : b`，這就是原始碼的文字。`abs_int` 是 `movl %edi, %eax; negl %eax; cmovsl %edi, %eax`，讀作 `-x < 0 ? x : -x`；AArch64 把它寫成 `cmp w0, #0; cneg w0, w0, mi`，讀作 `x < 0 ? -x : x`。`is_pos` 是 `xorl %eax, %eax; testl %edi, %edi; setg %al`，讀作 `(x > 0)`。以 GCC 的說法，這些是 `-fif-conversion`，以 LLVM 的說法，是 `cmov` 選擇。LLVM 的 `X86CmovConversion` pass，在判斷有利時，可以把 `cmov` 轉回分支，這就是同一份原始碼在迴圈中可以顯示任一種形狀的原因。

在 `-O0`，這些都不存在。`clamp_max` 是一次比較、一個 `jle`、對同一個 slot 的兩次 store 和一次 load。辨識器在 `-O0` 對它什麼也不回報，從 `-O1` 開始則回報 `branchless select`，這是圖表讓人看得見的事實之一。

`mul9` 是單一個 `leal (%rdi,%rdi,8), %eax`，而辨識器回報帶有 `x * 9` 的 `strength reduction`，因為原始碼乘以 9。只有在原始碼有該因子時它才這麼做；為表格 index 縮放的 `lea` 不是乘法。

## 兩個看起來像慣用寫法的強化功能 (Two Hardening Features That Look Like Idioms)

Stack protector 是編譯器的功能，它留下的形狀，agent 否則會讀成程式邏輯。以 `-fstack-protector-strong` 為 x86-64 Linux 編譯的 `canary.c` 給出：

```text
       7: 64 48 8b 04 25 28 00 00 00   	movq	%fs:0x28, %rax
      10: 48 89 44 24 40               	movq	%rax, 0x40(%rsp)
      26: 64 48 8b 0c 25 28 00 00 00   	movq	%fs:0x28, %rcx
      2f: 48 3b 4c 24 40               	cmpq	0x40(%rsp), %rcx
      34: 75 05                        	jne	0x3b <with_canary+0x3b>
      3b: e8 00 00 00 00               	callq	0x40 <plain_leaf>
		000000000000003c:  R_X86_64_PLT32	__stack_chk_fail-0x4
```

`%fs:0x28` 是 glibc 的 `tcbhead_t` 中的 `stack_guard`，位於位元組 offset 40，而 LLVM 的 `getIRStackGuard` 為 glibc、bionic 和 Fuchsia 指名該 slot。Prologue 把 guard 複製到 buffer 之上，epilogue 進行比較，而失敗路徑呼叫 `__stack_chk_fail`。在 AArch64 Linux 上，guard 透過 GOT，以 `adrp` 和 `ldr` 來自全域變數 `__stack_chk_guard`，而失敗是 `bl __stack_chk_fail`。辨識器回報 guard 來源、slot、比較與 fail call。沒有陣列的 `plain_leaf`，在 `strong` 下沒有 canary，而 `strong` 保護的是有陣列、`alloca` 或被取位址的區域變數的函式。研究筆記也記錄了 `/GS` 下的 MSVC 形式，也就是與 `%rsp` 混合的 cookie，並由 `__security_check_cookie` 檢查；這台機器上沒有 MSVC linker，所以那個形狀只根據編譯器的組合語言輸出描述。

`-fcf-protection=full` 把 `endbr64`（位元組 `f3 0f 1e fa`）放在每個函式的第一條指令。它是具備 control-flow enforcement 的處理器上 indirect branch 的 landing pad，在其他地方則作為 `nop` 執行。Ghidra 的 x86-64 函式起點 pattern，因此包含 `push rbp; mov rbp, rsp` 的 `endbr64` 前綴變體。辨識器將它回報為 `cet landing pad`，所以以四個不尋常的位元組開頭的函式，不會被誤認為資料。

## 反編譯器折疊什麼 (What the Decompilers Fold)

課程 37 中的每個反編譯器，都把其中一些推導當作改寫規則執行。Ghidra 的反編譯器在 `ruleaction.cc` 中為它們命名。`RuleDivOpt` 是「Convert INT_MULT and shift forms into INT_DIV or INT_SDIV」，其形式為 `sub( (zext(V)*c), d) >> e => V / (2^n/(c-1)) where n = d*8 + e` 以及帶 `sext` 和 `s>>` 的有號變體。`RuleModOpt` 是「Simplify expressions that optimize INT_REM and INT_SREM」。`RuleDivTermAdd` 處理加法修正，`RuleSignDiv2` 把 `(V + -1*(V s>> 31)) s>> 1` 變成 `V s/ 2`，而 `RulePositiveDiv` 把符號位元清除的值的有號除法，變成無號除法。課程 37 在課程 33 的 `-O1` `grade` 上擷取了 Ghidra 12.1.4，輸出讀作 `uVar2 = param_1 / 10 - 6;`。除法被折疊了。

這些規則的輸入不需要 Java 就看得見。Venv 的 pypcode 4.0.1 使用與 Ghidra 相同的 SLEIGH 規格進行 lift，而 `code/fixtures/idioms-x86_64-O2.lifts.json` 持有 `div10s` 的原始 P-code：七條指令的 112 個項目，其中 7 個是開啟每條指令的 `IMARK` 標記，包括在 `unique[95a00:8] = sext(0x66666667)` 之後的 `RAX = RAX * unique[95a00:8]`、`RCX = RCX >> unique[d1a00:4]` 以及 `RAX = RAX s>> unique[c5c00:4]`。`shrq` 帶有 41 個操作，`sarq` 帶有 40 個，其中大部分是計算 flag。`RuleDivOpt` 在這個串流中比對乘法與兩次 shift，而 dead-code elimination 移除沒有人讀取的 flag。

angr 10.0.3 在我們的 `-O2` object 上執行，而 `idioms-x86_64-O2.angr.json` 持有它對 20 個函式的反編譯器輸出。它折疊了無號的形式：`div10u` 是 `return a0 / 10;`。它沒有折疊有號的形式：`div10s` 是 `v1 = a0 * 1717986919; return ((long long)(v1) >> 0x22) + (v1 >> 63);`，而 `mod10s` 是 `a0 - (...) * 2 * 5`。對 `div7u`，它印出 `(a0 / 7 + ((a0 * 6 / 7 & 0xffffffff) >> 1 & 0xffffffff) & 0xffffffff) >> 2`，其中修正項以算術的形式保留。它把 `mul9` 折疊為 `a0 * 9`，把 `linear_switch` 折疊為 `(a0 < 7 ? a0 * 10 + 10 : 0xffffffff)`，把 `clamp_max` 折疊為 `((int)a1 < (int)a0 ? a0 : a1)`，把 `is_pos` 折疊為 `a0 > 0`，把 `fill` 折疊為 `return memset(a0, 0, a1);`。它把 `table_switch` 印成載入 `*(int *)(4 * a0 + 4195400)` 而沒有值，把 `sum_n` 印成帶有 `_mm_add_epi32` 和 `_mm_shuffle_epi32` intrinsic，並把 `tail_caller` 印成 `tail_target(); return;`，引數遺失了。在 `digest` 中，jump table 變成 `goto *((void *)(...))`，其表格算術被寫了出來。

從原始碼建構進研究快取的 radare2 6.2.4 與 rizin 0.9.1，在同一個 object 上執行。radare2 的 `pdc` 把 `div10s` 印成 `rax = rax * 0x66666667; rcx >>>= 0x3f; rax >>= 0x22; eax += ecx`，是暫存器層級的 pseudo-C，沒有折疊，並把 `table_switch` 印成 `eax = dword [rcx + rax*4]`，在 `lea` 上附有 `RELOC 32 .rodata` 註解。rizin 沒有內建反編譯器；它的 `aoi` 把 `imul` 的 RzIL 印成 128 位元乘積。lldb-1703.0.236.21 載入 ELF object，它的 `disassemble -n div10s` 印出與 `llvm-objdump` 相同的七條指令，因為兩者都使用 LLVM 的 disassembler。Capstone 5.0.7 解碼相同的位元組，並回報 `ret` 和 `mode64` 之類的 group；它是 decoder，而慣用寫法從不在 decoder 的輸出中。IDA Pro 與 Binary Ninja 沒有在這裡執行。它們的文件描述了發生這類改寫的階段，也就是 microcode 最佳化與 MLIL 到 HLIL 的 pass，而這堂課不主張它們折疊哪些慣用寫法。

Compiler Explorer 是正向的工具。貼上 `idioms.c`、挑一個編譯器項目（例如「x86-64 clang 18.1.0」或「ARM64 GCC 14.2.0」），並設定 flag。該服務列出約 1,050 個 C 編譯器項目，並有 API：`GET /api/compilers/c` 回傳識別碼，`POST /api/compiler/<id>/compile` 以 JSON 回傳組合語言。Matt Godbolt 2019 年的 ACM Queue 文章，用這個工具逐步說明除法、population count、串接條件式與加總的慣用寫法。用它取得這台機器無法產生的 GCC 輸出，並將 GCC 12 的 jump table 門檻與 `-O2` 向量化預設值，與 fixture 中的 clang 輸出比較。

## 動手做 (Build It)

執行 demo：

```bash
cd phases/13-tools-and-protocols/39-compiler-idioms-and-optimization
python3 code/main.py
```

輸出是一份 JSON 報告。對 18 個 build 中的每一個，它列出每個函式及其慣用寫法，包含種類、原始碼形式、證據位址、驗證，以及 detail 欄位。它以每個 build 的種類矩陣、兩種架構的 `-O2` 對 `-O3` 比較，以及 `unverified` 清單作結。

預期的形狀：

```json
{
  "builds": {
    "idioms-x86_64-O2": {
      "arch": "x86_64", "level": "O2", "flags": "x86_64",
      "functions": {
        "div7u": [{"kind": "division", "source": "x / 7", "evidence": ["0x30", "0x32", "0x39", "0x3d", "0x3f", "0x41", "0x43"], "verified": true, "form": "magic multiply with add fix-up", "magic": "0x24924925", "effectiveMagic": "0x124924925", "shift": 35, "...": "..."}],
        "table_switch": [{"kind": "switch", "source": "switch (k) as lookup table", "table": ".rodata+0x28", "entries": 7, "values": [10, 20, 30, 40, 50, 60, 71], "...": "..."}],
        "sum_n": [{"kind": "vector loop", "lanes": 4, "accumulators": 2, "elementsPerIteration": 8, "...": "..."}]
      }
    }
  },
  "kindMatrix": {"idioms-x86_64-O2": {"division": 5, "modulo": 3, "switch": 6, "vector loop": 3, "...": "..."}},
  "sameCodeAtO2AndO3": {"x86_64": {"identical": true}, "arm64": {"identical": false, "differences": ["..."]}},
  "unverified": []
}
```

依此順序閱讀 `code/main.py`：

1. `parse_listing()` 把 `llvm-objdump -d -r` 的文字，轉成兩種架構的函式、指令與附屬的 relocation。
2. `ElfFile` 讀取 section、symbol 與 `.rela.*` 項目，而 `words()` 讀取表格項目。
3. `magic_numbers()`、`divisor_from_magic()` 和 `evaluate_division()` 是 Granlund-Montgomery 推導的正向、反向，以及指令序列形式。
4. `parse_source()` 從 C 檔案擷取驗證所需的結構。
5. `Evaluator` 是逐 block 的符號 pass，用來命名暫存器與 stack slot。
6. `find_divisions()` 和 `find_modulos()` 比對乘法、shift、修正、乘回與 2 的冪的形式。
7. `find_switches()` 分類 jump table、查詢表、線性映射、select 鏈、比較樹與比較鏈。
8. `find_vector_loops()`、`find_unrolled_loops()`、`find_tail_calls()`、`find_inlined_calls()`、`find_selects()`、`find_memory_builtins()`、`find_stack_protector()`、`find_cet()` 和 `find_strength_reduction()` 涵蓋其餘部分。
9. `analyze_function()` 依序執行它們，把每個慣用寫法的結果名稱餵給下一個求值器 pass。

還有另外三個指令。`python3 code/main.py scan FILE.objdump.txt [FUNCTION]` 在一份清單上執行辨識器，並在旁邊找到 object 與對應的 C 檔案。`python3 code/main.py magic DIVISOR [signed]` 印出乘數、shift、誤差，以及恆等式在邊緣值上是否成立。`python3 code/main.py figure-data` 印出圖表內嵌的資料。

## 實際使用 (Use It)

依此順序，在你建構的新 object 上使用辨識器。

1. 從 object 擷取 `llvm-objdump -d -r`，並把確切的編譯器指令放在它旁邊。辨識器從清單 header 讀取架構。
2. 對清單執行 `scan`，並先閱讀 `unverified` 項目。每一個要不是原始碼 parser 沒看到的結構，就是辨識器誤讀的形狀，兩者都值得看看證據位址。
3. 對每個除法，閱讀 `algorithm` 欄位。當 `clangMatchesAlgorithm` 為 true 時，編譯器選擇了最小的有效 shift。當它為 false 時，編譯器有你還不知道的理由；把常數寫下來。
4. 對每個 switch，比較你建構的各個層級的 `lowering`。變成樹的表格，或變成表格的樹，標記了你能從 LLVM 原始碼指出名稱的門檻。
5. 對每個迴圈，閱讀 `elementsPerIteration` 和 `scalarRemainderLoop`，然後確認課程 33 回報的兩條 back edge 是同一個原始碼迴圈。
6. 對同一個 object 執行一個反編譯器，把它的文字引用在每個慣用寫法旁邊，並記錄它折疊了哪些。這份紀錄是課程 45 的 baseline。

## 互動實驗 (Interactive Lab)

### 實驗 A：推導 magic number，然後找到它

從 `code` 目錄開啟 Python：

```bash
cd phases/13-tools-and-protocols/39-compiler-idioms-and-optimization/code
python3 -q
```

執行：

```python
from main import *
magic_numbers(10), magic_numbers(10, signed=True)
magic_numbers(7), magic_numbers(7, signed=True)
divisor_from_magic(0x66666667, 34, signed=True), divisor_from_magic(0x66666667, 33, signed=True)
```

在第一次呼叫之前，先在紙上計算 `ceil(2^35 / 10)`。然後在 `idioms.c` 的副本中新增函式 `unsigned div13(unsigned x) { return x / 13; }`，用 `build_fixtures.sh` 中的指令以 `-O2` 建構，並對照 `magic_numbers(13)` 檢查清單中的常數。

### 實驗 B：表格與樹

```python
listing = load_listing("idioms-x86_64-O2.objdump.txt")
elf = load_elf("idioms-x86_64-O2.o")
sources = load_source("idioms.c")
[i.to_dict() for i in analyze_function(listing, "table_switch", elf, sources)]
[i.to_dict() for i in analyze_function(listing, "sparse_switch", elf, sources)]
elf.words(".rodata", 0x28, 7)
```

在 `table_switch` 的副本中把 `71` 改回 `70`，以 `-O2` 重新建構，並看查詢表變成線性映射。然後把一個 case 值改成 1000，看它保持為 `ArrayKind` 表格。

### 實驗 C：三種設定下的迴圈

```python
for name in ("idioms-x86_64-O1", "idioms-x86_64-O2", "idioms-x86_64-novec-O2"):
    print(name, [i.source for i in analyze_function(load_listing(name + ".objdump.txt"), "sum_n", load_elf(name + ".o"), sources)])
```

用手畫出 `-O2` 的 `sum_n` 的 control-flow graph，標記 guard、mask、向量本文、reduction 與餘數迴圈，並與 `detail` 欄位比較。

### 實驗 D：共用的商

```python
for name in ("idioms-x86_64-O1", "idioms-arm64-O1"):
    print([(i.kind, i.source, i.detail.get("sharedQuotient")) for i in analyze_function(load_listing(name + ".objdump.txt"), "digest", load_elf(name + ".o"), sources)])
```

閱讀 `digest` 的 `-O1` 清單，找出在 x86-64 上把商與其符號修正結合的兩條指令。然後說明為什麼 AArch64 的 build 只需要一條。

## 實作練習 (Practice Lab)

把位元技巧加進 fixture，並擴充辨識器。

需求：

- 用 GCC builtin 和 shift-or 旋轉，寫 `popcount`、`ctz` 和 `rotl` 函式，以 `-O2` 在有與沒有 `-mpopcnt -mbmi` 的情況下建構，並把 object 與它們的清單一起 commit。
- 辨識 `TargetLowering::expandCTPOP` 送出的 `0x55555555`、`0x33333333`、`0x0f0f0f0f`、`0x01010101` population-count 序列，並將它回報為 `popcount`，以四個常數作為證據。
- 辨識 `rep bsfl` 為 `tzcnt` 或 `bsf` 的編碼，並辨識 `roll %cl` 為旋轉，並將兩者解碼為 builtin。
- 把新的種類加入 `parse_source()` 的驗證，讓 `popcount` 慣用寫法只有在原始碼呼叫 `__builtin_popcount` 時才是 verified。
- 為每個慣用寫法新增一個從 fixture 清單讀取常數的測試。

成功的意思是，`python3 code/main.py` 在兩個 build 中都以 `verified: true` 回報三個新的種類，而測試套件仍然回報空的 `unverified` 清單。

## 課程成品 (Lesson Artifact)

這堂課包含 `outputs/skill-compiler-idioms-and-optimization.md`。用它把清單、object 與其原始碼，變成附帶反編譯器比較的已驗證慣用寫法報告。它要求每項主張都有證據位址、每個除法都有 magic-number 推導、每個 switch 都有降低種類與項目來源，以及每個慣用寫法都有一行引用的反編譯器輸出。

## 驗證 (Verify It)

執行 demo 與確定性的測試套件：

```bash
cd phases/13-tools-and-protocols/39-compiler-idioms-and-optimization
python3 code/main.py
python3 -m unittest discover -s code/tests -v
```

驗證應該證明：

- 清單 parser 從 `-O2` object 讀取 25 個函式、保留位元組，並把 `R_X86_64_PC32 .rodata+0x24` 附加到 `table_switch` 的 `leaq`
- ELF reader 從 `.rodata+0x28` 回傳 `[10, 20, 30, 40, 50, 60, 71]`，並回傳 `.rela.rodata` 中的十個 relocation
- `magic_numbers` 回傳 shift 35 的 `0xcccccccd`、34 的 `0x66666667`、35 帶加法形式的 `0x124924925`、32 的 `0x55555556`，以及 33 的 `0xaaaaaaab`
- 對從 2 到 64 之間每個不是 2 的冪的除數，有號與無號的恆等式在 32 位元邊緣值上都成立
- `divisor_from_magic` 從它們的配對還原 10、5、7 和 3，並拒絕錯誤的配對
- 加法修正序列精確地計算 `n // 7`
- 四個 x86-64 除法回報正確的乘數、shift、除數與形式，而四個 AArch64 除法解碼 `mov` 和 `movk`
- `-O0` 使用 `divl` 和 `idivl`，`-O0 -fast-isel=false` 使用乘法，而 `-O0` 的 `digest` 混用兩者
- modulo 被辨識為乘回、`msub`、2 的冪、`sdiv` 加 `mul` 加 `subs`，以及硬體 `idivl`
- 共用的商在 `-O1` 同時保留除法與 modulo
- 五種降低方式在兩種架構的 `-O2` 被分類，並附有公式、值、目標、pivot 與 select 表達式
- `-O0` 的 switch 在 x86-64 上是 jump table，在 AArch64 上是比較鏈
- `digest` 的 switch 在 x86-64 上依序是 jump table、折疊因子 16 的查詢表、再到有四個不同目標的 jump table，在 AArch64 上是帶範圍測試 9 到 10 的比較樹
- `-fno-jump-tables` 產生比較樹，而 16 個 case 的 `dispatch` 在 x86-64 上從 relocation 讀取目標，在 AArch64 上從壓縮的位元組表格讀取
- 向量迴圈回報 4 個 lane、2 個 accumulator、每次迭代 8 個元素、mask、guard、reduction 與餘數迴圈
- 固定 trip 的迴圈被完全向量化或完全展開，而 `-fno-vectorize` 展開 8 次
- `-O1` 沒有向量化，x86-64 上的 `-O2` 與 `-O3` 完全相同，而在 AArch64 上它們只在 `call_switch` 不同，其中 `-O3` 有 17 條指令，`-O2` 有 18 條
- `-fno-builtin` 把 store 迴圈變成向量化的 store 迴圈，並讓 `memcpy` 保持為 call
- tail call、`memset` 取代、inline 的 16 位元組複製，以及 `-O0` 的兩次 move 複製都被辨識
- 被 inline 的 callee 從 `-O1` 起由其本文形狀找到
- select 解碼為 `a > b ? a : b`、`-x < 0 ? x : -x`、`(x > 0)` 與 `sum % 10 < 5 ? sum % 10 : 5`
- stack protector 回報 `%fs:0x28`、slot 與 `__stack_chk_fail`，而 `endbr64` 在 CET build 的每個函式上都被回報
- 原始碼 parser 讀取函式指標參數、switch 的 case、call 與 accumulator 名稱
- 每個 build 中的每個慣用寫法都是 verified，且種類矩陣相符
- 互動式圖表內嵌了分析所產生的相同資料

## Production 的失敗模式 (Production Failure Modes)

| 失敗 | 薄弱的工具回報什麼 | 分析必須證明什麼 |
|---|---|---|
| 乘以大的奇數常數 | 「乘以 1717986919」 | 該常數對某個 `d` 是 `ceil(2^s / d)`，誤差滿足上界，而 shift 是最小的有效值 |
| 32 位元 shift 之後的加法修正 | 五條算術指令 | 乘數是 33 位元寬，而尾端是同一方法的加法形式 |
| 相同的乘數，不同的 shift | 相同的除數 | shift 決定除數：34 是 10，33 是 5 |
| 2 的冪的有號 modulo | 「加 7、遮蔽、減去」 | 偏移讓結果往零捨入，而結果是 `x % 8` |
| 從 `.rodata` 的索引 load | 「jump table」 | 值的查詢表，其項目從 addend 加 4 處的位元組讀取 |
| 仿射算術與 `cmov` | 「計算 10k + 10」 | 降低為線性映射的 switch，對每個原始碼 case 驗證 |
| 帶有序分割的比較鏈 | 「一連串 if」 | 對 case 常數的二分搜尋，並指名 pivot |
| 向後分支到 default block | 「迴圈」 | 目標以無條件跳轉離開區域；它是共用的尾端 |
| 一個函式中有兩條 back edge | 「兩個迴圈」 | 一個原始碼迴圈：向量主體與其純量餘數 |
| 沒有 `ret` 的 `jmp memset@PLT` | 「函式沒有 return」 | 透過 callee 回傳的 tail call，取代 store 迴圈 |
| Callee 不在 call graph 中 | 「從未被呼叫」 | 被 inline：callee 的本文形狀出現在 caller 中 |
| `%fs:0x28` 的 load 與比較 | 「讀取 thread-local 資料」 | Stack protector 的 canary，其 fail 路徑通往 `__stack_chk_fail` |
| 函式開頭的四個奇怪位元組 | 「函式之前的資料」 | `endbr64`，CET 的 landing pad |
| 反編譯器顯示原始算術 | 「工具是錯的」 | 工具沒有套用折疊；推導給出原始碼形式，而引用的輸出記錄了這個缺口 |

## 營運規則 (Operational Rule)

比對形狀，反向執行推導，並寫下證明它的常數。回報每個慣用寫法時，附上它的證據位址與解碼後的原始碼形式，對照你編譯的原始碼驗證每一個，並引用每個反編譯器對相同位元組印出的內容。

## 延伸閱讀 (Further Reading)

- [Granlund, Montgomery, "Division by Invariant Integers using Multiplication" (PLDI 1994)](https://gmplib.org/~tege/divcnst-pldi94.pdf)
- [Warren, Hacker's Delight, 2nd edition, Chapter 10: Integer Division By Constants](https://www.informit.com/store/hackers-delight-9780321842688)
- [Lemire, Kaser, Kurz, "Faster Remainder by Direct Computation" (2019)](https://arxiv.org/abs/1902.01961)
- [libdivide, header-only division by runtime constants](https://github.com/ridiculousfish/libdivide)
- [LLVM `DivisionByConstantInfo.cpp`](https://github.com/llvm/llvm-project/blob/main/llvm/lib/Support/DivisionByConstantInfo.cpp)
- [LLVM `SwitchLoweringUtils.cpp`](https://github.com/llvm/llvm-project/blob/main/llvm/lib/CodeGen/SwitchLoweringUtils.cpp) and [`TargetLoweringBase.cpp`](https://github.com/llvm/llvm-project/blob/main/llvm/lib/CodeGen/TargetLoweringBase.cpp)
- [LLVM `SimplifyCFG.cpp`, `simplifySwitchLookup`](https://github.com/llvm/llvm-project/blob/main/llvm/lib/Transforms/Utils/SimplifyCFG.cpp)
- [LLVM Auto-Vectorization documentation](https://llvm.org/docs/Vectorizers.html) and [LLVM 3.4 release notes](https://releases.llvm.org/3.4/docs/ReleaseNotes.html)
- [LLVM `LoopIdiomRecognize.cpp`](https://github.com/llvm/llvm-project/blob/main/llvm/lib/Transforms/Scalar/LoopIdiomRecognize.cpp)
- [LLVM `X86CmovConversion.cpp`](https://github.com/llvm/llvm-project/blob/release/20.x/llvm/lib/Target/X86/X86CmovConversion.cpp)
- [LLVM `StackProtector.cpp`](https://github.com/llvm/llvm-project/blob/main/llvm/lib/CodeGen/StackProtector.cpp) and [`X86ISelLoweringCall.cpp`, `getIRStackGuard`](https://github.com/llvm/llvm-project/blob/main/llvm/lib/Target/X86/X86ISelLoweringCall.cpp)
- [glibc `sysdeps/x86_64/nptl/tls.h`, `tcbhead_t`](https://sourceware.org/git/?p=glibc.git;a=blob;f=sysdeps/x86_64/nptl/tls.h)
- [GCC `tree-switch-conversion.cc`](https://github.com/gcc-mirror/gcc/blob/master/gcc/tree-switch-conversion.cc), [GCC Optimize Options](https://gcc.gnu.org/onlinedocs/gcc/Optimize-Options.html), and [GCC 12 release notes](https://gcc.gnu.org/gcc-12/changes.html)
- [Ghidra decompiler `ruleaction.cc`](https://github.com/NationalSecurityAgency/ghidra/blob/master/Ghidra/Features/Decompiler/src/decompile/cpp/ruleaction.cc)
- [angr decompiler documentation](https://docs.angr.io/en/latest/analyses/decompiler.html)
- [radare2 book: decompilers](https://book.rada.re/arch/decompile.html)
- [Compiler Explorer API](https://github.com/compiler-explorer/compiler-explorer/blob/main/docs/API.md)
- [Godbolt, "Optimizations in C++ Compilers: A practical journey", ACM Queue 17(5), 2019](https://queue.acm.org/detail.cfm?id=3372264)
- [Microsoft, `/GS` (Buffer Security Check)](https://learn.microsoft.com/en-us/cpp/build/reference/gs-buffer-security-check)
