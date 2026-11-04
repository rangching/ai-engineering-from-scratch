# 反組譯與控制流程：從位元組到可檢查的圖 (Disassembly and Control Flow: From Bytes to a Graph You Can Check)

> 反組譯清單是一項主張，說明哪些位元組是指令；這堂課建構用來檢查該主張的 decoder 與圖。

**Type:** Build
**Languages:** Python, C, x86-64 assembly
**Prerequisites:** Phase 13 · 32 (ELF from scratch)
**Time:** ~150 分鐘

## 學習目標 (Learning Objectives)

- 依 Intel SDM Volume 2 第 2 章的規則，解碼 ledger fixture 中的 x86-64 指令子集。
- 對每個 fixture build 的每一條指令，以 `llvm-objdump -d` 證明 decoder 的正確。
- 對相同的位元組執行 linear sweep 與 recursive descent，並說明各自錯在哪裡。
- 把指令切成 basic block，並建構帶有型別化 edge 的 control-flow graph。
- 用 Cooper、Harvey 與 Kennedy 的迭代演算法計算 immediate dominator，並從 back edge 找出迴圈。
- 從指向其表格的 relocation 項目，還原 indirect jump 的目標。

## 問題 (The Problem)

你用 `-O0` 編譯 `ledger.c`，並問 agent `grade` 做什麼。Agent 呼叫 disassembler、讀取清單，然後回答說該函式結束於沒有已知目標的 `jmpq *%rax`。接著它寫出只有單一 `return 0` 的 pseudocode。

你用 `llvm-objdump -d` 打開同一個 object，看到跳轉之後有四個 block，分別儲存 4、3、2 和 1。清單裡一直都有它們。清單中沒有任何地方說明執行如何到達它們，所以 tool 把它們當成 dead code。

清單只回答一個問題：哪些位元組組成哪條指令。它沒有說明哪些指令會執行、依什麼順序、從哪裡開始。要回答這些，你還需要三種結構：控制流程能到達的位元組、basic block，以及它們之間的 edge。

rea 6.5.0 沒有自己的 disassembler。進行深入分析時，它的 README 要求你自備 Ghidra、Hopper 或 IDA。進行離線檢視時，rea 也可以呼叫你安裝的 GNU objdump 或 Rizin。它的 call graph 止於未解析的目標：「Callers/callees 只包含已解析的函式……沒有目標的計算式流程仍然是未知的。」這堂課建構產生那些事實的部分，讓你在 tool 回報它們時能夠判斷。

```figure
re-33-sweep-vs-descent
```

## 指令格式每個欄位各有一條規則 (The Instruction Format Has One Rule Per Field)

Intel SDM Volume 2 的 Figure 2-1 給出每條指令的位元組順序。先是選擇性的 prefix，接著是一到三個 opcode 位元組，然後是選擇性的 ModRM 位元組與選擇性的 SIB 位元組。接下來是 0、1、2 或 4 位元組的 displacement，再來是 0、1、2 或 4 位元組的 immediate。整條指令最長 15 位元組（SDM 2.3.11）。AMD APM Volume 3 第 1 章的 1.1 到 1.7 節，用較少的頁數描述了相同的格式，是很好的交叉檢查。

每個欄位決定下一個欄位的意義。因此 decoder 是一連串的小決定，而每個決定背後，手冊裡都有一句話。

### Prefix 分成四組

SDM 2.1.1 列出 legacy prefix。第 1 組包含 `F0` lock 以及 repeat prefix `F2` 和 `F3`。第 2 組包含六個 segment override（`2E` 是 `%cs`）以及 branch hint。第 3 組是 `66`，也就是 operand-size override。第 4 組是 `67`，也就是 address-size override。

手冊說每一組可以使用一個 prefix。編譯器會送出更多。`ledger-x86_64-O0.o` 中 offset `0xf2` 的對齊 padding 讀作 `66 66 66 66 66 2e 0f 1f 84 00 00 00 00 00`，也就是連續五個 `66`。`llvm-objdump` 把它印成 `nopw %cs:(%rax,%rax)`，你建構的 decoder 也一樣。Decoder 收集它看到的每個 prefix 位元組，之後再問哪些有意義。

`66` 還有第二個工作。對 SSE 指令而言，它是用來選擇 opcode 的強制 prefix，所以 `66 0f ef c0` 是 `pxor %xmm0, %xmm0`，而不是任何 16 位元的東西。`F3 0f 6f` 是 `movdqu`。Decoder 先查找強制 prefix 與第二個 opcode 位元組的組合。只有當該組合沒有項目時，`66` 才表示 16 位元的 operand。

### REX 增加四個位元，且必須緊貼 opcode

SDM 2.2.1 把 REX 定義為從 `40` 到 `4F` 的一個位元組，形狀是 `0100WRXB`。`W` 選擇 64 位元 operand。`R` 擴充 ModRM 的 reg 欄位。`X` 擴充 SIB 的 index 欄位。`B` 擴充 ModRM 的 r/m 欄位、SIB 的 base 欄位，或 opcode 位元組中的暫存器（Table 2-4）。REX 必須緊貼在 opcode 或 escape 位元組之前，而後面跟著 legacy prefix 的 REX 會被忽略。

在 `48 89 e5` 中，位元組 `48` 是設定了 `W` 的 REX。Opcode `89` 是 `MOV r/m, r`，而 ModRM `e5` 把 `%rsp` 選入 `%rbp`。拿掉 REX，相同的位元組就是 `movl %esp, %ebp`。

REX 也會改變位元組暫存器的名稱。只要存在任何 REX，編碼 4 到 7 就選擇 `%spl`、`%bpl`、`%sil` 和 `%dil`，而不是 `%ah`、`%ch`、`%dh` 和 `%bh`。Decoder 因此保留兩張表。

### Opcode 存在於兩張 map 中

單一位元組 opcode 是常見的情況。`0F` 跳脫到雙位元組的 map，而 `0F 38` 或 `0F 3A` 跳脫到 fixture 從未使用的三位元組 map（SDM 2.1.2）。有些 opcode 把 ModRM 的 reg 欄位當作另外三個 opcode 位元，在手冊中寫成 `/digit`。`83 /7` 是帶 8 位元 immediate 的 `cmp`，`83 /5` 是 `sub`，而 `FF /4` 是 indirect 的 `jmp`。

`code/main.py` 中的 decoder 涵蓋四個 fixture object 中的 opcode，加上幾個鄰近的。清單包括八種算術形式、`mov`、`lea`、`movslq`、`movzbl`、`imul`、shift、`test` 和 `idiv`。接著是 `push`、`pop`、`call`、`ret`、十六個條件跳轉、多位元組 `nop`，以及五個 SSE2 opcode。任何其他 opcode 都會丟出帶有位址與原因的 `DecodeError`。會猜測的 decoder，比會停下來的 decoder 更糟。

### ModRM 與 SIB 選擇 operand

SDM 2.1.3 把 ModRM 拆成 `mod`（位元 7:6）、`reg`（5:3）和 `r/m`（2:0）。`mod = 11` 表示 r/m 是暫存器。`mod = 00`、`01` 和 `10` 表示記憶體，分別是沒有 displacement、8 位元 displacement，或 32 位元 displacement。`r/m = 100` 表示後面跟著 SIB 位元組。SIB 拆成 `scale`（7:6）、`index`（5:3）和 `base`（2:0），而 index 為 `100` 表示沒有 index 暫存器。

Table 2-2 中兩個特殊的列，在 64 位元模式下很重要。`mod = 00` 搭配 `r/m = 101`，不是 `(%rbp)`。它是相對於下一條指令的 32 位元 displacement（SDM 2.2.1.6）。`mod = 00` 搭配 SIB base 為 `101`，是沒有 base 暫存器的 32 位元 displacement。

以 `-O0` 的 `grade` 中的 `8d 44 08 fa` 為例。Opcode `8D` 是 `lea`。ModRM `44` 是 `01 000 100`：8 位元 displacement、reg 為 `%eax`，且後面跟著 SIB。SIB `08` 是 `00 001 000`：scale 1、index 為 `%rcx`、base 為 `%rax`。Displacement 位元組 `fa` 是 -6。AT&T 文字是 `leal -0x6(%rax,%rcx), %eax`，它計算 `score / 10 - 6`，也就是 switch 的 index。

### Displacement 是 link editor 寫入的地方

`48 8d 0d 00 00 00 00` 是 `leaq (%rip), %rcx`。四個零位元組是 RIP 相對 operand 的 displacement 欄位。在可重定位的 object 中，它們是 placeholder。`.rela.text` 在 offset `0x69` 的項目寫著 `R_X86_64_PC32.rodata-0x4`。psABI（Table 4.9）把 `R_X86_64_PC32` 定義為 `S + A - P`：symbol 位址加 addend 減去欄位的位址。

處理器把欄位加到下一條指令的位址，所以最終的目標是 `S + A + (end - P)`。對這個 `lea` 而言，欄位是最後四個位元組，所以 `end - P` 是 4，目標是 `.rodata + (-4) + 4 =.rodata + 0x0`。-4 的 addend 事先補償了那四個位元組。fixture 中每個 RIP 相對 operand，都在它的 addend 裡帶有這個 -4，所以 `-O1` 的 `lea` 上的 `.rodata+0x10`，意思是 `.rodata+0x14`。

Decoder 記錄每條指令內 displacement 欄位的 offset。`resolve_reference()` 在 relocation 表中查找該 offset，並回報連結後 operand 會指向哪裡。就是這一個函式，把 `leaq (%rip), %rcx` 變成「位於 `.rodata+0x0` 的表格」。

### Immediate 最多 32 位元，只有一個例外

SDM 2.2.1.5 說，在 64 位元模式下，immediate 最多維持在 32 位元，並且會做符號擴充。例外是 `MOV r64, imm64`（帶 `REX.W` 的 `B8+r`），AT&T 語法稱它為 `movabsq`。Decoder 在算術中把 8 位元 immediate 讀成有號（`83 c1 fa` 是 `addl $-0x6, %ecx`），在 shift 次數與 shuffle selector 中讀成無號（`pshufd $0xee`）。這個區分與 `llvm-objdump` 的輸出一致。

Near branch、`push` 和 `pop` 預設使用 64 位元 operand 並忽略 `66`（SDM 2.2.1.7）。這就是 decoder 在不讀取 operand-size prefix 的情況下，把它們命名為 `pushq`、`popq`、`callq` 和 `retq` 的原因。

## 手動解碼一條指令 (Decode One Instruction by Hand)

在信任程式碼之前，先在紙上解碼 `grade` 中的 `48 69 c0 67 66 66 66`。

1. `48` 位於 `40` 到 `4F`，所以它是 `W = 1` 的 REX。Operand 大小是 64 位元。
2. `69` 是 `IMUL r, r/m, imm32`。
3. `c0` 是 ModRM `11 000 000`：暫存器形式，reg 是 `%rax`，r/m 是 `%rax`。
4. 接下來的四個位元組是 immediate，little-endian：`0x66666667`。
5. AT&T 順序是 immediate、source、destination：`imulq $0x66666667, %rax, %rax`。

這是七個位元組，而 `llvm-objdump` 印出相同的文字。那個常數是倒數，讓 clang 能以乘法加上後面的 `sarq $0x22`（shift 34）來除以 10。你會在課程 34 再次遇到這一對，在那裡 lifter 會把它變回 `score / 10`。

Decoder 在 `Decoder.decode()`、`_State.modrm()` 和 `_State.finish()` 中做這五個步驟。依此順序閱讀它們，並把每個分支對應到其 docstring 中指名的 SDM 章節。

## Linear sweep

Schwarz、Debray 與 Andrews 在〈Disassembly of Executable Code Revisited〉（WCRE 2002）的 3.1 節描述 linear sweep。從程式碼 section 的第一個位元組開始，解碼一條指令，在它之後的位元組繼續，直到結尾才停止。`objdump` 這樣運作，`llvm-objdump` 也是。

```python
def linear_sweep(code: bytes, start: int = 0, end: int | None = None) -> SweepResult:
    position = start
    while position < end:
        try:
            instruction = decode(code[:end], position)
        except DecodeError:
            undecodable.append((position, code[position]))
            position += 1
            continue
        instructions.append(instruction)
        position = instruction.end
```

這個方法有一種失敗模式，論文的 Figure 1 顯示了它。屬於資料的位元組被解碼成指令，然後 decoder 在一段時間內錯過真正的指令邊界。在 x86 上，decoder 通常會在幾個位元組內重新同步，這使得錯誤很難被看出來。只有遇到無效 opcode 時才會被偵測到，而論文的結論是這些演算法「默默地失敗」。

這個 sweep 也是這堂課的 oracle。Decoder 必須在每個 fixture 的每條指令上，都與 `llvm-objdump -d` 一致，而 `llvm-objdump` 本身就是 linear sweep。擷取到的文字放在 `code/fixtures/*.objdump.txt`。測試把每一行解析成位址、位元組與文字，丟掉 `#` 之後的註解，然後比較。四個 object 分別有 86、56、84 和 11 條指令，共 237 條，而 decoder 全部符合。

## Recursive descent

同一篇論文的 3.2 節把第二種方法稱為 recursive traversal。從已知的進入點開始，一直解碼到控制轉移。對於直接跳轉，繼續於目標。對於條件分支，同時繼續於目標與下一條指令，而對於 call，則在它之後繼續，因為被呼叫者會返回。遇到 return 就停止。保留已造訪位址的集合，讓走訪能夠結束。

```python
if instruction.flow == "branch":
    worklist.append(instruction.target)
    address = instruction.end
elif instruction.flow == "indirect-jump":
    table = resolver(instruction, instructions) if resolver else None
    if table is None:
        unresolved.append(instruction)
        break
```

失敗模式正好相反。Indirect jump 的位元組中沒有目標。走訪在那裡停止，只能經由該跳轉到達的每個 block 都維持未知。論文的 Figure 2 顯示第二種風險：從 relocation 推導出來的目標集合，指向指令的中間。`main.py` 中的 descent 把那種情況記錄為衝突，並停止該路徑，讓重疊看得見，而不是被默默解碼兩次。

Andriesse 等人在 981 個編譯器產生的 binary 上量測了九個 disassembler（USENIX Security 2016）。在真實的編譯器輸出中，inline 資料與重疊的程式碼，比文獻所假設的更少見，而困難的基本操作是函式邊界的還原。請記住這兩項發現。下一節的陷阱在編譯器輸出中很少見。`grade` 中的 indirect jump 則很常見。

## Sweep 錯誤而 descent 正確的 fixture (A Fixture Where the Sweep Is Wrong and the Descent Is Right)

`code/fixtures/sweep-trap.s` 是 24 位元組的手寫組合語言，函式主體中有一個四位元組的表格。函式用 `movzbl (%rcx,%rax), %eax` 索引該表格，並跳過它。

```asm
lookup:
        movslq  %edi, %rax
        leaq    .Ltable(%rip), %rcx
        movzbl  (%rcx,%rax), %eax
        jmp     .Ldone
.Ltable:
        .byte   0x31, 0xc0, 0x90, 0x48
.Ldone:
        addl    $1, %eax
        retq
```

用 `clang -target x86_64-unknown-linux-gnu -c sweep-trap.s` 建構它。四個表格位元組是刻意選擇的，讓它們解碼成看似合理的指令。`31 c0` 是 `xorl %eax, %eax`。`90` 是 `nop`。`48` 是 REX prefix，所以 sweep 把它與真正的 `addl $1, %eax` 的三個位元組結合，並印出 `addq $0x1, %rax`。擷取到的 `llvm-objdump` 輸出正是如此：

```text
       e: eb 04                        	jmp	0x14 <lookup+0x14>
      10: 31 c0                        	xorl	%eax, %eax
      12: 90                           	nop
      13: 48 83 c0 01                  	addq	$0x1, %rax
      17: c3                           	retq
```

這份清單中沒有任何東西看起來壞掉。每個位元組都解碼了，每個 mnemonic 都是真的，`retq` 也存在。這份清單仍然在三個地方是錯的。位於 `0xe` 的跳轉以 `0x14` 為目標，而清單中沒有任何指令從 `0x14` 開始。在 `0x14` 實際執行的指令是 32 位元的 `addl`，而清單在一個位元組之前顯示的是 64 位元的 `addq`。四個表格位元組以程式碼的樣子出現。

從 `lookup` symbol 開始的 recursive descent 解碼六條指令，跟隨跳轉到 `0x14`，並回報 `addl $0x1, %eax`。它沒有認領四個表格位元組中的任何一個，也沒有認領 `plus_one` 之前的八個 padding 位元組。`main.py` 中的 `trap_report()` 印出比較結果：

```json
"linearSweep": {
  "instructions": 9,
  "atJumpTarget": {"starts": "0x13", "text": "addq\t$0x1, %rax", "startsAtTarget": false},
  "claimedDataBytes": 4,
  "claimedPaddingBytes": 8
},
"recursiveDescent": {
  "instructions": 6,
  "atJumpTarget": "addl\t$0x1, %eax",
  "claimedDataBytes": 0,
  "claimedPaddingBytes": 0
}
```

測試 `test_sweep_and_objdump_share_the_same_mistake` 斷言 `llvm-objdump` 在 `0x13` 印出了 `addq $0x1, %rax`。這個 oracle 是 linear sweep，繼承了它的盲點。與 oracle 一致，檢查的是 decoder；而 descent 檢查的是哪些指令會執行。

## Basic block

Allen 在〈Control Flow Analysis〉（1970）中定義了 basic block。它是一個指令序列，只能從它的第一條指令進入，也只能從它的最後一條指令離開。Yakdan 等人在〈No More Gotos〉（NDSS 2015）的 II.A.2 節，為反編譯重述了它，並以 header node 作為函式進入點。

建構方式是對依位址排序的已解碼指令做兩趟掃描。第一趟收集 leader。Leader 是進入點、分支或跳轉的目標、表格目標，或任何控制轉移之後的指令。第二趟在每個 leader 以及位址序列中的每個缺口，開始一個新 block。

```python
for instruction in self.instructions:
    if instruction.address in leaders or current is None or current.end != instruction.address:
        current = Block(instruction.address, [])
        self.blocks[instruction.address] = current
    current.instructions.append(instruction)
```

Edge 由每個 block 的最後一條指令決定。Return 沒有後繼者。跳轉有一個，標記為 `jump`。條件分支有兩個，標記為 `taken` 和 `fallthrough`。Block 結尾的一般指令會 fallthrough 到下一個 block，這發生在下一條指令是分支目標時。Indirect jump 有其表格的目標，標記為 `table`，或沒有，而該 block 被記錄在 `unresolved` 中。

同一次掃描還產生另外兩個事實。進入點到達不了的 block 是 unreachable，而對 `-O0` 的 `grade` 做 sweep 會產生五個。起始位址落在另一個 method 的清單中某條指令內的 block，就是陷阱中的重疊案例。

## total 的 control-flow graph

`-O1` 的 `total` 是 31 位元組、13 條指令：

```text
      40: 85 f6                        	testl	%esi, %esi
      42: 7e 18                        	jle	0x5c <total+0x1c>
      44: 89 f1                        	movl	%esi, %ecx
      46: 31 d2                        	xorl	%edx, %edx
      48: 31 c0                        	xorl	%eax, %eax
      4a: 66 0f 1f 44 00 00            	nopw	(%rax,%rax)
      50: 03 04 97                     	addl	(%rdi,%rdx,4), %eax
      53: 48 ff c2                     	incq	%rdx
      56: 48 39 d1                     	cmpq	%rdx, %rcx
      59: 75 f5                        	jne	0x50 <total+0x10>
      5b: c3                           	retq
      5c: 31 c0                        	xorl	%eax, %eax
      5e: c3                           	retq
```

Leader 是 `0x40`（進入點）、`0x44`（`jle` 之後）、`0x50`（`jne` 的目標）、`0x5b`（`jne` 之後）和 `0x5c`（`jle` 的目標）。五個 block，五條 edge：

| 從 | 到 | 種類 | 原因 |
|---|---|---|---|
| `0x40` | `0x44` | fallthrough | `jle` 沒有跳，count 是正數 |
| `0x40` | `0x5c` | taken | count 是零或負數，回傳 0 |
| `0x44` | `0x50` | fallthrough | `nopw` 是 block 內的對齊，執行會穿過它 |
| `0x50` | `0x50` | taken | `jne 0x50`，迴圈 |
| `0x50` | `0x5b` | fallthrough | 離開迴圈，回傳總和 |

位於 `0x4a` 的六位元組 `nopw` 是 padding，讓位於 `0x50` 的迴圈頭落在 16 位元組邊界。它是可到達的，所以屬於 block `0x44`。最後一個 `retq` 之後位於 `0x5f` 的一位元組 `nop` 也是 padding，而沒有任何東西到達它。對函式做 sweep 會為它建立第六個 block，並由 `unreachable()` 列出。

在 `-O2`，同一個函式是 140 位元組，有兩個迴圈。位於 `0x80` 的 SSE2 主體每趟用 `paddd` 加八個值，而純量的餘數在 `0xc0` 執行。CFG 有兩條 back edge：`0x80 -> 0x80` 與 `0xc0 -> 0xc0`，測試套件檢查了這兩條。

## 用迭代演算法計算 dominator (Dominators With the Iterative Algorithm)

當從進入點到 `b` 的每條路徑都經過 `d` 時，block `d` dominate block `b`。`b` 的 immediate dominator 是 `b` 的 dominator 中離 `b` 最近的那一個。這個概念可追溯到 Prosser（1959），經由 Allen（1970），而它正是課程 34 正確放置 `if` 與 `while` 所需要的結構。

Cooper、Harvey 與 Kennedy 寫了〈A Simple, Fast Dominance Algorithm〉（2001）。他們主張，以反向 postorder 執行、搭配精簡表示法的 data-flow 公式，在約一千個節點以下的圖上，勝過 Lengauer-Tarjan（1979）。他們的論文說，其目的是「不是要提出新的演算法」。它顯示簡單的演算法在實務上可以勝出，也是很好的教學工具。函式的 CFG 幾乎從來不會有一千個節點。

```python
def intersect(first: int, second: int) -> int:
    while first != second:
        while number[first] > number[second]:
            first = idom[first]
        while number[second] > number[first]:
            second = idom[second]
    return first

changed = True
while changed:
    changed = False
    for start in order[1:]:
        processed = [p for p in self.predecessors(start) if p in idom]
        candidate = processed[0]
        for other in processed[1:]:
            candidate = intersect(other, candidate)
        if idom.get(start) != candidate:
            idom[start] = candidate
            changed = True
```

`number` 保存每個 block 的反向 postorder 位置，進入點為 0。`intersect` 用兩根手指沿著目前的 dominator 樹往上走，直到相遇。論文用 postorder 編號和 `<` 來寫比較。這份程式碼使用反向 postorder 編號，所以比較是 `>`。其他一切都遵循論文的 pseudocode。

對 `-O1` 的 `total` 而言，結果是 `idom(0x44) = 0x40`、`idom(0x50) = 0x44`、`idom(0x5b) = 0x50`，以及 `idom(0x5c) = 0x40`。Edge `0x50 -> 0x50` 是 back edge，因為它的目標 dominate 它的來源，而 back edge 標記一個迴圈（Allen 1970）。`back_edges()` 會回傳它。測試 `test_dominators_on_a_hand_built_diamond_with_a_loop` 在帶有 join point 的 16 位元組序列上執行相同的程式碼，讓 `intersect` 看到兩個前驅。

## -O0 的 grade 中的 jump table (The Jump Table in grade at -O0)

在 `-O0`，clang 把 `grade` 中的 `switch` 編譯成與位置無關的 jump table 慣用寫法：

```text
      5d: 83 e8 04                     	subl	$0x4, %eax
      60: 77 38                        	ja	0x9a <grade+0x6a>
      62: 48 8b 45 f0                  	movq	-0x10(%rbp), %rax
      66: 48 8d 0d 00 00 00 00         	leaq	(%rip), %rcx
		0000000000000069:  R_X86_64_PC32	.rodata-0x4
      6d: 48 63 04 81                  	movslq	(%rcx,%rax,4), %rax
      71: 48 01 c8                     	addq	%rcx, %rax
      74: ff e0                        	jmpq	*%rax
```

把它讀成四個事實。`subl $0x4` 後面接著 `ja` 是邊界檢查：大於 4 的 index 會去位於 `0x9a` 的 default case，所以表格有五個項目。帶有 relocation 的 `leaq` 把表格位址放進 `%rcx`，而上一節顯示該位址是 `.rodata+0x0`。`movslq` 載入以 4 為 scale 的有號 32 位元項目。`addq` 加上表格基底，所以每個項目是從表格到其 case block 的距離。

在 object file 中，每個表格位元組都是零。目標只以 `.rela.rodata` 中的 relocation 形式存在：

```text
RELOCATION RECORDS FOR [.rodata]:
OFFSET           TYPE                     VALUE
0000000000000000 R_X86_64_PC32            .text+0x91
0000000000000004 R_X86_64_PC32            .text+0x8c
0000000000000008 R_X86_64_PC32            .text+0x87
000000000000000c R_X86_64_PC32            .text+0x82
0000000000000010 R_X86_64_PC32            .text+0x86
```

對位於 offset `4k` 的項目 `k` 套用 `S + A - P`。儲存的值是 `.text + A - (.rodata + 4k)`，而程式碼把 `.rodata + 0` 加到它上面，所以目標是 `.text + A - 4k`。項目 0 是 `0x91 - 0 = 0x91`，項目 1 是 `0x8c - 4 = 0x88`，項目 2 是 `0x87 - 8 = 0x7f`。項目 3 和 4 是 `0x82 - 12 = 0x76` 與 `0x86 - 16 = 0x76`。`score / 10` 的 case 6、7、8、9 和 10 前往儲存 1、2、3、4 和 4 的 block。這與 C 原始碼逐行相符，包括 9 和 10 共用的 case。

`recover_table()` 做這項算術。`jump_table_resolver()` 從 `jmpq` 往回比對這個慣用寫法。它找到產生暫存器的 `addq`、載入它的 `movslq`、產生基底的 `leaq`，以及作為邊界的 `ja`。接著 descent 推入四個不同的目標並繼續。連結後的可執行檔，其表格中有最終的值，且沒有 relocation。同一個函式於是讀取有號 32 位元項目並加上表格位址，而 `evidence` 記錄是哪條路徑產生了目標。

來自 `python3 code/main.py cfg code/fixtures/ledger-x86_64-O0.o grade` 的 `-O0` 下 `grade` 的數字：

| 方法 | 指令 | Block | Edge | 位元組 | 錯在哪裡 |
|---|---|---|---|---|---|
| linear sweep | 34 | 9 | 7 | 118 程式碼 + 10 padding | 5 個 unreachable block，沒有進入四個 case block 的 edge，padding 被解碼成 `nopw` |
| recursive descent | 25 | 3 | 2 | 82 程式碼 | 36 個程式碼位元組從未到達，位於 `0x74` 的 `jmpq *%rax` 未解析 |
| 帶表格的 descent | 33 | 8 | 11 | 118 程式碼 | 沒有錯 |

有了表格，dominator 樹就完整了。每個 case block 的 `idom` 是 `0x62`，也就是包含該跳轉的 block。`0xa1` 的 `idom` 是進入點 `0x30`，因為從 `0x9a` 出發的 default 路徑，不經過 `0x62` 就能到達 `0xa1`。

## 最佳化器在 -O1 對 grade 做了什麼 (What the Optimizer Did to grade at -O1)

你會看到 `grade` 被描述為編譯成 jump table 的 switch。位元組說的更具體。在 `-O1` 與 `-O2`，`grade` 裡根本沒有 indirect jump：

```text
      2c: 83 f9 04                     	cmpl	$0x4, %ecx
      2f: 77 0c                        	ja	0x3d <grade+0x2d>
      31: 89 c8                        	movl	%ecx, %eax
      33: 48 8d 0d 00 00 00 00         	leaq	(%rip), %rcx
		0000000000000036:  R_X86_64_PC32	.rodata+0x10
      3a: 8b 04 81                     	movl	(%rcx,%rax,4), %eax
      3d: c3                           	retq
```

Relocation addend `0x10` 加上慣例的 4，把表格放在 `.rodata+0x14`。載入是單純的 `movl`，而那裡的位元組是 `01 00 00 00 02 00 00 00 03 00 00 00 04 00 00 00 04 00 00 00`。編譯器用五個回傳值的表格取代了五個 case block。有 relocation、有表格、有邊界檢查。控制流程是三個 block、三條 edge，加上一個條件分支。`find_value_tables()` 把它回報為 `value-table`，項目是 `[1, 2, 3, 4, 4]`，且 `evidence: "bytes"`。

`-O2` 的 `grade_label` 又更進一步。五個字串指標的 `labels` 陣列，變成位於 `.rodata+0x0` 的五個 32 位元 offset 的表格。每個項目都帶有針對字串 section 的 `R_X86_64_PC32` relocation，而函式把表格位址加到它載入的項目上。`recover_table()` 讀取那五個 relocation，並回傳 `.rodata.str1.1` 中的 offset `0, 2, 4, 6, 8`，其中放著 `F`、`D`、`C`、`B` 和 `A`。還原 jump table 的同一種算術，也能還原相對的資料表，因為兩者都是已知 offset 上的 `S + A - P`。

從這兩個 build 得到的教訓是，要說出位元組顯示的內容。Jump table 會改變 control-flow graph。Value table 會改變一次載入。把兩者都稱為「switch」，會隱藏課程 34 必須重建的差異。

## rea 與 Ghidra 在這裡給你什麼 (What rea and Ghidra Give You Here)

搭配 Ghidra 12.1.4 provider 的 rea 6.5.0，從 `procedure_assembly` 以單一字串回傳函式的 Listing 文字。`inspect_native_instruction` 回傳一條指令的位元組、長度、decoder 文字、operand token、有型別的 reference 與 flow，其中 return 是依 p-code 分類的。`xrefs` 回傳 Ghidra 的 `ReferenceManager.getReferencesTo` 所給出的已排序唯一來源位址。`analyze_function` 帶有函式的 CFG 與 `body`。`procedure_info` 以包含邊界的範圍給出 body，並警告「空隙與共用的尾端，不得被視為擁有的位元組」。

這些事實每一個，都是 Ghidra 的 recursive descent 加上它的 analyzer，序列化後的結果。rea 的分析 profile 固定了 `instruction_flow_evidence`、`function_references` 和 `jump_table_evidence` 的合約版本。因此結果會指名產生它的規則。它做不到的是展示它的推導過程。當 Ghidra 沒有找到表格時，flow 是「unknown」，而 case block 會從 caller、callee 和 CFG 中消失。

你現在有了這些工作。對同一個 object 執行 `python3 code/main.py cfg`，並將 block 起點、edge 數量與還原的表格，和 provider 回傳的比較。一致的地方，你有兩個獨立的推導。不同的地方，你有一個具體的位址可以看，這正是課程 35 記錄為未知的東西。

關於把 decoder 當作程式的一點說明。Capstone 6.0.0-Alpha10（2026-07-21）修正了 CVE-2026-55893 和 CVE-2026-55894 的越界讀寫，而 5.0.9（2026-05-28）回補了 CVE 修正。Decoder 讀取的是它自己沒有寫的位元組。15 位元組的限制、`_Cursor.byte()` 中的截斷檢查，以及拒絕猜測未知 opcode，保護 `main.py` 不受它並非為其設計的輸入所影響。

## 動手做 (Build It)

執行 demo：

```bash
cd phases/13-tools-and-protocols/33-disassembly-and-control-flow
python3 code/main.py
```

輸出是一份 JSON 報告。對每個 fixture object，它給出 oracle 的一致性。對每個函式，它給出三種方法下的指令、block 與 edge 數量。它也列出 unreachable block、未解析的 indirect jump、back edge、immediate dominator，以及任何表格。最後以陷阱比較作結。

預期的形狀：

```json
{
  "fixtures": {
    "ledger-x86_64-O0.o": {
      "oracle": "86/86 instructions agree with llvm-objdump",
      "functions": {
        "grade": {
          "range": "0x30-0xa6",
          "paddingBytes": 10,
          "linearSweep": {"instructions": 34, "blocks": 9, "edges": 7, "unreachableBlocks": ["0x76", "0x7f", "0x88", "0x91", "0xa6"], "...": "..."},
          "recursiveDescent": {"instructions": 25, "blocks": 3, "edges": 2, "unresolvedIndirectJumps": ["0x74"], "...": "..."},
          "recursiveDescentWithTables": {"instructions": 33, "blocks": 8, "edges": 11, "...": "..."},
          "tables": [{"kind": "jump-table", "table": ".rodata+0x0", "count": 5, "entries": ["0x91", "0x88", "0x7f", "0x76", "0x76"], "evidence": "relocation", "...": "..."}]
        }
      }
    }
  },
  "sweepTrap": {"...": "..."}
}
```

依此順序閱讀 `code/main.py`：

1. `ElfObject` 讀取 section header、symbol 與 `.rela.*` 項目，而 `symbolize()` 以 `llvm-objdump` 的方式為位址命名。
2. `Decoder.decode()` 讀取 prefix 與 REX，然後在單位元組或雙位元組 opcode map 上分派。
3. `_State.modrm()` 讀取 ModRM、SIB 與 displacement，並記錄 displacement 欄位的 offset。
4. `resolve_reference()` 對 displacement 或 immediate 欄位上的 relocation 套用 `S + A - P`。
5. `linear_sweep()` 與 `recursive_descent()` 是兩種方法，各自記錄其失敗。
6. `recover_table()` 與 `jump_table_resolver()` 透過 relocation 或位元組讀取表格。
7. `ControlFlowGraph` 建構 leader、block 與型別化 edge，而 `dominators()` 是迭代演算法。
8. `analyze_function()` 對一個函式執行每種方法，而 `compare_with_oracle()` 檢查 decoder。

還有另外兩個指令。`python3 code/main.py disasm FILE.o` 印出清單，在每條有 relocation 的指令之後附上 relocation 目標。`python3 code/main.py cfg FILE.o FUNCTION` 印出單一函式的分析。

## 實際使用 (Use It)

對新的 object，依此順序使用這三種方法。

1. 先執行 oracle 比較。對一條單純的指令與 `llvm-objdump` 不一致的 decoder 有 bug，而之後的每個結構都會繼承它。
2. 從每個函式 symbol 執行 recursive descent，並閱讀 `unresolved`。每個項目都是你沒有目標的 indirect jump 或 call。
3. 對每個未解析的跳轉，尋找表格慣用寫法以及指向它的 relocation。如果慣用寫法不符，就閱讀跳轉之前的位元組，並寫下你看到的內容。那份筆記就是課程 35 中的第一個未知。
4. 逐位元組比較 sweep 與 descent。只有 sweep 認領的位元組是 padding 或資料。兩者都沒認領的位元組是 unreachable，或屬於你還沒找到的函式。
5. 用 descent 加上還原的表格建構 CFG，並閱讀 back edge。每一條都是課程 34 會變成 `while` 或 `for` 的迴圈。

把 oracle 文字檔與 object 放在一起。當你換到新的編譯器版本時，請重新擷取它們，因為清單改變就是 fixture 改變。

## 互動實驗 (Interactive Lab)

### 實驗 A：在紙上解碼，然後檢查

從 `code` 目錄開啟 Python：

```bash
cd phases/13-tools-and-protocols/33-disassembly-and-control-flow/code
python3 -q
```

執行：

```python
from main import *
decode(bytes.fromhex("48 8d 0d 00 00 00 00"), 0).text
decode(bytes.fromhex("8d 44 08 fa"), 0).text
decode(bytes.fromhex("66 66 66 66 66 2e 0f 1f 84 00 00 00 00 00"), 0).length
```

在每次呼叫之前，先寫下你的預期。然後從第一個移除 `48`，看 operand 大小的改變。接著試試 `decode(bytes.fromhex("0f 0a"), 0)`，並閱讀 `DecodeError`。

### 實驗 B：lea 上的 relocation

```python
elf = load_fixture("ledger-x86_64-O0.o")
code = elf.section(".text").data
lea = decode(code, 0x66)
lea.displacement_offset
resolve_reference(elf, lea)
```

Reference 寫著 `.rodata+0x0`。把 `grade_label` 中的位址改成 `0x116`，並讀取進入 `.data` 的 reference。然後解碼位於 `0x10e` 的 `callq` 並解析它。Immediate 欄位帶有 `R_X86_64_PLT32 grade-0x4`，而解析後的目標是 `0x30`，也就是 `grade`。

### 實驗 C：看兩種方法不一致

```python
trap = load_fixture("sweep-trap.o")
text = trap.section(".text").data
[i.text for i in linear_sweep(text, 0, 0x20).instructions]
[i.text for i in recursive_descent(text, [0], 0x20).sorted_instructions()]
```

計算每份清單中的指令數，並找出從 `0x13` 開始的那一條。然後把 `sweep-trap.s` 中最後一個表格位元組從 `0x48` 改成 `0xe8`，重新建構，再把兩者各執行一次。Sweep 現在把 `addl` 和 `retq` 吞進一個目標在 section 之外的 `callq`。函式看起來沒有 return。

### 實驗 D：手動計算 dominator

```python
analysis = analyze_function(load_fixture("ledger-x86_64-O1.o"), "total")
analysis.cfg.edges
analysis.cfg.reverse_postorder()
analysis.cfg.dominators()
analysis.cfg.back_edges()
```

在查看結果之前，先在紙上畫出五個 block 並計算 dominator。然後對 `-O0` 的 `grade` 執行同樣的四行，並說明為什麼 `idom(0xa1)` 是 `0x30` 而不是 `0x62`。

## 實作練習 (Practice Lab)

在 fixture 中加入 indirect call site，並擴充分析。

需求：

- 寫一個 C 函式，透過從小型陣列中選出的函式指標來呼叫。以 `-O0` 和 `-O1` 建構它，並把 object 與它們的 `llvm-objdump` 文字一起 commit。
- 讓 decoder 在兩個 object 上都與 oracle 一致。只有當 fixture 中的指令需要時，才加入 opcode。
- 擴充 `recursive_descent()`，讓 indirect call 以 `None` 作為目標，把 call site 記錄在 `calls` 中，並在其後繼續。
- 寫一個 resolver，透過 relocation 讀取函式指標陣列，並回報可能的 callee。
- 把 callee 集合以 `indirectCalls` 加入函式報告，並為每個目標附上證據種類。
- 寫一個測試，當 resolver 回傳不是函式 symbol 的目標時就失敗。

成功的意思是，`python3 code/main.py cfg` 指名每個可能的 callee 以及證明它的 relocation，而且 oracle 比較在每個 object 上仍然通過。

## 評分實驗 (Graded Lab)

關於這堂課 fixture 的五個問題。用你建構的程式碼找出每個答案，然後寫進 `code/lab_answers.json`。數字寫成數字或像 `"0x40"` 這樣的十六進位字串，清單寫成 JSON 清單。

1. `t1`：recursive descent 在 -O1 的 grade 中解碼的指令數。
2. `t2`：-O1 的 total 中的 basic block 數。
3. `t3`：-O1 的 total 中迴圈 header 的位址，以十六進位表示。
4. `t4`：在 sweep-trap.o 中，涵蓋 0x14 的 linear sweep 指令的起始位址，以十六進位表示。
5. `t5`：-O0 的 grade 的 jump table 中的項目數。

檢查你的答案：

```bash
cd phases/13-tools-and-protocols/33-disassembly-and-control-flow
python3 code/lab.py
```

檢查器會依據 `code/main.py` 在已 commit 的 fixture 上計算每個預期值。它為每個任務印出 `PASS`、`FAIL` 或 `TODO`，以及一個分數，只有當五個全部通過時才以 0 結束。`python3 code/lab.py my-answers.json` 檢查另一個檔案。`code/tests/test_lab.py` 證明檢查器接受正確的答案，並拒絕錯誤的答案。

### 看它執行

[Inside a Decompiler](https://aiengineeringfromscratch.com/inside-a-decompiler?program=trap&sweep=linear) 這台機器把這堂課當作它的 decode bed 與 block board 來執行。這個連結以 linear sweep 開啟 `sweep-trap.o`：decode bed 顯示四個資料位元組被解碼成程式碼，而一個 pin 標記了位於 `0x13` 的 `addq` 內、`0x14` 的跳轉目標。把 decoder 切換到 recursive descent，就能看到跳轉略過資料。

## 課程成品 (Lesson Artifact)

這堂課包含 `outputs/skill-disassembly-and-control-flow.md`。用它把 object 與函式名稱，變成經過檢查的清單、block、型別化 edge、dominator，以及每個未解析的轉移。它要求 oracle 比較、兩種反組譯方法、位元組帳目、每個還原表格的 relocation 證據，以及對維持未知之處的說明。

## 驗證 (Verify It)

執行 demo 與確定性的測試套件：

```bash
cd phases/13-tools-and-protocols/33-disassembly-and-control-flow
python3 code/main.py
python3 code/tests/test_main.py
```

驗證應該證明：

- decoder 在四個 fixture object 的全部 237 條指令上，都與 `llvm-objdump -d` 一致
- REX.W 選擇 64 位元 operand，而 REX.R 與 REX.B 擴充暫存器欄位
- ModRM 與 SIB 轉譯 AT&T 的記憶體 operand，包括 scale、index 與有號的 displacement
- RIP 相對的 displacement 與它在指令內的欄位 offset 一起被記錄
- 重複的 legacy prefix 與 segment override 解碼的結果和 oracle 相同
- 強制的 `66` prefix 選擇 SSE opcode，並讓暫存器維持 32 位元
- 8 位元 immediate 在算術中做符號擴充，在 shift 次數中則為無號
- 超過 15 位元組的指令、被截斷的指令與未知的 opcode 都會丟出 `DecodeError`
- `lea` 上的 `R_X86_64_PC32` 解析到表格，而 `call` 上的 `R_X86_64_PLT32` 解析到 callee
- sweep 把陷阱表格解碼為程式碼，並且在跳轉目標沒有指令
- descent 從不認領表格或 padding，並在目標處讀到 `addl $0x1, %eax`
- `llvm-objdump` 在陷阱上犯了與 sweep 相同的錯誤
- 位於指令內部的項目被記錄為衝突
- 未解析的 indirect jump 會停止 descent 並被列出
- `-O1` 的 `total` 有五個 block、五條型別化 edge、一條 back edge，以及預期的 immediate dominator
- 手工建構、帶迴圈的菱形，給出演算法所定義的 dominator
- `-O2` 的 `total` 有兩個迴圈
- `-O0` 的 `grade` 從 `.rela.rodata` 得出五個項目的 jump table，目標為 `0x91, 0x88, 0x7f, 0x76, 0x76`
- 沒有表格時，`-O0` 的 `grade` 有三個 block 與一個未解析的跳轉
- `-O1` 與 `-O2` 的 `grade` 是 value table `[1, 2, 3, 4, 4]`，沒有 indirect jump
- `-O2` 的 `grade_label` 透過五個 relocation 讀取相對的字串表格
- 互動式圖表內嵌了分析所產生的相同資料

## Production 的失敗模式 (Production Failure Modes)

| 失敗 | 薄弱的 tool 回報什麼 | 分析必須證明什麼 |
|---|---|---|
| `.text` 內的資料 | 每個位元組都被解碼、看似合理的清單 | descent 不認領任何資料位元組，且跳轉目標是一條指令的起點 |
| 最後一個 return 之後的 padding | 函式結尾有一條 `nopw` 指令 | 這些位元組從進入點無法到達 |
| 透過表格的 indirect jump | 「jmp rax」與四個 dead block | 表格位址來自 `lea` 的 relocation，項目數來自邊界檢查，目標來自 `.rela.rodata` |
| Switch 變成 value table | 「帶 relocation 的 jump table」 | 從 `.rodata+0x14` 的 `movl` 載入、三個 block、沒有 indirect 轉移 |
| displacement 為零的 call | `callq 0x113`，也就是下一條指令 | `R_X86_64_PLT32 grade-0x4` 解析到 `0x30` |
| Decoder 猜測未知的 opcode | 看起來正確的 mnemonic | 帶有位址的 `DecodeError`，而 sweep 記錄該位元組 |
| 超過 15 位元組的指令 | 非常長的「指令」 | cursor 在讀到第 16 個位元組之前就丟出例外 |
| 分支進入指令的中間 | 兩份重疊的清單 | 帶有兩個位址的衝突紀錄 |
| 迴圈被回報為直線程式碼 | pseudocode 中沒有 `while` | 目標 dominate 來源的 back edge |
| 從不同編譯器擷取的 oracle | 測試在過期的文字上通過 | fixture 的 README 指名確切的編譯器與指令 |

## 營運規則 (Operational Rule)

對照 oracle 解碼，從已知的進入點走訪控制流程，並把每個位元組都記錄為已認領、padding、資料或 unreachable。從你能指出的 relocation 或表格，還原每個 indirect 目標。其餘的記錄為未知，並附上位址。

## 延伸閱讀 (Further Reading)

- [Intel 64 and IA-32 Architectures Software Developer's Manual, Volume 2, Chapter 2: Instruction Format](https://www.intel.com/content/www/us/en/developer/articles/technical/intel-sdm.html)
- [AMD64 Architecture Programmer's Manual Volume 3, Chapter 1: Instruction Encoding](https://docs.amd.com/v/u/en-US/24594_3.38_APM_Vol3_PUB)
- [System V Application Binary Interface, AMD64 Architecture Processor Supplement, Table 4.9: Relocation Types](https://gitlab.com/x86-psABIs/x86-64-ABI)
- [Schwarz, Debray, Andrews, "Disassembly of Executable Code Revisited" (WCRE 2002)](https://www2.cs.arizona.edu/~debray/Publications/disasm.pdf)
- [Andriesse et al., "An In-Depth Analysis of Disassembly on Full-Scale x86/x64 Binaries" (USENIX Security 2016)](https://www.usenix.org/conference/usenixsecurity16/technical-sessions/presentation/andriesse)
- [Allen, "Control Flow Analysis" (1970)](https://doi.org/10.1145/800028.808479)
- [Cooper, Harvey, Kennedy, "A Simple, Fast Dominance Algorithm" (2001)](https://www.cs.tufts.edu/~nr/cs257/archive/keith-cooper/dom14.pdf)
- [Lengauer, Tarjan, "A fast algorithm for finding dominators in a flowgraph" (1979)](https://doi.org/10.1145/357062.357071)
- [Capstone 6.0.0-Alpha10 release notes](https://github.com/capstone-engine/capstone/releases/tag/6.0.0-Alpha10)
- 這堂課所描述的 rea 6.5.0 原始碼，位於 tag `3b79af2`：`README.md`、`docs/ghidra-provider.md`、`docs/ghidra-dos.md`、`docs/reverse-engineering-tools.md`、`bridge/ghidra/ReaGhidraBridge.java`、`src/contracts/officialToolContracts.ts`、`src/ghidra/GhidraAnalysisProfile.ts`。
