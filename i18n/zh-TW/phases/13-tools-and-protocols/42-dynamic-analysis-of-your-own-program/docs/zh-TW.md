# 動態分析你自己的程式：Debugger、Trace，以及確認靜態主張 (Dynamic Analysis of Your Own Program: Debuggers, Traces, and Confirming Static Claims)

> Breakpoint 命中，是關於某個 build 的一次執行的觀察。這堂課建構的 harness，把這些觀察變成對靜態課程所提出主張的 confirmed、refuted 與 not-observed 判定。

**Type:** Build
**Languages:** Python, C, arm64 assembly, lldb command language
**Prerequisites:** Phase 13 · 33 (disassembly and control flow), Phase 13 · 35 (evidence, confidence, and unknowns)
**Time:** ~130 分鐘

## 學習目標 (Learning Objectives)

- 說明 debugger 在 Linux（`ptrace`）和 macOS（Mach task port 與 exception port）上需要作業系統提供什麼，以及為什麼有 sandbox 的 shell 無法啟動被 debug 的程式。
- 依名稱與清單位址設定 software breakpoint，設定 hardware watchpoint，並閱讀處理器對兩者施加的限制。
- 在 Python 中以 batch 模式驅動 lldb，搭配 stop hook 在每次停止時記錄暫存器、記憶體與 frame 變數，並把 transcript 解析回時間線。
- 對照該時間線，檢查課程 32、33、34 和 41 的靜態主張：引數暫存器、struct 欄位 offset、條件分支走了哪條 edge，以及發生了哪些 call。
- 對 `-O2` build 執行相同的計畫，閱讀已解析但從未命中的 breakpoint，並說明這證明了什麼、沒證明什麼。
- 儲存並載入 core 檔案，並指出 stack 風格的 core 遺漏了什麼。
- 對應 gdb 與 lldb 的指令集，並把 `strace`、`ltrace`、`bpftrace`、`dtrace` 和 `rr` 放在正確的作業系統上，附上它們的版本與授權。
- 說明讓 lldb 能 debug 你自己建構的 binary 的 macOS 程式碼簽署事實，以及它會拒絕的兩個條件。

## 問題 (The Problem)

課程 41 從 DWARF 讀取 `struct account`，並說 `balance` 位於 offset 4。課程 33 建構了 `classify` 的 control-flow graph，並說當 balance 為負時，第一個分支會去往 `return -1` block。課程 32 說前兩個引數抵達 `x0` 和 `x1`。這些句子每一個都是從位元組推導出來的。沒有任何一個是被觀察到的。

接著，agent 讀取同一個程式並回報「main 呼叫 `bump`、`classify` 和 `sum_balances`」。Symbol table 列出全部三個，所以這個句子看起來很安全。你以 `-O2` 建構程式、在每個函式上設 breakpoint，並執行它。Process 印出它的四行並結束。每個 breakpoint 都回報 `hit count = 0`。

Agent 對原始碼沒有說錯。它是對它被問到的那個 build 說錯了。編譯器 inline 了全部三個 call，並把 `sum_balances(rows, 3)` 折疊成常數 315。要知道這個 binary 在這台機器上的情況，唯一的方法是在 debugger 下執行它並觀看。

這堂課建構一個進行觀看的 harness。它編譯 `account.c`、讀取清單、規劃一次 lldb session、執行它，並對照停止點所顯示的內容為每個靜態主張評分。評分有三種結果。`confirmed` 表示某次停止顯示該主張在這次執行中為真。`refuted` 表示某次停止顯示它為假。`not-observed` 表示沒有任何停止能測試它，而紀錄說明原因。課程 35 接著把每個判定儲存為權威為 `controlled-replay` 的 `observed` 紀錄。

```figure
re-42-run-timeline
```

## Debugger 是三個系統服務加一張表 (A Debugger Is Three System Services and a Table)

Debugger 對一個它不擁有程式碼的 process 做四件事。它停止 process、讀寫它的暫存器、讀寫它的記憶體，並在 process 再次停止時得到通知。Kernel 為每一件事提供服務，而這些服務依作業系統而異。

在 Linux 上，服務是 `ptrace(2)`。手冊頁說它「提供一種方法，讓一個 process（tracer）可以觀察並控制另一個 process 的執行」。`PTRACE_TRACEME` 讓 child 宣告它的 parent 將追蹤它，`PTRACE_ATTACH` 接管一個無關的 process，`PTRACE_PEEKDATA` 讀取「tracee 記憶體中位址 addr 的一個 word」，`PTRACE_GETREGS` 讀取暫存器，而 `PTRACE_SINGLESTEP` 執行一條指令。Yama 模組限制 attach：`/proc/sys/kernel/yama/ptrace_scope` 在預設值 1 時表示「目標 process 必須是呼叫者的後代」，2 表示只有管理員可以 attach，3 表示沒有人可以。

在 macOS 上，`ptrace` 是較小的介面。Apple 的手冊列出 `PT_TRACE_ME`、`PT_ATTACHEXC`、`PT_STEP`、`PT_CONTINUE` 和 `PT_KILL`，並說使用 `PT_ATTACHEXC` 時「來自 child 的 signal 會作為 Mach exception 送給 parent」。記憶體與暫存器完全不透過 `ptrace`。lldb 的 `debugserver` 呼叫 `task_for_pid()` 取得該 process 的 Mach task port，然後呼叫 `task_set_exception_ports()`，讓 breakpoint 以 exception 訊息的形式抵達。系統 daemon `taskgated` 決定誰可以呼叫 `task_for_pid`。

你可以在這堂課保留的 transcript 中看到那個設計的代價。Session 的 shell 在 Seatbelt sandbox 下執行。在 sandbox 之外產生十一次停止的同一份計畫，在 sandbox 內產生這個：

```text
(lldb) run
error: process exited with status -1 (attach failed ((os/kern) invalid address))
```

Process 被建立了，卻無法被接管。錯誤與 debugger 的權限有關，對程式本身沒有任何說明。Harness 把它記錄為啟動失敗，零次停止，也沒有判定。

那張表就是 breakpoint 表。Debugger 記得它在哪個位址取代了哪個指令位元組，所以在你繼續執行時可以把原本的放回去，並計算每個位置的命中次數。`breakpoint list` 印出那張表，而 harness 在每次執行結束時讀取它。

## Breakpoint：取代一個位元組或一個 word (Breakpoints: One Byte or One Word Replaced)

Software breakpoint 是處理器會 trap 的指令，寫在你想停下的那條指令的前幾個位元組上。在 x86-64 上，那是 `INT3`，opcode `CC`，長度一個位元組，Intel 手冊將它描述為「用來呼叫 debug exception handler」。在 arm64 上，它是 `BRK #imm16`，一條四位元組的指令，其 exception「使用 EC 值 0x3C 記錄在 ESR_ELx 中」。Arm 手冊補充說，偏好的回傳位址就是 `BRK` 指令本身，這就是為什麼停止點回報的是被取代那條指令的位址，而不是它之後的那一條。

lldb 依名稱、依檔案與行號，或依位址設定 breakpoint。依名稱時，它會略過 prologue：

```text
(lldb) breakpoint set -n bump
Breakpoint 1: where = account-O0`bump + 12 at account.c:13:19, address = 0x00000001000004bc
```

`bump + 12` 是第四條指令。它之前的三條是 `sub sp, sp, #0x10`、`str x0, [sp, #0x8]` 和 `str w1, [sp, #0x4]`，也就是 DWARF 行表歸屬給函式開頭那一行的 frame 設定。在停止時，`x0` 和 `x1` 仍持有引數，因為 prologue 只是複製了它們。

依位址時，lldb 需要知道該位址屬於哪個 module。在 process 存在之前，單純的 `breakpoint set -a 0x100000504` 是沒有 section 支撐的載入位址，而在第一次測試中，它在整個執行期間一直維持 `unresolved`，且 `hit count = 0`。Harness 使用的形式會指名 module：

```text
(lldb) breakpoint set -a 0x100000504 -s account-O0
Breakpoint 4: address = 0x0000000100000504
```

使用 `-s` 時，數字是 `account-O0` 內的檔案位址，lldb 加上載入 slide，而執行之後該位置讀作 `resolved, hit count = 2`。課程 33 的清單給出檔案位址，所以這是把靜態清單連接到動態停止點的形式。

### 手動解碼分支 (Decode the branch by hand)

`-O0` 的 `classify` 中第一個條件分支，是位於 `0x1000004f0` 的 word `36f800a8`，`llvm-objdump` 將它印成 `tbz w8, #0x1f, 0x100000504`。從 Arm A64 對 `TBZ` 的編碼解碼它。

1. 位元 31 是 `b5`，位元編號的高位元。它是 0。
2. 位元 30 到 25 是 `011011`，test-and-branch 群組。位元 24 是 0，這選擇的是 `TBZ` 而不是 `TBNZ`。
3. 位元 23 到 19 是 `b40`，位元編號的低五位元：`11111` = 31。搭配 `b5` = 0，被測試的位元是 31，也就是 32 位元值的符號位元。
4. 位元 18 到 5 是 `imm14`：`00000000000101` = 5。分支 offset 是 `imm14 * 4` = 20 = `0x14`，所以目標是 `0x1000004f0 + 0x14 = 0x100000504`。
5. 位元 4 到 0 是 `Rt`：`01000` = 8，所以暫存器是 `w8`。

`w8` 在前一條指令由 `ldr w8, [x8, #0x4]` 載入，也就是 offset 4 的 `a->balance`。`TBZ` 在位元為零時分支。當符號位元清除時分支被採取，也就是 `balance >= 0`。原始碼寫的是 `if (a->balance < 0) return -1;`，所以編譯器反轉了條件：被採取的 edge 略過 `return -1` block，而位於 `0x1000004f4` 的 fall-through 進入它。因此 harness 帶有的靜態主張，是帶有謂詞的 edge 主張：當 `a->balance >= 0` 時採取，否則 fall-through。第二個分支的 lift 以同樣方式運作。`subs w8, w8, #0xc8` 計算 `balance - 200`，而 `b.le 0x100000524` 在其結果最多為零時被採取，也就是 `balance <= 200`。

## Watchpoint 是硬體的，而且數量很少 (Watchpoints Are Hardware and Few)

Watchpoint 在某個位址的記憶體被存取時停止 process。它不可能是被取代的指令，因為任何地方的任何指令都可能進行該存取。因此處理器帶有一小組位址比較器。在 x86-64 上，它們是 debug 暫存器：`DR0` 到 `DR3` 持有「最多 4 個 breakpoint 的位址」，而 `DR7` 啟用每一個，並設定它的條件（執行、寫入或讀寫）與長度（1、2、4 或 8 位元組）。在 arm64 上，配對是「持有用於 watchpoint 比對的資料虛擬位址」的 `DBGWVR<n>_EL1`，以及控制用的 `DBGWCR<n>_EL1`，而 Arm 手冊說「實作可以包含 2 到 64 個 watchpoint」。

lldb 回報它被給予的數量：

```text
(lldb) watchpoint set variable a->balance
Watchpoint created: Watchpoint 1: addr = 0x16fdfe00c size = 4 state = enabled type = m
(lldb) watchpoint list
Number of supported hardware watchpoints: 4
```

這台機器上是四個。被監看的位址 `0x16fdfe00c` 是 `x0 + 4`：`bump` 停止時的 struct 指標加上 `balance` 的 DWARF offset。`size = 4` 是 `int`。`type = m` 表示 modify。當 slot 用完時，lldb 會拒絕下一個 watchpoint，而且因為未對齊或不是 2 的冪的區域可能佔用兩個 slot，你可以設定的 watchpoint 數量可能小於四個。

資料 watchpoint 在存取之後才回報。Intel 手冊說「寫入記憶體的指令，會在由資料 breakpoint 產生的 debug exception 產生之前，覆寫原始資料」，而 lldb 把舊值保存在自己的快取中，以便印出兩者：

```text
Watchpoint 1 hit:
old value: 250
new value: 255
* thread #1, queue = 'com.apple.main-thread', stop reason = watchpoint 1
    frame #0: 0x00000001000004d0 account-O0`bump(a=0x000000016fdfe008, by=5) at account.c:14:12
```

停止點在 `0x1000004d0`，也就是位於 `0x1000004cc` 的 `str w8, [x9, #0x4]` 之後的指令。在你解析輸出之前，有一個怪癖值得知道。在 `watchpoint set` 之後，lldb 立刻印出 `Watchpoint 1 hit:`，舊值是空的，而 `new value: 250`。那是初始讀取，而 harness 忽略沒有舊值的命中。

## Stepping、Stop Hook 與 Batch 模式 (Stepping, Stop Hooks, and Batch Mode)

lldb 的指令語法是 `<noun> <verb> [-options] [arguments]`。Stepping 的動詞是 `thread step-in`（`s`）、`thread step-over`（`n`）、`thread step-inst`（`si`）、`thread step-inst-over`（`ni`）和 `thread step-out`（`finish`）。`thread until 12` 執行到某一行。`thread return <expr>` 以某個值提早返回。Harness 不做 step。它在規劃好的位址停止，並讓 stop hook 進行讀取：

```text
target stop-hook add -o "register read x0 x1 pc" -o "memory read -s4 -fx -c24 $x0" -o "frame variable -P 1"
```

`-o` 可以重複，而指令在每次停止時依序執行。`register read` 指名兩個引數暫存器與 program counter。`memory read -s4 -fx -c24 $x0` 從 `x0` 中的位址以十六進位印出 24 個四位元組 word，也就是 96 位元組，三列 `struct account`。`frame variable -P 1` 印出引數與區域變數，並將指標展開一層，所以 `a` 顯示它的五個欄位。

Batch 模式是 `lldb -b`。手冊說 `--batch` 執行 `-s`、`-S`、`-o` 和 `-O` 中的指令然後結束，而如果某個指令「由於 signal 或 crash」而停止，「debugger 會回到互動式提示」。Harness 把計畫寫進檔案，並用 `-s` 傳入。有一個設定很重要：`settings set interpreter.stop-command-source-on-error false`。沒有它，process 結束後的第一個 `continue` 會結束整個檔案，而 `breakpoint list` 永遠不會執行。有了它，lldb 會印出 `error: Process must be launched.` 並繼續。

關於輸出形狀的兩項事實，決定 parser 如何運作。第一，hook 的輸出在 hook 執行時印出，而帶有 `frame #0` 行的 `Process N stopped` block 在之後才印出。因此 parser 收集暫存器、記憶體與變數行到待處理的 chunk 中，並將 chunk 附加到它接下來看到的 `frame #0` 行。Program counter 位於 `dyld` 的 `_dyld_start` 的 chunk，是程式執行之前的進入點停止，沒有任何 `frame #0` 認領它。第二，帶有 `--auto-continue` 的 breakpoint 根本不回報它的停止，而 stop hook 也不會為它執行。這個 harness 的第一版使用了 auto-continue，什麼也沒記錄到。計畫改為使用一般的 breakpoint 和固定預算的 `continue` 指令。

## 計畫來自清單 (The Plan Comes From the Listing)

`Plan.commands()` 從它將要執行的 binary 的 `llvm-objdump -d` 文字建構指令檔。對 `account-O0`：

```text
breakpoint set -n bump
breakpoint set -n classify
breakpoint set -n sum_balances
breakpoint set -a 0x100000504 -s account-O0
breakpoint set -a 0x1000004f4 -s account-O0
breakpoint set -a 0x100000524 -s account-O0
breakpoint set -a 0x100000514 -s account-O0
target stop-hook add -o "register read x0 x1 pc" -o "memory read -s4 -fx -c24 $x0" -o "frame variable -P 1"
run
watchpoint set variable a->balance
watchpoint list
continue
...
breakpoint list
watchpoint list -v
```

三個名稱 breakpoint 是函式進入點。四個位址 breakpoint 是 `Listing.conditional_branches()` 在 `classify` 中找到的每個條件分支的目標與 fall-through：位於 `0x1000004f0` 的 `tbz` 與位於 `0x100000510` 的 `b.le`。Watchpoint 在第一次停止時設定，計畫預期那是 `bump`。接著是十六個 `continue` 指令，比任何執行所需的都多，而多餘的在結束之後無害地失敗。這個預算不是關於程式的知識，它只是上限。

主張由 `claims_for()` 從同樣的輸入建構：

| Id | 種類 | 主張 | 來源 |
|---|---|---|---|
| `arg-registers` | argument-register | 在每個進入點，第一批引數抵達 `x0` 和 `x1` | 課程 32 與 AAPCS64 |
| `field-offsets` | field-offset | `struct account` 是 32 位元組，`kind` 在 0、`balance` 在 4、`code` 在 8、`rate` 在 16、`label` 在 24 | 課程 41 與 DWARF dump |
| `branch-1` | branch-taken | 位於 `0x1000004f0` 的 `tbz` 在 `a->balance >= 0` 時去往 `0x100000504`，否則去往 `0x1000004f4` | 課程 33 的 edge 與課程 34 的 lift |
| `branch-2` | branch-taken | 位於 `0x100000510` 的 `b.le` 在 `a->balance <= 200` 時去往 `0x100000524`，否則去往 `0x100000514` | 課程 33 的 edge 與課程 34 的 lift |
| `calls` | call-edge | `main` 呼叫 `bump` 一次、`classify` 三次、`sum_balances` 一次 | 原始碼文字與 `nm` |
| `store-once` | store | `bump` 寫入 `a->balance` 一次，且新值是舊值加 `by` | 課程 34 的 pseudocode |
| `stride` | struct-stride | `list[i].balance` 位於 `list + 32*i + 4` | 課程 41 與 psABI 的聚合規則 |
| `output` | behaviour | 程式印出 `class 0 1 -1` 與 `sum 315` | 原始碼的算術 |

欄位 offset 來自 `struct_layout()`，它從 dSYM dump 中讀取帶有 `DW_AT_byte_size (0x20)` 的 `DW_TAG_structure_type`，以及五個帶有 `DW_AT_data_member_location` 的 `DW_TAG_member` 項目。計畫從未輸入數字 4。它是讀取的。

## -O0 的執行：十一次停止 (The Run at -O0: Eleven Stops)

第一次停止是 `bump` breakpoint。Hook 在停止 block 之前印出這個：

```text
      x0 = 0x000000016fdfe008
      x1 = 0x0000000000000005
      pc = 0x00000001000004bc  account-O0`bump + 12 at account.c:13:19
0x16fdfe008: 0x00000073 0x000000fa 0x00000009 0x00000000
0x16fdfe018: 0x00000000 0x3ff40000 0x00000719 0x00000001
0x16fdfe028: 0x0000006c 0xffffffd8 0x00000003 0x00000000
0x16fdfe038: 0x00000000 0x40000000 0x00000721 0x00000001
(account *) a = 0x000000016fdfe008 {
  kind = 's'
  balance = 250
  code = 9
  rate = 1.25
  label = 0x0000000100000719 "savings"
}
(int) by = 5
```

對照 DWARF offset 閱讀第一列的八個 word。Offset 0 的 word 0 是 `0x73`，也就是 `'s'`，即 `kind` 位元組，後面是三個 padding 位元組。Offset 4 的 word 1 是 `0xfa` = 250，即 `balance`。Offset 8 的 word 2 是 `0x9`，即 `code`，低半部是 `short`，上面有兩個 padding 位元組。Offset 12 的 word 3 是 8 位元組對齊的 `double` 之前的 padding。Offset 16 的 word 4 和 5 是 `0x3ff4000000000000`，這是 1.25 的 IEEE-754 編碼。Offset 24 的 word 6 和 7 是 `0x100000719`，即指向 `__TEXT` 的 `label` 指標。接下來八個 word 以 `0x6c` = `'l'` 和 `0xffffffd8` = -40 開頭：第三列，`loan`，晚 32 個位元組。每個數字都與 `frame variable` 一致，所以 `_check_layout()` 在這次停止把 `field-offsets` 記錄為 `confirmed`，並在 `x0` 持有 struct 指標的十一次停止中的每一次再做一次。

`x0` 等於 `a`，`x1` 等於 `by`，所以 `arg-registers` 在這個進入點被確認。Harness 只比較宣告的引數。在 `classify` 的停止，`x1` 仍讀作 `0x5`，是 `bump` 留下來的，而 `classify` 只有一個引數，所以沒有東西與它比較。沒有引數住在其中的暫存器是雜訊，而課程 32 的主張對它什麼也沒說。

第二次停止是 watchpoint，舊值 250，新值 255，`by` = 5。整個執行中只命中一次，所以 `store-once` 被確認。接著是 `classify` 的三次啟用，每一次有其進入點停止，以及一或兩次 edge 停止：

| 停止 | 位址 | 停止原因 | 它測試什麼 |
|---|---|---|---|
| 2 | `0x1000004e8` | breakpoint 2.1 | 進入點，`a->balance` = 100，`x0` = `a` |
| 3 | `0x100000504` | breakpoint 4.1 | `tbz` 被採取，100 >= 0，`branch-1` 被確認 |
| 4 | `0x100000524` | breakpoint 6.1 | `b.le` 被採取，100 <= 200，`branch-2` 被確認 |
| 5 | `0x1000004e8` | breakpoint 2.1 | 進入點，`a->balance` = 255 |
| 6 | `0x100000504` | breakpoint 4.1 | `tbz` 被採取，255 >= 0 |
| 7 | `0x100000514` | breakpoint 7.1 | `b.le` fall through，255 > 200 |
| 8 | `0x1000004e8` | breakpoint 2.1 | 進入點，`a->balance` = -40 |
| 9 | `0x1000004f4` | breakpoint 5.1 | `tbz` fall through，-40 < 0 |
| 10 | `0x100000544` | breakpoint 3.1 | `sum_balances` 進入點，`x1` = 3 |

`_check_branches()` 把停止點分組為啟用，在進入點讀取 balance，並對每個 edge 停止計算被 lift 的謂詞所預測的後繼者。五次 edge 停止，五次一致：`branch-1` 被確認三次，`branch-2` 兩次。停止 5 顯示 balance 是 255，這是 watchpoint 看到被寫入的值。這次執行是一個程式，而證據相連。

在停止 10，hook 從 `list` 讀取 24 個 word。`_check_stride()` 取 word 1、word 9 和 word 17，也就是 offset 4、36 和 68：100、255 和 -40。它把每一個與 `classify` 在同一位址看到的 balance 比較，把它們加總成 315，並把總和與程式印出的 `sum 315` 行比較。然後 process 以狀態 0 結束，而 `breakpoint list` 顯示名稱的命中次數為 1、3、1，四個 edge 為 2、1、1、1。全部八項主張最終都是 `confirmed`，背後有 25 筆紀錄，其中 23 筆綁定到某次停止，兩筆綁定到完成的執行。

## -O2 的執行：零次停止 (The Run at -O2: Zero Stops)

同一份原始碼在 `-O2` 是不同的程式。`main` 的 `llvm-objdump -d` 有四條 `bl` 指令，全部都去往 `printf`。取代 call 的是 `mov w8, #-0x1`、`mov w8, #0x1`、透過 `xzr` 的零，以及 `mov w8, #0x13b`，也就是 315。編譯器在編譯期算出了每個結果。`nm` 仍然列出 `_bump`、`_classify` 和 `_sum_balances`，因為它們是外部函式，而 out-of-line 的副本必須存在，供其他 object 呼叫。dSYM 有全部四個函式的 `DW_TAG_subprogram` 項目，以及零個 `DW_TAG_inlined_subroutine` 項目，所以連偵錯資訊也沒有記錄 inline。

這個 build 的計畫有三個名稱 breakpoint，沒有 edge breakpoint。`-O2` 的 `classify` 是六條指令：`ldr`、`cmp`、`cset`、`cmp`、`csinv`、`ret`，裡面沒有條件分支。來自 `-O0` 清單的 edge 主張，在這個 binary 中沒有位址，而 harness 不會把它們帶過來。關於 edge 的主張屬於清單中有該 edge 的 build。

執行：

```text
(lldb) breakpoint set -n bump
Breakpoint 1: where = account-O2`bump at account.c:13:16, address = 0x0000000100000460
(lldb) run
sizeof 32 align 8
offsets 0 4 8 16 24
class 0 1 -1
sum 315
Process 48444 exited with status = 0 (0x00000000)
(lldb) watchpoint set variable a->balance
error: Command requires a process which is currently stopped.
(lldb) breakpoint list
1: name = 'bump', locations = 1, resolved = 1, hit count = 0
```

每個 breakpoint 都已解析。位址存在、symbol 存在，而 lldb 在那裡寫入了 `BRK`。命中次數是零，因為沒有任何指令擷取到達該位址。判定：

| 主張 | 判定 | 紀錄 |
|---|---|---|
| `calls` | refuted | 在以狀態 0 完整執行之後，進入點停止次數 bump 0、classify 0、sum_balances 0 |
| `output` | confirmed | 四行與結束狀態 0 |
| `arg-registers` | not-observed | 沒有停止測試它，命中次數全部為 0 |
| `field-offsets` | not-observed | 沒有停止測試它 |
| `store-once` | not-observed | `bump` 中沒有停止，所以 watchpoint 從未被設定 |
| `stride` | not-observed | `sum_balances` 中沒有停止 |

要精確說明這項推翻說了什麼。主張是 `main` 呼叫這個程式中的三個函式。對這個 build 和這次執行而言，它沒有，而 harness 檢查了執行已經完成，因為在結束之前命中次數為零，只表示執行還沒有到達那個 call。這項主張沒有說這些函式是 dead code，執行也沒有這麼說：它們被匯出，其他 object 可以呼叫它們。其他四項主張既不是 confirmed，也不是 refuted。Struct 在這個 build 中仍然把 `balance` 放在 offset 4，而 `-O2` 的 DWARF 這麼說，但這次執行沒有任何停止讀取記憶體來顯示它。`not-observed` 保留了這個區別，而課程 35 讓未知保持開啟，直到某次執行真正觀察到它。

## Core dump (Core Dumps)

Core 檔案是把 process 的記憶體與暫存器寫到磁碟，讓 debugger 在程式消失之後可以讀取已停止的程式。Apple 的 `core(5)` 說「少數幾個造成異常終止的 signal，也會使 process 的記憶體內狀態紀錄被寫到磁碟」，寫進 `/cores/core.pid`，「前提是被終止的 process 在該目錄有寫入權限，而且目錄存在」。在這台機器上，`sysctl` 回報 `kern.coredump: 1` 和 `kern.corefile: /cores/core.%P`，`/cores` 存在且權限為 `drwxrwxr-t root admin`，而 shell 的 `ulimit -c` 是 0。在你用 `ulimit -c unlimited` 提高限制，或以 `launchctl limit core unlimited` 在系統範圍提高限制之前，kernel 不會寫入任何 core。

lldb 可以在沒有 crash 的情況下依需求寫出 core。在 `bump` breakpoint：

```text
(lldb) process save-core --style stack account-O0-stack.core
Saving 49152 bytes of data for memory region at 0x16fdf4000

Modified-memory or stack-memory only corefile created.  This corefile may 
not show library/framework/app binaries on a different system, or when 
those binaries have been updated/modified. Copies are not included
in this corefile.  Use --style full to include all process memory.
```

檔案是 53,248 位元組。載入它需要 binary 在旁邊：

```text
(lldb) target create "./account-O0" --core "account-O0-stack.core"
(lldb) bt
* thread #1, stop reason = ESR_EC_BRK_AARCH64 (fault address: 0x0)
  * frame #0: 0x00000001000004bc account-O0`bump(a=0x000000016fdfe008, by=5) at account.c:13:19
    frame #1: 0x0000000100000640 account-O0`main at account.c:40:5
    frame #2: 0x000000019a949d54 dyld`start + 7184
(lldb) frame variable *a
(account) *a = (kind = 's', balance = 250, code = 9, rate = 1.25, label = "")
(lldb) frame variable a->label
(const char *) a->label = 0x0000000100000719 ""
```

Backtrace 與整數欄位完好，因為它們存在於 core 所保存的 stack 上。`label` 印成空字串，但指標完好。指標 `0x100000719` 是正確的；它所指的位元組是可執行檔 `__TEXT` segment 中的字面值 `"savings"`，而 stack 風格的 core 不包含該 segment。指標是觀察。字串則不可得，而 lldb 不會捏造它。`--style full` 包含所有 process 記憶體，而在研究機器上，這個 34 KB 程式的完整 core 是 2.1 GB，因為 process 對應了共享快取。停止原因讀作 `ESR_EC_BRK_AARCH64`：core 記錄了 breakpoint 引發的 `BRK` exception。

在 Linux 上，kernel 依照 `/proc/sys/kernel/core_pattern` 寫入 core，這是一個 template，其中 `%p` 是 PID，`%e` 是指令名稱，`%s` 是 signal，`%t` 是時間，而開頭的 `|` 把 dump 用管線傳給某個程式。systemd 的發行版把那個管線用於 `systemd-coredump`，而 `coredumpctl list`、`info`、`dump` 和 `debug` 取回它所儲存的內容。`gdb prog core` 與 `lldb -c core prog` 都能開啟結果。

## gdb 與 lldb 並排比較 (gdb and lldb Side by Side)

兩個 debugger 用不同的詞做同樣的工作。Harness 中的 `COMMAND_MAP` 有 35 列，由 `python3 code/main.py commands` 印出。這堂課用到的：

| 任務 | gdb | lldb | lldb alias |
|---|---|---|---|
| 在函式處 break | `break bump` | `breakpoint set -n bump` | `b bump` |
| 在一個 module 內的位址 break | `break *0x1000004f0` | `breakpoint set -a 0x1000004f0 -s account-O0` | |
| 每次停止時執行指令 | `define hook-stop` ... `end` | `target stop-hook add -o '...'` | |
| 列出帶命中次數的 breakpoint | `info breakpoints` | `breakpoint list` | `br l` |
| 監看變數的寫入 | `watch a->balance` | `watchpoint set variable a->balance` | `wa s v a->balance` |
| 列出 watchpoint | `info watchpoints` | `watchpoint list -v` | `wa l -v` |
| 讀取一個暫存器 | `p $x0` | `register read x0` | |
| 以 word 讀取記憶體 | `x/8xw 0x16fdfe008` | `memory read -s4 -fx -c8 0x16fdfe008` | `x -s4 -fx -c8 0x16fdfe008` |
| 顯示引數 | `info args` | `frame variable --no-locals` | |
| 單步執行一條指令 | `stepi` | `thread step-inst` | `si` |
| 執行到函式返回 | `finish` | `thread step-out` | `finish` |
| 寫出 core 檔案 | `gcore core` | `process save-core --style stack core` | |
| 開啟 core 檔案 | `gdb account-O0 core` | `lldb -c core account-O0` | |

lldb 的 `p` 是 `dwim-print` 的 alias，它透過 `frame variable` 路徑印出單純的變數，對其他任何東西則退回 `expression`。`v` 是 `frame variable`。在 harness 中優先使用 `frame variable`：它讀取 DWARF 位置，並且從不在被 debug 的程式中執行程式碼。

2026-10-14 的版本與授權。lldb 帶有 LLVM 的版本號，而 llvm.org 目前的 release 是 2026-10-06 的 LLVM 23.1.3，採用 Apache License 2.0 搭配 LLVM 例外條款。Homebrew 把它作為獨立的 formula `lldb` 提供。這台機器上的 Command Line Tools 隨附 `lldb-1703.0.236.21`（Apple 自己的編號），內嵌 Python 3.9.6，而 Apple 的 lldb 已經簽署好可供 debug。GDB 18.1 於 2026-09-25 以 GPL-3.0-or-later 發布。它的發布說明列出 Windows、`.gdb_index`、AArch64 `FPMR` 和 Python `gdb.Corefile` 的變更，沒有 Darwin 項目。到 18.1 為止，GDB 對 Apple silicon process 沒有原生支援：2025 年 10 月發布了 15 個 patch 的「Initial AArch64-Darwin support」系列，2026 年 2 月，一位維護者說明了先透過著作權轉讓和 BFD patch 再落地的路線。Homebrew 的 `gdb` formula 有 arm64 的 bottle，並以偽造的 `--target=x86_64-apple-darwin20` 建構，而新增它們的 pull request 說「無法透過 GDB 來 debug aarch64-darwin 的 binary」。macOS 上任何非 Apple 的 debugger 也需要自簽憑證與 `com.apple.security.cs.debugger` entitlement，否則 `task_for_pid` 會以「(os/kern) failure (0x5)」失敗。gdb 在這門課中的位置是 Linux，其中 Debian stable 為 arm64 提供 gdb 16.3，而且 `rr` 也在那裡。

## Trace：程式向 Kernel 要求了什麼 (Tracing: What the Program Asked the Kernel For)

Debugger 顯示程式在某次停止時的狀態。Tracer 在不長時間停止程式的情況下，顯示程式對系統的作用。這些工具依作業系統而異。

| 工具 | OS | 版本 | 授權 | 它記錄什麼 |
|---|---|---|---|---|
| `strace` | Linux | 7.2，2026-08-18 | LGPL-2.1-or-later | 帶引數與回傳值的 system call |
| `ltrace` | Linux | 最後的 tarball 是 0.7.3，Debian 使用 git snapshot | GPL-2.0 | 動態函式庫 call |
| `bpftrace` | Linux | v0.27.0，2026-09-10 | Apache-2.0 | 透過 eBPF 的 kernel 與 user probe |
| `dtrace` 和 `dtruss` | macOS | macOS 26.2 中的 `Sun D 1.19` | CDDL，來自 Sun 的 OpenSolaris | D probe；`dtruss` 印出 system call |

`dtruss` 在它的手冊中說「由於這使用 DTrace，只有具 root 權限的使用者才能執行這個指令」。Apple 的 `dtrace(1)` 補充說「在預設的 System Integrity Protection 設定下，D 程式不被允許存取 kernel 位址值或 kernel 記憶體內容」，而且在 SIP 下不允許具破壞性的動作。這堂課沒有在這台機器上執行 `sudo dtruss ./account-O0`，因為 harness 在沒有 root 的情況下執行，所以課程把它記錄為未驗證。Trace 回答的問題與 harness 不同：「這次執行是否為每一行 `printf` 發出了 `write`」是 trace 的問題，「`bump` 時 `x1` 是否持有 5」是 debugger 的問題。兩者都是針對每一次執行。

## 錄製與重播 (Record and Replay)

`rr` 把一次執行錄製一次，並在 gdb 下以確定性的方式重播，附帶反向執行：對你在意的值設一個 watchpoint，而 `reverse-cont` 會向後執行到寫入處。專案把自己描述為「你把失敗錄製一次，然後以確定性的方式、想debug幾次就debug幾次那份錄製」。目前的 release 是 2025-02-13 的 5.9.0，採用 MIT 授權加上 BSD-2-Clause 區塊。它只支援 Linux。它需要硬體效能計數器來計算分支，所以它在 Intel Nehalem 及更新、特定 AMD Zen 核心，以及 Linux 下「特定 AArch64 微架構（例如 ARM Neoverse N1 或 Apple Silicon M 系列）」上執行。在虛擬機器中，計數器必須被虛擬化，而專案 wiki 說「Mac 上沒有其他虛擬化做到這點」，所以這台機器上的 Docker 和 Colima 無法執行它。Pernosco 是建立在 rr 錄製之上的商業服務。

gdb 有自己的軟體錄製器。`record full` 記錄「process 執行的日誌」，預設上限是 200,000 條指令，而 `record btrace` 使用 Intel 的分支追蹤並「不記錄資料」。兩者在 lldb 中都不可用。在這台 macOS 機器上，harness 儲存的 transcript 就是紀錄：同一份計畫在同一個 binary 上以相同順序產生相同的停止點，而 fixture 測試固定了那個順序。

## 一次執行證明什麼 (What One Run Proves)

一次執行證明一個 build 上的一個輸入。C 標準將未定義行為定義為「對於不可攜或錯誤的程式結構，或錯誤的資料，本文件不施加任何要求的行為」，所以讓有號 `int` 溢位的程式，在 `-O0` 與 `-O2` 可以合法地印出不同的結果。課程 36 在 `percent_change` 的溢位格子中遇到這點。UBSan 把這個問題變成停止點：`-fsanitize=signed-integer-overflow -fsanitize-trap=undefined` 讓溢位「執行 trap 指令」，並在 lldb 中以帶有原因的停止點出現。

最佳化的程式碼也會移動你想監看的東西。GDB 的手冊說「最佳化器重新安排了你的程式碼；debugger 顯示的是真正在那裡的東西」，並說被最佳化掉的變數是「GDB 永遠看不到」的。GDB 把這種值印成 `<optimized out>`，而它的原始碼把它與 `<unavailable>` 分開：optimized out「實際上並不存在於程式中」，unavailable「存在於程式中（在機器層級），但 GDB 沒有辦法取得它」。當 program counter 在變數的 DWARF 位置清單的所有範圍之外時，lldb 印出 `<variable not available>`。對住在暫存器中的區域變數設定 watchpoint 永遠不會觸發，因為暫存器不是記憶體。上面的 `-O2` 執行顯示了極端情況：整個函式被折疊掉，而一個從未被設定的 watchpoint 的沉默，什麼也證明不了。

因此 harness 寫三種紀錄，絕不寫第四種。測試了主張並一致的停止是 `confirmed`。測試了並不一致的停止是 `refuted`。沒有任何停止能測試的主張是 `not-observed`，並附上原因，而 breakpoint 的命中次數是該原因的一部分。課程 35 的 ledger 把前兩種接受為權威為 `controlled-replay` 的 `observed` 紀錄，並把第三種保留為開放的未知。

## 在 macOS 上只 debug 你自己的 build (Debugging Only Your Own Builds on macOS)

在 Apple silicon 上，linker 替你簽署。Apple 的 `ld(1)` 對 `-adhoc_codesign` 說：「Apple Silicon binary 的預設是 ad-hoc 簽署。」fixture 上的 `codesign -dvvv account-O0` 印出：

```text
Format=Mach-O thin (arm64)
CodeDirectory v=20400 size=387 flags=0x20002(adhoc,linker-signed) hashes=9+0 location=embedded
Signature=adhoc
TeamIdentifier=not set
```

`0x20002` 是 `CS_ADHOC`（`0x2`）加上 `CS_LINKER_SIGNED`（`0x20000`），在 Apple 的程式碼簽署 header 中是「由 linker 自動簽署」。`codesign -d --entitlements -` 在可執行檔名稱之後什麼也沒印出：這個 binary 沒有 entitlement。lldb 啟動了它、命中了每個 breakpoint、設定了 hardware watchpoint，並寫出了 core，而 `DevToolsSecurity -status` 回報「Developer mode is currently disabled」，且 System Integrity Protection 為啟用。

有兩個 entitlement 控制誰可以 debug 什麼，而它們位於相對的兩側。`com.apple.security.cs.debugger` 是給 debugger 的：「一個布林值，指示 app 是否為 debugger，並且可以 attach 到其他 process 或取得 task port」。Apple 的 `debugserver` 帶有它的私有形式，而對 Command Line Tools 副本執行 `codesign -d --entitlements -` 會列出 `com.apple.private.cs.debugger`、`com.apple.private.set-exception-port`、`com.apple.private.thread-set-state` 和 `com.apple.private.logging.diagnostic`。`com.apple.security.get-task-allow` 是給目標的。Xcode 把它注入 debug build，在匯出時剝除它，而公證會拒絕仍然帶有它的 binary。Apple 的工程師陳述拒絕的規則：當目標啟用 hardened runtime 且沒有 `get-task-allow` 時，task port 會被拒絕。普通的 `clang` build 沒有 hardened，兩個條件都不會觸發。如果你曾經用 `codesign --options runtime` 簽署 fixture，請在 debug 之前用 `codesign -s - -f --entitlements get-task-allow.plist` 重新簽署它。

有三項事實為這一節作結。第一，在登入 session 中第一次時，你可能會看到對話框「Developer Tools Access needs to take control of another process for debugging to continue」；`sudo DevToolsSecurity -enable` 為 `admin` 或 `_developer` 群組的成員移除額外的密碼。第二，你不需要停用 SIP 來 debug 自己的程式碼。Apple 說「你不需要停用 SIP 就能從 Xcode 執行並 debug app」，而唯一與 debug 有關的 SIP 元件，是 `csrutil enable --without debug` 為其他人的 process 放寬的那一個。第三，這個 session 中的 Seatbelt sandbox 完全封鎖 `task_for_pid`，而 fixture `sandbox-O0.lldb.txt` 顯示了結果。無法啟動的執行是環境事實，而 harness 把它回報為環境事實。

這堂課的一切都適用於你編譯的程式。同樣的指令只有在另一個 process 的擁有者選擇加入時才能 attach 到它，而這裡沒有任何東西教你如何改變這點。

## 動手做 (Build It)

執行 demo：

```bash
cd phases/13-tools-and-protocols/42-dynamic-analysis-of-your-own-program
python3 code/main.py
```

輸出是一份針對已儲存 transcript 的 JSON 報告。對每個 build，它列出帶有解析狀況與命中次數的 breakpoint、hardware watchpoint 的數量、每次停止及其暫存器與引數、結束狀態、程式的輸出、lldb 印出的錯誤，以及帶有判定與背後證據紀錄的主張。它還加入 core 的來回、Python API 的結果、sandbox 失敗，以及這台機器上缺少哪些外部工具。

預期的形狀：

```json
{
  "program": "account.c",
  "variants": {
    "O0": {
      "binary": "account-O0",
      "breakpoints": {"1": {"spec": "name=bump", "address": "0x1000004bc", "resolved": true, "hits": 1}, "...": "..."},
      "hardwareWatchpoints": 4,
      "watchpoints": {"1": {"address": "0x16fdfe00c", "size": 4, "type": "m", "hits": [[250, 255]], "hitCount": 1}},
      "stops": [{"index": 0, "pc": "0x1000004bc", "function": "bump", "kind": "entry", "registers": {"x0": "0x16fdfe008", "x1": "0x5"}, "...": "..."}],
      "exitStatus": 0,
      "claims": [{"id": "arg-registers", "verdict": "confirmed", "confirmed": 5, "refuted": 0, "not-observed": 0, "...": "..."}],
      "evidence": [{"claim": "arg-registers", "verdict": "confirmed", "stop": 0, "observation": "bump: x0 = 0x16fdfe008 and a = 0x000000016fdfe008, x1 = 0x5 and by = 5"}]
    },
    "O2": {"stops": [], "claims": [{"id": "calls", "verdict": "refuted", "...": "..."}], "...": "..."}
  },
  "core": {"savedAt": {"function": "bump", "pc": "0x1000004bc"}, "labelString": "", "...": "..."},
  "types": {"python": "3.9.6", "size": 32, "...": "..."},
  "sandbox": {"stops": 0, "errors": ["process exited with status -1 (attach failed ((os/kern) invalid address))"]},
  "absentTools": ["bpftrace", "gdb", "llvm-objdump", "ltrace", "rr", "strace"]
}
```

依此順序閱讀 `code/main.py`：

1. `Listing` 把 `llvm-objdump -d` 的文字解析成函式與指令，而 `Function.conditional_branches()` 找到帶有目標與 fall-through 的分支。
2. `struct_layout()` 從 `llvm-dwarfdump` 文字讀取 structure 與成員項目。
3. `lift_condition()` 把一個條件分支變成謂詞，也就是課程 34 對 `tbz`、`cbz` 與 compare-and-branch 形式的步驟。
4. `Plan.commands()` 從清單寫出 lldb 指令檔，而 `Plan.lldb_command()` 是 batch 的呼叫。
5. `parse_transcript()` 依序走訪輸出，並建構帶有暫存器、記憶體、變數與 watchpoint 值的 `Stop` 紀錄，加上 breakpoint 表格與結束狀態。
6. `claims_for()` 陳述八項主張，而 `evaluate()` 用 `_check_*` 函式逐次停止為它們評分。
7. `rollup()` 把紀錄化簡為每項主張一個判定，其中 refuted 優先於 confirmed。
8. `capture()` 與 `capture_extras()` 產生了 fixture，而當 `clang` 和 `lldb` 可以執行時，`live()` 會在全新的 build 上重複整個迴圈。

還有另外四個指令。`python3 code/main.py plan O0` 印出某個 build 的指令檔。`python3 code/main.py commands` 印出 gdb 與 lldb 的表格。`python3 code/main.py tools` 回報 `PATH` 上有哪些外部工具。`python3 code/main.py live O0` 把 `account.c` 編譯到暫存目錄、根據它全新的清單進行規劃、執行 lldb，並對照那次執行為主張評分。在 sandbox 內，它回報 lldb 無法啟動 process，附上錯誤，並以 1 結束。

## 實際使用 (Use It)

依此順序，在你自己的程式上使用 harness。

1. 用 `-g` 與你將交付的最佳化層級建構它。把 binary、它的 dSYM 或它的 DWARF，以及 `llvm-objdump -d` 的文字放在一起。
2. 在執行之前先寫下主張。每一項都指名一個函式、一個暫存器、一個位址或 offset、一個預期的值或 edge，以及它來自的課程或文件。你在看到停止點之後才寫的主張，描述的是停止點，什麼也沒測試。
3. 從清單產生計畫並閱讀它。其中的每個位址都應該追溯到清單中的一行。如果某個 breakpoint 沒有這樣的行，就移除它。
4. 以 batch 模式執行一次，並把 transcript 連同其 header 中的指令列與 debugger 版本一起保存。Transcript 就是證據。判定是從它導出的，而任何人都可以重新導出。
5. 評分。在閱讀確認之前，先閱讀每一筆 `not-observed` 紀錄及其原因。這些原因告訴你哪些主張需要另一個輸入、另一個 build，或另一個 breakpoint。
6. 對最佳化的 build 以相同的主張執行。在編譯器 inline 的地方，預期會有已解析但零命中的 breakpoint，並讓那些主張保持開啟。
7. 把 confirmed 與 refuted 的紀錄餵進課程 35 的 ledger，作為權威為 `controlled-replay` 的 `observed`，每次停止一筆紀錄。把 not-observed 的留作未知，並附上 breakpoint 的命中次數。

## 互動實驗 (Interactive Lab)

### 實驗 A：手動讀取一次停止

從 `code` 目錄開啟 Python：

```bash
cd phases/13-tools-and-protocols/42-dynamic-analysis-of-your-own-program/code
python3 -q
```

執行：

```python
from main import *
t = load_transcript("session-O0.lldb.txt")
s = t.stops[0]
hex(s.pc), s.function, s.reason, s.registers
[hex(w) for w in s.memory[:8]]
s.struct_pointer().fields
```

在第二行之前，先寫下你預期哪個 word 持有 250，以及哪兩個 word 持有 1.25。然後執行 `struct.unpack("<d", struct.pack("<II", s.memory[4], s.memory[5]))` 並比較。接著查看 `t.stops[1].watch_hit` 和 `t.stops[1].pc`，並在 `account-O0.objdump.txt` 中找到該位址：它是 store 之後的指令。

### 實驗 B：分支與它的後繼者

```python
plan = make_plan("O0")
plan.branches()
classify = plan.listing.function("classify")
[lift_condition(classify, b, "a->balance")[0] for b in plan.branches()]
[(hex(s.pc), s.struct_pointer().field_int("balance")) for s in t.stops_in("classify")]
```

把每次 edge 停止與其進入點停止配對，並在查看證據之前，決定謂詞預測哪個後繼者。然後執行 `[e.to_dict() for e in evaluate(plan, t, struct_layout(open("fixtures/account-O0.dwarfdump.txt").read(), "account")) if e.claim.startswith("branch")]` 並比較。

### 實驗 C：從未停止的執行

```python
o2 = load_transcript("session-O2.lldb.txt")
len(o2.stops), o2.exit_status, o2.hits()
o2.errors[:3]
make_plan("O2").branches()
rollup(evaluate(make_plan("O2"), o2, struct_layout(open("fixtures/account-O2.dwarfdump.txt").read(), "account")))
```

用你自己的話說明每一個 `not-observed`，然後開啟 `account-O2.objdump.txt`，找出 `main` 中取代那些 call 的四個常數。接著用 `count_tags()` 計算 `account-O2.dwarfdump.txt` 中 `DW_TAG_inlined_subroutine` 的數量。

### 實驗 D：破壞證據

```python
text = open("fixtures/session-O0.lldb.txt").read().replace("x1 = 0x0000000000000005", "x1 = 0x0000000000000006", 1)
broken = parse_transcript(text)
[e.to_dict() for e in evaluate(plan, broken, struct_layout(open("fixtures/account-O0.dwarfdump.txt").read(), "account")) if e.claim == "arg-registers"][:2]
```

一個被編輯的暫存器，把第一筆 `arg-registers` 紀錄變成 `refuted`，rollup 也隨之改變，而其他四個項目仍然是 `confirmed`。然後刪除 watchpoint 停止之前的 `old value: 250` 行，看 `store-once` 掉到 `not-observed`。無法被弄失敗的評分器，就不是在評分。

如果 lldb 可以在你的 shell 中啟動 process，最後請執行 `python3 main.py live O0` 和 `python3 main.py live O2`。Stack 位址會與 fixture 不同。判定則不應該不同。

## 實作練習 (Practice Lab)

用一項關於 `sum_balances` 中迴圈的主張來擴充 harness。

需求：

- 用 `conditional_branches()` 在 `-O0` 清單中找到 `sum_balances` 的迴圈 back edge，並在迴圈 header 加入 breakpoint。
- 加入一項主張：對 `count` = 3，header 執行 `count + 1` 次，這是課程 33 的 back edge 加上離開測試。
- 擴充 stop hook 或計畫，讓 hook 在每次 header 停止時讀取迴圈計數器的 stack slot，並寫一個 `_check_loop()` 記錄計數器值的序列。
- 在 lldb 可以啟動的機器上用 `python3 main.py capture O0` 重新擷取 transcript，並重新執行測試套件。在已 commit 的指令檔符合新計畫之前，計畫測試會失敗，這正是該測試的用意。
- 新增一個測試，編輯 transcript 中的一個計數器值，並斷言主張變成 `refuted`。

成功的意思是，報告依序顯示 header 的停止及其計數器值，而 `-O2` 的執行讓新的主張維持 `not-observed`，原因是迴圈被向量化或折疊掉了。

## 課程成品 (Lesson Artifact)

這堂課包含 `outputs/skill-dynamic-analysis-of-your-own-program.md`。用它把你建構的程式、它的清單，以及一份靜態主張清單，變成一份儲存的 debugger transcript，以及每項主張一個判定，附上產生每個判定的停止點，以及每個維持 not-observed 的主張的原因。它適用於 Claude、Cursor、Codex、OpenClaw、Hermes，或任何讀取 SKILL.md 的 agent。

## 驗證 (Verify It)

執行 demo 與確定性的測試套件：

```bash
cd phases/13-tools-and-protocols/42-dynamic-analysis-of-your-own-program
python3 code/main.py
python3 -m unittest discover -s code/tests -v
```

測試套件在沒有 lldb 的情況下執行。它讀取一次擷取到 `code/fixtures/` 的 transcript。驗證應該證明：

- 清單 parser 找到四個函式及其範圍，以及 `-O0` 的 `classify` 的兩個條件分支及其目標與 fall-through
- `tbz w8, #0x1f` 的 lift 是 `a->balance >= 0`，而 `subs` 加 `b.le` 的 lift 是 `a->balance <= 200`
- `-O0` 的 `main` 呼叫 `bump`、`classify` 三次與 `sum_balances`，而 `-O2` 的 `main` 只呼叫 `printf`
- x86-64 的 `j<cc>` mnemonic 是條件式的，而 `jmp` 與 `callq` 不是
- DWARF dump 產生 32 位元組的 struct，成員位於 0、4、8、16 和 24
- `-O2` 的 DWARF 有四個 subprogram 和零個 inlined-subroutine 項目，而 `nm` 仍然列出三個函式
- 兩個 build 的計畫都等於已 commit 的指令檔，而 transcript header 記錄了執行它的 lldb 指令
- `-O0` 的 transcript 在預期的函式中有十一次停止、結束狀態 0 與四行輸出
- hook 輸出被附加到隨後的停止，其 `x1` = 5、記憶體從 `x0` 開始，並有 label 字串
- breakpoint 表格帶有解析狀況，以及命中次數 1、3、1、2、1、1、1
- watchpoint 是 `x0 + 4` 處的 4 位元組 modify 型別，命中一次，舊值 250，新值 255，並回報四個硬體 slot
- `-O2` 的 transcript 有零次停止、三個已解析但零命中的 breakpoint、沒有 watchpoint，以及每個未使用的 `continue` 一個錯誤
- 在 sandbox 中的啟動，在任何停止之前就以 attach 錯誤失敗
- core 來回保留 backtrace 與 `balance` = 250，並失去 `label` 背後的字串
- Python API 在沒有 process 的情況下，讀取五個欄位、大小 32 與函式範圍
- 手寫的 x86-64 transcript 以 `rdi`、`rsi` 和 `rip` 解析
- 每項主張在 `-O0` 都是 confirmed，紀錄數量為 5、11、3、2、1、1、1、1
- 分支證據指名值、條件與 edge
- 欄位 offset 證據透過 DWARF 版面解碼 word
- store 與 stride 證據帶有舊值與新值，以及三列的 balance
- `-O2` 的執行推翻 call 的 edge，並讓其他主張維持 not-observed
- 被編輯的暫存器值是 refuted，被刪除的 watchpoint 命中是 not-observed，且 refuted 贏得 rollup
- 指令對照表為每一列都有 gdb 與 lldb 的形式，而硬體表格指名四個 x86 debug 暫存器
- demo 列出缺少的工具而不是失敗，而 live 模式以名稱回報缺少的工具
- 圖表資料來自兩個 session，而網站模組內嵌相同的資料

## Production 的失敗模式 (Production Failure Modes)

| 失敗 | 薄弱的報告說什麼 | Harness 必須證明什麼 |
|---|---|---|
| 已解析但零命中的 breakpoint | 「函式有執行」或「函式是 dead」 | 帶有結束狀態的完整執行、命中次數，以及僅針對 call edge 的 `refuted` 判定 |
| 在錯誤的停止點讀取暫存器 | 「`classify` 內的 `x1` = 5」 | 只在進入點停止時比較宣告的引數 |
| 沒有 offset 的 struct 讀取 | 一欄十六進位 word | 每個 word 都對應到 `DW_AT_data_member_location`，並與 `frame variable` 比較 |
| 從 mnemonic 為分支評分 | 「`tbz` 表示為負時採取」 | 從編碼得到的 lift 謂詞，以及在後繼者處的停止 |
| 對住在暫存器的區域變數設 watchpoint | 「沒有發生寫入」 | watchpoint 被設在某個位址並被命中，否則紀錄是 `not-observed` |
| Auto-continue 的 breakpoint | 沒有停止的乾淨 transcript | 一般 breakpoint、stop hook 與 continue 預算 |
| 沒有 module 的位址 breakpoint | `unresolved, hit count = 0` | `-s MODULE`，讓清單位址解析成載入位址 |
| 第一個指令錯誤就結束 batch | 結尾沒有 `breakpoint list` | `interpreter.stop-command-source-on-error false` |
| Stack 風格 core 被讀成完整記憶體 | 把 `label = ""` 回報為程式的值 | core 風格、指標，以及它指向的 segment |
| 有 sandbox 的 shell | 「lldb 壞了」 | 記錄為環境事實的 `attach failed ((os/kern) invalid address)` |
| 沒有 `get-task-allow` 的 hardened 目標 | 「macOS 禁止 debug」 | Apple 規則的兩個條件，以及重新簽署你自己的 binary |
| Transcript 在評分後被丟棄 | 沒有來源的判定 | transcript 檔案、它的指令列與 debugger 版本 |

## 營運規則 (Operational Rule)

根據清單規劃停止點，以 batch 模式執行一次，保留 transcript，並在每個能測試它的停止點為每項靜態主張評分。只從停止點寫 confirmed 或 refuted，寫 not-observed 時附上命中次數與原因，並且絕不讓已解析的 breakpoint 取代一次執行。只 debug 你自己做的 build。

## 延伸閱讀 (Further Reading)

- [LLDB Tutorial: command grammar, breakpoints, watchpoints, stop hooks](https://lldb.llvm.org/use/tutorial.html)
- [lldb(1) manual page: --batch, --one-line, --source, --core](https://lldb.llvm.org/man/lldb.html)
- [GDB to LLDB Command Map](https://lldb.llvm.org/use/map.html)
- [LLDB Python Reference: SBTarget, SBFrame, SBValue, and the embedded interpreter](https://lldb.llvm.org/use/python-reference.html)
- [LLDB Caveats: the Python build lldb links against](https://lldb.llvm.org/resources/caveats.html)
- [LLVM 23.1.3 release](https://github.com/llvm/llvm-project/releases/tag/llvmorg-23.1.3)
- [GDB 18.1 release announcement, 2026-09-25](https://sourceware.org/pipermail/gdb-announce/2026/000150.html)
- [GDB manual: Process Record and Replay](https://sourceware.org/gdb/current/onlinedocs/gdb.html/Process-Record-and-Replay.html)
- [GDB manual: Debugging Optimized Code](https://sourceware.org/gdb/current/onlinedocs/gdb.html/Optimized-Code.html)
- [GDB wiki: Permissions on Darwin, code signing a debugger](https://sourceware.org/gdb/wiki/PermissionsDarwin)
- [gdb-patches: Initial AArch64-Darwin support series, October 2025](https://sourceware.org/pipermail/gdb-patches/2025-October/222196.html)
- [Homebrew gdb formula pull request adding arm64 bottles](https://github.com/Homebrew/homebrew-core/pull/209753)
- [ptrace(2), Linux manual page](https://man7.org/linux/man-pages/man2/ptrace.2.html)
- [lldb debugserver MachTask.mm: task_for_pid and task_set_exception_ports](https://github.com/llvm/llvm-project/blob/main/lldb/tools/debugserver/source/MacOSX/MachTask.mm)
- [Intel SDM Volume 2: INT n, INTO, INT3, INT1](https://www.intel.com/content/www/us/en/developer/articles/technical/intel-sdm.html)
- [Intel SDM Volume 3: debug registers DR0 to DR7 (Debug, Branch Profile, TSC, and Resource Monitoring Features)](https://www.intel.com/content/www/us/en/developer/articles/technical/intel-sdm.html)
- [Arm A64 ISA: BRK, Breakpoint instruction](https://developer.arm.com/documentation/ddi0602/latest/Base-Instructions/BRK--Breakpoint-instruction-)
- [Arm A64 ISA: TBZ, Test bit and branch if zero](https://developer.arm.com/documentation/ddi0602/latest/Base-Instructions/TBZ--Test-bit-and-branch-if-zero-)
- [Arm Architecture Reference Manual, D2.9 Watchpoint exceptions](https://developer.arm.com/documentation/ddi0487/latest/)
- [Apple ld(1): -adhoc_codesign](https://keith.github.io/xcode-man-pages/ld.1.html)
- [Apple ld64 LinkEdit.hpp: CS_LINKER_SIGNED](https://github.com/apple-oss-distributions/ld64/blob/59a99ab60399c5e6c49e6945a9e1049c42b71135/src/ld/LinkEdit.hpp)
- [Apple: Debugging Tool Entitlement, com.apple.security.cs.debugger](https://developer.apple.com/documentation/bundleresources/entitlements/com.apple.security.cs.debugger)
- [Apple: Hardened Runtime](https://developer.apple.com/documentation/security/hardened-runtime)
- [Apple: Resolving common notarization issues, the get-task-allow entitlement](https://developer.apple.com/documentation/security/resolving-common-notarization-issues)
- [Apple: Disabling and Enabling System Integrity Protection](https://developer.apple.com/documentation/security/disabling-and-enabling-system-integrity-protection)
- [Apple Developer Forums: debugserver may not get the taskport of a hardened process](https://developer.apple.com/forums/thread/127544)
- [Apple DevToolsSecurity(8)](https://keith.github.io/xcode-man-pages/DevToolsSecurity.8.html)
- [Apple core(5)](https://keith.github.io/xcode-man-pages/core.5.html)
- [Apple Technical Note TN2124: Mac OS X Debugging Magic, core dumps](https://developer.apple.com/library/archive/technotes/tn2124/_index.html)
- [lldb `process save-core` help text](https://github.com/llvm/llvm-project/pull/143126)
- [core(5), Linux manual page](https://man7.org/linux/man-pages/man5/core.5.html)
- [coredumpctl(1)](https://man7.org/linux/man-pages/man1/coredumpctl.1.html)
- [Apple dtrace(1)](https://keith.github.io/xcode-man-pages/dtrace.1.html)
- [Apple dtruss(1m)](https://keith.github.io/xcode-man-pages/dtruss.1m.html)
- [strace releases](https://github.com/strace/strace/releases)
- [ltrace](https://ltrace.org/)
- [bpftrace CHANGELOG](https://github.com/bpftrace/bpftrace/blob/master/CHANGELOG.md)
- [rr project](https://rr-project.org/)
- [rr 5.9.0 release](https://github.com/rr-debugger/rr/releases/tag/5.9.0)
- [rr wiki: Building And Installing, hardware and VM requirements](https://github.com/rr-debugger/rr/wiki/Building-And-Installing)
- [Clang UndefinedBehaviorSanitizer](https://clang.llvm.org/docs/UndefinedBehaviorSanitizer.html)
- [ISO/IEC 9899:2024 working draft N3220, clause 3.5 undefined behavior](https://www.open-std.org/jtc1/sc22/wg14/www/docs/n3220.pdf)
- [DWARF Debugging Information Format Version 5](https://dwarfstd.org/doc/DWARF5.pdf)
- [Procedure Call Standard for the Arm 64-bit Architecture (AAPCS64)](https://github.com/ARM-software/abi-aa/blob/main/aapcs64/aapcs64.rst)
- [System V Application Binary Interface, AMD64 Architecture Processor Supplement](https://gitlab.com/x86-psABIs/x86-64-ABI)
