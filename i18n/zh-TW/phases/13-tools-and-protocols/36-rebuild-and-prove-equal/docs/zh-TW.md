# 重建並證明相等：差異測試、位元組比較與證據迴圈 (Rebuild and Prove Equal: Differential Tests, Byte Comparison, and the Evidence Loop)

> 通過你的測試，只證明在你的測試上一致，所以相等需要一個涵蓋每個邊緣的格子，以及在遮蔽 relocation 後仍相符的位元組。

**Type:** Build
**Languages:** Python, C
**Prerequisites:** Phase 13 · 34 (lifting to pseudocode), Phase 13 · 35 (evidence, confidence, and unknowns)
**Time:** ~110 分鐘

## 學習目標 (Learning Objectives)

- 根據指令清單寫出重建版本，並指出它來自哪一筆證據紀錄。
- 建構輸入格，每種風險各有一個類別：zero guard、負值、截斷除法、32 位元溢位，以及除法 trap。
- 在同一個程式中執行原始版本與重建版本，並記錄每個格子，包括 trap。
- 用原始的 flag 重新編譯重建版本，並在遮蔽 relocation 欄位之後比較函式位元組。
- 讀取 relocation 所指向的常數並加以比較，讓錯誤的字面值無法通過。
- 用小型的 stdio client 驅動 rea 的 MCP server，並記錄每一步與其證據 id。
- 產生 fail-closed 的證明紀錄，其 id 是對它的主張與證據連結所做的 hash。

## 問題 (The Problem)

你向 rea 要我們 `ledger-x86_64-O1.o` fixture 中 `percent_change` 的函式檔案。Ghidra 12.1.4 回傳 pseudocode，而 agent 根據它寫出這段 C：

```c
int percent_change(int before, int after) {
    if (before == 0) return 0;
    return (int)(((long)after - before) * 100 / before);
}
```

四個測試通過：`(100, 150)` 得到 50、`(200, 100)` 得到 -50、`(0, 5)` 得到 0、`(50, 100)` 得到 100。Agent 也把 `grade` 重建成針對 90、80、70 和 60 的四個 `if` statement。另外六個測試通過。報告說「verified」。

那份報告沒有回答以下任何問題：

- 當 `(after - before) * 100` 放不進 32 位元時，原始版本回傳什麼？
- 對於分數 110，原始版本回傳什麼？
- 重建版本是否編譯成與原始版本相同的 16 個位元組？
- 當兩個編譯後的函式持有相同的位元組時，它們的 relocation 是否指向相同的常數？
- 重建版本來自哪一筆證據紀錄？agent 讀的是清單還是 pseudocode？
- 在沒有 C 編譯器的 host 上，這個證明會得到什麼狀態？

這堂課的每個目標，都是我們自己寫並編譯的程式。課程 32 的範圍與法律規則維持不變。

```figure
re-36-prove-equal
```

## 證據怎麼說 (What the Evidence Says)

`code/main.py` 中的 agent 迴圈，透過 stdio 對 rea 6.5.0 發出六次呼叫。已記錄的執行是 `code/fixtures/runs/agent-run.json`。Server 以 protocol version `2025-11-25` 回應 `initialize`。三次呼叫回傳了 Evidence 紀錄：

| 步驟 | Tool | 時間 | Evidence id |
|---|---|---|---|
| 2 | `analyze_function percent_change` | 6.31 秒 | `ev_e88b10dc0c6400e51bb6f7fdc3eaea8d2c97cc02463f0db3e6afeefc273197cd` |
| 3 | `analyze_function grade` | 0.02 秒 | `ev_67b6d3f992eaf49a2e16de084b6ebfdd9884e8c5c7f75e175b7bb20f4f748ce1` |
| 4 | `read_bytes 0x1000b4, 20` | 0.00 秒 | `ev_c26146cfedf4c1ee146d8ff00cfa9a5d3f5335bdd9c97ccf4e1af5f6ea3ed6bc` |

第一次呼叫負擔了 Ghidra 的匯入與自動分析。每筆紀錄都帶有 `confidence: "observed"`、`authority: "shipped-artifact"`，以及 object 的 SHA-256 `9a4a8082ba5e285a732552981a6935e0c5255725664f380e37c44ebfb04d803f`。Ghidra 載入可重定位的 object 時，把 `.text` 放在 `0x100000`，所以 `percent_change` 位於 `0x100000`，而 `grade` 位於 `0x100010`。

`percent_change` 的 pseudocode 就是 `long` 的來源：

```c
undefined1  [16] percent_change(int param_1,int param_2,ulong param_3)
{
  long lVar1;
  undefined1 auVar3 [16];
  if (param_1 != 0) {
    lVar1 = (long)((param_2 - param_1) * 100);
    auVar3._0_8_ = lVar1 / (long)param_1 & 0xffffffff;
    auVar3._8_8_ = lVar1 % (long)param_1 & 0xffffffff;
    return auVar3;
  }
  ...
}
```

反編譯器看到 `cltd` 和 `idivl` 同時寫入 `edx` 和 `eax`，於是用 64 位元除法為 16 位元組的回傳值建模。同一筆紀錄中的清單說得更平實：

```
0x100000: TEST EDI, EDI
0x100002: JZ 0x0010000d
0x100004: SUB ESI, EDI
0x100006: IMUL EAX, ESI, 0x64
0x100009: CDQ
0x10000a: IDIV EDI
0x10000c: RET
0x10000d: XOR EAX, EAX
0x10000f: RET
```

每個操作都是 32 位元寬。`IMUL EAX, ESI, 0x64` 以 32 位元相乘並保留低半部。`CDQ` 把 `eax` 符號擴充進 `edx`，而 `IDIV EDI` 以 32 位元除數除以 64 位元的配對。遵循清單的重建版本，是以 `int` 計算的 `100 * (after - before) / before`。

`grade` 的紀錄形狀相同。Pseudocode 指名一個表格 `*(uint *)(&DAT_001000b4 + (ulong)uVar2 * 4)`，以及一個用 `0x66666667` 建構、難以閱讀的預設表達式。清單把除以 10 顯示為 `IMUL RAX, RAX, 0x66666667`、`SAR RCX, 0x22`、`SHR RAX, 0x3f`、`ADD ECX, EAX`。接著 `ADD ECX, -0x6`、`CMP ECX, 0x4` 與 `JA` 保留從 6 到 10 的十位數。比較之前的 `XOR EAX, EAX` 是預設值 0。`LEA RCX, [0x1000b4]` 與 `MOV EAX, dword ptr [RCX + RAX*0x4]` 讀取五個 4 位元組項目的表格。

步驟 4 讀取了那 20 個位元組：`01000000 02000000 03000000 04000000 04000000`。對十位數 6、7、8、9、10，表格是 `1, 2, 3, 4, 4`。那就是 `ledger.c` 中的 `switch`，其中 `case 10` 與 `case 9` 共用值 4。

rea 自己證據頁面中的兩個證明，都從清單與常數的 `read_bytes` 開始。DX-Ball 的第一份 pseudocode 是沒有 stack 輸入的 `FUN_00406400(void)`，而指令視圖修正了它。先讀清單。引用紀錄的 id，然後再寫 C。

## 撰寫候選版本 (Write the Candidates)

資料夾 `code/fixtures/candidates/` 持有八個重建版本，每個檔案一個函式。四個遵循清單，或編譯成相同的程式碼。四個看似合理但是錯的，或者錯的方式是一個手寫測試就能發現的。

| 檔案 | 原始碼 | 它做什麼 |
|---|---|---|
| `percent_change_mul_first.c` | `100 * (after - before) / before` | 以 C 寫成的清單 |
| `percent_change_split.c` | `(after * 100 - before * 100) / before` | 不同的文字，clang 折疊之後得到相同的位元組 |
| `percent_change_wide.c` | `(int)(((long)after - before) * 100 / before)` | pseudocode 的 `long`，在它 wrap 之前先除 |
| `percent_change_div_first.c` | `(after - before) / before * 100` | 先除，所以捨入不同 |
| `grade_switch.c` | 帶五個 case 的 `switch (score / 10)` | 原始的原始碼 |
| `grade_table.c` | `unsigned slot = score / 10 - 6; slot < 5 ? points[slot] : 0` | 以 C 寫成的清單 |
| `grade_thresholds.c` | `if (score >= 90) return 4; ...` | 最高區間沒有上界 |
| `grade_case10_wrong.c` | `case 10: return 3;` | 一個表格項目是錯的 |

每個檔案都用 fixture 指令編譯：`clang -target x86_64-unknown-linux-gnu -O1 -fno-asynchronous-unwind-tables -c`。編譯後的 object 與原始碼一起 commit，讓測試可以在沒有 clang 的 host 上執行。

## 第一條腿：執行並比較 (Leg One: Execute and Compare)

McKeeman 在 1998 年對差異測試的定義，需要「兩個或更多可比較的系統」被餵入「機械式產生的測試案例」，並把不同的結果、hang 或 crash 視為 bug 候選。他指出了困難之處：「測試中最棘手的問題，是評估測試的結果。」原始函式在這裡就是 oracle。我們不需要知道正確的答案。我們需要的是，原始版本與重建版本在一個涵蓋每種風險的格子上，沒有任何地方不一致。

### 設計格子

`code/main.py` 中的 `input_grid()` 建構 92 個格子。對 `percent_change`，它取十個 `before` 值與七個 `after` 值的每一對：

```
before: 0, 1, -1, 3, -3, 7, 100, -100, 2147483647, -2147483648
after:  0, 5, -5, 105, 536870911, 2147483647, -2147483648
```

對 `grade`，它取 22 個分數：每個區間邊緣的兩側、負數、0、110，以及兩個 `int` 的極限。`classify()` 標記每個格子。溢位有 32 個、負值 17 個、邊界 14 個、單純 12 個，zero guard 7 個。負值截斷 5 個、捨入 2 個、超出 switch 的 2 個，trap 1 個。

每個類別的存在都有一個理由。Zero guard 格子檢查 `TEST EDI, EDI` 分支。負值格子檢查截斷除法：C 與 `idiv` 往零的方向捨入，所以 `-9 / 10` 是 0，而 `-10 / 10` 是 -1。捨入格子檢查乘法發生在除法之前，所以 `(105 - 100) * 100 / 100` 是 5 而不是 0。溢位格子讓 `(after - before) * 100` 超過 32 位元。那一個 trap 格子讓 wrap 後的乘積等於 `INT_MIN`，且 `before` 等於 -1。

Yang、Chen、Eide 與 Regehr 建構了 Csmith，用來產生沒有未定義行為的程式。Oracle 在未定義行為上的不一致只是雜訊，從來不是 bug 回報。有號溢位在 C 中是未定義的。我們的 oracle 是具有固定行為的編譯後 object，所以溢位格子在範圍之內。教訓是要知道哪些格子建立在標準上，哪些建立在編譯後的位元組上。

### Driver 與 trap

`code/driver.c` 從 stdin 每行讀取一個格子，並印出輸入、原始結果與重建結果。`build_harness()` 用 `-Dpercent_change=cand_percent_change -Dgrade=cand_grade` 編譯候選版本，所以兩個版本以不同名稱連結進同一個程式。在 x86-64 Linux host 上，原始版本就是 fixture object 本身。在任何其他 host 上，它是原生編譯的 `ledger.c`，而報告會記錄 `original_kind: fixture-source`。

Driver 用 `sigsetjmp` 和 `siglongjmp` 安裝 `SIGFPE` handler。對 `before = -1` 與 `after = 536870911`，差值是 536870912，而 32 位元的乘積 wrap 成 -2147483648。在 x86-64 上，被除數為此值、除數為 -1 的 `idiv` 會引發 divide error 例外。商放不進 32 位元（Intel SDM Volume 2，IDIV）。Handler 把它變成記錄的值 `trap`。

在 AArch64 上，`sdiv` 沒有 trap，對相同的 operand 回傳 -2147483648。我們在 arm64 Mac 上的執行，記錄的正是如此。Oracle 取決於架構，而報告說明是哪一個執行的。

### 結果

`prove_behaviour(PLAUSIBLE)` 建構 `long` 重建版本與 threshold 重建版本。十個單純案例通過。格子在 28 個格子上失敗：

| 函式 | 輸入 | 類別 | 原始 | 重建 |
|---|---|---|---|---|
| `percent_change` | 3, 2147483647 | 溢位 | -133 | -1431655899 |
| `percent_change` | 100, 2147483647 | 溢位 | -101 | 2147483547 |
| `percent_change` | 2147483647, 5 | 溢位 | 0 | -99 |
| `grade` | 110 | 超出 switch | 0 | 4 |
| `grade` | 2147483647 | 超出 switch | 0 | 4 |

`percent_change` 的全部 26 個失敗，都是 `|before| > 1` 的溢位格子。當 `before` 等於 1 或 -1 時，64 位元的商會截斷回 wrap 後的值，所以那些格子是意外地一致。Threshold 版 `grade` 失敗，是因為原始的 switch 對 11 以上的十位數沒有 case，而 `110 / 10` 是 11。

`prove_behaviour(EQUAL)` 建構 `percent_change_mul_first.c` 和 `grade_switch.c`。全部 92 個格子都一致。`percent_change_div_first.c` 在四個單純案例中有兩個失敗，這是手寫測試真的會發現的那一種案例。

`oracle_disagreements()` 檢查第二件事。`percent_change_model()` 和 `grade_model()` 是清單的 Python 模型，以 `wrap32()` 處理 32 位元的乘法，以 `idiv32()` 處理架構規則。報告在每個格子上，將模型與觀察到的原始版本比較。零個不一致，代表模型與 binary 在每個格子上都一致。模型是 derived 證據，而執行是 observed 證據。報告把它們放在不同的欄位。

## 第二條腿：重新編譯並比較 (Leg Two: Recompile and Compare)

格子上的行為是一條腿。第二條腿問的是，用相同的編譯器與 flag 編譯的重建版本，是否產生相同的位元組。DX-Ball 專案為它的 helper 回報了「63 個相符的位元組」。TH04 專案回報了 2,139 位元組的相符。每個都使用它那個年代的編譯器。rea 自己做不到這點。它的 `compare_functions` 以行為單位對 provider 文字做 diff，並附上限制「Assembly 是不透明的 provider 文字；relocation 正規化不可用。」

### 讀取 object

`code/main.py` 中的 `ElfObject` 用 `struct.unpack_from` 讀取一個 ELF64 可重定位 object。它檢查 magic、64 位元 class、little-endian 資料、`e_machine` 62 和 `e_type` 1。接著它走訪 section header、string table、`.symtab` 與每個 `.rela` section。在 fixture 中，`percent_change` 從 `.text` offset 0 開始，大小 16，`grade` 在 `0x10`，大小 46。`total` 從 `0x40` 開始，大小 31，`grade_label` 在 `0x60`，大小 63。

`percent_change` 的 16 個位元組是 `85 ff 74 09 29 fe 6b c6 64 99 f7 ff c3 31 c0 c3`。這個函式沒有 relocation。`grade` 有一個：在函式內的 offset `0x26`，型別 `R_X86_64_PC32`，symbol `.rodata`，addend `0x10`。它屬於位於 offset `0x23` 的 7 位元組 `leaq`，其最後四個位元組是 displacement 欄位。

### 遮蔽，然後比較紀錄

在 RELA object 中，addend 存在於 relocation 紀錄裡，而四個欄位位元組是零。候選的 `grade_switch.o` 在那裡也是零。單純的位元組比較會通過，而兩個 relocation 卻指名不同的地方。`mask_relocations()` 還是會把兩邊的欄位歸零，所以相同的方法對 REL object 以及欄位帶有 addend 的 Mach-O 也成立。

接著 `prove_bytes()` 依 offset、型別與目標 section 比較 relocation 紀錄。原始版本寫著 `.rodata+0x10`，而候選版本寫著 `.rodata-0x4`。Addend 不同，是因為候選 object 在它的 switch 表格前面沒有 `labels.rel` 表格。不同的 addend 是不同的 offset，而不是不同的函式。

### 讀取 relocation 指向的常數

對於結束於其指令結尾的 PC 相對欄位，CPU 會加上下一條指令的位址，所以目標是 symbol 加 addend 再加 4。在原始版本中，那是 `.rodata+0x14`，而在候選版本中是 `.rodata+0x0`。`read_table()` 在那裡讀取五個 4 位元組項目。兩者都持有 `1, 2, 3, 4, 4`。

五這個數量來自清單：`CMP ECX, 0x4` 搭配 `JA` 允許 index 0 到 4，而 `[RCX + RAX*0x4]` 每個 index 讀取 4 個位元組。`code/main.py` 中的 `CLAIMS` 為 `grade` 記錄了那個形狀，並為 `percent_change` 記錄零個 relocation。

這個步驟的存在，是因為 TH08 專案記錄過的 bug。它較舊的比較「把目標位址重播進指令的 relocation 欄位，然後在沒有檢查該位址的浮點字面值資料的情況下比較指令位元組。因此原始碼的 `0.0` 可能對目標的 `128.0` 顯得完全相符。」修正稽核了 1,548 個浮點字面值參照，並又更正了十二個。`grade_case10_wrong.c` 是我們版本的那個 bug。它的 46 個程式碼位元組與原始版本相等，它的 relocation 紀錄形狀相同，而它的表格是 `1, 2, 3, 4, 3`。只有常數檢查會讓它失敗。

### 結果

| 候選 | 大小 | 遮蔽後的位元組 | Relocation 紀錄 | 表格 | 狀態 |
|---|---|---|---|---|---|
| `percent_change_mul_first` | 16 對 16 | 相等 | 無 | 無 | pass |
| `percent_change_split` | 16 對 16 | 相等 | 無 | 無 | pass |
| `percent_change_wide` | 16 對 26 | 大小不同 | 無 | 無 | fail |
| `percent_change_div_first` | 16 對 18 | 大小不同 | 無 | 無 | fail |
| `grade_switch` | 46 對 46 | 相等 | 形狀相同 | 兩者皆 `1,2,3,4,4` | pass |
| `grade_table` | 46 對 46 | 相等 | 形狀相同 | 兩者皆 `1,2,3,4,4` | pass |
| `grade_thresholds` | 46 對 39 | 大小不同 | 1 對 0 | 無 | fail |
| `grade_case10_wrong` | 46 對 46 | 相等 | 形狀相同 | `1,2,3,4,4` 對 `1,2,3,4,3` | fail |

有兩個結果值得再看一眼。`percent_change_split.c` 有不同的原始碼文字但位元組相同，因為 clang 把有號溢位視為未定義，並把 `after * 100 - before * 100` 折疊成單一個乘法。`grade_table.c` 也逐位元組相符，所以編譯器從陣列 index 與從 `switch` 到達了相同的表格。位元組相符證明的是，這個編譯器搭配這些 flag，從候選版本產生了相同的程式碼。它不證明候選版本就是原始的原始碼文字。

大小是診斷，不是接受標準。rea 的證據頁面記錄了一個長度正確、卻有 467 個位元組不同的 1,825 位元組候選。`wide` 重建版本是 26 位元組，因為 `movslq`、`imulq`、`cqto` 和 `idivq` 取代了 32 位元的形式。清單早在任何編譯之前，就告訴了我們寬度。

## 經由 rea 的 agent 迴圈 (The Agent Loop Over rea)

rea 的 MCP tool 中有六十五個沒有 CLI 指令，而 `verify_reconstruction` 是其中之一。`code/main.py` 中的 `McpStdioClient` 是能到達它們的最小 client。它啟動 `npx -y rea-agents@6.5.0 mcp`，每行對 stdin 寫入一個 JSON-RPC 請求，並在一個 thread 上從 stdout 每行讀取一個 JSON 物件。它先送出 protocol version `2025-11-25` 的 `initialize`，然後是 `notifications/initialized`，再是 `tools/call`。每次呼叫預設的期限是 330 秒，與 rea 對 Ghidra cold start 允許的數字相同。

`agent_loop()` 發出六次呼叫並記錄每一次：

```json
{
  "step": 2,
  "tool": "analyze_function",
  "arguments": {"procedure": "percent_change"},
  "seconds": 6.31,
  "is_error": false,
  "evidence_id": "ev_e88b10dc0c6400e51bb6f7fdc3eaea8d2c97cc02463f0db3e6afeefc273197cd",
  "confidence": "observed",
  "authority": "shipped-artifact",
  "limitations": ["Resolved reference metadata identifies computed, indirect, external, call, jump, and data edges; unresolved targetless flows remain unknown, and function classifications distinguish thunks and externals.", "..."],
  "result_sha256": "..."
}
```

完整的 Evidence 紀錄以它們的 id 存放在同一個檔案中。重建版本所依據的 pseudocode 與清單，是 fixture 的一部分。`table_address()` 在 `grade` 的清單中找到 `LEA` operand，並把它餵給 `read_bytes`。`open_binary`、`get_evidence_bundle` 和 `close_binary` 不回傳 Evidence 紀錄，並以 null id 記錄。

當 `npx` 在 path 上，而且 `GHIDRA_INSTALL_DIR` 指名某個目錄時，`run_agent(live=True)` 會驅動 rea。否則它重播 fixture 並在 stderr 說明。測試又更進一步。`code/tests/fake_rea_server.py` 是替身 server，它用針對相同 tool 與引數的已記錄紀錄，回應每次 `tools/call`。Client 與迴圈在每次測試執行時，都針對它執行，不需要安裝 rea、Ghidra 或 Java。

### rea 的 verifier 能說與不能說什麼

rea 的 `verify_reconstruction` 接受指向比較 Evidence 的主張，並對每項主張回傳 pass、fail 或 unknown。它固定的限制寫著「Pass 表示每個宣告的主張都通過；它並不建立全域的實作等價性。」關於函式的主張，比較的是 `compare_functions` 的 provider 文字。行為主張則比較 `compare_process_captures` 的兩個已記錄的 process 擷取。當來源紀錄的 call 涵蓋範圍不完整時，關於 call 的主張是 `unknown`。rea 內部沒有任何東西會編譯重建後的函式，或在你的輸入格上執行它。這堂課的兩條腿，就是 rea 留給你的部分。

rea 的 obligation ledger 預期一筆證明紀錄，帶有 `predicate_type: "rea.reconstruction-proof"`、`operation: "verify_reconstruction_obligations"`、`confidence: "observed"` 與 `normalized_result.passed === true`。沒有任何公開的 rea tool 會發出一筆。`code/main.py` 中的 `build_proof()` 產生它：

```json
{
  "predicate_type": "rea.reconstruction-proof",
  "operation": "verify_reconstruction_obligations",
  "confidence": "observed",
  "authority": "controlled-replay",
  "normalized_result": {
    "passed": true,
    "status": "pass",
    "claims": [
      {"claim_id": "percent_change.behaviour", "kind": "behavioral", "status": "pass", "evidence_ids": ["ev_e88b10dc..."]},
      {"claim_id": "grade.behaviour", "kind": "behavioral", "status": "pass", "evidence_ids": ["ev_67b6d3f9..."]},
      {"claim_id": "percent_change.bytes", "kind": "structural-function", "status": "pass", "evidence_ids": ["ev_e88b10dc..."]},
      {"claim_id": "grade.bytes", "kind": "structural-function", "status": "pass", "evidence_ids": ["ev_67b6d3f9..."]},
      {"claim_id": "grade.referenced-constants", "kind": "structural-function", "status": "pass", "evidence_ids": ["ev_67b6d3f9..."]}
    ]
  },
  "evidence_links": ["ev_e88b10dc...", "ev_67b6d3f9..."],
  "evidence_id": "ev_..."
}
```

Id 是 `ev_` 加上不含 id 的紀錄，以 key 排序、緊湊的 JSON 的 SHA-256。`aggregate()` 是 fail-closed：`fail` 勝出，然後是 `unknown`，而空的主張清單是 `unknown`。在沒有 C 編譯器的 host 上，兩項行為主張是 `unknown`，所以即使兩項位元組主張都通過，紀錄仍然是 `unknown`。`authority` 是 `controlled-replay`，因為證明來自我們自己的 harness，而不只是 object file。

## 研究怎麼說 (What the Research Says)

Liu、Raff 與 Micinski 在 2026 年量測了九種設定下的八個反編譯系統。通過 benchmark 包含的每項測試的候選，在 fuzz 輸入上仍然有分歧。整體比率為 4.9%，某個系統高達 13%。在 300 個 GitHub 函式庫函式與 287 個以 CVE 為基礎的函式上，最強的改善模型把 Ghidra 的 build 成功率從 75% 提高到 90%。在相同的函式上，行為符合率從 74% 降到 62%。他們追溯到的原因，是被捏造出來的欄位、型別、callee 與 guard，取代了傳統反編譯器留下的看得見的未知。

我們的 `percent_change_wide.c` 就是那些被捏造的型別之一。Pseudocode 提供了 `long`，重建版本保留了它，結果能編譯、通過十個測試，並在 26 個溢位格子上失敗。Grade-threshold 的重建是被捏造的 guard。在 LLM4Decompile 的指標下，兩者都會算是 re-executable，該指標重新編譯輸出並執行 benchmark 的斷言。Ghidra 自己的 pseudo-C 在 HumanEval-Decompile 上通過了其中 20.12% 的斷言。

DecompileBench 量測了 130 個真實程式中的 23,400 個函式。它發現 LLM 方法在功能正確性上比工業級反編譯器低 52.2%，而可理解性分數較高。

這個領域使用的三種證明，正是你能建構的三條腿。Phoenix 與 DREAM 重新編譯輸出，並執行原始程式的測試。McKeeman 與 Csmith 把產生的輸入對原始版本執行，且輸入保持在已定義行為之內。Ramos 與 Engler 用有界的 symbolic execution 檢查兩個 C 函式，這證明在探索過的所有路徑上、直到某個輸入大小為止的相等。這堂課建構前兩個。第三個是課程 35 的證據 ledger 中的實作練習，用於格子小到無法信任的時候。

## 動手做 (Build It)

執行標準函式庫的 harness：

```bash
cd phases/13-tools-and-protocols/36-rebuild-and-prove-equal
python3 code/main.py
```

在 path 上有 clang 時，demo 會編譯兩對候選，並對每一對執行 92 個格子。它重新編譯八個候選、重播 agent 的執行，並產生證明紀錄。沒有 clang 時，它讀取已 commit 的 object，並把行為主張回報為 `unknown`。

在 arm64 Mac 上預期的形狀：

```json
{
  "host": "aarch64 Darwin",
  "originalKind": "fixture-source",
  "gridCells": 92,
  "plausible": {"naivePassed": true, "gridMismatches": 28, "oracleDisagreements": 0},
  "equal": {"naivePassed": true, "gridMismatches": 0, "oracleDisagreements": 0},
  "legTwo": {
    "percent_change_mul_first": {"status": "pass", "reasons": []},
    "percent_change_wide": {"status": "fail", "reasons": ["size 16 vs 26 bytes"]},
    "grade_case10_wrong": {"status": "fail", "reasons": ["referenced constants differ: [1, 2, 3, 4, 4] vs [1, 2, 3, 4, 3]"]}
  },
  "agentRun": {"mode": "replayed fixture", "server": {"name": "rea", "version": "6.5.0", "protocolVersion": "2025-11-25"}},
  "proof": {"status": "pass", "evidence_id": "ev_..."}
}
```

依此順序閱讀 `code/main.py`：

1. `wrap32()`、`c_div()`、`idiv32()` 以及兩個 `*_model()` 函式，持有清單的整數語意。
2. `input_grid()` 與 `classify()` 建構並標記 92 個格子。
3. `build_harness()` 與 `execute()` 從原始版本、重新命名的候選與 `driver.c` 編譯出一個程式，然後執行這些格子。
4. `prove_behaviour()` 回傳第一條腿的報告，包含不一致與模型的分歧。
5. `ElfObject` 讀取 header、symbol 與 `.rela` 紀錄。`mask_relocations()` 與 `read_table()` 為比較做準備。
6. `prove_bytes()` 回傳第二條腿的報告，包含大小、遮蔽範圍、relocation 形狀與表格。
7. `McpStdioClient` 與 `agent_loop()` 驅動 rea 並記錄每一步。
8. `build_proof()` 與 `aggregate()` 產生 fail-closed 的紀錄。

## 實際使用 (Use It)

要證明新的函式相等，請重複這些步驟：

1. 取得該函式的 `analyze_function` 並記錄 evidence id。在 pseudocode 之前先讀清單。
2. 根據清單的 operand 寬度撰寫候選版本。保留原始版本的函式名稱。
3. 列出清單中的每一個乘、加、減與除。加入在各自寬度上會 wrap 的格子輸入。
4. 加入清單中每個比較常數的兩側。對 `grade` 而言，那是減法之後的 6 和 4。
5. 加入在目標上可能 trap 的每個除數，並記錄是哪個架構執行了格子。
6. 執行兩條腿，並在看狀態之前先閱讀原因。
7. 把證明紀錄與候選版本放在一起。它的 `evidence_links` 告訴讀者主張所依據的是哪些紀錄。

任何編譯器更新之後，都要重新執行兩條腿。位元組相符，是關於一個編譯器搭配一組 flag 的陳述。

## 互動實驗 (Interactive Lab)

### 實驗 A：模型與執行

從 `code` 目錄開啟 Python：

```bash
cd phases/13-tools-and-protocols/36-rebuild-and-prove-equal/code
python3 -q
```

執行：

```python
from main import *
percent_change_model(3, INT_MAX)
percent_change_model(-1, 536870911, "x86_64")
percent_change_model(-1, 536870911, "aarch64")
report = prove_behaviour(PLAUSIBLE)
report.naive_passed, len(report.mismatches), report.mismatches[0]
```

第一次呼叫得到 -133，也就是 wrap 後的乘積除以 3。第二次得到 `'trap'`，第三次得到 -2147483648。報告顯示十個單純案例通過，28 個格子失敗。把 `report.oracle_disagreements` 與空清單比較。

### 實驗 B：讀取 object

```python
obj = ElfObject(ORIGINAL_OBJECT)
grade = obj.function("grade")
grade.size, grade.data[0x26:0x2a].hex(), grade.relocations
read_table(obj, grade.relocations[0], 5, 4)
```

欄位位元組是 `00000000`。Relocation 指名 `.rodata`，addend 為 `0x10`，而表格讀作 `[1, 2, 3, 4, 4]`。

### 實驗 C：破壞一個常數

```python
report = prove_bytes("grade", CANDIDATES / "grade_case10_wrong.o")
report.masked_equal, report.relocations_match, report.tables_equal, report.reasons()
```

位元組相符、紀錄相符，而表格不符。現在編輯 `grade_case10_wrong.c`，讓 `case 10` 再次回傳 4，用 fixture 指令重新編譯，並看狀態變成 `pass`。

### 實驗 D：驅動 rea

安裝了 rea 6.5.0 與 Ghidra 12.1.4 之後：

```bash
GHIDRA_INSTALL_DIR=/opt/homebrew/Cellar/ghidra/12.1.4/libexec \
JAVA_HOME=/opt/homebrew/opt/openjdk@21 \
python3 code/main.py --live
```

第一次 `analyze_function` 在 Ghidra 匯入 object 的期間要花幾秒。把新的 evidence id 與已 commit 的執行比較。同樣的位元組在相同的分析 profile 下得到相同的 id，因為 id 是排除檔案路徑的內容 hash。

## 實作練習 (Practice Lab)

把兩條腿擴充到同一個 object 的 `total` 和 `grade_label`。

需求：

- 對 `total`，輸入是陣列與 count。建構 count 為 0、負的 count、一個元素，以及在 32 位元 wrap 的總和的格子。
- 對 `grade_label`，函式 inline 呼叫 `grade` 並讀取 `labels.rel`，這是五個相對字串 offset 的表格，其 relocation 在 `.rela.rodata` 中。
- 為 `grade_label` 在 `CLAIMS` 中新增一個項目，包含兩個 relocation 與五個 4 位元組項目的表格，並比較項目所指向的字串。
- 記錄模型與執行不一致的格子（如果有的話），並從清單解釋每一個。
- 為每個函式每條腿，在證明紀錄中各新增一項主張，並保持彙總為 fail-closed。

成功的意思是，證明紀錄列出四個函式，每項主張都帶有 evidence id，而且當一個 label 字串不同時 `grade_label` 會失敗。

## 評分實驗 (Graded Lab)

關於這堂課 fixture 的五個問題。用你建構的程式碼找出每個答案，然後寫進 `code/lab_answers.json`。數字寫成數字或像 `"0x40"` 這樣的十六進位字串，清單寫成 JSON 清單。

1. `t1`：x86_64 上的 percent_change_model(3, 105)。
2. `t2`：x86_64 上的 percent_change_model(0, 5)。
3. `t3`：grade_model(100)。
4. `t4`：input_grid() 中的格子數量。
5. `t5`：x86_64 模型會 trap 的格子數量。

檢查你的答案：

```bash
cd phases/13-tools-and-protocols/36-rebuild-and-prove-equal
python3 code/lab.py
```

檢查器會依據 `code/main.py` 在已 commit 的 fixture 上計算每個預期值。它為每個任務印出 `PASS`、`FAIL` 或 `TODO`，以及一個分數，只有當五個全部通過時才以 0 結束。`python3 code/lab.py my-answers.json` 檢查另一個檔案。`code/tests/test_lab.py` 證明檢查器接受正確的答案，並拒絕錯誤的答案。

### 看它執行

[Inside a Decompiler](https://aiengineeringfromscratch.com/inside-a-decompiler) 這台機器以 pseudocode 作結，並把它歸為 inferred：頁面並不證明它。這堂課就是把那項主張變成經過檢查的主張的步驟。

## 輸出 Skill (Output Skill)

這堂課會產出 `outputs/skill-rebuild-and-prove-equal.md`。用它把函式證據與候選重建版本，變成兩條腿的證明與證明紀錄。這個 skill 要求在 pseudocode 之前先讀清單、涵蓋每種風險類別的格子、relocation 遮蔽加上常數檢查，以及 fail-closed 的彙總。

## 驗證 (Verify It)

執行 demo 與測試套件：

```bash
cd phases/13-tools-and-protocols/36-rebuild-and-prove-equal
python3 code/main.py
python3 -m unittest discover -s code/tests -v
```

35 個測試證明：

- `c_div()` 往零截斷，而 `wrap32()` 遵循 two's complement
- 模型對 `(3, INT_MAX)` 給出 -133，對 x86-64 上的 trap 格子給出 `trap`，而在 AArch64 上給出 -2147483648
- 格子有 92 個格子且涵蓋每種風險類別，而單純案例避開了每個困難類別
- ELF reader 在記錄的 offset 與大小找到四個函式
- `percent_change` 是沒有 relocation 的 16 位元組，而 `grade` 在 `0x26` 有一個 `R_X86_64_PC32`
- relocation 欄位的位元組是零，而遮蔽在這個 RELA object 中沒有改變任何東西
- 八個已 commit 的候選都達到預期的狀態，並附上已記錄的原因
- `grade_case10_wrong` 通過遮蔽後的位元組，只在常數檢查失敗
- 已記錄的第一條腿執行，顯示十個單純案例通過、28 個不一致，以及零個模型分歧
- 在有編譯器的情況下，兩對都重建並以相同的結果重新執行
- agent 的執行依序以 SHA-256 鎖定已 commit 的 object，並在每個 Evidence 步驟都有 `ev_` id
- `read_bytes` 的紀錄等於 ELF reader 找到的表格
- client 驅動替身 server，並記錄與真實執行相同的 evidence id
- server 的錯誤結果會進入執行紀錄
- `aggregate()` 是 fail-closed，而證明 id 能從它的內容重新計算

## Production 的失敗模式 (Production Failure Modes)

| 失敗 | 薄弱的報告說什麼 | Harness 必須證明什麼 |
|---|---|---|
| 重建版本保留反編譯器的 `long` | 「十個測試通過」 | 26 個溢位格子與原始版本不同 |
| 用 threshold 取代 switch | 「每個 grade 區間都通過」 | 110 以上在原始版本中回傳 0 |
| 先除後乘 | 「公式在代數上相等」 | 捨入格子不同，單純案例失敗 |
| 跳過 trap 格子 | 「沒有 crash」 | x86-64 引發 `#DE`，AArch64 回傳 `INT_MIN`，報告指名 host |
| 沒有遮蔽的位元組比較 | 「位元組相等」 | relocation 紀錄指名不同的 offset |
| 只遮蔽而不讀取常數 | 「遮蔽後位元組相等」 | `1,2,3,4,3` 不同於 `1,2,3,4,4` |
| 把大小當作接受標準 | 「長度相同」 | 在相同長度內有不同的位元組 |
| 不同的編譯器或 flag | 「位元組不同，重建是錯的」 | 相同的原始碼給出不同的位元組，所以比較無效 |
| 把模型當作觀察 | 「模型一致」 | 執行是 observed，模型是 derived，不一致處會被列出 |
| 沒有 evidence id 的主張 | 「verified」 | 每項主張都連結它所涵蓋函式的 `ev_` 紀錄 |
| CI host 上沒有編譯器 | 「通過」 | 行為主張是 `unknown`，彙總是 `unknown` |
| 把 pseudocode 當作原始碼 | 「反編譯器說是 long」 | 清單顯示 `IMUL EAX`、`CDQ`、`IDIV EDI` |

## 營運規則 (Operational Rule)

根據清單撰寫重建版本，並指名它來自哪一筆紀錄。在一個會讓每個中間值 wrap、並跨越每個比較常數的格子上，對照原始版本執行它。用原始的 flag 重新編譯、遮蔽 relocation 欄位、比較紀錄，並讀取它們指向的常數。為每項主張記錄其 evidence id，並讓未驗證的主張維持 unknown。

## 延伸閱讀 (Further Reading)

- [McKeeman, Differential Testing for Software, Digital Technical Journal 10(1), 1998](https://www.cs.tufts.edu/comp/150FP/archive/bill-mckeeman/DifferentailTesting.pdf)
- [Yang, Chen, Eide, Regehr, Finding and Understanding Bugs in C Compilers, PLDI 2011](https://doi.org/10.1145/1993498.1993532)
- [Ramos, Engler, Practical, Low-Effort Equivalence Verification of Real Code, CAV 2011](https://doi.org/10.1007/978-3-642-22110-1_55)
- [Schwartz, Lee, Woo, Brumley, Native x86 Decompilation Using Semantics-Preserving Structural Analysis, USENIX Security 2013](https://www.usenix.org/conference/usenixsecurity13/technical-sessions/presentation/schwartz)
- [Liu, Raff, Micinski, When LLM Decompilers Recompile More and Preserve Less, arXiv 2609.05370, 2026](https://arxiv.org/abs/2609.05370)
- [Gao et al., DecompileBench, Findings of ACL 2025](https://aclanthology.org/2025.findings-acl.1194/)
- [Tan et al., LLM4Decompile, arXiv 2403.05286, 2024](https://arxiv.org/abs/2403.05286)
- [System V gABI, Chapter 4, Relocation](https://www.sco.com/developers/gabi/latest/ch4.reloc.html)
- [x86-64 psABI repository, relocation types](https://gitlab.com/x86-psABIs/x86-64-ABI)
- [Intel 64 and IA-32 Architectures Software Developer's Manual, Volume 2, IDIV](https://www.intel.com/content/www/us/en/developer/articles/technical/intel-sdm.html)
- [Arm Architecture Reference Manual for A-profile, DDI 0487, SDIV](https://developer.arm.com/documentation/ddi0487/latest)
- [rea 6.5.0 MCP contracts](https://github.com/morluto/rea/blob/rea-agents-6.5.0/docs/mcp-contracts.md)
- [rea 6.5.0 reconstruction obligation ledgers](https://github.com/morluto/rea/blob/rea-agents-6.5.0/docs/reconstruction-obligation-ledgers.md)
