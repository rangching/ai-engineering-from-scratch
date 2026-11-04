# 二進位分析中的證據、信心與未知 (Evidence, Confidence, and Unknowns in Binary Analysis)

> 關於編譯後程式的發現，是一筆帶有 hash、信心、權威，以及它無法顯示之事項清單的紀錄。

**Type:** Build
**Languages:** Python, C
**Prerequisites:** Phase 13 · 32 (ELF from scratch), Phase 13 · 33 (disassembly and control flow), Phase 13 · 34 (lifting to pseudocode)
**Time:** ~105 分鐘

## 學習目標 (Learning Objectives)

- 建構以 rea 的 `Evidence` 為藍本的證據紀錄，具備 observed、derived 與 inferred 三種信心。
- 計算內容 hash 的 id，使檔案搬到另一個資料夾時 id 保持不變。
- 把每個 provider 無法顯示的事項，記錄為隨紀錄一起走的限制。
- 在函式的 observed 執行支持某個結果之前，拒絕陳述該結果。
- 把剩餘的未知保存為以 hash 鏈連結的有修訂版本的紀錄。
- 以確切的 key 快取 provider 的答案，並拒絕過期的重播。

## 問題 (The Problem)

你問 agent，`ledger-x86_64-O1.o` 中的 `grade` 對分數 55 回傳什麼。Agent 呼叫 rea、讀取 Ghidra 的 pseudocode，並回答 0。審查者問 agent 是怎麼知道的。Agent 手上有一段 Ghidra 印出的文字。對於預設路徑，那段文字寫著 `uVar1 = -(param_1 >> 0x1f) ^ (uint)((ulong)((long)param_1 * 0x66666667) >> 0x3f);`。而該處的位元組寫的是 `xor eax, eax`。

這個表達式與那條指令，對每個輸入都產生 0。Agent 並不知道這點。它信任這段文字，因為是 tool 產生的。2026 年有一項研究，在 24 個逆向工程任務上執行了三個 agent（[NDSS BAR 2026](https://www.ndss-symposium.org/wp-content/uploads/bar2026-58.pdf)）。它指出四種反覆出現的弱點：訓練偏差、對觀察結果過度信任、context 限制，以及計畫固著。另一項 2026 年的研究在 binary 中植入文字，發現模型在 40 個案例中的 35 個提出了被植入的不安全動作（[arXiv 2607.12507](https://arxiv.org/abs/2607.12507)）。

rea 6.5.0 為 agent 提供了處理這件事的詞彙。每個成功的 tool call 都產生一筆 `Evidence` 紀錄，帶有 `observed`、`derived` 或 `inferred` 的 `confidence`、一個 `authority`，以及一份 `limitations` 清單。封裝好的 skill 告訴 agent：「每個結論都必須區分觀察、推論與未知。引用 Evidence ID，保留限制與不完整的涵蓋範圍，並且絕不暗示靜態分析觀察到了執行。」

rea 把三件事留給你。它的 snapshot 快取的是確切的查詢結果，而不是分析。它的驗證在某個比較維度不變時通過一項主張，而它自己的註解說通過「並不建立全域的實作等價性」。它的 obligation ledger 只從 JavaScript 和 managed 圖取得候選，所以 native 函式永遠不會透過它到達 `verified`。

這堂課用單純的 Python 建構缺少的那一塊。你寫一個 ledger，把關於 fixture 的每個事實，都儲存為帶有內容 hash 的紀錄。你對 `percent_change` 和 `grade` 做出預測，把它們標記為 inferred，並對照你自己編譯的測試程式的執行結果來檢查。你匯入兩筆真實的 rea 紀錄，以及它們來源的 snapshot，而你的 ledger 重新計算 rea 寫下的每個 hash。

```figure
re-evidence-ledger
```

## 三個信心用詞，五個權威用詞 (Three Words for Confidence, Five for Authority)

`confidence` 是有三個值的 enum，從來不是數字。`authority` 有五個值。當 producer 沒有指定時，rea 使用 `shipped-artifact` 搭配 `observed`。

| 值 | 在 fixture 上的意義 | 這堂課中的紀錄 |
|---|---|---|
| `observed` | 從位元組或執行中讀取的事實 | ELF reader 對 section 和 symbol 的讀取、兩筆 Ghidra 紀錄、17 次 host 執行 |
| `derived` | 從 observed 紀錄計算出來的事實 | switch 表格模型、除法模型、每一次未知的變更 |
| `inferred` | 還沒有任何觀察陳述過的預期 | 18 個預測 |
| `shipped-artifact` | 來源是磁碟上的檔案 | 每一次對 `ledger-x86_64-O1.o` 的讀取 |
| `controlled-replay` | 你控制的執行產生了該事實 | `harness.c` 的 host 執行 |
| `analyst-inference` | 你自己的判斷 | 預測與未知的變更 |
| `historical-reference` | 較舊的原始碼樹或文件 | 這裡沒有 |
| `external-service` | 檔案與 host 之外的服務 | 這裡沒有 |

最耗時間的兩種混淆。第一種是把 pseudocode 當作行為的觀察。它是對 Ghidra 印出什麼的觀察。第二種是把空清單當作不存在的證明。rea 的設計筆記很嚴格：「缺少的涵蓋範圍是未知，而不是不存在。」`code/main.py` 中的 ledger 編碼了這兩條規則。

## 紀錄 (The Record)

紀錄恰好有十六個欄位，而 rea 6.5.0 會拒絕任何其他欄位。`code/main.py` 保留與 rea 相同的名稱，所以任何一方的紀錄都能在另一方通過驗證。第十六個欄位 `relationship_coverage`，說明 caller 或 callee 清單有多完整。它的值是 `complete`、`partial` 或 `unknown`，當操作沒有回報 call 關係時則是 `null`。

| 欄位 | 它持有什麼 |
| --- | --- |
| `evidence_id` | `ev_` 加 64 個十六進位數字，是內容的 hash |
| `subject` | `name`、`digest.sha256`、`format`、`architecture`、`local_path`，或 `null` |
| `provider` | `id`、`name`、`version`，或 `null` |
| `analysis_profile` | provider 的設定及其 digest，或 `null`，永遠存在 |
| `predicate_type` | 標籤，預設為 `rea.analysis` |
| `operation`、`parameters` | tool 及其輸入 |
| `raw_result`、`normalized_result` | provider 輸出與正規化的結果 |
| `confidence`、`authority` | 上面的 enum |
| `environment` | `id`、`platform`、`architecture`、`isolation`，或 `null` |
| `limitations` | 單純的句子 |
| `relationship_coverage` | `status` 為 `complete`、`partial` 或 `unknown` 並附診斷，或 `null` |
| `locations` | `address`、`address-range`、`artifact-path`、`file-offset`、`file-offset-range` |
| `evidence_links` | 這筆紀錄所依據的 `ev_` id |

這是 rea 為 `grade` 寫下的真實紀錄，裁剪到只剩這裡重要的欄位。完整的檔案是 `code/fixtures/rea-evidence-grade.json`，37,573 位元組。

```json
{
  "evidence_id": "ev_67b6d3f992eaf49a2e16de084b6ebfdd9884e8c5c7f75e175b7bb20f4f748ce1",
  "subject": {
    "name": "ledger-x86_64-O1.o",
    "digest": { "sha256": "9a4a8082ba5e285a732552981a6935e0c5255725664f380e37c44ebfb04d803f" },
    "format": "elf",
    "architecture": "x86_64",
    "local_path": "/private/tmp/aiefs-re/phases/13-tools-and-protocols/35-evidence-confidence-and-unknowns/code/fixtures/ledger-x86_64-O1.o"
  },
  "provider": { "id": "ghidra", "name": "Ghidra", "version": "12.1.4" },
  "analysis_profile": {
    "provider": { "id": "ghidra", "name": "Ghidra", "version": "12.1.4" },
    "parameters": { "jump_table_evidence": "typed-case-default-blocks-v1", "import_mode": "ephemeral-source-immutable" },
    "digest": "9c7f6b24be69294e61da046e4583e3856cb9e40433b149c1bf8bb534d2fe03ef"
  },
  "operation": "analyze_function",
  "parameters": { "procedure": "grade" },
  "confidence": "observed",
  "authority": "shipped-artifact",
  "environment": null,
  "limitations": [
    "Default-space addresses use lowercase 0x-prefixed hexadecimal; other address spaces use <percent-encoded-space>:0x<hex>.",
    "Resolved reference metadata identifies computed, indirect, external, call, jump, and data edges; unresolved targetless flows remain unknown, and function classifications distinguish thunks and externals."
  ],
  "relationship_coverage": {
    "status": "complete",
    "diagnostics": [
      "Only resolved Ghidra call references are returned; unresolved computed or indirect calls remain unknown, while function classifications distinguish thunks and externals."
    ]
  },
  "locations": [],
  "evidence_links": []
}
```

Profile 真實的 `parameters` 物件有 25 個 key。真實的 `limitations` 清單有十個句子。請保留它們。限制是內容的一部分，所以丟掉一個會改變 id，而你的 ledger 接著會拒絕這筆紀錄。

注意這筆紀錄沒有說什麼。`environment` 是 `null`，因為 Ghidra 讀取檔案而沒有執行任何東西。`locations` 是空的，因為這個操作把位址回報在 `normalized_result` 裡面。`relationship_coverage` 的 status 是 `complete`。它的診斷說明只回傳已解析的 Ghidra call reference，而未解析的計算式或 indirect call 仍然是未知。所以 caller 與 callee 清單只對已解析的 call 是完整的。

## Id 是內容的 hash，而路徑不在其中 (The Id Is a Hash of Content, and the Path Is Not in It)

rea 把 `evidence_id` 計算為 `ev_` 加上對紀錄投影的 RFC 8785 正規化 JSON 的 SHA-256。投影保留 subject 的 `digest`、`format` 和 `architecture`，並丟掉 `name` 和 `local_path`。當 `relationship_coverage` 的值為 `null` 時，它也會丟掉該欄位，所以沒有 call 關係的紀錄，其 id 與該欄位存在之前相同。其他一切，包括 `raw_result` 和每個限制，都會被 hash。

RFC 8785 對物件 key 排序、移除空白，並固定數字與字串的形式。對於數字都是整數的紀錄，Python 標準函式庫會產生相同的位元組：

```python
def canonical_json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
```

`code/main.py` 在 hash 之前拒絕浮點數，因為 RFC 8785 和 Python 印出某些浮點數的方式不同。兩筆 rea 紀錄與 snapshot 都不含浮點數，所以這個做法能重現每個 rea hash：

| rea 的值 | 由 `code/main.py` 重新計算 |
|---|---|
| `grade` 的 `ev_67b6d3f992eaf49a…` | 相等 |
| `percent_change` 的 `ev_e88b10dc0c6400e5…` | 相等 |
| 對 `{provider, parameters}` 的 profile digest `9c7f6b24be69294e…` | 相等 |
| `grade` snapshot 項目的 `query_26122aebe9d8abae…` | 相等 |
| `percent_change` 項目的 `query_0f95ef38038d3666…` | 相等 |

為什麼要排除路徑？對相同位元組的相同觀察，在任何機器、任何資料夾中，都得到相同的 id。上面清單中的 `local_path` 指名的是某台 Mac 上的資料夾。當你把 fixture 複製到 `/home/you/re` 時，id 不會改變。`rebind_subject_path()` 讓你以新的路徑顯示紀錄。當 `sha256`、`format` 或 `architecture` 不同時，它會拒絕。rea 6.5.0 在為位於新路徑的目標重播 snapshot 紀錄時，用 `rebindEvidenceToTargetPath` 做同樣的事。

`validate_evidence()` 重新計算 id，並用 rea 的句子「Evidence semantic identifier does not match its record」拒絕不一致。它也會拒絕 `analysis_profile.provider` 與其 `provider` 不同的紀錄。測試 `test_tampered_id_and_mismatched_profile_are_rejected` 會編輯真實 `grade` 紀錄內的一行 pseudocode，並看著 ledger 拒絕它。

## 自己觀察 object (Observe the Object Yourself)

在任何 provider 之前，ledger 先讀取檔案。`observe_object()` 用 `struct.unpack_from` 解析 ELF64 的 section header、`.symtab`、`.strtab` 和 `.rela.text`。它寫入一筆 `inspect_object` 紀錄，並為每個函式 symbol 寫入一筆 `read_function_bytes` 紀錄。五筆全都是 `observed`，權威為 `shipped-artifact`，因為來源是磁碟上的檔案。

| 紀錄 | 內容 | Id |
|---|---|---|
| `inspect_object` | 五個 section 的 offset、大小與 hash，四個函式，三個 relocation | `ev_645e0a145f28bd5c…` |
| `read_function_bytes percent_change` | `.text+0x00` 的 16 位元組 | `ev_2572a7d254ad15cb…` |
| `read_function_bytes grade` | `.text+0x10` 的 46 位元組，SHA-256 `ffcb75c2a7bb1104…` | `ev_a467c2427bfc0c56…` |
| `read_function_bytes total` | `.text+0x40` 的 31 位元組 | `ev_b1e845cea2543073…` |
| `read_function_bytes grade_label` | `.text+0x60` 的 63 位元組 | `ev_c50041be26e6b7f4…` |

`llvm-objdump -t` 同意：`grade` 位於 `0000000000000010`，大小為 `2e`。`.rela.text` 中的三個 relocation 是針對 `.rodata` 的 `R_X86_64_PC32`，位於 offset `0x36`、`0x84` 和 `0x93`，addend 為 `0x10`、`0x24` 和 `-0x4`。每筆紀錄都帶有 `file-offset-range` 位置與一個限制：「Section 位元組是從磁碟上的檔案讀取的。Reader 不套用 relocation，也不執行任何程式碼。」

課程 32 完整建構了這個 reader。這裡只有 60 行，因為這堂課要談的是你對這次讀取記錄了什麼。

## 推導表格，並說明你是推導出來的 (Derive the Table, and Say That You Derived It)

`grade` 的 46 個位元組包含課程 33 解碼過的四種模式：

| 位元組 | 指令 | 在 `grade` 中的 offset |
|---|---|---|
| `48 69 c0 67 66 66 66` | `imul rax, rax, 0x66666667` | 3 |
| `48 c1 f9 22` | `sar rcx, 34` | 13 |
| `83 c1 fa` | `add ecx, -6` | 23 |
| `83 f9 04 77 0c` | `cmp ecx, 4` 然後 `ja` | 28 |

乘以 `0x66666667` 再 shift 34，是有號的除以 10。檔案裡沒有 10。`derive_grade_model()` 把這個事實寫成限制：「除數 10 是從乘數 0x66666667 與 shift 34 推導出來的。檔案裡沒有 10。」

`add ecx, -6` 讓第一個 case 變成 6。帶有無號跳轉的 `cmp ecx, 4` 允許五個 case，也就是 6 到 10。

`.text+0x33` 的 `lea` 帶有 addend `0x10` 的 relocation。PC32 relocation 是從 4 位元組 displacement 的結尾開始量測，所以表格從 `.rodata+0x14` 開始。那裡的五個 little-endian word 讀作 `1, 2, 3, 4, 4`。`ledger.c` 中的 switch 對 60 到 69 回傳 1、七十幾分回傳 2、八十幾分回傳 3，而 90 到 100 回傳 4。編譯器把 switch 變成了回傳值的查詢表。

得到的紀錄 `ev_605106bff2d9998f…` 的 `confidence` 是 `derived`，而它的 `evidence_links` 指名它所依據的兩筆 observed 紀錄。把位元組紀錄從 ledger 中移除，derived 紀錄就有了懸空的連結。當你關掉 `ev_a467c242…` 時，上面的圖會顯示那種狀態。

現在與 Ghidra 比較。它對相同位元組的 pseudocode：

```c
uint grade(int param_1)
{
  uint uVar1;
  uint uVar2;

  uVar2 = param_1 / 10 - 6;
  uVar1 = -(param_1 >> 0x1f) ^ (uint)((ulong)((long)param_1 * 0x66666667) >> 0x3f);
  if (uVar2 < 5) {
    uVar1 = *(uint *)(&DAT_001000b4 + (ulong)uVar2 * 4);
  }
  return uVar1;
}
```

Ghidra 套用了 relocation，並在它把 `.text` 放在 `0x100000` 之後，把表格命名為 `DAT_001000b4`，也就是 `.rodata+0x14`。它還原了除以 10 與五個 case。對於預設值，它印出一個永遠是 0 的表達式，卻從來不說 0。`native_value_flow` 中的 p-code 把那些操作標記為 `is_dead: true`。這筆紀錄是對 Ghidra 的忠實觀察。你從中讀到的意義是 derived，而你預期從執行得到的值是 inferred。

`percent_change` 的紀錄顯示了另一種失敗。Ghidra 的簽章寫作 `undefined1 [16] percent_change(int param_1,int param_2,ulong param_3)`。原始碼寫的是 `int percent_change(int before, int after)`。`idiv` 寫入 `EDX:EAX`，而零的路徑不碰 `EDX`，所以反編譯器推論出 16 位元組的回傳值與第三個參數。它自己針對 `grade` 的 `native_api.return_type.confidence` 是 `"low"`。把還原的簽章視為 provider 的推論，並這樣記錄。

## 預測是 inferred 紀錄 (Predictions Are Inferred Records)

預測是關於行為的主張，沒有任何觀察陳述過它。`claim_suite()` 從 derived 模型和原始碼建構十八個預測。`record_prediction()` 把每個預測儲存為紀錄，`confidence` 為 `inferred`，`authority` 為 `analyst-inference`，並連結到它所依據的紀錄。

| 主張 | 預測值 | 依據 |
|---|---|---|
| `percent_change(12400, 15500)` | 25，來自 310000 / 12400 | 除法模型 `ev_076a9167…` |
| `percent_change(0, 5)` | 0，零的防護 | 除法模型 |
| `percent_change(100, 50)` | -50 | 除法模型 |
| `percent_change(-7, 3)` | -142，因為 `idiv` 往零的方向截斷 | 除法模型 |
| `grade(55)` | 0，index -1 無法通過無號比較 | 表格模型 `ev_605106bf…` |
| `grade(60)`、`grade(69)` | 1、1 | 表格模型 |
| `grade(70)`、`grade(85)` | 2、3 | 表格模型 |
| `grade(95)`、`grade(100)`、`grade(105)` | 4、4、4 | 表格模型 |
| `grade(110)`、`grade(-5)` | 0、0 | 表格模型 |
| `total(1, 2, 3)`、`total()` | 6、0 | 原始碼紀錄 `ev_2f1e6d25…` |
| `grade_label(85)` | `"B"` | 原始碼紀錄與表格模型 |
| 90 到 100 的 `grade(score)` | 全部十一個輸入皆為 4 | 表格模型 |

`predict()` 使用 C 的語意，而不是 Python 的語意。`c_div()` 往零截斷，而 `wrap32()` 把每個中間值折疊到 32 位元。Python 的 `//` 往負無限大捨入，所以 `(3 - -7) * 100 // -7` 得到 -143，而 C 程式得到 -142。課程 36 會把這個格子擴大很多。

每個預測開啟一個剩餘的未知。該未知要求權威為 `controlled-replay` 的 `observed` 證據，並建議一個探測：在 host 的 build 上執行 `execute_function`。下一節產生那份證據。第一個 grade 主張的問題寫作「當編譯後的函式執行時，grade(95) == 4 成立嗎？」

## 對照執行來檢查 (Check Them Against a Run)

`code/fixtures/harness.c` 包含 `ledger.c`，並在每次呼叫時印出一個結果。`observe_runs()` 用 PATH 上第一個找到的 `clang`、`cc` 或 `gcc` 編譯它，並對格子中的每個輸入各執行一次。每次執行變成一筆紀錄：

```json
{
  "operation": "execute_function",
  "parameters": { "function": "grade", "args": [95] },
  "normalized_result": { "result": 4, "stdout": "4\n", "exit_code": 0 },
  "confidence": "observed",
  "authority": "controlled-replay",
  "environment": { "id": "host-process", "platform": "darwin", "architecture": "arm64", "isolation": "process" },
  "limitations": [
    "The run executed a host build of ledger.c, not the x86-64 object. The claim about the object rests on the shared source and on the compiler.",
    "One run observes one input. It proves nothing about other inputs."
  ]
}
```

執行紀錄的 subject 是 harness 的 binary，在這台 Mac 上是 Mach-O arm64 檔案，帶有自己的 SHA-256。Provider 是 `course-harness`，帶有編譯器的版本行，這裡是 `Apple clang version 17.0.0 (clang-1700.6.3.2)`。`evidence_links` 指名原始碼紀錄 `ev_2f1e6d251543d017…`，因為執行與 object 除了 `ledger.c` 之外沒有共用任何東西。

把第一個限制讀兩遍。x86-64 object 從未在這台機器上執行過。執行的是同一份原始碼針對另一種架構的第二個 build。那仍然是觀察，而且是沒有模擬器時最強的一種，所以主張在附上該句話的情況下通過。省略它的報告，丟失了讀者所需的資訊。

17 次單一輸入的執行都與預測一致。`code/fixtures/run-evidence.json` 包含那 17 筆紀錄與原始碼紀錄，於 2026-10-13 以 `--record` 匯出。當沒有編譯器時，`build_ledger()` 改為匯入該 bundle 而不是執行，而報告會在它的 `runs` 欄位中說明這點。

## 升級規則 (Promotion Rules)

主張只有透過 observed 的執行，才能從 inferred 移到 supported。`check_claim()` 尋找函式與引數相同的 `execute_function` 紀錄，將每個結果與預測比較，並計算涵蓋的輸入：

| 狀態 | 條件 | rea 的 completion ledger |
|---|---|---|
| `pass` | 每個輸入都有一致的執行，且沒有不一致的 | `pass` |
| `fail` | 任何執行不一致 | `fail` |
| `unknown` | 有些輸入有執行，其他沒有 | `unknown` |
| `unsupported` | 沒有任何輸入有執行 | `unsupported` |

`promote_claim()` 是嚴格的形式。當任何 id 是 `inferred` 或 `derived` 時，它會丟出 `ClaimError`。當 id 指名的是另一個函式或其他引數的執行時，它會丟出例外。當清單為空，以及執行涵蓋的輸入少於主張時，它會丟出例外。對 `grade(55)` 而言，Ghidra 的 pseudocode 紀錄與表格模型都存在，而兩者都不能升級它。在圖中關掉 `ev_7be90c0c…`，看主張掉到 `unsupported`，而兩筆關於 `grade` 的紀錄依然存在。

範圍主張顯示了部分涵蓋的情況。它的十一個輸入中，只有兩個（95 和 100）出現在執行格中。`check_claim()` 回傳 `unknown`，`covered` 為 11 個中的 2 個，而 `promote_claim()` 以「2 of 11 inputs have observed runs」拒絕。Ledger 不會把它四捨五入。

`completion_ledger()` 對應 rea 的 `EvidenceCompletionLedger`。它的 id 是 `ecl_` 加上其紀錄的 hash，而只有當 `pass` 等於 `total` 時，`complete` 才為 true。Demo 印出 `ecl_da64933ea13befed…` 與 `complete: false`，原因就是那一個範圍主張。

## 未知是帶有 hash 鏈的紀錄 (Unknowns Are Records with a Hash Chain)

rea 把剩餘的未知儲存為嚴格的紀錄。`code/main.py` 保留相同的欄位與相同的不變量。

Id 是 `unk_` 加上對 `{domain, question, scope_digest}` 的 SHA-256，其中問題的空白被折疊成單一個空格。兩位分析者對同一個檔案問同樣的問題，會得到相同的 id。`scope_digest` 是 object 的 SHA-256。

每一次變更都是新的修訂版本。修訂版本 1 的 `previous_revision_digest` 是 `null`。之後的每個修訂版本都指名前一個版本的 `revision_digest`，而 digest 涵蓋除了它自己之外的每個欄位。`verify_unknown_chain()` 走訪清單並重新計算每個 digest。它檢查每個連結都指向前一個修訂版本。它檢查每個修訂版本所指名的變更證據都在 ledger 中。

每次變更本身就是一筆證據紀錄，就像 rea 以 provider `rea-unknown-registry` 所做的。我們的紀錄帶有 `confidence` `derived`、`authority` `analyst-inference`，而且沒有 subject。它的限制寫著「Registry 的變更記錄的是分析者決定了什麼。它不是答案的證明。」

這就是 demo 的 81 筆紀錄中包含 36 筆 derived 變更紀錄的原因：18 次開啟和 18 次更新。

每個修訂版本都強制執行三個不變量：

- 只有 `resolved` 的未知帶有 `resolution`，而且只有修訂版本 1 沒有前身。
- `contradicted` 的未知至少指名一個矛盾的 id，而且沒有任何 id 同時出現在兩份清單中。
- `verified` 的 resolution 至少指名一個具備所需信心與權威的證據 id。

第三條規則是 ledger 拒絕最多的地方。rea 的 workflow 筆記說「只有合格的 observed 證據才能驗證 resolution。」因此，`resolve_unknown()` 拒絕表格模型，拒絕權威為 `shipped-artifact` 的 observed 紀錄，並接受執行紀錄 `ev_f4ffbe8f922ec7fe…`。`grade(95)` 的 resolved 修訂版本是修訂版本 2，而它的 `previous_revision_digest` 是 `ffca840b55a8a188…`，也就是修訂版本 1 的 digest。

`update_unknown()` 接受 `expected_revision`，並在它過期時丟出 `revision-conflict`，就像 rea 的 `update_unknown` tool 那樣。正確的反應是 rea 筆記給的那個：重新讀取目前的修訂版本，然後重新決定。

Demo 之後仍有一個未知保持開啟：`unk_f2eb0d7f84c80a19…`，「當編譯後的函式執行時，90 到 100 的每個分數，grade(score) == 4 都成立嗎？」它的狀態是 `investigating`，它支持的 id 包含兩次執行，而它的探測仍然指名沒有執行的那九個輸入。

## 答案的快取不是分析的資料庫 (A Cache of Answers Is Not a Database of Analysis)

rea 有三樣東西看起來像快取。Ghidra 的目標 snapshot 是輸入的私有副本，以 `-readOnly -deleteProject` 匯入，並在關閉時刪除。記憶體內的 `AnalysisSnapshotCache` 為一個 binary 識別持有最多 10,000 個項目。Snapshot 檔案持有 `target`、`binding`、`entries`、`workflow_entries` 和一個 `evidence_bundle`。

快取 key 是 `query_` 加上對 `{target, binding, operation, parameters}` 的 SHA-256。`target` 是 `{sha256, kind, format, architecture}`，而 `raw-binary` 目標還有 `entry_point`。`binding` 是 provider 與整個分析 profile，包含 digest。改變 25 個 profile 參數中的任何一個，例如把 `jump_table_evidence` 從 `typed-case-default-blocks-v1` 改成 `v2`，digest 就會改變，所以每個 key 都會改變。

`code/fixtures/rea-snapshot.json` 是 rea 在兩次執行期間寫下的檔案，142,690 位元組。rea 以僅限擁有者的權限寫入它，而已 commit 的副本任何人都可讀。它有兩個項目，以及一個包含相同兩筆紀錄的 evidence bundle。`AnswerCache.import_snapshot()` 重新計算每個 `query_id`，並在任何一個不同時拒絕該檔案。它拒絕 `target` 或 `binding` 與目前使用中的不同的 snapshot，並使用 rea 的措辭 `profile_mismatch`。它拒絕 `format_version` 不是 `course-answer-cache-1` 的 snapshot。

這台機器上的三項量測。第一次 `grade` 查詢花了 8.6 秒，是全新的 Ghidra 匯入。針對同一個 snapshot 的 `percent_change` 查詢花了 8.5 秒，因為不同的查詢是快取 miss。第三次 `grade` 查詢花了 2.1 秒，並回傳具有相同 `evidence_id` 的相同紀錄。rea 中記憶體內的重播會加上限制「Loaded from a local REA analysis snapshot; this call did not re-run the provider.」。我們的 `lookup()` 加上同類的句子，讓讀者能分辨重播與全新的答案。

變更會結束快取。在 rea 6.5.0 中，重新命名或註解會在 rea 把呼叫送到 provider 之前，清除 snapshot 快取。失敗的回覆無法證明寫入已 rollback，所以失敗的寫入也會停止重播。改變資料庫的 metadata 匯入也一樣。

## 為什麼 agent 需要這個 (Why an Agent Needs This)

透過 MCP 驅動 rea 的 agent，在每次呼叫都看到完整的紀錄，並能在報告中引用 `evidence_id`。2026 年的研究指出問題會出在哪裡。對 tool 輸出的過度信任，是 NDSS BAR 研究中的四種弱點之一。在植入文字的研究中，單單把 binary 內容顯示為資料，40 個中仍有 15 個不安全的提議，而出處檢查搭配 tool 授權才消除了它們。第三項研究觀察 agent 透過 MCP 驅動 Ghidra（[arXiv 2606.06838](https://arxiv.org/abs/2606.06838)）。在有數百個函式的 binary 中，在 agent 產生 stub 之前，只有 10% 到 15% 得到真正的分析。

Ledger 用一筆紀錄回應這每一項，而 agent 只回報 ledger 所持有的內容。沒有 observed 執行的主張是 `unsupported`。有部分執行的主張是 `unknown`。Provider 的文字對 provider 而言是 `observed`，除此之外什麼也不是。把 `outputs/` 中的 skill 檔案交給 agent，並讓 ledger 成為它寫下結論的唯一地方。這適用於 Claude、Cursor、Codex、OpenClaw、Hermes，或任何讀取 SKILL.md 的 agent。

## 動手做 (Build It)

執行標準函式庫的 demo：

```bash
cd phases/13-tools-and-protocols/35-evidence-confidence-and-unknowns
python3 code/main.py
```

Demo 讀取 object、匯入兩筆 rea 紀錄、驗證 rea 的 snapshot，並推導兩個模型。接著它記錄 18 個預測、開啟 18 個未知、編譯並執行 harness、檢查每個主張，並 resolve 17 個未知。預期的形狀：

```json
{
  "records": 81,
  "by_confidence": { "observed": 25, "derived": 38, "inferred": 18 },
  "rea_snapshot": { "entries": 2, "replay_hit": true, "stale_profile_refused": true },
  "grade_table": [1, 2, 3, 4, 4],
  "runs": "live: Apple clang version 17.0.0 (clang-1700.6.3.2)",
  "claims": [ { "claim_id": "grade(95)", "status": "pass", "covered": "1/1" }, { "claim_id": "grade(90..100)", "status": "unknown", "covered": "2/11" } ],
  "completion_ledger_id": "ecl_da64933ea13befed9483f8b00447bb3f4ceb74324c01e2515eff03f3af3ca750",
  "complete": false,
  "unknowns": { "resolved": 17, "investigating": 1 }
}
```

沒有 C 編譯器時，`runs` 讀作 `imported 17 recorded run records from run-evidence.json`，其餘相同。

依此順序閱讀 `code/main.py`：

1. `canonical_json()` 與 `compute_evidence_id()` 持有 hash 的做法與語意投影。
2. `create_evidence()` 與 `validate_evidence()` 建構並檢查紀錄。
3. `EvidenceLedger.record()` 與 `import_bundle()` 儲存紀錄並拒絕衝突。
4. `open_unknown()`、`update_unknown()` 與 `resolve_unknown()` 維持修訂版本鏈。
5. `observe_object()` 讀取 ELF，而 `derive_grade_model()` 讀取表格。
6. `claim_suite()`、`predict()` 與 `record_prediction()` 寫入 inferred 層。
7. `observe_runs()` 把每次 harness 執行變成 observed 紀錄。
8. `check_claim()`、`promote_claim()` 與 `completion_ledger()` 套用升級規則。
9. `AnswerCache` 像 rea 的 snapshot 一樣為答案建立 key，並拒絕過期的答案。

## 實際使用 (Use It)

每個目標、每次調查各保留一份 ledger。把目標的 SHA-256 放進每個 subject，並把問題的 scope digest 放進每個未知。當你換 tool 時，換掉 `provider` 並讓 id 分歧。當你重建 fixture 時，subject digest 會改變，而每筆舊紀錄都變成關於另一個檔案的證據。

在你知道限制的當下就寫下它們。不套用 relocation 的 reader 要這麼說。host build 的執行要這麼說。推論出除數的模型要這麼說。沒有寫下限制的紀錄，看起來比事實所允許的更強，而之後的讀者無法修補這點。

把每個 provider 的文字都視為資料。來自目標的字串與 pseudocode，是由目標控制的文字。它們永遠不會變成指令，也永遠不會變成行為的觀察。rea 6.5.0 用 `_meta.provenance.untrusted_content: true` 標記這類結果。它的 server 指令告訴模型，把目標的內容視為資料，並把其中任何要求執行、安裝、下載或傳送東西的請求，回報為 prompt injection 的嘗試。這個標記有幫助，但你的 ledger 仍然必須讓目標的文字遠離指令。

## 互動實驗 (Interactive Lab)

### 實驗 A：重新計算 rea 的 id

從 `code` 目錄開啟 Python：

```bash
cd phases/13-tools-and-protocols/35-evidence-confidence-and-unknowns/code
python3 -q
```

執行：

```python
from main import *
record = load_evidence_file(FIXTURES / "rea-evidence-grade.json")
record["evidence_id"] == compute_evidence_id(record)
record["subject"]["local_path"] = "/home/you/re/ledger-x86_64-O1.o"
record["evidence_id"] == compute_evidence_id(record)
record["limitations"].pop()
record["evidence_id"] == compute_evidence_id(record)
```

前兩個比較為 true。路徑不在 hash 中。第三個為 false，因為限制是內容。

### 實驗 B：拒絕一次升級

```python
ledger, report = build_ledger(live_runs=False)
claim = [c for c in claim_suite(ledger.find(operation="derive_switch_model")[0], ledger.find(operation="derive_arithmetic_model")[0], ledger.find(operation="read_source")[0]) if c.claim_id == "grade(55)"][0]
promote_claim(ledger, claim, [ledger.find(operation="derive_switch_model")[0]["evidence_id"]])
```

這個呼叫會丟出 `ClaimError` 並指名 derived 的 id。現在傳入執行的 id：

```python
run = run_records_for(ledger, "grade", [55])[0]
promote_claim(ledger, claim, [run["evidence_id"]])["status"]
```

狀態是 `pass`，而回傳的 `limitations` 帶有 host-build 的句子。

### 實驗 C：拒絕過期的 snapshot

```python
import json
snapshot = json.loads(REA_SNAPSHOT_PATH.read_text())
target, binding = rea_snapshot_binding(snapshot)
AnswerCache(target, binding).import_snapshot(snapshot)
stale = dict(binding["analysis_profile"]["parameters"], jump_table_evidence="typed-case-default-blocks-v2")
AnswerCache(target, {"provider": binding["provider"], "analysis_profile": make_profile(binding["provider"], stale)}).import_snapshot(snapshot)
```

第一次匯入回傳 2。第二次丟出帶有 `profile_mismatch` 的 `CacheError`。只改變了一個版本字串，整個 snapshot 就被拒絕。

### 實驗 D：破壞鏈

```python
unknown_id = [u for u, r in ledger.unknowns.items() if r[-1]["status"] == "investigating"][0]
ledger.verify_unknown_chain(unknown_id)
ledger.unknowns[unknown_id][0]["severity"] = "critical"
ledger.verify_unknown_chain(unknown_id)
```

第一次呼叫回傳 2。第二次丟出 `EvidenceIntegrityError`，因為修訂版本 1 不再符合它自己的 digest，而修訂版本 2 指向一個已不存在的 digest。

## 實作練習 (Practice Lab)

關閉那個開啟的未知，並新增第二個 provider。

需求：

- 擴充 `run_grid()`，讓範圍主張缺少的九個輸入都被執行，然後確認未知在修訂版本 3 移到 `resolved`。
- 新增 `llvm-objdump -d` 作為第二個 provider，擁有自己的 `provider` 物件，並把它對 `grade` 的清單記錄為 `observed`。
- 寫一筆 `derived` 紀錄，陳述兩份清單（Ghidra 的與 llvm-objdump 的）解碼出相同的 14 條指令，並連結兩者。
- 為 `grade_label(105)` 開啟嚴重程度為 `high` 的未知，因為表格 index 4 讀取 `labels[4]`，並用一次執行 resolve 它。
- 用 `export_bundle()` 匯出 ledger，匯入到全新的 ledger，並證明每條鏈仍然驗證通過。

成功的意思是，產生一份報告，其中每個句子都引用 bundle 中的 id，而且那一個 `unknown` 主張已經消失。

## 評分實驗 (Graded Lab)

關於這堂課 fixture 的五個問題。用你建構的程式碼找出每個答案，然後寫進 `code/lab_answers.json`。數字寫成數字或像 `"0x40"` 這樣的十六進位字串，清單寫成 JSON 清單。

1. `t1`：compute_evidence_id 對 rea-evidence-grade.json 給出的 evidence id。
2. `t2`：grade 紀錄中的限制數量。
3. `t3`：grade 紀錄的 relationship_coverage status。
4. `t4`：grade 紀錄的 confidence 與 authority，以兩個字串的清單表示。
5. `t5`：grade 紀錄的 subject 的 SHA-256。

檢查你的答案：

```bash
cd phases/13-tools-and-protocols/35-evidence-confidence-and-unknowns
python3 code/lab.py
```

檢查器會依據 `code/main.py` 在已 commit 的 fixture 上計算每個預期值。它為每個任務印出 `PASS`、`FAIL` 或 `TODO`，以及一個分數，只有當五個全部通過時才以 0 結束。`python3 code/lab.py my-answers.json` 檢查另一個檔案。`code/tests/test_lab.py` 證明檢查器接受正確的答案，並拒絕錯誤的答案。

### 看它執行

[Inside a Decompiler](https://aiengineeringfromscratch.com/inside-a-decompiler?symbols=off) 這台機器維護一份證據 ledger，有四個托盤：observed、derived、inferred 和 unknown。這個連結開啟關閉 symbol table 的執行，所以函式的名稱與結尾歸入 unknown，而起點歸入 inferred。

## 課程成品 (Lesson Artifact)

這堂課會產出 `outputs/skill-evidence-ledger.md`。它告訴 agent 如何把每個發現寫成帶有信心、權威、限制與連結的紀錄。它告訴 agent 如何把未知保存為修訂版本，以及主張何時是 `pass`、`fail`、`unknown` 或 `unsupported`。

## 驗證 (Verify It)

執行 demo 與確定性的測試套件：

```bash
cd phases/13-tools-and-protocols/35-evidence-confidence-and-unknowns
python3 code/main.py
python3 -m unittest discover -s code/tests -v
```

28 個測試證明：

- 兩個 rea 的 `evidence_id` 值、profile digest，以及兩個 snapshot 的 `query_id` 值，都能從內容重新計算
- 改變的路徑保留 id，而改變的限制會改變它
- 被竄改的紀錄、profile provider 不一致、浮點數，以及未知的 enum 值都被拒絕
- 重複的紀錄回傳 `duplicate`，衝突的紀錄丟出例外，而 bundle 匯入是 atomic 的
- 已記錄的執行 bundle 匯入 18 筆紀錄，沒有懸空的連結
- 未知以修訂版本 1 開啟，拒絕過期的 `expected_revision`，且只在有 observed 的 `controlled-replay` 證據時才 resolve
- 被編輯過的修訂版本會破壞鏈
- 升級拒絕 inferred、derived、空的與不相關的證據，並在有一致的執行時通過
- 部分涵蓋是 `unknown`，而不一致的執行是 `fail`
- 快取在確切的 key 上命中，在其他參數上 miss
- 快取拒絕被提升版本的 profile、錯誤的目標、錯誤的 format version，以及被竄改的 key
- ELF reader 在 offset、大小、位元組與 relocation 上與 `llvm-objdump` 一致，而表格讀作 `1, 2, 3, 4, 4`
- 預測使用 C 的除法，而已記錄的執行與全部 17 個單一輸入主張一致
- 在有編譯器的情況下，即時的執行也一致

## Production 的失敗模式 (Production Failure Modes)

| 失敗 | 薄弱的報告說什麼 | Ledger 要求什麼 |
|---|---|---|
| Pseudocode 被讀成行為 | 「grade 對不及格的分數回傳 0」 | 針對該輸入的 `execute_function` 紀錄，否則是 `unsupported` |
| 信任反編譯器的簽章 | 「percent_change 回傳 16 位元組」 | 把簽章標記為 provider 的推論，並附上它的 `low` 信心 |
| 空清單被讀成不存在 | 「這個函式沒有 caller」 | `unknown` 狀態，或 tool 提供時的涵蓋欄位 |
| 路徑被 hash 進 id | 複製後「紀錄改變了」 | 不含 `name` 和 `local_path` 的語意投影 |
| 副本中遺失限制 | 看起來比來源更強的紀錄 | 匯入時 id 不一致並被拒絕 |
| 部分格子被四捨五入 | 「90 以上的所有分數都通過」 | `unknown`，`covered` 為 11 個中的 2 個 |
| Host 執行被當成 object 的執行 | 「x86-64 的 object 回傳了 25」 | 主張上帶有 host-build 的限制 |
| 未知被一句話關閉 | 「已解決，見上文」 | 帶有所需權威的 observed id 的 `verified` resolution |
| 過期的修訂版本被覆寫 | 兩位分析者的編輯衝突 | `revision-conflict` 與重新讀取 |
| 發布後重播舊 snapshot | 來自已改變合約的答案 | 匯入時 `profile_mismatch` |
| 重新命名後仍保留快取 | 隱藏 metadata 寫入的重播 | 在 dispatch 之前失效，且匯出失敗 |

## 營運規則 (Operational Rule)

把你讀到的、你計算的與你預期的，記錄為三種紀錄，用三個詞區分。用 id 關閉未知，絕不用句子。只在儲存時使用的確切 key 之下重播答案，並在任何一個版本字串不同時拒絕它。

## 延伸閱讀 (Further Reading)

- [rea 6.5.0 source, `src/domain/evidence.ts` and `src/domain/residualUnknown.ts`](https://github.com/morluto/rea)
- [RFC 8785, JSON Canonicalization Scheme](https://www.rfc-editor.org/rfc/rfc8785)
- [Ghidra 12.1.4 release](https://github.com/NationalSecurityAgency/ghidra/releases)
- [Agent Failure Patterns in CTF Reverse Engineering, NDSS BAR 2026](https://www.ndss-symposium.org/wp-content/uploads/bar2026-58.pdf)
- [When Binaries Talk Back: Representation-Confusion Attacks on LLM-Assisted Reverse Engineering, arXiv 2607.12507](https://arxiv.org/abs/2607.12507)
- [Agentic decompilation study, arXiv 2606.06838](https://arxiv.org/abs/2606.06838)
- [When LLM Decompilers Recompile More and Preserve Less, arXiv 2609.05370](https://arxiv.org/abs/2609.05370)
- 這堂課所描述的 rea 6.5.0 原始碼，位於 tag `3b79af2`：`src/domain/evidence.ts`、`.agents/skills/reverse-engineer-anything/SKILL.md`、`docs/cli.md`、`src/domain/reconstructionVerification.ts`、`src/application/ReconstructionObligationCandidates.ts`、`docs/tool-design.md`、`src/domain/evidenceCompletionLedger.ts`、`src/domain/residualUnknown.ts`、`src/application/investigation/UnknownEvidence.ts`、`.agents/skills/reverse-engineer-anything/references/evidence-workflows.md`、`src/application/binary/AnalysisSnapshotCache.ts`、`src/domain/analysisSnapshot.ts`、`src/application/binary/BinarySession.ts`、`src/application/binary/BinarySessionRecords.ts`、`docs/mcp-contracts.md`。
