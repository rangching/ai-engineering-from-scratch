# 超越原生程式碼：Python、JVM、.NET 與 WebAssembly Bytecode (Beyond Native Code: Python, JVM, .NET, and WebAssembly Bytecode)

> Bytecode 檔案是程式加上編譯器對它的筆記，而這堂課建構的 reader，會顯示每種容器保留了哪些筆記。

**Type:** Build
**Languages:** Python, Java, C#, WebAssembly text
**Prerequisites:** Phase 13 · 32 (ELF from scratch), 33 (disassembly and control flow)
**Time:** ~150 分鐘

## 學習目標 (Learning Objectives)

- 讀取 PEP 552 的 16 位元組 `.pyc` header，並在不使用 `marshal` 模組的情況下，為四個 interpreter 版本寫出的檔案 unmarshal code object。
- 解析 JVMS 第 4 章的 JVM class 檔案：constant pool、欄位、method、`Code` attribute，以及 `javac -g` 加入的表格。
- 反組譯 `Ledger.class` 中的 JVM opcode，並在同一份原始碼的四個 build 上，對照 `javap -c` 證明清單。
- 逐 section 讀取 WebAssembly 模組、解碼每個函式本文，並對照 `wasm-objdump -d` 證明結果。
- 走訪 .NET assembly 的 ECMA-335 metadata，從 PE data directory 到 `MethodDef` 列，以及每個 method 的 CIL，並對照 ILSpy 所用的 reader 函式庫檢查。
- 針對五種容器中的一個函式，量測每種格式保留了哪些名稱、型別與控制結構，又遺失了哪些。

## 問題 (The Problem)

你建構了 `ledger.py`，並在 `__pycache__` 中找到 `ledger.cpython-314.pyc`。你問 agent 這個檔案包含什麼。Agent 執行了反編譯器。工具印出 `Unsupported opcode` 以及一行 `# WARNING: Decompyle incomplete`。Agent 結論說這個檔案被混淆了。

這個檔案沒有被混淆。它的前兩個位元組是 `2b 0e`，也就是 little-endian 的 word 3627。那是 CPython 3.14 寫下的數字，而反編譯器的 opcode 表格停在 3.13。是工具過時了。Agent 沒有讀 header，所以分辨不出差異。

課程 33 顯示了原生程式碼失去什麼：參數名稱、型別，以及變成 jump table 的 `switch`。Bytecode 容器失去得較少，而且每一種失去的都不同。pyc 保留每個變數名稱，但沒有型別。Class 檔案保留型別，只有在你要求時才保留區域變數名稱。Wasm 模組保留控制結構，除非 custom section 攜帶名稱，否則丟掉名稱。.NET assembly 把參數名稱保留在表格中，並把區域變數名稱留給獨立的檔案。在你知道容器一開始有哪些筆記之前，你無法判斷任何一種反編譯器的輸出。

這堂課以 Python 標準函式庫建構四個 reader。每一個從容器的第一個位元組讀起、列出一個函式，並在你建構的 fixture 上，對照該格式自己的工具檢查。最後一節把 `ledger` 的 `grade` 並排放在全部五種容器中。

```figure
re-43-five-containers
```

## pyc header 是四個 word (The pyc Header Is Four Words)

PEP 552 將 header 定義為四個 32 位元 word。第一個是 magic number。第二個是位元欄位。位元 0 設定表示檔案是 hash-based。位元 1 設定表示 `check_source`：interpreter 在 import 時對原始碼做 hash 並比較。位元 0 清除時，第三和第四個 word 是原始碼的 mtime 與大小。位元 0 設定時，它們持有原始碼的 64 位元 SipHash。

Magic number 位於 `Include/internal/pycore_magic_number.h`。自 3.11 以來它的規則是，Python 3.n 從 2900 + 50n 開始，而一個 release 週期內每次 bytecode 變更都加一。儲存的 token 是 16 位元的數字後面接 `\r\n`，所以以文字模式開啟的 pyc 會得到錯誤的 magic。最終的值是 3.11 的 3495、3.12 的 3531、3.13 的 3571、3.14 的 3627，以及 3.15 的 3666。Python 3.15.0 於 2026-10-09 發布，並未安裝在建構這些 fixture 的機器上，所以它的數字來自 header 檔案，而不是這裡的檔案。

Fixture 為每個 interpreter 持有兩個 pyc 檔案。`compile_pyc.py` 用 `py_compile.compile(..., dfile="ledger.py")` 寫出它們，讓儲存的檔名是 `ledger.py`，而不是建構機器的路徑。3.14 檔案的前 16 個位元組：

```text
checked-hash   2b 0e 0d 0a  03 00 00 00  c7 16 c7 fc c5 1d 81 01
timestamp      2b 0e 0d 0a  00 00 00 00  8d 60 cf 6a  f1 01 00 00
```

逐 word 閱讀 timestamp header。`03 00 00 00` 或 `00 00 00 00` 是 flag word。`8d 60 cf 6a` 是 0x6acf608d = 1791975565，也就是建構檔案時 `ledger.py` 的 mtime。`f1 01 00 00` 是 497，原始碼大小。Checked-hash 檔案則攜帶 flag 3 與八個 hash 位元組。四個 interpreter 在 header 中除了 magic word 之外的一切都一致：

| Interpreter | Magic 位元組 | Word | 檔案位元組 | `grade.co_code` 位元組 | `grade` 指令 |
|---|---|---|---|---|---|
| 3.11.14 | `a7 0d 0d 0a` | 3495 | 1054 | 80 | 31 |
| 3.12.12 | `cb 0d 0d 0a` | 3531 | 937 | 62 | 26 |
| 3.13.12 | `f3 0d 0d 0a` | 3571 | 943 | 70 | 26 |
| 3.14.6 | `2b 0e 0d 0a` | 3627 | 965 | 96 | 35 |

`code/main.py` 中的 `read_pyc_header()` 回傳這些欄位，並從 word 命名版本。`importlib.util.MAGIC_NUMBER` 為執行中的 interpreter 給出相同的四個位元組。`python3 code/main.py compare` 用在 `PATH` 上找到的每個 `python3.11` 到 `python3.15` 編譯原始碼。對每一個，它印出我們的 reader 所看到的 header，旁邊是那個 interpreter 回報的關於它自己的事實。

## Marshal 逐欄位寫出 code object (Marshal Writes the Code Object Field by Field)

每個 fixture 的位元組 16 是 `e3`。那是 `0x80 | ord('c')`：設定了 `FLAG_REF` 的 code object 的 marshal 型別代碼，它告訴 reader 記住該物件供之後的 `r` 參照使用。Marshal 文件說這個格式是刻意不寫成文件的，而且 code object 在版本之間不相容。兩項陳述都是真的，而這個格式仍小到足以閱讀。`Python/marshal.c` 列出型別代碼：`N` None、`i` 32 位元 int、`s` bytes、`z` 和 `Z` 短 ASCII 字串、`(` 和 `)` tuple、`c` code object、`r` 回溯參照。容器在其子項之前先保留它的參照 slot，所以 reader 也做同樣的事。

`w_complex_object()` 以固定的順序寫出 code object。對 3.11 到 3.15，順序是 `argcount`、`posonlyargcount`、`kwonlyargcount`、`stacksize`、以 32 位元 int 表示的 `flags`，然後是物件 `code`、`consts`、`names`、`localsplusnames`、`localspluskinds`、`filename`、`name`、`qualname`，然後是 `firstlineno`，接著是 `linetable` 和 `exceptiontable`。從位元組 17 開始，手動解碼 3.14 檔案的 module code object：

```text
00 00 00 00   argcount 0
00 00 00 00   posonlyargcount 0
00 00 00 00   kwonlyargcount 0
01 00 00 00   stacksize 1
00 00 00 00   flags 0
f3 22 00 00 00   type 's' with FLAG_REF, 34 bytes of co_code follow
80 00  52 06  74 00  52 01  17 00  74 01 ...
```

Code 位元組每條指令兩個：`80 00` 是 `RESUME 0`、`52 06` 是 `LOAD_CONST 6`、`74 00` 是 `STORE_NAME 0`，而 `17 00` 是 `MAKE_FUNCTION`。Fixture 中的 3.14 清單印出相同的五個模組層級 statement。`code` 之後是常數，其中持有四個巢狀的 code object，然後是 `names`：`('LABELS', 'percent_change', 'grade', 'total', 'grade_label')`。

`code/main.py` 中的 `Unmarshaller` 讀取這個版面，以及 3.9 和 3.10 使用的較舊版面，其中 `nlocals`、`varnames`、`freevars` 和 `cellvars` 是獨立的欄位。它建構一個 `CodeObject`，屬性名稱與真正的物件相同。`varnames` 依 `localspluskinds` 的 `CO_FAST_LOCAL` 位元過濾 `localsplusnames`。對四個檔案中每一個的 `grade`，它回傳 `('score', 'bucket')`、`firstlineno` 10，以及非空的 `linetable`。測試 `test_unmarshaller_agrees_with_the_running_interpreter` 把每個欄位與對執行中 interpreter 所寫檔案的 `marshal.loads()` 比較。另外三個檔案可由我們的程式碼讀取，卻無法由這個 interpreter 下的 `marshal` 讀取。

## 四個 interpreter，grade 的四種 bytecode (Four Interpreters, Four Bytecodes for grade)

dis 文件在每個版本都帶有一項警告：bytecode 是 CPython 的實作細節，不保證它會保持不變。Fixture 顯示這對十一行 Python 意味著什麼。為每個 interpreter 擷取了 `python3.1N -m dis ledger.py`。

在 3.11 中，`grade` 這樣開始：

```text
 11           2 LOAD_FAST                0 (score)
              4 LOAD_CONST               1 (10)
              6 BINARY_OP                2 (//)
             10 STORE_FAST               1 (bucket)
 12          12 LOAD_FAST                1 (bucket)
             14 LOAD_CONST               2 (9)
             16 COMPARE_OP               5 (>=)
             22 POP_JUMP_FORWARD_IF_FALSE     2 (to 28)
```

`BINARY_OP` 位於 offset 6，而下一條指令在 10。它們之間的兩個位元組是 PEP 659（specializing adaptive interpreter）的 inline cache 項目。位於 16 的 `COMPARE_OP` 後面接 22，所以它擁有兩個 cache 項目。除非你傳入 `show_caches=True`，否則 cache 是隱藏的。3.12 把 `LOAD_CONST` 加 `RETURN_VALUE` 換成 `RETURN_CONST`，把跳轉改名為 `POP_JUMP_IF_FALSE`，並把 `grade` 從 80 縮小到 62 位元組。3.13 印出標籤 `L1` 到 `L4` 而不是 offset，並為 CLI 加上顯示 cache 的 `-C` 與顯示 offset 的 `-O`。3.14 又移除了 `RETURN_CONST`，加入 `LOAD_SMALL_INT` 和 `LOAD_FAST_BORROW`，並在每個條件跳轉之後插入 `NOT_TAKEN`：

```text
 12         20       LOAD_FAST_BORROW         1 (bucket)
            22       LOAD_SMALL_INT           9
            24       COMPARE_OP             188 (bool(>=))
            26       CACHE                    0 (counter: 0)
            28       POP_JUMP_IF_FALSE        3 (to L1)
            30       CACHE                    0 (counter: 0)
            32       NOT_TAKEN
```

`-C -O` 的清單顯示 `grade` 內有 13 個 `CACHE` 項目：`BINARY_OP` 之後五個、每個 `COMPARE_OP` 之後一個、每個 `POP_JUMP_IF_FALSE` 之後一個。Adaptive 的 fixture 顯示 cache 的用途。在 200 次呼叫之後，`dis.dis(grade, adaptive=True)` 在一般清單有 `RESUME` 和 `COMPARE_OP` 的地方，印出 `RESUME_CHECK` 和 `COMPARE_OP_INT`，而 `total` 顯示 `FOR_ITER_LIST`、`BINARY_OP_ADD_INT` 和 `JUMP_BACKWARD_NO_JIT`。特化後的形式存在於相同的 `co_code` 位元組中。pyc 從不包含它們，因為特化是在記憶體中發生的。

這就是反編譯器有版本上限的原因。uncompyle6 3.9.3 陳述它接受從 Python 1.0 到 3.8 的 bytecode。decompyle3 3.9.3 讀取 3.7 和 3.8。xdis 6.3.0，也就是兩者背後的跨版本反組譯器，最高讀取到 3.13。pycdc 的 build 檔案中有最高到 `python_3_13.cpp` 的 opcode 表格，並在遇到未知的 opcode 時印出 `Unsupported opcode` 並將輸出標記為不完整。PyLingual 宣稱涵蓋自 3.6 起每個已發布的版本，並使用從 3.8 到 3.15 的 interpreter 來重新編譯並檢查它自己的輸出。對於 3.14 的檔案，前三個拒絕或失敗，只有後兩個適用。在挑選工具之前，先讀 magic word。

## Class 檔案保留了自己的字典 (The Class File Keeps a Dictionary of Itself)

JVMS Java SE 25 的 4.1 節把版面給成單一個結構：`magic`、`minor_version`、`major_version`、`constant_pool_count`、constant pool、`access_flags`、`this_class`、`super_class`、interface、欄位、method、attribute。Magic 是 `0xCAFEBABE`。Java SE 25 的虛擬機器接受 major 版本 45 到 69，而 minor 為 65535 則標記帶有 preview 功能的檔案。`Ledger-g.class` 的前十個位元組是 `ca fe ba be 00 00 00 41 00 5e`：major 65 是 Java 21，而 `0x5e` = 94 是 constant pool 的數量。由 javac 25.0.2 從同一份原始碼建構的 `Ledger-jdk25.class` 以 `ca fe ba be 00 00 00 45` 開頭：major 69。

Constant pool 是 class 檔案保留它自己字典的地方。`-g` build 中的項目 `#13` 是 `Methodref`，指向 class `#8` 和 name-and-type `#14`。跟隨這些 index 得到 `Ledger.grade:(I)I`，這正是 `javap -v` 在 `//` 之後印出的。Descriptor `(I)I` 是一個 `int` 參數與一個 `int` 結果。`total` 上的 `([I)I` 說明參數是 `int` 陣列。程式碼碰到的每個 method、欄位與 class 都有這樣的項目，而 4.10 節的 verifier 會拒絕指令與它們不一致的檔案。`code/main.py` 中的 `ClassFile` class 讀取表 4.4 的全部 17 個 constant tag，並把 `Long` 或 `Double` 視為兩個 slot。它在未知的 tag 以及最後一個 attribute 之後還有位元組時丟出例外。

Attribute 掛在 class、每個欄位、每個 method 以及 `Code` attribute 本身之下。JVMS 說虛擬機器必須忽略它不認識的 attribute，這就是為什麼可以在不改變 method 本文的情況下，新增或丟掉 debug 表格。javac 手冊說 `-g` 產生所有偵錯資訊，包括區域變數，預設只產生行號與原始碼檔案，而 `-g:none` 什麼都不產生。這三個 build 量測了這點：

| Build | 位元組 | Constant pool 數量 | `LineNumberTable` | `LocalVariableTable` | `SourceFile` | `grade` 程式碼位元組 |
|---|---|---|---|---|---|---|
| `javac -g` | 1832 | 94 | 7 個 method | 6 個 method | 有 | 50 |
| `javac` | 1581 | 82 | 7 個 method | 無 | 有 | 50 |
| `javac -g:none` | 1404 | 79 | 無 | 無 | 無 | 50 |

`grade` 的 50 個程式碼位元組在三個檔案中完全相同，測試也檢查了這點。`-g` 為 `grade` 加入的，是一列 `LocalVariableTable`：start 0、length 50、slot 0、名稱 `score`、descriptor `I`。沒有它，反編譯器看到的是型別為 `int` 的 slot 0，必須自己發明名稱。`StackMapTable` 留在每個 build 中，因為 verifier 需要它，而 `BootstrapMethods` attribute 也留下，因為 `main` 透過 `invokedynamic` 串接字串。

## 手動解碼 tableswitch (Decode tableswitch by Hand)

JVMS 第 6 章把每條指令定義為一個 opcode 位元組，加上該 opcode 所指名的 operand。分支 offset 是相對於分支 opcode 位址的 16 位元有號值，這與 x86-64 不同，後者的 displacement 是相對於下一條指令。`tableswitch` 和 `lookupswitch` 以零到三個 padding 位元組開頭，讓它們的 32 位元 operand 從 4 的倍數的位址開始。

`Ledger-g.class` 中 `grade` 從 pc 4 開始的位元組：

```text
aa                 tableswitch at pc 4
00 00 00           padding to pc 8
00 00 00 2c        default: 4 + 44 = 48
00 00 00 06        low 6
00 00 00 0a        high 10
00 00 00 2a        case 6:  4 + 42 = 46
00 00 00 28        case 7:  4 + 40 = 44
00 00 00 26        case 8:  4 + 38 = 42
00 00 00 24        case 9:  4 + 36 = 40
00 00 00 24        case 10: 4 + 36 = 40
```

三十六個位元組、一條指令，以及整個帶有 9 和 10 共用 case 的 `switch (score / 10)`。課程 33 中同一個 `switch` 的原生 build，需要邊界檢查、`lea`、`movslq`、`add`、indirect jump，以及另一個 section 中的五個 relocation。在這裡，目標就在指令裡面，它們屬於同一組這個事實也一樣。

`disassemble_jvm()` 涵蓋全部 202 個已定義的 opcode 與它們的 operand 寬度，遵循 padding 規則，並以 `javap` 的方式轉譯 operand。`compare_with_javap()` 解析擷取到的 `javap -c -p` 文字，包括多行的 `tableswitch` block，並對每條指令檢查 offset、mnemonic 與 operand。四個 build 在 108 條指令中全部 108 條一致。測試 `test_branch_offsets_are_relative_to_the_branch_instruction` 檢查位於 pc 1 的 `ifne` 落在 6、位於 13 的 `if_icmpge` 落在 33，以及位於 30 的 `goto` 跳回 10。

為這些課程從原始碼建構的 radare2 6.2.4 讀取同一個檔案。`rabin2 -I` 回報 `arch java`、`lang java 21`、`linenum true` 和 `lsyms true`，這是它對版本與兩個 debug 表格的解讀。`pd 14 @ sym.Ledger.grade` 印出 `tableswitch`，附有指向它五個目標的箭頭。它也在這個檔案上印出 `Unable to parse code attribute: 2`，這是工具自己指出的限制，也是保留 `javap` 作為 oracle 的理由。相較於 Python，這個格式的反編譯器已經很成熟。CFR 0.152 採用 MIT 授權。Procyon 0.6.0 和 Vineflower 1.12.0 是 Apache-2.0，而 Vineflower 是支援 Java 21 的 Fernflower fork。Fernflower 本身位於 IntelliJ Community repository 內。JADX 1.5.6 涵蓋 Android DEX 格式，而 JD-GUI 的最後一次發布是 2019 年。Harrand、Soto-Valero、Monperrus 和 Baudry 在 2020 年量測了其中八個。最好的一個為 84% 的 class 產生語法正確的輸出，並為 78% 產生語意等價的輸出。他們的 meta-decompiler 還原了沒有任何單一工具能處理的 class 的 37.6%。型別、名稱與經過驗證的 stack 讓這個格式可被反編譯。它們不會讓任何單一反編譯器是完整的。

## Wasm 模組是 section 的清單 (A Wasm Module Is a List of Sections)

核心規格把模組定義為前導碼 `00 61 73 6d 01 00 00 00`（字串 `\0asm` 與 version 1），後面接 section。每個 section 是一個 id 位元組、LEB128 表示的 `u32` 大小，以及它的內容。Id 1 到 13 是已知的 section，每個最多出現一次且依順序。Id 0 是帶名稱的 custom section，可以出現在任何地方。所有整數都是 LEB128，所以 `e4 00` 是 100，`80 08` 是 1024。`ledger.wasm` 上的 `wasm-objdump -h` 列出七個 section，而 `WasmModule` 找到相同的七個，payload offset 也相同：

| Section | Payload | 數量 | 它為 ledger 持有什麼 |
|---|---|---|---|
| Type | 10 到 26 | 3 | `(i32) -> nil`、`(i32, i32) -> i32`、`(i32) -> i32` |
| Import | 28 到 39 | 1 | 型別為 0 的 `env.log` |
| Function | 41 到 46 | 4 | 每個已定義函式的型別 index |
| Memory | 48 到 51 | 1 | 一頁，沒有最大值 |
| Export | 53 到 110 | 5 | `memory` 與四個函式名稱 |
| Code | 113 到 271 | 4 | 23、63、44 和 23 位元組的本文 |
| Data | 273 到 290 | 1 | offset 1024 處的 `F\0D\0C\0B\0A\0` |

Function 與 Code section 的分離，是為了讓瀏覽器能在整個檔案抵達之前就開始編譯。每個 code 項目是一個大小、一個區域變數宣告的 vector，以及一個 expression。對 `grade` 而言，區域變數 vector 是 `01 7f`：一組一個 `i32`。那就是 `bucket`，有型別但沒有名稱。

控制流程在建構上就是結構化的。規格說分支只能以外圍的 `block`、`loop` 或 `if` 結構為目標。標籤以巢狀深度為 index。`br_table` 透過 index 標籤清單的 operand 進行 indirect branch。在檔案 offset `0xa3` 解碼 `grade` 的 switch：

```text
0e                  br_table
0b                  11 labels follow
04 04 04 04 04 04   buckets 0 to 5: label 4, the outermost block, returns 0
03 02 01 00         buckets 6, 7, 8, 9: labels 3, 2, 1, 0
00                  bucket 10: label 0, shared with 9
04                  default label 4
```

每個標籤都是深度。標籤 0 是最內層的 `block`，而它的 `end` 之後的程式碼是 `i32.const 4` 後面接 `br 4`，跳出 result block。表格之前的六個 `block` 指令、十一個標籤，以及表格之後的五個 `br` 指令，就是整個 `switch`，而 validator 會拒絕任何離開該結構的分支。`disassemble_wasm()` 追蹤深度，讓清單像 `wasm-objdump` 一樣縮排，涵蓋從 `unreachable` 到 `i64.extend32_s` 的每個單一位元組 opcode，並在兩個 build 的 77 條指令中全部 77 條與擷取到的 `wasm-objdump -d` 相符。

名稱是普通模組丟掉的一樣東西。`wasm-objdump` 仍然印出 `<grade>`，因為 Export section 把那個字串對應到函式 index 2，而 reader 也這麼做。區域變數名稱不存在於 290 位元組的檔案的任何地方。`wat2wasm --debug-names` 加入名為 `name` 的 custom section，帶有模組名稱、函式名稱與區域變數名稱的子 section。檔案增長到 425 位元組。Reader 為函式 2 找到 `{0: 'score', 1: 'bucket'}`，為函式 3 找到 `{0: 'values', 1: 'count', 2: 'sum', 3: 'i'}`。`wasm-strip` 再次移除該 section。tool-conventions repository 描述了另外兩種載體。DWARF 可以內嵌為 `.debug_*` custom section。`sourceMappingURL` custom section 可以指向 source map，其欄號是進入 binary 的位元組 offset。

工具在 2026 年期間有所移動。2026-09-15 發布的 wabt 1.0.42 移除了 `wasm-decompile`，而這台機器上的 Homebrew build 恰好有其餘的十一個工具。wasm-tools 1.261.0 提供 `print`、`objdump` 和 `demangle`。Binaryen version_133 提供 `wasm-dis`。radare2 6.2.4 也讀取該模組。它的 `rabin2 -I` 說 `arch wasm`。它為 `grade` 給出的 symbol 起點是 `0x8e`，也就是區域變數 vector 開始的地方，而 `wasm-objdump` 把函式標記在 `0x8b`，也就是它的 size 欄位開始的地方。它的清單使用較舊的 mnemonic `get_local`、`set_local`，以及帶有標籤數量的 `brtable 11 4 4 ...`。三個工具的位元組相同，拼寫則不同，所以比較必須指名它擷取所用的工具。這堂課的研究在 Ghidra 的 processor 目錄中沒有找到 WebAssembly processor 模組，而該清單沒有被完整閱讀。第三方的 ghidra-wasm-plugin v2.4.0 以 Ghidra 12.0 為目標。規格本身是一份持續更新的文件。`wasm-core-2` 的 W3C 頁面是 2026 年 10 月 13 日的 Candidate Recommendation Draft，其摘要描述 3.0 版，而編輯草稿帶有相同的日期。

## ECMA-335 metadata 是程式自己的資料庫 (ECMA-335 Metadata Is the Program's Own Database)

.NET assembly 是 PE 檔案。`LedgerNet.dll` 以 `MZ` 開頭，有三個名為 `.text`、`.rsrc` 和 `.reloc` 的 section，並宣告 machine `0x14c`，因為編譯器送出與平台無關的程式碼。Optional header 的 data directory 14，也就是 CLR Runtime Header，指向 RVA `0x2008`。ECMA-335 II.25.3.3 定義了那裡的 72 個位元組：

```text
48 00 00 00   Cb 72
02 00 05 00   MajorRuntimeVersion 2, MinorRuntimeVersion 5
78 21 00 00   MetaData RVA 0x2178
84 07 00 00   MetaData size 1924
01 00 00 00   Flags: COMIMAGE_FLAGS_ILONLY
05 00 00 06   EntryPointToken 0x06000005, MethodDef row 5, which is Main
```

位於 RVA `0x2178` 的 metadata root 以簽章 `42 53 4a 42`（讀作 `BSJB`）開頭，然後是 version 1.1 與字串 `v4.0.30319`。接著是五個 stream header：offset `0x6c` 的 `#~`，然後是 `#Strings`、`#US`、`#GUID` 和 `#Blob`。II.24.2 說每種 stream 最多出現一次。`#~` stream 以 `HeapSizes` 位元組開頭（這裡為零），所以每個字串、GUID 和 blob index 都是兩個位元組寬，以及一個指名存在的表格的 64 位元 `Valid` mask。設定了十四個位元：

| 表格 | Id | 列數 | 列大小 | 這些列說明什麼 |
|---|---|---|---|---|
| Module | 0x00 | 1 | 10 | 模組名稱與它的 GUID |
| TypeRef | 0x01 | 21 | 6 | `System.Object`、`System.Console`、另外 19 個 |
| TypeDef | 0x02 | 4 | 14 | `<Module>`、`Ledger`，以及兩個編譯器 helper |
| Field | 0x04 | 2 | 6 | `Labels` 與一個 static 陣列的初始化器 |
| MethodDef | 0x06 | 6 | 14 | 五個 method 與 `.cctor` |
| Param | 0x08 | 5 | 6 | `before`、`after`、`score`、`values`、`score` |
| MemberRef | 0x0a | 16 | 6 | 12 個 attribute constructor 與 `Main` 呼叫的 4 個 method |
| CustomAttribute | 0x0c | 12 | 6 | 編譯器與 SDK 加入的 attribute |
| StandAloneSig | 0x11 | 3 | 2 | 三個區域變數簽章 |
| Assembly、AssemblyRef、ClassLayout、FieldRva、NestedClass | | 1、2、1、1、1 | | identity、reference、版面 |

列大小並沒有儲存在任何地方。II.22 為每個欄位給出種類，而 II.24.2.6 給出大小規則。Heap index 依 `HeapSizes` 是兩或四個位元組。簡單的表格 index，在該表格少於 65536 列時是兩個位元組。可以指向 n 個表格、帶有 b 個 tag 位元的 coded index，在這些表格每一個都少於 2^(16-b) 列時是兩個位元組。`CliAssembly` 從這些規則與列數計算 14 個列大小。Fixture 中的 System.Reflection.Metadata dump 回報相同的大小與相同的表格 offset，測試就是這樣證明算術的。

III.1.9 定義 token：高位元組是表格，低三個位元組是 1 起算的列。`0x06000002` 是 `MethodDef` 第 2 列。該列的 RVA 為 `0x2060`、名稱為 `Grade`，簽章 blob 為 `00 01 08 08`：預設呼叫慣例、一個參數、回傳 `int32`、參數 `int32`。它的 `ParamList` 指向 `Param` 的第 3 列，其名稱是 `score`。Param 表格就是 ILSpy 能從去除符號的 release build 印出 `Grade(int score)` 的原因。在 RVA `0x2060`，method 本文以 fat header 開始：

```text
13 30         flags 0x13: fat, init locals; size 3 dwords
02 00         MaxStack 2
2d 00 00 00   CodeSize 45
01 00 00 11   LocalVarSigTok 0x11000001
```

`StandAloneSig` 第 1 列持有 `07 01 08`：`LOCAL_SIG`、一個區域變數、`int32`。這個區域變數有型別而沒有名稱。ECMA-335 把區域變數名稱保存在 Partition V 的 debug 交換格式中，而 .NET SDK 把它們寫進 PDB。這個 fixture 是用 `DebugType` none 建構的，所以沒有任何表格或檔案帶有名稱 `bucket`。`PercentChange` 則有 tiny header：它的第一個位元組 `3a` 是 `0x0e << 2 | 2`，一個 14 位元組的本文，沒有區域變數，預設 stack 為 8。

`Grade` 的 45 個程式碼位元組以 Partition III 的 opcode 表格解碼。`switch` 是 opcode `0x45`，後面是 `u32` 的數量，以及那麼多個相對於表格之後那條指令的 `int32` 目標：

```text
IL_0000: 02           ldarg.0
IL_0001: 1f 0a        ldc.i4.s 10
IL_0003: 5b           div
IL_0004: 0a           stloc.0
IL_0005: 06           ldloc.0
IL_0006: 1c           ldc.i4.6
IL_0007: 59           sub
IL_0008: 45 05 00 00 00 08 00 00 00 06 00 00 00 04 00 00 00 02 00 00 00 02 00 00 00
                      switch (IL_0029, IL_0027, IL_0025, IL_0023, IL_0023)
IL_0021: 2b 08        br.s IL_002b
IL_0023: 1a 2a        ldc.i4.4, ret
```

編譯器在 `switch` 之前減去 6，所以 case 6 是第一個目標。`disassemble_cil()` 透過表格轉譯 token：`ldsfld Labels`、`call Grade`、`call System.Console::WriteLine`，以及來自 `#US` heap 的 `ldstr ' '`。這一切的 oracle 是 `MetaDump`，這是 fixture 中一個建立在 ILSpy 內部的 `System.Reflection.Metadata` 函式庫之上的簡短 C# 程式。它印出每個表格的列數、大小與 offset、每個 method 的 RVA、簽章、參數名稱與 IL 位元組，以及 `System.Reflection.Emit.OpCodes` 的 226 個欄位。`compare_with_metadump()` 在全部 246 項檢查上一致。Runtime 表格有 218 個 opcode 加上八個 `prefix` placeholder。課程的表格有 219 個項目，因為它包含了 III.2.2 的 `no.`，而 Emit 表格從未有過它。

這個格式的工具：ILSpy 11.0（MIT，2026-08-12 發布）透過它的 Avalonia 桌面 build 在 macOS 上執行。`ilspycmd` 11.1.0.9782 用 `-il` 印出 IL，並用 `-m 0x06000002` 依 token 選擇 method。dnSpyEx 6.6.0 延續 dnSpy，僅限 Windows。IL 反組譯器 `ildasm` 以 NuGet 套件發行，而不是在 SDK 內。radare2 6.2.4 也讀取 metadata：`rabin2 -I` 說 `arch cil` 和 `class PE32`，而 `is` 列出 `Ledger.Grade(int:score):int`，其參數名稱與兩個型別取自 Param 與簽章表格。這個 build 中沒有 CIL 反組譯器。這個 fixture 是用 .NET SDK 8.0.205 建構的，因為那是建構機器上安裝的 SDK。.NET 10 自 2025-11-11 起是目前版本，並寫入相同的 ECMA-335 結構。

## 每種容器保留什麼 (What Each Container Keeps)

把五種容器中的 `grade` 放在一張表中。每一列都來自 `python3 code/main.py keeps`，它在 fixture 上執行這些 reader：

| 容器 | 本文位元組 | 函式名稱 | 參數名稱 | 參數與回傳型別 | 區域變數名稱 | 行號 | 控制流程 |
|---|---|---|---|---|---|---|---|
| ELF object，clang -O0 | 118 | `.symtab` 中的 `grade` | 無 | 無 | 無 | 無，沒有 `.debug_*` section | 5 個跳轉與透過 relocation 表格的 1 個 indirect jump |
| CPython 3.14 pyc | 96 | `co_qualname` | `score` | 無 | `bucket` | `co_linetable`，第一行為 10 | 4 組 compare 與 `POP_JUMP_IF_FALSE` |
| JVM class，`javac -g` | 50 | constant pool | `LocalVariableTable` 中的 `score` | `(I)I` | 除了參數之外沒有 | `LineNumberTable`，6 列 | 1 個 `tableswitch` |
| WebAssembly | 63 | Export section | 僅在有 name section 時 | `(i32) -> i32` | 僅在有 name section 時 | 無 | 6 個巢狀 block、1 個 `br_table`、4 個 `br` |
| .NET assembly | 45 | `MethodDef` | `Param` 中的 `score` | `static int32(int32)` | 無，型別只在 `StandAloneSig` | 沒有 PDB 就沒有 | 1 個 `switch`、1 個 `br.s` |

這張表解釋了你會看到的反編譯器結果。課程 34 比較了 `int total(const int *, int)` 與 Ghidra 從原生程式碼推論出的 `int total(long param_1, uint param_2)`，因為原生程式碼沒有保留這些欄位的任何一個。pyc 保留每個名稱而沒有型別，而它的控制流程是形狀每個 release 都會改變的跳轉，所以 Python 的反編譯器依版本壞掉，並免費還原名稱。Class 檔案保留型別與經過驗證的 stack，在 `-g` 開啟時保留名稱，並把 `switch` 表達成單一個指令。Wasm 模組在建構上就保留結構，所以 `wasm2wat` 是無損的來回，唯一的損失是名稱。.NET assembly 保留最完整的描述，因為 metadata 表格是規範性的，卻仍需要 PDB 才能取得區域變數的名稱。

這張表告訴你反編譯器有什麼可以使用，以及它的哪些輸出是複製儲存的事實、哪些是推論一個。ILSpy 輸出中的參數名稱是複製。同一份輸出中的區域變數名稱是推論，除非有 PDB 存在。課程 35 把這個差異記錄為每項主張的權威。

## 動手做 (Build It)

執行 demo：

```bash
cd phases/13-tools-and-protocols/43-bytecode-and-managed-formats
python3 code/main.py
```

輸出是一份 JSON 報告。對每個 pyc，它給出兩個 header、檔案與 `grade` 的大小、來自擷取清單的指令數量與 opcode 名稱，以及執行中的 interpreter 是否能讀取它。對每個 class build，它給出版本、大小、debug 表格的數量，以及與 `javap` 的一致程度。對每個 Wasm build，它給出 section、型別、函式、name section 的內容，以及與 `wasm-objdump` 的一致程度。對 assembly，它給出 CLI header、stream、表格列數、method，以及與 System.Reflection.Metadata 的一致程度。它以 keeps 表格以及在 `PATH` 上找到的 interpreter 清單作結。

依此順序閱讀 `code/main.py`：

1. `Reader` 持有 cursor 與整數寬度。`uleb()` 與 `sleb()` 是 Wasm 和 .NET blob heap 需要的 LEB128 reader。
2. `read_pyc_header()`、`Unmarshaller` 和 `PycFile` 讀取 pyc。`parse_dis_text()` 讀取擷取到的 `dis` 清單，讓跨版本比較在測試時不需要四個 interpreter。
3. `ClassFile` 讀取 class 檔案，而 `disassemble_jvm()` 解碼它的程式碼。`parse_javap_code()` 與 `compare_with_javap()` 是 oracle 的檢查。
4. `WasmModule` 讀取 section，而 `disassemble_wasm()` 解碼帶深度的 expression。`compare_with_wasm_objdump()` 是 oracle 的檢查。
5. `CliAssembly` 走訪 PE、CLI header、metadata root、stream 與表格。`TABLE_SCHEMAS` 和 `CODED_INDEXES` 是以資料形式表示的 II.22 和 II.24.2.6。`disassemble_cil()` 解碼本文，而 `compare_with_metadump()` 檢查它。
6. `NativeObject` 讀取足夠的 ELF 來列出函式 symbol，並以課程 33 的清單作為它的文字。`what_each_keeps()` 與 `figure_data()` 建構比較。

還有另外五個指令。`python3 code/main.py pyc FILE`、`class FILE`、`wasm FILE` 和 `dll FILE` 印出你給它們的檔案中的結構，以及每個函式的清單。`python3 code/main.py compare` 用 `PATH` 上的每個 interpreter 編譯 `ledger.py`，並印出每一個寫了什麼。

## 實際使用 (Use It)

依此順序，在你建構的容器上使用這些 reader。

1. 讀取前幾個位元組並說出格式。兩個位元組後接 `0d 0a` 是 pyc，`ca fe ba be` 是 class 檔案，`00 61 73 6d` 是模組，設定了 data directory 14 的 `4d 5a` 是 assembly。
2. 讀取 header，並寫下它指名的 producer 版本。對 pyc 是 magic word，對 class 檔案是 major 版本，對 assembly 是 runtime 與 metadata 的版本字串。
3. 在執行你打算使用的每個反編譯器之前，先對照那個版本檢查它。記錄上限，以及工具在哪裡有文件記載。
4. 對同一個檔案執行 reader 與該格式自己的工具，並逐列比較。不一致是 reader 的 bug 或工具的變更，而兩者在清單成為證據之前都必須被理解。
5. 為你在意的函式填寫 keeps 表格。將每個事實標記為保留或遺失，並指名它來自的表格、attribute 或 section。
6. 之後才閱讀反編譯器的輸出。將其中的每個名稱與型別，標記為已保留事實的複製，或是推論。

把 oracle 文字檔與容器放在一起。當 interpreter、JDK、SDK 或 wabt 的版本改變時，用 `build_fixtures.sh` 重建並重新擷取，因為清單改變就是 fixture 改變。

## 互動實驗 (Interactive Lab)

### 實驗 A：讀取 header，然後讀取 code object

從 `code` 目錄開啟 Python：

```bash
cd phases/13-tools-and-protocols/43-bytecode-and-managed-formats/code
python3 -q
```

執行：

```python
from main import *
pyc = load_pyc("ledger.cpython-311.pyc")
pyc.header.to_dict()
pyc.module.names
pyc.functions()["grade"].varnames
len(pyc.functions()["grade"].code)
```

在第一次呼叫之前，先寫下 magic word。然後載入 `ledger.cpython-314.pyc`，並比較兩個 `code` 的長度。接著在你的 interpreter 下，對 3.11 檔案嘗試 `import marshal; marshal.loads(pyc.data[16:])` 並閱讀錯誤。

### 實驗 B：constant pool 的鏈

```python
cls = load_class("Ledger-g.class")
cls.constant(13)
cls.constant(14)
cls.describe_constant(13)
[row["text"] for row in jvm_listing(cls, "grade")]
```

在呼叫 `describe_constant` 之前，先手動從 `#13` 跟隨到 `#8` 和 `#14`，再到 `Utf8` 項目。然後載入 `Ledger-gnone.class`，並對兩者都呼叫 `cls.method("grade").attribute_names()`。`Code` 陣列相同，attribute 則不同。

### 實驗 C：深度與標籤

```python
module = load_wasm("ledger.wasm")
[(row["depth"], row["text"]) for row in wasm_listing(module, "grade")]
module.function_name(2)
load_wasm("ledger-names.wasm").names["locals"][2]
```

找到 `br_table` 那一列並閱讀它的深度。計算它之後的 `end` 列，並把每一個對應到表格中的標籤編號。然後開啟 `ledger.wat`，找到標籤 3 所指名的 `block $six`。

### 實驗 D：從 token 到位元組

```python
asm = load_assembly("LedgerNet.dll")
asm.method("Grade")
body = asm.method_body(asm.method("Grade")["RVA"])
body.fat, body.max_stack, body.code_size, hex(body.local_signature_token)
asm.token_text(0x11000001)
[row["text"] for row in cil_listing(asm, "GradeLabel")]
```

從 `.text` section header 手動計算 RVA `0x2060` 的檔案 offset，並與 `asm.rva_to_offset(0x2060)` 比較。然後查找 `asm.row(0x08, 3)`，並找到名稱 `score` 住在哪裡。

## 實作練習 (Practice Lab)

在比較中加入第五種原始碼語言，或在表格中加入第六項事實。

需求：

- 用你有編譯器的語言，寫一個帶有迴圈與區域變數的函式，例如 Kotlin 編譯到 class 檔案，或 Rust 編譯到 Wasm 模組。把輸出與 oracle 清單，連同 `build_fixtures.sh` 中確切的指令一起 commit。
- 讓對應的 reader 在每條指令上都與 oracle 一致。只有當 fixture 需要時才加入 opcode。
- 用新的容器擴充 `what_each_keeps()`，並為每個容器多加一項事實：本文是否攜帶例外表格或 `try` 區域。
- 把新的容器加進 `figure_data()` 並重新產生 `site/figures-re-43.js`。
- 寫一個測試，當 reader 回報某項事實為已保留、卻沒有指名持有它的表格、attribute 或 section 時就失敗。

成功的意思是，`python3 code/main.py keeps` 顯示新的一列，且每個格子都有證據，而每項 oracle 比較仍然一致。

## 課程成品 (Lesson Artifact)

這堂課包含 `outputs/skill-bytecode-and-managed-formats.md`。用它把 bytecode 容器變成 header 報告、一個函式的經過檢查的清單，以及將每個名稱、型別與結構標記為已保留或遺失的事實表。它要求以該格式自己的工具作為 oracle、在任何反編譯器之前先做版本檢查，以及說明每項遺失的事實原本會住在哪裡。

## 驗證 (Verify It)

執行 demo 與確定性的測試套件：

```bash
cd phases/13-tools-and-protocols/43-bytecode-and-managed-formats
python3 code/main.py
python3 -m unittest discover -s code/tests -v
```

驗證應該證明：

- 四個 pyc header 帶有 magic word 3495、3531、3571 和 3627、flag 3，以及 8 位元組的 hash，而 timestamp 檔案帶有 flag 0、mtime 1791975565 與大小 497
- 3.11 及之後的每個 magic word，都位於 `2900 + 50n` 規則為其 release 所給出的範圍內
- unmarshaller 回傳的名稱、常數、程式碼位元組、變數名稱與行表，與對執行中 interpreter 所寫檔案的 `marshal.loads()` 相同
- 四個檔案都保留模組名稱、`grade` 的 `('score', 'bucket')`、第一行 10，以及作為檔名的 `ledger.py`
- `grade.co_code` 在四個版本中分別是 80、62、70 和 96 位元組，而擷取的清單有 31、26、26 和 35 條指令
- 3.11 清單有 `POP_JUMP_FORWARD_IF_FALSE`，3.12 和 3.13 有 `RETURN_CONST`，而 3.14 有 `LOAD_SMALL_INT`、`LOAD_FAST_BORROW` 和 `NOT_TAKEN`
- 3.14 的 `-C -O` 清單在 `grade` 中顯示 13 個 cache 項目，其中五個在 `BINARY_OP` 之後，而 adaptive 清單顯示 `COMPARE_OP_INT`
- 沒有 `0d 0a` 的 header、被截斷的 header，以及不是 code object 的本文會丟出 `FormatError`
- 指令清單只存在於執行中 interpreter 所寫的 pyc，並與擷取的 `dis` 輸出相符
- class 檔案的 major 為 65 或 69，大小為 1581、1832、1404 和 1581，constant pool 數量為 82、94、79 和 82
- 常數 `#13` 解析為 `Method Ledger.grade:(I)I`，而 `#47` 解析為 `String F`
- JVM 反組譯器在四個 build 中的 108 條指令中全部 108 條與 `javap -c -p` 一致
- 位於 pc 4 的 `tableswitch` 是 36 位元組，low 為 6，high 為 10，default 為 48，目標為 46、44、42、40、40
- 三個 debug 層級改變 attribute 的數量，並讓 `grade` 的 50 個程式碼位元組保持相同
- `ifne`、`if_icmpge` 和 `goto` 的目標都相對於分支 opcode
- 錯誤的 magic、尾端位元組，以及未定義的 opcode 會丟出 `FormatError`
- 七個 Wasm section 具有 `wasm-objdump -h` 印出的 payload offset，而 names build 在 293 到 425 加入 `name` custom section
- 型別、`env.log` import、五個 export，以及位於 1024 的 data segment，如 `ledger.wat` 所寫的那樣解碼
- Wasm 反組譯器在兩個 build 的 77 條指令中全部 77 條與 `wasm-objdump -d` 一致
- `grade` 有 6 個巢狀 block、一個標籤為 `4 4 4 4 4 4 3 2 1 0 0` 且 default 在深度 6 為 4 的 `br_table`，以及一個 `i32` 區域變數
- 沒有 name section 時，函式名稱來自 export，區域變數名稱只有在有 name section 時才有
- LEB128 `e4 00` 是 100，`7f` 是 -1，`80 08` 是 1024，而 block 型別的轉譯像 `wasm-objdump` 一樣
- 錯誤的前導碼、version 2 與未知的 opcode 會丟出 `FormatError`
- assembly 是 PE32，machine 為 `0x14c`，CLI header 位於 RVA `0x2008`，runtime 2.5，`ILONLY`，進入點為 `0x06000005`
- metadata 版本是 `v4.0.30319`，存在五個 stream，且 14 個表格的列數與列大小與 oracle 回報的相同
- .NET reader 在 246 項檢查中全部 246 項與 System.Reflection.Metadata 一致
- `Grade` 有參數 `score`、簽章 `static int32(int32)`、`MaxStack` 為 2 且有 45 個程式碼位元組的 fat header，以及一個沒有名稱的 `int32` 區域變數
- CIL 的 `switch` 目標、`br.s`、`ldsfld`、`call` 和 `ldstr` operand，透過表格與 heap 解析
- CIL opcode 表格涵蓋每個 runtime opcode，並且只多加了 `no.`
- keeps 表格回報參數名稱在四種容器中保留，在原生程式碼中則沒有，而型別在三種容器中保留
- demo 中的每一行 oracle 都讀作 n of n
- 互動式圖表內嵌了 `figure_data()` 產生的相同資料

## Production 的失敗模式 (Production Failure Modes)

| 失敗 | 薄弱的工具回報什麼 | 分析必須證明什麼 |
|---|---|---|
| 反編譯器比 interpreter 舊 | `Unsupported opcode`、輸出標記為不完整，或聲稱被混淆 | magic word、它所指名的 release，以及工具文件記載的上限 |
| 用錯誤的 interpreter 讀取 pyc | `marshal.loads` 丟出例外或回傳垃圾 | 我們的 unmarshaller 能從任何 3.9 到 3.15 的檔案讀取名稱、常數與大小，而指令來自對應的 `dis` |
| 特化的 opcode 被誤認為檔案內容 | 清單中的 `COMPARE_OP_INT` | pyc 持有未特化的形式，特化是在暖機之後於記憶體中發生 |
| 對預設 javac build 回報區域變數名稱 | 把 `var1`、`var2` 當作原始碼名稱呈現 | `LocalVariableTable` 只在有 `-g` 時才存在，兩種情況下的 50 個程式碼位元組都相同 |
| 分支目標偏差了指令長度 | 跳轉落在指令中間的 JVM 清單 | 相對於分支 opcode 的 offset，並對照 `javap` 檢查 |
| 略過 switch 的 padding | default 與 case 錯位的 `tableswitch` | 零到三個 padding 位元組，對齊到 4 的倍數的 pc，五個 case 是 36 位元組 |
| 假設 Wasm 名稱被 strip | 「沒有 symbol」 | Export 為每個匯出的函式命名，區域變數需要 `name` section，而 `wasm-strip` 只移除那個 section |
| 三個工具，三種拼寫 | `get_local` 對 `local.get`，位於 `0x8b` 對 `0x8e` 的函式 | 位元組相同，而且比較時要使用擷取時所用的清單 |
| 猜測表格的列大小 | 第一個表格之後的每一列都被誤讀 | 列大小來自 `HeapSizes`、列數與 coded index 寬度，等於 oracle 的 |
| 預期 release assembly 中有區域變數名稱 | 把 `V_0` 當作原始碼名稱呈現 | 區域變數的型別在 `StandAloneSig`，而名稱只在 `DebugType` none 沒有寫出的 PDB 中 |
| Reader 猜測未知的 opcode 或型別代碼 | 看似合理的 mnemonic | 帶有 offset 的 `FormatError`，而 oracle 比較在那裡停止 |
| 從另一個 build 擷取的 oracle | 測試在過期的文字上通過 | `build_fixtures.sh` 指名每個工具版本並寫出 `versions.txt` 和 `sha256.txt` |

## 營運規則 (Operational Rule)

先讀 header，並指名它指向的 producer。用拒絕它不認識之事物的 reader 讀取容器，並在同一個檔案上，對照該格式自己的工具檢查它的清單。然後，對反編譯器輸出中的每個名稱與型別，說明它是容器儲存的，還是工具推論的。

## 延伸閱讀 (Further Reading)

- [PEP 552: Deterministic pycs](https://peps.python.org/pep-0552/)
- [CPython `Include/internal/pycore_magic_number.h`](https://github.com/python/cpython/blob/main/Include/internal/pycore_magic_number.h)
- [Python `marshal` module documentation](https://docs.python.org/3/library/marshal.html)
- [CPython `Python/marshal.c`](https://github.com/python/cpython/blob/main/Python/marshal.c)
- [Python 3.14 `dis` module documentation](https://docs.python.org/3.14/library/dis.html)
- [PEP 659: Specializing Adaptive Interpreter](https://peps.python.org/pep-0659/)
- [PEP 626: Precise line numbers for debugging and other tools](https://peps.python.org/pep-0626/)
- [PEP 790: Python 3.15 Release Schedule](https://peps.python.org/pep-0790/)
- [uncompyle6 on PyPI](https://pypi.org/project/uncompyle6/), [decompyle3 on PyPI](https://pypi.org/project/decompyle3/), [xdis on PyPI](https://pypi.org/project/xdis/), [pycdc on GitHub](https://github.com/zrax/pycdc), [PyLingual on GitHub](https://github.com/syssec-utd/pylingual)
- [The Java Virtual Machine Specification, Java SE 25, Chapter 4: The class File Format](https://docs.oracle.com/javase/specs/jvms/se25/html/jvms-4.html)
- [The Java Virtual Machine Specification, Java SE 25, Chapter 6: The Java Virtual Machine Instruction Set](https://docs.oracle.com/javase/specs/jvms/se25/html/jvms-6.html)
- [javac manual, Java SE 25](https://docs.oracle.com/en/java/javase/25/docs/specs/man/javac.html) and [javap manual](https://docs.oracle.com/en/java/javase/25/docs/specs/man/javap.html)
- [CFR](https://www.benf.org/other/cfr/), [Procyon](https://github.com/mstrobel/procyon), [Vineflower](https://github.com/Vineflower/vineflower), [Fernflower in intellij-community](https://github.com/JetBrains/intellij-community/tree/master/plugins/java-decompiler/engine), [JADX](https://github.com/skylot/jadx)
- [Harrand, Soto-Valero, Monperrus, Baudry, "Java Decompiler Diversity and its Application to Meta-decompilation" (JSS 2020)](https://arxiv.org/abs/2005.11315)
- [ECMA-335, Common Language Infrastructure, 6th edition (June 2012)](https://ecma-international.org/publications-and-standards/standards/ecma-335/)
- [PE Format, Microsoft Learn](https://learn.microsoft.com/en-us/windows/win32/debug/pe-format)
- [System.Reflection.Metadata.Ecma335.TableIndex](https://learn.microsoft.com/en-us/dotnet/api/system.reflection.metadata.ecma335.tableindex)
- [ILSpy on GitHub](https://github.com/icsharpcode/ILSpy), [ilspycmd on NuGet](https://www.nuget.org/packages/ilspycmd), [dnSpyEx on GitHub](https://github.com/dnSpyEx/dnSpy)
- [WebAssembly Core Specification: Binary Format, Modules](https://webassembly.github.io/spec/core/binary/modules.html) and [Values](https://webassembly.github.io/spec/core/binary/values.html)
- [WebAssembly Core Specification: Custom Sections (name section)](https://webassembly.github.io/spec/core/appendix/custom.html)
- [WebAssembly tool-conventions: Debugging](https://github.com/WebAssembly/tool-conventions/blob/main/Debugging.md)
- [wabt 1.0.42 release notes](https://github.com/WebAssembly/wabt/releases/tag/1.0.42), [wasm-tools](https://github.com/bytecodealliance/wasm-tools), [Binaryen](https://github.com/WebAssembly/binaryen), [ghidra-wasm-plugin](https://github.com/nneonneo/ghidra-wasm-plugin)
- [radare2 on GitHub](https://github.com/radareorg/radare2), version 6.2.4 built from source for this lesson
