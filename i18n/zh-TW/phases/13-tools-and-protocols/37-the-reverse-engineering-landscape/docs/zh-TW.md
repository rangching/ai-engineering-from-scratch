# 逆向工程全景：歷史、工具家族，以及反編譯器如何建構 (The Reverse Engineering Landscape: History, Tool Families, and How a Decompiler Is Built)

> 五個工具讀同一個 object，給出四個位址、三種 block 數量，以及對同一個 switch 的兩種答案，而這堂課建構的 harness 會顯示哪個是哪個。

**Type:** Build
**Languages:** Python, C, x86-64 assembly, Java (one Ghidra script)
**Prerequisites:** Phase 13 · 32 (ELF from scratch), Phase 13 · 33 (disassembly and control flow), Phase 13 · 34 (lifting to pseudocode)
**Time:** ~150 分鐘

## 學習目標 (Learning Objectives)

- 把這個領域的主要工具放在 1973 年到 2026 年的標有日期的時間線上，並指出每個工具所屬的家族。
- 說出反編譯器的八個階段，以及 Ghidra、Binary Ninja、Hex-Rays、radare2、rizin 與 angr 對每個階段使用的術語。
- 把一條指令 lift 成 ESIL、RzIL、P-code 與 VEX，並說明其中哪些會積極（eagerly）寫入 flag，哪些儲存 thunk。
- 把五個工具的位址化簡為單一個 `.text` offset，並在一個函式上證明這個化簡。
- 建構比較 harness，讀取擷取到的工具輸出，並回報指令起點、block、edge 與函式起點上的一致程度。
- 從擷取的證據中指出，每個工具在哪裡捏造程式碼、漏掉程式碼，或停在 indirect jump。

## 問題 (The Problem)

你請 agent 摘要三個逆向工程工具對 `ledger-x86_64-O1.o` 中 `grade` 的說法。Agent 執行每個工具、讀取三份清單，並回報該函式在一個工具中位於 `0x08000050`，在另一個中位於 `0x400010`，在第三個中位於 `0x100010`。它的結論是這些工具對函式在哪裡有分歧。

這些工具其實一致。Object 是可重定位的，沒有載入位址，所以每個工具自己選一個。減去每個工具給 `.text` 的位址，每個答案都是 `0x10`。Agent 比較原始位址，是因為沒有東西告訴它，這個位址在各個工具中代表什麼。

之後同一個 agent 讀取 `-O0` build 中的 `grade`，並回報四個工具分別有 3、8、8 和 9 個 basic block。四個數字都是真的。一個工具停在 `jmp rax`。兩個工具透過 relocation 讀取 jump table。一個工具為了放置標籤，在跳轉處切開了 block。沒有原因支持的 block 數量，只是一個沒有主張的數字。

這堂課是你在信任任何那些數字之前需要的地圖。課程 32 到 36 建構了一個 decoder、一個 lifter、一個證據紀錄和一個重建。這條路線的後半部分，把它們放在人們使用的工具旁邊。要做到這點，你需要這個領域的歷史、工具所屬的家族、每個反編譯器都會執行的管線，以及一個比較工具對你自己編譯的 object 所說內容的 harness。

```figure
re-37-field-map
```

## 五十三年的有日期步驟 (Fifty-Three Years in Dated Steps)

這個領域比它大多數的工具都要古老。下面每個日期在延伸閱讀中都有第一手來源。

### 1973 到 1991：先有 debugger，然後是 disassembler

最早的反編譯論文來自 1973 年。史丹佛的 Hollander 與普渡的 Housel 寫了它們，而兩者今天都是透過引用而為人所知。那個時期實用的工具是 debugger。Maranzano 與 Bourne 在 1977 年於 Bell Labs 寫了 `adb` 教學，而 debugger 在 1979 年隨 UNIX V7 出貨。Tim Paterson 的 `DEBUG` 帶有 8086 disassembler，於 1981 年 8 月隨 PC-DOS 1.00 出貨。Richard Stallman 的 GDB 在 1986 年進入散布。SoftICE 在 1987 年把 kernel debugger 放到 DOS 之下。

批次 disassembler 接著出現。V Communications 在 1988 年的 Sourcer，對 DOS 程式執行多趟處理，並寫出帶註解的清單。GNU binutils 以沒有編號的 snapshot 在 1988 年開始，而 `objdump` 帶有 1990 年的版權。Ilfak Guilfanov 在 1991 年 1 月寫下 IDA 的第一行，並於 1991 年 5 月 20 日以 shareware 形式發布 IDA 0.1。

### 1991 到 2009：互動式 disassembler 與最早的反編譯器

Cristina Cifuentes 在 1994 年 7 月於 QUT 完成〈Reverse Compilation Techniques〉。她的反編譯器 `dcc` 把 i80286 DOS 程式轉成 C，而她 1995 年與 Gough 合寫的論文，設定了之後每個反編譯器都沿用的階段。REC 在 1997 年跟進，Boomerang 在 2002 年出現，這是第一個開源反編譯器，採 BSD 授權。

IDA 在 1996 年於 DataRescue 成為商業產品，1999 年獲得 plugin SDK，並移到 2005 年成立的 Hex-Rays。Hex-Rays Decompiler 1.0 在 2007 年 9 月 17 日出貨。同一年，Mike Van Emmerik 提交了〈Static Single Assignment for Decompilation〉，這篇論文把 SSA 放在反編譯器資料流的中心。Halvar Flake 的〈Structural Comparison of Executable Objects〉（2004）開啟了 diff 家族，而 BinDiff 在 2004 年 8 月開賣。

radare 在 2006 年 2 月以具備 64 位元 offset 的鑑識 hex editor 之姿開始。radare2 是採用 LGPL-3.0 的模組化重寫，於 2009 年 3 月 7 日首次發布。

### 2009 到 2019：框架與中介表示法

LLDB 在 2010 年 6 月 8 日被宣布為 LLVM 子專案。BAP，也就是帶有 BIL 中介語言的 Binary Analysis Platform，出現在 CAV 2011。Google 在 2011 年 3 月 1 日收購了 BinDiff 的製作者 zynamics。Hopper 大約在 2011 年底出現在 Mac App Store。Matt Godbolt 在 2012 年開始做 Compiler Explorer，讓 C++ 程式設計師可以看到編譯器產生的組合語言。

angr 在 2013 年 8 月收到第一行程式碼。Phoenix 論文（Schwartz、Lee、Woo、Brumley，USENIX Security 2013）展示了保留語意的結構化，在 8,676 個函式上有 40 個 `goto` statement。DREAM（Yakdan 等人，NDSS 2015）用與模式無關的結構化移除了 `goto` statement。Vector 35 在 2015 年 1 月 14 日成立，而 Binary Ninja 在 2016 年 7 月 23 日開賣，帶有它的 Low Level IL 與 Python API。angr 在 2015 年公開發布，並在 2016 年 8 月的 DARPA Cyber Grand Challenge 中名列第三。BinDiff 在 2016 年 3 月變成免費。Cutter 1.0 在 2017 年 12 月為 radare2 帶來 Qt GUI。2018 年，Guilfanov 在 Recon 發表〈Decompiler internals: microcode〉，這是對 Hex-Rays IR 的第一次公開描述。

### 2019 到 2026：開源反編譯器與研究的轉向

Ghidra 9.0 由 NSA 在 2019 年 3 月 5 日的 RSA Conference 上以 Apache-2.0 發布，一個月後在 GitHub 上提供完整原始碼。angr 8.19.2.4 在同一年加入初步的反編譯器。Binary Ninja 2.0 在 2020 年 5 月 20 日出貨它的 High Level IL 反編譯器。Rizin 在 2020 年 12 月 8 日被宣布為 radare2 的 fork，而 Cutter 團隊轉移到它。IDA Freeware 7.6 在 2021 年 5 月加入雲端 x64 反編譯器。

BinDiff 8 在 2023 年 9 月 25 日以 Apache-2.0 開源。SAILR（Basque 等人，USENIX Security 2024）成為 angr 預設的 structurer，並主張只應該移除虛假的 `goto` statement，因為 Linux 6.1 的原始碼中就有 3,754 個。Binary Ninja 4.0 在 2024 年 2 月加入 Free 版，而 IDA 9.0 加入了供 headless 使用的 `idalib`。學習式反編譯器的論文從 SLaDe（2023）經 LLM4Decompile（2024）到 DecompileBench（2025），以及 2026 年關於 agent 輔助可讀性，與反編譯器重新編譯更多卻保留更少的兩項研究。

這堂課所執行的版本，是 2026 年 10 月 14 日的最新版本：Ghidra 12.1.4（2026 年 9 月 21 日）、radare2 6.2.4（2026 年 10 月 5 日）、rizin 0.9.1（2026 年 6 月 29 日）以及 angr 10.0.3（2026 年 10 月 10 日）。Binary Ninja 6.0「Krypton」（2026 年 9 月 3 日）與 IDA 9.4（2026 年 7 月 13 日）是目前的閉源工具。這堂課沒有執行它們。

## 工具家族 (The Tool Families)

工具屬於哪個家族，取決於它產生什麼。家族在同一個程式中會重疊，而重疊正是大部分混淆開始的地方。

### Disassembler

Linear sweep 從程式碼 section 的第一個位元組解碼到最後一個。GNU `objdump -d` 和 `llvm-objdump -d` 這樣運作，而兩者都不建構 control-flow graph。GNU `objdump` 把 symbol 視為指令邊界，並在它們之間 sweep。binutils 2.47（2026 年 7 月 26 日）採用 GPL-3.0-or-later。LLVM 23.1.3 採用 Apache-2.0 搭配 LLVM 例外條款。Apple 的 Command Line Tools 隨附 `llvm-objdump`，名為「Apple LLVM version 17.0.0」，這就是這堂課擷取的副本。

互動式 disassembler 從已知的進入點執行 recursive descent，為函式起點加入啟發式，並維護資料庫。IDA Pro（processor module、autoanalysis queue、FLIRT signature）、Ghidra（`Disassembler` 加上 Auto Analysis analyzer）、Binary Ninja（recursive descent，然後對未知區域做 linear sweep）、Hopper、radare2 與 rizin（`aa`、`aaa`、`af`，並以 Cutter 和 iaito 作為 GUI），全都屬於這個家族。Andriesse 等人（USENIX Security 2016）在編譯器產生的 binary 上量測了其中九個。指令的準確率接近 100%，而困難的基本操作是函式起點，其 false negative 比率達 20% 或更多。

### 反編譯器

商業反編譯器：Hex-Rays Decompiler（從 x86 到 RISC-V 的十二個目標家族）、Binary Ninja 的 HLIL 及其 Pseudo C 轉譯，以及 Hopper 的 pseudo-code 視圖。開源反編譯器：Ghidra 反編譯器，這是採用 Apache-2.0 的 C++ 引擎，r2ghidra 與 rz-ghidra 把它重用為 plugin，pypcode 則把它重用為 lifter；以 AIL 為基礎、搭配 SAILR 結構化、採用 BSD-2-Clause 的 angr 反編譯器；以 JavaScript 寫成的 r2dec 與 jsdec；以及作為歷史的 Boomerang 和 `dcc`。radare2 也有內建的 `pdc`，「類 C 語法的 pseudo disassembler 輸出」。rizin 沒有內建的反編譯器。指令 `pdc` 在 rizin 0.9.1 中不存在，而 `pdg` 需要 rz-ghidra plugin。

### Debugger

GDB（1986）搭配 GEF 與 pwndbg 前端、LLDB（2010）、搭配 Time Travel Debugging（2017）的 WinDbg、OllyDbg（2014 年停止更新）、x64dbg（GPL-3.0，2014）、帶有 Python connector 的 Ghidra Debugger、Binary Ninja debugger，以及 radare2 與 rizin 內建的 debugger。課程 42 在你自己的 build 上使用其中兩個。

### 二進位分析框架

angr（VEX IR、symbolic execution、`CFGFast` 與 `CFGEmulated`、反編譯器）、BAP（BIL、OCaml）、Dyninst、Jakstab 和 ByteWeight，是 Andriesse 所指名的框架。Ghidra 透過 PyGhidra 成為函式庫，Binary Ninja 透過 `import binaryninja`，IDA 透過 `idalib`。

### 格式與引擎函式庫

LIEF 1.0.0（2026 年 7 月 12 日，Apache-2.0）解析 ELF、PE、Mach-O、COFF 與 Android 格式。CLE，「CLE Loads Everything」，是 angr 的 loader。Capstone 5.0.9（2026 年 5 月 28 日，BSD-3-Clause）負責解碼；Keystone 0.9.2（GPL-2.0）負責組譯；Unicorn 2.1.4（GPL-2.0）負責模擬。radare2 6.2.4 內含 Capstone 5，而 rizin 0.9.1 內含 Capstone、Zydis 和 tree-sitter。

### Diff 工具與編譯器端的檢視器

BinDiff 8（Apache-2.0）搭配其適用於 IDA Pro、Binary Ninja 與 Ghidra 的 BinExport 12 plugin；作為 IDA plugin 的 Diaphora 3.4.3（2026 年 10 月 11 日，AGPL-3.0）；`radiff2` 和 `rz-diff`。Compiler Explorer（BSD-2-Clause）顯示編譯器為原始碼片段送出的組合語言，這是這份清單中其他一切的正向。課程 44 使用 diff 家族。

### 版本、授權，以及這堂課執行了什麼

| 工具 | 2026-10-14 讀到的版本 | 授權 | Headless 進入點 | 此處是否執行 |
|---|---|---|---|---|
| Ghidra | 12.1.4 | Apache-2.0 | `analyzeHeadless`、`pyghidra.start()` | 是，在全部四個 object 上 |
| radare2 | 6.2.4 | LGPL-3.0 | `r2 -qc '<cmds>' file`、`rabin2` | 是 |
| rizin | 0.9.1 | LGPL-3.0 | `rizin -qc '<cmds>' file`、`rz-bin` | 是 |
| angr | 10.0.3 | BSD-2-Clause | Python 中的 `angr.Project` | 是 |
| llvm-objdump | Apple LLVM 17.0.0 | Apache-2.0 搭配 LLVM 例外條款 | `llvm-objdump -d -r` | 是 |
| Binary Ninja | 6.0 "Krypton" | 商業；Free 版供非商業使用 | 搭配 Commercial、Ultimate 或 headless 授權的 `binaryninja.load()` | 否，僅查閱文件 |
| IDA Pro | 9.4 | 商業訂閱；IDA Free 供非商業的 x86 與 x86-64 使用 | `idat`、`ida -A -S`、`idalib` | 否，僅查閱文件 |
| Hopper | 6.6.0 | 商業，永久授權 | 沒有文件記載 | 否，僅查閱文件 |
| Cutter | 2.5.0 | GPL-3.0 | 位於 rizin 之上的 GUI | 否 |

Binary Ninja 的 Free 版只反編譯 x86、x86-64、ARMv7 與 AArch64，而且沒有 API。IDA Free 沒有 IDAPython、沒有 SDK，也沒有 `idalib`，而且它的反編譯器在雲端執行。這堂課中關於那兩個工具的每項陳述，都來自它們的文件與發布說明，而圖表把它們標記為未執行。

## 反編譯器管線，逐階段說明 (The Decompiler Pipeline, Stage by Stage)

Cifuentes 在 1994 年為這些階段命名：語法分析、語意分析、中間碼產生、control-flow graph 產生、資料流分析、控制流程分析，以及程式碼產生。她把它們分成與機器相關的前端、與機器和語言無關的中間，以及與語言相關的後端。之後的每個反編譯器都保留那個形狀，並重新命名各個部分。下表給出這堂課使用的八個階段，以及每個工具給它們的名稱。圖表顯示相同的表格，並可選擇工具欄。

| 階段 | Ghidra | Binary Ninja | Hex-Rays / IDA | radare2 | rizin | angr |
|---|---|---|---|---|---|---|
| Loader | `ElfLoader`、`MachoLoader`、`PeLoader`；`.ldefs` 選擇語言 | `BinaryView`、`BinaryViewType` | loader module (`LDSC`) | RBin (`rabin2`) | RzBin (`rz-bin`) | CLE (`cle.Loader`、`main_object`) |
| Disassembly | `Disassembler`；SLEIGH `.slaspec` 編譯成 `.sla`；analyzer「Disassemble Entry Points」、「Function Start Search」 | architecture plugin；Recursive Descent，然後是 linear sweep 階段 | processor module (`ana.cpp`、`emu.cpp`、`out.cpp`)；autoanalysis queue；FLIRT | 通往 `RAnalOp` 的 RArch plugin；`aa`、`aaa`、`af` | RzAnalysis plugin；`aa`、`aaa`、`af` | Capstone block；`CFGFast` 函式起點搜尋 |
| CFG 還原 | 「Generate Basic Blocks and the CFG」；`BlockGraph`；「Decompiler Switch Analysis」 | `Function`、`BasicBlock`；「Control Flow Graph (CFG) Analysis」階段 | `mbl_graph_t`，於 `MMAT_LOCOPT` 就緒 | `afb`、`agf`、`aab` | `afb`、`agf`、`aab` (Nucleus) | `CFGFast`、`CFGEmulated`；`kb.functions`；IRSB |
| Lifting | 「Generate Raw P-code」；`ram`、`register`、`const`、`unique` 中的 Varnode | Lifted IL、LLIL、Mapped MLIL、MLIL | microcode (`minsn_t`、`mop_t`)；`MMAT_GENERATED` | ESIL (`asm.esil`、`ae`) | RzIL (`aoi`、`aez`) | VEX (`pyvex.lift`、IRSB)，然後是 AIL |
| SSA | `Heritage`、`ActionHeritage`；`MULTIEQUAL`、`INDIRECT` | LLIL SSA、MLIL SSA、HLIL SSA；`HLIL_VAR_PHI` | 沒有命名的 SSA；`MMAT_GLBOPT1` 到 `3` 中的 use-def 資料流 | 無；ESIL 是為模擬而建 | 無；RzIL 是為模擬而建 | `_stage_transform_to_ssa_level0` 與 `level1`；`SeqNodeDephication` |
| 資料流與型別 | `ActionDeadCode`、`ActionInferTypes`、`Rule` pool；來自 `.cspec` 的 prototype | MLIL 常數傳播、變數與型別；`analyzeReturns` | 「Transform microcode」；`MMAT_CALLS`、`MMAT_GLBOPT`、`MMAT_LVARS`；TIL | `afv`、`afc`、`aaft`；ESIL 模擬 | `afv`、`afc`、`aft` | `CallingConventionAnalysis`、`VariableRecoveryFast`、`Typehoon` |
| 結構化 | `CollapseStructure`；`BlockIf`、`BlockWhileDo`、`BlockSwitch`、`BlockGoto` | `HLIL_IF`、`HLIL_WHILE`、`HLIL_SWITCH`；位於 `root` 的 AST | 「Generate ctree」；結構分析；goto 區域 | `pdc`；`pdd` (r2dec)；`pdg` (r2ghidra) | `pdg` (rz-ghidra)；jsdec | `RegionIdentifier`；搭配 `SAILRStructurer` 的 `RecursiveStructurer` |
| 輸出 | 「Emit Final C Tokens」；`PrintC`、`ActionNameVars`、`ActionSetCasts` | HLIL 轉譯；Pseudo C、Pseudo Rust、Pseudo Python | 「Beautify ctree; Print ctree」；`cfunc_t` | `pdc`、`pdd`、`pdg` 的文字 | `pdg` 文字 | `CStructuredCodeGenerator`；`codegen.text` |

### Loader

Loader 解析容器、對應 section，並讀取 symbol 與 relocation。在可重定位的 object 中，沒有任何東西有位址，所以 loader 自己挑一個。radare2 與 rizin 把檔案對應在 `0x08000000`，這讓 `.text` 位於 `0x08000040`，因為 section 從檔案 offset `0x40` 開始。CLE 把 `.text` 放在 `0x400000`。Ghidra 的 `ElfLoader` 把它放在 `0x100000`。`llvm-objdump` 印出 section 的 offset，根本沒有 base。

Loader 在 relocation 上也不同。Ghidra 在匯入時處理它們，而 CLE 套用它們。radare2 在開啟 object 時印出提示：relocation 沒有被套用，而 `bin.relocs.apply=true` 會改變這點。Loader 對 relocation 做了什麼，決定了之後每個階段能看到什麼，你會在下面的 `-O0` jump table 中遇到這個效果。

### Disassembly

課程 33 建構了兩種方法。表格中除了 `llvm-objdump` 之外的每個工具，都執行 recursive descent，然後再加上函式起點的搜尋。Binary Ninja 的文件依序列出階段：Recursive Descent、Call Target Analysis、作為「對未知區域的 linear disassembly」的 CFG Analysis，以及 Tail Call Analysis。Ghidra 依優先順序執行它的 analyzer，「Disassemble Entry Points」與「Function Start Search」在前。IDA 維護 autoanalysis queue，每個都有優先順序，並為函式庫程式碼提供 FLIRT signature。angr 的 `CFGFast` 文件記載了四種函式起點來源：symbol、prologue 掃描、call 目標，以及對剩餘位元組的完整掃描。最後一種來源就是 angr 在 padding 上捏造函式的地方。

### CFG 還原

每個工具都像課程 33 一樣切出 block，只有一個你會量測到的差異。angr 在 `call` 結束一個 block，因為它的 IRSB 是單一進入點、多個離開點的 block，而 call 是一個離開點。Ghidra、radare2 與 rizin 則穿過 call 繼續。在呼叫 `grade` 的 `-O0` 的 `grade_label` 上，angr 回報 2 個 block，另外三個回報 1 個。指令是相同的。

### Lifting

Binary Ninja 將 lifting 定義為「從較低階的語言，轉譯成被視為較高階的中介語言」。這些 IR 的用途不同。P-code 與 VEX 是為分析與反編譯而建。ESIL 與 RzIL 是為模擬而建，兩者都沒有 SSA 階段。Hex-Rays microcode 一開始是有 72 個 opcode 的類 RISC 程式碼，並通過從 `MMAT_GENERATED` 到 `MMAT_LVARS` 的成熟度層級。Binary Ninja 先 lift 到 Lifted IL，然後在 LLIL 中把 flag 折疊成條件，而 MLIL「把暫存器與記憶體存取轉譯成變數」。下一節把其中四種 IR，並排放在同一條指令上。

### SSA

Van Emmerik 的論文列出 SSA 為反編譯器帶來的好處：表達式傳播、不假設 ABI 的參數與回傳值辨識、保留分析、作為稀疏資料流問題的型別分析，以及 indirect jump 與 call 的分析。Ghidra 把它的 SSA 建構稱為 `Heritage`，並加入兩個只存在於圖中的操作：phi 用的 `MULTIEQUAL` 與副作用用的 `INDIRECT`。Binary Ninja 在每個 IL 層級都公開 SSA 形式。angr 在 `Clinic` 內有兩個 SSA 階段。Hex-Rays 沒有命名的 SSA 形式，而是在它的全域最佳化 pass 中做 use-def 資料流。ESIL 與 RzIL 則沒有。

### 資料流與型別

Ghidra 主要的簡化迴圈執行「Eliminate Dead Code」、「Propagate Local Types」和「Perform Term Rewriting」，直到沒有任何改變為止，其中改寫由 `Rule` 物件池完成。Hex-Rays 轉換 microcode、在 `MMAT_CALLS` 偵測 call 引數，並在 `MMAT_LVARS` 配置區域變數。angr 執行 `CallingConventionAnalysis`、`VariableRecoveryFast` 和 `Typehoon`，後者實作了 Noonan、Loginov 與 Cok 的 polymorphic type inference。課程 41 談的是這個階段。

### 結構化

Ghidra 的 `CollapseStructure` 搜尋符合某種結構的子圖、把它們折疊成一個節點，並在卡住時移除一條 edge 並標記為未結構化。Hex-Rays 把 CFG 化簡為 `if`、`while`、`do` 和 `switch`，並把其餘部分留作 goto 區域。angr 辨識單一進入點、單一離開點的區域，並預設以 `SAILRStructurer` 結構化它們，另有 `PhoenixStructurer` 與 `DreamStructurer` 作為替代。課程 34 手寫了一個 structurer。

### 輸出

Ghidra 在 `ActionNameVars` 和 `ActionSetCasts` 之後，透過 `PrintC` 搭配 Oppen pretty-printer 送出 C token。Hex-Rays 美化並印出 ctree。Binary Ninja 把 HLIL 轉譯為 Pseudo C、Pseudo Rust 或 Pseudo Python。angr 回傳 `codegen.text`。radare2 印出 `pdc`、`pdd` 或 `pdg`。

## 四種 IR 中的一條指令 (One Instruction in Four IRs)

以 `48 83 ec 10`（`subq $0x10, %rsp`）為例。用課程 33 的規則在紙上解碼它。

1. `48` 是 `W = 1` 的 REX。Operand 是 64 位元。
2. `83` 是帶 8 位元 immediate 的 group-1 算術 opcode。ModRM 的 reg 欄位挑選運算。
3. `ec` 是 ModRM `11 101 100`。`mod = 11` 是暫存器形式。`reg = 101` 是 `/5`，也就是 `sub`。`r/m = 100` 是 `%rsp`。
4. `10` 是 immediate，16。

四個 lifter 都拿到了這四個位元組。Fixture 持有確切的指令與完整的文字。

radare2 6.2.4 的 ESIL 是一個以逗號分隔、含 30 個 token 的字串：

```text
16,0x8000000000000000,-,!,16,rsp,-=,63,$o,^,of,:=,63,$s,sf,:=,$z,zf,:=,$p,pf,:=,64,$b,cf,:=,4,$b,af,:=
```

它是 stack machine。`16,rsp,-=` 原地相減。接著每個 flag 都用內部運算子寫入：`$o` overflow、`$s` sign、`$z` zero、`$p` parity、`$b` borrow。

rizin 0.9.1 的 RzIL 是帶有 14 個 `set` 效果的 S-expression：

```text
(seq (set op1 (var rsp)) (set op2 (bv 64 0x10)) (set sub (- (var op1) (var op2))) (set rsp (var sub))
     (set _result (var sub)) (set pf (! (lsb (let _val (cast 8 false (var _result)) ...))))
     (set zf (is_zero (var _result))) (set sf (msb (var _result))) ... (set cf (|| ...)) (set of (|| ...)) (set af (|| ...)))
```

每個 flag 都是對 bitvector 的布林表達式。光是 parity flag 就是四層的 `let`。

來自 SLEIGH 規格 `x86:LE:64:default`、經由 pypcode 4.0.1 的 P-code，在 `IMARK` 之後是九個操作：

```text
CF = RSP < 0x10
OF = sborrow(RSP, 0x10)
RSP = RSP - 0x10
SF = RSP s< 0x0
ZF = RSP == 0x0
unique[58300:8] = RSP & 0xff
unique[58400:1] = popcount(unique[58300:8])
unique[58500:1] = unique[58400:1] & 0x1
PF = unique[58500:1] == 0x0
```

這是暫存器之間的轉移，暫存變數放在 `unique` 空間。Flag 在指令處被積極計算，而 `CF` 和 `OF` 在減法覆寫 `RSP` 之前就先計算好了。

來自 pyvex 10.0.3 的 VEX，在 `IMark` 之後是七個 statement：

```text
t2 = GET:I64(offset=48)
t0 = Sub64(t2,0x0000000000000010)
PUT(offset=144) = 0x0000000000000008
PUT(offset=152) = t2
PUT(offset=160) = 0x0000000000000010
PUT(offset=168) = 0x0000000000000000
PUT(offset=48) = t0
```

沒有任何 flag 出現。VEX 以暫存器在 guest 狀態中的 offset 為其命名，而 offset 48 是 `rsp`。Offset 144 到 168 是 `cc_op`、`cc_dep1`、`cc_dep2` 和 `cc_ndep`。Lifter 儲存運算代碼 8（也就是 64 位元減法）以及兩個 operand。之後讀取 flag 的分支，會從 thunk 計算它。這就是為什麼 harness 報告中 `pcode.writesFlags` 為 true 而 `vex.computesFlags` 為 false，也是為什麼 VEX 是四者中最短的。

同樣的四個 lifter 處理 `48 63 c7`（`movslq %edi, %rax`，`-O1` 的 `grade` 的第一條指令），分別給出 5 個 ESIL token（`32,edi,~,rax,=`）、1 個 RzIL `set`、1 個 P-code 操作（`RAX = sext(EDI)`），以及 4 個 VEX statement。沒有任何 lifter 寫入 flag，因為該指令沒有 flag。Ghidra 自己透過 headless 腳本擷取的該指令清單，把相同的操作印成 `(register, 0x0, 8) INT_SEXT (register, 0x38, 4)`：暫存器 offset 0 是 `RAX`，offset `0x38` 是 `RDI`，而大小是 8 和 4 位元組。

`grade` 的邊界檢查中的 `83 f9 04`（`cmpl $0x4, %ecx`），給出 30 個 ESIL token、13 個 RzIL `set` 效果、10 個 P-code 操作，以及 7 個 VEX statement，其 `cc_op` 為 7，即 32 位元減法。沒有任何東西寫入 `%ecx`。你會在 VEX 中遇到的兩種減法寬度是 32 位元的 7 和 64 位元的 8，而你建構的 translator 帶有那張表。

## 一個函式的四個位址基底 (Four Address Bases for One Function)

在 harness 能比較任何東西之前，它必須先與自己就函式的位置達成一致。為 `ledger-x86_64-O1.o` 中的 `grade` 手動做一次算術。

| 工具 | `.text` 的位址 | `grade` 的位址 | Offset |
|---|---|---|---|
| `llvm-objdump` | `0x0` | `0x10` | `0x10 - 0x0 = 0x10` |
| radare2、rizin | `0x08000040` | `0x08000050` | `0x08000050 - 0x08000040 = 0x10` |
| angr | `0x400000` | `0x400010` | `0x400010 - 0x400000 = 0x10` |
| Ghidra | `0x100000` | `0x100010` | `0x100010 - 0x100000 = 0x10` |

Symbol table 說 `grade` 位於 `.text + 0x10`，大小為 46。一旦移除 base，每個工具都一致。Harness 從每份擷取讀取 base：radare2 與 rizin 中的 `iSj`、angr 擷取中的 `.text` section 項目、Ghidra 擷取中的 `.text` 記憶體區塊，以及 sweep 的零。測試 `test_every_tool_places_grade_at_offset_0x10_from_its_own_base` 固定了這項事實。

## 工具在哪裡一致，在哪裡分歧 (Where the Tools Agree and Where They Split)

`code/main.py` 中的 harness 讀取擷取、化簡每個位址，並進行比較。這些是它回報的數字，而測試斷言了每一個。

### 一致的案例：-O1 的 ledger

四個函式，sweep 中有 56 條指令，其中 54 條在 symbol 內，2 條在 padding。每個工具在每個函式中都回報相同的指令起點：9、14、13 和 18。四個圖建構器回報 3、3、5 和 4 個 block，以及 2、3、5 和 4 條 edge，而且 edge 集合在 radare2、rizin、angr 與 Ghidra 之間逐對相同。`total` 的五個 block 和五條 edge，就是課程 33 手繪的那些。

唯一的差異在 padding。函式之間的 3 個位元組（`grade` 之後的 `66 90`、`total` 之後的 `90`）被 sweep 解碼（它沒有理由停下來），也被 angr 解碼（它建立了沒有 symbol 的 `sub_40003e` 和 `sub_40005f`）。radare2、rizin 和 Ghidra 不認領其中任何一個。

### 向量化的案例：-O2 的 ledger

`total` 在四個圖建構器的每一個中，都是 10 個 block、13 條 edge 的 41 條指令。課程 33 的 SSE2 迴圈與純量餘數迴圈，在每個地方都以相同的方式切開。angr 再次在 padding 上加入兩個函式。

### Jump table：-O0 的 grade

工具在這裡分歧了，而這個分歧就是這堂課。

| 工具 | Block | Edge | 擷取說了什麼 |
|---|---|---|---|
| radare2 | 3 | 2 | `0x62` 的 block 沒有後繼者；位元組 `0x76` 到 `0x9a` 從未到達；缺少 8 個指令起點 |
| rizin | 8 | 11 | `0x62` 的 block 帶有一個 5 個 case、4 個不同目標的 `switch_op` |
| angr | 8 | 7 | case block 存在，而 `0x62` 的轉移去往無法解析的目標節點 |
| Ghidra | 9 | 12 | `0x74` 的 `switchD` 標籤切開了 block；四個 `COMPUTED_JUMP` flow 去往 `0x76`、`0x7f`、`0x88`、`0x91` |

四者在到達相同的 33 條指令時都解碼出同樣的結果。radare2 停在 `jmp rax`，留下 36 個位元組沒有被認領，而 harness 把這個範圍命名為漏掉的程式碼。rizin 與 Ghidra 找到了課程 33 從 `.rela.rodata` 的五個 relocation 讀到的四個不同目標。angr 在 `grade` 內列出 case block，但從跳轉 block 到它們仍然沒有 edge；它的擷取記錄 `0x62` 的轉移通往無法解析的目標節點，而它的反編譯器印出了計算式 `goto`。對於相同的控制流程，Ghidra 比 rizin 多一個 block 和一條 edge，因為它的 switch 分析在 `jmp rax` 上放了一個標籤，而 block 模型會在標籤處開始新的 block。

課程 33 的圖有 8 個 block 與 11 條 edge。rizin 完全吻合。重點是要看出，9 和 11 以及 7 和 2，不是四次「找到多少程式碼」的量測。它們是四個答案，說明工具跟隨一個 indirect jump 跟到多遠，以及它選擇在哪裡切開。

### 程式碼中的資料：sweep-trap

`lookup` 是 24 位元組，其中 4 個表格位元組 `31 c0 90 48` 位於 `0x10`。Sweep 在那個範圍回報 8 個指令起點。Harness 列出它與多數的差異：位於 `0x10`、`0x12` 和 `0x13` 的多餘起點，以及缺少的 `0x14` 起點。那就是課程 33 中早了一個位元組的 `addq $0x1, %rax`。radare2、rizin 和 Ghidra 回報 6 個起點、2 個 block、1 條 edge，並把 `0x10` 到 `0x14` 留作未認領。angr 對 `lookup` 回報相同的 2 個 block 和 1 條 edge，然後在表格位元組上捏造 `sub_400010`：一個結束於 `0x14` 的 4 位元組 block 中的 2 條指令，而 `0x14` 正是 `lookup` 的第二個 block 開始的地方，所以位於 `0x13` 的 REX 位元組不屬於任何指令。Harness 對這個 object 的最後發現是，沒有任何圖建構器解碼 `0x10` 到 `0x14`，所以那些位元組位於函式內，是資料。

### 依工具列出的捏造與漏掉 (Invented and missed, by tool)

在四個 object 中，harness 指出 angr 捏造了 9 個沒有 symbol 的函式，每一個都位於 padding 或 trap 表格。它指出 radare2 漏掉一個範圍，即 `-O0` 的 `grade` 中 36 位元組的 case。它指出 9 個範圍是 linear sweep 解碼了、但沒有圖建構器視為程式碼：八個 padding 空隙加上 trap 表格。它指出沒有任何工具漏掉函式。

## grade 上的三個反編譯器 (Three Decompilers on grade)

三個被擷取的工具，對 `-O1` 的 `grade` 送出類 C 的文字。原始碼是帶有五個 case 的 `switch (score / 10)`，編譯成邊界檢查與回傳值表格。

Ghidra 12.1.4：

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

angr 10.0.3：

```c
extern unsigned int g_4000b4[4];

unsigned int grade(int a0)
{
    unsigned int v1;  // ecx
    unsigned int v2;  // eax

    v1 = (unsigned int)((long long)(a0 * 1717986919) >> 0x22) + (unsigned int)(a0 * 1717986919 >> 63) - 6;
    v2 = 0;
    if (v1 <= 4)
        v2 = g_4000b4[v1];
    return v2;
}
```

radare2 6.2.4，`pdc`：

```c
void sym.grade (int64_t arg1) {
    rax = edi
    rax = rax * 0x66666667
    rcx = rax
    rcx >>= 0x22
    rax >>>= 0x3f
    ecx += eax
    ecx += 0xfffffffa
    eax = 0
    v = ecx - 4
    if (((unsigned) v) <= 0) {
        eax = ecx
        rcx = rip
        eax = dword [rcx + rax*4]
    }
    return
}
```

對照管線閱讀這三者。Ghidra 的 term rewriting 把乘以 `0x66666667` 與 shift 34，折回成 `param_1 / 10`，這是課程 34 的慣用寫法。angr lift 了相同的位元組，並把算術留成 `a0 * 1717986919 >> 0x22`。radare2 的 `pdc` 是 ESIL 模擬加上 pseudo 語法：暫存器名稱、沒有變數、沒有型別，而邊界檢查在無號減法之後被印成 `<= 0`。Ghidra 透過 `DAT_001000b4` 為表格載入設定型別；angr 宣告了 `g_4000b4[4]`，四個元素，而原始碼有五個。Ghidra 和 angr 都沒有還原 case 的列舉，因為在 `-O1`，位元組裡已經沒有 switch 了。在存在 jump table 的 `-O0`，Ghidra 印出帶有 case 6 到 10 與 default 的 `switch(param_1 / 10)`，而 angr 印出計算式 `goto`。

rizin 沒有自己的反編譯器，而擷取記錄了 `pdc` 在那裡不存在。Binary Ninja 的 HLIL 與 Hex-Rays 的 ctree，是這堂課無法擷取的兩個輸出。

## 動手做 (Build It)

執行 demo：

```bash
cd phases/13-tools-and-protocols/37-the-reverse-engineering-landscape
python3 code/main.py
```

輸出是一份 JSON 報告。它一開始是擷取時記錄的工具版本，以及你的機器上存在的工具（可能全都沒有）。接著，對四個 object 中的每一個，它給出 symbol 範圍、padding 空隙、每個工具使用的 base、每個工具找到或捏造的函式起點，以及對每個函式：指令起點的一致程度、block 數量、edge 數量，以及每個工具缺少的 edge。它以每個工具的位元組帳目、工具自己的註記、以白話句子列出的發現清單，以及 IR 區段作結：translator 對三條指令的輸出，旁邊是在擷取的 lift 上量測到的數量。

預期的形狀：

```json
{
  "captureTools": {"llvm-objdump": "Apple LLVM version 17.0.0", "radare2": "radare2 6.2.4 ...", "rizin": "rizin 0.9.1 ...", "angr": "10.0.3", "ghidra": "12.1.4"},
  "localTools": {"radare2": "absent", "...": "..."},
  "fixtures": {
    "ledger-x86_64-O0.o": {
      "bases": {"llvm-objdump": "0x0", "radare2": "0x8000040", "angr": "0x400000", "ghidra": "0x100000", "...": "..."},
      "functions": {"grade": {"blocks": {"radare2": 3, "rizin": 8, "angr": 8, "ghidra": 9}, "edges": {"radare2": 2, "rizin": 11, "angr": 7, "ghidra": 12}, "...": "..."}},
      "findings": ["radare2 never reaches bytes 0x76-0x9a inside grade that the other graph builders decode", "..."]
    }
  },
  "ir": {"translator": [{"text": "subq $0x10, %rsp", "flags": {"pcodeLike": "eager", "vexLike": "thunk"}, "...": "..."}], "observed": {"...": "..."}}
}
```

依此順序閱讀 `code/main.py`：

1. `read_text_symbols()` 讀取 section header 與 `.text` 的 `STT_FUNC` symbol。Symbol table 是 harness 唯一的基準真相，而 `TextSection.gaps()` 從它推導出 padding。
2. `parse_objdump()`、`parse_r2_family()`、`parse_angr()` 和 `parse_ghidra()` 各自把一份擷取變成 `ToolView`：一個 base、以 offset 對應長度的指令，以及每個函式一個 `FunctionView`，內含 block、edge 與任何反編譯後的文字。每個 parser 只認得一個工具的詞彙，其他一概不知。
3. `majority()` 與 `ranges()` 是每次比較都使用的兩個輔助函式。
4. `function_report()` 將指令起點與圖建構器的多數比較，然後是 block 數量，然後是 edge 集合，並列出每個工具缺少什麼。
5. `byte_report()` 把每個位元組記為 symbol 內的程式碼、padding、未認領、捏造或漏掉，並指出沒有任何圖建構器解碼的 symbol 內位元組。
6. `findings()` 把這些變成句子，而 `compare()` 為每個 object 組裝一份報告。
7. `decode_small()`、`to_pcode_like()` 和 `to_vex_like()` 是 translator：三種暫存器形式的編碼，一種是積極風格，一種是 thunk 風格。
8. `observed_ir()` 量測擷取到的 lift，而 `figure_data()` 組裝圖表所顯示的時間線、階段表與證據。

還有另外三個指令。`python3 code/main.py compare FIXTURE.o` 印出單一 object 的報告。`python3 code/main.py ir 4883ec10` 印出 translator 對一條指令的兩種形式。`python3 code/main.py figure-data` 印出內嵌在圖表中的資料。

擷取是由 `code/fixtures/capture_tools.py` 一次性完成，它在 `captures.json` 中記錄每個指令與版本。Harness 與測試從不啟動任何工具。如果你有這些工具，請執行擷取腳本來更新檔案，並閱讀 fixture 的 README 以了解它所需的環境。

## 實際使用 (Use It)

依此順序，在新的 object 上使用 harness。

1. 自己建構 object，並保留編譯器指令。在開啟任何工具之前，先讀取它的 symbol table，讓你知道有多少函式，以及 padding 在哪裡。
2. 以 headless 模式擷取每個工具一次。把原始輸出與確切的指令和 object 存在一起。你無法重新閱讀的分歧，不算發現。
3. 檢查 base。挑一個函式，確認在減去各自的 `.text` 位址之後，每個工具都把它放在相同的 offset。如果有一個不是，就停下來查明那個工具載入了什麼。
4. 先比較指令起點。如果圖建構器在這裡就有分歧，其餘的比較談的是解碼，而不是分析。
5. 比較 block 與 edge，並在判斷某個數量之前，先閱讀每個工具的註記。`switch_op`、`COMPUTED_JUMP` 或無法解析的目標節點，能解釋大部分差異。
6. 為位元組做帳。工具認領的 padding 是啟發式在作用。工具留下的 symbol 位元組是停止的走訪。沒有圖建構器認領的 symbol 位元組是資料。
7. 寫下你沒有執行的工具。來自文件的描述有用，但與擷取是不同種類的證據。

## 互動實驗 (Interactive Lab)

### 實驗 A：一個函式，四個 base

從 `code` 目錄開啟 Python：

```bash
cd phases/13-tools-and-protocols/37-the-reverse-engineering-landscape/code
python3 -q
```

執行：

```python
from main import *
views = load_views("ledger-x86_64-O1.o")
{tool: hex(view.base) for tool, view in views.items()}
{tool: hex(view.functions["grade"].start + view.base) for tool, view in views.items()}
```

在執行第二行之前，先寫下四個原始位址，並對照這堂課的表格檢查。然後把 `grade` 換成 `grade_label`，並手動計算它在 Ghidra 中位於何處。

### 實驗 B：從四個角度看 jump table

```python
report = compare("ledger-x86_64-O0.o")
report["functions"]["grade"]["blocks"]
report["functions"]["grade"]["edgesMissingPerTool"]["radare2"]
report["notes"]["rizin"]
report["notes"]["ghidra"]
report["bytes"]["radare2"]["misses"]
```

計算 radare2 缺少的 edge，並把每一條對應到課程 33 的 `-O0` 清單中的一行。然後閱讀 Ghidra 的註記，並在 Ghidra 擷取中找出 `switchD` 標籤。

### 實驗 C：積極的 flag 對 thunk

```python
translate("48 83 ec 10")["pcodeLike"]
translate("48 83 ec 10")["vexLike"]
translate("83 f9 04")["vexLike"]
{row["name"]: (row["pcode"]["ops"], row["vex"]["statements"]) for row in observed_ir()["instructions"]}
```

在兩個輸出中找出 `cc_op` 常數，並說明為什麼一個是 8，另一個是 7。然後試試 `translate("48 01 c8")` 並閱讀 `TranslateError`。

### 實驗 D：五個工具下的 trap

```python
report = compare("sweep-trap.o")
report["functions"]["lookup"]["instructionStarts"]
report["bytes"]["dataInsideSymbols"]
report["functionStarts"]["angr"]
```

先在紙上，像 sweep 一樣解碼位元組 `31 c0 90 48 83 c0 01`，然後像 descent 在 `jmp` 之後那樣解碼。把你的兩份清單，對應到 harness 回報的多餘與缺少的起點。

## 實作練習 (Practice Lab)

新增第五份擷取，並擴充 harness。

需求：

- 挑一個這堂課沒有擷取、而你可以在自己機器上執行的工具：GNU `objdump`、搭配 Capstone 的 LIEF、手動匯出的 Binary Ninja Free session，或使用不同 analyzer 集合的 Ghidra。在四個 object 上擷取它的輸出，並在 `captures.json` 中記錄確切的指令與版本。
- 寫一個 parser，把擷取變成 `ToolView`，包含工具的 base、指令、函式、該工具有的 block，以及該工具有的 edge。
- 把工具加入 `TOOLS`，如果它建構圖，也加入 `DESCENT_TOOLS`。執行 demo，並閱讀多數集合中改變了什麼。
- 寫一個斷言該工具在 `ledger-x86_64-O1.o` 上 block 數量的測試，以及一個斷言它如何處理 trap 表格的測試。
- 為這堂課寫一段話，用該工具自己的詞語，說明它把 loader、disassembler 與 CFG 稱為什麼。

成功的意思是，`python3 code/main.py` 在 `captureTools` 中回報新的工具，`ledger-x86_64-O1.o` 中的每個函式都保持 `allDescentToolsAgree` 為 true，而 `sweep-trap.o` 的發現說明了新工具對位元組 `0x10` 到 `0x14` 做了什麼。

## 課程成品 (Lesson Artifact)

這堂課包含 `outputs/skill-the-reverse-engineering-landscape.md`。用它在你建構的一個 object 上執行數個工具、把它們的位址化簡為單一 base，並針對每個函式回報它們在指令起點、block 與 edge 上的一致之處，以及每個工具捏造或漏掉了哪些位元組。它要求有記錄指令與版本的擷取、從檔案讀取的 symbol table，以及說明哪些工具只是根據文件描述。

## 驗證 (Verify It)

執行 demo 與確定性的測試套件：

```bash
cd phases/13-tools-and-protocols/37-the-reverse-engineering-landscape
python3 code/main.py
python3 -m unittest discover -s code/tests -v
```

驗證應該證明：

- ELF reader 在 `ledger-x86_64-O1.o` 中找到四個函式 symbol，大小為 16、46、31 和 63，以及兩個 padding 空隙
- 四個 object 的 sweep 擷取分別包含 86、56、84 和 11 條指令
- radare2 與 rizin 在 `ledger-x86_64-O1.o` 上回報相同的指令、block 與 edge，而只有 radare2 有 `pdc` 文字
- angr 把 `sub_40003e` 和 `sub_40005f` 標記為沒有 symbol 的函式
- Ghidra 對 `total` 的 block 是課程 33 的五個 block，而它的 call flow 不是圖的 edge
- 每個工具在移除 base 之後，都把 `grade` 放在 offset `0x10`
- 每個工具在 `ledger-x86_64-O1.o` 的每個函式中，對每個指令起點都一致
- 四個圖建構器在那裡回報 3、3、5、4 個 block，以及 2、3、5、4 條 edge，且 edge 集合相同
- `-O2` 的 `total` 在每個圖建構器中都有 10 個 block 與 13 條 edge
- `-O0` 的 `grade` 在 radare2、rizin、angr 與 Ghidra 中，分別有 3、8、8 和 9 個 block，以及 2、11、7 和 12 條 edge，原因記在各工具的註記中
- `-O0` 的 `grade_label` 在除了 angr 之外的每個地方都是一個 block，而 angr 在 call 處結束 block
- angr 為每個 padding 空隙捏造一個函式，外加 trap 表格上的一個，而沒有其他工具捏造或漏掉函式
- sweep 在 trap 上多了起點 `0x10`、`0x12`、`0x13`，並失去 `0x14`，且沒有圖建構器解碼 `0x10` 到 `0x14`
- 發現指出 radare2 漏掉的範圍，以及 angr 捏造的每個函式
- manifest 為全部 21 個擷取檔案記錄版本與指令
- 三個工具反編譯 `grade`，而只有 Ghidra 折疊了除法慣用寫法
- translator 在一種風格中積極寫入 CF、OF、SF、ZF、PF，在另一種風格中寫入 `cc_op` thunk，對 `cmp` 不寫入暫存器，對 `movslq` 不寫入 flag
- translator 拒絕記憶體形式、其他 opcode、額外的 REX 位元，以及錯誤的長度
- translator 的 `cc_op` 數字 8 和 7，等於擷取的 VEX 中的那些
- 擷取的 lift 為這三條指令給出 9、1、10 個 P-code 操作，以及 7、4、7 個 VEX statement
- 時間線依序從 1973 年走到 2026 年，管線有八個階段，六個工具在每個格子都有術語，而兩個閉源工具被標記為未執行
- 互動式圖表內嵌了 harness 產生的相同資料

## Production 的失敗模式 (Production Failure Modes)

| 失敗 | 薄弱的報告說什麼 | Harness 必須證明什麼 |
|---|---|---|
| 來自不同工具的原始位址 | 「這些工具對 grade 在哪裡有分歧」 | 移除各自 base 之後的單一 offset，並在一個函式上檢查 |
| 把 block 數量當作找到多少程式碼的指標 | 「Ghidra 找到最多程式碼」 | 相同的指令起點，多出來的 block 由標籤解釋 |
| 把單一工具當作 oracle | 「objdump 顯示表格位元組是指令」 | sweep 是 linear sweep；三個圖建構器不認領這些位元組 |
| 把 padding 函式算成發現 | 「angr 多找到七個函式」 | 每一個都位於 padding 空隙或資料上，且沒有一個有 symbol |
| 把停止的走訪回報為 dead code | 「四個 case block 是 unreachable」 | 一個工具漏掉 36 個位元組，而三個工具到達了它們 |
| 在沒有註記的情況下比較 edge 數量 | 「rizin 和 Ghidra 對 grade 有分歧」 | 相同的四個計算式目標，在標籤處多一次切分 |
| 把反編譯器的算術讀成程式本身 | 「函式乘以 1717986919」 | 課程 34 的慣用寫法，一個工具折疊了它，另一個留下了它 |
| 把閉源工具描述得像是執行過 | 「Binary Ninja 回報五個 block」 | 圖表與表格將它標記為僅查閱文件 |
| 沒有指令的擷取 | 「我們執行了 radare2」 | 為每個檔案記錄版本、指令與日期的 manifest |
| 把沒有 flag 的 IR 讀成有損的 lift | 「VEX 丟掉了 flag」 | 位於 `cc_op`、`cc_dep1`、`cc_dep2` 的 thunk，在分支處求值 |

## 營運規則 (Operational Rule)

用指令和版本把每個工具擷取一次，把每個位址化簡為單一 base，並先比較指令起點，再比較 block，再比較 edge。把每個位元組記為已認領、padding、資料或漏掉。在判斷某個工具的數量之前，先閱讀它自己的註記，並說明你只是讀到哪些工具而沒有執行。

## 延伸閱讀 (Further Reading)

- [Cifuentes, "Reverse Compilation Techniques", PhD thesis, QUT (1994)](https://eprints.qut.edu.au/36820/)
- [Cifuentes and Gough, "Decompilation of Binary Programs", Software: Practice and Experience (1995)](https://doi.org/10.1002/spe.4380250706)
- [Van Emmerik, "Static Single Assignment for Decompilation", PhD thesis, University of Queensland (2007)](https://espace.library.uq.edu.au/view/UQ:158682)
- [Schwartz, Lee, Woo, Brumley, "Native x86 Decompilation Using Semantics-Preserving Structural Analysis and Iterative Control-Flow Structuring" (USENIX Security 2013)](https://www.usenix.org/conference/usenixsecurity13/technical-sessions/presentation/schwartz)
- [Yakdan et al., "No More Gotos" (NDSS 2015)](https://www.ndss-symposium.org/ndss2015/ndss-2015-programme/no-more-gotos-decompilation-using-pattern-independent-control-flow-structuring-and-semantics/)
- [Basque et al., "Ahoy SAILR! There is No Need to DREAM of C" (USENIX Security 2024)](https://www.usenix.org/conference/usenixsecurity24/presentation/basque)
- [Andriesse et al., "An In-Depth Analysis of Disassembly on Full-Scale x86/x64 Binaries" (USENIX Security 2016)](https://www.usenix.org/conference/usenixsecurity16/technical-sessions/presentation/andriesse)
- [Flake, "Structural Comparison of Executable Objects" (DIMVA 2004)](https://dl.gi.de/items/7f573931-cb51-49ef-a12c-0452a9fb0ea4)
- [Maranzano and Bourne, "A Tutorial Introduction to ADB", Bell Labs (1977)](http://squoze.net/UNIX/v7/files/doc/18_adb.pdf)
- [Stallman, KTH lecture on GDB entering distribution (30 October 1986)](https://www.gnu.org/philosophy/stallman-kth.html)
- [Hex-Rays, "IDA: celebrating 30 years of binary analysis innovation"](https://hex-rays.com/blog/ida-celebrating-30-years-of-binary-analysis-innovation)
- [Guilfanov, "Decompiler internals: microcode", Recon Brussels 2018 (slides)](https://www.recon.cx/2018/brussels/resources/slides/RECON-BRX-2018-Decompiler-internals-microcode.pdf)
- [Hex-Rays, microcode view and maturity levels](https://docs.hex-rays.com/core/decompiler/how-tos/microcode-view)
- [Hex-Rays, IDA 9.4 release notes](https://docs.hex-rays.com/release-notes/9_4)
- [Hex-Rays, IDA Free](https://hex-rays.com/ida-free)
- [Ghidra releases (12.1.4)](https://github.com/NationalSecurityAgency/ghidra/releases)
- [Ghidra P-code reference](https://github.com/NationalSecurityAgency/ghidra/blob/master/GhidraDocs/languages/html/pcoderef.html)
- [Ghidra decompiler source, main work flow and Action classes](https://github.com/NationalSecurityAgency/ghidra/tree/master/Ghidra/Features/Decompiler/src/decompile/cpp)
- [Ghidra analyzeHeadless README](https://github.com/NationalSecurityAgency/ghidra/blob/master/Ghidra/RuntimeScripts/support/analyzeHeadlessREADME.md)
- [Binary Ninja, BNIL overview](https://docs.binary.ninja/dev/bnil-overview.html)
- [Binary Ninja, "Architecture Agnostic Function Detection in Binaries" (2017)](https://binary.ninja/2017/11/06/architecture-agnostic-function-detection-in-binaries.html)
- [Binary Ninja, batch processing and headless licences](https://docs.binary.ninja/dev/batch.html)
- [Binary Ninja 6.0 "Krypton" release post](https://binary.ninja/2026/09/03/binary-ninja-6.0-krypton.html)
- [Binary Ninja Free](https://binary.ninja/free/)
- [radare2 book, ESIL](https://book.rada.re/emulation/esil.html)
- [radare2 book, decompilers](https://book.rada.re/arch/decompile.html)
- [radare2 releases (6.2.4)](https://github.com/radareorg/radare2/releases)
- [rizin, RzIL design notes](https://github.com/rizinorg/rizin/blob/dev/doc/rzil.md)
- [rizin releases (0.9.1)](https://github.com/rizinorg/rizin/releases)
- [Announcing Rizin (8 December 2020)](https://rizin.re/posts/announcing-rizin/)
- [Cutter releases (2.5.0)](https://github.com/rizinorg/cutter/releases)
- [angr documentation, CFG recovery](https://docs.angr.io/en/latest/analyses/cfg.html)
- [angr documentation, the decompiler and its analysis passes](https://docs.angr.io/en/latest/analyses/decompiler.html)
- [angr documentation, intermediate representation (VEX)](https://docs.angr.io/en/latest/advanced-topics/ir.html)
- [angr on PyPI (10.0.3)](https://pypi.org/project/angr/)
- [pypcode on PyPI](https://pypi.org/project/pypcode/)
- [llvm-objdump command guide](https://llvm.org/docs/CommandGuide/llvm-objdump.html)
- [GNU objdump manual](https://sourceware.org/binutils/docs/binutils/objdump.html)
- [LIEF documentation](https://lief.re/doc/stable/index.html)
- [Capstone releases](https://github.com/capstone-engine/capstone/releases)
- [BinDiff releases (v8, open source)](https://github.com/google/bindiff/releases/tag/v8)
- [Diaphora](https://github.com/joxeankoret/diaphora)
- [Hopper Disassembler downloads](https://www.hopperapp.com/download.html)
- [Compiler Explorer](https://github.com/compiler-explorer/compiler-explorer)
- [LLDB announced as an LLVM subproject (2010)](https://blog.llvm.org/2010/06/new-lldb-debugger.html)
- [Boomerang decompiler](https://boomerang.sourceforge.net/)
- [Tan et al., "LLM4Decompile" (EMNLP 2024)](https://aclanthology.org/2024.emnlp-main.203/)
- [Gao et al., "DecompileBench" (Findings of ACL 2025)](https://aclanthology.org/2025.findings-acl.1194/)
- [Liu, Raff, Micinski, "When LLM Decompilers Recompile More and Preserve Less" (2026)](https://arxiv.org/abs/2609.05370)
