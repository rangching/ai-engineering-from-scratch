# 程式的解剖：從零開始寫 ELF Reader (Program Anatomy: An ELF Reader From Scratch)

> 編譯後的程式是一組互相指向的表。你自己寫的 reader，輸出的結果可以逐位元組檢查。

**Type:** Build
**Languages:** Python (stdlib), C (fixture only)
**Prerequisites:** Phase 0 · 01 (Dev Environment) with clang installed, Python 3.10 or later, hexadecimal arithmetic
**Time:** ~120 分鐘

## 學習目標 (Learning Objectives)

- 說明這條路線中每個目標所適用的授權規則，以及 17 U.S.C. 1201(f) 的四項限制。
- 用 `struct.unpack` 讀取 ELF header、section header table 與兩個 string table。
- 用 gABI 與 x86-64 psABI 的名稱，解碼 `.symtab` 項目與 `.rela` 紀錄。
- 回傳具名函式的確切位元組，並把每個檔案 offset 對應到擁有它的結構。
- 讀取 program header，並透過 `PT_LOAD` 把虛擬位址對應到檔案 offset。
- 讀取 Mach-O 物件的 load command，並說出每一個對應到哪個 ELF 概念。

## 開始之前：授權與法律 (Before You Start: Authorization and the Law)

課程 32 到 36 的每個目標，都是你自己寫、自己編譯的程式。原始碼是 `code/fixtures/ledger.c`，四個簡短的 C 函式。你擁有原始碼、編譯指令，以及它產生的 object file。沒有授權條款要讀、沒有保護機制要通過，也沒有第三方的程式碼被你打開。這條路線不涵蓋軟體保護、授權檢查，或你不擁有的程式。

這條規則之所以存在，是因為法律把反組譯視為複製。在 *Sega Enterprises Ltd. v. Accolade, Inc.*, 977 F.2d 1510 (9th Cir. 1992) 中，法院認為反組譯期間製作的中間副本，依著作權法屬於副本。接著法院認為：「當反組譯是取得受著作權保護的電腦程式中所體現之概念與功能性元素的唯一方式，且有正當理由尋求這種存取時，反組譯依法屬於對該著作的合理使用（fair use）。」

第二個案例是 *Sony Computer Entertainment, Inc. v. Connectix Corp.*, 203 F.3d 596 (9th Cir. 2000)。Connectix 在建構模擬器的過程中複製了 PlayStation BIOS，法院認為這些副本屬於合理使用。兩個案例保護的都是研究。兩者都沒有保護最終產品中被複製的程式碼。

兩條法規設定了限制。17 U.S.C. 1201(f) 讓「已合法取得電腦程式副本使用權」的人，可以規避存取控制，「其唯一目的，是辨識並分析該程式中，為了讓獨立建立的電腦程式與其他程式達成互通性所必需的那些元素」。四項限制疊加：合法的副本、互通性的目的、不是已經「容易取得」的元素，以及分析本身在其他方面不構成侵權。1201(f) 只是反規避規則的例外。複製的問題仍歸合理使用處理。

在歐盟，Directive 2009/24/EC 第 5(3) 條涵蓋研究。有權使用副本的人可以「觀察、研究或測試程式的功能，以判定構成程式任何元素之基礎的概念與原則」。這項權利適用於載入、顯示、執行、傳輸或儲存程式的期間。

第 6 條允許在沒有權利人授權的情況下進行反編譯，只要它「對於取得讓獨立建立的電腦程式達成互通性所必需的資訊而言是不可或缺的」。其條件與 1201(f) 相符。由被授權人執行這些行為、資訊先前不是容易取得的，且這些行為限於所需的部分。第 6(2) 條禁止將結果用於任何其他目標，或用於在表達上實質相似的程式。第 8 條使違反第 6 條或第 5(3) 條的契約條款無效。

| 來源 | 它允許什麼 | 條件 |
|---|---|---|
| 17 U.S.C. 1201(f) | 為了分析程式而規避存取控制 | 合法的副本、互通性的目的、資訊不是容易取得的、不構成侵權 |
| Directive 2009/24/EC Art. 5(3) | 觀察、研究與測試執行中的程式 | 有使用該副本的權利、行為在正常載入與執行的範圍內 |
| Directive 2009/24/EC Art. 6 | 反編譯 | 被授權人、互通性的目的、資訊不是容易取得的、限於所需的部分 |
| *Sega v. Accolade* | 反組譯作為合理使用 | 是取得概念與功能性元素的唯一方式、有正當理由 |
| *Sony v. Connectix* | 逆向工程期間的中間副本 | 最終產品不構成侵權 |

這些筆記來自法規與判決意見，不是法律建議。當未來的目標不是你的時，請在開啟檔案之前取得書面授權，並把授權與你的證據一起保存。Reverse Engineering 101 手冊的第 2.1 節提供完整的引用與 CJEU 案例。

## 問題 (The Problem)

你編譯了 `ledger.c`，並對 object 執行 `llvm-objdump -t`。輸出列出 `percent_change`，value 為 `0000000000000000`，size 為 `0000000000000010`。在 1,768 位元組的檔案中，哪 16 個位元組是那個函式？清單也顯示 `R_X86_64_PC32 .rodata+0x10` 位於 offset `0x36`，而檔案中該處的四個位元組是零。它們會變成什麼？為什麼 addend 是 `0x10`，而表格從 `0x14` 開始？

rea 用 `inspect_binary_layout` 回答版面配置的問題，而該 tool 只存在於 Linux x64，且需要在 `REA_PWNTOOLS_PYTHON` 背後有 pwntools 4.15.0。在這台 Mac 上它不存在。`address_to_file_offset` 需要 Ghidra session 與原始檔案位元組的完整副本，且 SHA-256 要相符。`inspect_artifact` 依路徑與內容 hash 列出容器裡有什麼。它不會打開 object file 內部的表。所以你要自己寫 reader，用大約 950 行 Python，搭配 `struct.unpack`，沒有相依套件。測試會把每個表與一次性擷取到 fixture 中的 `llvm-objdump` 文字做比較。

```figure
elf-byte-map
```

## 檔案是指向表的表 (The File Is Tables That Point at Tables)

System V gABI 第 4 章的 Figure 4-1，畫出同一個檔案的兩種視圖。連結視圖有 ELF header、選擇性的 program header table、section，以及必要的 section header table。執行視圖有 ELF header、必要的 program header table、segment，以及選擇性的 section header table。gABI 加了一句話，主導整個 reader：「只有 ELF header 在檔案中有固定的位置」。其他每個結構，都是靠另一個結構中儲存的 offset 找到的。

gABI 也固定了純量型別，並說「ELF 不使用 bit-field」。`Elf64_Addr` 和 `Elf64_Off` 是 8 位元組，`Elf64_Half` 是 2，`Elf64_Word` 是 4，`Elf64_Xword` 是 8。x86-64 psABI 的 4.1.1 節要求 `ELFCLASS64`、`ELFDATA2LSB`、`e_machine` 為 `EM_X86_64` (62)，以及 `e_flags` 為 0。Little-endian 加上固定寬度，意味著帶有 `<` 前綴的 `struct.unpack` 可以一次呼叫就讀取每筆紀錄。

`ledger-x86_64-O1.o` 的前 64 個位元組：

```text
00000000: 7f45 4c46 0201 0100 0000 0000 0000 0000  .ELF............
00000010: 0100 3e00 0100 0000 0000 0000 0000 0000  ..>.............
00000020: 0000 0000 0000 0000 2804 0000 0000 0000  ........(.......
00000030: 0000 0000 4000 0000 0000 4000 0b00 0100  ....@.....@.....
```

對照 gABI 的「ELF Header」頁面閱讀。`e_ident` 是 16 個位元組：`7f 45 4c 46`（magic）、class 2（64 位元）、data 1（little-endian）、version 1、OS ABI 0（System V），以及 padding。

接著 `e_type` `0x0001` 是 `ET_REL`，也就是可重定位的 object。`e_machine` `0x003e` 是 62，`EM_X86_64`。`e_entry` 是零，因為 object 沒有進入點。`e_phoff` 是零，因為 object 沒有 program header。`e_shoff` 是 `0x428`，是 section header table 的檔案 offset。最後一列包含 `e_ehsize` `0x40`、`e_phentsize` 0、`e_phnum` 0、`e_shentsize` `0x40`、`e_shnum` `0x0b`（11 個 header），以及 `e_shstrndx` `0x0001`。

Reader 把那一列變成一次呼叫：

```python
fields = struct.unpack_from("<HHIQQQIHHHHHH", data, 16)
```

格式字串就是由左到右閱讀的 gABI 圖表：兩個 halfword、一個 word、三個 8 位元組的值、一個 word，以及六個 halfword。Reader 會拒絕 magic、class 或位元組順序不同的檔案，也會拒絕 `e_shentsize` 或 `e_phentsize` 不是 64 和 56 的值。錯誤的寬度會讓之後每個 offset 都錯，所以檢查放在最前面。

## Section header 與兩個 string table (Section Headers and Two String Tables)

每個 section header 是 64 位元組的 `Elf64_Shdr`（gABI Figure 4-8）：`sh_name`、`sh_type`、`sh_flags`、`sh_addr`、`sh_offset`、`sh_size`、`sh_link`、`sh_info`、`sh_addralign` 與 `sh_entsize`。格式字串是 `<IIQQQQIIQQ`。`sh_name` 是 index 為 `e_shstrndx` 的 section 內的 offset，名稱在第一個零位元組處結束。在這個 object 中，`e_shstrndx` 是 1，而 clang 把那個 section 命名為 `.strtab`，所以一個表同時服務 section 名稱與 symbol 名稱。

Reader 印出 -O1 object 的 11 個 header：

```text
Idx Name             Type         Flags   Offset     Size       Addr Link Info Align
  0                  NULL                    0x0      0x0        0x0    0    0     0
  1 .strtab          STRTAB                0x368     0xbe        0x0    0    0     1
  2 .text            PROGBITS     AX        0x40     0x9f        0x0    0    0    16
  3 .rela.text       RELA         I        0x2a8     0x48        0x0   10    2     8
  4 .rodata          PROGBITS     A         0xe0     0x50        0x0    0    0     8
  5 .rela.rodata     RELA         I        0x2f0     0x78        0x0   10    4     8
  6 .rodata.str1.1   PROGBITS     AMS      0x130      0xa        0x0    0    0     1
  7 .comment         PROGBITS     MS       0x13a     0x2f        0x0    0    0     1
  8 .note.GNU-stack  PROGBITS              0x169      0x0        0x0    0    0     1
  9 .llvm_addrsig    LLVM_ADDRSIG E        0x368      0x0        0x0   10    0     1
 10 .symtab          SYMTAB                0x170    0x138        0x0    1    9     8
```

Index 0 是保留的 null header。`.text` 在檔案 offset `0x40` 持有 159 位元組的程式碼，就在 header 之後，flag 為 `A`（在記憶體中配置）與 `X`（可執行）。`.rodata` 持有 80 位元組的唯讀資料。`.rodata.str1.1` 持有五個標籤字串 `F\0D\0C\0B\0A\0`，共 10 位元組，flag 為 `M`（可合併）與 `S`（字串）。

`.comment` 持有 47 位元組的編譯器標記 `\0Apple clang version 17.0.0 (clang-1700.6.3.2)\0`。`.llvm_addrsig` 的 type 是 `0x6fff4c03`，落在 gABI 保留的 OS 專屬範圍 `SHT_LOOS` 到 `SHT_HIOS` 之內，而 LLVM 的 `ELF.h` 為它命名。它的 flag `E` 是 `SHF_EXCLUDE`，所以 linker 會丟棄它。

`sh_link` 和 `sh_info` 欄位是表與表之間的指標。`.rela.text` 的 `sh_info` 為 2、`sh_link` 為 10：它修補 section 2，並從 section 10 取得它的 symbol。`.symtab` 的 `sh_link` 為 1，也就是 symbol 名稱的 string table，而 `sh_info` 為 9，也就是第一個非 local symbol 的 index。每一列的 `sh_addr` 都是零，因為 object 還沒有位址。Linker 會指派它們。

兩條 gABI 規則讓 reader 能畫出檔案的完整地圖。Section「是連續的，而且不會重疊」，而檔案「可能包含未使用的空間」。Reader 依 offset 對每個結構排序，並填補空洞：

```text
     0x0 +64    header   ELF header
    0x40 +159   section  .text
    0xdf +1     gap      inactive space
    0xe0 +80    section  .rodata
   0x130 +10    section  .rodata.str1.1
   0x13a +47    section  .comment
   0x169 +7     gap      inactive space
   0x170 +312   section  .symtab
   0x2a8 +72    section  .rela.text
   0x2f0 +120   section  .rela.rodata
   0x368 +190   section  .strtab
   0x426 +2     gap      inactive space
   0x428 +704   table    section header table
```

三個空隙總共 10 位元組，每一個都是對齊造成的。`.rodata` 的 `sh_addralign` 是 8，所以它從 `0xe0` 而不是 `0xdf` 開始。`.symtab` 的對齊是 8，所以 `.comment` 之後有七個位元組沒有使用。Section header table 本身也對齊到 8。

類型為 `SHT_NOBITS` 的 section 完全不佔用檔案位元組，所以地圖會跳過它們。這個 object 中沒有任何東西是那個類型。連結後程式中的 `.bss` 才是，而信任那裡的 `sh_size` 的 reader 會讀過檔案的結尾。

## Symbol：Binding、Type、Section、Value、Size

`Elf64_Sym` 是 24 位元組（gABI Figure 4-18）：`st_name`、`st_info`、`st_other`、`st_shndx`、`st_value`、`st_size`，格式 `<IBBHQQ`。`st_info` 包裝兩個欄位：高四位元是 binding，低四位元是 type。gABI 把 binding 命名為 `STB_LOCAL` 0、`STB_GLOBAL` 1、`STB_WEAK` 2，並把 type 命名為 `STT_NOTYPE` 0、`STT_OBJECT` 1、`STT_FUNC` 2、`STT_SECTION` 3、`STT_FILE` 4。`st_other` 的低兩位元持有 visibility，這裡全部都是 `STV_DEFAULT` 0。

```text
Idx            Value   Size Bind   Type    Section          Name
  0              0x0      0 LOCAL  NOTYPE  *UND*
  1              0x0      0 LOCAL  FILE    *ABS*            ledger.c
  2              0x0     20 LOCAL  OBJECT  .rodata          labels.rel
  3              0x0      0 LOCAL  SECTION .rodata          .rodata
  4              0x2      2 LOCAL  OBJECT  .rodata.str1.1   .L.str.1
  5              0x4      2 LOCAL  OBJECT  .rodata.str1.1   .L.str.2
  6              0x6      2 LOCAL  OBJECT  .rodata.str1.1   .L.str.3
  7              0x8      2 LOCAL  OBJECT  .rodata.str1.1   .L.str.4
  8              0x0      0 LOCAL  SECTION .rodata.str1.1   .rodata.str1.1
  9              0x0     16 GLOBAL FUNC    .text            percent_change
 10             0x10     46 GLOBAL FUNC    .text            grade
 11             0x40     31 GLOBAL FUNC    .text            total
 12             0x60     63 GLOBAL FUNC    .text            grade_label
```

Index 0 是保留的 undefined symbol。`st_shndx` 指名定義該 symbol 的 section，有三個保留值：`SHN_UNDEF` 0、檔案名稱用的 `SHN_ABS` `0xfff1`，以及 `SHN_COMMON` `0xfff2`。Section symbol 的名稱是空的，所以 `llvm-objdump` 會為它們印出該 section 自己的名稱，而 reader 也這麼做。gABI 規則「所有 STB_LOCAL binding 的 symbol，都在 weak 與 global symbol 之前」，表現為前九列，而 `.symtab` 就是因為這個原因在 `sh_info` 中記錄了 9。

在可重定位的 object 中，`st_value` 是 symbol 所屬 section 內的 offset，因為 `sh_addr` 是零。因此函式的檔案 offset 是 `sh_offset + st_value - sh_addr`，而它的長度是 `st_size`：

```python
section = elf.sections[symbol.st_shndx]
file_offset = section.sh_offset + (symbol.st_value - section.sh_addr)
code = elf.data[file_offset : file_offset + symbol.st_size]
```

對 `percent_change` 而言，那是 `0x40 + 0`，16 個位元組是 `85 ff 74 09 29 fe 6b c6 64 99 f7 ff c3 31 c0 c3`。大小和起點一樣重要。`grade` 結束於 `.text` offset `0x3e`。接在後面的兩個位元組 `66 90` 是編譯器加入的 `nop`，讓 `total` 從 `0x40` 開始，也就是 16 位元組的邊界。這兩個位元組不屬於任何 symbol。課程 33 使用 `st_size` 作為 disassembler 的停止條件，而課程 36 在比較重建結果之前，會把它們遮掉。

## Relocation：零位元組代表什麼 (Relocations: What the Zero Bytes Mean)

`Elf64_Rela` 是 24 位元組（gABI Figure 4-23）：`r_offset`、`r_info`，以及有號的 `r_addend`，格式 `<QQq`。gABI 用兩個巨集拆開 `r_info`：`ELF64_R_SYM(i)` 是 `i >> 32`，`ELF64_R_TYPE(i)` 是 `i & 0xffffffff`。在可重定位的檔案中，`r_offset` 是該表 `sh_info` 所指名的 section 內的 offset。型別名稱與公式來自 x86-64 psABI 的 Table 4.9。其中 `S` 是 symbol 的值，`A` 是 addend，`P` 是被修補的位置，`L` 是 PLT 項目。`R_X86_64_64` 是 `S + A`，`R_X86_64_PC32` 是 `S + A - P`，`R_X86_64_PLT32` 是 `L + A - P`。

-O1 object 有八筆紀錄：

```text
  .rela.text       0x36 R_X86_64_PC32          .rodata+0x10
  .rela.text       0x84 R_X86_64_PC32          .rodata+0x24
  .rela.text       0x93 R_X86_64_PC32          .rodata-0x4
  .rela.rodata      0x0 R_X86_64_PC32          .rodata.str1.1
  .rela.rodata      0x4 R_X86_64_PC32          .L.str.1+0x4
  .rela.rodata      0x8 R_X86_64_PC32          .L.str.2+0x8
  .rela.rodata      0xc R_X86_64_PC32          .L.str.3+0xc
  .rela.rodata     0x10 R_X86_64_PC32          .L.str.4+0x10
```

把第一筆拆開來看。Offset `0x36` 位於 `grade` 之內，範圍是 `0x10` 到 `0x3e`。位於 `0x33` 的指令是 `48 8d 0d 00 00 00 00`，是 RIP 相對位址的 `lea`，它的四位元組 displacement 位於 `0x36`。處理器把 displacement 加到下一條指令的位址 `0x3a`。

編譯器希望結果是 `.rodata + 0x14`，也就是一個包含 32 位元值 `1, 2, 3, 4, 4` 的五項表格的起點。所以儲存的值必須是 `(.rodata + 0x14) - 0x3a`。公式 `S + A - P` 在 `P = 0x36` 時，得到 `.rodata + A - 0x36`。令兩者相等，得到 `A = 0x14 - 4 = 0x10`。Addend 帶的是目標 offset 減去 displacement 到指令結尾之間的四個位元組。

那個表格就是 clang 在 -O1 編譯 `grade` 中 `switch` 的方式。程式碼用乘以 `0x66666667` 加兩次 shift 算出 `score / 10`，減 6，與 4 比較，然後索引該表格。-O1 的 ELF object 沒有保留該表格的 symbol。參照方式是 section symbol `.rodata` 加上 addend，只有同一份原始碼的 arm64 build 才把它命名為 `l_switch.table.grade`。第二筆紀錄位於 `grade_label` 的 `0x84`，指向 `.rodata + 0x28` 的第二個 8 位元組值表格。第三筆位於 `0x93`，指向 `.rodata + 0x0`，也就是 `labels.rel` 的起點。

五筆 `.rela.rodata` 紀錄填滿 `labels.rel`。每個項目位於 offset `4i`，並得到 addend `4i`，所以 `S + A - P` 化簡為字串的位址減去表格的基底。`grade_label` 用 `movslq` 讀取一個項目，把表格基底加回去，就得到指標。第一筆紀錄指名 section symbol `.rodata.str1.1`，addend 為 0，因為字串 `F` 是該 section 的第一個位元組。其他的指名 `.L.str.1` 到 `.L.str.4`，也就是編譯器為 `D`、`C`、`B` 和 `A` 產生的 local 標籤。

同一份原始碼在三個最佳化層級，編譯成三種不同的表格。Reader 不用 disassembler 就能顯示差異：

| Build | `labels` 位於何處 | `grade` 如何選擇 | `grade_label` 如何到達 `grade` |
|---|---|---|---|
| -O0 | `.data`，五個 8 位元組指標，五筆 `R_X86_64_64` 紀錄 | `.rodata` 中的 jump table：五筆指向 `.text` 的 `R_X86_64_PC32` 紀錄 | 帶有 `R_X86_64_PLT32 grade-0x4` 的 call |
| -O1 | `.rodata` 中的 `labels.rel`，五個 32 位元相對項目 | 位於 `.rodata+0x14` 的值表格，一筆 `R_X86_64_PC32` | 內嵌（inline），沒有 call |
| -O2 | 與 -O1 相同 | 與 -O1 相同 | 內嵌，`total` 增長到 140 位元組 |

在 -O0，`switch` 是真正的 jump table，其五個項目指回 `.text` 的 `+0x91`、`+0x8c`、`+0x87`、`+0x82` 和 `+0x86`。對 `grade` 的 call 帶有與 `lea` 相同的 `-0x4` addend，原因也相同。在 -O1，最佳化器發現每個 case 都回傳常數，就把 jump 換成結果表格。課程 33 會解碼這兩種形狀。

## Program header：loader 的視圖 (Program Headers: The Loader's View)

可重定位的 object 沒有 program header，所以 `e_phnum` 是 0。可執行檔或 shared object 有，而 loader 只讀它們。`Elf64_Phdr` 是 56 位元組（gABI Figure 5-1）：`p_type`、`p_flags`、`p_offset`、`p_vaddr`、`p_paddr`、`p_filesz`、`p_memsz`、`p_align`，格式 `<IIQQQQQQ`。gABI 對 `PT_LOAD` 項目固定了三條規則。它們以 `p_vaddr` 遞增的順序出現，`p_memsz` 大於 `p_filesz` 表示補零。`p_vaddr` 與 `p_offset` 在模 `p_align` 下同餘。

這條路線中的每個目標，都是你自己建構的程式，而這台 Mac 上沒有 linker 能從 `ledger.o` 產生 Linux 可執行檔。所以 fixture `tiny-x86_64` 來自 `code/fixtures/build_tiny_elf.py`，它用 `struct.pack` 寫出 736 位元組。檔案包含一個 ELF header、兩個 program header、36 位元組的 `_start`、3 位元組的 `.rodata`、一個 symbol table，以及六個 section header。`_start` 把 `ok\n` 寫到檔案描述符 1，並以狀態 42 結束。在 Linux x86-64 上它可以執行。在這裡，它是 program header 路徑的 fixture。

```text
Program headers:
  PT_LOAD          r-x offset=0x0 vaddr=0x400000 filesz=0x2e0 memsz=0x2e0 align=0x1000
  PT_GNU_STACK     rw- offset=0x0 vaddr=0x0 filesz=0x0 memsz=0x0 align=0x10

entry 0x4000b0 is file offset 0xb0 through PT_LOAD
```

單一的 `PT_LOAD` 把整個 736 位元組的檔案對應到 `0x400000`。Reader 找出包含某個位址的 `PT_LOAD`，並計算 `p_offset + (address - p_vaddr)`，藉此把位址對應到檔案 offset。對進入點而言，那是 `0 + (0x4000b0 - 0x400000) = 0xb0`，也就是 `.text` 的起點。

這就是 rea 的 `address_to_file_offset` 背後的計算。對 Ghidra 目標，該 tool 回傳 `{address, file_offset}`。Hopper provider 還會回傳 `provider_file_offset` 與 `image_base_file_offset`。該 tool 需要原始檔案的完整副本，而該副本的 SHA-256 必須與目標相符。rea 的 `inspect_binary_layout` 文件加了一句值得記住的話：「連結後的位址不是 runtime 位址」。與位置無關的可執行檔，會載入到 kernel 選擇的基底，而檔案 offset 是唯一穩定的座標。

Program header 也解釋了為什麼被 strip 的可執行檔仍然可以執行。Linker 可以完全移除 section header table，使 `e_shoff` 變成零。走訪 section 的 tool 於是什麼都看不到。Loader 從來沒有看過它們。

## Mach-O 中的相同概念 (The Same Ideas in Mach-O)

Apple 沒有發布 Mach-O 的文字規格。XNU 原始碼中的 header 就是規格：`EXTERNAL_HEADERS/mach-o/loader.h`、`nlist.h` 和 `reloc.h`。Fixture `ledger-arm64-O1.o` 是同一份 `ledger.c`，以 `clang -target arm64-apple-macos -O1 -fno-asynchronous-unwind-tables -c` 建構。

`mach_header_64` 是 32 位元組：`magic`、`cputype`、`cpusubtype`、`filetype`、`ncmds`、`sizeofcmds`、`flags`、`reserved`。Magic 是 `0xfeedfacf`，即 `MH_MAGIC_64`。`cputype` 是 `0x0100000c`，即 `CPU_TYPE_ARM64`。`filetype` 是 1，即 `MH_OBJECT`。`ncmds` 是 5，`sizeofcmds` 是 616：load command 佔據 header 之後的 616 位元組，每一個都以 `{cmd, cmdsize}` 開頭。Flag `0x2000` 是 `MH_SUBSECTIONS_VIA_SYMBOLS`。

```text
Load commands:
  0 LC_SEGMENT_64                cmdsize=472 at 0x20
  1 LC_BUILD_VERSION             cmdsize=24 at 0x1f8
  2 LC_LINKER_OPTIMIZATION_HINT  cmdsize=16 at 0x210
  3 LC_SYMTAB                    cmdsize=24 at 0x220
  4 LC_DYSYMTAB                  cmdsize=80 at 0x238
```

`LC_SEGMENT_64` (0x19) 是 Mach-O 對兩個 ELF 表的答案。它的 `segment_command_64` 有 `segname`、`vmaddr`、`vmsize`、`fileoff`、`filesize`、`maxprot`、`initprot`、`nsects` 和 `flags`。接著在同一個 command 中，是 `nsects` 筆 `section_64` 型別的紀錄，每筆 80 位元組。`loader.h` 中關於 `MH_OBJECT` 的註解說，section 存在於「一個沒有 segment padding 的未命名 segment」中。Fixture 顯示了這點：一個名稱為空的 segment，`fileoff` 648，`filesize` 456，以及五個 section。

```text
  Idx Section                    Addr   Size Offset Reloff Nreloc Type
    1 __TEXT,__text               0x0   0xd0    648   1104      6 S_REGULAR
    2 __DATA,__const             0xd0   0x28    856   1152      5 S_REGULAR
    3 __TEXT,__cstring           0xf8    0xa    896      0      0 S_CSTRING_LITERALS
    4 __TEXT,__const            0x108   0x40    912      0      0 S_REGULAR
    5 __LD,__compact_unwind     0x148   0x80    976   1192      4 S_REGULAR
```

每個 section 在 `segname` 中指名它的上層 segment，所以兩個叫做 `__const` 的 section 是不同的，因為一個是 `__DATA,__const`，另一個是 `__TEXT,__const`。Relocation 是透過 `reloff` 與 `nreloc` 以每個 section 為單位，而 ELF 則是每個目標有獨立的 `.rela.X` section。

每個 `relocation_info` 是 8 位元組。有號的 `r_address` 在最前面。後面是 32 位元的欄位，包裝為 `r_symbolnum`（24 位元）、`r_pcrel`（1）、`r_length`（2）、`r_extern`（1）和 `r_type`（4）。六筆 `__text` 紀錄是三對 `ARM64_RELOC_PAGE21` 與 `ARM64_RELOC_PAGEOFF12`，每個表格參照一對。arm64 的 `adrp` 加 `add` 做的事，等於 x86-64 的一個 `lea`，所以每個參照需要兩筆紀錄。五筆 `__DATA,__const` 紀錄是 `r_length` 為 3 的 `ARM64_RELOC_UNSIGNED`，所以每一筆修補一個 8 位元組的絕對指標。

在 -O1，arm64 的 build 把 `labels` 保留為單純的指標，而 x86-64 的 build 則做成相對表格。

`LC_SYMTAB` (0x2) 持有 `symoff` 1240、`nsyms` 17、`stroff` 1512、`strsize` 176，而 1512 加 176 是 1688，也就是檔案大小。每個 `nlist_64` 是 16 位元組：`n_strx`、`n_type`、`n_sect`、`n_desc`、`n_value`。`n_type` 是位元欄位。`N_TYPE`（mask `0x0e`）等於 `N_SECT`（`0xe`）表示定義在 1 起算的序號為 `n_sect` 的 section 中。`N_EXT`（`0x01`）標記 external symbol。

`nlist_64` 沒有 size 欄位。要取得 `_percent_change` 的 24 個位元組，reader 取同一個 section 中的下一個 symbol 位址，也就是 `0x18` 的 `_grade`。C 的 symbol 在這裡帶有開頭的 `_`，所以函式是 `_percent_change`。

`LC_DYSYMTAB` (0xb) 記錄與 ELF `sh_info` 相同的 local 在先規則：`ilocalsym` 0、`nlocalsym` 13、`iextdefsym` 13、`nextdefsym` 4。`LC_BUILD_VERSION` (0x32) 記錄 `platform` 1（macOS）、`minos` 26.0，以及 `sdk` 26.2，以包裝成 `XXXX.YY.ZZ` 的整數表示。這個 command 就是 arm64 fixture 以位元組形式 commit 的原因。在使用其他 SDK 的 Mac 上建構，會產生不同的檔案。

Reader 的 `compare_formats` 把兩個檔案並排呈現：

| 概念 | ELF，`ledger-x86_64-O1.o` | Mach-O，`ledger-arm64-O1.o` |
|---|---|---|
| 檔案識別 | `e_ident` `7f 45 4c 46`，class 64，data LSB | `magic` `0xfeedfacf` |
| 機器 | `e_machine` 62，`EM_X86_64` | `cputype` `0x100000c`，`CPU_TYPE_ARM64` |
| 檔案種類 | `e_type` 1，`ET_REL` | `filetype` 1，`MH_OBJECT` |
| 表格在哪裡 | `e_shoff` `0x428`，11 個 section header | header 之後 616 位元組中的 5 個 load command |
| 程式碼 | `.text` 位於 `0x40`，大小 `0x9f`，flag `AX` | `__TEXT,__text` 位於 648，大小 `0xd0`，在未命名的 segment 中 |
| Symbol 名稱 | `.symtab`（section 10），名稱在 `.strtab` 中，經由 `sh_link` 1 | `LC_SYMTAB` `symoff` 1240，`nsyms` 17，名稱位於 `stroff` 1512 |
| Symbol 大小 | `st_size`：`percent_change` 是 16 位元組 | 沒有欄位：下一個 symbol 減去這一個，24 位元組 |
| Relocation | `.rela.text`，3 筆紀錄，`R_X86_64_PC32` | `__text` `reloff` 1104，6 筆紀錄，`PAGE21` 與 `PAGEOFF12` |
| 函式位元組 | `percent_change`：位於 `0x40` 的 16 位元組 | `_percent_change`：位於 `0x288` 的 24 位元組 |

rea 透過 `inspect_macho` 讀取 Mach-O，它會透過 `xcrun` 執行 `otool -h -l`、`nm -gjU`、`dyld_info -imports -exports` 和 `dwarfdump --uuid`。你的 reader 不需要其中任何一個，而且它讀取的是 `otool -l` 印出的同一組 `symoff` 與 `stroff`。rea 的 `open_binary` tool 從 magic 位元組偵測 ELF、MZ、PE，以及單一架構或 universal 的 Mach-O。它也自己讀取 offset 18 的 `e_machine`。那就是你在這裡建構的前 20 個位元組。對於沒有 header 的檔案，你要提供 `format: raw-binary`、架構與進入點位址。Ghidra provider 會拒絕 universal binary，並要求提供瘦身後的 slice。你的 reader 看到 `0xcafebabe` 時也做同樣的事。

## 動手做 (Build It)

執行標準函式庫的 reader：

```bash
cd phases/13-tools-and-protocols/32-program-anatomy-elf-from-scratch
python3 code/main.py
```

Demo 讀取 `ledger-x86_64-O1.o`，並印出它的 header、section、symbol、relocation 與檔案 offset 地圖。接著它印出四個函式的位元組字串，讀取 `ledger-arm64-O1.o`，並印出比較表。最後以 `tiny-x86_64` 的 program header 作結。預期的行：

```text
percent_change: 16 bytes at file offset 0x40: 85 ff 74 09 29 fe 6b c6 64 99 f7 ff c3 31 c0 c3
grade: 46 bytes at file offset 0x50: 48 63 c7 48 69 c0 67 66 66 66 ...
entry 0x4000b0 is file offset 0xb0 through PT_LOAD
```

傳入路徑可讀取單一檔案：`python3 code/main.py code/fixtures/ledger-x86_64-O0.o`。Reader 從前四個位元組挑選 ELF 或 Mach-O。

依此順序閱讀 `code/main.py`：

1. `ElfFile._read_header()` 檢查 magic、class、位元組順序與項目大小，然後解開 header。
2. `ElfFile._read_sections()` 與 `read_cstring()` 讀取 section header table，並透過 `e_shstrndx` 解析名稱。
3. `ElfFile._read_symbols()` 拆開 `st_info`，透過 `sh_link` 解析名稱，並讓保留的 section index 保持可讀。
4. `ElfFile._read_relocations()` 走訪每個 `SHT_RELA` section，拆開 `r_info`，並依 Table 4.9 為型別命名。
5. `ElfFile.function_span()` 與 `function_bytes()` 把 symbol 變成檔案範圍及其中的位元組。
6. `ElfFile.file_offset_map()` 依 offset 為每個結構排序，並填補未使用的空間。
7. `ElfFile._read_program_headers()` 與 `address_to_file_offset()` 處理可執行檔。
8. `MachOFile` 讀取 header、load command、帶有 section 的 `LC_SEGMENT_64`、`LC_SYMTAB`、`LC_BUILD_VERSION`，以及每個 section 的 relocation。
9. `compare_formats()` 用從兩個檔案各自讀取的值，為兩者中的同一個概念命名。

## 實際使用 (Use It)

把這堂課的 `code` 目錄加進 `sys.path` 之後，用 `from main import ElfFile` 在後面的課程中匯入 reader。課程 33 向它要 `function_bytes` 和 `relocations_for(".text")`。課程 35 把它的輸出記錄為已觀察的證據。課程 36 向它要 relocation offset，在位元組比較之前把它們遮掉。

用相同的 `clang -target x86_64-unknown-linux-gnu -c` 指令把你自己的 C 編譯成 object，並對它執行 reader。只有當 x86-64 psABI 的 Table 4.9 列出某個 relocation 名稱時，才加入它。只有當 gABI 或產生它的 toolchain 有文件記載時，才加入 section 型別。當 reader 遇到沒有名稱的值時，它會以十六進位印出數字，而不是猜測。

## 互動實驗 (Interactive Lab)

### 實驗 A：手動讀 header

從 `code` 目錄開啟 Python：

```bash
cd phases/13-tools-and-protocols/32-program-anatomy-elf-from-scratch/code
python3 -q
```

執行：

```python
import struct
from main import ElfFile, FIXTURES
data = (FIXTURES / "ledger-x86_64-O1.o").read_bytes()
data[:4], data[4], data[5]
struct.unpack_from("<HHIQQQIHHHHHH", data, 16)
elf = ElfFile(data)
elf.header.e_shoff, elf.header.e_shnum, elf.sections[elf.header.e_shstrndx].name
```

Tuple 顯示 `e_type` 1、`e_machine` 62 和 `e_shoff` 1064，也就是 `0x428`。最後一行顯示 section 1 `.strtab` 持有 section 名稱。

### 實驗 B：switch 背後的值表格

```python
rodata = elf.section_bytes(elf.section(".rodata"))
[int.from_bytes(rodata[i : i + 4], "little") for i in range(0x14, 0x28, 4)]
[int.from_bytes(rodata[i : i + 8], "little") for i in range(0x28, 0x50, 8)]
[r for r in elf.relocations_for(".text") if r.symbol == ".rodata"]
```

兩個表格都讀作 `1, 2, 3, 4, 4`。對照 `ledger.c` 檢查：`grade(65)` 計算 `65 / 10 - 6 = 0` 並讀取項目 0，也就是 1，即 `case 6` 的值。`grade(100)` 讀取項目 4，也就是 4，即 `case 10` 和 `case 9` 共用的值。

### 實驗 C：同一份原始碼的三個 build

```python
for level in ("O0", "O1", "O2"):
    obj = ElfFile.from_path(FIXTURES / f"ledger-x86_64-{level}.o")
    print(level, [s.name for s in obj.sections if s.name in (".data", ".rodata")])
    print(level, sorted({r.type_name for r in obj.relocations}))
    print(level, {s.name: s.st_size for s in obj.symbols if s.type == "FUNC"})
```

-O0 的 object 有一個 `.data` section 和三種 relocation 型別。-O1 與 -O2 的 object 沒有 `.data`，只有一種型別。`total` 在 -O0 是 66 位元組，在 -O1 是 31，在 -O2 是 140。

### 實驗 D：Mach-O 中的同一個函式

```python
from main import MachOFile
macho = MachOFile.from_path(FIXTURES / "ledger-arm64-O1.o")
[(c.name, c.cmdsize) for c in macho.load_commands]
macho.function_span("_percent_change")
macho.function_bytes("_percent_change")[:4].hex()
```

範圍是檔案 offset 648 的 24 位元組。前四個位元組是 `a0000034`，是指令 `0x340000a0` 的 little-endian 形式，`llvm-objdump -d` 把它印成 `cbz w0, 0x14`。那就是 `before == 0` 的防護，與 x86-64 build 編碼成 `85 ff 74 09` 的是同一個測試。

## 實作練習 (Practice Lab)

把 reader 擴充到連結後的程式。在 Linux x86-64 機器上，建構 `clang -O1 -shared -fPIC code/fixtures/ledger.c -o libledger.so` 與 `clang -O1 code/fixtures/ledger.c -o ledger_main`（加上你自己加入的 `main`），然後對兩者執行 reader。

需求：

- 用相同的 `Elf64_Sym` 版面讀取 `.dynsym`，因為 `SHT_DYNSYM` 是 11，而紀錄沒有改變。
- 把 `.dynamic` 讀作由 `d_tag` 與 `d_val` 組成的 `Elf64_Dyn` 紀錄，並依 gABI 的「Dynamic Section」頁面，為 `DT_NEEDED`、`DT_STRTAB`、`DT_SYMTAB`、`DT_RELA` 和 `DT_JMPREL` 命名。
- 把 `st_shndx` 為 `SHN_UNDEF` 的 undefined symbol 列為 import，並把 `.rela.plt` 中的 `R_X86_64_JUMP_SLOT` 紀錄列為到達它們的 call site。
- 依 gABI 規則「object file 的 segment 包含一個或多個 section」，把每個 `PT_LOAD` segment 對應到它包含的 section。
- 用 `address_to_file_offset` 證明，shared object 中每個已定義函式的位元組，就是 `function_bytes` 回傳的位元組。
- 把兩個檔案的 `llvm-objdump -h -t -r -p` 與 `llvm-objdump -T` 文字擷取到 `code/fixtures/`，並在測試中將 reader 與它們比較。

成功的意思是，import 清單、segment 地圖與每個函式範圍，都與擷取到的 tool 輸出一致，且測試期間不執行任何 tool。

## 評分實驗 (Graded Lab)

關於這堂課 fixture 的五個問題。用你建構的程式碼找出每個答案，然後寫進 `code/lab_answers.json`。數字寫成數字或像 `"0x40"` 這樣的十六進位字串，清單寫成 JSON 清單。

1. `t1`：ledger-x86_64-O1.o 的 e_shoff，以數字表示。
2. `t2`：ledger-x86_64-O1.o 中 percent_change 的 st_size。
3. `t3`：ledger-x86_64-O1.o 中 .text 的檔案 offset，以十六進位表示。
4. `t4`：ledger-x86_64-O0.o 中的 relocation 型別名稱，排序後，以清單表示。
5. `t5`：ledger-x86_64-O1.o 中 .rodata+0x14 處的五個 int 值，以清單表示。

檢查你的答案：

```bash
cd phases/13-tools-and-protocols/32-program-anatomy-elf-from-scratch
python3 code/lab.py
```

檢查器會依據 `code/main.py` 在已 commit 的 fixture 上計算每個預期值。它為每個任務印出 `PASS`、`FAIL` 或 `TODO`，以及一個分數，只有當五個全部通過時才以 0 結束。`python3 code/lab.py my-answers.json` 檢查另一個檔案。`code/tests/test_lab.py` 證明檢查器接受正確的答案，並拒絕錯誤的答案。

### 看它執行

[Inside a Decompiler](https://aiengineeringfromscratch.com/inside-a-decompiler) 這台機器把這堂課當作它的前兩個站點來執行。捲軸讀取持有 `percent_change` 的 object file 的 header，而櫃子打開 `.text` 並讀取 symbol。關掉 Symbols，就能看到 symbol table 攜帶了哪些事實。

## 課程成品 (Lesson Artifact)

這堂課會產出 `outputs/skill-program-anatomy-elf-from-scratch.md`。可搭配 Claude、Cursor、Codex、OpenClaw、Hermes，或任何讀取 SKILL.md 的 agent 使用。這個 skill 會把使用者建構的編譯後 object 或可執行檔，變成版面配置報告。報告給出每個結構、symbol 與 relocation 的檔案範圍，並與擷取到的 tool 輸出核對過。

## 驗證 (Verify It)

執行 demo 與確定性的測試套件：

```bash
cd phases/13-tools-and-protocols/32-program-anatomy-elf-from-scratch
python3 code/main.py
python3 -m unittest discover -s code/tests -v
```

測試套件證明：

- 三個 x86-64 object 與 tiny 可執行檔的每個 section index、名稱、大小、位址與種類，都與擷取到的 `llvm-objdump -h` 文字相符
- 每個 symbol 的值、binding、type、section、大小與名稱，都與擷取到的 `llvm-objdump -t` 文字相符
- 每個 relocation 的 offset、型別名稱、symbol 與 addend，都與擷取到的 `llvm-objdump -r` 文字相符，包括 -O0 的 `R_X86_64_PLT32` 和 `R_X86_64_64`
- `tiny-x86_64` 的兩個 program header 與擷取到的 `llvm-objdump -p` 文字相符，而 object 沒有 program header
- `percent_change` 是檔案 offset `0x40` 的 16 位元組，具有預期的位元組，而且每個函式都位於 `.text` 之內
- 檔案 offset 地圖涵蓋每個 fixture 的每個位元組恰好一次，-O1 object 的三個對齊空隙也在各自的位置
- `tiny-x86_64` 的進入點透過 `PT_LOAD` 對應到檔案 offset `0xb0`，而可重定位的 object 拒絕位址對應
- Mach-O 的 header、每個 load command、segment、五個 section、`LC_SYMTAB` 與 `LC_BUILD_VERSION`，都與擷取到的 `--private-headers` 文字相符
- Mach-O 的 symbol 與 15 筆 relocation 與擷取到的 `llvm-objdump` 文字相符，而 `_percent_change` 是 offset 648 的 24 位元組
- 錯誤的 magic、32 位元 class、big-endian 檔案、被截斷的表、universal binary，以及 C 原始檔，都會丟出 `FormatError`
- `build_tiny_elf.py` 逐位元組重現已 commit 的 `tiny-x86_64`
- 圖表模組 `site/figures-re-32.js` 帶有 `ledger-x86_64-O1.o` 的確切位元組

## Production 的失敗模式 (Production Failure Modes)

| 失敗 | 薄弱的解讀回報什麼 | Reader 必須證明什麼 |
|---|---|---|
| 把 `st_value` 當作檔案 offset | 「`percent_change` 在 0」 | Offset 是 `sh_offset + st_value - sh_addr`，這裡是 `0x40` |
| 把 padding 算成程式碼 | 「`grade` 是 48 位元組」 | `st_size` 是 46，而它後面的 `66 90` 不屬於任何 symbol |
| 從錯誤的表讀取 section 名稱 | 亂碼的名稱或例外 | `e_shstrndx` 指名該表，這裡是 section 1 |
| 對 `SHT_NOBITS` 信任 `sh_size` | 讀過檔案結尾 | `.bss` 沒有檔案位元組，地圖會跳過它 |
| 把零位元組讀成 operand | 「`lea 0(%rip)`」 | 位於 `0x36` 的 `.rela.text` 紀錄指名 `.rodata+0x10` |
| 把 addend 讀成目標 offset | 「表格從 `0x10` 開始」 | 加上指令結尾的 `S + A - P` 得到 `0x14` |
| 在被 strip 的可執行檔上走訪 section | 「檔案沒有程式碼」 | `PT_LOAD` 對應 loader 使用的每個位址 |
| 把連結後的位址當成 runtime 位址 | 在隨機基底下得到錯誤的 offset | 來自 `PT_LOAD` 的檔案 offset 是穩定的座標 |
| 只用 `sectname` 查找 Mach-O section | 兩個 `__const` section 合成一個 | `segname,sectname` 才是名稱 |
| 預期 `nlist_64` 有 size 欄位 | 例外或零位元組 | 同一個 section 中的下一個 symbol 結束該函式 |
| 把 universal binary 讀成單一 Mach-O | 「magic `0xcafebabe` 是未知的」 | 拒絕它，並要求提供瘦身後的 slice |
| 在另一台 Mac 上重建 arm64 object | 「fixture 改變了」 | `LC_BUILD_VERSION` 記錄 SDK，所以已 commit 的位元組就是 fixture |

## 營運規則 (Operational Rule)

讀取你能信任的 header，跟隨它所持有的 offset，並從定義它的規格為每個欄位命名。把你比對所用的 tool 輸出保留為已 commit 的 fixture，讓比較在沒有該 tool 的情況下也能執行。當某個值在規格中沒有名稱時，就印出數字。能逐位元組檢查的 reader，正是之後每一堂課所依賴的基礎。

## 延伸閱讀 (Further Reading)

- [System V gABI, chapter 4: Object Files](https://www.sco.com/developers/gabi/latest/ch4.intro.html) 以及 [ELF Header](https://www.sco.com/developers/gabi/latest/ch4.eheader.html)、[Sections](https://www.sco.com/developers/gabi/latest/ch4.sheader.html)、[Symbol Table](https://www.sco.com/developers/gabi/latest/ch4.symtab.html) 和 [Relocation](https://www.sco.com/developers/gabi/latest/ch4.reloc.html) 頁面
- [System V gABI, chapter 5: Program Header](https://www.sco.com/developers/gabi/latest/ch5.pheader.html) 與 [Dynamic Linking](https://www.sco.com/developers/gabi/latest/ch5.dynamic.html)
- [System V ABI, AMD64 Architecture Processor Supplement](https://gitlab.com/x86-psABIs/x86-64-ABI)，Table 4.9 Relocation Types
- [Apple XNU `loader.h`](https://github.com/apple-oss-distributions/xnu/blob/main/EXTERNAL_HEADERS/mach-o/loader.h)、[`nlist.h`](https://github.com/apple-oss-distributions/xnu/blob/main/EXTERNAL_HEADERS/mach-o/nlist.h) 和 [`reloc.h`](https://github.com/apple-oss-distributions/xnu/blob/main/EXTERNAL_HEADERS/mach-o/reloc.h)
- [17 U.S.C. 1201](https://uscode.house.gov/view.xhtml?req=granuleid:USC-prelim-title17-section1201&num=0&edition=prelim) 與 [Directive 2009/24/EC](https://eur-lex.europa.eu/eli/dir/2009/24/oj/eng)
- [Sega v. Accolade, 977 F.2d 1510](https://law.resource.org/pub/us/case/reporter/F2/977/977.F2d.1510.92-15655.html) 與 [Sony v. Connectix, 203 F.3d 596](https://law.resource.org/pub/us/case/reporter/F3/203/203.F3d.596.99-15852.html)
- [rea 6.5.0 `docs/binary-diagnostics.md`](https://github.com/morluto/rea/blob/rea-agents-6.5.0/docs/binary-diagnostics.md) 與 [`docs/ghidra-provider.md`](https://github.com/morluto/rea/blob/rea-agents-6.5.0/docs/ghidra-provider.md)
