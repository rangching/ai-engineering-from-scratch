# 連結、載入與被 strip 的 Binary：PLT、GOT、Import、Fixup 與函式識別 (Linking, Loading, and Stripped Binaries: PLT, GOT, Imports, Fixups, and Function Identification)

> 連結後的程式會呼叫它自己沒有包含的函式。這堂課閱讀 link editor 為這些呼叫在三種格式中寫下的表格，跟隨一次 call 到達 loader 填入的 slot，並在名稱消失之後找出 build 中的函式。

**Type:** Build
**Languages:** Python, C, x86-64 and arm64 assembly
**Prerequisites:** Phase 13 · 32 (ELF from scratch), 33 (disassembly and control flow), 38 (calling conventions)
**Time:** ~160 分鐘

## 學習目標 (Learning Objectives)

- 從 relocation 紀錄與 symbol table 判斷 link editor 會把它變成什麼：固定的距離、帶有 GOT slot 的 PLT 項目，或是動態 relocation。
- 讀取與位置無關的 ELF 可執行檔的 dynamic section、PLT 與 GOT，並用 psABI 的公式跟隨一次 call 通過它們。
- 讀取 COFF object（20 位元組的檔案 header、40 位元組的 section 項目、18 位元組的 symbol、10 位元組的 relocation），以及 PE32+ image 的 import、export 與 base relocation 目錄。
- 逐 word 解碼 Mach-O 的 chained fixup 鏈，並讓每個 word 都與 `dyld_info` 相符。
- 說明 `strip` 在每種格式中移除了什麼、留下什麼，並執行被 strip 的程式以證明它仍然能運作。
- 用 `LC_FUNCTION_STARTS`、prologue 掃描與 call 目標，還原被 strip 的 arm64 build 的函式起點，對照未 strip 的孿生版本的 symbol table 為每種方法評分，並用 FLIRT 形狀的 signature 為函式命名。

## 問題 (The Problem)

你把 `main.c` 與 `libledger.dylib` 連結在一起，並問 agent，`main` 呼叫的 `0x1000005a0` 是哪個函式。Agent 反組譯了該指令並讀到 `bl 0x100000694`。那個位址持有十二個位元組：`adrp x16`、`ldr x16, [x16, #0x38]`、`br x16`。Agent 跟隨 load 到 `0x100004038`，讀取八個位元組 `00 00 00 00 00 00 10 80`，並回報這是透過損壞的指標呼叫到 kernel 空間。

接著你 strip 同一個程式的靜態 build，並問它有多少個函式。Agent 執行了 `nm`，看到一個名稱 `__mh_execute_header`，於是回答一個。

兩個答案都把檔案當成執行中的 process 來讀。`0x100004038` 的 word 是 chained fixup。位元 63 說明 bind，十二個位元說明下一個 fixup 在哪裡，而低位元說明要綁定哪個 import。dyld 在 `main` 執行之前，把這個 word 換成 `_puts` 的位址。被 strip 的檔案失去了 symbol table，卻保留了 `LC_FUNCTION_STARTS`，它用九個位元組列出全部五個函式起點。

這堂課以標準函式庫的 Python 建構五個 reader。一個用於我們自己 object 的 COFF reader，以及一個用於 import、export 與 base relocation 目錄的 PE32+ reader。一個用於 load command、stub、indirect symbol table、chained fixup 鏈與 exports trie 的 Mach-O reader，並逐 word 對照 `dyld_info` 檢查。一個用於 dynamic section、PLT 與 GOT 的 ELF reader，加上一個說明對我們的 object 做 PIE 連結會建立什麼的 planner。一個針對被 strip 的 build、有四種方法且每種各有分數的函式尋找器。以及一個 FLIRT 形狀的 signature 建構器，具有遮蔽 relocation 位元組，以及尾端的 CRC16。

這台機器上只有 Mach-O 能連結。arm64 的可執行檔與 dylib 已被連結、strip 並執行過。ELF 與 COFF 的 object 在這裡用 `clang -target` 編譯，沒有安裝 ELF 或 PE 的 link editor，所以那兩種格式的兩個連結後 image，是由 fixtures 資料夾中的兩個 Python 腳本，依規格逐位元組寫成的。`llvm-objdump` 讀取它們，函式庫探測也一樣。它們沒有被執行過，而文字在重要之處這麼說。

```figure
re-40-call-through-three-formats
```

## Link editor 如何處理 relocation (What the Link Editor Does With a Relocation)

課程 32 讀取 relocation 紀錄，而課程 33 對 `lea` 套用 `S + A - P`。紀錄指名位置 `P`、型別、symbol `S` 和 addend `A`。x86-64 psABI 的 Table 4.9 為每種型別給出一個公式，其中 `L` 是 symbol 的 PLT 項目位置，`G` 是它的 GOT offset，`GOT` 是 GOT 位址，而 `B` 是 image 的載入 base。三種結果涵蓋了我們的 object。

Link editor 把值寫進欄位，紀錄就消失了。針對輸入中所定義 symbol 的 `R_X86_64_PC32` 就是這種情況。同一個 image 的兩個位元組之間的距離，無論載入位址為何都不會改變。

Link editor 建立 PLT 項目與 GOT slot。針對未定義函式的 `R_X86_64_PLT32` 就是這種情況。Call 得到 `L + A - P`，也就是到該項目的距離。該項目透過 slot 跳轉。`.rela.plt` 中的 `R_X86_64_JUMP_SLOT` 紀錄，告訴 dynamic linker 去填入 slot。

Link editor 為 loader 保留一個 relocation。`R_X86_64_64` 是八位元組的絕對位址，而與位置無關的可執行檔在載入之前並不知道自己的位址。該紀錄變成 `.rela.dyn` 中公式為 `B + A` 的 `R_X86_64_RELATIVE`。

`main-x86_64-O1.o` 有 18 筆紀錄：`.text` 中 14 筆、`.data.rel.ro` 中 2 筆、`.rodata` 中 2 筆。`code/main.py` 中的 `plan_dynamic_link()` 對它們以及 `ledger-x86_64-O1.o` 套用這三條規則：

| 紀錄 | 數量 | 決定 | 原因 |
|---|---|---|---|
| 指向 `.rodata`、`.L.str`、`fns`、`.rodata` 中表格的 `R_X86_64_PC32` | 16 | 原地 | symbol 是在輸入中定義的 |
| 指向 `percent_change`、`grade`、`total`、`grade_label` 的 `R_X86_64_PLT32` | 4 | 直接，沒有 PLT | 定義在 `ledger-x86_64-O1.o` 中，所以 link editor 把 PLT32 化簡為 PC32 |
| 指向 `puts` 兩次和 `printf` 兩次的 `R_X86_64_PLT32` | 4 | PLT 項目與 GOT slot | 未定義，由共享函式庫提供 |
| `fns[]` 中指向 `grade` 的 `R_X86_64_64` | 2 | `R_X86_64_RELATIVE` | PIE 中的絕對欄位 |

兩個 import 給出 48 位元組的 `.plt`（PLT0 加兩個 16 位元組的項目）、40 位元組的 `.got.plt`（三個保留 slot 加兩個），以及 48 位元組的 `.rela.plt`。兩筆 relative 紀錄給出 48 位元組的 `.rela.dyn`。這些數字來自把規格套用到我們的紀錄。手工建構的 `tiny-dyn-x86_64` 針對相同的兩個 import，實作這個計畫的 PLT 部分，而它的大小與計畫相符。

自 binutils 2.31（H.J. Lu，2018-02-13）與 LLVM 7.0.0（D43383，2018-02-20）以來，編譯器對每個 call，不論已定義或未定義，都寫 `PLT32`。由 link editor 決定。因此 object 中的 `PLT32` 表示「一個 call」，並沒有說明 PLT 項目是否會存在。在其他 build 中還會出現兩種型別。`R_X86_64_GOTPCRELX` 和 `R_X86_64_REX_GOTPCRELX`（`G + GOT + A - P`）要求不帶 PLT 的 GOT slot，這是 `-fno-plt` 產生的，而當 symbol 是 local 時，link editor 可以把它放寬為直接的 `lea` 或 `call`（psABI 附錄 B）。`R_X86_64_32` 和 `R_X86_64_32S` 在 PIE 中放不下位址，連結會失敗。這兩種都沒有出現在我們的兩個 object 中。

## Dynamic section、PLT 與 GOT (The Dynamic Section, the PLT, and the GOT)

gABI 第 5 章說，參與動態連結的可執行檔有一個 `PT_INTERP` program header（必須在任何可載入 segment 之前），以及一個涵蓋 `.dynamic` section 的 `PT_DYNAMIC`。該 section 是 `Elf64_Dyn` 配對的陣列，每一對是有號的 tag 與一個值，以 `DT_NULL` 結尾。

`tiny-dyn-x86_64` 是 5,184 位元組，由 `build_tiny_dyn_elf.py` 依 gABI 和 psABI 寫成。它的 dynamic section 在 `0x1000` 持有十二對：

| Tag | 編號 | 值 | 意義 |
|---|---|---|---|
| `DT_NEEDED` | 1 | `libc.so.6` | string table offset `0xb`，要載入的函式庫 |
| `DT_HASH` | 4 | `0x1a0` | SysV hash table |
| `DT_STRTAB` | 5 | `0x188` | `.dynstr` |
| `DT_SYMTAB` | 6 | `0x140` | `.dynsym`，每個項目 24 位元組 |
| `DT_STRSZ` | 10 | `0x15` | 21 位元組的字串 |
| `DT_SYMENT` | 11 | `0x18` | 24 |
| `DT_PLTGOT` | 3 | `0x10c0` | `.got.plt` |
| `DT_PLTRELSZ` | 2 | `0x30` | 兩筆 24 位元組的紀錄 |
| `DT_PLTREL` | 20 | 7 | 紀錄是 `DT_RELA` |
| `DT_JMPREL` | 23 | `0x1b8` | `.rela.plt` |
| `DT_FLAGS_1` | `0x6ffffffb` | `0x08000000` | `DF_1_PIE`，GNU tag |
| `DT_NULL` | 0 | 0 | 結束 |

gABI 說明了為什麼 `DT_JMPREL` 是獨立的：它「持有僅與 procedure linkage table 相關聯的 relocation 項目的位址。在啟用 lazy binding 的情況下，把這些 relocation 項目分開，讓 dynamic linker 在 process 初始化期間可以忽略它們。」`DT_FLAGS` 中的 `DF_BIND_NOW`、`DT_FLAGS_1` 中的 `DF_1_NOW`，或環境中的 `LD_BIND_NOW`，會關閉 lazy binding，而 dynamic linker 會在啟動時處理它們。

### PLT0 與 puts 的項目

psABI 5.2 節在 Figure 5.2 給出兩種形狀。這是我們檔案的位元組：

```text
     1f0: ff 35 d2 0e 00 00    pushq  0xed2(%rip)      # 0x10c8 = GOT+8
     1f6: ff 25 d4 0e 00 00    jmpq   *0xed4(%rip)     # 0x10d0 = GOT+16
     1fc: 0f 1f 40 00          nopl   (%rax)

     200: ff 25 d2 0e 00 00    jmpq   *0xed2(%rip)     # 0x10d8 = GOT+24, the slot for puts
     206: 68 00 00 00 00       pushq  $0x0             # index 0 into DT_JMPREL
     20b: e9 e0 ff ff ff       jmp    0x1f0            # PLT0
```

算術只做一次。RIP 相對的 displacement 是相對於下一條指令的。`0x1f6 + 0xed2 = 0x10c8`，也就是 `.got.plt + 8`。`0x1fc + 0xed4 = 0x10d0`，也就是 `.got.plt + 16`。`0x206 + 0xed2 = 0x10d8`，第四個 slot。`0x210 - 0x20 = 0x1f0`，PLT0。每個項目 16 位元組，所以 `exit` 的項目從 `0x210` 開始、push `$0x1`，並讀取 `0x10e0`。

GOT 保留三個 slot。psABI 5.2：項目零持有 `_DYNAMIC` 的位址，而「global offset table 中的項目一和二也是保留的」。我們的 `.got.plt` 讀作 `0x1000, 0, 0, 0x206, 0x216`。glibc 在載入檔案時，把 slot 一填入它的 `struct link_map`，把 slot 二填入 `_dl_runtime_resolve`。Slot 三在檔案中持有 `0x206`，也就是 `puts@plt` 內的 `pushq`。psABI：該 slot「起初……持有後續 pushq 指令的位址」。

所以第一次 call 執行 `call 0x200`，`jmpq *0x10d8` 落在 `pushq $0` 上，`jmp 0x1f0` push GOT+8 並透過 GOT+16 跳進 resolver，它讀取 `DT_JMPREL` 的紀錄 0 `R_X86_64_JUMP_SLOT puts`、在 `libc.so.6` 中查找 `puts`、把它的位址寫進 `0x10d8`，並跳到那裡。第二次 call 執行 `call`、`jmpq *0x10d8`，然後抵達。Drepper 寫道，「因此除了第一次 call 之外，所有 call 的開銷都『只有』一次 indirect jump」。psABI 關於 `%r11` 的註腳，解釋了課程 38 中的一個暫存器選擇：它是暫存變數暫存器，所以「PLT 中的程式碼不需要 spill 任何暫存器」。

`.text` 中的 call site 在 `0x227` 讀作 `e8 d4 ff ff ff`，而 `0x22c - 0x2c = 0x200`。`llvm-objdump` 為它印出 `callq 0x200 <puts@plt>`，並從 `.rela.plt` 紀錄為該項目命名，`plt_entries()` 也是這樣命名它的。

### 檔案中沒有的東西 (What the file does not have)

檔案沒有 `PT_GNU_RELRO`。Jakub Jelinek 於 2004-01-06 在 binutils 郵件列表上提出 RELRO：把只在 relocation 期間被寫入的 section 移到可寫 segment 的開頭，讓 dynamic linker 之後可以對它們 `mprotect` 為唯讀。GNU ld 的預設腳本把 `.init_array`、`.fini_array`、`.data.rel.ro`、`.dynamic` 和 `.got` 放在裡面，並只把 `.got.plt` 的三個保留 slot（24 位元組）留在裡面，因為其他 slot 是 lazy 寫入的。那是 partial RELRO。`-z now` 設定 `DF_BIND_NOW` 並把整個 `.got.plt` 移進去，就是 full RELRO，因為啟動後不再寫入任何東西。glibc 的 `_dl_protect_relro` 對對齊到頁面的範圍呼叫 `mprotect(start, end - start, PROT_READ)`。`-fno-plt` 在編譯器端又更進一步：call 透過 `jmpq *name@GOTPCREL(%rip)` 進行，不可能有 lazy binding，而 GCC 手冊說「所有外部 symbol 都在載入時解析」。

檔案也沒有 version section。對 `libc.so.6` 的真正連結，會加入 `.gnu.version`（每個 `.dynsym` 項目一個 `Elfxx_Half`）和 `.gnu.version_r`（每個函式庫一個 `Verneed`，內含每個 version 名稱的 `Vernaux` 紀錄）。`printf@GLIBC_2.2.5` 的意思是：`printf` 在 `.gnu.version` 中的項目持有一個 index，`vna_other` 等於該 index 的 `Vernaux` 的 `vna_name` 是「GLIBC_2.2.5」，位於 `vn_file` 為「libc.so.6」的 `Verneed` 之下。Index 0 是 local，1 是 global，而位元 15（`VERSYM_HIDDEN`，`0x8000`）對 link editor 隱藏某個 version。

Hash table 是 SysV 的 `DT_HASH`，有一個 bucket 和三個項目的鏈，共 24 位元組。Dynamic linker 需要 `DT_HASH` 或 `DT_GNU_HASH` 來查找檔案中的 symbol。這個檔案不匯出任何東西，也沒有在這裡執行，所以 glibc 是否接受一個 bucket 的表格，是未驗證的。

### .dynsym、.symtab 與 strip

gABI 把 `.dynsym` 和 `.dynstr` 定為 allocated section，把 `.symtab` 和 `.strtab` 定為 unallocated 的。Dynamic linker 讀取前者，從不讀後者。Drepper：「一般的 symbol table 在 runtime 不會被使用，所以沒有必要把它從 binary 中 strip 掉。它對效能沒有影響。」

GNU `strip --strip-all` 移除 `.symtab`、`.strtab`、`.debug_*` 和 `.comment`。每個 allocated section 都會保留：`.interp`、`.dynsym`、`.dynstr`、`.gnu.hash`、`.gnu.version*`、`.rela.dyn`、`.rela.plt`、`.plt`、`.text`、`.rodata`、`.eh_frame`、`.init_array`、`.dynamic`、`.got`、`.got.plt`、`.data`、`.bss` 和 note。Section header 會保留，除非你加上 `--strip-section-headers`（binutils 2.41）或執行 `llvm-strip --strip-sections`。`--strip-debug` 保留 symbol。`--only-keep-debug` 寫出 debug 檔案，而 `--add-gnu-debuglink` 從被 strip 的檔案指名它，並在 `.gnu_debuglink` 中帶有 CRC32，而 `ld --build-id` 寫入 `.note.gnu.build-id`，讓 debugger 可以比對兩者。

## COFF object (The COFF Object)

`clang -target x86_64-pc-windows-msvc -O1 -fno-asynchronous-unwind-tables -c` 產生了 `main-x86_64.obj` 和 `ledger-x86_64.obj`。PE Format 頁面定義了該 object：20 位元組的檔案 header、沒有 optional header、由 40 位元組項目組成的 section table、由 18 位元組紀錄組成的 symbol table，以及每個 section 的 10 位元組 relocation 紀錄。

`code/main.py` 中的 `CoffObject` 讀取全部四者。檔案 header 說明 machine 為 `0x8664`、10 個 section、37 個 symbol 紀錄、`SizeOfOptionalHeader` 為 0。Section table 指名 `.text`、`.data`、`.bss`、六個叫做 `.rdata` 的 section，以及 `/16`。以 `/` 開頭的名稱，是進入 string table 的十進位 offset，而該 string table 以自己的四位元組大小開頭；`/16` 解析為 `.llvm_addrsig`，這是 `llvm-objdump -h` 印出的名稱。

這 37 筆紀錄包含 auxiliary 紀錄。Section symbol（storage class 3）帶有一個，持有 section 長度、relocation 數量、checksum 與 COMDAT 選擇。五個 `.rdata` section 的選擇是 2，「any」，並各持有一個以 MSVC 修飾名稱命名的字串常數：`??_C@_04JDDJEAHN@even?$AA@` 就是字串「even」。Link editor 每個只保留一份。最後一筆紀錄是 `.file`，帶有 auxiliary 紀錄「main.c」。`llvm-objdump -t` 把這些印成 `[ 0]`、`AUX scnlen ...`、`[ 2]` 等，並略過 auxiliary 的 index，而測試將我們的 index 與名稱配對，與那份文字比較。

Relocation：`.text` 中 14 個 `IMAGE_REL_AMD64_REL32`，以及 `.rdata` 中 4 個 `IMAGE_REL_AMD64_ADDR64`。`REL32` 是「相對於 relocation 之後那個位元組的 32 位元相對位址」。那就是內建 `-4` 的 `R_X86_64_PC32`：ELF 紀錄帶有 addend `-4`，因為該紀錄有 addend 欄位，而 COFF 紀錄沒有。`ledger-x86_64.obj` 在 `0x36`、`0x84` 和 `0x93` 有三個 `REL32`，而課程 32 的 `ledger-x86_64-O1.o` 的三個 `R_X86_64_PC32` 在相同的三個 offset。程式碼相同，格式只在四被寫在哪裡有所不同。`ADDR64` 是八位元組的絕對位址，也就是字串指標的 `labels` 表格，並且在 image 中變成 base relocation，就像 `R_X86_64_64` 變成 `R_X86_64_RELATIVE`。

| | ELF `Elf64_Rela` | COFF relocation | Mach-O `relocation_info` |
|---|---|---|---|
| 紀錄大小 | 24 位元組 | 10 位元組 | 8 位元組 |
| 欄位 | offset、`r_info` 中的型別與 symbol、addend | offset、symbol index、型別 | address，然後是 symbolnum:24、pcrel:1、length:2、extern:1、type:4 的 32 位元 word |
| addend | 明確 | 無，存在欄位中 | 無，或真正紀錄之前的 `ARM64_RELOC_ADDEND` 紀錄 |
| 對 `puts` 的 call | `R_X86_64_PLT32 puts-0x4` | `IMAGE_REL_AMD64_REL32 puts` | `ARM64_RELOC_BRANCH26 _puts` |

## PE image：Import、Export 與 Base Relocation (The PE Image: Imports, Exports, and Base Relocations)

`tiny-x86_64.exe` 是 2,560 位元組，由 `build_tiny_pe.py` 寫成。它與 tiny ELF 的工作相同：從 `msvcrt.dll` 呼叫 `puts` 和 `exit`。它也匯出一個名稱並帶有一個 base relocation，所以這堂課需要的三個目錄都存在。`pefile` 2024.8.26 和 `llvm-objdump -p` 都讀取它，而兩份擷取都是 fixture。

走過 header。MS-DOS stub 以 `MZ` 開頭，並在 offset `0x3c` 持有 `e_lfanew`，即 `0x40`。在 `0x40` 是簽章 `PE\0\0`。然後是 COFF 檔案 header：machine `0x8664`、4 個 section、`SizeOfOptionalHeader` 240、characteristics `0x22`（可執行 image，large address aware）。Optional header 以 `Magic` 欄位 `0x20b` 開頭，代表 PE32+。`AddressOfEntryPoint` 是 `0x1000`，`ImageBase` 是 `0x140000000`，`SectionAlignment` 是 `0x1000`，`FileAlignment` 是 `0x200`，`SizeOfImage` 是 `0x5000`，`Subsystem` 是 3（console）。`DllCharacteristics` `0x8160` 解碼為 `HIGH_ENTROPY_VA 0x0020`、`DYNAMIC_BASE 0x0040`、`NX_COMPAT 0x0100` 和 `TERMINAL_SERVER_AWARE 0x8000`。接著是 16 個 RVA 與大小的 data directory。其中四個有設定：Export 在 `0x2080`，大小 `0x4d`；Import 在 `0x2000`，大小 `0x28`；Base Relocation 在 `0x4000`，大小 `0xc`；IAT 在 `0x2040`，大小 `0x18`。

Image 中的位址是 RVA，也就是相對於載入 base 的 offset。Section table 把它們對應到檔案 offset：`.text` RVA `0x1000` 在檔案 `0x200`，`.rdata` RVA `0x2000` 在 `0x400`，`.data` RVA `0x3000` 在 `0x600`，`.reloc` RVA `0x4000` 在 `0x800`。`PeImage.rva_to_offset()` 就是那張表。

### Import 目錄

在 RVA `0x2000` 是以零 descriptor 結尾的 20 位元組 import descriptor 陣列。我們的有一個：Import Lookup Table `0x2028`、timestamp 0、forwarder chain 0、name `0x2068`、Import Address Table `0x2040`。Name RVA 指向「msvcrt.dll」。Lookup table 持有以零結尾的 64 位元項目。位元 63 設定表示依 ordinal import，ordinal 在位元 15 到 0。位元 63 清除表示位元 30 到 0 是 hint/name 項目的 RVA：16 位元的 hint（「進入 export name pointer table 的 index」），接著是 ASCII 名稱。我們的是 `0x2058`（hint 0，「puts」）和 `0x2060`（hint 0，「exit」）。

規格說，IAT「在 image 被 bind 之前，與 import lookup table 的內容相同」。我們位於 `0x2040` 的 IAT 持有相同的兩個 RVA，而測試檢查 `0x2028` 的 24 位元組等於 `0x2040` 的 24 位元組。「在 binding 期間，import address table 中的項目會被覆寫為所 import 之 symbol 的 32 位元（PE32）或 64 位元（PE32+）位址。」Windows loader 在進入點執行之前做這件事。沒有 lazy binding，也沒有 stub。程式碼直接到達 slot：

```text
    100b: ff 15 2f 10 00 00    call qword ptr [rip + 0x102f]   # 0x1011 + 0x102f = 0x2040, the IAT slot of puts
    1013: ff 15 2f 10 00 00    call qword ptr [rip + 0x102f]   # 0x1019 + 0x102f = 0x2048, the IAT slot of exit
```

`FF /2` 是 indirect near call。這兩條指令有相同的位元組，因為每一條都位於另一條之後八個位元組，並讀取另一個 slot 之後八個位元組的 slot。`iat_calls()` 掃描程式碼 section 中的 `ff 15`，加上六與 displacement，並從 import 目錄為 slot 命名。

### Export 目錄

在 RVA `0x2080` 是 40 位元組的 export directory table：flags、timestamp、version、指向「tiny-x86_64.exe」的 name RVA、ordinal base 1、1 個 address table 項目、1 個 name pointer。後面跟著三個表格。位於 `0x20a8` 的 Export Address Table 持有一個 RVA `0x1000`。位於 `0x20ac` 的 Export Name Pointer Table 持有指向字串「tiny_entry」的一個 RVA。位於 `0x20b0` 的 Export Ordinal Table 持有一個 16 位元無 bias 的 index 0。名稱 `k` 對應到 address 項目 `ordinals[k]`，而匯出的 ordinal 是 `ordinals[k] + ordinal base`。規格補充了一條我們不使用的規則：指向 export section 內部的 address 項目是 forwarder 字串，例如 `MYDLL.expfunc`，而 name pointer table「以字典序排列，以便二分搜尋」。

### Base relocation

`.reloc` 持有多個 block。每個 block 以頁面 RVA 和 block 大小開頭，然後是 16 位元項目，型別在高四位元，頁面 offset 在低十二位元。我們唯一的 block：頁面 `0x3000`、大小 12、項目 `0xa000`、型別 10 `IMAGE_REL_BASED_DIR64`、offset 0，以及型別 0 `IMAGE_REL_BASED_ABSOLUTE` 的 padding 項目。RVA `0x3000` 的八個位元組持有 `0x1400020d0`，也就是 image base 加上字串「linked by hand」的 RVA。如果 loader 把 image 放在別的位置，它會把差值加到那個欄位。規格：「Image 檔案不包含 COFF relocation ……image 以 base relocation 的形式在 .reloc section 中包含 relocation 資訊（除非 image 有 IMAGE_FILE_RELOCS_STRIPPED 屬性）。」目前的 link.exe 預設啟用 `/DYNAMICBASE`，所以 image 帶有 `.reloc` 與 `DYNAMIC_BASE`。

PE image 沒有可 strip 的 symbol table。檔案 header 的 `PointerToSymbolTable`「對 image 而言應為零，因為 COFF 偵錯資訊已被棄用」，而名稱存在於另一個 PDB 中。每個 image 中都留下來的，是 import 目錄、export 目錄，以及對 64 位元程式碼而言的 `.pdata`，其 `RUNTIME_FUNCTION` 項目指名每個配置 stack 或呼叫其他函式的函式的起點與終點。Leaf 函式不在其中。

## Mach-O image：Load Command、Stub 與 Indirect Symbol Table (The Mach-O Image: Load Commands, Stubs, and the Indirect Symbol Table)

`ledger_main` 是連結後的 arm64 可執行檔。`otool -h` 印出 19 個 load command 與 flag `0x00200085`。`MH_NOUNDEFS 0x1`、`MH_DYLDLINK 0x4`、`MH_TWOLEVEL 0x80` 和 `MH_PIE 0x200000` 加總為該值。這 19 個 command 是四個 `LC_SEGMENT_64`（`__PAGEZERO`、`__TEXT`、`__DATA_CONST`、`__LINKEDIT`）、兩個 `LC_LOAD_DYLIB`（先是 `@rpath/libledger.dylib`，然後是 `/usr/lib/libSystem.B.dylib`），以及各一個 `LC_DYLD_CHAINED_FIXUPS`、`LC_DYLD_EXPORTS_TRIE`、`LC_SYMTAB`、`LC_DYSYMTAB`、`LC_LOAD_DYLINKER`、`LC_UUID`、`LC_BUILD_VERSION`、`LC_SOURCE_VERSION`、`LC_MAIN`、`LC_RPATH`、`LC_FUNCTION_STARTS`、`LC_DATA_IN_CODE` 和 `LC_CODE_SIGNATURE`。`MachoFile.command_names()` 以與 `otool -l` 相同的順序回傳相同的清單。

`__TEXT` 從 `0x100000000` 開始。`__text` 是位於 `0x100000558` 的 256 位元組，是 `main` 且沒有別的東西，因為四個 ledger 函式位於 dylib 中。`__stubs` 是位於 `0x100000658` 的 84 位元組，七個各十二位元組的 stub。`__DATA_CONST` 從 `0x100004000` 開始，包含 64 位元組的 `__got`（八個 section 類型為 `S_NON_LAZY_SYMBOL_POINTERS` 的指標 slot），以及 32 位元組的 `__const`（`main.c` 的 `fns[]` 和 `names[]` 表格）。

### 手動看一個 stub

`_puts` 的 stub 位於 `0x100000694`：

```text
    30 00 00 90    adrp x16, 0x100004000
    10 1e 40 f9    ldr  x16, [x16, #0x38]
    00 02 1f d6    br   x16
```

`0x90000030` 是 `adrp`。暫存器是低五個位元，16。`immlo` 是位元 30 到 29，0。`immhi` 是位元 23 到 5，1。值是 `immhi << 2 | immlo = 4`，以 4,096 位元組的頁面為單位，所以目標頁面是指令所在的頁面 `0x100000000` 加上 `0x4000`。`0xf9401e10` 是帶無號 12 位元 offset、以 8 為 scale 的 `ldr`：位元 21 到 10 是 7，乘以 8 是 `0x38`，所以 load 讀取 `0x100004038`。`0xd61f0200` 是 `br x16`。ld64 原始碼中的 arm64 stub 就是這個確切的三條指令 `StubAtom`。

這個 slot 屬於哪個名稱，來自 indirect symbol table。`LC_DYSYMTAB` 指向進入 symbol table 的 15 個 index。Stubs section 在 `reserved1` 說明它的 index 序列從哪裡開始，並在 `reserved2` 說明每個 stub 多大（12）。指標 section 使用 `reserved1` 和每個 slot 八個位元組。`__stubs` 從 index 0 開始，而 `__got` 從 index 7 開始，所以 stub 5 對應到 symbol table 項目 10 `_puts`，而 `__got` 的 slot 7 對應到同一個項目。`otool -Iv` 印出兩個序列，而 `indirect_table()` 重現每一列。

### Two-level namespace

`nm -m` 印出 `(undefined) external _puts (from libSystem)` 和 `(undefined) external _grade (from libledger)`。函式庫來自 symbol 紀錄本身。`nlist.h` 把 `GET_LIBRARY_ORDINAL(n_desc)` 定義為 `n_desc` 的位元 15 到 8：第一個 `LC_LOAD_DYLIB` 為 1、第二個為 2、動態查找為 `0xfe`、可執行檔為 `0xff`。`ld(1)`：「使用 -two_levelnamespace（預設）時，linker 只在命令列上的 dylib 中搜尋 symbol，並記錄是在哪個 dylib 中找到的。」`nm_lines()` 從 `n_type`、`n_sect`、`n_desc` 和 `n_value` 轉譯這十二筆紀錄，而測試逐行將它的輸出與擷取的 `nm -m` 比較。用 `-Wl,-flat_namespace` 重新連結並閱讀 `otool -h`，就能看到 `MH_TWOLEVEL` 位元消失。

## 逐位元組看 Chained Fixup (Chained Fixups, Byte for Byte)

`LC_DYLD_CHAINED_FIXUPS` 是 `linkedit_data_command`：`dataoff 0x8000`、`datasize 208`。這 208 個位元組遵循 `fixup-chains.h`。

Header 是七個 32 位元欄位：`fixups_version 0`、`starts_offset 32`、`imports_offset 80`、`symbols_offset 112`、`imports_count 8`、`imports_format 1`（`DYLD_CHAINED_IMPORT`）、`symbols_format 0`（未壓縮）。`llvm-objdump --macho --chained-fixups` 印出這七個值，而測試比較它們。

在 offset 32 是 `dyld_chained_starts_in_image`：`seg_count 4` 和四個 offset，每個 segment 一個，沒有 fixup 的 segment 為零。只有 `__DATA_CONST` 有。它的 `dyld_chained_starts_in_segment` 讀作 `size 24`、`page_size 0x4000`、`pointer_format 6`（`DYLD_CHAINED_PTR_64_OFFSET`，「目標是 vm offset」）、`segment_offset 0x4000`、`max_valid_pointer 0`、`page_count 1`，以及一個 0 的 `page_start`。`DYLD_CHAINED_PTR_START_NONE`（`0xffff`）則表示沒有 fixup 的頁面。

鏈從檔案 offset `0x4000 + 0` 開始。每個 64 位元 word 帶有兩樣東西：要寫在那裡的內容，以及下一個 word 在哪裡。

```text
  dyld_chained_ptr_64_bind    ordinal:24  addend:8  reserved:19  next:12  bind:1 = 1
  dyld_chained_ptr_64_rebase  target:36   high8:8   reserved:7   next:12  bind:1 = 0
```

閱讀第一個 word `0x8010000000000000`。位元 63 已設定，是 bind。`next` 是位元 62 到 51，`0x002`，以 4 位元組為單位，所以下一個 fixup 在 8 位元組之後。Ordinal 是 0。閱讀位於 `0x4058` 的第十二個 word `0x00000000000006db`。位元 63 清除，是 rebase。`next` 是 0，鏈的結尾。目標是 `0x6db`，是 vm offset，而 image base `0x100000000` 使它成為 `0x1000006db`，也就是 `__cstring` 中的字串「odd」。它之前的 word `0x00100000000006d6` 把 `names[0]` rebase 到「even」。

走訪造訪十二個 word：`__got` 中八個 ordinal 為 0 到 7 的 bind、`__const` 中兩個 ordinal 為 0 的 bind（`fns[]` 的兩個項目都指向 `_grade`），以及 `names[]` 的兩個 rebase。`chained_fixups()` 回傳它們，附上檔案 offset 與原始 word，而測試把十二個 `(offset, raw)` 配對與 `dyld_info -fixup_chain_details` 比較，後者對第一個印出 `raw: 0x8010000000000000 bind: (next: 002, bindOrdinal: 0x000000)`。

位於 offset 80 的 imports table 有八個 32 位元項目：`lib_ordinal:8`、`weak_import:1`、`name_offset:23`，名稱位於 offset 112。Ordinal 0 到 3 是來自函式庫 1 的 `_grade`、`_grade_label`、`_percent_change`、`_total`，而 4 到 7 是來自函式庫 2 的 `___stack_chk_fail`、`___stack_chk_guard`、`_printf`、`_puts`。`fixup_rows()` 把兩個表格結合成 `dyld_info -fixups` 印出的行，例如 `__DATA_CONST __got 0x100004038 bind libSystem/_puts`，而測試比較全部十二行。

所以位於 `0x100004038` 的「損壞指標」是 import 7。dyld 在載入時走訪這條鏈一次，並把 `_puts` 的位址寫進 slot。Apple 的 WWDC22 session 是這麼說的：「在 DATA 中的 64 位元指標位置，有些位元包含到下一個 fixup 位置的 offset。同時也打包了一個位元，說明 fixup 是 bind 還是 rebase。」Xcode 13 讓這個格式成為 macOS 12 或 iOS 15 部署目標的預設，而 Xcode 13.3 把它擴展到 macOS 11 與 iOS 13.4。這個格式沒有 lazy binding。每個 slot 都在 `main` 執行之前被 bind。

### 它所取代的版面 (The layout it replaced)

`ledger_main_nochain` 是用 `-Wl,-no_fixup_chains` 做的相同連結。它改用 `LC_DYLD_INFO_ONLY`，有四個位元組編碼的表格：rebase 8 位元組、bind 72、lazy bind 120、export 64。它也有三個 chained 檔案沒有的 section：`__TEXT,__stub_helper`（108 位元組）、只有 16 位元組、持有 `___stack_chk_guard` 和 `dyld_stub_binder` 的 `__got`，以及 `__DATA,__la_symbol_ptr`（56 位元組，七個 slot）。`dyld_info -fixups` 列出七個 `lazy-bind` 列與七個 rebase，把每個 lazy slot 指向它在 `__stub_helper` 中自己的 trampoline。Trampoline push 一個 index 並跳到 `dyld_stub_binder`，後者解析名稱並寫入 slot。這就是上一節 PLT 與 GOT 的 Mach-O 形狀：`__la_symbol_ptr` 是 `.got.plt`，`__stub_helper` 是 PLT0 和 `pushq $index`，而 `dyld_stub_binder` 是 `_dl_runtime_resolve`。`code/main.py` 回報這個檔案的 offset 與 section，但不解碼 bind opcode；實作練習會要求你做這件事。

### Exports trie

`LC_DYLD_EXPORTS_TRIE` 是前綴樹。每個節點是一個 uleb128 的 terminal 大小，然後（如果它非零）是 uleb128 的 flags 與 uleb128 的 offset，然後是一個位元組的子節點數量，以及對每個子節點而言，一條字串邊與一個 uleb128 的節點 offset。`exports_trie()` 走訪它。在 `libledger.dylib` 上，它回傳 `_percent_change 0x410`、`_grade 0x428`、`_total 0x464`、`_grade_label 0x498`，也就是 `dyld_info -exports` 印出的四個 offset。在可執行檔上，它回傳 `__mh_execute_header`、`_fns` 和 `_main`。

## strip 留下什麼 (What strip Leaves)

Apple 的 `strip(1)` 說，在動態連結的可執行檔上不帶選項時，其行為類似 `-u -r`：保存未定義的 symbol 與被動態參照的 symbol。`-x` 移除 local symbol。`-N` 移除整個 symbol table 與 string table。「對動態共享函式庫而言，最高程度的 strip 通常是 -x」，因為「動態連結過程只使用 global symbol」。

在我們的 fixture 上的觀察。`strip ledger_main` 讓 `LC_SYMTAB` 從 12 筆紀錄變成 10 筆，並移除了 `_main`、`_fns` 和 `_names`。八個未定義的 import 留下，因為 dyld 需要它們的名稱來 bind，而 `__mh_execute_header` 留下，因為它是 `REFERENCED_DYNAMICALLY`。第十筆紀錄是名為 `radr://5614542` 的 `N_OPT` 型別 stab，這是 Apple 的工具寫入的，使表格永遠不會是空的；`nm` 不會印出它。被 strip 的程式印出與原本相同的三行並以 0 結束。`LC_FUNCTION_STARTS` 仍然有一個起點，`main`。

不帶選項的 `strip libledger.dylib` 比 man page 通常的 `-x` 走得更遠：`LC_SYMTAB` 有 0 筆紀錄，而 `nm` 印出「no symbols」。Exports trie 仍然在相同的 offset 列出四個函式，因為 dyld 是透過 trie 而不是 `LC_SYMTAB` 來 bind。`LC_FUNCTION_STARTS` 仍然列出四個起點。實驗 C 把全新的 `main.c` 與被 strip 的 dylib 連結，顯示 link editor 同樣由 trie 提供服務。

| 什麼 | `--strip-all` 之後的 ELF | PE image | `strip` 之後的 Mach-O |
|---|---|---|---|
| 靜態 symbol table | 消失 | image 中從未存在 | local 與已定義的名稱消失，未定義的 import 留下 |
| loader 要 bind 的名稱 | `.dynsym` 與 `.dynstr` 留下 | import 目錄留下 | import 留在 `LC_SYMTAB` 與 chained imports table 中 |
| 匯出的名稱 | `.dynsym` 留下 | export 目錄留下 | exports trie 留下 |
| 函式起點 | `.eh_frame` 的 FDE，除非建構時不帶 unwind table | 非 leaf 函式的 `.pdata` | `LC_FUNCTION_STARTS` 留下 |
| 偵錯資訊 | `.debug_*` 消失 | 在 PDB 中，從不在 image 中 | 在 dSYM 中，從不在 image 中 |
| section 或 load command header | 留下，除非 `--strip-section-headers` | 留下 | 留下 |

## 在沒有 symbol 的情況下尋找函式 (Finding Functions Without Symbols)

`static-arm64-O2` 把 `main-arm64-O2.o` 和 `ledger-arm64-O2.o` 連結成一個可執行檔，`__text` 中有五個函式。未 strip 檔案的 `nm -n` 是基準真相：`_main` 位於 `0x100000500`（244 位元組）、`_percent_change` 位於 `0x1000005f4`（24）、`_grade` 位於 `0x10000060c`（60）、`_total` 位於 `0x100000648`（248）、`_grade_label` 位於 `0x100000740`（72）。`static-arm64-O2.stripped` 在 `__text` 內沒有 `N_SECT` symbol。`FunctionFinder` 在它上面執行四種方法，而 `score()` 把每份清單與真值比較。

### LC_FUNCTION_STARTS

該 command 指向 `__LINKEDIT` 中的九個位元組：`80 0a f4 01 18 3c f8 01 00`。格式是 uleb128 的 delta，第一個相對於 `__TEXT` 的位址。`80 0a` 是 `0x500`，所以第一個起點是 `0x100000500`。`f4 01` 是 `0xf4`，所以是 `0x1000005f4`。`18` 給出 `0x10000060c`，`3c` 給出 `0x100000648`，`f8 01` 給出 `0x100000740`，而 `00` 結束清單。五個中五個，沒有 false positive，在 `-O0` 與 `-O2` 皆然。Link editor 為 unwinder 和 crash reporter 寫下這個表格，而 `strip` 不碰它。Ghidra 在它的「Mach-O Function Starts」analyzer 中讀取它，`dyld_info -function_starts` 印出它，而 LIEF 把它公開為 `function_starts`。

### Prologue 掃描

Ghidra 的 AARCH64 起點 pattern 以 `fd 7b bf a9` 開頭，也就是 `stp x29, x30, [sp, #-0x10]!`。在我們的檔案上，它在兩個層級都是五個中找到 0 個。clang 先送出 `sub sp, sp, #N`，然後是帶正 offset 的 `stp x29, x30, [sp, #M]`，所以 pre-index 形式從不出現。針對 `sub sp, sp, #imm` 的 mask `(word & 0xff0003ff) == 0xd10003ff`，在 `-O0`（每個函式都有 frame）找到五個中的 5 個，在 `-O2` 找到五個中的 1 個，因為 `percent_change`、`grade`、`total` 和 `grade_label` 是完全沒有 frame 的 leaf 函式。`percent_change` 以 `cbz w0, ...` 開頭。Prologue 掃描回報的是編譯器選擇送出什麼，而課程 38 顯示那常常是什麼都沒有。

這裡兩種掃描都沒有產生 false positive，因為程式很小。像 x86-64 的 `push rbp` 的 `55` 這種單一位元組的 pattern，會命中其他指令的內部，這就是為什麼 Ghidra 的 x86-64 pattern 檔案，把像 `55 48 89 e5` 這樣的 post-pattern，與像 `ret` 或 `nop` 這樣必須結束前一個函式的 pre-pattern 配對。

### Call 目標

`__text` 中的每個 `bl` 都指名一個函式起點，因為編譯器不會為 C 送出進入函式中間的 `bl`。編碼是 `0x94000000` 加上有號 26 位元的 word offset。位於 `0x1000005a0` 的 `bl` 讀作 `3d 00 00 94`：`0x3d` 是 61，乘以 4 是 `0xf4`，所以目標是 `0x100000694`。在 `ledger_main` 中那是 `_puts` 的 stub。在靜態 build 中，`call_targets()` 收集 `__text` 內的目標、丟掉落在 `__stubs` 的那些，並加入 `LC_MAIN` 的進入點，因為沒有人呼叫 `main`。兩個層級都是五個中的五個。Nucleus（Andriesse、Slowinska、Bos，2017）把它對編譯器產生程式碼的函式偵測建立在這個想法上，而 Andriesse 等人在 2016 年量測到，函式邊界是反組譯中困難的部分，比指令邊界更難。

## FLIRT 形狀的 signature (A FLIRT-Shaped Signature)

Hex-Rays 在它的文件中描述 FLIRT：一個 pattern 是「函式的前 32 個位元組，其中所有可變位元組都被標記」，可變位元組來自 fixup 與外部 reference，pattern 存在樹中，並且在 pattern 之外，「我們計算從位置 33 開始、直到第一個可變位元組為止的位元組的 CRC16」並儲存該長度。有相同 pattern 與 CRC 的兩個函式，會在它們不同的位置以尾端位元組區分，然後以它們所參照的名稱區分。無法區分的衝突「由自然智慧解決」。沒有外部 reference 且短於四位元組的函式被排除，而且「signature 檔案不包含原始函式庫的任何位元組，除了函式的名稱」。

`signatures_from_object()` 從 `ledger-arm64-O2.o` 和 `main-arm64-O2.o` 建構它，方式就像 FLAIR 從函式庫的 object 建構一樣。每個 `relocation_info` 紀錄涵蓋一個四位元組的 word，而這些位元組成為可變的。`.pat` 的行版面（記錄在 rizin 的 parser 中）依序是 pattern、CRC 長度、CRC16、函式長度、`:0000` 與名稱，以及每個被參照名稱的 `^offset name`：

```text
A00000342800004B890C8052087D091B000DC01AC0035FD6 00 0000 0018 :0000 _percent_change
E8CC8C52C8CCAC72087C289B09FD7FD308FD62930801090B081900511F110071 04 9AAF 003C :0000 _grade ^0000 l_switch.table.grade ^0001 l_switch.table.grade
E8CC8C52C8CCAC72087C289B09FD7FD308FD62930801090B081900511F110071 04 9AAF 0048 :0000 _grade_label ^0000 l_switch.table.grade_label ^0001 l_switch.table.grade_label ^0002 _labels ^0003 _labels
```

`_percent_change` 是 24 位元組，所以它的 pattern 是整個函式，而 CRC 長度是 0。`_grade` 是 60 位元組，其第一個 relocation 在位元組 36，所以 CRC 涵蓋四個位元組。CRC16 是公開 `.pat` parser 所用的那個：多項式 `0x8408`、初始值 `0xffff`、取補數，然後交換兩個位元組。它對 ASCII 字串「123456789」的值是 `0x6e90`，而測試固定了這點。

`match_signatures()` 在被 strip 的 build 的每個函式起點嘗試每個 signature，先比對 pattern，再比對 CRC。在 `-O0`，五個函式全都得到名稱，且沒有衝突。在 `-O2`，上面兩行發生衝突。`grade_label` inline 了 `grade`，所以兩個函式共用前 36 個位元組與四個 CRC 位元組。兩個 signature 都在 `0x10000060c` 與 `0x100000740` 符合。尾端把它們分開：在位元組 44，`_grade` 是 `20`，而 `_grade_label` 是 `28`，也就是 `ldr w0, [x9, w8, uxtw #2]` 對 `ldr x8, [x9, w8, uxtw #3]`，而長度是 60 對 72。報告記錄了兩個衝突以及決定它們的位元組。在兩個層級都是五個命名了五個。

測試 `test_object_bytes_equal_linked_bytes_outside_relocated_words` 檢查前提：每個 object 函式的每個非可變位元組，都等於連結後可執行檔中相同 offset 的位元組，所以 link editor 只改變了被 relocation 的 word。`_total` 沒有 relocation，也完全沒有可變位元組。

### 工具對相同想法做了什麼

IDA Pro 9.4 為每個處理器提供 FLIRT signature，以及用於建構它們的 `sigmake` 工具；IDA Free 只涵蓋 x86 與 x86-64，並供非商業使用。Lumina 是以函式的 md5 為 key 來存放函式 metadata 的 server，並跳過小於 32 位元組的函式。Ghidra 12.1.4（Apache-2.0）有 Function ID：一個涵蓋 mnemonic 與 addressing mode 的完整 hash，「對連結造成的變更具穩健性」，以及一個加入常數 operand 的特定 hash，依每個符合的指令計分，並以 parent 與 child 函式作為平手時的判斷依據；它內附的資料庫只涵蓋 Visual Studio 函式庫，而你要用自己的 binary 建立自己的。Binary Ninja 6.0 Krypton（商業，附 Free 版）使用 WARP（Apache-2.0）：函式 GUID 是對 basic block GUID 的 UUIDv5，每個 block GUID 又是對把「所有可重定位指令」歸零並丟掉 NOP 之後的 block 位元組的 UUIDv5，並以被呼叫、caller 與相鄰的函式作為限制。BinDiff 8（Apache-2.0）以固定順序的步驟在兩個 binary 之間比對函式，從 hash 比對、flow graph MD index 到 call sequence 比對，並從符合的 edge、block、指令與 flow graph MD index 加權相似度。Diaphora 3.4.3（AGPL-3.0，僅限 IDA）加入小質數乘積，其中每個 mnemonic 對應一個質數，而函式就是乘積，這是與順序無關的 multiset 指紋。radare2 6.2.4（LGPL-3.0）有 zignature（`zg`、`z/`、`zb`），並用 `zfs` 讀取 FLIRT `.sig` 檔案；rizin 0.9.1（LGPL-3.0）用 `Fc` 從開啟的 binary 建立 `.pat` 或 `.sig`，並用 `Fs` 套用它。

IDA 與 Binary Ninja 是閉源的，在這裡根據它們的文件描述。radare2 與 rizin 沒有安裝在這台機器上，它們的指令引用自它們的書與原始碼。在這些 fixture 上執行的工具是 Command Line Tools 的 `clang`、`ld`、`llvm-objdump`、`otool`、`nm`、`strip` 和 `dyld_info`，以及透過探測的 LIEF 1.0.0、pyelftools 0.33 和 pefile 2024.8.26。LIEF 讀取這裡的每種格式，包括它自 0.17.0 起就支援的 COFF object，並在 `ledger_main` 上回報十個 binding 與兩個 rebase，也就是相同的十二個 fixup。pyelftools 只讀取 ELF 檔案。pefile 拒絕 `main-x86_64.obj`，丟出 `PEFormatError: 'DOS Header magic not found.'`，並讀取 tiny image。

## 動手做 (Build It)

執行 demo：

```bash
cd phases/13-tools-and-protocols/40-linking-loading-and-stripped-binaries
python3 code/main.py
```

輸出是一份有五個部分的 JSON 報告。`elf` 持有兩個 object 的 relocation 數量、連結計畫，以及 `tiny-dyn-x86_64` 的 dynamic section、PLT 和 GOT。`coff` 持有兩個 object 的 section、relocation、函式與未定義名稱。`pe` 持有 `tiny-x86_64.exe` 的 header、目錄、import、export、base relocation 與 IAT call。`macho` 持有 `ledger_main` 及其 stub、chained fixup、fixup 列與 export、`ledger_main_nochain` 的舊版版面、dylib，以及兩份 strip 報告。`identification` 持有真值、有分數的四種方法，以及 `-O0` 與 `-O2` 下帶有衝突的 signature。

預期的形狀：

```json
{
  "elf": {"linkPlan": {"pltSymbols": ["puts", "printf"], "sizes": {".plt": 48, ".got.plt": 40, ".rela.plt": 48, ".rela.dyn": 48}, "...": "..."}, "tiny-dyn-x86_64": {"...": "..."}},
  "coff": {"main-x86_64.obj": {"...": "..."}},
  "pe": {"imports": [{"dll": "msvcrt.dll", "iat": "0x2040", "...": "..."}], "...": "..."},
  "macho": {"ledger_main": {"flags": "0x200085", "fixupRows": ["__DATA_CONST __got 0x100004000 bind libledger/_grade", "..."], "...": "..."}, "strip": {"...": "..."}},
  "identification": {"O2": {"methods": {"functionStarts": {"recall": "5/5"}, "prologueSub": {"recall": "1/5"}, "...": "..."}, "correctNames": 5}}
}
```

依此順序閱讀 `code/main.py`：

1. `ElfFile` 讀取 header、section、兩個 symbol table 與 relocation，而 `dynamic_entries()`、`jump_slots()`、`plt_entries()` 和 `plt0()` 讀取連結後的 image。
2. `plan_dynamic_link()` 把 Table 4.9 和 5.2 節套用到可重定位 object 的 relocation 紀錄。
3. `CoffObject` 讀取檔案 header、section table、帶 auxiliary 紀錄的 18 位元組 symbol，以及 10 位元組的 relocation。
4. `PeImage` 讀取 MS-DOS stub、PE 簽章、optional header、data directory、section，以及 import、export 與 base relocation 目錄，而 `iat_calls()` 解碼 `ff 15`。
5. `MachoFile` 讀取 load command、segment、`nlist` 紀錄、indirect symbol table、stub、chained fixup、exports trie 與 object relocation。
6. `FunctionFinder` 執行四個起點尋找器，而 `score()` 把每一個與真值比較。
7. `signatures_from_object()` 與 `match_signatures()` 建構並套用 FLIRT 形狀的 signature。
8. `parse_*` 函式讀取測試比較所用的、擷取到的工具文字。

更多指令：`python3 code/main.py elf FILE`、`coff FILE`、`pe FILE` 和 `macho FILE` 印出一份報告。`python3 code/main.py plan OBJECT...` 印出任何一組 ELF object 的連結計畫。`python3 code/main.py functions STRIPPED UNSTRIPPED OBJECT...` 在你自己的另一對 build 上，為尋找器與 signature 評分。

## 實際使用 (Use It)

依此順序，在你建構的程式上使用這些 reader。

1. 讀取 header，並說出格式與種類：可重定位 object、可執行檔，或共享函式庫。Reader 會拒絕它不認識的東西，而那個拒絕就是資訊。
2. 列出 import 及其提供者：`DT_NEEDED` 與 `.dynsym`、import 目錄，或 chained imports table 與 `LC_LOAD_DYLIB`。對這份清單上的名稱的每次 call，都離開 image。
3. 手動跟隨一次 call 到它的 slot。寫下 call 位元組、stub 或 PLT 位元組、slot 位址、檔案中的 word，以及 loader 在那裡寫入什麼。如果檔案中的 word 看起來像垃圾，在稱它為垃圾之前，先把它解碼為 fixup。
4. 比較 symbol table 與 export 和函式起點。Loader 需要的名稱在 `strip` 之後留下。只有 debugger 需要的名稱則不會。說明遺失的名稱屬於哪一組。
5. 在被 strip 的 build 中，執行你能執行的尋找器，並在你還有未 strip 的孿生版本時，對照它評分。優先使用 link editor 寫下的表格（`LC_FUNCTION_STARTS`、`.pdata` 或 `.eh_frame`），而不是位元組 pattern，並用 call 目標填補缺口。
6. 從你自己的 object 建構 signature 並套用它們。記錄每個衝突以及區分它的位元組。兩個 signature 都認領的名稱，在尾端或被參照的名稱決定之前，是未知的。

把擷取到的工具文字與檔案放在一起。當 toolchain 改變時，重新擷取，因為清單改變就是 fixture 改變。

## 互動實驗 (Interactive Lab)

### 實驗 A：手動看 PLT

從 `code` 目錄開啟 Python：

```bash
cd phases/13-tools-and-protocols/40-linking-loading-and-stripped-binaries/code
python3 -q
```

執行：

```python
from main import *
elf = load_elf("tiny-dyn-x86_64")
[(e.tag_name, e.text) for e in elf.dynamic_entries()]
elf.plt0()
[(e.symbol, hex(e.address), hex(e.slot), hex(e.slot_initial)) for e in elf.plt_entries()]
[hex(w) for w in elf.got_plt_words()]
```

在每次呼叫之前，根據上面文字中的位元組算出答案。然後把 `build_tiny_dyn_elf.py` 中的 `IMPORTS` 改成三個名稱、重建，並在閱讀之前預測 `.plt`、`.got.plt` 和 `.rela.plt` 的新大小。

### 實驗 B：一個 chained fixup

```python
exe = load_macho("ledger_main")
stub = next(s for s in exe.stubs() if s.symbol == "_puts")
stub.lines()
raw = int.from_bytes(exe.read(stub.slot, 8), "little")
hex(raw), raw >> 63, (raw >> 51) & 0xfff, raw & 0xffffff
[(i.index, i.name, i.library) for i in exe.chained_fixups()["imports"]]
```

先在紙上解碼 bind word：bind 位元、以四位元組為單位的 `next` 欄位、ordinal。然後閱讀再往後八個位元組的 word，並確認它是下一個 fixup。接著閱讀位於 `0x100004058` 的鏈的最後一個 word，並說明為什麼它的 `next` 是零。

### 實驗 C：strip 與重新連結

在 fixtures 目錄中，把全新的程式與完全被 strip 的 dylib 連結：

```bash
cd phases/13-tools-and-protocols/40-linking-loading-and-stripped-binaries/code/fixtures
clang -target arm64-apple-macos -O1 main.c -L. -lledger.stripped -Wl,-rpath,@executable_path -o /tmp/relinked
nm /tmp/relinked | grep ' U '
dyld_info -exports libledger.stripped.dylib
```

連結成功，即使函式庫的 `LC_SYMTAB` 是空的，因為 link editor 讀取 exports trie。然後執行 `python3 ../main.py macho libledger.stripped.dylib`，並比較 `symtab`、`exports` 和 `functionStarts`。接著改用 `-N` strip 靜態 build，看 `dyld_info -function_starts` 仍然印出什麼。

### 實驗 D：衝突

```python
from main import *
stripped = load_macho("static-arm64-O2.stripped")
sigs = signatures_from_object(load_macho("ledger-arm64-O2.o"))
[s.pat_line() for s in sigs]
starts = FunctionFinder(stripped).function_starts()
match_signatures(sigs, stripped, starts)
```

閱讀兩筆衝突紀錄。然後從 `ledger-arm64-O0.o` 建構 signature，與 `static-arm64-O0.stripped` 比對，並從 `.pat` 行說明為什麼 `-O0` 沒有發生衝突。接著把 `PATTERN_LENGTH` 縮短為 16，並計算有多少函式發生衝突。

## 實作練習 (Practice Lab)

解碼舊版的 binding 表格，並加入 PE 尋找器。

需求：

- 把 `ledger_main_nochain` 的 `LC_DYLD_INFO_ONLY` rebase、bind 與 lazy bind opcode 串流，解碼成帶有 segment、section、位址、型別與目標的列，並讓它們等於擷取的 `dyld_info -fixups` 文字中的 21 列。
- 在 `build_tiny_dyn_elf.py` 中加入 `PT_GNU_RELRO` program header 和帶有 `DF_BIND_NOW` 的 `DT_FLAGS`，把 `.got.plt` 移進被涵蓋的範圍，並讓 `llvm-objdump -p` 顯示新的 header。在 fixtures 的 README 中說明這重現了 full RELRO 的哪一部分，以及哪一部分只有 glibc 才能執行。
- 為 PE32+ image 寫一個 `.pdata` reader，以及一個使用 `RUNTIME_FUNCTION` begin 位址的函式尋找器。用不帶 `-fno-asynchronous-unwind-tables` 的 `clang -target x86_64-pc-windows-msvc -c` 建構 object，讀取它的 `.pdata` relocation，並說明它的哪些函式是表格省略的 leaf 函式。
- 為每個 reader 寫一個與擷取到的工具文字比較的測試。

成功的意思是，`python3 code/main.py macho ledger_main_nochain` 印出與 `dyld_info` 相同的 21 列 fixup，tiny ELF 帶有 `llvm-objdump` 列出的 `PT_GNU_RELRO`，而 `.pdata` 尋找器指名它能指名的函式，並列出它不能指名的 leaf 函式。

## 課程成品 (Lesson Artifact)

這堂課包含 `outputs/skill-linking-loading-and-stripped-binaries.md`。用它把你建構的程式，變成它的 import 及其提供者清單、追蹤到 loader 所填 slot 的一次 call、關於 `strip` 移除了什麼與留下了什麼的陳述，以及對被 strip 的 build 中函式起點與名稱的評分還原。它要求以未 strip 的孿生版本作為基準真相、每項主張都有擷取到的工具清單，以及每個衝突的紀錄。

## 驗證 (Verify It)

執行 demo 與確定性的測試套件：

```bash
cd phases/13-tools-and-protocols/40-linking-loading-and-stripped-binaries
python3 code/main.py
python3 code/tests/test_main.py
```

驗證應該證明：

- `tiny-dyn-x86_64` 的 dynamic section 逐 tag 符合 `llvm-objdump -p`，且 `DT_FLAGS_1` 帶有 `DF_1_PIE`
- 兩筆 `R_X86_64_JUMP_SLOT` 紀錄符合 `llvm-objdump -R`，並指名 `puts` 和 `exit`
- PLT0 push GOT+8 並透過 GOT+16 跳轉，每個項目讀取它的 slot 並 push 它的 index，而每個 slot 持有它自己的 `pushq` 的位址
- tiny ELF 的 section 和 program header 符合 `llvm-objdump -h -p`
- 兩個 ELF object 的 relocation 紀錄符合 `llvm-objdump -r`
- 連結計畫找到兩個 PLT symbol、四個直接 call、兩個 relative relocation，以及 tiny image 的大小
- COFF 的 section table、帶 auxiliary 紀錄的 37 筆 symbol 紀錄，以及 relocation 符合 `llvm-objdump -h -t -r`，且 `/16` 解析為 `.llvm_addrsig`
- `ledger` 的三個 COFF `REL32` 紀錄，位於三個 ELF `PC32` 的 offset
- PE 的 header、`DllCharacteristics` 和 data directory 符合 `llvm-objdump -p`
- import 目錄、hint/name 項目和 IAT 符合 `llvm-objdump` 與 `pefile`，且 IAT 在 bind 之前等於 lookup table
- export 目錄與 `DIR64` base relocation 能解碼，而被 relocation 的指標指名字串「linked by hand」
- RVA `0x100b` 的 `ff 15` 解析到 `puts` 的 IAT slot
- Mach-O 的 flag、19 個 load command 名稱，以及兩個函式庫符合 `otool`
- 七個 stub 解碼到八個 `__got` slot，且 indirect symbol table 符合 `otool -Iv`
- chained fixups header 符合 `llvm-objdump --macho --chained-fixups`
- 鏈的走訪重現 `dyld_info -fixup_chain_details` 的全部十二個原始 word
- fixup 列與 imports table 符合 `dyld_info -fixups` 和 `-imports`
- exports trie 對 dylib 及其被 strip 的副本，都符合 `dyld_info -exports`
- `nm_lines()` 對可執行檔及其被 strip 的副本，都等於 `nm -m`
- `strip` 移除了 `_main`、`_fns` 和 `_names`，保留九個名稱和一個 `radr://5614542` stab，且被 strip 的程式印出相同的輸出
- 舊版版面有 `LC_DYLD_INFO_ONLY`、七個 lazy 指標，以及進入 `__stub_helper` 的 rebase
- arm64 object 的 relocation 符合 `llvm-objdump -r`
- `LC_FUNCTION_STARTS` 在兩個層級都還原五個中的五個，並等於 `nm` 與 `dyld_info -function_starts`
- `sub sp` 掃描在 `-O0` 找到五個中的五個，在 `-O2` 找到五個中的一個，而 `stp` frame 掃描什麼也沒找到
- `bl` 目標加上 `LC_MAIN` 進入點，還原五個中的五個
- signature 在兩個層級都為全部五個函式命名，並記錄由位元組 44 區分的 `grade` 衝突
- "123456789" 的 CRC16 是 `0x6e90`，而 `main` 的 GOT-load 配對之下的八個位元組是可變的
- 每個 object 中非可變位元組都等於連結後的位元組
- demo 和圖表資料回報相同的位址，而網站模組以 ES5 內嵌相同的資料

## Production 的失敗模式 (Production Failure Modes)

| 失敗 | 薄弱的工具回報什麼 | 分析必須證明什麼 |
|---|---|---|
| `bl` 或 `call` 落在 stub 上 | 「呼叫位於 0x694 的無名函式」 | stub 位元組、它載入的 slot，以及該 slot bind 的 import |
| 從檔案讀取 GOT 的 word | 「損壞的指標 0x8010000000000007」 | 位元 63 是 bind、`next` 是 2、ordinal 7 是函式庫 2 的 `_puts` |
| object 中的 `PLT32` | 「每個 call 都經過 PLT」 | 已定義的 symbol 被化簡為直接的距離，只有未定義的才得到項目 |
| Partial RELRO 被讀成 full | 「GOT 是唯讀的」 | 沒有 `-z now` 時，三個保留 slot 之後的 `.got.plt` slot 位於 `PT_GNU_RELRO` 之外 |
| 把 COFF `REL32` 與 ELF `PC32` 比較 | 「Windows build 有不同的 relocation」 | offset 相同，`-4` 存在於 ELF 的 addend 和 COFF 的欄位中 |
| 在載入之前讀取 IAT | 「import 指向字串」 | IAT 在 loader bind 之前等於 lookup table，而字串是 hint/name 項目 |
| 被 strip 的 dylib | 「沒有 symbol，沒有匯出任何東西」 | exports trie 與 `LC_FUNCTION_STARTS` 列出每個函式 |
| 對被 strip 的可執行檔執行 `nm` | 「一個函式」 | `LC_FUNCTION_STARTS` 列出五個起點，長度九個位元組 |
| `-O2` 下的 prologue 掃描 | 「一個函式」 | 四個 leaf 函式沒有 frame，call 目標與起點表格找得到它們 |
| 兩個函式共用 36 個位元組 | 「grade_label 就是 grade」 | 位於 44 的尾端位元組與長度 60 和 72 區分衝突 |
| 在另一台機器的 build 上執行工具 | 這份文字中的數字 | 從你自己的 fixture 重新擷取每份清單 |

## 營運規則 (Operational Rule)

跟隨每個外部 call，從 call site 到 loader 填入的 slot，並在稱它為位址之前，依格式的定義解碼該 slot 中的 word。只有在你記錄了 loader 仍然需要什麼之後，才 strip。在掃描位元組之前，先從 link editor 寫下的表格還原函式起點，對照有 symbol 的孿生版本為每個尋找器評分，並記錄每個 signature 衝突以及區分它的位元組。

## 延伸閱讀 (Further Reading)

- [System V gABI, chapter 5: Program Loading and Dynamic Linking](https://www.sco.com/developers/gabi/latest/ch5.dynamic.html)
- [System V Application Binary Interface, AMD64 Architecture Processor Supplement: section 5.2 and Table 4.9](https://gitlab.com/x86-psABIs/x86-64-ABI)
- [Drepper, "How To Write Shared Libraries", version 4.1.2](https://akkadia.org/drepper/dsohowto.pdf)
- [Drepper, "ELF Symbol Versioning"](https://akkadia.org/drepper/symbol-versioning)
- [Jelinek, "Little hardening DSOs/executables against exploits", binutils list, 2004-01-06](https://www.sourceware.org/ml/binutils/2004-01/msg00070.html)
- [Taylor, "Linker relro", 2008](https://www.airs.com/blog/archives/189)
- [binutils commit bd7ab16b: PLT32 for calls, 2018-02-13](https://gnu.googlesource.com/binutils-gdb/+/bd7ab16b4537788ad53521c45469a1bdae84ad4a) and [LLVM D43383](https://reviews.llvm.org/D43383)
- [GNU Binutils: strip](https://sourceware.org/binutils/docs/binutils/strip.html) and [llvm-objcopy](https://llvm.org/docs/CommandGuide/llvm-objcopy.html)
- [Microsoft, PE Format](https://learn.microsoft.com/en-us/windows/win32/debug/pe-format)
- [CALL instruction reference, FF /2](https://www.felixcloutier.com/x86/call)
- [Apple, mach-o/loader.h](https://github.com/apple-oss-distributions/xnu/blob/main/EXTERNAL_HEADERS/mach-o/loader.h) and [mach-o/nlist.h](https://github.com/apple-oss-distributions/xnu/blob/main/EXTERNAL_HEADERS/mach-o/nlist.h)
- [Apple dyld, mach-o/fixup-chains.h](https://github.com/apple-oss-distributions/dyld/blob/main/include/mach-o/fixup-chains.h)
- [Apple ld64, stub_arm64.hpp](https://github.com/apple-oss-distributions/ld64/blob/main/src/ld/passes/stubs/stub_arm64.hpp)
- [Xcode 13 release notes: chained fixups](https://developer.apple.com/documentation/xcode-release-notes/xcode-13-release-notes) and [WWDC22 session 110362, "Link fast: Improve build and launch times"](https://developer.apple.com/videos/play/wwdc2022/110362/)
- [strip(1)](https://keith.github.io/xcode-man-pages/strip.1.html), [ld(1)](https://keith.github.io/xcode-man-pages/ld.1.html), and [dyld(1)](https://keith.github.io/xcode-man-pages/dyld.1.html)
- [Hex-Rays, IDA F.L.I.R.T. Technology: In-Depth](https://docs.hex-rays.com/core/flirt/concepts/ida-f.l.i.r.t.-technology-in-depth) and [Lumina overview](https://docs.hex-rays.com/add-ons/lumina/overview)
- [Ghidra Function ID help](https://github.com/NationalSecurityAgency/ghidra/blob/master/Ghidra/Features/FunctionID/src/main/help/help/topics/FunctionID/FunctionID.html) and [AARCH64 function start patterns](https://github.com/NationalSecurityAgency/ghidra/blob/master/Ghidra/Processors/AARCH64/data/patterns/AARCH64_LE_patterns.xml)
- [Vector 35, WARP](https://github.com/Vector35/warp) and [Binary Ninja WARP guide](https://docs.binary.ninja/guide/warp.html)
- [BinDiff manual](https://www.zynamics.com/bindiff/manual/) and [BinDiff source](https://github.com/google/bindiff)
- [Diaphora](https://github.com/joxeankoret/diaphora)
- [rizin pat.c, the .pat parser](https://github.com/rizinorg/rizin/blob/dev/librz/sign/pat.c), [radare2 flirt.c](https://github.com/radareorg/radare2/blob/master/libr/anal/flirt.c), and [radare2 book: Zignatures](https://book.rada.re/signatures/zignatures.html)
- [Andriesse et al., "An In-Depth Analysis of Disassembly on Full-Scale x86/x64 Binaries" (USENIX Security 2016)](https://www.usenix.org/conference/usenixsecurity16/technical-sessions/presentation/andriesse) and [Andriesse, Slowinska, Bos, "Compiler-Agnostic Function Detection in Binaries" (EuroS&P 2017)](https://danielandriesse.com/papers/eurosp-2017.pdf)
- [Bao et al., ByteWeight (USENIX Security 2014)](https://www.usenix.org/conference/usenixsecurity14/technical-sessions/presentation/bao) and [Shin, Song, Moazzezi, "Recognizing Functions in Binaries with Neural Networks" (USENIX Security 2015)](https://www.usenix.org/conference/usenixsecurity15/technical-sessions/presentation/shin)
- [LIEF changelog](https://lief.re/doc/latest/changelog.html), [pyelftools](https://github.com/eliben/pyelftools), and [pefile](https://github.com/erocarrera/pefile)
