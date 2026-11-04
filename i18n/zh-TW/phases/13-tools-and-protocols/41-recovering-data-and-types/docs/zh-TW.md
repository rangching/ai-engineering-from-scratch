# 還原資料與型別：Struct、陣列、C++ 物件與 DWARF (Recovering Data and Types: Structs, Arrays, C++ Objects, and DWARF)

> 當反編譯器有型別表時，它印出 `a->balance`；沒有時，它印出 `*(int *)(param_1 + 4)`。這堂課從存取動作建構型別表、從 DWARF 5 讀取真正的型別表，並顯示兩者在哪裡一致。

**Type:** Build
**Languages:** Python, C, C++, x86-64 assembly
**Prerequisites:** Phase 13 · 33 (disassembly and control flow), Phase 13 · 34 (lifting to pseudocode)
**Time:** ~140 分鐘

## 學習目標 (Learning Objectives)

- 依 Intel SDM 的定址規則，把 struct 成員存取讀成 base 暫存器加上 displacement，把陣列存取讀成 base 加上 index 乘以 stride。
- 手動對 `struct account` 套用 System V 的對齊規則，並說明每個 padding 位元組。
- 建構型別還原 pass，把清單中的存取動作變成欄位 offset、寬度、種類、stride 與大小，並附上每一項的證據。
- 為可重定位 object 的 `.debug_info`、`.debug_abbrev`、`.debug_str_offsets` 與 `.debug_addr` 寫一個 DWARF 5 reader，並對照 `llvm-dwarfdump` 檢查。
- 比較還原的版面與 DWARF 的版面，並把每個差異命名為缺少證據、被合併的存取，或無法證明的種類。
- 從 relocation 讀取 Itanium C++ vtable、它的 typeinfo 與修飾後的名稱，並把虛擬 call 解析到某個 slot。

## 問題 (The Problem)

你不帶 `-g` 編譯 `account.c`，並問 agent `bump` 做什麼。Agent 把 object 送給 Ghidra 12.1.4，得到：

```c
void bump(long param_1,int param_2)
{
  *(int *)(param_1 + 4) = param_2 + *(int *)(param_1 + 4);
  return;
}
```

同一份原始碼帶 `-g` 編譯，同一個反編譯器給出：

```c
int bump(account *a,int by)
{
  a->balance = by + a->balance;
  return a->balance;
}
```

兩個 object 中的指令是相同的。差別是型別表。有了這張表，反編譯器知道第一個參數指向一個 32 位元組的結構，其 offset 4 的欄位是名為 `balance` 的 4 位元組 `int`。沒有它，反編譯器只知道有個 8 位元組寬的東西送進了 `%rdi`，而 `+4` 處的 4 個位元組被讀取並寫入。

那張表中的每一樣東西都有來源。Offset 4 來自 `addl 0x4(%rax), %ecx` 的 displacement 位元組。寬度 4 來自 `l` 後綴。名稱來自 `.debug_info` 中的 `DW_TAG_member` 項目。這堂課建構兩半：一個還原指令所能證明之事的 pass，以及一個解碼編譯器所記錄之事的 reader。兩者之間的差距，就是型別名稱的代價。

```figure
re-41-struct-recovery
```

## 成員存取是一個 displacement (A Member Access Is a Displacement)

Intel SDM Volume 1 的 3.7.5 節，把有效位址定義為 `Offset = Base + (Index * Scale) + Displacement`，其中 scale 為 1、2、4 或 8，displacement 為 8、16 或 32 位元。3.7.5.1 節加入 64 位元模式的形式 `RIP + Displacement`。Struct 成員存取使用沒有 index 的第一種形式：base 暫存器持有物件的位址，而 displacement 是成員的 offset。陣列存取加入 index 項，而 scale 或獨立的乘法承載元素大小。

以 `-O0` 的 `bump` 為例。`llvm-objdump -d -r` 的清單讀作：

```text
       4: 48 89 7d f8                  	movq	%rdi, -0x8(%rbp)
       e: 48 8b 45 f8                  	movq	-0x8(%rbp), %rax
      12: 03 48 04                     	addl	0x4(%rax), %ecx
      15: 89 48 04                     	movl	%ecx, 0x4(%rax)
```

用課程 33 的規則解碼 `03 48 04`。`03` 是 `ADD r32, r/m32`。ModRM `48` 是 `01 001 000`：`mod = 01` 表示帶 8 位元 displacement 的記憶體 operand，`reg = 001` 是 `%ecx`，`r/m = 000` 是 `%rax`。Displacement 位元組是 `04`。這條指令讀取 `%rax + 4` 的 4 個位元組並加進 `%ecx`。`llvm-objdump` 印出的 `l` 後綴是 operand 大小，所以該欄位是 4 位元組寬。`%rax` 來自 `-0x8(%rbp)` 的 frame slot，而該 slot 收到的是 `%rdi`，也就是 System V 呼叫慣例下的第一個整數引數。由此得出三項事實：第一個參數是指標，它所指的物件在 offset 4 有一個 4 位元組的欄位，而 `bump` 讀取並寫入該欄位。

這些位元組裡沒有任何東西說 `int`、`balance` 或 `struct account`。也沒有任何東西說明物件有多大。Struct 剩下的 28 個位元組，對 `bump` 是看不見的。

## System V 的版面規則 (The System V Layout Rules)

x86-64 psABI 的 3.1.2 節，給出三條關於聚合型別的規則。「Structure 與 union 採用其對齊要求最嚴格的元件的對齊。每個成員被指派到具有適當對齊的最低可用 offset。任何物件的大小永遠是該物件對齊的倍數。」同一份文件的 Figure 3.1 給出純量的大小與對齊：`char` 1 和 1、`short` 2 和 2、`int` 4 和 4、`double` 8 和 8、任何指標 8 和 8。

把這些規則套用到 `account.c` 中的 struct：

```c
struct account {
    char kind;
    int balance;
    short code;
    double rate;
    const char *label;
};
```

| 成員 | 大小 | 對齊 | 放得下的最低 offset | 之前的 padding |
|---|---|---|---|---|
| `kind` | 1 | 1 | 0 | 0 |
| `balance` | 4 | 4 | 4 | 3（位元組 1 到 3） |
| `code` | 2 | 2 | 8 | 0 |
| `rate` | 8 | 8 | 16 | 6（位元組 10 到 15） |
| `label` | 8 | 8 | 24 | 0 |

最後一個成員結束於 32。最嚴格的對齊是 8，而 32 是 8 的倍數，所以大小是 32，沒有尾端 padding。32 個位元組中有九個是 padding，而 psABI 說「任何 padding 的內容都是未定義的」。

成員的順序決定大小。同樣五個成員以 `double, pointer, int, short, char` 的順序，填滿 offset 0、8、16、20、22，並結束於 23，向上捨入為 24。`code/main.py` 中的 `sysv_layout()` 套用這三條規則並產生兩個結果。Python 的 `ctypes` 模組透過 host 編譯器的慣例套用相同的規則，而對同一個 struct，`ctypes.sizeof`、`ctypes.alignment` 以及每個欄位的 `.offset` 給出 32、8，以及 `0, 4, 8, 16, 24`。當你在這台機器上建構並執行程式時，程式本身會印出第三份副本：

```text
sizeof 32 align 8
kind 0 balance 4 code 8 rate 16 label 24
sum 315 tag -3 projected 318.75 label_length 7
```

那個原生 build 是 arm64。Arm AAPCS64 在它的 5.10.1 節陳述相同的聚合規則，而 Apple 的 arm64 文件保留這五種純量大小，所以兩個目標的數字一致。這堂課的 x86-64 object 是用 `-target x86_64-unknown-linux-gnu` 編譯的，讓課程 33 的 ELF reader 與下面的 DWARF reader 都能在上面運作。

## 手動讀出 struct 的其餘部分 (Read the Rest of the Struct by Hand)

`bump` 只看到一個欄位。`account.c` 中其他四個函式看到其餘部分，而每一個都顯示不同的指令形式。以下所有位址都來自 `account-x86_64-O0-g.objdump.txt`。

`sum_balances` 對 struct 的陣列做 index：

```text
      51: 48 8b 45 f8                  	movq	-0x8(%rbp), %rax
      55: 48 63 4d ec                  	movslq	-0x14(%rbp), %rcx
      59: 48 c1 e1 05                  	shlq	$0x5, %rcx
      5d: 48 01 c8                     	addq	%rcx, %rax
      60: 8b 40 04                     	movl	0x4(%rax), %eax
```

`%rax` 是 `list`，`%rcx` 是符號擴充為 64 位元的 `i`，而 `shlq $0x5` 把它乘以 32。在 `addq` 之後，`%rax` 是 `list + 32 * i`，而 load 讀取元素 `i` 的 offset 4 處的 4 個位元組。移位數是 struct 大小在這個 object 中唯一以數字出現的地方。包含這些指令的迴圈，從 `0x49` 執行到位於 `0x72` 的 `jmp 0x49`。

`tag` 讀取兩個窄欄位：

```text
      8c: 0f be 00                     	movsbl	(%rax), %eax
      98: 0f bf 48 08                  	movswl	0x8(%rax), %ecx
```

`movsbl` 載入 offset 0 的一個位元組並做符號擴充，所以 0 處的欄位是有號的 1 位元組整數。`movswl` 載入 offset 8 的兩個位元組並做符號擴充，所以 8 處的欄位是有號的 2 位元組整數。在 `-O1`，第一次存取變成 `cmpb $0x6c, (%rdi)`，這仍然說明 offset 0 處有 1 個位元組，但沒有說明符號。

`projected` 讀取浮點數：

```text
      cc: f2 0f 2a 40 04               	cvtsi2sdl	0x4(%rax), %xmm0
      d5: f2 0f 59 40 10               	mulsd	0x10(%rax), %xmm0
```

`cvtsi2sdl` 把記憶體中有號的 32 位元整數轉換成 double，這確認 offset 4 持有有號的 4 位元組整數。`mulsd` 乘以記憶體中的 double，所以 offset 16 持有 8 位元組的浮點值。

`label_length` 讀取指標，然後跟隨它：

```text
      f3: 48 8b 40 18                  	movq	0x18(%rax), %rax
      f7: 48 63 4d f4                  	movslq	-0xc(%rbp), %rcx
      fb: 80 3c 08 00                  	cmpb	$0x0, (%rax,%rcx)
```

`movq` 讀取 offset 24 的 8 個位元組。單看這一點，它是 8 位元組整數或指標。`cmpb` 把載入的值當作記憶體 operand 的 base。被當作位址使用的值就是指標。`cmpb` 從中以 index `%rcx` 讀取 1 個位元組，而在 `-O1`，指標在迴圈內以 `leaq 0x1(%rcx), %rcx` 前進，所以被指向的元素彼此相距 1 個位元組。

把這五個函式放在一起，struct 就是 `{ i8 at 0, i32 at 4, i16 at 8, f64 at 16, pointer at 24 }`，stride 為 32。1 到 3 和 10 到 15 的空隙，正是 psABI 規則所預測的 padding。指令沒有說明的是：成員名稱、指標是 `const char *`、位於 0 的有號位元組的宣告型別（原始碼寫為 `char`），以及這五個函式收到的是同一個型別。最後一項主張來自 `main`，它把一個 frame 物件傳給全部五個。在 `-O0`，call site 讓這點看得見：在 `callq bump` 之前是 `leaq -0x70(%rbp), %rdi` 後面接 `addq $0x20, %rdi`，在 `callq tag` 之前是 `addq $0x40, %rdi`。Offset 0、`0x20` 和 `0x40` 是同一個陣列的三個元素，而填滿它的 `0x60` 位元組的 `memcpy` 是 96 位元組，也就是 32 位元組的三個元素。

## 這個 Pass (The Pass)

`code/main.py` 中的 `trace_function()` 依位址順序走訪一個函式的指令，並維護一個小型的抽象狀態。Holder 是 64 位元暫存器，或是像 `-0x8(%rbp)` 這樣的 frame slot。每個 holder 有三種值之一。`Pointer` 是 root 加上固定 offset，再加上一組 stride，其中 stride 記錄某個未知倍數的這麼多位元組被加進來。`Index` 是值未知、但是某個 scale 的倍數的整數。其他任何東西都是 nothing。

狀態一開始，六個 System V 引數暫存器分別持有 `Pointer("param0", 0)` 到 `Pointer("param5", 0)`。然後每條指令套用一條規則：

- 64 位元暫存器與 frame slot 之間的 `movq` 複製該值。對暫存器的 32 位元或更窄的寫入，會讓它成為 scale 為 1 的 `Index`。
- 帶有指標 base 的 `leaq` 產生一個加上 displacement 的指標，而帶有 index 暫存器時，它把 `scale * index_scale` 加到 stride 中。Base 為 `%rbp` 或 `%rsp` 的 `leaq` 指名一個 frame 物件，這就是 `main` 傳遞 `rows` 的方式。
- 對 `Index` 的 `shlq $n` 把它的 scale 乘以 `2^n`，而 `imulq $k` 乘以 `k`。
- 指標與 index 的 `addq` 保留指標，並把 index 的 scale 加進 stride。對指標的 `addq $k` 移動它的 offset，同時也被記錄為該暫存器的遞增。
- Call 讓每個 caller-saved 暫存器變成未知，而持有指標的引數暫存器則被記錄為 call site。
- 任何 base 暫存器持有指標的記憶體 operand，都變成一筆 `Access`，帶有指標的 root 與 offset、來自 mnemonic 的寬度與種類，以及 store 旗標。

寬度與種類只來自 mnemonic。`access_width()` 把 `b`、`w`、`l` 和 `q` 後綴對應到 1、2、4 和 8 位元組，把 `movsbl` 和 `movswl` 對應到有號的窄 load，把 `movzbl` 對應到無號，把純量 SSE 指令 `sd` 和 `ss` 對應到 8 位元組與 4 位元組的浮點數，把 `cvtsi2sdl` 對應到有號的 4 位元組整數來源，並把 16 位元組的 `movaps` 家族對應到向量。`leaq` 和 `nop` 的形式不是存取。

```python
if memory.base in FRAME_REGISTERS:
    return Pointer(f"frame({memory.base}{format_signed(memory.displacement) if memory.displacement else ''})", 0)
base = env.get(memory.base)
if not isinstance(base, Pointer):
    return Unknown
strides = set(base.strides)
if memory.index is not None:
    index_value = env.get(memory.index)
    strides.add((index_value.scale if isinstance(index_value, Index) else 1) * memory.scale)
return Pointer(base.root, base.offset + memory.displacement, frozenset(strides), base.origin)
```

透過指標做 8 位元組的 load，會產生一個新的指標，其 root 指名它來自的欄位，例如 `param0->0x18`。當該值之後被用作 base，或以 `leaq` 前進時，它來自的欄位就被標記為已證明的指標。那條規則把 offset 24 從「8 位元組」變成「指標」。

### 四種 stride 證據

`infer_layout()` 合併每個見證函式的存取，並從四個來源收集 stride，每個都附上產生它的指令。

| 種類 | 來自哪裡 | 顯示它的 build |
|---|---|---|
| `scale` | 一個 scale 大於 1 的 `Index` 到達了指標，來自 `shlq $0x5` 或 `imulq` | `-O0`，位於 `0x59` 的 `shlq $0x5, %rcx` |
| `loop-step` | 在同時包含該存取的迴圈內，對 base 或 index 暫存器的 64 位元遞增 | `-O1`，位於 `0x24` 的 `addq $0x20, %rdx` |
| `offset-delta` | 一個迴圈內，透過同一個 base 暫存器、offset 彼此相差固定值的相同寬度存取 | `-O2`，相隔 32 的八個 load |
| `call-site` | 一個 frame 物件被傳給見證函式時的 offset | `-O0`，`main` 傳遞 `rows`、`rows+0x20`、`rows+0x40` |

迴圈是向後的 `jmp` 或條件分支，而只有當迴圈範圍同時包含遞增與存取時，遞增才算數。這條規則排除了 `-O0` 中推進 `i` 的 `addl $0x1, %eax`，因為它是 32 位元的 add，而且存取不經過 `%eax`。最終的 stride 是找到的每個值的最大公因數。當 stride 至少與最後一個欄位的結尾一樣大時，大小就是 stride。

### -O2 的向量化迴圈

在 `-O2`，clang 向量化 `sum_balances`。從 `0x50` 到 `0xaa` 的迴圈本文，每趟讀取八個元素：

```text
      3b: 48 8d 87 e4 00 00 00         	leaq	0xe4(%rdi), %rax
      50: 66 0f 6e 50 80               	movd	-0x80(%rax), %xmm2
      55: 66 0f 6e 98 60 ff ff ff      	movd	-0xa0(%rax), %xmm3
      61: 66 0f 6e 90 40 ff ff ff      	movd	-0xc0(%rax), %xmm2
      69: 66 0f 6e a0 20 ff ff ff      	movd	-0xe0(%rax), %xmm4
      7d: 66 0f 6e 10                  	movd	(%rax), %xmm2
      81: 66 0f 6e 58 e0               	movd	-0x20(%rax), %xmm3
      8a: 66 0f 6e 50 c0               	movd	-0x40(%rax), %xmm2
      8f: 66 0f 6e 60 a0               	movd	-0x60(%rax), %xmm4
      a0: 48 05 00 01 00 00            	addq	$0x100, %rax
      aa: 75 a4                        	jne	0x50 <sum_balances+0x40>
```

`%rax` 從 `list + 0xe4` 開始。八個 displacement 把 load 放在距離 `list` 的 offset `0x04, 0x24, 0x44, 0x64, 0x84, 0xa4, 0xc4, 0xe4`，也就是 `4, 36, 68, 100, 132, 164, 196, 228`。遞增是 `0x100`，也就是 256。看到什麼就記錄 offset 的 pass，會回報八個欄位與至少 232 位元組的 struct。清單是正確的，結論卻是錯的。

`fold_unrolled()` 套用一條規則。在一個迴圈內，取透過同一個 base 暫存器的相同寬度存取。它們的 offset 相隔 `d`，這裡是 32。如果迴圈步長是 `k * d`，而且恰好有 `k` 個不同的 offset，則該迴圈被展開了 `k` 次，而每個存取都是 offset `(offset - lowest) mod d` 的同一個元素存取。這裡 `256 = 8 * 32` 且有八個 offset，所以全部八個 load 都折疊到 offset 4，而 stride 證據記錄 `offset-delta 32`、`unrolled 32` 和 `loop-step 256`。位於 `0xe0` 的純量餘數迴圈，從 `shlq $0x5, %rdx` 加入 `scale 32`，並從 `addq $0x20, %rdx` 加入 `loop-step 32`。全部五個值的最大公因數是 32。

圖表有一個關閉折疊的按鈕。關閉時，`-O2` 的 build 顯示超過位元組 32 之後的七個欄位，而沒有大小。那個狀態就是信任 displacement 卻不看迴圈的讀者所看到的。

## 從位元組看 DWARF 5 (DWARF 5 From the Bytes)

`account.c` 的 `-g` build 帶有八個偵錯 section。`.debug_info` 是 493 位元組、`.debug_abbrev` 179、`.debug_str` 230、`.debug_str_offsets` 124、`.debug_addr` 112、`.debug_line` 312、`.debug_line_str` 12，以及 `.debug_frame` 264。DWARF 5 標準的簡介，列出 `.debug_str_offsets`、`.debug_addr` 和 `.debug_line_str` 是 Version 5 新增的。`code/main.py` 中的 reader 需要前五個。

### Unit header

標準的 7.5.1.1 節給出完整的 compilation unit header：`unit_length`、`version`、`unit_type`、`address_size` 和 `debug_abbrev_offset`。`.debug_info` 的前十二個位元組是：

```text
e9 01 00 00   unit_length = 0x1e9 = 489
05 00         version = 5
01            unit_type = DW_UT_compile
08            address_size = 8
00 00 00 00   debug_abbrev_offset = 0
```

`unit_type` 位元組是 Version 5 新增的，標準在同一節這麼說。長度計算的是它自身之後的位元組，所以下一個 unit 會從 `4 + 489 = 493` 開始，也就是 section 的結尾。`llvm-dwarfdump --debug-info` 在它的 `Compile Unit:` 行印出相同的五個值，而第一個測試解析那一行並比較。

### Abbreviation

2.1 節說，每個 debugging information entry「由一個識別用的 tag 與一系列 attribute 組成」。項目本身只儲存 abbreviation code。該 code 選擇 `.debug_abbrev`（7.5.3 節）中的某個宣告，它指名 tag、說明是否有 child 跟隨，並列出每個 attribute 及其 form。這個 object 有 14 個宣告。其中兩個在這裡很重要：

```text
[12] DW_TAG_structure_type  DW_CHILDREN_yes
     DW_AT_name              DW_FORM_strx1
     DW_AT_byte_size         DW_FORM_data1
     DW_AT_decl_file         DW_FORM_data1
     DW_AT_decl_line         DW_FORM_data1
[13] DW_TAG_member           DW_CHILDREN_no
     DW_AT_name              DW_FORM_strx1
     DW_AT_type              DW_FORM_ref4
     DW_AT_decl_file         DW_FORM_data1
     DW_AT_decl_line         DW_FORM_data1
     DW_AT_data_member_location  DW_FORM_data1
```

`parse_abbreviations()` 讀取 ULEB128 的 code、ULEB128 的 tag、一個 children 位元組，然後是 attribute 與 form 的配對，直到 `0, 0` 這一對。7.5.6 節給出 form 的編碼：`data1` 是一個位元組，`ref4` 是相對於 unit 的 4 位元組 offset，`strx1` 是進入 string offsets table 的 1 位元組 index，`sec_offset` 與 `data4` 是 4 位元組，`addrx` 是進入 address table 的 ULEB128 index，`exprloc` 是 ULEB128 長度加上那麼多個位元組，而 `flag_present` 完全沒有位元組。

### 手動解碼 balance 成員

`account` 的 structure 項目從 offset `0x195` 開始，位元組是 `0c 15 20 00 03`。Abbreviation code 是 12，也就是 structure。`DW_AT_name` 是字串 index `0x15`，`DW_AT_byte_size` 是 `0x20` = 32，`DW_AT_decl_file` 是 0，而 `DW_AT_decl_line` 是 3，也就是原始碼中 `struct account {` 所在的行。

第二個成員項目從 `0x1a3` 開始，位元組是 `0d 10 88 01 00 00 00 05 04`：

1. `0d` 是 abbreviation 13，成員。
2. `10` 是以 `strx1` 表示的 `DW_AT_name`：字串 index 16。
3. `88 01 00 00` 是以 `ref4` 表示的 `DW_AT_type`：距離 unit 起點的 offset `0x188`，也就是名為 `int` 的 `DW_TAG_base_type` 項目。
4. `00` 是 `DW_AT_decl_file`，而 `05` 是 `DW_AT_decl_line`：`account.c` 的第 5 行。
5. `04` 是 `DW_AT_data_member_location`：offset 4。

5.7.6 節說成員項目「可以有 DW_AT_data_member_location attribute 或 DW_AT_data_bit_offset attribute」，並說整數常數「是從所包含實體開頭算起的位元組 offset」。那個位元組與 `addl 0x4(%rax), %ecx` 中的 displacement 是同一個數字，而 `compare_layouts()` 就是把兩者並排的函式。

### 字串與位址要透過兩個表格

字串 index 16 不是進入 `.debug_str` 的 offset。Reader 從 unit 項目取得 `DW_AT_str_offsets_base`，也就是 8，並讀取 `.debug_str_offsets` 中 `8 + 4 * 16` 處的 4 位元組項目。那個項目持有 `"balance"` 在 `.debug_str` 中的 offset。同樣的間接層也適用於位址。`sum_balances` 的項目以位元組 `07 07` 開頭：abbreviation 7，也就是 subprogram，然後是以 `addrx` index 7 表示的 `DW_AT_low_pc`。`.debug_addr` 中 `8 + 8 * 7` 處的 8 位元組項目是 `0x30`，也就是函式在 `.text` 中的位址。該表格的前六個項目是六個字串常數的位址。

在可重定位 object 中，這兩個表格都沒有持有最終的值。`.debug_str_offsets` 中的每個項目在檔案中都是零，而 `.rela.debug_str_offsets` 持有 29 個針對 `.debug_str` 的 `R_X86_64_32` relocation，其真正的 offset 在 addend 中。`.debug_addr` 有 13 個針對 `.text` 和 `.rodata.str1.1` 的 `R_X86_64_64` relocation。`.debug_info` 有它自己的四個 relocation：abbreviation offset、作為 `.debug_str_offsets+0x8` 的 `DW_AT_str_offsets_base`、`DW_AT_stmt_list`，以及作為 `.debug_addr+0x8` 的 `DW_AT_addr_base`。在 reader 解析某個 section 之前，`ElfObject.relocated()` 對 32 位元與 64 位元的型別套用 `S + A`。Section symbol 在 object file 中的值為 0，所以欄位收到 addend。`llvm-dwarfdump` 做相同的解析，這就是它的輸出顯示每個 index 都已解析的名稱與位址的原因。

### 型別、函式與位置

`DwarfInfo.type_name()` 跟隨 `DW_AT_type` 參照。`label` 成員指向 `DW_TAG_pointer_type`，它指向 `DW_TAG_const_type`，再指向 `char` 基本型別，所以名稱組成為 `const char *`。`type_size()` 對基本型別與 structure 回傳 `DW_AT_byte_size`，對指標回傳 8，對陣列回傳元素數量乘以元素大小。`type_kind()` 把基本型別的 `DW_AT_encoding` 化簡為帶符號的 `int` 或 `float`，並把指標型別化簡為 `pointer`，這就是比較所使用的詞彙。

`DwarfInfo.functions()` 讀取六個 `DW_TAG_subprogram` 項目。`bump` 從 `0x0` 執行到 `0x21`，也就是 `DW_AT_low_pc` 加上 `DW_AT_high_pc` 的長度。它的 frame base 是 `DW_OP_reg6 RBP`，而它的參數 `a` 有位置表達式 `91 78`：`DW_OP_fbreg` 搭配 SLEB128 值 `0x78`，也就是 -8。那就是清單在位址 `0x4` 把 `%rdi` 存入的 frame slot `-0x8(%rbp)`。偵錯資訊與指令從兩側描述同一個 slot。

## 比較兩種版面 (Compare the Two Layouts)

`compare_layouts()` 走訪 DWARF 成員，並尋找位於相同 offset 的推論欄位。沒有欄位的成員是 `no-evidence`。大小不同的欄位是 `size-mismatch`。從未被 dereference 的指標成員是 `pointer-unproven`。種類不一致，例如指令使用整數算術卻為浮點數，是 `kind-mismatch`。起點落在成員內部或 padding 內部的推論欄位是 `unexpected`。大小則把 stride 與 `DW_AT_byte_size` 比較。

三個 build 的數字，來自 `python3 code/main.py`：

| Build | 透過第一個參數的存取 | 欄位 | Stride 證據 | 符合的成員 | 大小 |
|---|---|---|---|---|---|
| `-O0` | 11 | 5 | `scale 32`、`call-site 32`、`call-site 64` | 5 個中的 5 個 | 32，相符 |
| `-O1` | 9 | 5 | `loop-step 32` | 5 個中的 5 個 | 32，相符 |
| `-O2` | 17 | 折疊後 5 個，折疊前 12 個 | `offset-delta 32`、`unrolled 32`、`loop-step 256`、`scale 32`、`loop-step 32` | 5 個中的 5 個 | 32，相符 |

在每個 build 中，推論出的空隙都是位元組 1 到 3 和 10 到 15，而它們正是 DWARF 版面的 padding 位元組。每個推論欄位都位於自己大小的倍數，這是 psABI 規則所要求的，而 `natural_alignment_ok()` 記錄了這點。

一次加入一個函式，畫面就會改變。只有 `bump` 時，版面是 offset 4 的一個欄位，大小未知，下界是 8。`sum_balances` 加入 stride。`tag` 加入 offset 0 和 8。`projected` 加入位於 16 的 float。`label_length` 加入位於 24 的指標，並證明它是指標。`main` 在 `-O0` 加入 call site，在 `-O1` 和 `-O2` 什麼也不加，因為編譯器 inline 了每個 call。圖表為每個 build 依此順序逐步呈現。

有三樣東西仍然遙不可及。名稱不在指令中。`const` 限定詞不在指令中。`label` 的 pointee 型別，只是來自字串迴圈中 `leaq 0x1(%rcx), %rcx` 的「相距 1 位元組的元素」，這與 `char`、`unsigned char` 和 `_Bool` 一致。在 `-O1`，offset 0 處位元組的符號是未知的，因為 `cmpb $0x6c, (%rdi)` 在比較時不做擴充。Pass 在那裡回報 `int8`，在 `movsbl` 做符號擴充的 `-O0` 則回報 `i8`。這些就是課程 35 記錄為未知的事實。

## C++ 物件 (C++ Objects)

`shapes.cpp` 定義了帶有虛擬 destructor、純虛擬 `area()` 和虛擬 `name()` 的 `Shape`，以及從 `Shape` 衍生並加入兩個 `int` 成員的 `Rect`。Itanium C++ ABI 規範 clang 如何在 x86-64 Linux 上安排這些。

### 修飾後的名稱

ABI 的 5.1.2 節說「名稱的修飾方式，是在名稱的編碼前面加上 `_Z`」。這個 object 有 19 個這種 symbol。`_ZNK4Rect4areaEv` 讀作 `_Z`，然後 `N` 表示巢狀名稱，`K` 表示 const 成員函式，`4Rect` 和 `4area` 是帶長度前綴的原始碼名稱，`E` 結束巢狀名稱，而 `v` 表示空的參數清單。`llvm-cxxfilt` 印出 `Rect::area() const`。`_Z8describeRK5Shapei` 是 `describe(Shape const&, int)`：`R` 是參照，`K` 是 const，`5Shape` 是類別，`i` 是 `int`。5.1.4 節中的特殊名稱是：vtable 用 `_ZTV`，typeinfo 用 `_ZTI`，typeinfo 名稱字串用 `_ZTS`。Constructor 和 destructor 是 `C1`、`C2`、`D0`、`D1` 和 `D2`，所以 `_ZN4RectD0Ev` 和 `_ZN4RectD2Ev` 都印成 `Rect::~Rect()`：deleting destructor 與 base-object destructor 共用原始碼名稱。

`code/main.py` 中的 `Demangler` 涵蓋該子集，加上 5.1.5.2 節的內建型別代碼與 5.1.10 節的 `S_` 替換。測試對全部 19 個名稱執行它，並與擷取到的 `llvm-cxxfilt --no-strip-underscore` 輸出比較。這個旗標在 macOS 上很重要，因為該工具預設會剝除一個開頭的底線，然後無法辨識 `Z8describe...`。

### 從 relocation 看 vtable

ABI 的 2.5.2 節依序列出虛擬表格的元件：虛擬 call offset、虛擬 base offset、offset to top、typeinfo 指標，然後是虛擬函式指標。位址點（address point），也就是物件的 vptr 所持有的值，是第一個函式指標的位址（2.6.2 節）。`Rect` 沒有虛擬 base，所以它的表格是 offset to top、typeinfo 和函式。

Symbol `_ZTV4Rect` 是 0x30 位元組，六個 8 位元組的 slot。在 object 中，每個位元組都是零。內容在 `.rela.data.rel.ro._ZTV4Rect` 中：

```text
0000000000000008 R_X86_64_64              _ZTI4Rect
0000000000000010 R_X86_64_64              _ZN4RectD2Ev
0000000000000018 R_X86_64_64              _ZN4RectD0Ev
0000000000000020 R_X86_64_64              _ZNK4Rect4areaEv
0000000000000028 R_X86_64_64              _ZNK4Rect4nameEv
```

Slot 0 持有 offset to top，為 0 且不需要 relocation。Slot 1 是 typeinfo。位址點在 `+16`。從那裡開始，slot 0 是 complete-object destructor，slot 1 是 deleting destructor，slot 2 是 `area`，slot 3 是 `name`。`read_vtable()` 讀取 symbol 大小、其範圍內的 relocation，以及 offset 0 的零。`Shape` 的表格形狀相同，slot 2 是 `__cxa_pure_virtual`，也就是純虛擬 call 會到達的 runtime 函式。

### RTTI

2.9.4 節把 `std::type_info` 描述為 vptr 後面接一個指向修飾後型別名稱的指標，並列出衍生的類別。`_ZTI4Rect` 有三個 relocation：`_ZTVN10__cxxabiv120__si_class_type_infoE+0x10`、`_ZTS4Rect` 和 `_ZTI5Shape`。第一個是 `__si_class_type_info` 的 vtable，ABI 用這個類別表示在 offset 零處有單一公開非虛擬 base 的型別，而 `+0x10` 是該 vtable 的位址點。第二個指向字串 `4Rect`，也就是修飾後的型別名稱，而 `type_info` 上的 `name()` 回傳的正是那個字串。第三個是 base。`_ZTI5Shape` 使用 `__class_type_info` 且沒有 base 指標。`read_typeinfo()` 回傳這三項事實。

### 虛擬 call

`-O0` 的 `describe` 做一次虛擬 call：

```text
       f: 48 8b 7d f8                  	movq	-0x8(%rbp), %rdi
      13: 48 8b 07                     	movq	(%rdi), %rax
      16: ff 50 10                     	callq	*0x10(%rax)
```

第一個 `movq` 載入物件指標。第二個讀取物件 offset 0 的 8 個位元組，也就是 vptr，因為 ABI 把它放在有虛擬函式的類別的最前面。`callq` 經過 vptr 之後的 `0x10`，而 `0x10 / 8` 是 slot 2。`find_virtual_calls()` 比對這個模式，並回報兩個候選：來自 `_ZTV4Rect` 的 `Rect::area() const`，以及來自 `_ZTV5Shape` 的 `__cxa_pure_virtual`。哪一個會執行，取決於物件，這就是為什麼這個 call 是 indirect 的。

DWARF 記錄相同的 slot。兩個類別中 `area` 的 `DW_TAG_subprogram` 項目都帶有 `DW_AT_vtable_elem_location (DW_OP_constu 0x2)`，`name` 帶有 slot 3，而 destructor 帶有 slot 0。`Rect` 是 16 位元組的 `DW_TAG_structure_type`，帶有位於 offset 0、針對 `Shape` 的 `DW_TAG_inheritance` 項目，以及位於 8 的成員 `w` 和位於 12 的 `h`。`Shape` 是 8 位元組，有一個人工成員 `_vptr$Shape`。兩個獨立的推導給出 `area` 的 slot 2：relocation 表格與偵錯項目。

在 `-O1`，object 根本沒有 vtable。`main` 把 `Rect` 建構為區域變數，所以編譯器在編譯期解析了 `r.name()` 和 `describe(r, 2)` 並 inline 了它們。沒有東西取表格的位址，所以表格不會被送出。`describe` 仍然包含 `movq (%rdi), %rax` 和 `callq *0x10(%rax)`，因為它的參數可以是任何 `Shape`。遺失的 vtable 是關於某個 object file 的事實，而測試 `test_the_optimised_object_keeps_the_call_pattern_but_drops_the_vtables` 檢查這兩半。

Microsoft ABI 在這每一個細節上都不同。修飾後的名稱以 `?` 開頭並編碼呼叫慣例，所以編譯成 C++ `__cdecl` 的 `void test(void)` 是 `?test@@ZAXXZ`。RTTI locator 位於 `vfptr[-1]`，作為 `_RTTICompleteObjectLocator`，而 symbol 是 vftable 用的 `??_7` 和 locator 用的 `??_R4`。Microsoft 自己的文件說「在編譯器廠商之間，沒有 C++ 命名的標準」。這堂課的 fixture 只有 Itanium。

## 工具如何呈現還原的型別 (How the Tools Present Recovered Types)

下表中每一列指名輸出的，都是在這台機器上產生並擷取到 `code/fixtures/tools/` 的。這裡沒有安裝的工具，其列描述的是它們的文件，並且這麼說明。

| 工具 | 版本、授權 | 型別存放在哪裡 | 在我們的 fixture 上 |
|---|---|---|---|
| Ghidra | 12.1.4，2026-09-21，Apache-2.0 | 包含內建、使用者定義與衍生型別的 Data Type Manager；`.gdt` archive；程式使用的型別會被複製進去 | 有 DWARF：`/DWARF/account.c/account size=32 align=8` 與 `a->balance = by + a->balance;`。沒有：`*(int *)(param_1 + 4)` 搭配 `long param_1`、`tag(char *param_1)`，以及 `*(char **)(param_1 + 0x18)` |
| IDA Pro | 9.4，2026-07-13，商業；IDA Free 供非商業使用，且有僅限 x86 的雲端反編譯器 | Local Types 視窗、`.til` 型別函式庫、匯入 DWARF 與 PDB 到 `dwarf` 或 `pdb` 資料夾 | 未安裝。僅查閱文件 |
| Binary Ninja | 6.0，2026-09-03，商業，附 Free 版 | 型別編輯、header 匯入與匯出、type archive、來自 DWARF 與 PDB 的偵錯資訊、型別函式庫、平台型別 | 未安裝。僅查閱文件 |
| radare2 | 6.2.4，2026-10-05，LGPLv3 | `ts`、`tsc`、`afvd` | 在 `bin.dbginfo=true` 下，`ts` 和 `tsc account` 對可重定位 object 什麼也沒印出；`pdf` 顯示 `mov dword [rax + 4], ecx` 與來自 DWARF 的原始碼行註解 |
| rizin | 0.9.1，2026-06-29，LGPL 核心 | `ts`、`tsc`、`afvd` | `tsc account` 從 DWARF 印出 `struct account { char kind; int balance; short code; double rate; const char *label; };`；沒有 `-g` 的 build 只列出它內附的 header 型別 |
| lldb | lldb-1703.0.236.21 | `image lookup -t` | 來自 ELF object 的 `byte-size = 32` 與完整的 `compiler_type` 文字 |
| gdb | 此處未安裝 | `ptype /o` 印出「struct 中欄位的 offset 與大小」，並標記空洞 | 僅查閱文件 |
| pyelftools | 0.33，public domain | 位於 `.debug_info` 之上的 Python API | 一個 43 行的腳本印出相同的五個成員與 `unit version 5` |
| llvm-dwarfdump | Apple LLVM 17.0.0 | `--debug-info`、`--debug-abbrev`、`--debug-str-offsets` | 每個 reader 測試的 oracle |

沒有 DWARF 的 Ghidra 輸出，顯示了同樣兩條規則在運作。它的反編譯器為 `tag` 的參數指派 `char *`，因為該函式透過它比較一個位元組，並為位於 `0x18` 的欄位指派 `char **`，因為 `label_length` 對載入的值做 dereference 並比較位元組。這就是這堂課的 pass 所套用的同樣兩條規則：寬度來自指令，指標來自 dereference。Ghidra 的反編譯器文件說，p-code 的 varnode「不一定有與之相關聯的資料型別」，而反編譯器「最終透過傳播來指派正式的資料型別」。選項「Infer constant pointers」把同樣的傳播延伸到常數。Ghidra 12.1 為反編譯器加入了 bitfield 還原，這堂課的 fixture 沒有演練到它。

手工製作的表格是故事的另一半。Ghidra 的 Data Type Manager、IDA 的 Local Types、Binary Ninja 的 type archive 以及 rizin 的 `td`，都讓你根據還原的版面為 struct 設定型別，並觀察反編譯器重新轉譯。Pass 告訴你要設定什麼型別，以及哪些欄位是你在猜的。

搭配 Ghidra provider 的 rea 6.5.0，把反編譯器文字作為紀錄回傳。Ghidra 那一列的所有內容都適用於它，而且該紀錄帶有型別名稱，卻沒有產生它們的證據。對同一份清單執行 `python3 code/main.py infer`，就能給你那份證據。

## 型別推論研究 (Type Inference Research)

這裡的 pass 是型別推論系統中限制產生的那一半，且僅限於從參數到達的一個 struct。這個領域已經走得更遠。

- TIE（Lee、Avgerinos、Brumley，NDSS 2011）把型別重建構想為對 binary IR 的限制產生與求解，量測了保守性與精確度，並在 87 個 coreutils 程式上與商業反編譯器比較。
- Retypd（Noonan、Loginov、Cok，PLDI 2016）加入了 subtyping、polymorphism 與遞迴型別，並回報以 98% 的 recall 還原指標的 `const` 標註。GrammaTech 名下有一個開源實作。
- OSPREY（Zhang 等人，IEEE S&P 2021）把每個記憶體位置視為隨機變數，並推論最可能的型別。
- DIRTY（Chen 等人，USENIX Security 2022）與 ReSym（Xie 等人，CCS 2024）用學習式模型對反編譯器輸出做後處理，而 ReSym 在完整 struct 版面上回報 72.9% 的 precision。
- RecStruct（Chen 等人，USENIX Security 2026）不靠反編譯器，以 flow-insensitive 的 symbolic execution 和對 stack slot 的 union-find 來還原巢狀 struct。

Stack 上的 struct 是這堂課的 pass 跳過的第一件事。在 `-O1`，`main` 把 `rows` 留在 stack 上，而 inline 後的程式碼讀取 `0x24(%rsp)`、`0x40(%rsp)`、`0x48(%rsp)`、`0x30(%rsp)` 和 `0x38(%rsp)`。這些 offset 模 32 是 4、0、8、16 和 24：同一個 struct，沒有可供播種的參數。實作練習要求你還原它。

## 動手做 (Build It)

執行 demo：

```bash
cd phases/13-tools-and-protocols/41-recovering-data-and-types
python3 code/main.py
```

輸出是一份有五個部分的 JSON 報告。`dwarf` 持有 `-O0` object 的 unit header、structure 與函式。`oracles` 持有 psABI 規則版面、`ctypes` 版面、DWARF 版面、原生執行，以及 `agree` 旗標。`builds` 持有三個 object 各自的存取、推論的版面與比較。`cxx` 持有還原的名稱、兩個 vtable、兩筆 typeinfo 紀錄、虛擬 call 與 DWARF slot。`tools` 列出擷取到的工具輸出。

預期的形狀：

```json
{
  "dwarf": {"unit": {"unitLength": 489, "version": 5, "unitType": "DW_UT_compile", "addressSize": 8, "abbrevOffset": 0}, "...": "..."},
  "oracles": {"agree": true, "psabiRules": {"size": 32, "align": 8, "...": "..."}, "...": "..."},
  "builds": {
    "O0": {"layout": {"stride": 32, "size": 32, "gaps": [1, 2, 3, 10, 11, 12, 13, 14, 15], "...": "..."}, "comparison": {"matches": 5, "members": 5, "...": "..."}},
    "O1": {"...": "..."},
    "O2": {"...": "..."}
  },
  "cxx": {"virtualCalls": [{"slot": 2, "candidates": {"Shape": "__cxa_pure_virtual", "Rect": "_ZNK4Rect4areaEv"}, "...": "..."}], "...": "..."}
}
```

依此順序閱讀 `code/main.py`：

1. `ElfObject` 讀取 section、symbol 與 relocation，而 `relocated()` 對偵錯 section 套用 `S + A`。
2. `parse_abbreviations()` 與 `DwarfInfo._parse_unit()` 讀取 header、abbreviation 表格與項目，然後透過兩個表格解析 `strx` 與 `addrx`。
3. `DwarfInfo.struct_layouts()` 與 `functions()` 把項目變成 `StructLayout` 與 `FunctionInfo` 紀錄，搭配 `type_name()`、`type_size()` 與 `type_kind()`。
4. `parse_listing()`、`parse_memory()` 與 `access_width()` 讀取 `llvm-objdump` 的文字。
5. `trace_function()` 是抽象狀態機，而 `fold_unrolled()` 與 `infer_layout()` 把存取變成版面。
6. `compare_layouts()` 產生狀態列，而 `sysv_layout()` 與 `ctypes_layout()` 是另外兩個 oracle。
7. `Demangler`、`read_vtable()`、`read_typeinfo()` 與 `find_virtual_calls()` 涵蓋 C++ 物件。

還有另外四個指令。`python3 code/main.py dwarf FILE.o` 印出任何 DWARF 5 可重定位 object 的 structure 與函式。`python3 code/main.py infer LISTING.txt [FILE.o]` 在清單上執行 pass，並在給定 object 時，與它的 DWARF 比較。`python3 code/main.py demangle NAME...` 與 `python3 code/main.py vtable FILE.o` 涵蓋 C++ 的那一半。

## 實際使用 (Use It)

在你自己建構的新 object 上：

1. 擷取 object 的 `llvm-objdump -d -r`，並把它與 object 放在一起。Pass 讀取那份文字。
2. 在清單上執行 `python3 code/main.py infer`。先閱讀見證者那一行：它指名其第一個參數被用作記憶體 base 的函式。如果某個函式缺少，表示它的參數從未到達記憶體 operand，或是透過 pass 沒有追蹤的路徑到達。
3. 閱讀每個欄位及其存取次數。一次存取就是一個見證。把只有單一見證的欄位記錄為推論。
4. 閱讀 stride 證據。`call-site` 或 `scale` 項目是強的。沒有其他種類、單獨一個 `loop-step` 項目是弱的，因為展開的迴圈回報的是數個元素的步長。如果出現 `unrolled`，請手動檢查折疊。
5. 如果存在 `-g` build，執行比較並閱讀每個不是 `match` 的列。`no-evidence` 列指名沒有任何函式碰過的成員。`unexpected` 列指名 pass 無法安置的存取。
6. 把版面當作 struct 輸入你的工具的型別管理員，並記錄哪些欄位、名稱與種類是猜測。

保持三個 oracle 隨手可得。當規則、`ctypes` 與 DWARF 對你寫的版面意見不一時，你就找到了 packed struct、bit-field，或是規則不同的目標。

## 互動實驗 (Interactive Lab)

### 實驗 A：解碼成員項目

從 `code` 目錄開啟 Python：

```bash
cd phases/13-tools-and-protocols/41-recovering-data-and-types/code
python3 -q
```

執行：

```python
from main import *
elf = load_object("account-x86_64-O0-g.o")
dwarf = DwarfInfo(elf)
dwarf.header()
elf.section(".debug_info").data[0x1a3:0x1ac].hex(" ")
dwarf.dies[0x1a3].attributes
dwarf.type_name(dwarf.type_of(dwarf.dies[0x1a3]))
```

在查看 dictionary 之前，先用 abbreviation 13 在紙上解碼這九個位元組。然後閱讀沒有 relocation 的原始 section `elf.section(".debug_str_offsets").data[8:16].hex()`，以及已套用 relocation 的 `elf.relocated(".debug_str_offsets")[8:16].hex()`，並說明差異。

### 實驗 B：追蹤一個函式

```python
listing = load_listing("account-x86_64-O0-g.objdump.txt")
trace = trace_function("label_length", listing["label_length"])
[(hex(a.address), a.root, a.offset, a.width, a.kind) for a in trace.accesses]
trace.pointer_uses
trace_function("sum_balances", listing["sum_balances"]).loops
```

在執行之前，先寫下是哪次存取證明 offset 24 是指標。然後在 `-O1` 的清單上執行同樣的幾行，並在 `trace.increments` 中找到 `leaq 0x1(%rcx), %rcx`。

### 實驗 C：折疊向量化迴圈

```python
o2 = load_listing("account-x86_64-O2-g.objdump.txt")
trace = trace_function("sum_balances", o2["sum_balances"])
sorted({a.offset for a in trace.accesses})
folded, evidence = fold_unrolled(trace, trace.accesses)
sorted({a.offset for a in folded}), [(e.kind, e.value, e.text) for e in evidence]
```

然後執行 `analyze_build("O2", fold=False)["comparison"]` 並計算 `unexpected` 列的數量。修改折疊規則，要求迴圈步長恰好是 `k * d` 且 `k` 至少為 2，並確認 `-O1` 的迴圈（一次存取、步長 32）不受影響。

### 實驗 D：vtable 與名稱

```python
shapes = load_object("shapes-x86_64-O0-g.o")
read_vtable(shapes, "_ZTV4Rect")
read_typeinfo(shapes, "_ZTI4Rect")
demangle("_ZNK4Rect4areaEv"), demangle("_Z8describeRK5Shapei")
find_virtual_calls(load_listing("shapes-x86_64-O0-g.objdump.txt"), [read_vtable(shapes, "_ZTV5Shape"), read_vtable(shapes, "_ZTV4Rect")])
```

從 `callq *0x10(%rax)` 在紙上算出 slot，然後對照 `DwarfInfo(shapes).struct("Rect").virtual_methods` 確認。接著載入 `shapes-x86_64-O1-g.o` 並對它嘗試 `read_vtable`。閱讀 `ElfError`，並說明為什麼該 symbol 不存在。

## 實作練習 (Practice Lab)

還原 `-O1` 的 `main` 中的 stack struct。

需求：

- 擴充 `trace_function()`，讓 frame slot 區域可以作為 root。用從 `.data.rel.ro` 填滿 `0x0(%rsp)` 到 `0x5f(%rsp)` 的 `movaps` store 來播種，這就是 `rows` 的 96 位元組初始化器。
- 把 inline 後的存取 `0x24(%rsp)`、`0x40(%rsp)`、`0x48(%rsp)`、`0x30(%rsp)` 和 `0x38(%rsp)` 歸屬到該 root，並回報它們對候選 stride 取模的 offset。
- 從 `0x120` 到 `0x12c` 的迴圈推導出 stride 32，其中 `%r14` 從 4 開始，以 `0x20` 前進直到 `0x64`，並顯示迴圈上界 `0x64` 是三個元素加上欄位 offset。
- 將結果與 DWARF 版面比較，達到五個符合，並記錄哪些種類仍未被證明。
- 新增一個測試，當 frame root 被歸屬到某個參數時就失敗。

成功的意思是，在 `-O1` 的清單上執行 `python3 code/main.py infer`，將 `main` 列為帶有 frame root 的見證者，而比較顯示僅從 `main` 就有五個符合。

## 課程成品 (Lesson Artifact)

這堂課包含 `outputs/skill-recovering-data-and-types.md`。用它把清單與選擇性的偵錯 build，變成每個欄位附證據的 struct 版面、附來源的 stride、對照 DWARF 的比較，以及仍然未知的清單。它要求存取表、stride 證據、折疊檢查、存在時的 DWARF oracle，以及關於沒有任何指令能提供的名稱與限定詞的陳述。

## 驗證 (Verify It)

執行 demo 與確定性的測試套件：

```bash
cd phases/13-tools-and-protocols/41-recovering-data-and-types
python3 code/main.py
python3 -m unittest discover -s code/tests -v
```

驗證應該證明：

- unit header、14 個 abbreviation 和五個成員 offset 與 `llvm-dwarfdump` 一致
- 名稱經由已 relocation 的 `.debug_str_offsets` 表格抵達，且 unit 的 `str_offsets_base` 和 `addr_base` 是 8
- 六個函式帶有 dump 所顯示的範圍、`DW_OP_reg6 RBP` 與 `DW_OP_fbreg` 位置，且它們的 low address 等於清單的函式起點
- 沒有 `.debug_info` 的 object 會丟出例外，而 LEB128 解碼器把 `0x78` 處理為 -8
- `pyelftools` 和 `lldb` 印出與 reader 相同的版面
- psABI 規則、`ctypes`、DWARF 與原生執行給出 offset `0, 4, 8, 16, 24` 與大小 32，而重新排序的 struct 是 24 位元組
- 清單 parser 在 `bump` 中讀取 13 條指令，把 `R_X86_64_PLT32 bump-0x4` 附加到位於 `0x194` 的 call，並找到位於 `0x72` 的向後跳轉
- 記憶體 operand、寬度、種類與符號，從 AT&T 文字與 mnemonic 解碼
- `-O0` 的 `bump` 從 frame slot 讀回它的指標，並在 offset 4 做出三次存取
- stride 在 `-O0` 來自 `shlq $0x5`，在 `-O1` 來自 `addq $0x20, %rdx`，在 `-O2` 來自以步長 256 折疊的八個 load
- `label_length` 透過 dereference 證明位於 offset 24 的指標
- `main` 在 `-O0` 於 offset 0、`0x20` 和 `0x40` 傳遞一個 frame 物件，並在 `-O1` 除了 `printf` 之外什麼也不呼叫
- 每個 build 都符合五個成員中的五個，大小為 32，且空隙等於 DWARF 的 padding
- 沒有折疊時，`-O2` 的 build 回報超過位元組 32 的七個欄位，而且沒有大小
- demangler 重現全部 19 個 `llvm-cxxfilt` 字串
- 兩個 vtable、兩筆 typeinfo 紀錄，以及位於 `0x16` 的虛擬 call 都解析到 slot 2，而 DWARF 同意
- `-O1` 的 C++ object 保留 call 的模式，且沒有 vtable
- Ghidra、rizin 與 radare2 的擷取，在有 DWARF 時顯示 struct，在沒有時顯示原始 offset
- 互動式圖表內嵌了分析所產生的相同資料

## Production 的失敗模式 (Production Failure Modes)

| 失敗 | 薄弱的工具回報什麼 | 分析必須證明什麼 |
|---|---|---|
| 展開的迴圈 | 相隔 32 位元組的八個欄位，以及 232 位元組的 struct | 迴圈步長 256 等於 offset delta 32 的 8 倍，所以是 offset 4 的單一欄位 |
| 從一次遞增讀取 stride | 從 `addq $0x100` 得到 `sizeof` 256 | 每個 stride 來源的最大公因數，包含純量尾端的 `addq $0x20` |
| 8 位元組欄位被稱為指標 | offset 24 處的 `void *` | 之後某次存取把載入的值用作 base，否則種類維持為 `int64` |
| 指標被稱為整數 | offset 24 處的 `long` | 位於 `0xfb` 的 dereference，記錄為指標證明 |
| 窄的比較 | 帶符號的 `char` | `cmpb` 不帶符號；只有 `movsbl` 或 `movzbl` 才有 |
| 假設五個函式共用一個型別 | 沒有別名證據的單一 struct | `main` 中傳遞單一 frame 物件的 call site，或明確的分析者斷言 |
| 以 index 從 `.debug_str` 讀取字串 | 來自可重定位 object 的亂碼名稱 | `str_offsets_base` 加上已 relocation 的 `.debug_str_offsets` 項目 |
| 位址被讀成數字 | 每個函式的 `low_pc` 都是 0 | 透過已 relocation 的 `.debug_addr` 表格的 `addrx` |
| DWARF 4 object | 誤解析 header 的 reader | 檢查第二個 header 欄位的 version；4 沒有 `unit_type` 位元組 |
| 遺失的 vtable | 「這個類別沒有虛擬函式」 | `-O1` 的 object inline 了每個 call；`-O0` 的 object 有表格及其 relocation |
| 從錯誤的 base 讀取 vtable slot | 差二的 slot 編號 | 位址點在 symbol 之後 16 位元組，位於 offset-to-top 和 typeinfo 之後 |
| 修飾後的名稱被剝除 | `Z8describeRK5Shapei` 保持原樣 | macOS 上的 `--no-strip-underscore`，或會檢查 `_Z` 的 demangler |
| 從某個工具的猜測設定 struct 型別 | 把 `char *param_1` 當作欄位型別 | 寫成證據的寬度與 dereference，以及標記為猜測的名稱 |

## 營運規則 (Operational Rule)

從存取還原欄位，從迴圈還原 stride，並為每一項附上指令。在你信任超過 stride 之後的 offset 之前，先折疊展開的迴圈。透過 abbreviation 與已 relocation 的表格讀取 DWARF，絕不以 index 讀取 string section。逐欄位比較，並把每個沒有證據的成員、每個未證明的指標與每個名稱都寫為未知。在 C++ object 中，從 relocation 讀取 vtable，從 call 讀取 slot，然後用 DWARF 確認你讀到的 slot。

## 延伸閱讀 (Further Reading)

- [System V Application Binary Interface, AMD64 Architecture Processor Supplement, section 3.1.2: Data Representation](https://gitlab.com/x86-psABIs/x86-64-ABI)
- [Procedure Call Standard for the Arm 64-bit Architecture (AAPCS64), section 5.10: Composite Types](https://github.com/ARM-software/abi-aa/blob/main/aapcs64/aapcs64.rst)
- [Writing ARM64 code for Apple platforms](https://developer.apple.com/documentation/xcode/writing-arm64-code-for-apple-platforms)
- [Intel 64 and IA-32 Architectures Software Developer's Manual, Volume 1, section 3.7.5: Specifying an Offset](https://www.intel.com/content/www/us/en/developer/articles/technical/intel-sdm.html)
- [DWARF Debugging Information Format Version 5, sections 2.1, 5.7.6, 7.5.1.1, 7.5.3, 7.5.6](https://dwarfstd.org/doc/DWARF5.pdf)
- [Itanium C++ ABI, sections 2.5 (Virtual Table Layout), 2.9 (Run-Time Type Information), 5.1 (External Names)](https://itanium-cxx-abi.github.io/cxx-abi/abi.html)
- [Microsoft Learn: Decorated names](https://learn.microsoft.com/en-us/cpp/build/reference/decorated-names?view=msvc-170) and [Name Decoration](https://learn.microsoft.com/en-us/cpp/error-messages/tool-errors/name-decoration?view=msvc-170)
- [llvm-dwarfdump command guide](https://llvm.org/docs/CommandGuide/llvm-dwarfdump.html) and [llvm-cxxfilt command guide](https://llvm.org/docs/CommandGuide/llvm-cxxfilt.html)
- [Python ctypes: Structure layout, sizeof, alignment](https://docs.python.org/3/library/ctypes.html)
- [Ghidra Data Type Manager](https://github.com/NationalSecurityAgency/ghidra/blob/master/Ghidra/Features/Base/src/main/help/help/topics/DataTypeManagerPlugin/data_type_manager_description.htm), [Decompiler Concepts](https://github.com/NationalSecurityAgency/ghidra/blob/master/Ghidra/Features/Decompiler/src/main/help/help/topics/DecompilePlugin/DecompilerConcepts.html), [Decompiler Options](https://github.com/NationalSecurityAgency/ghidra/blob/master/Ghidra/Features/Decompiler/src/main/help/help/topics/DecompilePlugin/DecompilerOptions.html), and [Ghidra 12.1.4 release](https://github.com/NationalSecurityAgency/ghidra/releases/tag/Ghidra_12.1.4_build)
- [IDA 9.1 user guide: Types](https://docs.hex-rays.com/9.1/user-guide/types), [IDA 9.4 release post](https://hex-rays.com/blog/ida-9.4-release-a-new-dyld-shared-cache-swift-analysis-new-teams-add-on-and-more-), and [IDA Free](https://hex-rays.com/ida-free)
- [Binary Ninja user guide: Types](https://docs.binary.ninja/guide/types/) and [Binary Ninja 6.0 Krypton](https://binary.ninja/2026/09/03/binary-ninja-6.0-krypton.html)
- [radare2 6.2.4 release](https://github.com/radareorg/radare2/releases/tag/6.2.4) and [rizin v0.9.1 release](https://github.com/rizinorg/rizin/releases/tag/v0.9.1)
- [GDB manual: Examining the Symbol Table, ptype /o](https://sourceware.org/gdb/current/onlinedocs/gdb.html/Symbols.html)
- [pyelftools on PyPI](https://pypi.org/project/pyelftools/)
- [Lee, Avgerinos, Brumley, "TIE: Principled Reverse Engineering of Types in Binary Programs" (NDSS 2011)](https://www.ndss-symposium.org/ndss2011/tie-principled-reverse-engineering-of-types-in-binary-programs/)
- [Noonan, Loginov, Cok, "Polymorphic Type Inference for Machine Code" (PLDI 2016)](https://arxiv.org/abs/1603.05495) and [the retypd repository](https://github.com/GrammaTech/retypd)
- [Zhang et al., "OSPREY: Recovery of Variable and Data Structure via Probabilistic Analysis for Stripped Binary" (IEEE S&P 2021)](https://ieeexplore.ieee.org/document/9519451/)
- [Chen et al., "Augmenting Decompiler Output with Learned Variable Names and Types" (USENIX Security 2022)](https://www.usenix.org/conference/usenixsecurity22/presentation/chen-qibin)
- [Xie et al., "ReSym: Harnessing LLMs to Recover Variable and Data Structure Symbols from Stripped Binaries" (CCS 2024)](https://dl.acm.org/doi/10.1145/3658644.3670340)
- [Chen et al., "RecStruct" (USENIX Security 2026)](https://www.usenix.org/conference/usenixsecurity26/presentation/chen-yuxin)
