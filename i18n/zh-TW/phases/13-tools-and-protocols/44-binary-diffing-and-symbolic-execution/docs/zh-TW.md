# 二進位差異比對與符號執行 (Binary Diffing and Symbolic Execution)：比對函式並求解輸入

> 兩個函式之間的比對，是由一個簽章 (signature) 與一種品質等級共同提出的主張；求解出的輸入，則是某條路徑條件 (path condition) 的一個模型。本課兩者都親手建出來，並與 radiff2、Ghidra、angr、z3 對照檢查。

**Type:** Build
**Languages:** Python, C, x86-64 assembly, Java (one Ghidra script)
**Prerequisites:** Phase 13 · 33 (disassembly and control flow), 34 (lifting to pseudocode), 36 (rebuild and prove equal)
**Time:** ~150 分鐘

## 學習目標 (Learning Objectives)

- 從一個目的檔 (object) 與其反組譯清單建出五種函式簽章：完整位元組、遮蔽重定位欄位後的位元組、助記符 (mnemonic) 序列、流程圖形狀、符號名稱。
- 以 BinDiff 風格執行分階段的貪婪比對器 (greedy matcher)，只有簽章在兩側都唯一時才算比對成功，並記錄每一次比對背後的階段與品質。
- 重現兩個位元組完全相同的函式在載入映像 (loaded image) 上通不過位元組相關器 (byte correlator)、遮蔽後卻能通過的案例，並以重定位算術解釋那四個改變的位元組。
- 由 Levenshtein 距離重現 radiff2 的相似度比值，並說明為什麼其中一個比值只有在比較使用 radare2 實際取得的 82 個位元組時才會出現。
- 在小型 IR 上列舉函式的所有路徑，為每條路徑攜帶一個條件，並以有界搜尋為每個葉節點求解輸入。
- 在帶有符號化上界的迴圈上量測路徑爆炸 (path explosion)，並將路徑數量、類別與模型與 angr 10.0.3 和 z3 5.1.0 比較。

## 問題 (The Problem)

你以 `-O2` 重新建置了 `ledger.c`，並問代理 (agent) 自 `-O1` 建置以來有沒有任何改變。代理執行了一個差異比對工具，回報 `grade` 被修改了，因為它在兩個載入程式中的位元組不同。接著它把預算都花在解釋一個根本不存在的改變上。`grade` 的 46 個位元組在兩個目的檔中完全相同。差異比對工具比較的是重定位之後的映像，而 `.rodata` 在 `-O2` 程式中往後移了 0x70 個位元組，所以 `grade` 裡的一個位移量 (displacement) 增加了 0x70。

同一個下午，你問哪個分數會讓 `grade` 回傳 4。代理讀了偽程式碼後回答「90」。這是一個答案。這個函式對 90 到 109 的每個分數都回傳 4，而問題並沒有設定界限，所以代理的答案只是一個樣本，而你需要的是一個集合。

兩次失敗來自同一個缺口。比對與求解出的輸入，各自都是背後帶有方法的主張。本課以標準函式庫的 Python 建出這兩種方法，並把正式的工具並排執行。工具與程式碼一致時，你就有了兩種推導；兩者不一致時，你就有了一個位址、一個位元組或一個界限可以去檢查。

```figure
re-44-matches-and-paths
```

## 函式簽章是帶有品質等級的主張 (A Function Signature Is a Claim With a Quality)

Flake 在 2004 年把二進位差異比對界定為：在兩個相似的可執行檔的函式集合之間建構同構 (isomorphism)。Dullien 與 Rolles 在 2005 年將它延伸到基本區塊 (basic block) 與指令。BinDiff 的概念頁面描述了由這項工作產生的策略：為兩側的每個函式計算一個屬性。在每一側都恰好出現一次的值就是一次比對。出現超過一次的值是有歧義的，會在該等價類別內以下一個屬性往下細分。沒有對應項的值則保持未比對。每一輪之後，新比對的呼叫者 (caller) 與被呼叫者 (callee) 會形成小集合，其中下一個屬性很可能是唯一的，然後流程以下一個屬性重複進行。

每個比對演算法都帶有一個品質等級。該頁面將原始函式位元組的雜湊比對與名稱雜湊比對評為非常好 (very good)，將流程圖 MD 索引與質數簽章比對（每個助記符對應一個小質數，對整個函式取乘積）評為好 (good)，將字串參照評為中等 (medium)，將迴圈數或位址序列評為差 (poor)。比對的信心度是產生它的演算法之平均品質，以 sigmoid 加權。函式相似度大約是 25% 的已比對流程圖邊、15% 的已比對基本區塊、10% 的已比對指令，以及 50% 的流程圖 MD 索引差異，再乘以信心度。

本課保留其中五個屬性與唯一重要的規則：兩側皆唯一，否則不比對。它捨棄 MD 索引，因為概念頁面只點名而沒有給出公式，並改用單純的度數序列 (degree sequence) 代替。品質標籤保留下來，因為一次比對的可靠程度，只取決於產生它的最弱事實。

## 一筆函式紀錄產生五種簽章 (Five Signatures From One Function Record)

差異比對工具對每個目的檔讀取兩樣東西。ELF 提供位元組、符號大小與 `.rela.*` 紀錄。`llvm-objdump -d` 清單提供指令邊界與助記符，而第 33 課已證明解碼器在這些目的檔的每一道指令上都與該清單相等。`function_record()` 會檢查清單中每道指令的位元組是否等於目的檔在該位址的位元組，否則就拋出例外。過時的清單就等於被改動過的測試資料 (fixture)。

以 `ledger-x86_64-O1.o` 中的 `grade` 為例。符號表指出偏移量為 0x10、大小為 46，所以這筆紀錄涵蓋 0x10 到 0x3e，並排除了用來對齊下一個函式的兩位元組 `nop`。

完整位元組簽章 (exact-bytes signature) 是這 46 個位元組的 SHA-256。遮蔽位元組簽章 (masked-bytes signature) 則先把每個重定位欄位歸零。這裡只有一個欄位：偏移量 0x36 處的紀錄 `R_X86_64_PC32 .rodata+0x10`，寬度四個位元組，是位於 0x33 的 `leaq (%rip), %rcx` 的位移量。在 RELA 目的檔中，這四個位元組本來就是零，所以這兩種簽章在此處每個目的檔的每個函式上都一致。這個遮蔽是為下一節而存在的。

助記符簽章 (mnemonic signature) 對 14 個助記符的序列取雜湊：

```text
movslq imulq movq sarq shrq addl addl xorl cmpl ja movl leaq movl retq
```

流程圖形狀簽章 (flow-graph shape signature) 來自第 33 課的基本區塊與具型別的邊，並限制在入口可到達的區塊。`grade` 在 `-O1` 有三個區塊與三條邊。區塊 0x10 以 `ja 0x3d` 結尾，帶有一條跳轉邊 (taken edge) 與一條直通邊 (fallthrough edge)，區塊 0x31 直通到 0x3d，而 0x3d 負責回傳。度數序列依排序後的順序列出每個區塊的 (in, out)，所以簽章讀作 `3b 3e [0/2,1/1,2/0]`。對於 `-O0` 的 `grade`，這筆紀錄如同第 33 課那樣，透過五筆 `.rela.rodata` 項目解析間接跳轉，得出八個區塊與十一條邊，其中有四條 `table` 邊。

名稱簽章 (name signature) 就是符號名稱。它存在時是最強的事實，而當建置已被剝除符號 (stripped) 時就毫無作用。

後面的修補 (patch) 案例還會用到另一個欄位。當函式具有指向 `.rodata` 的 `lea` 重定位、一個邊界檢查與一次索引載入時，`referenced_table()` 會依第 36 課的做法讀取該表：`ja` 之前的 `cmpl $0x4` 給出五個項目，而載入 `movl (%rcx,%rax,4), %eax` 給出每項四個位元組。對 `-O1` 的 `grade`，位於 `.rodata+0x14` 的表讀作 `1, 2, 3, 4, 4`。

## 貪婪比對器 (The Greedy Matcher)

`match_functions()` 依序執行各個階段。每個階段會依該階段的簽章，把每一側仍未比對的函式分組，並接受每個在兩側各恰有一個函式的值。比對結果會記錄該階段、該階段的品質、同樣一致的其他每一種簽章，以及一個相似度比值。

```python
for stage in order:
    for signature, candidates in left_groups.items():
        partners = right_groups.get(signature, [])
        if len(candidates) != 1 or len(partners) != 1:
            continue
        a, b = candidates[0], partners[0]
        agreeing = [s.id for s in STAGES if a.signatures()[s.id] == b.signatures()[s.id]]
        matches.append(Match(a, b, stage, agreeing, radare2_ratio(a.raw, b.raw), data_equal))
```

本課的順序先執行四個結構性階段、最後才執行名稱，使結構性階段先接受測試，之後再由名稱來評分。以 `-O1` 對 `-O2` 為例：

| 函式 | 階段 | 一致的簽章 | 相似度 |
|---|---|---|---|
| `percent_change` | exact bytes | 五種全部 | 1.000 |
| `grade` | exact bytes | 五種全部 | 1.000 |
| `total` | name | 僅名稱 | 0.171 |
| `grade_label` | exact bytes | 五種全部 | 1.000 |

`total` 從 31 個位元組、13 道指令成長為 140 個位元組、41 道指令，因為 clang 以 SSE2 將迴圈向量化，並保留了純量的餘數迴圈。它的形狀從五個區塊、五條邊變成十個區塊、十三條邊。沒有任何結構性的東西留存下來，只剩符號把兩者連在一起。

對於 `-O0` 對 `-O1`，同樣的順序會產生一次錯誤的比對：

| `-O0` 的函式 | 在 `-O1` 比對到 | 階段 | 名稱是否一致 |
|---|---|---|---|
| `percent_change` | `grade_label` | flow-graph shape | 否 |
| `grade` | `grade` | name | 是 |
| `total` | `total` | name | 是 |
| `grade_label` | 未比對 | | |

`percent_change` 在 `-O0` 是個四區塊的菱形：一個以 `before == 0` 分支的入口、兩個分支臂，以及一個回傳的匯合點。它的度數序列是 `0/2, 1/1, 1/1, 2/0`。`-O1` 的 `grade_label` 將 `grade` 行內展開 (inline)，並在邊界檢查周圍有同樣的菱形。這個形狀在兩側都唯一，所以規則接受了它。中等品質的標籤就是警告，而之後執行的名稱階段與它相矛盾。這次錯誤的比對還耗用了右側的 `grade_label`，使真正的 `percent_change` 與 `grade_label` 配對都保持未比對。

當真實名稱存在時，BinDiff 的順序會把名稱比對排在第二階段，程式碼中的 `BINDIFF_ORDER` 做的是同一件事。以該順序，四個 `-O0` 函式全都依名稱比對到其 `-O1` 版本，這也是下文 Ghidra 回報的結果。本課在示範中保留兩種順序，讓失敗的情形看得見。四個節點的流程圖形狀，就是每個在回傳前匯合的 `if` 與 `else` 的形狀。

## 位元組相同、運算元被重定位 (Identical Bytes, Relocated Operand)

Ghidra 在匯入目的檔時會套用重定位。因此它的位元組相關器比較的是載入映像，而第 36 課正是為了這個原因遮蔽重定位欄位。`load_image()` 重現了這個匯入過程。它依標頭順序走訪所有被配置的區段 (section)，將每個區段對齊到其 `sh_addralign`，並從 0x100000 開始排列。然後套用每一筆紀錄：對 `R_X86_64_PC32` 與 `R_X86_64_PLT32` 寫入有號 32 位元值的 `S + A - P`，對 `R_X86_64_64` 寫入 64 位元的 `S + A`。

這個排列與 Ghidra 12.1.4 在兩個程式中的每個區段起點都一致，測試套件會對照所擷取的 Version Tracking 輸出來檢查：

| 區段 | `-O1` 程式 | `-O2` 程式 |
|---|---|---|
| `.text` | 0x100000，159 個位元組 | 0x100000，271 個位元組 |
| `.rodata` | 0x1000a0 | 0x100110 |
| `.rodata.str1.1` | 0x1000f0 | 0x100160 |

`.text` 在 `-O2` 長了 112 個位元組，也就是 `total` 的增長，所以 `.rodata` 的起點往後移了 0x70 個位元組。`grade` 在兩個程式中都位於 0x100010。它的 `leaq` 在偏移量 0x33，位移量欄位在 0x36，結束於 0x3a。重定位的加數 (addend) 0x10 加上慣例的 4，得出表位於 `.rodata+0x14`。

在 `-O1` 映像中：`0x1000a0 + 0x14 - 0x10003a = 0x7a = 122`。在 `-O2` 映像中：`0x100110 + 0x14 - 0x10003a = 0xea = 234`。欄位位元組在一個映像中是 `7a 00 00 00`，在另一個中是 `ea 00 00 00`。46 個位元組中有一個不同，Levenshtein 比值為 0.978，而 exact-bytes 階段失敗。masked-bytes 階段將該欄位歸零後通過。`grade_label` 與它的表一同移動了相同的 0x70，所以它的位移量保持相等，在兩種視角下都能以完整位元組通過。

Ghidra 回報了同樣的分歧。在 `-O1` 對 `-O2` 上，Exact Function Bytes Match 找到 `percent_change` 與 `grade_label`，Exact Function Instructions Match 找到這兩個加上 `grade`，Exact Function Mnemonics Match 找到相同的三個。指令相關器會遮蔽運算元，這就涵蓋了位移量。本課的映像視角給出的 exact-bytes 比對，恰好是 Ghidra 位元組相關器找到的那兩個函式，而 masked-bytes 一致的，恰好是它的指令相關器找到的那三個。測試套件斷言了這兩個相等關係。

rea 6.5.0 的 `compare_functions` 以逐行方式比對提供者文字。在組合語言維度上，它附上了「relocation normalization is unavailable」這項限制。本課的載入器與遮蔽提供了這種正規化。

## radiff2 量測什麼 (What radiff2 Measures)

radare2 6.2.4 附帶 `radiff2`。加上 `-AC` 時，它會分析兩個檔案，並對每個函式輸出一列：左側的大小與位址、MATCH 或 UNMATCH、一個比值，以及右側的大小與位址。針對我們的目的檔所擷取的各列如下：

| 配對 | `percent_change` | `grade` | `total` | `grade_label` |
|---|---|---|---|---|
| `-O0` vs `-O1` | UNMATCH 0.130435 | UNMATCH 0.426829 | UNMATCH 0.106061 | UNMATCH 0.269841 |
| `-O1` vs `-O2` | MATCH 1.000000 | MATCH 1.000000 | UNMATCH 0.171429 | MATCH 1.000000 |
| `-O1` vs `-O1` | MATCH 1.000000 | MATCH 1.000000 | MATCH 1.000000 | MATCH 1.000000 |

radare2 原始碼中的比值是 `1 - distance / max(len_a, len_b)`，其中 distance 是兩個函式指紋 (fingerprint) 之間的 Levenshtein 距離，而指紋是函式的基本區塊依位址順序串接而成。比對門檻是 0.7。`code/main.py` 中的 `radare2_ratio()` 對原始函式位元組實作了相同的公式，並將 0.130435、0.106061、0.269841 與 0.171429 重現到小數點後六位。

它無法重現 `-O0` 的 `grade` 的 0.426829。以該函式的 118 個位元組計算，比值是 0.296610。radare2 在大小欄位印出 `118`，但它的指紋較短。沒有套用重定位時，它無法解析跳轉表，所以它的分析只到達入口區塊、以 `jmpq *%rax` 結尾的區塊、預設情況 (default case) 與回傳：共 82 個位元組，與第 33 課的單純遞迴下降 (recursive descent) 所宣稱的是同樣的 82 個。這 82 個位元組與 `-O1` 的 46 個位元組之間的 Levenshtein 距離是 47，而 `1 - 47/82 = 0.426829`。使用 `-e bin.relocs.apply=true` 時，radare2 找到全部 118 個位元組並印出 0.288136，這等於本課在重定位後映像上的比值，其中四個位移量位元組與目的檔的零欄位不同。兩個工具只有在比較同一批位元組時才會一致，而大小欄位並不會告訴你輸入的是哪些位元組。

`-O` 將指紋限制在運算碼 (opcode) 位元組，並將運算元歸零。`-O0` 對 `-O1` 的比值升至 0.239130、0.512195、0.303030 與 0.523810，而 `-O1` 對 `-O2` 的 `total` 升至 0.192857。整個檔案的距離也存在：`-s` (Myers) 回報兩個目的檔之間的相似度為 0.956、距離為 162，`-ss` (Levenshtein) 則為 0.927 與 137。這些數字包含標頭、符號表與重定位區段，對函式什麼也說不上。

## Ghidra Version Tracking 相關器 (Ghidra Version Tracking Correlators)

Ghidra 的 Version Tracking 工具用相關器 (correlator) 把來源程式與目的程式配對。函式相關器有 Exact Function Bytes Match、Exact Function Instructions Match（運算元被遮蔽）、Exact Function Mnemonics Match，以及 Duplicate Function Instructions Match (1:N 與 N:M)。資料相關器有 Exact、Duplicate 與 Similar Data Match。符號相關器以完全相同、重複或相似的方式比對名稱。參照相關器 (Reference correlator) 利用已接受的比對去找出更多比對。函式必須已在目的程式中定義，且最小函式長度預設為 10 個位元組。Auto Version Tracking 依序執行 Exact Symbol Name、Exact Data、Exact Function Bytes、Exact Function Instructions、Exact Function Mnemonics、Duplicate Function Instructions，然後是參照相關器，並為已接受的比對建立隱含比對 (implied match)。11.2 版新增了 BSim 相關器。

本課以無頭 (headless) 模式擷取相關器。`capture_ghidra_vt.sh` 把三個目的檔匯入專案並執行 `ListVersionTrackingMatches.java`，這是一個繼承 `AbstractGhidraVersionTrackingScript` 的短腳本，它依路徑開啟來源與目的程式，依 Auto Version Tracking 的順序執行六個非參照相關器，並為每個區塊、函式、相關器與比對印出一行。本節中的每個數字都來自那兩個文字檔。

| 相關器 | `-O0` 到 `-O1` | `-O1` 到 `-O2` |
|---|---|---|
| Exact Symbol Name Match | 4 個函式 | 4 個函式、5 個資料項目 (`labels.rel`、`.L.str.1` 到 `.L.str.4`) |
| Exact Data Match | 0 | 0 |
| Exact Function Bytes Match | 0 | 2 (`percent_change`、`grade_label`) |
| Exact Function Instructions Match | 0 | 3（加上 `grade`） |
| Exact Function Mnemonics Match | 0 | 3 |
| Duplicate Function Instructions Match | 0 | 0 |

每個函式在 `-O0` 與 `-O1` 之間都改變了，所以只有名稱比對成功，這就是本課的比對器在 BinDiff 順序下給出的結果。每一筆比對都印出相似度 1.000 與信心度 10.000。說明頁面把 exact 相關器描述為相似度 1.0、信心度 1.0，所以所印出信心度的尺度是個懸而未決的問題，本課將它記錄下來而不加以解決。

Tools 底下的 Program Differences 是另一回事。它在相同位址空間的兩個程式之間，比較位於相同位址的位元組、標籤、程式碼單元、參照、註解與函式。它是依位址對齊的差異比對，對任何移動過的東西都比對不到。

## Diaphora 與 Binary Ninja，依其文件所述 (Diaphora and Binary Ninja, From Their Documentation)

Diaphora 是 IDA Pro 的外掛，2015 年首次釋出，截至 2026-08-18 為 3.4.2 版。自 2.0 起採用 AGPL-3.0，並需要 IDAPython，所以它在 IDA Home 或 IDA Pro 中執行，而不能在 IDA Free 中執行。它把每個資料庫匯出成 SQLite，並以 SQL 查詢執行其啟發式方法 (heuristic)。每個啟發式方法有名稱、Best、Partial 或 Unreliable 的類別、比值類型，以及如 SAME_CPU 或 SLOW 的選用旗標。目前原始碼中的名稱讀起來就像本課的各個階段，只是數量更多：Bytes hash、Same address and mnemonics、Same cleaned assembly、Mnemonics small-primes-product、Same nodes, edges, loops and strongly connected components、Topological sort hash，以及需要 Hex-Rays 反編譯器的偽程式碼模糊雜湊。它的發行說明展示了調校的歷史：慢速的啟發式方法只在函式數量低於約 2,000 時執行，而「Same graph」被移到 unreliable 類別。這個移動，就是本課那個四區塊菱形在大規模下的樣子。

2026-09-03 釋出的 Binary Ninja 6.0 新增了 Binary Similarity 框架，包含提供者 (provider) 與解析器 (resolver)。它隨附 BinDiff 作為結構性提供者，並附帶 WARP，用於容忍可重定位運算元的精確函式比對；它的指南說，當函式除了可重定位指令外都吻合時使用 WARP，而當函式在重新編譯後仍保持結構相關時使用 BinDiff。此功能僅在 Ultimate 版本中提供。Binary Ninja 在這裡沒有套件管理員可安裝，也沒有執行過，所以這一段只來自其文件，別無其他。

BinDiff 本身自 2023-09-25 起以 Apache-2.0 開源，最新釋出版本為 v8。它需要一個反組譯器和 BinExport 外掛，後者支援 IDA Pro 9.4 或更新版本、Binary Ninja 與 Ghidra。學習者的免費路徑是 Ghidra 加 BinExport 再加 BinDiff 命令列。本課沒有執行它，而它的 Practice Lab 要求你去執行。

## 相同的程式碼，不同的常數 (Same Code, Different Constant)

第 36 課保留了一個 `grade` 的候選重建，其表的最後一項是 3，而原版是 4。`grade_case10_wrong.o` 與 `-O1` 的 `grade` 有相同的 46 個程式碼位元組。比對器以完整位元組將它們配對，且每種簽章都一致。參照表檢查透過一個重定位讀到 `1, 2, 3, 4, 4`，透過另一個讀到 `1, 2, 3, 4, 3`，並回報 `dataEqual: false`。其他三個 `-O1` 函式在左側未比對，因為候選目的檔只包含一個函式。

這就是修補分析 (patch analysis) 的形態。只看到位元組為止的函式差異比對工具會說什麼都沒改變。Ghidra 的 Exact Data Match 與 BinDiff 的字串參照比對正是為了同樣的理由而存在。本課的版本很小：對每一對已比對的函式，比較各函式的重定位所指向的位元組，項目數取自邊界檢查，寬度取自載入指令。

## 符號執行：路徑、條件與求解器 (Symbolic Execution: Paths, Conditions, Solvers)

King 在 1976 年描述了符號執行。輸入變成符號。每道敘述更新一個符號儲存區 (symbolic store)。每個分支為路徑加上一個條件，由求解器 (solver) 判斷該路徑是否可行，並產生一個遵循該路徑的具體輸入。帶有分支的函式會變成一棵路徑樹，而葉節點是附帶一個條件的回傳。

Baldoni、Coppa、D'Elia、Demetrescu 與 Finocchi 在 2018 年對此領域做了綜述。他們的第 5 節指出路徑爆炸是核心成本：執行器可能在每個分支分叉，所以狀態數量可能隨分支數呈指數成長，而界限取決於符號輸入的迴圈則給出無界的路徑集合。他們列出的緩解方式有：在每個分支求解以剪除不可實現的路徑 (unrealizable path)、為函式與迴圈建立摘要 (summary)、狀態合併 (state merging)，以及欠近似 (under-approximation)。具體符號執行 (concolic execution) 同時以具體與符號方式執行程式，記錄具體執行過程中的條件，將其中一個取反，然後請求解器給出下一個輸入。DART (2005)、CUTE (2005) 與 SAGE (2008) 是有名的系統。

KLEE (Cadar、Dunbar、Engler，OSDI 2008) 是建立在 LLVM bitcode 上的符號虛擬機，所以它的輸入是 `clang -emit-llvm` 的輸出而不是二進位檔。它附帶 POSIX 環境模型與重播函式庫。在 89 個 GNU coreutils 程式上，它達到 84.5% 的行覆蓋率，而開發者的測試是 67.7%，並找到 56 個嚴重的錯誤。KLEE 3.2 於 2025-12-23 以 NCSA 授權釋出，建議搭配 LLVM 16。

angr 處理二進位檔。`angr.Project` 持有一個映射目的檔並套用重定位的 CLE 載入器、一個把區塊轉成 VEX IR (`block.vex`) 的 lifter（以 pypcode 作為替代），以及一個狀態工廠。`SimState` 持有暫存器、記憶體與一個求解器。`SimulationManager` 逐步推進名為 `active`、`deadended`、`found`、`avoided` 與 `errored` 的狀態堆疊 (stash)，而 `explore(find=..., avoid=...)` 是對它的包裝。每一步各引擎依序執行：failure、syscall、hook、處理具體資料的 unicorn，以及作為後備的 VEX。claripy 是 Z3 之上的求解器前端，在 angr 10.0 中它是一個內嵌的 Rust 實作，以 `from angr import claripy` 匯入。angr 10.0.3 於 2026-10-10 上傳，採用 BSD-2-Clause，需要 Python 3.12 或更新版本。2026-08-16 以 MIT 釋出的 Z3 5.1.0，是 angr、Triton 以及 KLEE 選用後端背後的求解器。

## 建立在第 34 課 IR 之上的小型執行器 (A Tiny Executor Over the Lesson 34 IR)

`code/main.py` 中的執行器使用一種小型 IR，其敘述形態與第 34 課相同：`Assign`、`Branch`、`Jump`、`Return`，外加第 34 課 Practice Lab 要求的 `Switch` 終結指令 (terminator)。運算式有 `Const`、`Var`、帶寬度的 `Bin`、`Cast`、從具名具體表格載入的 `Load`，以及 `Not`。四個測試資料函式是依第 34 課在簡化後所達到的層級手動提升 (lift) 而成，其中 `0x66666667` 的乘法已變成 `score / 10`，且每個 IR 區塊都列出它所來自的指令位址。`check_ir_against_listing()` 斷言這些位址恰好涵蓋函式的清單一次。

`-O0` 的 `grade` 在該 IR 中如下：

```text
0x30: local_8 = score; local_10 = score / 10 - 6; if (local_10 >u 4) goto 0x9a else 0x62
0x62: switch local_10 -> 0x91, 0x88, 0x7f, 0x76, 0x76
0x76: local_4 = 4; goto 0xa1        0x7f: local_4 = 3; goto 0xa1
0x88: local_4 = 2; goto 0xa1        0x91: local_4 = 1; goto 0xa1
0x9a: local_4 = 0                    0xa1: return local_4
```

switch 的目標來自 `jump_table_targets()`，與第 33 課相同的重定位算術，所以這個 IR 讀的是測試資料。

`execute()` 維護一個路徑狀態的工作清單 (worklist)。狀態包含目前區塊、區塊的軌跡 (trace)、條件清單，以及一個把每個變數對應到輸入上之運算式的環境。`Assign` 把環境代入來源並儲存結果。`Branch` 把環境代入條件，然後詢問求解器兩次：一次加入該條件，一次加入它的否定。求解器無法滿足的一側會被丟棄，這就是綜述中所描述的積極剪枝 (eager pruning)。`Switch` 依目標將表格項目分組，並對每個不同的目標各分叉一次，附帶各項目相等式的析取 (disjunction)，angr 處理符號化跳轉目標的方式也是如此。`Return` 把環境代入回傳值並關閉葉節點。軌跡超過步數上限時，該路徑會被標記為截斷並停止。

用手追蹤通往 `return 4` 的路徑。入口指派 `local_10 = score / 10 - 6`。分支條件 `local_10 >u 4` 在這條路徑上為假，所以條件清單持有 `!(score / 10 - 6 >u 4)`。switch 的第 3 與第 4 項都指向 0x76，所以分叉到 0x76 會加上 `score / 10 - 6 == 3 || score / 10 - 6 == 4`。區塊 0x76 指派 `local_4 = 4`，跳到 0xa1，而回傳代入後得到 `4`。葉節點就是該條件與該值所成的一對。

求解器是 `BoundedSearch`。它為每個輸入持有一個有限的值域，列舉各值域的乘積，並以 C 語意計算每個約束：32 位元環繞 (wraparound)、截斷除法、有號與無號比較，以及算術或邏輯位移。除以零或表格索引超出表格，會使該次賦值不滿足約束。對於 -200 到 1200 的 `score`，`grade` 的五條路徑花費 44,460 次求值與數毫秒。為每個葉節點求出輸入集合，得到：

| 葉節點 | 回傳 | 條件 | 界限內的分數 |
|---|---|---|---|
| 0x76 | 4 | `!(q >u 4) && (q == 3 \|\| q == 4)` | 90 到 109，20 個輸入 |
| 0x7f | 3 | `!(q >u 4) && q == 2` | 80 到 89，10 個輸入 |
| 0x88 | 2 | `!(q >u 4) && q == 1` | 70 到 79，10 個輸入 |
| 0x91 | 1 | `!(q >u 4) && q == 0` | 60 到 69，10 個輸入 |
| 0x9a | 0 | `q >u 4` | 1,351 個輸入，界限內其餘的部分 |

`q` 代表 `score / 10 - 6`。`-O1` 的 `grade` 也得出同樣的類別，那裡的 IR 有兩條路徑，表格路徑回傳 `lookup_14[q]`，預設路徑回傳 0，而問題變成對回傳值的約束，因為沒有可到達的 case 區塊。表格 `lookup_14` 是從目的檔 `.rodata` 的 0x14 讀取的。

界限就是這個方法的極限。有界搜尋在它的值域之內是完備的，在值域之外則保持沉默。在 -200 到 1200 之內，60 到 109 中沒有任何分數回傳 0，110 或以上也沒有任何分數回傳非零值。z3 對全部 2^32 個分數證明了這兩點，而下一節會顯示這兩種方法在哪裡回傳了不同的見證 (witness)。

## 量測路徑爆炸 (Path Explosion Measured)

`-O1` 的 `total` 就是第 33 課的迴圈：對 `count` 的測試、一個加上 `values[rdx]` 並遞增 `rdx` 的迴圈本體，以及在 `rdx != rcx` 時 `jne` 回到本體。在 IR 中，`rcx` 是 `zext64(count)`，而 `rdx` 從 0 開始。每次經過回邊 (back edge) 都會在 `rdx != count` 上分叉，其中 `rdx` 是常數，所以 `count = c` 的路徑會執行本體 `c` 次然後離開。當 `count` 限制在 0 到 2 時，執行器找到 3 條路徑。限制在 0 到 4 時找到 5 條，限制在 0 到 8 時找到 9 條。每條路徑都回傳其計數的三角形數和：(0, 0)、(1, 1)、(2, 3)，直到 (8, 36)。angr 在三個最佳化層級（包括 `-O2`）都回報 3、5 與 9 個 deadended 狀態，以及相同的九組計數與總和配對；`-O2` 的向量迴圈需要至少 8 的計數，所以在這個範圍內從不改變路徑數。

沒有界限時執行器永遠不會結束，因為每多一次迭代就多一條可行路徑。本課每條路徑 24 個區塊的上限，在 `count` 最高到 64 的情況下，回傳 22 條路徑、截斷 2 條，並且從未啟動存在的 65 條中的其他 41 條。報告給出這三個數字。求解成本也隨之成長：界限 2 時 33 次求值，界限 4 時 115 次，界限 8 時 531 次，而有上限的那次執行為 6,900 次。

## angr 與 z3 回答同樣的問題 (angr and z3 on the Same Questions)

`capture_angr.py` 對三個目的檔執行了 angr 10.0.3。CLE 以 ELF 後端將每個可重定位目的檔載入到 0x400000，為 `-O0` 套用 13 筆重定位、其餘為 8 筆，並解析了四個符號。對於帶有一個符號化 64 位元引數的 `grade`，`run()` 在 `-O0` 以 5 個 deadended 狀態結束，在 `-O1` 與 `-O2` 為 2 個，耗時分別為 0.475、0.408 與 0.318 秒。對每個 deadended 狀態加上 `eax == v`，並請求解器給出有號最小值與最大值，在每個層級中，`v` 從 1 到 4 分別得到 60 到 69、70 到 79、80 到 89 與 90 到 109，而 0 得到 -2147483648 到 2147483647。兩個交叉檢查皆為 unsat。

到達區塊 (reach-a-block) 的形式對 `-O0` 的 `grade` 使用 CFGFast，列出第 33 課找到的八個區塊，然後以 `explore(find=0x400076)` 尋找儲存 4 的區塊。它在 0.47 秒內找到一個狀態，另有三個仍為 active，引數的最小值為 90、最大值為 109，樣本為 91、93、94、97 與 101。

對於 `percent_change`，有兩個 deadended 狀態，其約束為 `before[31:0] != 0` 與 `before[31:0] == 0`。對非零狀態詢問 `eax == 50`，在 `-O0` 得到 `before = 21055744, after = 332315210`，在 `-O1` 得到 `40566804, 834296877`，在 `-O2` 得到 `-24484867, 993934694`。-16 到 16 內的有界搜尋回答 `before = -10, after = -15`。z3 對第 34 課的清單詢問同樣的問題，回答 `before = -25092599, after = 305712160`。測試套件用 IR 求值器計算全部五組配對，每一組都得到 50。模型證明有輸入存在，但它不是那個輸入，三個正確的工具給出了三個不同的答案。

`capture_z3.py` 以 32 位元位元向量 (bit-vector) 對 `-O1` 的 `grade` 清單建模，包含乘法、兩次位移與表格，並再問四個問題。清單對某個 32 位元分數是否與原始碼的 `switch` 不同：0.068 秒內 unsat，所以編譯後的函式在全部 2^32 個輸入上都等於原始碼。清單與第 36 課的門檻式重建是否不同：sat，分數為 706346536，此時 `score / 10` 為 70634653，switch 落入預設情況，而門檻式回傳 4。把同一個差異限制在 0 到 109：unsat，這就是第 36 課的網格看不見它的原因。對於 `percent_change`，`before == 3` 且結果為 -133 是 sat，`after = 1073741823`，即第 36 課的溢位格 (overflow cell)，而有界搜尋在它的值域內找到第二個見證 `after = -1`，此時 `(-1 - 3) * 100 / 3` 截斷為 -133，完全沒有溢位。

## 動手做 (Build It)

執行示範：

```bash
cd phases/13-tools-and-protocols/44-binary-diffing-and-symbolic-execution
python3 code/main.py
```

輸出是一份 JSON 報告。`diffs` 持有 `-O0` 對 `-O1` 與 `-O1` 對 `-O2` 的目的檔與載入映像比對，外加 BinDiff 順序下的 `-O0` 配對。`toolComparison` 把本課的階段與 radiff2 的判定與比值，以及每個函式的 Ghidra 相關器並排放在一起。`patch` 是 `grade_case10_wrong.o` 的案例。`radare2Ratios` 重現 radiff2 的數字，`gradeRelocationCase` 顯示排列與兩個位移量，而 `symbolic` 持有路徑、類別與爆炸計數。

請依此順序閱讀 `code/main.py`：

1. `ElfObject` 讀取區段、符號與 `.rela.*` 紀錄。`parse_listing()` 讀取 `llvm-objdump` 文字。
2. `jump_table_targets()` 與 `build_cfg()` 給出可到達的區塊與具型別的邊。`FunctionRecord.signatures()` 建出五種簽章。
3. `load_image()` 排列被配置的區段，並套用 `S + A - P` 與 `S + A`。
4. `match_functions()` 是分階段的比對器。`levenshtein()` 與 `radare2_ratio()` 給出相似度。
5. `parse_radiff2()` 與 `parse_ghidra_vt()` 讀取所擷取的工具輸出以供比較。
6. `evaluate()`、`substitute()` 與 `BoundedSearch` 是運算式語意與求解器。
7. `execute()` 列舉路徑。`solve_returns()` 為每個葉節點求出輸入集合。
8. `grade_ir_o0()`、`grade_ir_o1()`、`percent_change_ir_o1()` 與 `total_ir_o1()` 是附帶其指令位址的手動提升。

另外還有三個指令。`python3 code/main.py diff A.o B.o` 比對兩個旁邊放著清單的目的檔，`--image` 則比對它們的載入映像。`python3 code/main.py paths grade O0` 印出單一函式的路徑與類別，`paths total 8` 則印出帶界限的迴圈。`python3 code/main.py figure-data` 印出互動圖嵌入的資料。

## 實際使用 (Use It)

請依此順序，對你自己程式的兩次建置使用差異比對工具與執行器。

1. 為每個目的檔擷取一份清單，並讓 `function_record()` 對照位元組驗證它。在做其他事之前先修正清單。
2. 先比對目的檔。閱讀每一次比對的階段，並在第二種簽章一致之前，把流程圖形狀的比對視為建議。
3. 比對載入映像。從 exact bytes 降到 masked bytes 的函式，是相對於它的資料移動了，而位移量算術會指出是哪個區段移動了。
4. 對每一次比對，比較被參照的表格。相同的程式碼加上不同的常數，就是被修改過的函式。
5. 執行 names 階段並比較。當建置帶有符號時，執行 BinDiff 順序，並只用結構性結果來為各階段評分。
6. 當工具回報一個比值時，先弄清楚是哪些位元組進入了它，再拿它與你自己的比值比較。
7. 對你想要求解其輸入的函式，寫出每道敘述一個位址的 IR，對每個符號化迴圈計數器設界限後執行 `execute()`，並為每個葉節點求解。報告結果時要連同界限一起報告。
8. 當 angr 或 z3 可用時，用它們執行同樣的查詢，並比較類別的界限。一個樣本只是一個模型。

## 互動實驗 (Interactive Lab)

### 實驗 A：手動建出五種簽章

在 `code` 目錄下開啟 Python：

```bash
cd phases/13-tools-and-protocols/44-binary-diffing-and-symbolic-execution/code
python3 -q
```

執行：

```python
from main import *
grade = {r.name: r for r in fixture_records("ledger-x86_64-O1.o")}["grade"]
grade.raw.hex(" ")
grade.masked_bytes() == grade.raw
grade.mnemonics
grade.cfg_shape()
grade.signatures()
```

在印出 `grade.raw` 之前，先寫下函式偏移量 0x26 處的四個位元組。然後對 `ledger-x86_64-O0.o` 的 `grade` 執行同樣的幾行，並數一數 `grade.edges` 中的 `table` 邊。

### 實驗 B：觀察重定位如何破壞位元組雜湊

```python
case = grade_relocation_case()
case["layout"]
case["displacements"], case["rodataShift"]
case["objectBytesEqual"], case["imageBytesEqual"], case["maskedImageBytesEqual"]
```

在讀 `displacements` 之前，先在紙上計算 `0x100110 + 0x14 - 0x10003a`。然後把 `load_image()` 中的基底改成 0x400000，確認位移量沒有改變，因為兩個運算元一起移動了。

### 實驗 C：錯誤的比對，以及避免它的順序

```python
report = diff_fixtures("ledger-x86_64-O1.o", "ledger-x86_64-O2.o")
[(m.left.name, m.right.name, m.stage) for m in report.matches]
report = diff_fixtures("ledger-x86_64-O0.o", "ledger-x86_64-O1.o")
[(m.left.name, m.right.name, m.stage, m.names_agree) for m in report.matches]
report = diff_fixtures("ledger-x86_64-O0.o", "ledger-x86_64-O1.o", stages=BINDIFF_ORDER)
[(m.left.name, m.right.name, m.stage) for m in report.matches]
```

對 `-O0` 印出 `records(...)["percent_change"].cfg_shape()`，對 `-O1` 印出 `grade_label` 的，並閱讀兩個度數序列。然後寫出第六個階段，對排序後的助記符取雜湊，讓順序不再重要，看看它是否能把兩者區分開。

### 實驗 D：路徑與界限

```python
report = grade_paths("O0")
[(p["trace"], p["condition"], p["returns"]) for p in report["paths"]]
report["returns"]["4"]
total_paths(2), total_paths(4), total_paths(8)
total_paths(64, step_limit=24)["truncated"]
```

把 `SCORE_DOMAIN` 改成 `range(0, 110)`，再次執行 `grade_paths("O0")`。0 的類別縮小為 60 個輸入，兩個交叉檢查仍然成立。然後把值域設成 `range(90, 100)`，閱讀哪些葉節點完全沒有模型。

## 實作練習 (Practice Lab)

在測試資料上執行 BinDiff，並與你已有的三個結果比較。

需求：

- 在 Ghidra 12.1.4 中安裝 BinExport 擴充套件，並匯出三個 ledger 程式。記錄 BinExport 的釋出標籤與 Ghidra 版本。
- 安裝 BinDiff 8 命令列，並以僅輸出相似度的選項產生 `-O0` 對 `-O1` 與 `-O1` 對 `-O2` 的結果。
- 用 `build_fixtures.sh` 中的確切指令，把每個函式的相似度與信心度擷取到 `code/fixtures/`。
- 為 `tool_comparison()` 擴充一個 `bindiff` 欄，並新增一個測試，將 BinDiff 在 `-O1` 對 `-O2` 上的已比對集合與本課的 masked-bytes 集合比較。
- 寫一段文字，指出 BinDiff 對 `grade` 使用了哪個演算法，以及它是否比較了載入映像。

成功的標準是示範為每個函式印出四個工具欄，且新測試陳述一個相等關係或一項已解釋的差異。

## 課程成品 (Lesson Artifact)

本課包含 `outputs/skill-binary-diffing-and-symbolic-execution.md`。用它把兩次建置與一個函式，轉成一份分階段的差異比對，其中每次比對都有明示的品質、每個未比對或已改變的函式都有說明，以及一份路徑報告，內含每個葉節點求得的輸入集合與產生它的界限。

## 驗證 (Verify It)

執行示範與測試套件：

```bash
cd phases/13-tools-and-protocols/44-binary-diffing-and-symbolic-execution
python3 code/main.py
python3 -m unittest discover -s code/tests -v
```

驗證應證明：

- 符號表給出四個函式，大小在 `-O0` 為 46、118、66、39，在 `-O1` 為 16、46、31、63，在 `-O2` 為 16、46、140、63
- `-O1` 目的檔的 `.rela.text` 紀錄，其描述方式與 `llvm-objdump -r` 的輸出一致
- 清單中的每道指令都落在其符號範圍內，且其位元組等於目的檔，改動一個位元組就會拋出例外
- `-O1` 的 `total` 有五個區塊、五條邊與一個自迴圈 (self loop)，且其形狀字串是確定性的
- `-O0` 的 `grade` 從 `.rela.rodata` 復原出五個跳轉表目標，有八個區塊與十一條邊
- 沒有該表時，`-O0` 的 `grade` 在四個區塊中到達 82 個位元組
- `-O0` 的 `percent_change` 與 `-O1` 的 `grade_label` 有相同的四區塊形狀
- exact 與 masked 雜湊在可重定位目的檔的每個函式上都一致
- `-O1` 與 `-O2` 有三個函式共用位元組，`-O0` 與 `-O1` 一個都沒有
- `grade` 的參照表讀作 `1, 2, 3, 4, 4`，而候選的讀作 `1, 2, 3, 4, 3`
- 載入器把 `.text`、`.rodata` 與 `.rodata.str1.1` 放在 Ghidra 在兩個程式中放置的位置
- `R_X86_64_PC32` 以 `S + A - P` 寫入並給出位移量 122，而 `.data` 中的 `R_X86_64_64` 項目取得字串位址
- `grade` 的位元組在目的檔中一致，在映像中因 0x70 的位移而不同，遮蔽後恢復比對
- `-O1` 對 `-O2` 以 exact bytes 比對三個函式、以名稱比對 `total`；映像視角以 masked bytes 比對 `grade`，相似度為 1 - 1/46
- 本課的順序在 `-O0` 對 `-O1` 上產生一次錯誤的流程圖形狀比對，而 BinDiff 順序以名稱比對全部四個
- 修補候選在位元組上比對成功，在被參照的資料上失敗
- Levenshtein 比值重現 0.130435、0.106061、0.269841、0.171429，以及 82 個位元組上的 0.426829，還有 radiff2 在套用重定位時印出的重定位後位元組比值
- radiff2 的 MATCH 集合等於 exact-bytes 階段，而 Ghidra 的位元組與指令相關器等於映像視角的 exact 與 masked 集合
- Ghidra 在 `-O0` 對 `-O1` 上只找到名稱，每筆比對的相似度為 1.0、信心度為 10.0
- 手動提升的 IR 恰好涵蓋每道列出的指令一次，且求值器遵循 C 語意
- `grade` 在 `-O0` 有五條路徑，帶有 switch 條件，在 `-O1` 有兩條路徑，類別相同
- 有界的類別 60 到 69、70 到 79、80 到 89、90 到 109，在三個層級上都等於 angr 的，也等於 z3 的，其中 deadended 狀態為 5 與 2 個，到達區塊的範圍為 90 到 109
- 兩個交叉檢查在界限內、在 angr 中、在 z3 中都沒有模型，且 z3 證明清單等於原始碼的 switch
- 有界搜尋、angr 與 z3 給出的每個 `percent_change` 模型，其求值都等於所要求的值
- `total` 給出 3、5、9 條路徑與三角形數和，與 angr 相同，而步數上限回傳 22 並截斷 2
- 圖的資料同時帶有兩種視角、重定位的數字，以及五個葉節點與其輸入集合，且網站模組嵌入相同的資料

## Production 的失敗模式 (Production Failure Modes)

| 失敗 | 較弱的工具回報什麼 | 分析必須證明什麼 |
|---|---|---|
| 對載入映像做位元組雜湊 | 兩次建置之間程式碼相同卻回報「grade modified」 | 欄位偏移量、兩個位移量與區段位移；masked 雜湊比對成功 |
| 對可重定位目的檔做遮蔽雜湊 | 「masking made no difference」 | 欄位本來就是零；由重定位紀錄與被參照的常數承載改變 |
| 只靠流程圖形狀 | 一對毫不相干的函式被自信地配對 | 第二種簽章一致，否則名稱相矛盾，該比對就被丟棄 |
| 在被剝除符號的建置上先比名稱再比結構 | 什麼都比對不到 | 帶有品質等級的結構性階段，以及未比對清單 |
| 跨工具比較比值 | 「radiff2 says 0.43 and the lesson says 0.30」 | 雙方各自對哪些位元組取了指紋；沒有重定位時為 82 個位元組，有重定位時為 118 個 |
| 相同的程式碼，不同的常數 | 「exact match」 | 透過重定位與邊界檢查讀出的表格 |
| 把一個模型當成輸入回報 | 「the score is 90」 | 類別 90 到 109 連同其數量，以及找到它的界限 |
| 把有界搜尋當成 unsat 回報 | 「no input gives -133」 | 僅在界限內 unsat；z3 在界限外找到 1073741823 |
| 沒有界限的符號化迴圈 | 逾時 | 界限或上限，以及已回傳、已截斷與從未啟動的路徑 |
| 把封閉的工具描述成已執行 | 沒有擷取檔的數字 | 帶有指令與版本的擷取檔，或一句說明僅依文件的句子 |

## 營運規則 (Operational Rule)

一次只依一種簽章比對，只接受兩側皆唯一的值，並讓階段與其品質隨比對保留。同時比對目的檔與載入映像，並說明在兩者之間改變階段的每個函式。在每個分支都以求解器列舉路徑，為每個符號化迴圈設界限，並連同其界限一起報告輸入集合。沒有界限的單一模型只是一個樣本。

## 延伸閱讀 (Further Reading)

- [BinDiff concepts: matching strategy, algorithms, and quality ratings](https://github.com/google/bindiff/blob/main/docs/concepts.md)
- [BinDiff repository and the v8 open-source release](https://github.com/google/bindiff/releases/tag/v8)
- [BinExport README: IDA Pro, Binary Ninja, and Ghidra exporters](https://github.com/google/binexport)
- [Flake, "Structural Comparison of Executable Objects" (DIMVA 2004) and Dullien, Rolles, "Graph-based comparison of Executable Objects" (SSTIC 2005), shipped in the BinDiff repository](https://github.com/google/bindiff/tree/main/docs/papers)
- [Diaphora README and heuristics source](https://github.com/joxeankoret/diaphora)
- [Ghidra Version Tracking correlators help page](https://github.com/NationalSecurityAgency/ghidra/blob/master/Ghidra/Features/VersionTracking/src/main/help/help/topics/VersionTrackingPlugin/VT_Correlators.html)
- [Ghidra Auto Version Tracking help page](https://www.ghidradocs.com/11.0_PUBLIC/help/VersionTracking/help/topics/VersionTrackingPlugin/VT_AutoVT.html)
- [Ghidra 12.1.4 release](https://github.com/NationalSecurityAgency/ghidra/releases)
- [Binary Ninja 6.0 "Krypton" release notes](https://binary.ninja/2026/09/03/binary-ninja-6.0-krypton.html) and [Binary Similarity user guide](https://docs.binary.ninja/guide/similarity.html)
- [radare2 book: radiff2](https://book.rada.re/tools/radiff2/intro.html) and [radare2 6.2.4 source](https://github.com/radareorg/radare2/releases/tag/6.2.4)
- [King, "Symbolic execution and program testing" (CACM 1976)](https://dl.acm.org/doi/10.1145/360248.360252)
- [Baldoni et al., "A Survey of Symbolic Execution Techniques" (ACM Computing Surveys 2018)](https://dl.acm.org/doi/10.1145/3182657)
- [Cadar, Sen, "Symbolic execution for software testing: three decades later" (CACM 2013)](https://dl.acm.org/doi/10.1145/2408776.2408795)
- [Cadar, Dunbar, Engler, "KLEE: Unassisted and Automatic Generation of High-Coverage Tests for Complex Systems Programs" (OSDI 2008)](https://www.usenix.org/legacy/events/osdi08/tech/full_papers/cadar/cadar.pdf) and [KLEE 3.2 release](https://github.com/klee/klee/releases/tag/v3.2)
- [Shoshitaishvili et al., "SoK: (State of) The Art of War: Offensive Techniques in Binary Analysis" (IEEE S&P 2016)](https://oaklandsok.github.io/papers/shoshitaishvili2016.pdf)
- [angr documentation: loading, simulation managers, and the simulation pipeline](https://docs.angr.io/en/latest/core-concepts/toplevel.html) and [angr 10.0.3 on PyPI](https://pypi.org/project/angr/)
- [Z3 5.1.0 release](https://github.com/Z3Prover/z3/releases/tag/z3-5.1.0)
- [Triton dynamic binary analysis library](https://github.com/JonathanSalwan/Triton)
- [System V Application Binary Interface, AMD64 Architecture Processor Supplement, Table 4.9: Relocation Types](https://gitlab.com/x86-psABIs/x86-64-ABI)
