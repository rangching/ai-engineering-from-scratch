# ASCII Art 與視覺 Jailbreak (ASCII Art and Visual Jailbreaks)

> Jiang、Xu、Niu、Xiang、Ramasubramanian、Li、Poovendran，「ArtPrompt: ASCII Art-based Jailbreak Attacks against Aligned LLMs」(ACL 2024, arXiv:2402.11753)。把有害請求中與安全相關的 token 遮蔽，用同樣字母的 ASCII art 呈現取代，再送出偽裝過的 prompt。GPT-3.5、GPT-4、Gemini、Claude、Llama-2 都無法穩健地辨識 ASCII art token。這個攻擊能繞過 PPL（perplexity 過濾）、Paraphrase 防禦與 Retokenization。相關：ViTC benchmark 測量對非語意視覺 prompt 的辨識；StructuralSleight 則推廣到 Uncommon Text-Encoded Structures（樹、圖、巢狀 JSON），形成一整個編碼攻擊家族。

**Type:** Build
**Languages:** Python (stdlib, ArtPrompt token-masking harness)
**Prerequisites:** Phase 18 · 12 (PAIR), Phase 18 · 13 (MSJ)
**Time:** ~60 分鐘

## 學習目標 (Learning Objectives)

- 描述 ArtPrompt 攻擊：辨識關鍵字的步驟、ASCII art 替換、最終偽裝的 prompt。
- 解釋為什麼標準防禦（PPL、Paraphrase、Retokenization）對 ArtPrompt 失效。
- 定義 ViTC，並描述它測量什麼。
- 描述 StructuralSleight 如何推廣到任意的 Uncommon Text-Encoded Structures。

## 問題 (The Problem)

透過改寫與角色扮演（第 12 課）以及長 context（第 13 課）的攻擊，都是在文字層級的模式上運作。ArtPrompt 則在辨識層級運作：模型並沒有解析被禁止的 token。它解析的是用字元渲染出來的影像。安全過濾器看到的是無害的標點符號。模型看到的是一個單字。

## 核心概念 (The Concept)

### ArtPrompt 的兩個步驟

步驟 1。單字辨識 (Word Identification)。給定有害請求，攻擊者用 LLM 找出與安全相關的單字（例如「how to make a bomb」中的「bomb」）。

步驟 2。偽裝 prompt 的產生 (Cloaked Prompt Generation)。把每個被辨識出的單字，換成它的 ASCII art 呈現（由 7x5 或 7x7 的字元區塊組成字母形狀）。模型收到的是一格格標點與空白，夠強的模型能把它辨識為那個單字；安全過濾器看到的只有格子。

結果：GPT-4、Gemini、Claude、Llama-2、GPT-3.5 全部失敗。在他們的 benchmark 子集上，attack success rate 超過 75%。

### 為什麼標準防禦失效

- **PPL（perplexity 過濾）。** ASCII art 的 perplexity 很高，但所有新穎輸入也都一樣。能擋下 ArtPrompt 的門檻選擇，也會擋下合法的結構化輸入。
- **Paraphrase。** 改寫 prompt 會破壞 ASCII art。實務上，改寫用的 LLM 常常會保留或重建這個圖案。
- **Retokenization。** 用不同方式切分 token，並不會改變模型的視覺正在辨識字母形狀這件事。

根本問題是安全過濾器作用在 token 或語意層級；ArtPrompt 則在視覺辨識層級運作。

### ViTC benchmark

對非語意視覺 prompt 的辨識。測量模型閱讀 ASCII art、wingdings 與其他非文字語意視覺內容的能力。ArtPrompt 的效果與 ViTC 準確率相關：模型越擅長閱讀視覺文字，ArtPrompt 對它就越有效。這是能力與安全之間的取捨。

### StructuralSleight

推廣 ArtPrompt：Uncommon Text-Encoded Structures (UTES)。樹、圖、巢狀 JSON、CSV-in-JSON、diff 風格的程式碼區塊。如果某種結構在訓練安全資料中罕見，但模型可以解析，它就能藏匿有害內容。

對防禦的含意：安全必須泛化到模型能解析的所有結構化表示。這個集合很大，而且還在成長。

### 影像模態的類比

視覺 LLM（GPT-5.2、Gemini 3 Pro、Claude Opus 4.5、Grok 4.1）擴大了攻擊面。使用真實影像的 ArtPrompt 式攻擊，比 ASCII art 類比更強，因為影像編碼器產生更豐富的訊號。

### 這在 Phase 18 中的位置

第 12 到 14 課描述三個正交的攻擊向量：反覆精煉（PAIR）、context 長度（MSJ）與編碼（ArtPrompt/StructuralSleight）。第 15 課從以模型為中心的攻擊，轉向系統邊界的攻擊（indirect prompt injection）。第 16 課描述防禦工具的回應。

```figure
al-ascii-cloak
```

## 實際使用 (Use It)

`code/main.py` 建立一個玩具 ArtPrompt。你可以用 ASCII art 字形偽裝有害查詢中的特定單字，驗證偽裝後的字串能通過關鍵字過濾，並（可選）用簡單的辨識器把偽裝後的字串解碼回來。

## 交付 (Ship It)

這堂課會產出 `outputs/skill-encoding-audit.md`。給它一份 jailbreak 防禦報告，它會列舉涵蓋的編碼攻擊家族（ASCII art、base64、leet-speak、UTF-8 homoglyph、UTES），以及攔下每一種的防禦層。

## 練習 (Exercises)

1. 執行 `code/main.py`。驗證偽裝後的字串能通過簡單的關鍵字過濾。回報所需的字元層級改動。

2. 實作第二種編碼：對同一個目標單字做 base64。比較它與 ArtPrompt 繞過過濾的比率，以及還原的難度。

3. 閱讀 Jiang et al. 2024 第 4.3 節（五個模型的結果）。提出一個理由，說明為什麼在同一個 benchmark 上，Claude 對 ArtPrompt 的抵抗力比 Gemini 更高。

4. 設計一個生成前的防禦，偵測 prompt 中形似 ASCII art 的區域。測量它在合法程式碼、表格與數學符號上的誤報率。

5. StructuralSleight 列出 10 種編碼結構。勾勒一個能處理全部 10 種的通用防禦，並估計每個受防禦 prompt 的運算成本。

## 關鍵術語 (Key Terms)

| 術語 | 一般說法 | 實際意義 |
|------|-----------------|------------------------|
| ArtPrompt | 「the ASCII-art attack」 | 用 ASCII art 呈現遮蔽安全字詞的兩步驟 jailbreak |
| Cloaking | 「hide the word」 | 把被禁止的 token 換成模型讀得懂、過濾器卻讀不懂的視覺表示 |
| UTES | 「uncommon structure」 | Uncommon Text-Encoded Structure，用來夾帶內容的樹、圖、巢狀 JSON 等 |
| ViTC | 「visual-text capability」 | 衡量模型閱讀非語意視覺編碼能力的 benchmark |
| Perplexity filter | 「PPL defense」 | 拒絕高 perplexity 的 prompt；會失敗，因為合法的結構化輸入分數也很高 |
| Retokenization | 「tokenizer shift defense」 | 用不同 tokenizer 預處理 prompt；會失敗，因為辨識是視覺性的 |
| Homoglyph | 「lookalike characters」 | 外觀與拉丁字母相同的 Unicode 字元；能繞過子字串檢查 |

## 延伸閱讀 (Further Reading)

- [Jiang et al. — ArtPrompt (ACL 2024, arXiv:2402.11753)](https://arxiv.org/abs/2402.11753)：ASCII art jailbreak 論文
- [Li et al. — StructuralSleight (arXiv:2406.08754)](https://arxiv.org/abs/2406.08754)：UTES 的推廣
- [Chao et al. — PAIR (Lesson 12, arXiv:2310.08419)](https://arxiv.org/abs/2310.08419)：互補的反覆攻擊
- [Anil et al. — Many-shot Jailbreaking (Lesson 13)](https://www.anthropic.com/research/many-shot-jailbreaking)：互補的長度攻擊
