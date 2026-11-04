# 文件與圖表理解 (Document and Diagram Understanding)

> 文件不是照片。PDF、科學論文、發票或手寫表單，都帶有版面、表格、圖表、註腳、標頭與語意結構，是單純的圖片理解無法捕捉的。VLM 出現之前的堆疊是一條管線：Tesseract OCR + LayoutLMv3 + 表格擷取的啟發式規則。VLM 浪潮用 OCR-free 模型取代了它，例如 Donut (2022)、Nougat (2023)、DocLLM (2023)，它們直接輸出結構化標記。到 2026 年，前沿做法就是「把頁面圖片以 2576px 原生解析度餵給 Claude Opus 4.7」，結構化標記輸出免費附贈。這堂課讀文件 AI 的三個時代演進。

**Type:** Build
**Languages:** Python (stdlib, layout-aware document parser skeleton)
**Prerequisites:** Phase 12 · 05 (LLaVA), Phase 5 (NLP)
**Time:** ~180 分鐘

## 學習目標 (Learning Objectives)

- 解釋文件 AI 的三個時代：OCR 管線、OCR-free、VLM-native。
- 描述 LayoutLMv3 的三條輸入串流：文字、版面 (bbox)、圖片 patch，並搭配統一的 masking。
- 比較 Donut（OCR-free，圖片 → 標記）、Nougat（科學論文 → LaTeX）、DocLLM（感知版面的生成式）、PaliGemma 2（VLM-native）。
- 為新任務（發票、科學論文、手寫表單、中文收據）挑選文件模型。

## 問題 (The Problem)

「理解這份 PDF」看似簡單，其實很難。資訊存在於：

- 文字內容（90% 的訊號）。
- 版面（標頭、註腳、側欄、雙欄格式）。
- 表格（列、欄、合併儲存格）。
- 圖片與圖表。
- 手寫註記。
- 字型與排版（標題 vs 內文）。

單純的 OCR 只倒出文字，其餘全部流失。在意發票的系統需要知道「Total: $1,245」來自右下角，而不是註腳。

## 核心概念 (The Concept)

### 時代 1：OCR 管線 (2021 年以前) (Era 1 — OCR pipeline)

經典堆疊：

1. PDF → 每頁一張圖片。
2. Tesseract（或商用 OCR）擷取文字，附每個字的 bounding box。
3. 版面分析器辨識區塊（標頭、表格、段落）。
4. 表格結構辨識器解析表格。
5. 領域規則 + regex 擷取欄位。

對乾淨的印刷文字有效。遇到手寫、歪斜的掃描、複雜表格、非英文文字就壞掉。每種失敗模式都需要自訂的例外處理路徑。

### TrOCR (2021)

TrOCR (Li et al., arXiv:2109.10282) 用在合成 + 真實文字圖片上訓練的 transformer encoder-decoder，取代 Tesseract 經典的 CNN-CTC。在手寫與多語言文字上明顯勝出。仍然是管線（偵測器、TrOCR、再版面），但 OCR 這一步大幅進步。

### 時代 2：OCR-free (2022-2023) (Era 2 — OCR-free)

第一批 OCR-free 模型說：完全跳過偵測，直接把圖片像素對應到結構化輸出。

Donut (Kim et al., arXiv:2111.15664)：
- Encoder-decoder transformer，encoder 是 Swin-B。
- 輸出是表單理解用的 JSON、摘要用的 markdown，或任何任務專屬的 schema。
- 沒有 OCR、沒有版面、沒有偵測。

Nougat (Blecher et al., arXiv:2308.13418)：
- 專門在科學論文上訓練。
- 輸出是 LaTeX / markdown。
- 處理方程式、多欄版面、圖片。
- 每個 arXiv 解析器都會呼叫的模型。

這些是專家，不是通才。Donut 用在科學論文會失敗；Nougat 用在發票會失敗。

### LayoutLMv3 (2022)

另一條路線。LayoutLMv3 (Huang et al., arXiv:2204.08387) 保留 OCR，但加入版面理解：

- 三條輸入串流：OCR 文字 token、每個 token 的 2D bounding box、圖片 patch。
- 跨三種 modality 的 masked 訓練目標（masked 文字、masked patch、masked 版面）。
- 下游：分類、實體擷取、表格 QA。

LayoutLMv3 是以 OCR 為基礎的文件理解的巔峰。在表單和發票上很強。上游需要 OCR。在標準化文件 benchmark 上是 VLM 之前最好的準確率。

### DocLLM (2023)

DocLLM (Wang et al., arXiv:2401.00908，於 2023 年 12 月 31 日提交，所以 ID 帶有 2024 年 1 月的前綴) 是 LayoutLM 的生成式兄弟。以版面 token 為條件，生成自由格式的答案。更適合文件上的 QA；仍然依賴 OCR 輸入。

### 時代 3：VLM-native (2024 年起) (Era 3 — VLM-native)

2024 年的 VLM 已好到足以完全取代這條管線。把整頁圖片以高解析度餵給 VLM，提出問題，得到答案。

- LLaVA-NeXT 336-tile AnyRes 適用於小型文件。
- Qwen2.5-VL 動態解析度原生處理 2048+ 像素。
- Claude Opus 4.7 支援 2576px 的文件。
- PaliGemma 2 (2024 年 12 月) 專門針對文件 + 手寫訓練。

VLM-native 與 OCR 管線之間的差距迅速縮小。到 2026 年，VLM-native 在以下方面勝出：

- 場景文字（手寫 + 印刷、混合文字）。
- 帶合併儲存格的複雜表格。
- 嵌在文字中的數學方程式。
- 帶文字註記的圖片。

OCR 管線仍在以下方面勝出：

- 大規模的純掃描工作負載，每頁延遲很重要。
- 管線可靠性（確定性的失敗 vs VLM 的幻覺）。
- 需要可稽核 OCR 輸出的受管制環境。

### Claude 4.7 / GPT-5 前沿 (The Claude 4.7 / GPT-5 frontier)

在 2576 像素原生輸入下，前沿 VLM 的文件理解已接近人類準確率。2026 年初的 benchmark 數字：

- DocVQA：Claude 4.7 約 95.1，PaliGemma 2 約 88.4，Nougat 約 77.3，管線式 LayoutLMv3 約 83。
- ChartQA：Claude 4.7 約 92.2，GPT-4V 約 78。
- VisualMRC：Claude 4.7 約 94。

封閉模型的差距主要在解析度和基礎 LLM 的規模。7B 的開源模型落後幾分，但正在追上。

### 數學方程式與 LaTeX 輸出 (Math equations and LaTeX output)

科學論文需要方程式的精確 LaTeX 輸出。Nougat 就是為此訓練的。以 LaTeX 為目標訓練的 VLM（Qwen2.5-VL-Math、Nougat 衍生模型）能產生可用的 LaTeX。沒有明確的 LaTeX 訓練，VLM 產生的是可讀但不精確的轉錄。

2026 年的科學論文管線：先對 PDF 串接 Nougat，再對棘手的頁面使用 VLM。

### 手寫 (Handwriting)

仍是最困難的子任務。印刷 + 手寫混合（醫師筆記、填寫過的表單）是 OCR 管線在成本上仍勝過 VLM 的地方。純手寫的 VLM 正在進步 (Claude 4.7、PaliGemma 2)。

### 2026 年的配方 (2026 recipe)

對新的文件 AI 專案：

- 大規模純印刷發票：LayoutLMv3 + 規則，成本效益高。
- 混合文件（科學 + 手寫 + 表單）：VLM-native (PaliGemma 2 或 Qwen2.5-VL)。
- 完整的 arXiv 輸入：數學用 Nougat，圖片用 VLM。
- 法規：OCR 管線 + VLM 驗證器做交叉檢查。

```figure
mm-doc-layout
```

## 實際使用 (Use It)

`code/main.py`：

- 玩具版感知版面的 tokenizer：給定 (text, bbox) 配對，產生 LayoutLMv3 風格的輸入。
- Donut 風格的任務 schema 產生器：表單的 JSON 模板。
- 比較 OCR 管線、Donut、Nougat 與 VLM-native 每頁的 token 預算。

## 交付 (Ship It)

這堂課產出 `outputs/skill-document-ai-stack-picker.md`。給定文件 AI 專案（領域、規模、品質、法規），它會在 OCR 管線、OCR-free 專家與 VLM-native 之間做選擇。

## 練習 (Exercises)

1. 你的專案每天有 1000 萬張發票。哪個堆疊在不損失準確率的情況下，能讓每頁成本最低？

2. 為什麼 LayoutLMv3 在表單 QA 上勝過純 CLIP-VLM，卻在場景文字上表現較差？bbox 串流放棄了什麼？

3. Nougat 生成 LaTeX。提出一個 VLM-native 輸出在 LaTeX 保真度上勝過 Nougat 的測試案例，以及一個 Nougat 勝出的案例。

4. 閱讀 PaliGemma 2 論文（Google, 2024）。相較於 PaliGemma 1，提升文件準確率的關鍵訓練資料新增項目是什麼？

5. 設計一個符合法規的混合方案：OCR 管線為主，VLM 作為次要的交叉檢查。如何解決兩者的分歧？

## 關鍵術語 (Key Terms)

| 術語 (Term) | 一般說法 (What people say) | 實際意義 (What it actually means) |
|------|-----------------|------------------------|
| OCR pipeline | "Tesseract-style" | 分階段的堆疊：偵測 -> OCR -> 版面 -> 規則；確定性、脆弱 |
| OCR-free | "Donut-style" | 跳過明確 OCR 的圖片對輸出 transformer；單一模型 |
| Layout-aware | "LayoutLM" | 輸入包含每個 token 的 bbox 座標；跨 modality 的統一 masking |
| VLM-native | "Frontier VLM" | 把頁面圖片以高解析度直接餵給 Claude/GPT/Qwen VLM；沒有管線 |
| DocVQA | "Doc benchmark" | 文件 VQA 標準；最常被引用的分數 |
| Markup output | "LaTeX / MD" | 結構化輸出格式，而不是自由格式文字；使下游自動化成為可能 |

## 延伸閱讀 (Further Reading)

- [Li et al. — TrOCR (arXiv:2109.10282)](https://arxiv.org/abs/2109.10282)
- [Blecher et al. — Nougat (arXiv:2308.13418)](https://arxiv.org/abs/2308.13418)
- [Huang et al. — LayoutLMv3 (arXiv:2204.08387)](https://arxiv.org/abs/2204.08387)
- [Kim et al. — Donut (arXiv:2111.15664)](https://arxiv.org/abs/2111.15664)
- [Wang et al. — DocLLM (arXiv:2401.00908)](https://arxiv.org/abs/2401.00908)
