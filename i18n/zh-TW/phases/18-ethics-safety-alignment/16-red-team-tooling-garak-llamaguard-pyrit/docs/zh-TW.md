# Red-Team 工具：Garak、Llama Guard、PyRIT (Red-Team Tooling — Garak, Llama Guard, PyRIT)

> 三個量產工具構成 2026 年的 red-team 堆疊。Llama Guard (Meta)：以 14 個 MLCommons 危害類別 fine-tune 的 Llama-3.1-8B 分類器；2025 年的 Llama Guard 4 是從 Llama 4 Scout 剪枝而來、原生多模態的 12B 分類器。Garak (NVIDIA)：開源 LLM 弱點掃描器，具備靜態、動態與 adaptive 的 probe，涵蓋 hallucination、資料外洩、prompt injection、toxicity 與 jailbreak。PyRIT (Microsoft)：以 Crescendo、TAP 和自訂 converter 鏈進行多輪 red-team 活動，做深度利用。Llama Guard 3 記載於 Meta 的「Llama 3 Herd of Models」(arXiv:2407.21783)；Llama Guard 3-1B-INT4 在 arXiv:2411.17713；Garak 的 probe 架構在 github.com/NVIDIA/garak。這些工具是 2026 年 red-team 研究（第 12 到 15 課）與部署（第 17 課以後）之間的量產介面。

**Type:** Build
**Languages:** Python (stdlib, tool-architecture simulator and Llama Guard-style classifier mock)
**Prerequisites:** Phase 18 · 12-15 (jailbreaks and IPI)
**Time:** ~75 分鐘

## 學習目標 (Learning Objectives)

- 描述 Llama Guard 3/4 在安全堆疊中的位置：輸入分類器、輸出分類器，或兩者皆是。
- 說出 14 個 MLCommons 危害類別，並指出一個不明顯的類別（Code Interpreter Abuse）。
- 描述 Garak 的 probe 架構：probe、detector、harness。
- 描述 PyRIT 的多輪活動結構，以及它如何與 Garak probe 組合。

## 問題 (The Problem)

第 12 到 15 課呈現了攻擊面。量產部署需要可重複、可擴展的評估。2026 年有三個工具居於主導：Llama Guard（防禦分類器）、Garak（掃描器）、PyRIT（活動協調器）。每一個針對 red-team 生命週期中的不同層。

## 核心概念 (The Concept)

### Llama Guard (Meta)

Llama Guard 3 是一個為輸入 / 輸出分類 fine-tune 的 Llama-3.1-8B 模型，涵蓋 MLCommons AILuminate 的 14 個類別：
- 暴力犯罪、非暴力犯罪、性相關、CSAM、誹謗
- 專業建議、隱私、IP、無差別殺傷武器、仇恨
- 自殺 / 自傷、性內容、選舉、code-interpreter 濫用

支援 8 種語言。用法：放在 LLM 之前（輸入審核）、LLM 之後（輸出審核），或兩者皆放。這兩種用法會產生不同的訓練分佈，Llama Guard 3 以單一模型同時處理兩者。

Llama Guard 3-1B-INT4 (arXiv:2411.17713, 440MB, 在行動 CPU 上約 30 tokens/s) 是量化的邊緣版本。

Llama Guard 4（2025 年 4 月）為 12B，原生多模態，從 Llama 4 Scout 剪枝而來。它以一個能同時接收文字與影像的分類器，取代前代的 8B 文字版與 11B 視覺版。

### Garak (NVIDIA)

開源弱點掃描器。架構：
- **Probe。** 針對 hallucination、資料外洩、prompt injection、toxicity、jailbreak 的攻擊產生器。分為靜態（固定 prompt）、動態（產生的 prompt）、adaptive（回應目標的輸出）。
- **Detector。** 依預期的失敗模式為輸出打分，如有毒、外洩、被 jailbreak。
- **Harness。** 管理 probe 與 detector 的配對、執行活動、產生報告。

TrustyAI 把 Garak 與 Llama-Stack shield（Prompt-Guard-86M 輸入分類器、Llama-Guard-3-8B 輸出分類器）整合，做端對端的有防護目標評估。以分級為基礎的評分 (TBSA) 取代二元的通過 / 失敗：同一個 probe 上，模型可能在嚴重度第 3 級通過、在第 5 級失敗。

### PyRIT (Microsoft)

Python Risk Identification Toolkit。多輪 red-team 活動。圍繞下列元件建構：
- **Converter。** 轉換種子 prompt，如改寫、編碼、翻譯、角色扮演。
- **Orchestrator。** 執行活動：Crescendo（逐步升級）、TAP（分支）、RedTeaming（自訂迴圈）。
- **評分 (Scoring)。** 以 LLM 當裁判或以分類器當裁判。

PyRIT 是 Garak 較重量級的表親。Garak 執行數千個單輪 probe；PyRIT 執行為突破特定失敗模式而設計的深度多輪活動。

### 堆疊

把 Llama Guard 放在模型兩側。每晚執行 Garak 做回歸測試。發布前執行 PyRIT 活動。這是 2026 年多數量產部署的預設配置。

### 評估的陷阱

- **裁判身分。** 三個工具都可以使用 LLM 裁判；裁判的校準會決定回報的 ASR（第 12 課）。請在工具之外一併註明裁判。
- **Probe 老化。** 隨著模型針對 Garak probe 打補丁，這些 probe 會老化。Adaptive probe（PAIR 形式）比靜態 probe 老化得慢。
- **Llama Guard 對無害內容的 FPR。** 早期的 Llama Guard 版本對政治與 LGBTQ+ 內容過度標記；Llama Guard 3/4 的校準已改善，但並非針對每個部署校準。

### 這在 Phase 18 中的位置

第 12 到 15 課是攻擊家族。第 16 課是量產工具。第 17 課 (WMDP) 是針對雙用途能力的評估。第 18 課是把這些工具包進政策結構的 frontier 安全框架。

```figure
al-guard-stack
```

## 實際使用 (Use It)

`code/main.py` 建立一個玩具 Llama Guard 風格分類器（針對 14 個類別的關鍵字加語意特徵）、一個玩具 Garak harness（probe 與 detector 迴圈），以及一個 PyRIT 風格的多輪 converter 鏈。你可以對一個模擬目標執行這三個工具，並觀察它們不同的涵蓋特徵。

## 交付 (Ship It)

這堂課會產出 `outputs/skill-red-team-stack.md`。給它一份部署描述，它會指出三個工具中哪些適用、每個工具該設定什麼，以及應該執行什麼樣的回歸頻率。

## 練習 (Exercises)

1. 執行 `code/main.py`。比較 Llama-Guard 風格分類器在單輪與多輪攻擊上的偵測率。

2. 實作一個新的 Garak probe：base64 編碼的有害請求。測量 Llama-Guard 風格分類器對它的偵測。

3. 為 PyRIT 風格的 converter 鏈加入「翻譯成法文，再改寫」的 converter。重新測量攻擊成功率。

4. 閱讀 Llama Guard 3 的危害類別清單。找出兩個類別，其訓練資料實際上會對合法的開發者內容產生高誤報率。

5. 比較 Garak 與 PyRIT 的設計原則。為各自是正確工具的部署情境提出論證。

## 關鍵術語 (Key Terms)

| 術語 | 一般說法 | 實際意義 |
|------|-----------------|------------------------|
| Llama Guard | 「the classifier」 | 以 14 個危害類別 fine-tune 的 Llama-3.1-8B/4-12B 安全分類器 |
| Garak | 「the scanner」 | NVIDIA 開源的弱點掃描器；probe、detector、harness |
| PyRIT | 「the campaign tool」 | Microsoft 的多輪 red-team 協調器；converter、orchestrator、scoring |
| Prompt-Guard | 「the small classifier」 | Meta 的 86M prompt-injection 分類器，與 Llama Guard 搭配 |
| TBSA | 「tier-based scoring」 | Garak 以分級為基礎的通過 / 失敗，取代二元結果 |
| Converter chain | 「paraphrase + encode + ...」 | PyRIT 用來建構多步驟攻擊的組合基本單元 |
| MLCommons hazard categories | 「the 14 taxonomies」 | Llama Guard 所針對的業界標準分類法 |

## 延伸閱讀 (Further Reading)

- [Meta — Llama Guard 3 (in Llama 3 Herd paper, arXiv:2407.21783)](https://arxiv.org/abs/2407.21783)：8B 分類器
- [Meta — Llama Guard 3-1B-INT4 (arXiv:2411.17713)](https://arxiv.org/abs/2411.17713)：量化的行動端分類器
- [NVIDIA Garak — GitHub](https://github.com/NVIDIA/garak)：掃描器的 repo 與文件
- [Microsoft PyRIT — GitHub](https://github.com/Azure/PyRIT)：活動工具包
