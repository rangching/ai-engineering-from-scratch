# Watermarking：SynthID、Stable Signature、C2PA (Watermarking — SynthID, Stable Signature, C2PA)

> 三項技術構成 2026 年 AI 生成內容來源追溯的骨幹。SynthID (Google DeepMind)：影像 watermarking 於 2023 年 8 月推出，文字與影片於 2024 年 5 月推出（Gemini + Veo），文字部分於 2024 年 10 月透過 Responsible GenAI Toolkit 開源，統一的多媒體偵測器於 2025 年 11 月隨 Gemini 3 Pro 推出。文字 watermarking 以難以察覺的方式調整 next-token 取樣機率；影像 / 影片 watermark 能撐過壓縮、裁切、濾鏡與影格率變更。Stable Signature (Fernandez et al., ICCV 2023, arXiv:2303.15435)：fine-tune latent diffusion 的 decoder，使每個輸出都含有固定訊息；裁切後（保留 10% 內容）的生成影像，在 FPR<1e-6 下偵測率 >90%。後續研究「Stable Signature is Unstable」(arXiv:2405.07145, May 2024)：fine-tuning 能在保留品質的同時移除 watermark。C2PA：經密碼學簽署、可偵測竄改的 metadata 標準（C2PA 2.2 Explainer 2025）。Watermarking 與 C2PA 互補：metadata 可以被剝除，但攜帶更豐富的來源資訊；watermark 能撐過轉碼，但攜帶的資訊較少。

**Type:** Build
**Languages:** Python (stdlib, token-watermark embed + detect)
**Prerequisites:** Phase 10 · 04 (sampling), Phase 01 · 09 (information theory)
**Time:** ~75 分鐘

## 學習目標 (Learning Objectives)

- 描述 token 層級的 watermarking（SynthID-text 風格），以及它可被偵測的機制。
- 描述 Stable Signature，以及 2024 年破解它的移除攻擊。
- 說明 C2PA 的角色，以及它為什麼與 watermarking 互補。
- 描述主要的限制：針對特定模型的訊號、在改寫下的穩健性，以及保留語意的攻擊 (arXiv:2508.20228)。

## 問題 (The Problem)

2023-2024 年，deepfake 與 AI 生成內容大規模進入政治與消費情境。Watermarking 是被提出的技術性來源訊號：在生成時標記，之後再偵測。2025 年的證據：沒有任何 watermark 是無條件穩健的，但與 C2PA metadata 分層搭配後，兩者合起來能提供一個可用的來源追溯方案。

## 核心概念 (The Concept)

### 文字 watermarking（SynthID-text 風格）

Kirchenbauer et al. 2023 的機制，由 Google 量產化：

1. 在每個解碼步驟，對前 K 個 token 做 hash，產生把詞彙表偽隨機分成「green」與「red」兩組的分割。
2. 藉由對 green 的 logit 加上 δ，讓取樣偏向 green 組。
3. 生成內容中的 green token 會比隨機機率所產生的更多。

偵測：對每個前綴重新 hash，計算生成內容中的 green token 數，再算出 z-score。有 watermark 的文字 z-score >0，人類文字約為 ~0。

特性：
- 讀者難以察覺（δ 小到品質損失輕微）。
- 能取得詞彙表分割函式時可偵測。
- 對改寫不穩健：重寫文字就會破壞訊號。

SynthID-text 於 2024 年 10 月透過 Google 的 Responsible GenAI Toolkit 開源。

### Stable Signature（影像）

Fernandez et al. ICCV 2023。Fine-tune latent diffusion 的 decoder，使每張生成影像都含有嵌入在 latent 表示中的固定二進位訊息。偵測是用神經網路 decoder 從 latent 解碼出來。裁切後（保留 10% 內容）的影像，在 FPR<1e-6 下偵測率 >90%。

2024 年 5 月的「Stable Signature is Unstable」(arXiv:2405.07145)：對 decoder 做 fine-tuning 能在保留影像品質的同時移除 watermark。生成後的 adversarial fine-tuning 成本低廉；這個 watermark 的 adversarial 穩健性有限。

### SynthID 統一偵測器（2025 年 11 月）

隨 Gemini 3 Pro 推出：一個多媒體偵測器，用單一 API 讀取文字、影像、音訊與影片中的 SynthID 訊號。統一了 Google 的來源追溯堆疊。

### C2PA

Coalition for Content Provenance and Authenticity。經密碼學簽署、可偵測竄改的 metadata 標準。C2PA 2.2 Explainer (2025)。C2PA manifest 記錄來源主張（誰建立、何時、經過哪些轉換），並以建立者的金鑰簽署。

與 watermarking 互補：
- Metadata 可以被剝除；watermark（不容易）剝除。
- Metadata 內容豐富（完整的來源鏈）；watermark 攜帶的是少量位元。
- C2PA 依賴平台採用；watermark 則是自動嵌入。

Google 在 Search、Ads 與「About this image」中整合了兩者。

### 限制

- **針對特定模型。** SynthID 只對啟用 SynthID 的模型的生成內容加 watermark。沒有 SynthID 的模型所生成的內容不會有 watermark，因此「沒有 SynthID 訊號」不能證明內容是真實的。
- **改寫。** 文字 watermark 撐不過保留語意的改寫。
- **轉換攻擊。** arXiv:2508.20228 (2025) 顯示，保留語意的攻擊能同時破壞文字 watermark 與許多影像 watermark。
- **Fine-tune 移除。** 依「Stable Signature is Unstable」，生成後的 fine-tuning 能移除嵌入的 watermark。

### EU AI Act 第 50 條

針對 AI 生成內容標示的 Transparency Code（第一版草案於 2025 年 12 月，第二版草案於 2026 年 3 月，依[歐盟執委會狀態頁面](https://digital-strategy.ec.europa.eu/en/policies/code-practice-ai-generated-content)預計於 2026 年 6 月定稿）。截至 2026 年 4 月，該 Code 仍為草案，時程可能變動。這是要求技術層的監管層。Deepfake 必須被標示。

### 這在 Phase 18 中的位置

第 22 到 23 課談的是模型輸出什麼（私人資料、來源訊號）。第 27 課涵蓋訓練資料治理。第 24 課是要求這些技術措施的監管框架。

```figure
an-watermark-greenlist
```

## 實際使用 (Use It)

`code/main.py` 建立一個玩具文字 watermark。Token 是整數 0..N-1；加了 watermark 的取樣會偏向由 hash 定義的 green 組。偵測器計算 green token 的 z-score。你可以觀察在 1000 個 token 的生成上偵測到 watermark、看著改寫破壞訊號，並測量人類文字的誤報率。

## 交付 (Ship It)

這堂課會產出 `outputs/skill-provenance-audit.md`。給它一個帶有來源主張的內容部署，它會稽核：watermark 機制（若有）、C2PA 簽署鏈（若有）、各自的 adversarial 穩健性，以及各模態的涵蓋範圍。

## 練習 (Exercises)

1. 執行 `code/main.py`。回報加了 watermark 的 1000 個 token 生成與人類撰寫文字的 z-score。找出在 95% 信賴門檻下的誤報率。

2. 實作一個把 30% 的 token 換成同義詞的改寫攻擊。重新測量 z-score。

3. 閱讀 Kirchenbauer et al. 2023 第 6 節關於穩健性的內容。為什麼文字 watermark 在改寫下失敗，而影像 watermark 卻撐得過裁切？

4. 設計一個使用 SynthID-text 加 C2PA metadata 的部署。描述消費者看到的來源鏈。找出每個元件的一種失敗模式。

5. 2024 年「Stable Signature is Unstable」的結果顯示 fine-tuning 能移除影像 watermark。設計一個限制此攻擊的部署控制，例如要求 fine-tune 後的 checkpoint 必須經過簽署發布。

## 關鍵術語 (Key Terms)

| 術語 | 一般說法 | 實際意義 |
|------|-----------------|------------------------|
| SynthID | 「Google's watermark」 | 跨模態的來源訊號；文字、影像、音訊、影片 |
| Token watermark | 「Kirchenbauer-style」 | 以偏向性取樣產生、可由 green token z-score 偵測的文字 watermark |
| Stable Signature | 「image watermark」 | 以 fine-tune decoder 產生的 watermark；ICCV 2023 |
| C2PA | 「the metadata standard」 | 經密碼學簽署、可偵測竄改的來源 metadata |
| Paraphrase robustness | 「does rewording break it」 | 文字 watermark 的特性；目前有限 |
| Fine-tune removal | 「adversarial unwatermark」 | 透過 decoder fine-tuning 移除影像 watermark 的攻擊 |
| Cross-modal detector | 「unified SynthID」 | 2025 年 11 月跨模態的統一 API |

## 延伸閱讀 (Further Reading)

- [Kirchenbauer et al. — A Watermark for Large Language Models (ICML 2023, arXiv:2301.10226)](https://arxiv.org/abs/2301.10226)：token watermark 機制
- [Fernandez et al. — Stable Signature (ICCV 2023, arXiv:2303.15435)](https://arxiv.org/abs/2303.15435)：影像 watermark 論文
- ["Stable Signature is Unstable" (arXiv:2405.07145)](https://arxiv.org/abs/2405.07145)：移除攻擊
- [Google DeepMind — SynthID](https://deepmind.google/models/synthid/)：跨模態 watermark
- [C2PA 2.2 Explainer (2025)](https://c2pa.org/specifications/specifications/2.2/explainer/Explainer.html)：metadata 標準
