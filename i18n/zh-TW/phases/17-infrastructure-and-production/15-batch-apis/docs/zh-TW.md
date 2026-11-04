# Batch APIs：50% 折扣成為業界標準 (the 50% Discount as Industry Standard)

> 每一家主要供應商都提供非同步 batch API，折扣 50%，約 24 小時內交付。OpenAI、Anthropic、Google，以及多數 inference 平台（Fireworks batch tier、Together batch）都實作同樣的模式。把 batch 與 prompt caching 疊加，夜間 pipeline 可降到同步未快取成本的約 ~10%。規則極其簡單：只要能容忍數小時的 latency，就該放進 batch。內容生成 pipeline、文件分類、資料擷取、報告生成、大量標註、目錄標籤，凡是能容忍 24 小時 latency 的任何工作，在搬到 batch 之前都是把錢留在桌上。2026 年的 production 模式，是把每個新的 LLM 工作負載分流到三條車道：互動式（同步加 caching）、半互動式（非同步佇列加 fallback）、batch（夜間，疊加快取輸入）。那些假裝是互動式、其實能容忍數分鐘 latency 的工作負載，浪費最多。

**Type:** Learn
**Languages:** Python (stdlib, toy batch-vs-sync cost simulator)
**Prerequisites:** Phase 17 · 14 (Prompt & Semantic Caching)
**Time:** ~45 分鐘

## 學習目標 (Learning Objectives)

- 說出三家供應商的 batch API（OpenAI、Anthropic、Google），以及共通的 50% 折扣 + 24 小時交付保證。
- 計算在夜間分類工作負載上疊加 batch + 快取輸入的成本，並與同步未快取的基準比較。
- 把工作負載分流到互動式 / 半互動式 / batch，並說明該車道的理由。
- 說出兩個陷阱：部分互動性（使用者期待比 24 小時更快）與輸出 schema 漂移（batch 檔案格式因供應商而異）。

## 問題 (The Problem)

你的團隊交付了一條夜間報告生成 pipeline。50,000 份文件，逐份摘要、將摘要分群、草擬一份主管簡報。同步執行要 4 小時，每晚 $2,000。你聽說了 batch API。

Batch 讓你省 50%。你也對 system prompt（在全部 50k 次呼叫間共用）啟用 prompt caching。疊加之後，帳單降到每晚 $180，約為基準的 ~9%。同一條 pipeline，三個設定改動。

Batch 是 LLM 成本工具箱裡最便宜、卻沒人去拉的槓桿。原因多半是組織性的：團隊想的是「即時」，而 SLA 其實是「早上之前」。這堂課就是要讓你不把 90% 的帳單留在桌上。

## 核心概念 (The Concept)

### 三個 batch API

**OpenAI Batch API**：上傳包含 request 清單的 JSONL 檔案。承諾 24 小時內交付（實務上通常 ~2-8 小時）。輸入與輸出 tokens 皆享 50% 折扣。`/v1/batches` endpoint。符合快取資格的輸入還會再疊加快取輸入定價。

**Anthropic Message Batches**：上傳 JSONL。24 小時交付。50% 折扣。支援 `cache_control`：cache 寫入是明確的，讀取則在 batch 內自動發生。

**Google Vertex AI Batch Prediction**：輸入為 BigQuery 或 GCS。Gemini 同樣享有 50% 折扣。與 Vertex pipelines 整合。

### 語意：非同步，不是慢

Batch 是「我承諾在 24 小時內回傳」，不是「這會花 24 小時」。典型 P50 為 2-6 小時。供應商會在 GPU 庫存使用率偏低的離峰時段排程你的 batch。

### 與 caching 疊加

50k 份文件摘要，共用同一個 4K-token system prompt：

- 同步未快取：50000 × ($input × 4000 + $output × 200)，按全額費率。
- 同步已快取：system prompt 在第一次寫入後被快取；其餘 49999 次的輸入便宜 10x。
- Batch 已快取：以上全部，再加上讀取與寫入都享有 50% 折扣。

疊加結果：batch + cache = 同步未快取帳單的 ~10%。任何在夜間執行、且有共用 system prompt 的工作負載，都該用這招。

### 工作負載分流

**互動式**：使用者等待回應。TTFT 很重要。同步呼叫加 prompt caching。不能用 batch。

**半互動式**：使用者提交任務，幾分鐘後回來查看。非同步佇列，若 batch 無法使用就 fallback 到同步。例如中等量級的 RAG 索引建立。

**Batch**：使用者預期「早上前」或「下班前」拿到結果。內容 pipeline、大規模分類、離線分析。一律用 batch，一律疊加 caching。

常見錯誤：因為 pipeline 是 production，就把所有東西都歸為互動式。Production 不是 latency 規格，SLA 才是。

### 部分互動性陷阱

有些功能看起來是互動式，其實能容忍 5-10 分鐘。例如：有「重新整理」按鈕的夜間客戶健康報告。使用者點重新整理，等 10 分鐘沒問題。團隊卻把它做成同步。50 個並行重新整理的成本，是改成 batch 並以 email 交付的 10 倍。

該問的問題是：「對這個使用者而言，24 小時意味著什麼？」如果答案是「他們不會注意到」，就用 batch。

### 輸出 schema 陷阱

Batch 檔案格式因供應商而異：

- OpenAI：JSONL，每行一個 request。
- Anthropic：JSONL，每行一則 message；回應格式內嵌。
- Vertex：BigQuery 資料表，或帶 TFRecord 的 GCS 前綴。

要寫出跨供應商的「單一 batch client」，意味著每個供應商都要寫 adapter 程式。標榜多供應商 batch 的 gateway（Portkey、部分方案的 LiteLLM）仍只是薄薄包裝原始格式。

### 該記住的數字

- 各供應商的 batch 折扣：輸入 + 輸出一律 50%。
- 交付 SLA：保證 24 小時，典型 P50 為 2-6 小時。
- 疊加 batch + 快取輸入：同步未快取成本的 ~10%。
- 工作負載分流規則：如果可接受 24 小時 latency，就一律用 batch。

```figure
batch-lane-triage
```

## 實際使用 (Use It)

`code/main.py` 針對 50k 文件的工作負載，計算同步、同步+cache、batch 與 batch+cache 的成本。回報節省的金額與百分比。

## 交付 (Ship It)

這堂課產出 `outputs/skill-batch-triager.md`。給定工作負載特性，分流到互動式/半互動式/batch，並估算節省。

## 練習 (Exercises)

1. 執行 `code/main.py`。對 3K-token system prompt、500-token 輸出的 100k 文件 pipeline，計算完整 stack（batch + cache）相對同步基準的節省。
2. 在你熟悉的真實產品中挑三個功能。把每個分流到互動式/半互動式/batch。
3. 一位使用者抱怨他的報告花了 3 小時。那是 batch 分流錯誤，還是合理的互動式？寫出判斷準則。
4. 你的 batch API 回傳 SLA 是 24h，但 P99 是 20 小時。你如何向使用者溝通？在極端情況下，下游系統的行為是什麼？
5. 計算損益平衡：在共用前綴多長時，batch + cache 會比在自己的預留 GPU 上跑一整晚更便宜？

## 關鍵術語 (Key Terms)

| 術語 | 一般說法 | 實際意義 |
|------|----------------|------------------------|
| Batch API | 「非同步折扣」 | 24h 交付、五折 |
| JSONL | 「batch 格式」 | 每行一個 JSON request；OpenAI/Anthropic 標準 |
| Message Batches | 「Anthropic batch」 | Anthropic 的 batch API 產品名稱 |
| Batch prediction | 「Vertex batch」 | Vertex AI 的 batch API 產品 |
| Turnaround SLA | 「24h 承諾」 | 是保證而非典型值；典型是 2-6h |
| Workload triage | 「互動性決策」 | 互動式 / 半互動式 / batch 的路由決策 |
| Output schema | 「回應格式」 | 各供應商自己的 JSONL 版面；不可攜 |
| Stacked discount | 「batch + cache」 | 兩者皆適用時，約為未快取同步帳單的 ~10% |

## 延伸閱讀 (Further Reading)

- [OpenAI Batch API](https://platform.openai.com/docs/guides/batch)：JSONL 格式與 `/v1/batches` 語意。
- [Anthropic Message Batches](https://docs.anthropic.com/en/docs/build-with-claude/batch-processing)：batch 格式與 `cache_control` 的互動。
- [Vertex AI Batch Prediction](https://cloud.google.com/vertex-ai/generative-ai/docs/multimodal/batch-prediction-gemini)：Gemini batch 語意。
- [Finout — OpenAI vs Anthropic API Pricing 2026](https://www.finout.io/blog/openai-vs-anthropic-api-pricing-comparison)
- [Zen Van Riel — LLM API Cost Comparison 2026](https://zenvanriel.com/ai-engineer-blog/llm-api-cost-comparison-2026/)
