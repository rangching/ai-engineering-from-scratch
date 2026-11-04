# 把 Model Routing 當作降低成本的基本元件 (Model Routing as a Cost-Reduction Primitive)

> 動態 broker 會評估每個 request（任務類型、token 長度、embedding 相似度、信心度），把簡單的查詢送到便宜的模型，把複雜的升級到前沿模型。也稱為 model cascading。Production 案例研究顯示，在美國/英國/歐盟的部署中，於同等品質下可降低 20-60% 成本；高流量 SaaS 上 30% 的 routing 效率提升，會變成六位數美元的年度節省。2026 年的背景是：固定能力等級的 LLM inference 價格大約每年下降 10x（a16z LLMflation 估計）；以 GPT-4 等級而言，降幅較慢，從 2022 年底到 2026 年由 $20/M 降到約 ~$0.40/M（約 50x）。大部分降幅來自更好的 serving stack (Phase 17 · 04-09)，而不是硬體。Routing 就是把這個價格降幅轉成利潤、又不讓產品退步的方法。失敗模式是便宜模型漂移：路由把 40% 推給較弱的模型，推理任務的品質掉 3-5%，一整季都沒人發現。請用線上品質指標來把關路由，而不只是離線 eval 集。

**Type:** Learn
**Languages:** Python (stdlib, toy cascading router simulator)
**Prerequisites:** Phase 17 · 01 (Managed LLM Platforms), Phase 17 · 19 (AI Gateways)
**Time:** ~60 分鐘

## 學習目標 (Learning Objectives)

- 解釋 model cascading：便宜優先並做信心檢查，信心低時升級。
- 列舉四個 routing 訊號（任務分類、prompt 長度、與已知困難集合的 embedding 相似度、第一輪的自我信心）。
- 在目標 routing 分流比例與品質損失容忍度下，計算預期的混合成本。
- 說出能抓到便宜模型悄悄蔓延的漂移監控指標（線上品質關卡）。

## 問題 (The Problem)

你的服務在 GPT-5 上每月花 $80k。分析顯示 70% 的查詢很簡單：「巴黎現在幾點？」「改寫這句話。」Haiku 等級的模型以 3% 的成本就能完美處理。30% 需要 GPT-5 的推理能力：寫程式、數學、多步驟規劃。

如果你把 70% 路由到便宜模型、30% 到昂貴模型，帳單在相同的產品品質下大約降低 ~65%。這就是 routing。訣竅在於建立 broker 又不讓品質退步。

## 核心概念 (The Concept)

### 四個 routing 訊號

1. **任務分類**：簡單/複雜/codegen/數學/聊天。可以是規則式分類器、小型 LLM（$0.25/M 的 Haiku 等級），或是對有標籤的桶子做 embedding 相似度比對。輸出：route = cheap / balanced / frontier。

2. **Prompt 長度**：>4K tokens 的 prompt 通常需要前沿模型才能維持連貫。<500 tokens 的 prompt 通常不需要。

3. **與已知困難集合的 embedding 相似度**：如果查詢接近（cosine > 0.88）已知困難的桶子，直接升級到前沿模型。

4. **第一輪的自我信心**：先送到便宜模型；如果模型的 log-probs 顯示低信心，或它拒絕回答，或輸出含有模稜兩可的語句，就在前沿模型上重試。約 10% 的流量會增加 P95 latency，但其餘 90% 可省 50%+。

### 三種模式

**Pre-route**（先放分類器）：增加約 ~5-10ms latency；整體最快。

**Cascade**（便宜優先，信心低時升級）：中位數 latency 約 ~1.2x（便宜模型執行加驗證），升級的約 ~2x。品質下限最佳。

**Ensemble route**（對樣本同時執行便宜與前沿模型，由 reward model 挑選）：品質最高、成本最高；僅用於關鍵的 A/B。

### 實作

AI gateway (Phase 17 · 19) 提供 routing。LiteLLM 有帶 fallback 與成本路由的 `router` 設定。Portkey 有 guards + routing。Kong AI Gateway 有外掛式的 routing。OpenRouter 的模型市集提供推薦 API。

開源：RouteLLM (LMSYS)、Prompt Mule。商業：Not Diamond。

### 2026 年的價格曲線

| 模型等級 | 2022 年底 | 2026 | 變化 |
|-------------|-----------|------|--------|
| GPT-4 等級品質 | ~$20/M | ~$0.40/M | 便宜 50x |
| 前沿 (GPT-5, Claude 4) | — | ~$3-10/M | 新的一級 |

大部分進步來自 serving 效率，Phase 17 · 04-09 的核心課程轉化成供應商端的成本下降。Routing 讓你在應用層就拿到這些收益，而不必等所有使用者都遷移到便宜的等級。

### 漂移才是真正的風險

你的路由把 40% 送到便宜模型。六個月後，任務分布改變（使用者變得更成熟，問題更長）。Router 沒有察覺，因為它的分類器是用 Q1 的資料訓練的。品質悄悄下降。沒人抱怨得夠大聲。你是在輸掉的競爭對手 benchmark 上才發現。

用線上品質指標把關路由：

- 每條路由的使用者按讚 / 按倒讚。
- 對每條路由的保留樣本 (5%) 做自動 LLM-judge。
- 升級率：如果 cascade 往上升級的比例 >30%，代表便宜模型被過度路由。
- 每條路由的拒絕率。

### 該記住的數字

- 2026 年同等品質下的 routing 節省：案例研究顯示 20-60%。
- 固定能力等級的 LLM 價格下降：約每年 ~10x（a16z LLMflation 估計）；GPT-4 等級從 2022 到 2026 總共約 50x。
- GPT-4 等級 2022 vs 2026：~$20/M → ~$0.40/M。
- Cascade 的 latency 影響：中位數約 ~1.2x，升級者約 ~2x（約 ~10% 的流量）。

```figure
model-cascade-router
```

## 實際使用 (Use It)

`code/main.py` 在混合工作負載上模擬 pre-route、cascade 與 ensemble。回報混合成本、品質損失與升級率。

## 交付 (Ship It)

這堂課產出 `outputs/skill-router-plan.md`。給定工作負載與品質預算，挑選 routing 模式與訊號。

## 練習 (Exercises)

1. 執行 `code/main.py`。準確度下限為多少時，cascade 會勝過 pre-route？
2. 你的使用者群是 30% 企業（複雜查詢）、70% 免費方案（簡單）。設計 routing 分流比例。哪個線上指標負責把關？
3. 某條路由讓品質下降 2%，但省下 40%。要上線嗎？這取決於產品，請正反兩面都辯論。
4. 使用 OpenAI / Anthropic API 的 logprobs 實作信心檢查。你一開始的門檻是多少？
5. 六個月內，升級率從 8% 爬升到 22%。診斷三個原因，以及每個的修法。

## 關鍵術語 (Key Terms)

| 術語 | 一般說法 | 實際意義 |
|------|----------------|------------------------|
| Model routing | 「成本 broker」 | 每個 request 動態選擇模型 |
| Model cascade | 「便宜優先、升級」 | 先跑便宜的，信心低時落到前沿模型 |
| Pre-route | 「先分類」 | 先放分類器；不重新執行 |
| Ensemble route | 「平行挑選」 | 執行多個，由 reward model 挑出最好的 |
| Escalation rate | 「往上路由 %」 | cascade request 中被升級的比例 |
| RouteLLM | 「LMSYS router」 | 開源 router 函式庫 |
| Not Diamond | 「商業 router」 | SaaS 模型 routing 產品 |
| Drift | 「便宜蔓延」 | 分布改變而 router 沒有察覺 |
| Online quality gate | 「即時檢查」 | 對即時流量取樣的自動 LLM-judge |

## 延伸閱讀 (Further Reading)

- [AbhyashSuchi — Model Routing LLM 2026 Best Practices](https://web.archive.org/web/20260413143335/https://abhyashsuchi.in/model-routing-llm-2026-best-practices/)
- [Lukas Brunner — Rise of Inference Optimization 2026](https://dev.to/lukas_brunner/the-rise-of-inference-optimization-the-real-llm-infra-trend-shaping-2026-4e4o)
- [RouteLLM paper / code](https://github.com/lm-sys/RouteLLM)
- [Not Diamond — model routing](https://www.notdiamond.ai/)
- [OpenRouter](https://openrouter.ai/)：具備 routing 基本元件的多模型 gateway。
