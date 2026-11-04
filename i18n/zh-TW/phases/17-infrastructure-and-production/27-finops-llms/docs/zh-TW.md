# LLM 的 FinOps：單位經濟學與多租戶 Attribution (FinOps for LLMs)

> 傳統 FinOps 在 LLM 支出上行不通。成本是 token 交易，不是資源運作時間。Tags 對不上：API 呼叫是一筆交易，不是資產。工程決策（prompt 設計、context window、輸出長度）就是財務決策。2026 年的作法有三個 attribution 維度，要在第一天就做 instrumentation：每位使用者 (`user_id`)，用於座位定價與擴充；每項任務 (`task_id` + `route`)，用於產品介面的成本與優先順序；每個 tenant (`tenant_id`)，用於單位經濟學與續約。四個 token 層：prompt、tool、memory、response，只用一個桶子就會藏住花費。多租戶產品的強制執行階梯：每個 tenant 的 rate limit（預期尖峰的 2-3x，清楚的 429 + retry-after）；每日花費上限（合約上限的 1.5-3x；觸發 rate 收緊 + 告警）；花費 z-score > 4 時的 kill switch（自動暫停 + 呼叫 on-call）。Attribution 模式：tag-and-aggregate、telemetry-joiner（trace-ID → 帳單；準確度最高）、sampling-and-extrapolation、model-based allocation、event-sourced、real-time streaming。單位指標：每個已解決查詢的成本、每個生成產物的成本，而不是 $/M tokens。事後補 tag 一定會漏掉；請在建立 request 時就做 instrumentation。

**Type:** Learn
**Languages:** Python (stdlib, toy cost-attribution simulator with kill switch)
**Prerequisites:** Phase 17 · 13 (Observability), Phase 17 · 14 (Caching)
**Time:** ~60 分鐘

## 學習目標 (Learning Objectives)

- 解釋為什麼傳統 FinOps（tags + 分層）在 LLM 支出上行不通，並說出三個新的 attribution 維度。
- 列舉四個 token 層（prompt、tool、memory、response），以及為什麼單一桶子計費會藏住成本。
- 為多租戶產品設計強制執行階梯（rate → 花費上限 → kill switch）。
- 挑選單位指標（每個已解決查詢 / 產物的成本），而不是 $/M tokens。

## 問題 (The Problem)

你的帳單寫著 $40,000。你不知道：
- 是哪個 tenant 花的。
- 是哪個產品功能造成的。
- 有沒有任何個別使用者在濫用。
- 元凶是 prompt 膨脹、tool 呼叫，還是 memory 放大。

供應商端的 tag-and-aggregate 適用於雲端資源（EC2、S3），因為 tags 會傳播到明細項目。LLM API 呼叫不會自動加 tag，你必須在呼叫端蓋上 user/task/tenant 並一路帶下去。事後歸因一定會漏掉邊緣案例。

## 核心概念 (The Concept)

### 三個 attribution 維度

**每位使用者** (`user_id`)：誰花了多少。驅動座位定價、擴充的對話，並找出重度使用者。

**每項任務** (`task_id` + `route`)：哪個產品介面花多少。驅動功能優先順序、砍掉昂貴功能的決策。

**每個 tenant** (`tenant_id`)：哪位客戶有獲利。驅動單位經濟學、續約定價、方案門檻。

第一天就在呼叫端對這三者做 instrumentation。事後補一定比較差。

### 四個 token 層

| 層 | 範例 | 占總量的典型 % |
|-------|---------|---------------------|
| Prompt | system + 使用者輸入 | 40-60% |
| Tool | 回饋進來的 tool 呼叫結果 | 20-40%（agent 工作負載） |
| Memory | 先前對話 / 檢索到的文件 | 10-30% |
| Response | 模型輸出 | 10-30% |

把四者合在一起分桶，會讓優化變成盲目。請在你的 attribution schema 中把它們拆開。

### 強制執行階梯

1. 每個 tenant 的 **rate limit**。預期尖峰的 2-3x。回傳 429 並附 `Retry-After`。Tenant 會感受到摩擦，但不會有意外的帳單。

2. 每個 tenant 的**每日花費上限**。合約上限的 1.5-3x。觸發：收緊 rate limit + 通知客戶成功團隊。

3. 相對於 tenant 基準，花費 z-score > 4 時啟動 **kill switch**。自動暫停 tenant；呼叫 on-call；升級給 ops + CS。

### Attribution 模式

- **Tag-and-aggregate**：蓋上 metadata header；稍後彙總。簡單；粗略。
- **Telemetry joiner**：透過 trace ID 把 traces 與帳單 join 起來。準確度最高。成熟的團隊都這樣做。
- **Sampling + extrapolation**：取樣 5-10%，再乘上去。對粗略的花費來說划算；會漏掉長尾。
- **Model-based allocation**：用迴歸推論成本驅動因素。適用於沒有 tags 的舊資料。
- **Event-sourced**：成本作為串流（Kafka / Kinesis）中的事件。即時。
- **Real-time streaming**：儀表板亞秒級更新。

### 每個 X 的成本才是單位指標

$/M tokens 是供應商的說法。產品指標是：

- 每張已解決支援工單的成本。
- 每篇生成文章的成本。
- 每個成功 agent 任務的成本。
- 每個使用者 session 分鐘的成本。

把成本綁到產品成果上。否則優化就沒有錨點。

### 成本 attribution 的 trace 形狀

```
trace_id: abc123
  user_id: u_42
  tenant_id: t_7
  task_id: task_classify_doc
  route: model_haiku
  layers:
    prompt_tokens: 1800
    tool_tokens: 600
    memory_tokens: 400
    response_tokens: 150
  cost_usd: 0.0135
  cached_input: true
  batch: false
```

每次呼叫都發出。存到資料湖。依各維度彙總。Phase 17 · 13 的 observability stack 就是它存在的地方。

### 疊加節省的 stack

Stack：cache + batch + route + gateway。四者齊備時：
- Cache L2 (Phase 17 · 14)：輸入便宜約 ~10x。
- Batch (Phase 17 · 15)：五折。
- 路由到便宜模型 (Phase 17 · 16)：成本降低 60%。
- Gateway 效率 (Phase 17 · 19)：備援 + retries。

最佳情況疊加：約為單純基準的 ~5-10%。多數團隊啟用了 2-3 個槓桿；很少有團隊把四個全疊上。

### 該記住的數字

- Attribution 維度：每位使用者、每項任務、每個 tenant。
- 四個 token 層：prompt、tool、memory、response。
- Kill switch：花費 z-score > 4。
- 單位指標：每個已解決查詢的成本，而不是 $/M tokens。
- 疊加優化：可能降到基準的約 ~5-10%。

```figure
i4-spend-ladder
```

## 實際使用 (Use It)

`code/main.py` 以三層強制執行階梯模擬多租戶 LLM 服務。注入一個濫用的 tenant，並展示 kill switch 啟動。

## 交付 (Ship It)

這堂課產出 `outputs/skill-finops-plan.md`。給定產品與規模，設計 attribution schema 與強制執行階梯。

## 練習 (Exercises)

1. 執行 `code/main.py`。kill switch 在哪個 z-score 啟動？你如何挑選門檻？
2. 設計每個 tenant、每項任務的成本儀表板。你最先建立的 5 個視圖是什麼？
3. 你最大的 tenant 單位經濟學為負。依對客戶影響的大小排序，提出三個介入措施。
4. 計算支援產品每張已解決工單的成本：每張工單 3M tokens，每天約 ~800 張工單，所有輸入都以 GPT-5 快取費率 $0.125/M（發布時的定價；請查看即時費率表）。
5. 辯論事後補 tag 是否可能有效。什麼時候可以接受？

## 關鍵術語 (Key Terms)

| 術語 | 一般說法 | 實際意義 |
|------|----------------|------------------------|
| Per-user attribution | 「使用者層級成本」 | 每次呼叫都蓋上 `user_id` |
| Per-task attribution | 「功能成本」 | `task_id` + `route` 識別產品介面 |
| Per-tenant attribution | 「客戶成本」 | `tenant_id`；驅動單位經濟學 |
| Four token layers | 「成本層」 | prompt + tool + memory + response |
| Rate limit | 「429 防護」 | 在 gateway 強制執行的每個 tenant 上限 |
| Daily spend cap | 「每日上限」 | 帶告警、以 tenant 為範圍的預算 |
| Kill switch | 「自動暫停」 | 花費 z-score > 4 時觸發自動停權 |
| Cost per resolved | 「產品單位指標」 | 綁定產品成果而非 tokens 的成本 |
| Telemetry joiner | 「trace 對帳單」 | 準確度最高的 attribution 模式 |
| Stacked optimization | 「cache+batch+route+gateway」 | 疊加的節省，可降到基準的約 ~5-10% |

## 延伸閱讀 (Further Reading)

- [FinOps Foundation — FinOps for AI Overview](https://www.finops.org/wg/finops-for-ai-overview/)
- [FinOps School — Cost per Unit 2026 Guide](https://finopsschool.com/blog/cost-per-unit/)
- [Digital Applied — LLM Agent Cost Attribution 2026](https://www.digitalapplied.com/blog/llm-agent-cost-attribution-guide-production-2026)
- [PointFive — Managed LLMs in Azure OpenAI](https://www.pointfive.co/blog/finops-for-ai-economics-of-managed-llms-in-azure-open-ai)
