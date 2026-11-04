# Capstone 11 — LLM 可觀測性與評測 Dashboard (LLM Observability & Eval Dashboard)

> Langfuse 走向 open-core。Arize Phoenix 發佈了 2026 GenAI semconv 的對應。Helicone 與 Braintrust 都加碼投入每位使用者的成本歸屬。Traceloop 的 OpenLLMetry 成為事實上的 SDK 儀表化標準。正式環境的形態是：ClickHouse 存 trace、Postgres 存 metadata、Next.js 做 UI，再加上一小隊在取樣 trace 上執行的評測工作 (DeepEval、RAGAS、LLM-judge)。自行架設一套，從至少四個 SDK 家族匯入資料，並展示在五分鐘內抓到被注入的退步。

**Type:** Capstone
**Languages:** TypeScript (UI), Python / TypeScript (ingest + evals), SQL (ClickHouse)
**Prerequisites:** Phase 11 (LLM engineering), Phase 13 (tools), Phase 17 (infrastructure), Phase 18 (safety)
**Phases exercised:** P11 · P13 · P17 · P18
**Time:** 25 小時

## 問題 (Problem)

2026 年每個跑正式流量的 AI 團隊，都在模型旁邊維護一個可觀測性平面。成本歸屬。幻覺偵測。Drift 監控。Jailbreak 訊號。SLO dashboard。PII 洩漏警示。開源的參考專案 Langfuse、Phoenix、OpenLLMetry，都收斂到以 OpenTelemetry GenAI semantic conventions 作為匯入 schema。現在你可以用一套 SDK 對 OpenAI、Anthropic、Google、LangChain、LlamaIndex 和 vLLM 做儀表化，並送出相容的 span。

你要打造一個自行架設的 dashboard，從至少四個 SDK 家族匯入資料，在取樣的 trace 上執行一小組評測工作，偵測 drift 並發出警示。衡量標準：給定一個刻意注入的退步 (某個 prompt 開始產生 PII)，dashboard 要在五分鐘內抓到並觸發警示。

## 核心概念 (Concept)

匯入是 OTLP HTTP。SDK 產生 GenAI-semconv 的 span：`gen_ai.system`、`gen_ai.request.model`、`gen_ai.usage.input_tokens`、`gen_ai.response.id`、`llm.prompts`、`llm.completions`。Span 進入 ClickHouse 做欄式分析；metadata (使用者、session、app) 進入 Postgres。

評測以批次工作在取樣的 trace 上執行。DeepEval 為 faithfulness、toxicity 與 answer relevance 評分。當 trace 帶有檢索 context 時，RAGAS 為檢索指標評分。自訂 LLM-judge 執行領域專屬的檢查 (PII 洩漏、偏離政策的回應)。評測結果以連結到父 trace 的 eval span 寫回同一個 ClickHouse。

Drift 偵測監看 embedding 空間分布隨時間的變化 (對 prompt embedding 做 PSI 或 KL divergence)，加上評測分數的趨勢。警示送到 Prometheus Alertmanager，再到 Slack / PagerDuty。UI 是 Next.js 15 搭配 Recharts。

## 架構 (Architecture)

```
production apps:
  OpenAI SDK  +  Anthropic SDK  +  Google GenAI SDK
  LangChain + LlamaIndex + vLLM
       |
       v
  OpenTelemetry SDK with GenAI semconv
       |
       v  OTLP HTTP
  collector (ingest, sample, fan-out)
       |
       +-------------+-----------+
       v             v           v
   ClickHouse    Postgres    S3 archive
   (spans)       (metadata)  (raw events)
       |
       +---> eval jobs (DeepEval, RAGAS, LLM-judge)
       |     sampled or all-trace
       |     write eval spans back
       |
       +---> drift detector (PSI / KL on prompt embeddings)
       |
       +---> Prometheus metrics -> Alertmanager -> Slack / PagerDuty
       |
       v
   Next.js 15 dashboard (Recharts)
```

## 技術堆疊 (Stack)

- 匯入：OpenTelemetry SDK + GenAI semantic conventions；OTLP HTTP 傳輸
- Collector：具備 tail-sampling processor 的 OpenTelemetry Collector (用於成本控制)
- 儲存：ClickHouse 存 span、Postgres 存 metadata、S3 存原始事件封存
- 評測：DeepEval、RAGAS 0.2、Arize Phoenix evaluator pack、自訂 LLM-judge
- Drift：每週對合併的 prompt embedding (sentence-transformers) 做 PSI / KL
- 警示：Prometheus Alertmanager -> Slack / PagerDuty
- UI：Next.js 15 App Router + Recharts + server actions
- 開箱即用支援的 SDK：OpenAI、Anthropic、Google GenAI、LangChain、LlamaIndex、vLLM

```figure
ce-otel-drift
```

## 動手做 (Build It)

1. **Collector 設定。** 具備 OTLP HTTP receiver 的 OpenTelemetry Collector、保留 100% 錯誤 trace 與 10% 成功 trace 的 tail-sampler，以及輸出到 ClickHouse 與 S3 的 exporter。

2. **ClickHouse schema。** 資料表 `spans`，欄位對應 GenAI semconv：`gen_ai_system`、`gen_ai_request_model`、`input_tokens`、`output_tokens`、`latency_ms`、`prompt_hash`、`trace_id`、`parent_span_id`，外加存放長 payload 的 JSON 欄位。依 user_id 與 app_id 加上次級索引。

3. **SDK 涵蓋度測試。** 用每個 SDK (OpenAI、Anthropic、Google、LangChain、LlamaIndex、vLLM) 搭配 OpenLLMetry 自動儀表化，寫一個小型 client 應用程式。驗證每個都產生標準的 GenAI span 並進入 ClickHouse。

4. **評測工作。** 排程工作讀取最近 15 分鐘取樣的 trace，執行 DeepEval 的 faithfulness、toxicity 與 answer relevance。輸出是連結到父 trace 的 eval span。

5. **自訂 LLM-judge。** PII 洩漏評審：給定一個回應，呼叫一個守門 LLM，評估 PII 洩漏的可能性。高分的回應進入分流佇列。

6. **Drift 偵測。** 每週工作計算本週合併 prompt embedding 與前 4 週基準線之間的 PSI。如果 PSI 超過閾值，就發出警示。

7. **Dashboard。** Next.js 15，頁面包含：總覽 (spans/sec、cost/user、p95 延遲)、traces (搜尋 + 瀑布圖)、evals (faithfulness 趨勢、toxicity)、drift (PSI 隨時間變化)、警示。

8. **警示鏈。** Prometheus exporter 讀取評測分數彙總與延遲百分位；Alertmanager 將警告路由到 Slack，嚴重違規路由到 PagerDuty。

9. **退步探測。** 注入一個 bug：被評測的 chatbot 有 1% 的機率洩漏假的 SSN。衡量偵測時間 (MTTD)：從 bug 部署到 Slack 警示。偵測是 dashboard 負責的 MTTR 部分；修復與重新部署接在後面。

## 實際使用 (Use It)

```
$ curl -X POST https://my-otel-collector/v1/traces -d @trace.json
[collector]  accepted 1 trace, 3 spans
[clickhouse] inserted 3 spans (app=chat, user=u_42)
[eval]       DeepEval faithfulness 0.82, toxicity 0.03
[drift]      weekly PSI 0.08 (below 0.2 threshold)
[ui]         live at https://obs.example.com
```

## 交付 (Ship It)

`outputs/skill-llm-observability.md` 是交付成果。給定一個 LLM 應用程式，dashboard 會匯入它的 trace、執行評測、對 drift 發出警示，並在 Next.js 中呈現每位使用者的成本拆解。

| 權重 | 評分項目 | 衡量方式 |
|:-:|---|---|
| 25 | Trace schema 涵蓋度 | 產生標準 GenAI span 的 SDK 家族數量 (目標：6+) |
| 20 | 評測正確性 | DeepEval / RAGAS 分數與人工標註集的比較 |
| 20 | Dashboard UX | 注入退步的 MTTD (目標低於 5 分鐘) |
| 20 | 成本 / 規模 | 持續以 1k spans/sec 匯入且無積壓 |
| 15 | 警示 + drift 偵測 | Prometheus/Alertmanager 鏈端到端運作 |
| **100** | | |

## 練習 (Exercises)

1. 為 Haystack 框架加入自訂儀表化。驗證標準 span 進入 ClickHouse，且 `gen_ai.*` 屬性忠實。

2. 在同樣的 trace 上，把 DeepEval 換成 Phoenix evaluators。衡量兩個評測引擎之間的分數偏移。

3. 讓 drift 偵測器更精準：依 app-id 計算 PSI，而不是全域計算。展示每個 app 的 drift 軌跡。

4. 加入「使用者影響」頁面：每位使用者的成本與失敗率，附 sparkline。

5. 建立 tail-sampling 政策：保留 100% toxicity > 0.5 的 trace，加上其餘 trace 的 10% 分層抽樣。衡量引入的取樣偏差。

## 關鍵術語 (Key Terms)

| 術語 | 一般說法 | 實際意義 |
|------|-----------------|------------------------|
| GenAI semconv | 「OTel LLM 屬性」 | 2025 年 OpenTelemetry 針對 LLM span 屬性 (system、model、tokens) 的規格 |
| Tail sampling | 「Trace 後取樣」 | Collector 在 trace 完成後才決定保留或丟棄 (可以偷看錯誤) |
| PSI | 「Population stability index」 | 比較兩個分布的 drift 指標；> 0.2 通常代表有意義的 drift |
| LLM-judge | 「用模型做評測」 | 由一個 LLM 依評分標準 (faithfulness、toxicity、PII) 為另一個 LLM 的輸出評分 |
| Tail-sampling policy | 「保留規則」 | 決定哪些 trace 要保存、哪些要丟棄的規則；錯誤 + 取樣率 |
| Eval span | 「連結的評測 trace」 | 帶有評測分數、連結到原始 LLM 呼叫 span 的子 span |
| Cost per user | 「單位經濟」 | 一段時間內歸屬於某個 user_id 的美元成本；關鍵產品指標 |

## 延伸閱讀 (Further Reading)

- [Langfuse](https://github.com/langfuse/langfuse) — 參考用的 open-core 可觀測性平台
- [Arize Phoenix](https://github.com/Arize-ai/phoenix) — 另一個參考，drift 支援強
- [OpenLLMetry (Traceloop)](https://github.com/traceloop/openllmetry) — 自動儀表化 SDK 家族
- [OpenTelemetry GenAI semantic conventions](https://opentelemetry.io/docs/specs/semconv/gen-ai/) — 匯入 schema
- [Helicone](https://www.helicone.ai) — 另一個託管的可觀測性服務
- [Braintrust](https://www.braintrust.dev) — 另一個以評測為先的平台
- [ClickHouse documentation](https://clickhouse.com/docs) — 欄式 span 儲存
- [DeepEval](https://github.com/confident-ai/deepeval) — 評測函式庫
