# Agent Observability：Langfuse、Phoenix、Opik (Agent Observability: Langfuse, Phoenix, Opik)

> 2026 年有三個可自架的 agent observability 平台佔據主導。Langfuse（MIT）：每月 6M+ 安裝，提供 tracing + prompt 管理 + eval + session 重播。Arize Phoenix（Elastic 2.0）：深入的 agent 專屬 eval、RAG 相關性、OpenInference 自動 instrumentation。Comet Opik（Apache 2.0）：自動 prompt 最佳化、guardrail、LLM-judge 幻覺偵測。

**Type:** Learn
**Languages:** Python (stdlib)
**Prerequisites:** Phase 14 · 23 (OTel GenAI)
**Time:** ~45 分鐘

## 學習目標 (Learning Objectives)

- 說出三個頂尖的可自架 agent observability 平台及其授權，並指出哪個授權不是 OSI 開源。
- 區分各自最強的地方：Langfuse（prompt 管理 + session）、Phoenix（RAG + 自動 instrumentation）、Opik（最佳化 + guardrail）。
- 解釋為什麼到 2026 年有 89% 的組織回報已建立 agent observability。
- 用 stdlib 實作從 trace 到儀表板的 pipeline，並搭配 LLM-judge 評估。

## 問題 (The Problem)

OTel GenAI（Lesson 23）給了你 schema。你仍然需要一個平台來匯入 span、執行評估、儲存 prompt 版本，並找出退化。這三個競爭者各自強調生命週期的不同部分。

## 核心概念 (The Concept)

### Langfuse (MIT)

- 每月 6M+ SDK 安裝，19k+ GitHub star。
- 功能：tracing、帶版本與 playground 的 prompt 管理、評估（LLM-as-judge、使用者回饋、自訂）、session 重播。
- 2025 年 6 月：先前屬於商業版的模組（LLM-as-a-judge、標註佇列、prompt 實驗、Playground）以 MIT 授權開源。
- 最強的地方：端到端 observability，加上緊密的 prompt 管理 loop。

### Arize Phoenix (Elastic License 2.0)

- 更深入的 agent 專屬評估：trace 分群、異常偵測、RAG 的 retrieval 相關性。
- 原生 OpenInference 自動 instrumentation。
- 搭配託管的 Arize AX 用於正式環境。
- 沒有 prompt 版本管理，定位是與更廣泛平台並用的 drift/行為退化工具。
- 最強的地方：RAG 相關性、行為 drift、異常偵測。

### Comet Opik (Apache 2.0)

- 透過 A/B 實驗做自動 prompt 最佳化。
- Guardrail（PII 遮蔽、主題限制）。
- LLM-judge 幻覺偵測。
- 來自 Comet 自己測量的 benchmark：Opik 記錄 + 評估花 23.44 秒，Langfuse 花 327.15 秒（約 14 倍差距）；廠商 benchmark 請視為僅供參考的方向。
- 最強的地方：最佳化 loop、自動化實驗、guardrail 強制執行。

### 產業資料 (Industry data)

根據 Maxim（2026 年實地分析）：89% 的組織已建立 agent observability；品質問題是正式環境的最大障礙（32% 的受訪者提到）。

### 挑選一個 (Picking one)

| 需求 | 選擇 |
|------|------|
| 內含 prompt 管理的全方位平台 | Langfuse |
| 深入的 RAG 評估 + drift | Phoenix |
| 自動最佳化 + guardrail | Opik |
| 開放授權，不要 ELv2 | Langfuse (MIT) 或 Opik (Apache 2.0) |
| Datadog / New Relic 整合 | 任一個：它們都輸出 OTel |

### 這個模式哪裡會出錯 (Where this pattern goes wrong)

- **沒有 eval 策略。** 只有 tracing 而沒有評估，只是昂貴的 logging。
- **自己做的 LLM-judge 沒有依據。** CRITIC 模式（Lesson 05）同樣適用：judge 需要外部 tool 來做事實驗證。
- **Prompt 版本沒有綁定 trace。** 當正式環境退化時，你無法二分查找出造成問題的 prompt。

```figure
wb-trace-ingest
```

## 動手做 (Build It)

`code/main.py` 實作 stdlib 的 trace 收集器 + LLM-judge 評估器：

- 匯入 GenAI 形狀的 span。
- 依 session 分組，標記失敗的執行（guardrail 觸發、低信心評估）。
- 一個腳本化的 LLM-judge，依評分標準為 agent 的回應打分。
- 類似儀表板的摘要：失敗率、主要失敗原因、eval 分數分佈。

執行：

```
python3 code/main.py
```

輸出：每個 session 的 eval 分數與失敗分類，與 Langfuse/Phoenix/Opik 會顯示的相符。

## 實際使用 (Use It)

- **Langfuse** 自架或雲端；透過 OTel 或他們的 SDK 接上。
- **Arize Phoenix** 自架；自動 instrument OpenInference。
- **Comet Opik** 自架或雲端；自動最佳化 loop。
- **Datadog LLM Observability** 適合已經在跑 Datadog 的混合維運+ML 團隊。

## 交付 (Ship It)

`outputs/skill-obs-platform-wiring.md` 挑選一個平台，並把 trace + eval + prompt 版本接進既有的 agent。

## 練習 (Exercises)

1. 把一週的 OTel trace 匯出到 Langfuse cloud（免費方案）。哪些 session 失敗了？為什麼？
2. 為你的領域寫一份 LLM-judge 評分標準（事實正確性、語氣、範圍遵守度）。在 50 筆 trace 上測試。
3. 比較 Langfuse 的 prompt 版本管理與 Phoenix 的 trace 分群。哪個能更快告訴你是什麼壞掉了？
4. 讀 Opik 的 guardrail 文件。把 PII 遮蔽 guardrail 接到你其中一次 agent 執行。
5. 在你的語料上對這三者做 benchmark。忽略廠商公布的數字；量測你自己的。

## 關鍵術語 (Key Terms)

| Term | 一般說法 | 實際意義 |
|------|----------------|------------------------|
| Tracing | 「Span 收集器」 | 匯入 OTel / SDK span；依 session 建立索引 |
| Prompt management | 「Prompt CMS」 | 與 trace 綁定的版本化 prompt |
| LLM-as-judge | 「自動化 eval」 | 另一個 LLM 依評分標準為 agent 輸出打分 |
| Session replay | 「Trace 回放」 | 逐步檢視過去的執行以便 debug |
| RAG relevancy | 「Retrieval 品質」 | 取回的 context 是否符合查詢 |
| Trace clustering | 「行為分群」 | 將相似的執行分群以偵測 drift |
| Guardrail enforcement | 「記錄時的政策」 | 對記錄內容做 PII/毒性/範圍檢查 |

## 延伸閱讀 (Further Reading)

- [Langfuse docs](https://langfuse.com/)：tracing、eval、prompt 管理
- [Arize Phoenix docs](https://docs.arize.com/phoenix)：自動 instrumentation、drift
- [Comet Opik](https://www.comet.com/site/products/opik/)：最佳化 + guardrail
- [OpenTelemetry GenAI semantic conventions](https://opentelemetry.io/docs/specs/semconv/gen-ai/)：三者都使用的 schema
