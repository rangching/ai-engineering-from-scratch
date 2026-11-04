# LLM Routing 層：LiteLLM、OpenRouter、Portkey (LLM Routing Layer — LiteLLM, OpenRouter, Portkey)

> 被單一 provider 綁死的代價很高。不同的 tool-calling 工作負載適合不同的模型。Routing gateway 提供單一 API 介面、重試、failover、成本追蹤與 guardrail。2026 年有三種典型：LiteLLM（開源、自行託管）、OpenRouter（託管式 SaaS）、Portkey（production 等級，2026 年 3 月開源）。這堂課會列出決策準則，並走過一個 stdlib 的 routing gateway。

**Type:** Learn
**Languages:** Python (stdlib, routing + failover + cost tracker)
**Prerequisites:** Phase 13 · 02 (function calling), Phase 13 · 17 (gateways)
**Time:** ~45 分鐘

## 學習目標 (Learning Objectives)

- 區分自行託管、託管式與 production 等級的 routing 選項。
- 實作 fallback chain，在 provider 失敗時依定義好的優先順序重試。
- 跨 provider 追蹤每個請求的成本與 token 用量。
- 針對特定的 production 限制，在 LiteLLM、OpenRouter 與 Portkey 之間做出決定。

## 問題 (The Problem)

Provider routing 重要的情境：

1. **成本。** Claude Sonnet 的價格是 Haiku 的 3 倍。分流（triage）任務用 Haiku 就夠；綜合整理的任務值得用 Sonnet。依每個請求來路由。

2. **Failover。** OpenAI 有一個小時狀況不好，每個請求都失敗。你想要在不重新部署的情況下，自動 fallback 到 Anthropic。

3. **延遲。** 即時聊天 UI 需要快速的 time-to-first-token。批次摘要器則不需要。依延遲 SLA 路由。

4. **合規。** 歐盟的使用者必須留在歐盟區域。依區域路由。

5. **實驗。** 在同一個工作負載上對兩個模型做 A/B。依測試分組路由。

為每個整合都手寫這一切，是重複的工作。Routing gateway 提供一個與 OpenAI 相容的 API，並處理其餘的事。

## 核心概念 (The Concept)

### 與 OpenAI 相容的 proxy 形狀

每個人都說 OpenAI 的形狀。Routing gateway 公開 `/v1/chat/completions`，接受 OpenAI 的 schema，並在內部 proxy 到 Anthropic / Gemini / Cohere / Ollama / 任何東西。Client 不必在意。

### Model alias

你的程式碼不寫固定的 snapshot id，而是寫 `our_smart_model`。Gateway 把 alias 對應到真正的模型。當 provider 推出新一代時，你在 server 端改 alias；你的程式碼完全不用動。

### Fallback chain

```
primary: openai/gpt-4o
on 5xx: anthropic/claude-3-5-sonnet
on 5xx: google/gemini-1.5-pro
on 5xx: refuse
```

Gateway 在設定檔中定義這個。重試會計入預算，所以 fallback 的連鎖反應不會讓成本爆炸。

### 語意快取 (Semantic caching)

完全相同或近乎相同的 prompt 會命中快取，而不是送到 provider。對重複的 agent 迴圈，節省可達 30% 到 60%。Key 是以 embedding 為基礎；近乎相同的 prompt 共用一個快取位置。

### Guardrail

Gateway 層級：

- **PII 遮蔽。** 在送出 prompt 之前，用 regex 或以 ML 為基礎的處理。
- **政策違規。** 拒絕含有禁止內容的 prompt。
- **輸出過濾。** 清除 completion 中的洩漏。

Portkey 與 Kong 都提供有主見的 guardrail。LiteLLM 則讓它們維持選用。

### 每個 key 的 rate limit

一個 API key 對應一個團隊。每個 key 的預算，避免單一團隊耗盡共用的配額。大多數 gateway 都支援。

### 自行託管 vs 託管式的取捨

| 因素 | LiteLLM（自行託管） | OpenRouter（託管式） | Portkey（production） |
|--------|----------------------|----------------------|----------------------|
| 程式碼 | 開源、Python | 託管式 SaaS | 開源（2026 年 3 月）+ 託管 |
| 設定 | 部署 proxy | 註冊 | 兩者皆可 |
| Provider | 100+ | 300+ | 100+ |
| 計費 | 你自己的 key | OpenRouter 點數 | 你自己的 key |
| Observability | OpenTelemetry | Dashboard | 完整 OTel + PII 遮蔽 |
| 最適合 | 想要完全掌控的團隊 | 快速原型 | 需要合規的 production |

當你有 SRE 團隊並想要資料主權時，LiteLLM 勝出。當你想要單一訂閱且不想碰基礎設施時，OpenRouter 勝出。當你需要現成的 guardrail 與合規時，Portkey 勝出。

### 成本追蹤

每個請求都帶有 `provider`、`model`、`input_tokens`、`output_tokens`。乘以每個模型每個 token 的價格（取自 gateway 維護的價格表）。再依使用者 / 團隊 / 專案彙總。

### MCP 加上 routing

Gateway 可以同時路由 LLM call 和 MCP sampling 請求。當 sampling 請求的 modelPreferences 偏好特定模型時，gateway 會轉譯到正確的 backend。這就是 Phase 13 · 17（MCP gateway）與這堂課的 routing gateway 有時會合併成一個服務的地方。

### Routing 策略

- **靜態優先順序。** 清單中的第一個；出錯時 fallback。
- **負載平衡。** Round-robin 或加權。
- **成本感知。** 選擇符合延遲 / 品質要求的最便宜模型。
- **延遲感知。** 選擇最近 N 分鐘內最快的模型。
- **任務感知。** 由 prompt 分類器把寫程式路由到一個模型，把摘要路由到另一個。

```figure
tp-router-failover
```

## 實際使用 (Use It)

`code/main.py` 用約 150 行實作 routing gateway：接受 OpenAI 形狀的請求，轉譯為每個 provider 的 stub，執行優先順序的 fallback chain，追蹤每個請求的成本，並對輸入套用 PII 遮蔽處理。用三個情境執行它：正常請求、主要 provider 中斷而觸發 fallback，以及被遮蔽攔下的 PII 洩漏。

觀察重點：

- `ROUTES` dict：alias -> 依優先順序排列的具體 provider 清單。
- Fallback 迴圈在 5xx 時重試。
- 成本追蹤器把 token 用量乘以每個模型的費率。
- PII 遮蔽器在轉發之前，清除類似 SSN 的模式。

## 交付 (Ship It)

這堂課會產出 `outputs/skill-routing-config-designer.md`。給定一份工作負載輪廓（延遲、成本、合規），這個 skill 會挑選 LiteLLM / OpenRouter / Portkey，並產出 routing 設定。

## 練習 (Exercises)

1. 執行 `code/main.py`。觸發中斷情境；確認 fallback 落在第二個 provider，且成本被正確歸屬。

2. 加入語意快取：prompt 的 SHA256 作為查找 key；快取命中時立即回傳。測量重複呼叫時節省的成本。

3. 加入 prompt 分類器，把「code ...」開頭的 prompt 路由到偏重智慧的 alias，把「summarize ...」開頭的 prompt 路由到偏重速度的 alias。

4. 設計每個團隊的預算：每個團隊有每月的花費上限；gateway 達到上限後就拒絕請求。選擇一個強制執行的粒度（每個請求，或視窗式）。

5. 並排閱讀 LiteLLM、OpenRouter 和 Portkey 的文件。指出每一個提供、但另外兩個都沒有的功能。

## 關鍵術語 (Key Terms)

| 術語 | 一般說法 | 實際意義 |
|------|----------------|------------------------|
| Routing gateway | 「LLM proxy」 | 位於多個 provider 前面、提供單一 API 介面的層 |
| OpenAI-compatible | 「說 OpenAI 的 schema」 | 接受 `/v1/chat/completions` 的形狀，並轉譯到任何 backend |
| Model alias | 「our_smart_model」 | 你程式碼中的名稱，由 gateway 對應到具體的模型 |
| Fallback chain | 「重試清單」 | 失敗時依序嘗試的 provider 有序清單 |
| Semantic caching | 「Prompt embedding 快取」 | Key 是 prompt 的 embedding；近乎重複的 prompt 共用一次快取命中 |
| Guardrails | 「輸入/輸出過濾」 | 遮蔽 PII、拒絕政策違規 |
| Per-key rate limit | 「團隊預算」 | 限定在某個 API key 的配額 |
| Cost tracking | 「每個請求的花費」 | 彙總 token 用量 x 每個模型的價格 |
| LiteLLM | 「開放的 proxy」 | 可自行託管的開源 routing gateway |
| OpenRouter | 「託管式 SaaS」 | 以點數計費的託管 gateway |
| Portkey | 「Production 選項」 | 開源加託管，內建 guardrail |

## 延伸閱讀 (Further Reading)

- [LiteLLM — docs](https://docs.litellm.ai/) — 自行託管的 routing gateway
- [OpenRouter — quickstart](https://openrouter.ai/docs/quickstart) — 託管式 routing SaaS
- [Portkey — docs](https://portkey.ai/docs) — 附 guardrail 的 production routing
- [TrueFoundry — LiteLLM vs OpenRouter](https://www.truefoundry.com/blog/litellm-vs-openrouter) — 決策指南
- [Relayplane — LLM gateway comparison 2026](https://relayplane.com/blog/llm-gateway-comparison-2026) — 廠商調查
