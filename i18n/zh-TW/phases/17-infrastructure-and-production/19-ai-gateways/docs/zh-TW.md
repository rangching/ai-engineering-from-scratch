# AI Gateways：LiteLLM、Portkey、Kong AI Gateway、Bifrost

> Gateway 位於你的 app 與模型供應商之間。核心功能是供應商路由、fallback、retries、rate limiting、secret 參照、observability、guardrails。2026 年的市場分野：**LiteLLM** 是 MIT 開源，支援 100+ 供應商、相容 OpenAI，但在約 ~2000 RPS 時就撐不住（已發布的 benchmark 中 8 GB 記憶體、連鎖失敗）；最適合 Python、<500 RPS、開發/原型。**Portkey** 定位為 control plane（guardrails、PII redaction、jailbreak 偵測、audit trails），2026 年 3 月改為 Apache 2.0 開源，latency 開銷 20-40 ms，production 方案 $49/mo。**Kong AI Gateway** 建立在 Kong Gateway 之上，Kong 自己在相同 12 個 CPU 上的 benchmark：比 Portkey 快 228%、比 LiteLLM 快 859%；定價 $100/model/month（Plus 方案最多 5 個）；如果你已經在用 Kong，就很合適企業。**Bifrost** (Maxim AI)：自動 retries 且 backoff 可設定，OpenAI 429 時 fallback 到 Anthropic。**Cloudflare / Vercel AI Gateways**：託管、零維運、基本 retry。資料落地決定了是否自架；Portkey 與 Kong 處於中間，提供 OSS 加可選的託管。

**Type:** Learn
**Languages:** Python (stdlib, toy gateway-routing simulator)
**Prerequisites:** Phase 17 · 01 (Managed LLM Platforms), Phase 17 · 16 (Model Routing)
**Time:** ~60 分鐘

## 學習目標 (Learning Objectives)

- 列舉七個核心 gateway 功能（routing、fallback、retries、rate limits、secrets、observability、guardrails）。
- 把四個 2026 年的 gateway（LiteLLM、Portkey、Kong AI、Bifrost）對應到規模上限與使用情境。
- 引用 Kong benchmark（比 Portkey 快 228%、比 LiteLLM 快 859%），並解釋它為何對 >500 RPS 很重要。
- 在給定資料落地與維運預算下，選擇自架或託管。

## 問題 (The Problem)

你的產品呼叫 OpenAI、Anthropic 和自架的 Llama。每個供應商都有不同的 SDK、錯誤模型、rate limit 與驗證機制。你想要 failover（OpenAI 回 429 就試 Anthropic）、單一憑證儲存、統一的 observability，以及每個 tenant 的 rate limit。

在應用層重新發明這些，會讓每個服務與每個供應商耦合。Gateway 層把它整合進一個行程、一個 API（通常相容 OpenAI），再向外分發到各供應商。

## 核心概念 (The Concept)

### 七個核心功能

1. **供應商路由**：OpenAI、Anthropic、Gemini、自架等，都放在一個 API 後面。
2. **Fallback**：遇到 429、5xx 或品質失敗時，改到別處重試。
3. **Retries**：指數 backoff，有界的嘗試次數。
4. **Rate limits**：每 tenant、每 key、每模型。
5. **Secret 參照**：在 runtime 從 vault 拉取憑證（絕不放在 app 裡）。
6. **Observability**：OTel + GenAI 屬性 (Phase 17 · 13) + 成本 attribution。
7. **Guardrails**：PII redaction、jailbreak 偵測、允許主題的過濾器。

### LiteLLM：MIT OSS、Python

- 100+ 供應商、相容 OpenAI、router 設定、fallback、基本 observability。
- 在 Kong 的 benchmark 中約 2000 RPS 就撐不住；8 GB 記憶體占用，持續負載下連鎖失敗。
- 最適合：Python app、<500 RPS、dev/staging gateway、實驗性路由。
- 成本：OSS 為 $0；雲端有免費方案。

### Portkey：control plane 定位

- 自 2026 年 3 月起為 Apache 2.0 OSS。Guardrails、PII redaction、jailbreak 偵測、audit trails。
- 每個 request 的 latency 開銷為 20-40 ms。
- Production 方案每月 $49，含保留期與 SLA。
- 最適合：需要 guardrails + observability 打包的受監管產業。

### Kong AI Gateway：規模化方案

- 建立在 Kong Gateway 之上（成熟的 API gateway 產品，lua+OpenResty）。
- Kong 自己在相當於 12 個 CPU 的 benchmark：比 Portkey 快 228%，比 LiteLLM 快 859%。
- 定價：$100/model/month，Plus 方案最多 5 個。
- 最適合：已經在用 Kong；>1000 RPS；願意購買授權。

### Bifrost (Maxim AI)

- 自動 retries，backoff 可設定。
- OpenAI 429 時 fallback 到 Anthropic 是經典配方。
- 較新的參與者；商業產品。

### Cloudflare AI Gateway / Vercel AI Gateway

- 託管、零維運。基本的 retry 與 observability。
- 最適合：部署在 Cloudflare/Vercel 的 edge-serving JavaScript app。
- 在 guardrails 與 rate limits 上，比 Kong/Portkey 有限。

### 自架 vs 託管

資料落地是強制因素。醫療與金融預設自架（LiteLLM 或 Portkey OSS 或 Kong）。消費性產品預設託管 (Cloudflare AI Gateway) 或中間層 (Portkey 託管)。混合：受監管的 tenant 自架，其他託管。

### Latency 預算

- LiteLLM：典型開銷 5-15 ms。
- Portkey：開銷 20-40 ms。
- Kong：開銷 3-8 ms。
- Cloudflare/Vercel：開銷 1-3 ms（edge 優勢）。

Gateway latency 會直接加到 TTFT 上。TTFT P99 < 100 ms 的 SLA，用 Kong 或 Cloudflare。P99 < 500 ms，任何一個都行。

### Rate-limit 語意很重要

簡單的 token-bucket 在中等規模以內可行。多租戶需要 sliding-window + 突發額度 + 每 tenant 分級。LiteLLM 提供 token-bucket；Kong 提供 sliding-window；Portkey 提供分級。

### Gateway + observability + routing 會組合在一起

Phase 17 · 13（observability）+ 16（model routing）+ 19（gateways）在 production 中是同一層。挑一個涵蓋三者的工具，或小心地串接：2026 年多數部署把 Helicone（observability）或 Portkey（guardrails）與 Kong（規模）搭配，分擔不同角色。

### 該記住的數字

- LiteLLM：約 ~2000 RPS 撐不住，8 GB 記憶體。
- Portkey：開銷 20-40 ms；自 2026 年 3 月起 Apache 2.0。
- Kong：比 Portkey 快 228%，比 LiteLLM 快 859%。
- Kong 定價：$100/model/month，Plus 方案最多 5 個。
- Cloudflare/Vercel：edge 上開銷 1-3 ms。

```figure
mx-gateway-fallback
```

## 實際使用 (Use It)

`code/main.py` 在注入 429/5xx 的情況下，模擬跨 3 家供應商的 gateway 路由與 fallback。回報 latency、retry 率與 fallback 命中率。

## 交付 (Ship It)

這堂課產出 `outputs/skill-gateway-picker.md`。給定規模、維運立場、合規與 latency 預算，挑選 gateway。

## 練習 (Exercises)

1. 執行 `code/main.py`。設定 OpenAI→Anthropic→自架 的 fallback。在 5% 供應商錯誤率下，預期的命中率是多少？
2. 你的 SLA 是 TTFT P99 < 320 ms，而模型單獨在 P99 就量到 300 ms。哪些 gateway 能維持在預算內？
3. 一位醫療客戶要求自架 + PII redaction + audit。在 Portkey OSS 與 Kong 之間挑一個。
4. 比較 LiteLLM 與 Kong：團隊該在 RPS 上限多高時遷移？
5. 為多租戶 SaaS 設計 rate-limit 政策：免費方案、試用方案、付費方案。用 token-bucket 還是 sliding-window？

## 關鍵術語 (Key Terms)

| 術語 | 一般說法 | 實際意義 |
|------|----------------|------------------------|
| Gateway | 「API broker」 | 位於 app 與供應商之間的行程 |
| LiteLLM | 「MIT 那個」 | Python OSS、100+ 供應商、在 2K RPS 撐不住 |
| Portkey | 「guardrails gateway」 | Control plane + observability，Apache 2.0 |
| Kong AI Gateway | 「規模化那個」 | 建立在 Kong Gateway 之上，benchmark 領先 |
| Bifrost | 「Maxim 的 gateway」 | Retries + Anthropic fallback 配方 |
| Cloudflare AI Gateway | 「edge 託管」 | 部署在 edge 的託管 gateway，零維運 |
| PII redaction | 「資料清洗」 | 送到模型前，以 Regex + NER 遮罩 |
| Jailbreak detection | 「prompt injection 防護」 | 針對使用者輸入的分類器 |
| Audit trail | 「受監管 log」 | 每次 LLM 呼叫的不可變紀錄 |
| Token-bucket | 「簡單 rate limit」 | 以補充為基礎的 rate limiter |
| Sliding-window | 「精確 rate limit」 | 以時間視窗為基礎的 rate limiter；更公平 |

## 延伸閱讀 (Further Reading)

- [Kong AI Gateway Benchmark](https://konghq.com/blog/engineering/ai-gateway-benchmark-kong-ai-gateway-portkey-litellm)
- [TrueFoundry — AI Gateways 2026 Comparison](https://www.truefoundry.com/blog/a-definitive-guide-to-ai-gateways-in-2026-competitive-landscape-comparison)
- [Techsy — Top LLM Gateway Tools 2026](https://techsy.io/en/blog/best-llm-gateway-tools)
- [LiteLLM GitHub](https://github.com/BerriAI/litellm)
- [Portkey GitHub](https://github.com/Portkey-AI/gateway)
- [Kong AI Gateway docs](https://docs.konghq.com/gateway/latest/ai-gateway/)
