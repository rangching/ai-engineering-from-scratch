# LLM Production 的 Chaos Engineering (Chaos Engineering for LLM Production)

> 2026 年，LLM 的 chaos engineering 已是一門獨立的學問。在 production 執行實驗之前的先決條件：已定義的 SLI/SLO、trace+metric+log 的 observability、自動化 rollback、runbooks、on-call。架構有四個平面：control（實驗排程器）、target（服務、基礎設施、資料儲存）、safety（guards + abort + 流量過濾器）、observability（metrics + traces + logs），再加上 feedback（回饋到 SLO 調整）。Guardrails 是強制的：如果每日 error-budget 消耗 > 預期的 2x，burn-rate 告警會暫停實驗；抑制視窗 + trace-ID 關聯可以去除告警雜訊。節奏：每週小型 canary + SLO 檢視；每月 game day + 事後檢討；每季跨團隊韌性稽核 + 相依性對映。LLM 專屬實驗：記憶體過載、網路故障、供應商停機、格式錯誤的 prompt、KV cache 驅逐風暴。工具：Harness Chaos Engineering（LLM 衍生的建議、blast-radius 縮小、MCP tool 整合）；LitmusChaos (CNCF)；Chaos Mesh（CNCF Kubernetes 原生）。

**Type:** Learn
**Languages:** Python (stdlib, toy chaos experiment runner)
**Prerequisites:** Phase 17 · 23 (SRE for AI), Phase 17 · 13 (Observability)
**Time:** ~60 分鐘

## 學習目標 (Learning Objectives)

- 說出 chaos engineering 的五個先決條件（SLI/SLO、observability、rollback、runbooks、on-call），並解釋為什麼少了任何一個都會破壞這個實務。
- 畫出四個平面（control、target、safety、observability）以及回饋到 SLO 的迴圈。
- 列舉五個 LLM 專屬實驗（記憶體過載、網路故障、供應商停機、格式錯誤的 prompt、KV 驅逐風暴）。
- 依據 stack，在 Harness、LitmusChaos、Chaos Mesh 中挑選工具。

## 問題 (The Problem)

傳統 stack 的 chaos 測試已經成熟。LLM stack 則增加了新的失敗模式。一個帶有毒字元的 4K-token prompt，會讓 tokenizer 停滯 12 秒。上游供應商回 429；你的 gateway 重試；你的服務因 retry 放大的並行而 OOM。突發負載下的 KV cache 驅逐風暴，造成重新 prefill 的連鎖反應，讓運算飽和。

這些都不會出現在單元測試中。Chaos engineering 是讓你在使用者之前發現它們的方法。

## 核心概念 (The Concept)

### 先決條件

沒有以下條件，不要在 production 執行 chaos：

1. **SLI/SLO**：已定義的服務等級指標與目標。
2. **Observability**：traces、metrics、logs，並接到儀表板。
3. **自動化 rollback**：Phase 17 · 20 的政策旗標 rollback。
4. **Runbooks**：結構化，Phase 17 · 23。
5. **On-call**：有人可以回應。

少了任何一項，chaos 就會變成真正的事件。

### 四個平面 + 回饋

**Control plane**：實驗排程器（Litmus workflow、Chaos Mesh schedule、Harness UI）。

**Target plane**：服務、pods、節點、負載平衡器、資料儲存。

**Safety plane**：kill switch、抑制視窗、blast-radius 限制、error-budget 關卡。

**Observability plane**：一般 metrics + trace-ID 關聯，用以區分 chaos 引起的與自然發生的失敗。

**Feedback loop**：發現的問題回饋到 SLO 調整、runbook 更新、程式碼修正。

### Guardrails 是強制的

- **Burn-rate 告警**：如果每日 error-budget 消耗超過預期的 2x，就暫停實驗。
- **抑制視窗**：實驗期間，在 blast radius 內靜音非實驗的告警。
- **Trace-ID 關聯**：所有實驗引起的錯誤都帶有標記，讓 on-call 可以去重。

### 五個 LLM 專屬實驗

1. **記憶體過載**：以高並行送出長 context 的 request，強制引發 KV cache 搶佔風暴。觀察：服務是優雅地卸載，還是崩潰？

2. **網路故障**：切斷 inference gateway 與供應商之間的連線。觀察：fallback 是否在 SLA 內啟動？(Phase 17 · 19)

3. **供應商停機模擬**：來自 OpenAI 的 429 達 100%。觀察：路由是否 failover 到 Anthropic？(Phase 17 · 16, 19)

4. **格式錯誤的 prompt**：注入會讓 tokenizer 停滯的 payload（例如深度巢狀的 unicode、巨大的 UTF-8 codepoint）。觀察：單一 request 會不會鎖死一個 worker？

5. **KV 驅逐風暴**：透過讓 vLLM 的 block 預算飽和來強制驅逐。觀察：LMCache 能恢復，還是服務退化？

### 節奏

- **每週**：在 staging 做小型 canary 實驗，也許 5% 的 prod。
- **每月**：針對特定情境排程 game day；跨團隊參與；事後檢討。
- **每季**：跨團隊韌性稽核；更新相依性圖。

### 工具

- **Harness Chaos Engineering**：商業；AI 衍生的實驗建議；blast-radius 縮小；MCP tool 整合。
- **LitmusChaos**：CNCF graduated；以 Kubernetes workflow 為基礎。
- **Chaos Mesh**：CNCF sandbox；Kubernetes 原生的 CRD 風格。
- **Gremlin**：商業；支援廣泛。
- **AWS FIS** / **Azure Chaos Studio**：託管的雲端服務。

### 從小處開始

第一個實驗：在穩定流量下，pod-kill 一個 decode replica。觀察重新路由與恢復。如果這個有效且看起來安全，再進階到網路 chaos。

第一個 LLM 專屬實驗：注入某個供應商的 429 達 5 分鐘。觀察 fallback。多數團隊會發現他們的 fallback 並沒有被完整測試過。

### 該記住的數字

- 四個平面：control、target、safety、observability。
- Burn-rate 暫停：每日預算消耗為預期的 2x。
- 節奏：每週 canary、每月 game day、每季稽核。
- 五個 LLM 實驗：記憶體、網路、供應商、格式錯誤的 prompt、KV 風暴。

```figure
i4-chaos-guard
```

## 實際使用 (Use It)

`code/main.py` 模擬三個帶有 safety plane 關卡的 chaos 實驗。回報哪些實驗會觸發 burn-rate 中止。

## 交付 (Ship It)

這堂課產出 `outputs/skill-chaos-plan.md`。給定 stack 與成熟度，挑出前三個實驗與工具。

## 練習 (Exercises)

1. 執行 `code/main.py`。哪個實驗觸發了 burn-rate 關卡，為什麼？
2. 為以 vLLM 為基礎的 RAG 服務設計前五個 chaos 實驗。包含成功準則。
3. 你的 burn-rate 告警暫停了一個實驗。你如何判定根本原因是 chaos 還是自然發生？
4. 辯論 chaos 應該在 production 執行還是只在 staging。什麼時候 production 才是正確答案？
5. 說出三個一般網路 chaos 無法重現的 LLM 專屬失敗模式。

## 關鍵術語 (Key Terms)

| 術語 | 一般說法 | 實際意義 |
|------|----------------|------------------------|
| SLI / SLO | 「服務目標」 | 指標 + 目標；必要的先決條件 |
| Blast radius | 「範圍」 | 受實驗影響的服務 / 使用者集合 |
| Burn-rate alert | 「預算關卡」 | error-budget 消耗速率 > 預期 2x 時觸發 |
| Game day | 「每月演練」 | 排程的跨團隊 chaos 演習 |
| LitmusChaos | 「CNCF workflow」 | CNCF graduated 的 Kubernetes chaos 工具 |
| Chaos Mesh | 「CNCF CRD」 | CNCF sandbox 的 Kubernetes 原生 chaos |
| Harness CE | 「商業 AI 輔助」 | 附 AI 建議的 Harness chaos |
| Malformed prompt | 「tokenizer 炸彈」 | 會讓 tokenization 停滯的輸入 |
| KV eviction storm | 「搶佔連鎖」 | 大量驅逐引發重新 prefill |

## 延伸閱讀 (Further Reading)

- [DevSecOps School — Chaos Engineering 2026 Guide](https://devsecopsschool.com/blog/chaos-engineering/)
- [Ankush Sharma — Observability for Large Language Models (Apress, 2026)](https://link.springer.com/book/10.1007/979-8-8688-2827-0)
- [LitmusChaos (CNCF)](https://litmuschaos.io/)
- [Chaos Mesh (CNCF)](https://chaos-mesh.org/)
- [Harness Chaos Engineering](https://www.harness.io/products/chaos-engineering)
- [AWS FIS](https://aws.amazon.com/fis/)
