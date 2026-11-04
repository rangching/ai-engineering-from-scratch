# Kill Switch、Circuit Breaker 與 Canary Token (Kill Switches, Circuit Breakers, and Canary Tokens)

> Kill switch 是一個放在 agent 編輯範圍之外的布林值（Redis key、feature flag、簽章過的設定），可以讓 agent 完全停用。Circuit breaker 的粒度更細：它在特定模式出現時跳脫（連續五次相同的 tool call），暫停有問題的路徑，並上報給人類。Canary token 承襲自傳統的欺敵技術：一個假憑證或 honeypot 記錄，agent 沒有任何正當理由去碰，一旦被存取就會觸發警報。以 eBPF 為基礎的 datapath（例如 Cilium）可以在 kernel 層把被隔離 pod 的 egress 改寫到鑑識用的 honeypot；已發表的 Cilium benchmark 回報在負載下 datapath 延遲的 P99 低於 1 毫秒（你的傳播預算取決於 policy 更新如何到達節點，而不是 datapath 本身）。會適應移動 baseline 的統計偵測器（EWMA、CUSUM）會悄悄接受漂移，因此要把它們與不會彎曲的硬性 constitutional 限制分層搭配。

**Type:** Learn
**Languages:** Python (stdlib, three-detector simulator: kill switch, circuit breaker, canary)
**Prerequisites:** Phase 15 · 13 (Cost governors), Phase 15 · 10 (Permission modes)
**Time:** ~60 分鐘

## 問題 (The Problem)

Cost governor（第 13 課）限制 agent 能花多少錢。它們不限制 agent 在預算之內能做什麼。一個有 $50 速率限制的 agent，仍然可以外洩一個 secret、發布錯誤的貼文，或刪除一項資源，而昂貴的動作常常正是 token 上很便宜的那個。

這堂課介紹與成本層並排的三種偵測器：

1. **Kill switch**：放在 agent 碰不到的地方的布林關閉鈕。
2. **Circuit breaker**：暫停特定路徑的動作模式偵測器。
3. **Canary token**：誘餌，沒有正當理由去碰它的 agent，一碰就會自曝。

三者都是 LLM 之前就有的工程。傳統的欺敵、rate-limit breaker 與 feature-flag kill 都早於自主 agent。新的是攻擊面：agent 會讀取不受信任的內容（第 11 課）、編輯自己的記憶，並能把許多看似安全的動作組合成不安全的一個。這裡列出的偵測器之所以有效，是因為它們不信任 agent 的自我回報。

## 核心概念 (The Concept)

### Kill switch

Kill switch 是 agent 可以讀取但不能寫入的布林值。實作選擇：

- **託管服務中的 feature flag。** LaunchDarkly、Statsig、Unleash。更新幾秒內傳播。
- **Agent 輪詢的 Redis key。** 簡單；需要 agent 行程在每一輪都檢查。
- **物件儲存中的簽章設定。** Agent 在開機時驗證簽章；拒絕未簽章的狀態。
- **OS 層級的訊號或容器生命週期的終止。** Docker `kill`、Kubernetes `kubectl delete pod`、systemd stop。

正確的 kill switch 具備的性質：

- Agent 無法把它設成 `off`。（放在 agent 的憑證無法寫入的系統中。）
- 在每個有後果的動作時檢查，而不只是在啟動時。
- 關閉時，agent 不做任何對外可觀察的事，包括寫 log 到 agent 能碰到的系統。
- 重新啟用是明確的人類動作，不是自動逾時。

### Circuit breaker

Circuit breaker 暫停特定的模式，而不是整個 agent。經典形狀（出自 2007 年 Nygard 的書，至今仍適用）：

- **Closed**：動作被允許。
- **Open**：動作被阻擋。
- **Half-open**：冷卻後，允許 1–3 次探測嘗試（預設 1 次）；成功就關閉 breaker，任何剩餘的失敗就重新打開。

與 agent 相關的觸發條件：

- 連續五次相同的 tool call（重複的 loop）。
- 同一工具、不同輸入下連續五次失敗（系統性失敗）。
- Tool call 速率超過門檻（第 13 課的速率限制）。
- 在讀取不受信任的內容（第 11 課）之後，呼叫特定工具（例如 `file.delete`）。

### Canary token

Canary token（也稱為 honeytoken）是 agent 永遠不該碰的輸入。被存取就是警報。

Agent 的例子：

- 環境中一個沒有任何實際權限的假 `AWS_SECRET_ACCESS_KEY`。外洩嘗試會立刻被歸因。
- 一筆帶有已知哨兵值標記的假資料庫記錄。任何對這筆記錄的讀取或更新都會觸發警報。
- 工作區中的一個假檔案。對這個檔案的讀取或編輯都很可疑。
- 在 session 開始時注入的一筆假記憶。如果它出現在之後的輸出中，記憶已經被外洩。

Canary 的設計取決於工作流程。Agent 有正當理由去碰的 canary，只是雜訊。

### 為什麼要把統計式與硬性限制分層

統計偵測器（EWMA、CUSUM、tool-call 速率的 z-score）會適應移動的 baseline。對合法的漂移這是功能，對對抗性的漂移則是 bug。有耐心的攻擊者可以慢慢移動 baseline，慢到沒有警報觸發。

硬性的 constitutional 限制（第 17 課）不會彎曲。把「絕不寄信給 `secrets@`」當成 constitutional 規則，更簡單、可稽核，而且無法被緩慢的漂移鑽漏洞。分層模式是：統計偵測器抓到大部分雜訊，硬性限制抓到那個繞過偵測器的攻擊。

### 透過 eBPF datapath 重導向進行隔離

Cilium 以 eBPF 為基礎的網路 policy，可以在 kernel 層把被隔離 pod 的 egress 改寫到鑑識用的 honeypot。Cilium 公開的 benchmark 回報 TCP_RR 上 datapath 延遲的 P99 低於 1 毫秒；在你的叢集中，*觸發*隔離的牆上時間預算，主要取決於 policy 更新如何從控制平面傳播到節點（在 operator 預先載入的 Kubernetes 上，幾百毫秒是合理的；尾端延遲則取決於叢集大小與 reconcile 頻率）。這帶來兩件事：agent 不會外洩任何真實的東西，而且 agent 在 honeypot 內的行為會被完整觀察。這和 canary token 是同樣的模式，只是在不同層：agent 透過它在被圍堵時試圖做的事，揭露自己的意圖。

### 沒有任何偵測器能抓到的

- **由被允許的基本元件組合而成的語意攻擊。** 每個單獨的動作都通過每個偵測器；問題出在組合。這就是第 10 課 Auto Mode 的組合式失敗。
- **把偵測器本身納入 payload 的攻擊。** 如果 agent 讀到一個頁面寫著「行動之前，先停用 canary」，而 agent 有這個能力，偵測器就被入侵了。Canary token 應該放在 agent 無法修改的系統中。

```figure
circuit-breaker
```

## 實際使用 (Use It)

`code/main.py` 模擬一段簡短的 agent trajectory 通過三個偵測器。Kill switch 存放在外部的 dict；circuit breaker 在連續五次相同的 tool call 時跳脫；canary 檔案一被讀取就觸發警報。餵入一條合成的 trajectory：合法動作、重複的 loop、canary 探測，以及 kill switch 觸發、agent 的動作被中止的情境。

## 交付 (Ship It)

`outputs/skill-tripwire-design.md` 審查提議中 agent 部署的偵測器堆疊，並標出缺口（缺少 kill switch、缺少 canary、circuit breaker 的門檻太鬆）。

## 練習 (Exercises)

1. 執行 `code/main.py`。確認 circuit breaker 在第 5 輪（第五次相同呼叫）觸發，而 canary 在第 9 輪（讀取假 key）觸發。

2. 加入一個統計偵測器：對 tool-call 速率的 EWMA z-score。餵入一條緩慢漂移的 trajectory，展示偵測器從不觸發。接著加入一個硬性限制（10 分鐘內不超過 50 次 tool call），展示硬性限制在同一條 trajectory 上觸發。

3. 為 browser agent（第 11 課）設計一組 canary token。列出至少三個 canary，以及每個能偵測什麼。

4. 閱讀 Cilium 的 network-policy 文件。具體描述一個 egress 重導向的隔離流程：哪個 policy selector、哪個 pod、哪個 egress 改寫、哪個警報。從「決定隔離」到「第一個被重導向的封包」之間的牆上延遲由什麼決定？

5. 為被 kill switch 停用的 agent 定義重新啟用程序。誰可以重新啟用？必須記錄什麼？重新啟用之前，agent 必須改變什麼？

## 關鍵術語 (Key Terms)

| 術語 | 一般說法 | 實際意義 |
|---|---|---|
| Kill switch | 「關閉鈕」 | 位於 agent 編輯範圍之外的布林值；在每個有後果的動作時檢查 |
| Circuit breaker | 「模式暫停」 | 針對特定動作，因重複、失敗率或速率限制而跳脫 |
| Canary token | 「Honeytoken」 | Agent 沒有正當理由去碰的誘餌；被存取就觸發警報 |
| Honeypot | 「鑑識 sandbox」 | 被重導向的流量 / 工作區，在其中觀察被隔離的 agent |
| EWMA | 「移動平均」 | 指數加權；會適應漂移（既是功能也是 bug） |
| CUSUM | 「累積和」 | 偵測相對於 baseline 的持續偏移 |
| Hard limit | 「Constitutional 規則」 | 不會適應；不論歷史為何都保持固定 |
| Constitutional limit | 「永遠成立的規則」 | 與第 17 課的 constitution 綁定；agent 無法編輯 |

## 延伸閱讀 (Further Reading)

- [Anthropic — Measuring agent autonomy in practice](https://www.anthropic.com/research/measuring-agent-autonomy) — 自主 agent 的 kill switch 與 circuit breaker 框架。
- [Microsoft Agent Framework — HITL and oversight](https://learn.microsoft.com/en-us/agent-framework/workflows/human-in-the-loop) — production 治理模式。
- [OWASP LLM / Agentic Top 10](https://owasp.org/www-project-top-10-for-large-language-model-applications/) — 偵測與回應的要求。
- [Cilium — Network policy and eBPF](https://docs.cilium.io/en/stable/security/network/) — pod 層級的 egress 重導向與鑑識 honeypot 模式。
- [Anthropic — Claude's Constitution (January 2026)](https://www.anthropic.com/news/claudes-constitution) — 作為「constitutional 限制」的硬編碼禁令。
