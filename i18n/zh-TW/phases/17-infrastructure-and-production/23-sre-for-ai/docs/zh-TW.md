# AI 時代的 SRE：多 Agent 事件應變、Runbooks、預測式偵測 (SRE for AI)

> AI SRE 使用以基礎設施資料（logs、runbooks、服務拓撲）透過 RAG 為基礎的 LLM，自動化調查、文件化與協調階段。2026 年的架構模式是多 agent 編排：由 supervisor 協調專門的 agents（logs、metrics、runbooks）；AI 提出假設與查詢，人類批准需要判斷的決定。Datadog Bits AI 和 Azure SRE Agent 以託管產品提供這種做法。Runbooks 也在演進：NeuBird Hawkeye 使用對抗式評估（兩個模型分析同一個事件；一致 = 信心，不一致 = 不確定性）；營運記憶在團隊人員異動後仍然保留。自動修復保持謹慎：AI 建議，人類批准。完全自主的行動範圍狹窄（重啟 pod、回滾特定部署），且有嚴格的 guardrails，任何兜售「設定好就不用管」的人都誇大了。新興前沿：事件發生前的預測。MIT 研究報告指出，以歷史 logs + GPU 溫度 + API 錯誤模式訓練的 LLM，提前 10-15 分鐘預測出 89% 的停機。預測：到 2026 年底，95% 的企業 LLM 會有自動化 failover。

**Type:** Learn
**Languages:** Python (stdlib, toy multi-agent incident triage simulator)
**Prerequisites:** Phase 17 · 13 (Observability), Phase 17 · 24 (Chaos Engineering)
**Time:** ~60 分鐘

## 學習目標 (Learning Objectives)

- 畫出多 agent AI SRE 架構：supervisor + 專門 agents（logs、metrics、runbooks）+ 人工批准關卡。
- 解釋為什麼自動修復的範圍狹窄（重啟 pod、還原部署），而不是廣泛（重新設計服務架構）。
- 說出對抗式評估模式 (NeuBird Hawkeye)：兩個模型一致 = 信心；不一致 = 升級。
- 引用 MIT 89% 提早偵測的結果，以及營運上的限制：沒有執行動作的預測只是儀表板。

## 問題 (The Problem)

凌晨 3 點，值班工程師被呼叫。「checkout 錯誤率偏高。」他們查看 Datadog、Loki、三份 runbook、部署記錄。30 分鐘後他們發現根本原因是 KV cache 暴增導致的 vLLM OOM。他們重啟 pod；錯誤消失。

在 2026 年，那場調查的前 20 分鐘是可以自動化的。依服務分組 logs、與最近的部署做關聯、比對 runbooks，全都是 RAG + tool-use。受監督的 agent 可以做第一輪分流，並在人類打開 Datadog 之前就提出假設。

完全自主的修復是另一個問題。重啟 pod：安全。擴充 GPU pool：政策允許時安全。重新設計服務架構：絕對不行。紀律在於劃出那條狹窄的線。

## 核心概念 (The Concept)

### 多 agent 架構

```
          Incident
             │
             ▼
        Supervisor
        /    |    \
       ▼     ▼     ▼
  Log agent  Metric agent  Runbook agent
       │     │     │
       └─────┴─────┘
             │
             ▼
        Hypothesis + evidence
             │
             ▼
        Human approval
             │
             ▼
        Action (narrow set)
```

Supervisor 把事件拆成子查詢。專門的 agents 擁有 tool 存取權（log 搜尋、PromQL、文件檢索）。Supervisor 綜合結果，把假設 + 證據呈現給人類。人類批准或重新導向。

### 自動修復的範圍

**安全（狹窄）**：重啟 pod、還原特定部署、在預先核准的界線內擴充 pool、啟用預先核准的 feature flag。

**不安全（廣泛）**：變更服務拓撲、修改資源限制、部署新程式碼、變更 IAM、修改資料庫。

任何兜售「設定好就不用管」的人都誇大了。隨著 AI SRE 成熟，安全的集合會擴大，但界線是真實存在的。

### 對抗式評估 (NeuBird Hawkeye)

兩個模型獨立分析同一個事件。如果它們對根本原因看法一致，信心就高。如果不一致，就升級給人類，並同時呈現兩個假設。簡單的模式，卻是過濾幻覺式根本原因的有效手段。

### 營運記憶

人員流動是傳統 SRE 的隱形殺手，部落知識會隨人離開。AI SRE 把 runbooks + 事後檢討存在向量 DB 中；agents 在每個新事件發生時檢索。新工程師加入時，AI 擁有完整的歷史。

### 事件發生前的預測

MIT 2025 研究：以歷史 logs、GPU 溫度、API 錯誤模式訓練的 LLM，在測試集上於停機發生前 10-15 分鐘預測出 89% 的停機。

現實檢查：沒有執行動作的預測只是儀表板。營運上的問題是「我們預測到的時候，要做什麼？」預先 drain？呼叫值班？自動擴充？答案取決於政策。

### 2026 年的產品

- **Datadog Bits AI**：Datadog 內的託管 SRE copilot。
- **Azure SRE Agent**：Azure 原生。
- **NeuBird Hawkeye**：對抗式 eval + 營運記憶。
- **PagerDuty AIOps**：分流 + 去重。
- **Incident.io Autopilot**：事件指揮官 + 協調。

### Runbooks 即程式碼

Runbooks 從 Confluence 頁面演進為帶有結構化章節（症狀、假設、驗證、行動）的版本化 markdown。結構化的 runbooks 能讓 RAG 檢索更好。任何 AI-SRE rollout 都從把非結構化 runbooks 轉成結構化開始。

### 該記住的數字

- MIT 提早偵測：89% 的停機，提前 10-15 分鐘。
- 多 agent 分流：supervisor + (logs, metrics, runbooks) + 人類。
- 安全的自動修復集合：重啟 pod、還原部署、在界線內擴充。
- 對抗式 eval：兩個模型獨立；一致 = 信心。

```figure
i4-incident-agents
```

## 實際使用 (Use It)

`code/main.py` 模擬多 agent 分流：log agent 找出錯誤，metric agent 找出 CPU 暴增，runbook agent 比對到已知問題。Supervisor 為假設排序。

## 交付 (Ship It)

這堂課產出 `outputs/skill-ai-sre-plan.md`。給定目前的值班狀況、事件量、團隊成熟度，設計 AI SRE rollout。

## 練習 (Exercises)

1. 執行 `code/main.py`。如果 log 和 metric agents 意見不一致，supervisor 如何解決？
2. 為你的服務定義三個「安全」的自動修復動作。為每個說明理由。
3. 寫一份結構化 runbook 模板：章節、必填欄位、驗證指令。
4. 預測式偵測在提前 12 分鐘觸發。你的政策是什麼：呼叫值班、預先 drain，還是兩者都做？
5. 辯論 3 人團隊在 2026 年應該採用 AI SRE，還是等待。考量成熟度、事件量、風險。

## 關鍵術語 (Key Terms)

| 術語 | 一般說法 | 實際意義 |
|------|----------------|------------------------|
| AI SRE | 「值班用的 agent」 | 以 LLM 為基礎的事件調查 + 協調 |
| Supervisor agent | 「編排者」 | 把事件拆成子查詢的頂層 agent |
| Specialized agent | 「領域 agent」 | 擁有 tool 存取權的子 agent（logs、metrics、runbooks） |
| Auto-remediation | 「AI 修好它」 | 狹窄、預先核准的動作；不是廣泛的重新架構 |
| Operational memory | 「向量 runbooks」 | 存在向量 DB 中供 RAG 使用的事後檢討 + runbooks |
| Adversarial eval | 「雙模型檢查」 | 獨立分析；一致 = 信心 |
| NeuBird Hawkeye | 「對抗式那個」 | 採用對抗式 eval + 記憶模式的產品 |
| Bits AI | 「Datadog 的 SRE agent」 | Datadog 託管的 AI SRE |
| Pre-incident prediction | 「提早偵測」 | 停機預測有 10-15 分鐘的領先時間 |

## 延伸閱讀 (Further Reading)

- [incident.io — AI SRE Complete Guide 2026](https://incident.io/blog/what-is-ai-sre-complete-guide-2026)
- [InfoQ — Human-Centred AI for SRE](https://www.infoq.com/news/2026/01/opsworker-ai-sre/)
- [DZone — AI in SRE 2026](https://dzone.com/articles/ai-in-sre-whats-actually-coming-in-2026)
- [Datadog Bits AI](https://www.datadoghq.com/product/bits-ai/)
- [NeuBird Hawkeye](https://www.neubird.ai/)
- [awesome-ai-sre](https://github.com/agamm/awesome-ai-sre)
