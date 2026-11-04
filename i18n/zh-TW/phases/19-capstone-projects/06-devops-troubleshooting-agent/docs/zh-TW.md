# Capstone 06 — Kubernetes 的 DevOps 疑難排解 Agent (DevOps Troubleshooting Agent)

> AWS 的 DevOps Agent 正式上市 (GA)，Resolve AI 發佈了它的 K8s playbook，NeuBird 展示了語意監控，Metoro 把 AI SRE 綁到每個服務的 SLO。正式環境的樣貌已經底定：警報 webhook 觸發，agent 讀取遙測資料，走訪 K8s 物件的 graph，為根因假說排序，並在 Slack 貼出附有核准按鈕的簡報。預設唯讀。每個修復動作都要由人核准。這個 capstone 就是那個 agent，用 20 個合成事件評測，並在三個共同案例上與 AWS 的 Agent 比較。

**Type:** Capstone
**Languages:** Python (agent), TypeScript (Slack integration)
**Prerequisites:** Phase 11 (LLM engineering), Phase 13 (tools and MCP), Phase 14 (agents), Phase 15 (autonomous), Phase 17 (infrastructure), Phase 18 (safety)
**Phases exercised:** P11 · P13 · P14 · P15 · P17 · P18
**Time:** 30 小時

## 問題 (Problem)

2025-2026 年 SRE 的主流說法變成：「AI agent 負責事件分流，人類核准修復。」AWS DevOps Agent、Resolve AI、NeuBird、Metoro、PagerDuty AIOps 都在正式環境提供這種形態。Agent 讀取 Prometheus 指標、Loki 日誌、Tempo trace、kube-state-metrics，以及 K8s 物件的知識圖譜。它在五分鐘內產出附有遙測引用的根因假說排名。沒有經過 Slack 的明確人工核准，它絕不執行破壞性指令。

困難的工作多半在界定範圍與安全，而不是推理。Agent 需要預設唯讀的 RBAC 介面、強化過的 MCP 工具伺服器，以及每個「考慮過」與「執行過」指令的稽核日誌。它需要知道自己何時超出能力範圍並上報。而且它的執行成本要夠低，才不會因為 OOM-kill 連鎖反應而產生 $5k 的 agent 帳單。

## 核心概念 (Concept)

Agent 在知識圖譜上運作。節點是 K8s 物件 (Pods、Deployments、Services、Nodes、HPAs、PVCs) 加上遙測來源 (Prometheus series、Loki streams、Tempo traces)。邊編碼擁有關係 (Pod -> ReplicaSet -> Deployment)、排程 (Pod -> Node) 與觀測 (Pod -> Prometheus series)。圖譜透過 kube-state-metrics 同步保持更新，並在每次警報時重新取樣。

警報觸發時，agent 從受影響的物件開始找根因。它沿著邊走訪，取出相關的遙測切片 (最近 15 分鐘)，並草擬假說。假說依證據排序：有多少遙測引用支持、多近期、多具體。前三名假說連同 graph 路徑視覺化和修復動作的核准按鈕，一併送到 Slack。

修復有關卡。預設允許的動作都是唯讀。破壞性動作 (縮減規模、回滾、刪除 Pod) 需要 Slack 核准；ArgoCD 回滾 hook 需要 agent 永遠不持有的 auth token。稽核日誌記錄 agent *考慮過* 的每個指令，而不只是執行過的，讓審查流程能抓到險些出事的狀況。

## 架構 (Architecture)

```
PagerDuty / Alertmanager webhook
           |
           v
     FastAPI receiver
           |
           v
   LangGraph root-cause agent
           |
           +---- read-only MCP tools ----+
           |                             |
           v                             v
   K8s knowledge graph              telemetry slices
     (Neo4j / kuzu)              Prometheus, Loki, Tempo
   ownership + scheduling          last 15m, scoped
           |
           v
   hypothesis ranking (evidence weight)
           |
           v
   Slack brief + approval buttons
           |
           v (approved)
   ArgoCD rollback hook / PagerDuty escalate
           |
           v
   audit log: considered vs executed, every command
```

## 技術堆疊 (Stack)

- 可觀測性來源：Prometheus、Loki、Tempo、kube-state-metrics
- 知識圖譜：Neo4j (託管) 或 kuzu (內嵌)，存放 K8s 物件 + 遙測邊
- Agent：LangGraph，每個工具有 allow-list，預設唯讀
- 工具傳輸：透過 StreamableHTTP 的 FastMCP；破壞性工具使用獨立伺服器，並置於核准關卡之後
- 模型：根因推理用 Claude Sonnet 4.7，日誌摘要用 Gemini 2.5 Flash
- 修復：ArgoCD 回滾 webhook、PagerDuty 上報、Slack 核准卡片
- 稽核：僅追加的結構化日誌 (considered、executed、approved、outcome)
- 部署：K8s deployment，使用自己的窄範圍 RBAC 角色，並放在獨立 namespace

```figure
ce-rootcause-walk
```

## 動手做 (Build It)

1. **Graph 匯入。** 每 30 秒把 kube-state-metrics 同步到 Neo4j/kuzu。節點：Pod、Deployment、Node、Service、PVC、HPA。邊：OWNED_BY、SCHEDULED_ON、EXPOSES、MOUNTS、SCALES。遙測疊加邊：OBSERVED_BY (Pod 被一條 Prometheus series 觀測)。

2. **警報接收器。** 一個接受 PagerDuty 或 Alertmanager webhook 的 FastAPI endpoint。擷取受影響的物件與 SLO 違反情形。

3. **唯讀工具介面。** 透過 FastMCP 包裝 kubectl、Prometheus query、Loki logql、Tempo traceql。每個工具都只有狹窄的 RBAC 動詞 (「get」、「list」、「describe」)。預設伺服器中沒有「delete」、「exec」、「scale」。

4. **根因 agent。** 三個節點的 LangGraph：`sample` 取出最近 15 分鐘的遙測切片、`walk` 查詢 graph 取得相鄰物件、`hypothesize` 草擬附有遙測引用、經過排序的根因候選。

5. **證據評分。** 每個假說的分數 = recency * specificity * graph 路徑長度倒數 * 引用數量。回傳前三名。

6. **Slack 簡報。** 貼出附件，內含假說、graph 路徑視覺化 (伺服器端渲染的子圖圖片)，以及最多一個修復動作的核准按鈕。

7. **修復關卡。** 破壞性工具 (縮減規模、回滾、刪除) 放在第二個 MCP 伺服器，並置於核准 token 之後。Agent 只有在 Slack 卡片被人核准後才能呼叫。

8. **稽核日誌。** 僅追加的 JSONL：對每個候選指令，記錄它是否被考慮、是否被執行、由誰核准。每天送到 S3。

9. **合成事件套件。** 建立 20 個情境：OOMKill 連鎖反應、DNS 抖動、HPA 來回震盪、PVC 滿載、吵鬧的鄰居、有問題的 sidecar、錯誤的 ConfigMap 發佈、憑證輪替、image-pull backoff 等等。依根因準確度與找出假說的時間，為 agent 評分。

## 實際使用 (Use It)

```
webhook: alert.pagerduty.com -> checkout-api SLO breach, error rate 14%
[graph]   affected: Deployment checkout-api (3 Pods, Node ip-10-2-3-4)
[walk]    neighbors: ReplicaSet checkout-api-abc, Service checkout-api,
           recent rollout 14m ago
[sample]  prometheus error_rate 14%, up-trend; loki 500s on /api/v2/pay
[hypo]    #1 bad rollout: latest image checkout-api:v2.41 fails /healthz
          citations: deploy.yaml (rev 42), prometheus errorRate, loki 500 stack
[slack]   [ROLL BACK to v2.40]  [ESCALATE]  [IGNORE]
          (approval required; agent does not roll back unilaterally)
```

## 交付 (Ship It)

`outputs/skill-devops-agent.md` 是交付成果。給定 K8s 叢集與警報來源，agent 會產出排序過的根因假說，以及由 Slack 把關的修復流程。

| 權重 | 評分項目 | 衡量方式 |
|:-:|---|---|
| 25 | 情境套件上的 RCA 準確度 | 20 個合成事件中，根因正確率 ≥80% |
| 20 | 安全性 | 稽核日誌顯示，破壞性動作防護在沒有 Slack 核准時從未觸發 |
| 20 | 找出假說的時間 | 從警報到 Slack 簡報的 p50 低於 5 分鐘 |
| 20 | 可解釋性 | 每個假說都附有 graph 路徑與遙測引用 |
| 15 | 整合完整度 | PagerDuty、Slack、ArgoCD、Prometheus 端到端運作 |
| **100** | | |

## 練習 (Exercises)

1. 讓你的 agent 處理 AWS 的 DevOps Agent 所展示的那三個相同事件。發佈並排比較，回報 agent 在哪裡出現分歧。

2. 加入「險些出事」稽核，標記 agent *考慮過*、且在沒有核准時會造成破壞的任何指令。衡量一週內的險些出事比率。

3. 把假說模型從 Claude Sonnet 4.7 換成自行架設的 Llama 3.3 70B。衡量 RCA 準確度差異與每個事件的美元成本。

4. 建立因果過濾器：區分相關的遙測尖峰與真正的根因。用 20 個情境的標籤訓練一個小型分類器。

5. 加入回滾 dry-run：用相同的 manifest 對 staging 叢集執行 ArgoCD 回滾。在 Slack 核准按鈕出現之前，先在真實叢集中驗證回滾計畫。

## 關鍵術語 (Key Terms)

| 術語 | 一般說法 | 實際意義 |
|------|-----------------|------------------------|
| K8s knowledge graph | 「叢集 graph」 | 節點 = K8s 物件 + 遙測 series；邊 = 擁有、排程、觀測關係 |
| Read-only-by-default | 「限縮範圍的 RBAC」 | Agent 的 service account 只有 get/list/describe 動詞；破壞性動詞放在核准關卡後的獨立伺服器 |
| Audit log | 「考慮過 vs 執行過」 | 僅追加的紀錄，包含每個候選指令、是否執行、由誰核准 |
| Hypothesis ranking | 「證據分數」 | Recency × specificity × graph 路徑長度倒數 × 引用數量 |
| Slack approval card | 「HITL 關卡」 | 附有修復按鈕的互動式 Slack 訊息；人類點擊之前 agent 不能繼續 |
| Telemetry citation | 「證據指標」 | 支持某個主張的 Prometheus query、Loki selector 或 Tempo trace URL |
| MTTR | 「解決所需時間」 | 從警報觸發到 SLO 恢復的實際經過時間 |

## 延伸閱讀 (Further Reading)

- [AWS DevOps Agent GA](https://aws.amazon.com/blogs/mt/announcing-general-availability-of-aws-devops-agent/)：2026 年的標準參考
- [Resolve AI K8s troubleshooting](https://resolve.ai/blog/kubernetes-troubleshooting-in-resolve-ai) — 競爭者的參考
- [NeuBird semantic monitoring](https://www.neubird.ai) — 語意 graph 做法
- [Metoro AI SRE](https://metoro.io) — 以 SLO 為先的正式環境框架
- [kube-state-metrics](https://github.com/kubernetes/kube-state-metrics) — 叢集狀態來源
- [LangGraph](https://langchain-ai.github.io/langgraph/) — 參考用的 agent 編排器
- [FastMCP](https://github.com/jlowin/fastmcp) — Python MCP 伺服器框架
- [ArgoCD rollback](https://argo-cd.readthedocs.io/en/stable/user-guide/commands/argocd_app_rollback/) — 受關卡把關的修復目標
