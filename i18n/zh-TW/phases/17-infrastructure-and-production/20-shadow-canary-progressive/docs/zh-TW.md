# LLM 的 Shadow Traffic、Canary Rollout 與漸進式部署 (Shadow Traffic, Canary Rollout, and Progressive Deployment for LLMs)

> LLM rollout 結合了軟體部署中最困難的部分：沒有單元測試、失敗模式分散、訊號延遲。順序是：(1) shadow mode，把 prod request 複製給候選模型、記錄、比較，對使用者零影響；能抓到明顯的分布問題，但不是品質保證；(2) canary rollout，漸進式流量轉移 10% → 25% → 50% → 75% → 100%，每一步都有關卡；追蹤 latency 百分位數、每個 request 的成本、錯誤/拒絕率、輸出長度分布、使用者回饋率；(3) 確認穩定後，對差異明顯的替代方案做 A/B 測試。非決定性無法消除：由於 GPU 浮點不具結合律，加上 batch-size 變異，即使輸入完全相同，多次執行之間的準確度最多可差 15%。成本是變數，不是常數：好 20% 的模型每次呼叫可能貴 3x。Rollback 速度是決定性的：如果 rollback 需要重新部署，你就太慢了。政策存在 config/flags 裡；模型存在有固定 digest 的 registry 中；rollback = 翻轉政策 + 還原門檻 + 在幾秒內固定舊模型。

**Type:** Learn
**Languages:** Python (stdlib, toy canary-progression simulator)
**Prerequisites:** Phase 17 · 13 (Observability), Phase 17 · 21 (A/B Testing)
**Time:** ~60 分鐘

## 學習目標 (Learning Objectives)

- 區分 shadow mode（零影響比較）、canary（漸進式即時流量）與 A/B（穩定後的比較）。
- 列舉五個 LLM 專屬的 canary 指標（latency、每個 request 成本、錯誤/拒絕、輸出長度分布、使用者回饋）。
- 解釋為什麼 LLM 的非決定性（最高 15%）會改變 rollout 中「穩定」的意義。
- 設計只需幾秒（翻轉政策）而不是幾小時（重新部署）的 rollback 路徑。

## 問題 (The Problem)

你交付了新模型。離線 eval 顯示準確度提升 3%。你在 production 打開它。24 小時內，成本上升 40%，使用者倒讚上升 8%，三張客戶工單回報「奇怪的回答」。你 rollback。重新部署要 3 小時。你的週末毀了。

這一切本來都可以避免。Shadow mode 能在任何使用者看到之前就抓到那個 40% 的成本暴增。Canary 會在倒讚上升時停在 10%。用政策旗標 rollback 只要 30 秒。這套紀律填補的，就是「離線 eval 看起來很好」和「真實使用者滿意」之間的落差。

## 核心概念 (The Concept)

### Shadow mode

候選模型接收與 production 相同的 request；輸出被記錄，但不回傳給使用者。對使用者零影響。記錄：

- 輸出內容（與 production 做 diff）。
- Token 數量（成本差異）。
- Latency。
- 拒絕與錯誤。

能抓到：成本暴增、長度退化、明顯的拒絕變化、硬性錯誤。抓不到：使用者會感受到的品質差異。Shadow 是 smoke test，不是品質測試。

### Canary rollout

帶關卡的漸進式流量轉移。典型進程：1% → 10% → 25% → 50% → 75% → 100%。每一步以 5 個指標把關：

1. **Latency 百分位數**：P50、P95、P99。違反：canary 的 P99 > 基準的 1.5x。
2. **每個 request 的成本**：混合 $。違反：高於基準 >20%。
3. **錯誤 / 拒絕率**：5xx 加明確拒絕。違反：基準的 2x。
4. **輸出長度分布**：平均 + P99。違反：分布位移。
5. **使用者回饋率**：倒讚 / 工單提交。違反：基準的 1.5x。

### 非決定性是新的變異

相同的輸入會產生不同的輸出。原因：

- GPU 浮點不具結合律（浮點 reduction 順序會隨 batch 而變）。
- Batch-size 變異（同一個 prompt 在 128 的 batch 與 16 的 batch 中）。
- Sampling（temperature > 0）。

實測：在相同的 eval 集上，逐次執行的準確度變異最高可達 15%。Rollout 中的「穩定」意味著指標落在預期變異之內，而不是與基準完全相同。請把關卡設在雜訊底線之上。

### 成本是變數

好 20% 的模型每次呼叫可能貴 3x。每個 request 的成本是五個關卡之一。上線一個破壞單位經濟學的「更好」模型，就是 rollback 的案例。

### Rollback 是武器

- 政策旗標（feature flag 系統）：在 config 中翻轉百分比；只需幾秒。
- 模型固定（registry digest）：固定的模型不會自動升級。
- Rollback = 還原旗標 + 把固定的 digest 設回前一個。幾秒，不是幾小時。

如果你的 stack 需要重新部署才能 rollback，請在 rollout 之前先修好。

### 工具

**Argo Rollouts** / **Flagger**：Kubernetes 漸進式交付控制器。與 Istio/Linkerd 的加權路由整合。

**Istio weighted routing**：service-mesh 層級的流量切分。

**KServe / Seldon Core**：內建 canary 的模型 serving。

**Feature flags**：LaunchDarkly、Flagsmith、Unleash。政策層級的翻轉，不需重新部署。

### 指標節奏

Canary 關卡每 5-15 分鐘檢查一次，依流量而定。1000 req/min 的 1%（canary 每分鐘 10 個 request），每個 5-15 分鐘視窗內有 50-150 個資料點，對 latency 夠用，但對使用者回饋太吵。10% 則多約 10x。進程應該在每一步停留夠久，累積足夠的樣本。

### A/B 步驟是選用的

如果新模型明顯不同（不同的行為、不同的成本曲線、不同的語氣），在 canary 通過後以 50% 做 A/B 測試。如果它只是同一種行為的改良版，canary 關卡通過後就直接跳到 100%。所謂改良，是指離線 eval 顯示了增益。Canary 關卡並不衡量任務完成度，所以跳過 A/B 就是對這點信任 eval。

### 該記住的數字

- Canary 進程：1% → 10% → 25% → 50% → 75% → 100%。
- 非決定性上限：相同輸入下，逐次執行變異最高 15%。
- 五個 canary 指標：latency、成本、錯誤/拒絕、輸出長度、使用者回饋。
- 成本關卡：高於基準 >20% 即為違反。
- Rollback：幾秒，不是幾小時。

```figure
i4-canary-ramp
```

## 實際使用 (Use It)

`code/main.py` 模擬注入了退化的 canary rollout。回報 rollout 停在哪個階段，以及觸發了哪個關卡。

## 交付 (Ship It)

這堂課產出 `outputs/skill-rollout-runbook.md`。給定候選模型、基準與風險容忍度，設計 shadow→canary→100% 計畫。

## 練習 (Exercises)

1. 執行 `code/main.py`。注入 25% 的成本退化。Canary 在哪個階段停止？
2. 你的新模型離線準確度提升 3%，但每個 request 成本 +18%。要上線嗎？這取決於政策，請寫出兩條路徑。
3. 設計一個端到端不到 60 秒的 rollback。列出所需的基礎設施。
4. 你的 eval 顯示非決定性為 ±7%。設定 canary 關卡，避免誤報。你用什麼倍數？
5. Shadow mode 在 canary 之前就抓到 40% 的成本暴增。寫出在 shadow 中觸發的告警規則。

## 關鍵術語 (Key Terms)

| 術語 | 一般說法 | 實際意義 |
|------|----------------|------------------------|
| Shadow mode | 「複製給新的」 | 為了記錄而零影響地送給候選模型 |
| Canary | 「漸進式流量」 | 帶關卡、逐步讓使用者暴露的 rollout |
| Gates | 「rollout 檢查」 | 阻擋進程的指標門檻 |
| Non-determinism | 「LLM 變異」 | 無法消除的逐次執行差異 |
| Policy flag | 「旗標翻轉 rollback」 | config 層級的 rollback，幾秒而非幾小時 |
| Model pin | 「registry digest」 | 對模型版本的不可變參照 |
| Argo Rollouts | 「K8s 漸進式」 | Kubernetes 原生的 canary/rollback 控制器 |
| KServe | 「K8s inference」 | 具備 canary 基本元件的模型 serving |
| Istio weighted | 「mesh 切分」 | Service-mesh 流量切分器 |

## 延伸閱讀 (Further Reading)

- [TianPan — Releasing AI Features Without Breaking Production](https://tianpan.co/blog/2026-04-09-llm-gradual-rollout-shadow-canary-ab-testing)
- [MarkTechPost — Safely Deploying ML Models](https://www.marktechpost.com/2026/03/21/safely-deploying-ml-models-to-production-four-controlled-strategies-a-b-canary-interleaved-shadow-testing/)
- [APXML — Advanced LLM Deployment Patterns](https://apxml.com/courses/mlops-for-large-models-llmops/chapter-4-llm-deployment-serving-optimization/advanced-llm-deployment-patterns)
- [Argo Rollouts docs](https://argo-rollouts.readthedocs.io/)
- [Flagger docs](https://docs.flagger.app/)
