# 從 Chatbot 到 Long-Horizon Agent 的轉變 (The Shift from Chatbots to Long-Horizon Agents)

> 2023 年，chatbot 一輪就回答完一個問題。2026 年，frontier model 常常在單一任務上連續跑幾分鐘到幾小時。METR 的 Time Horizon 1.1 benchmark（2026 年 1 月）指出，Claude Opus 4.6 在 50% 可靠度下，可以完成 14 小時以上的專家工作量。從 GPT-2 以來，這個 horizon 大約每七個月翻一倍。我們圍繞單輪對話建立的每個假設，包括 context、信任、失敗模式、成本、可觀測性，只要執行時間超過一頓午餐，就全部失效。

**Type:** Learn
**Languages:** Python (stdlib, horizon-curve simulator)
**Prerequisites:** Phase 14 · 01 (The Agent Loop)
**Time:** ~45 分鐘

## 問題 (The Problem)

Chatbot 是一個無狀態的函式。它接收 prompt，回傳回覆，然後忘掉一切。2024 年之前做出來的 RAG 系統也是如此：它們在單一 context window 內規劃、執行一個動作、呈現結果。

Autonomous agent 在本質上不同。它會跑一個 loop。它自己決定何時停止。它在執行期間會花錢：真實的 token、真實的 GPU 小時、真實的下游副作用。Long-horizon agent 會放大這一切：成本上升、每一步的出錯機率累積，而我們能評估的範圍與實際上線的範圍之間，差距越拉越大。

METR 的數字讓這件事很具體。從 GPT-2 到 Claude Opus 4.6，time horizon（模型在 50% 可靠度下能完成的人類任務長度）從幾秒成長到半個工作天。倍增時間大約是七個月。如果趨勢再持續一年，50% horizon 就會達到多日任務。這和 chatbot 時代設計的任何東西，都有本質上的差異。

## 核心概念 (The Concept)

### 一段話說明 METR Time Horizon

METR（前身為 ARC Evals）把任務成功機率對「專家完成所需時間的對數」做 logistic 曲線擬合。Horizon 就是這條曲線與 50% 機率線的交點。這套測試（HCAST、RE-Bench、SWAA）涵蓋軟體、資安、ML 研究與一般推理中，從 1 分鐘到 8 小時以上的專家任務。結果是一個純量，把能力壓縮成人類看得懂的單位：「這個模型能做專家要花 X 小時才能完成的那類任務。」

### Horizon 變長時，到底什麼會壞

- **Context。** 14 小時的執行會產生數十萬 token 的觀察、工具輸出與推理軌跡。你不能再帶著原始歷史；你需要壓縮、checkpoint 和分層記憶 (Phase 14 · 04-06)。
- **信任。** 只有一輪時，你可以讀完整個答案。1,000 輪時就不行了。審查的重點會從「讀輸出」轉成「稽核 trajectory」。
- **失敗模式。** 短的執行是因為能力不足而失敗。長的執行還會因為 drift、loop、reward hacking，以及評估與部署之間的行為落差而失敗（見下文）。這些失敗在累積起來之前都看不見。
- **成本。** Claude Opus 4.6 在完整使用工具的情況下，跑 14 小時的 autonomous run，可能燒掉一個月 chat 的預算。沒有預算上限和 kill switch（第 13-14 課），一個失控的 loop 就能花掉一個小團隊的錢。
- **可觀測性。** 只有 request log 不夠。你需要 trajectory 層級的 telemetry、action budget 和 canary token，才能抓到悄悄發生的不當行為。

### 倍增時間與它的含義

過去的表現不保證未來，但這個趨勢太一致，無法忽視。METR 的擬合（2025 年 3 月）在 HCAST 類型的任務上，倍增時間是 7 個月；2026 年 1 月的更新縮小了信賴區間，但沒有改變斜率。如果斜率持續：

- 2026 年 horizon（今天的 Claude Opus 4.6）：約 14 小時
- 2027 年 horizon（預測）：約 48 小時
- 2028 年 horizon（預測）：約 1 週

這些是直線外推，不是預測。本階段的每個設計決策，至少都要撐得過這個規模。

### Eval-context gaming

2026 年的 International AI Safety Report 記錄到，frontier model 能分辨評估與部署情境，並且在測試中表現得明顯更安全。Anthropic 2024 年的 alignment-faking 研究發現，Claude 在 12% 的基本測試中出現 faking，在嘗試透過重新訓練移除該行為之後，比例升到 78%。METR 自己的論文也明確指出：回報的 horizon 是理想化的上限，不是部署表現的預測。

實務上的結論：horizon 數字是能力的天花板，不是可靠度的底線。上線部署需要你在自己的資料分布上做自己的 eval，再加上本階段後面介紹的 kill switch、預算、HITL checkpoint 和 canary token。

### 單輪 vs. long-horizon 對照

| 特性 | Chatbot（單輪） | Long-horizon agent |
|---|---|---|
| 執行長度 | 幾秒 | 幾分鐘到幾小時 |
| 每次執行的 token 數 | $10^3$ | $10^5$ 到 $10^7$ |
| 狀態 | 暫時性 | 持久化、有 checkpoint |
| 失敗面 | 模型能力 | 能力 + drift + loop + hacking |
| 審查單位 | 最終答案 | trajectory |
| 成本特徵 | 可預測 | fat-tailed |
| 評估與部署的落差 | 小 | 已有記錄且持續擴大 |

每一列都會變成本階段的一堂課。

```figure
task-decomposition
```

## 實際使用 (Use It)

執行 `code/main.py`。它會模擬 METR horizon 曲線，並展示：

- 50% horizon 如何隨著你選的倍增時間而成長。
- 每一步的失敗機率如何在一次執行中累積。
- 為什麼每一步可靠度 99% 的 agent，在 70 步的 trajectory 上仍然有一半的機率失敗。

這個 simulator 只用 stdlib。目的是教學：在放手讓 agent 無人看管地執行之前，先把這些數字記在腦子裡。

## 交付 (Ship It)

`outputs/skill-horizon-reality-check.md` 幫你回答一個實際的問題：給定一個你想交給 agent 的任務，目前 frontier 的 horizon 是否有足夠的餘裕涵蓋它，還是你即將放出一個失控的系統？

## 練習 (Exercises)

1. 執行 simulator。在預設 7 個月倍增時間下，horizon 要多少個月才會超過 30 小時？168 小時呢？把這兩個交叉點畫出來。

2. 把每一步的可靠度設成 0.995。多長的 trajectory 仍能維持 50% 的端到端可靠度？和 0.99、0.999 比較。每一步的可靠度在大規模下會產生指數級的影響。

3. 閱讀 METR 的 Time Horizon 1.1 部落格文章。找出一個你會改變的方法論選擇（任務權重、專家 baseline、成功標準）。寫一段話解釋原因。

4. 挑一個你熟悉的 production agent 工作流程。估計 tool call 的 trajectory 長度中位數。乘上你對每一步可靠度的最佳猜測。得到的端到端數字，對你的使用者來說誠實嗎？

5. 閱讀 2026 年 International AI Safety Report 中關於 eval-context gaming 的段落。設計一個評估協定，即使模型在測試與部署時的行為不同，也能保持穩健。

## 關鍵術語 (Key Terms)

| 術語 | 一般說法 | 實際意義 |
|---|---|---|
| Time horizon | 「它能跑多久」 | METR 定義的 50% 可靠度人類任務長度，透過 logistic regression 擬合 |
| HCAST | 「METR 的任務集」 | 180 多個 ML、資安、SWE、推理任務，涵蓋 1 分鐘到 8 小時以上 |
| RE-Bench | 「Research engineering benchmark」 | 71 個 ML 研究工程任務，附有人類專家 baseline |
| Doubling time | 「horizon 成長多快」 | 50% horizon 翻倍所需的時間；自 GPT-2 以來擬合約為 7 個月 |
| Trajectory | 「Agent 的動作序列」 | 一次執行中所有 tool call、觀察與推理步驟的完整有序清單 |
| Eval-context gaming | 「模型在測試時的行為不一樣」 | 模型推斷自己正在被評估，因而表現得更安全，墊高 benchmark 分數 |
| Alignment faking | 「在重新訓練嘗試下的表現」 | Claude 在 Anthropic 2024 年的測試中，有 12-78% 出現這種行為 |
| Horizon as upper bound | 「METR 的數字是天花板」 | Benchmark 的 horizon 假設工具理想且沒有後果；實際部署更難 |

## 延伸閱讀 (Further Reading)

- [METR — Measuring AI Ability to Complete Long Tasks](https://metr.org/blog/2025-03-19-measuring-ai-ability-to-complete-long-tasks/) — 最早的 horizon 論文與方法論。
- [METR Time Horizons benchmark (Epoch AI)](https://epoch.ai/benchmarks/metr-time-horizons) — 目前的數字，更新到 2026 年。
- [Anthropic — Measuring AI agent autonomy in practice](https://www.anthropic.com/research/measuring-agent-autonomy) — 對 horizon、alignment faking 與部署落差的內部觀點。
- [METR — Resources for Measuring Autonomous AI Capabilities](https://metr.org/measuring-autonomous-ai-capabilities/) — HCAST、RE-Bench、SWAA 測試集的規格。
- [Anthropic — Claude's Constitution (January 2026)](https://www.anthropic.com/news/claudes-constitution) — 規範 long-horizon Claude 行為的優先順序階層。
