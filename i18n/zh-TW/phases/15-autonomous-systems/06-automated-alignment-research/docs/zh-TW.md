# 自動化 Alignment 研究 (Automated Alignment Research, Anthropic AAR)

> Anthropic 讓多組 Claude Opus 4.6 Autonomous Alignment Researcher 在各自獨立的 sandbox 中平行執行，並透過一個共用論壇協調，論壇的紀錄存放在任何 sandbox 之外（所以 agent 無法刪除自己的紀錄）。在 weak-to-strong training 問題上，AAR 的表現勝過人類研究者。Anthropic 自己的摘要指出，預先規定的工作流程常常限制 AAR 的彈性並降低表現。自動化 alignment 研究會壓縮時程，而且是建立在 RSP 想要偵測的同一批 misalignment 風險之上。

**Type:** Learn
**Languages:** Python (stdlib, parallel-research-forum simulator)
**Prerequisites:** Phase 15 · 05 (AI Scientist v2), Phase 15 · 04 (DGM)
**Time:** ~60 分鐘

## 問題 (The Problem)

Alignment 研究很耗費人類研究者的時間。像 scalable oversight、reward specification 或 weak-to-strong training 這類問題，每次迭代的實驗都要花好幾週。隨著 frontier 能力推進，alignment 的工作量成長得比合格研究者的供給還快。

自動化 Alignment 研究 (Automated Alignment Research, AAR) 問的是：能力正在超越 alignment 的那些 frontier model，能不能幫忙縮小這個差距？Anthropic 2026 年關於 AAR 執行 weak-to-strong-training 研究的報告 (alignment.anthropic.com/2026/automated-w2s-researcher/)，是這類已部署系統最早的公開成果之一。

結果確實是正面的：AAR 解決一個研究問題的表現，優於人類 baseline。這個結果同時也把本階段發展出來的每一項疑慮，集中在單一系統上。如果 alignment 研究可以自動化，那麼會削弱安全措施的部分也可以。RSP 對 AI R&D 能力的門檻，就是以這個 loop 為前提寫的。

## 核心概念 (The Concept)

### 架構（依公開描述）

- **平行 agent。** 多個 Claude Opus 4.6 實例作為 AAR 執行，每個都在獨立的 sandbox 中。Sandbox 限制檔案系統、網路與行程的接觸面。
- **共用論壇。** 發現會發布到一個共用論壇，其儲存後端位於每個 sandbox 之外。Agent 可以讀取論壇；agent 無法從自己的 sandbox 內刪除或編輯論壇紀錄。
- **任務分配。** 人類設計的任務佇列把研究問題指派給 AAR。AAR 提出拆解方式、執行實驗、撰寫發現。
- **聚焦 weak-to-strong training。** 已發表的研究處理的是 weak-to-strong training，也就是用較弱的 supervisor model 來訓練較強的 student model。AAR 在這個任務上勝過人類研究者。

### 為什麼 sandbox 之外的 log 很重要

Log 完整性這項性質，是受控實驗與可能已被汙染的實驗之間的差別。如果 AAR 能編輯自己的 log，它們就能掩蓋失敗或可疑的實驗。Anthropic 直接點出：這項性質是信任研究輸出的必要條件，而且要驗證並不簡單。

基礎設施上的類比，是 append-only log 加上 write-through 到外部儲存（例如 S3 object-lock、不可變的 Datadog log）。Agent 沒有修改過往紀錄的憑證。這和 2026 年企業資安團隊用來建立抵抗勒索軟體的稽核軌跡，是同一套模式。

### 預先規定工作流程的取捨

Anthropic 的報告指出，預先規定的工作流程常常限制 AAR 的彈性並降低表現。換句話說：如果人類設計者一步一步寫好研究計畫，AAR 的表現會比只給高層目標、並讓它自由拆解時更差。這與整個 agentic 研究的結果一致（LangGraph、Microsoft 的 Agent Framework）：過度規定的 scaffold 會傷害有能力的 agent。

安全上的含意比較微妙。提升 AAR 表現的那種彈性，同樣也讓未經核准的拆解、新穎的工具使用，以及目標漂移成為可能。這個取捨是設計空間的決策：要給多少彈性才足以擊敗人類，又不會讓 agent 有空間把目標泛化到分布之外？

### 時程壓縮的風險

RSP v3.0（第 19 課）引入了 AI R&D 能力門檻：相較於人類 + AI 工具，能以具競爭力的成本完全自動化 AI R&D pipeline。DeepMind 的 FSF v3 也包含類似的 ML R&D 自主性等級。兩個框架都把這個門檻視為提高控制措施的觸發點。

AAR 離這個門檻只差一步：它自動化了 pipeline 的一部分（針對特定、範圍明確任務的 alignment 研究），但不是端到端的能力開發 loop。時程的問題在於這個差距縮小得有多快。

時程壓縮就是複利式失敗的疑慮。如果 alignment 研究與能力研究以相近的速度複利成長，misalignment 風險面至少會與能力一樣快地擴大。如果能力複利更快（歷史趨勢），差距就會拉大。這就是 AAR 是「有條件的好事」的論點：只有在研究過程值得信任的前提下，每多一項 alignment 成果才會縮小差距。

### AAR 取代不了什麼

人類研究者設定任務佇列、審查結果，並握有 constitutional 的權限。AAR 加速的是 pipeline 的中段，不是兩端。Anthropic 已發表的成果同時包含 AAR 的貢獻，以及人類研究者對於該發表什麼、撤回什麼、改進什麼的判斷。

這與第 15 課的 propose-then-commit 模式相符，只是套用在研究本身：AAR 提議；人類 commit。

```figure
aar-forum
```

## 實際使用 (Use It)

`code/main.py` 模擬一個多 AAR 的研究論壇。三個獨立的「研究 agent」平行執行，把發現張貼到 append-only 論壇，並可選擇加入竄改的嘗試。

觀察：

- Sandbox 之外的 logging 如何抓到竄改 log 的嘗試。
- 允許自由拆解的任務分配，如何在較難的問題上勝過固定工作流程。
- 如果 log 不具備竄改可偵測性，單一被汙染的 AAR 如何汙染論壇的訊號。

## 交付 (Ship It)

`outputs/skill-aar-deployment-review.md` 是自動化 alignment 研究 pipeline 的部署前審查。

## 練習 (Exercises)

1. 執行 `code/main.py`。比較「fixed-workflow」與「free-decomposition」兩種設定。自由拆解一定會贏嗎？還是有些問題類型中，固定工作流程也有競爭力？

2. 修改 simulator，讓其中一個 agent 嘗試竄改 log。確認 append-only log 能偵測到。寫一段話，描述偵測特徵在 log 中的確切樣貌。

3. 閱讀 Anthropic 的 weak-to-strong AAR 報告。找出 AAR 勝過人類研究者的具體子任務。是什麼讓它適合自動化？

4. 設計一個任務佇列分配策略，在 AAR 的彈性（更好的結果）與預先規定工作流程的限制（更容易稽核）之間取得平衡。描述你會如何對兩者做 A/B 測試。

5. 閱讀 RSP v3.0 的 AI R&D-4 門檻。用一段話描述，你認為什麼樣的能力會跨過它，而 AAR 目前還沒有。

## 關鍵術語 (Key Terms)

| 術語 | 一般說法 | 實際意義 |
|---|---|---|
| AAR | 「Automated Alignment Researcher」 | 自主處理 alignment 問題的 Claude Opus 4.6 實例 |
| Weak-to-strong training | 「用較弱的 supervisor 訓練較強的模型」 | AAR 表現勝過人類的經典 scalable-oversight benchmark |
| Shared forum | 「agent 發布發現的地方」 | Append-only、位於 sandbox 之外的儲存 |
| Out-of-sandbox log | 「agent 無法編輯自己的紀錄」 | 可偵測竄改、write-through 到外部儲存 |
| Prescribed workflow | 「人類設計者給的逐步計畫」 | 限制 AAR；相較於自由拆解，常常降低表現 |
| Free decomposition | 「agent 自己決定如何拆解任務」 | 能力更強，但更難稽核 |
| AI R&D threshold | 「RSP/FSF 的能力等級」 | 以具競爭力的成本完全自動化 R&D pipeline |
| Compressed timeline | 「alignment 對能力的競賽」 | 如果能力複利成長得比 alignment 快，misalignment 風險就會增加 |

## 延伸閱讀 (Further Reading)

- [Anthropic — Automated Weak-to-Strong Researcher](https://alignment.anthropic.com/2026/automated-w2s-researcher/) — 第一手來源。
- [Anthropic Responsible Scaling Policy v3.0](https://anthropic.com/responsible-scaling-policy/rsp-v3-0) — AI R&D 門檻的框架。
- [Anthropic — Measuring AI agent autonomy](https://www.anthropic.com/research/measuring-agent-autonomy) — 更廣的 agent 自主性框架。
- [DeepMind Frontier Safety Framework v3](https://deepmind.google/blog/strengthening-our-frontier-safety-framework/) — 與 RSP 平行的 ML R&D 自主性等級。
- [Burns et al. (2023). Weak-to-Strong Generalization (OpenAI)](https://openai.com/index/weak-to-strong-generalization/) — AAR 所攻克的底層問題。
