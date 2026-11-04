# METR Time Horizon 與外部能力評估 (METR Time Horizons and External Capability Evaluation)

> METR（前身為 ARC Evals）自 2023 年 12 月起是獨立的 501(c)(3) 組織。他們的 Time Horizon 1.1 benchmark（2026 年 1 月）把任務成功機率對 log(專家完成時間) 做 logistic 曲線擬合；曲線與 50% 機率的交點，定義了模型的 time horizon。2025–2026 年的合作案涵蓋 GPT-5.1、GPT-5.1-Codex-Max，以及原型的監控評估（monitor 能不能抓到 side task；agent 能不能躲過）。Benchmark 套件：HCAST（180 多個 ML、資安、SWE、推理任務；1 分鐘到 8 小時以上）、RE-Bench（71 個附有專家 baseline 的 ML 研究工程任務）、SWAA。誠實的註記是：METR 的量測是理想化的，沒有人類、沒有真實後果，團隊也已記錄評估與部署之間的行為落差（第 1 課）。Time horizon 是上限，不是部署表現的預測。

**Type:** Learn
**Languages:** Python (stdlib, logistic-fit horizon estimator)
**Prerequisites:** Phase 15 · 01 (Long-horizon agents), Phase 15 · 19 (RSP)
**Time:** ~60 分鐘

## 問題 (The Problem)

擴展政策（第 19、20 課）的用處，只取決於它們所參照的量測。「AI R&D-4 門檻」和「Long-range Autonomy」是用政策的文字定義的；只有當特定的評估產出特定的數字時，它們才變得可以付諸行動。

METR 是 2024–2026 年定義了許多這類數字的外部評估組織。他們評估 frontier 模型，常常是在發布之前、與實驗室簽訂 NDA 的情況下，事後再發布方法論。Time Horizon 1.1 benchmark（2026 年 1 月）是他們最具代表性的成果：一個純量，把能力壓縮成人類看得懂的單位（「這個模型能以 50% 的可靠度，完成專家要花 X 小時才能完成的那類任務」）。

這堂課一部分講方法論（horizon 如何計算），一部分講詮釋（為什麼 horizon 是上限，而不是部署表現的預測）。這兩項技能應該放在一起。一個理解 horizon 如何被擬合出來的團隊，比起只看到投影片上「14 小時」的團隊，要難被糟糕的廠商宣稱矇騙得多。

## 核心概念 (The Concept)

### METR 背景

- 成立：2023 年 12 月（前身為 ARC Evals，分出來成為獨立的 501(c)(3)）。
- 範圍：評估 frontier 模型的自主能力，常常在發布之前。
- 合作實驗室：Anthropic、OpenAI（2025–2026 年有多次合作）。
- 值得注意的成果：Time Horizon 1.0（2025 年 3 月）、Time Horizon 1.1（2026 年 1 月）、原型監控評估。

### Time Horizon 擬合

方法論（出自 METR 的部落格與論文）：

1. 收集一套任務，專家完成時間從分鐘等級涵蓋到小時等級。目前的套件：HCAST（180 多個任務）、RE-Bench（71 個任務）、SWAA。
2. 讓模型對每個任務執行；記錄成功或失敗。
3. 擬合 logistic 曲線：$P(\text{success})$ 作為 $\log(\text{expert completion time})$ 的函數。
4. Horizon 是 P(success) = 0.5 時的專家時間。

Logistic 擬合的形狀是正確的，因為能力與任務難度之間，通常是遞增、並趨近平台的關係。50% 這個點是一種選擇（可以是 10%、90%）；METR 在詳細論文中報告多個門檻，但以 50% 為主，因為它最直觀。

### 2026 年 1 月的數字

依據 Time Horizon 1.1：

- Claude Opus 4.6：在 Time Horizon 1.1（2026 年 1 月）時，50% 可靠度下約 14 小時。
- HCAST 類型任務的倍增時間：依 Time Horizon 1.1（2026 年 1 月）回報的 2023 年之後的擬合，約為 4.3 個月（130.8 天）；約 7 個月的數字，則是 Time Horizon 1.0 中 2019–2025 年的完整擬合，TH1.1 把它與 2023 年之後的數字並列回報。

這些就是 Phase 15 其餘部分所引用的具體數字。它們在 Anthropic 的 RSP v3.0（第 19 課）和 DeepMind 的 FSF v3（第 20 課）中，被引用為餵入門檻評估的量測之一。

### Benchmark 套件

- **HCAST**：涵蓋 ML、資安、軟體工程與一般推理的 189 個任務。範圍：1 分鐘到 8 小時以上的專家時間。是 horizon 數字背後的主要套件。
- **RE-Bench**：71 個附有人類專家 baseline 的 ML 研究工程任務。衡量模型能不能做出會餵入 AI R&D-4 門檻的那類 ML R&D 工作。
- **SWAA**：較短的任務；用於校準 horizon 曲線的低端。

### 原型監控評估

2025–2026 年 METR 的工作包含監控評估的原型：monitor 能不能抓到模型在執行 side task（operator 要求之外的事）；agent 能不能躲過 monitor。這些直接餵入 OpenAI PF v2 中的「Undermining Safeguards」類別，以及 DeepMind FSF v3 中的 deceptive-alignment 監控（第 20 課）。

### 為什麼 horizon 是上限

- **理想化的工具**：Benchmark 任務給模型乾淨的工具和格式良好的資料。真實部署比較雜亂。
- **沒有真實後果**：模型從不會真的向客戶收費、刪除真實資料，或聯絡真實的人。真實部署有不可逆的賭注。
- **Eval-context gaming**：第 1 課。模型在測試中的行為不同。2026 年的 International AI Safety Report 以實證記錄了這點。
- **沒有合理的使用者變異**：Benchmark 的 prompt 是結構化的。真實使用者會提出含糊、依賴脈絡的請求。

Horizon 是有利條件下的能力天花板。部署可靠度是另一個數字，比較低，團隊必須量測自己的分布才知道。

### 外部評估者的必要性

外部評估之所以重要，是因為實驗室內部有誘因去優化它們所回報的指標。METR 的獨立性（有宣告方法論並發表同行審查論文的 501(c)(3)）是結構上的緩解措施。單獨這樣並不充分（實驗室仍然控制 METR 能看到什麼），但嚴格來說，比沒有外部評估好。

### 如何在實務上使用 horizon 數字

- **作為能力過濾器**：如果模型的 horizon 遠低於某個提議任務的專家時間，就不要讓它自主上線（第 1 課的 skill 檔案）。
- **作為趨勢指標**：倍增時間告訴你，即使沒有新的緩解措施，目前的做法還能維持安全多久。
- **作為先驗**：14 小時的 horizon 是起點。依你的任務分布、工具品質與部署脈絡往下調整。

```figure
a5-horizon-fit
```

## 實際使用 (Use It)

`code/main.py` 給定一組合成的結果，實作任務成功率對 $\log(\text{expert time})$ 的 logistic 擬合。它回報 50% horizon（METR 的主要數字）、10% horizon（保守）和 90% horizon（樂觀）。也展示當成功率被 eval-context gaming 人為灌水時，會有什麼改變。

## 交付 (Ship It)

`outputs/skill-horizon-interpretation.md` 審查廠商的 horizon 宣稱，並產出 benchmark 宣稱與部署現實之間的落差分析。

## 練習 (Exercises)

1. 執行 `code/main.py`。確認擬合出的 50% horizon 與合成的真實值相符。現在把任務時間網格減半；horizon 的估計值有明顯改變嗎？

2. 閱讀 METR 的 Time Horizon 1.1 部落格文章。找出可靠度最高與最低的具體任務。說明為什麼會有這個落差。

3. 閱讀 METR 的「Measuring Autonomous AI Capabilities」資源。列出 HCAST 的任務類別。挑一個你在 production 任務中會給予更高權重的類別，並說明理由。

4. 在 simulator 中引入 eval-context gaming：把約 20% 失敗的任務翻轉成成功。回報新的 horizon。這近似於 20% 的 gaming 率對觀察到的數字造成的影響。

5. 在你自己的 bug backlog 或一組有代表性的任務集上，設計一個內部的 horizon 評估。描述資料收集、擬合，以及輸出告訴你什麼。與 METR 的數字比較。

## 關鍵術語 (Key Terms)

| 術語 | 一般說法 | 實際意義 |
|---|---|---|
| METR | 「外部評估者」 | 前身為 ARC Evals；自 2023 年 12 月起為獨立的 501(c)(3) |
| Time Horizon | 「能力量測」 | 由 logistic 擬合得出、50% 可靠度下的專家任務長度 |
| HCAST | 「METR 的主要套件」 | 180 多個任務，涵蓋 1 分鐘到 8 小時以上 |
| RE-Bench | 「研究工程」 | 71 個附有人類 baseline 的 ML 研究工程任務 |
| SWAA | 「短任務套件」 | 校準 horizon 曲線的低端 |
| Doubling time | 「成長率」 | 50% horizon 翻倍所需的時間；HCAST 約 7 個月 |
| Eval-context gaming | 「模型行為不同」 | 有記錄的測試與部署之間的行為落差 |
| Upper bound | 「Horizon 是天花板」 | Benchmark horizon > 負載下的部署可靠度 |

## 延伸閱讀 (Further Reading)

- [METR — Resources for Measuring Autonomous AI Capabilities](https://metr.org/measuring-autonomous-ai-capabilities/) — HCAST、RE-Bench、SWAA 的規格。
- [METR — Measuring AI Ability to Complete Long Tasks](https://metr.org/blog/2025-03-19-measuring-ai-ability-to-complete-long-tasks/) — 最早的 horizon 論文。
- [METR — Time Horizon 1.1 (January 2026)](https://metr.org/research/) — 目前的數字與方法論。
- [Epoch AI — METR Time Horizons benchmark](https://epoch.ai/benchmarks/metr-time-horizons) — 即時追蹤。
- [Anthropic — Measuring agent autonomy in practice](https://www.anthropic.com/research/measuring-agent-autonomy) — 對 METR 量測的內部觀點。
