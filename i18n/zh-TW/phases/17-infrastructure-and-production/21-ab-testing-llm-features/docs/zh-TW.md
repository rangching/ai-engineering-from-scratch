# LLM 功能的 A/B 測試：GrowthBook、Statsig 與憑感覺的問題 (A/B Testing LLM Features)

> 傳統的 A/B 測試不是為非決定性的 LLM 設計的。關鍵區別：evals 回答「模型能不能做好這份工作？」A/B 測試回答「使用者在乎嗎？」兩者都必須做；憑感覺檢查就上線的時代結束了。2026 年要測什麼：prompt engineering（措辭）、模型選擇（GPT-4 vs GPT-3.5 vs OSS；準確度 vs 成本 vs latency）、生成參數（temperature、top-p）。真實案例：某聊天機器人的 reward-model 變體帶來 +70% 的對話長度與 +30% 的留存；Nextdoor 的 AI 主旨行實驗在 reward 函式調整後帶來 +1% CTR；Khan Academy 的 Khanmigo 在 latency 對數學準確度的軸線上反覆迭代。平台分野：**Statsig**（2025 年 9 月被 OpenAI 以 $1.1B 收購），支援 sequential testing、CUPED，一站式。**GrowthBook** 開源、warehouse-native，具備 Bayesian + Frequentist + Sequential 引擎、CUPED、SRM 檢查、Benjamini-Hochberg + Bonferroni 校正。你要依自己對 warehouse-SQL 的偏好，以及「被 OpenAI 收購」對你的組織是否重要來挑選。

**Type:** Learn
**Languages:** Python (stdlib, toy sequential test simulator)
**Prerequisites:** Phase 17 · 13 (Observability), Phase 17 · 20 (Progressive Deployment)
**Time:** ~60 分鐘

## 學習目標 (Learning Objectives)

- 區分 evals（「模型能不能做好這份工作」）與 A/B 測試（「使用者在乎嗎」）。
- 列舉三個可測試的軸線（prompt、模型、參數），並為每個挑選指標。
- 解釋 CUPED、sequential testing，以及 Benjamini-Hochberg 多重比較校正。
- 依 warehouse-SQL 立場與企業收購立場，在 Statsig 與 GrowthBook 之間挑選。

## 問題 (The Problem)

你手工調整了 system prompt。感覺比較好。你上線了。轉換率的變化只是雜訊。你怪罪指標。或者你上線了新模型，轉換率沒有動，是模型退化了，還是變動太小偵測不到？你不知道，因為你沒有 A/B 就上線了。

Evals 回答的是模型能否在有標註的集合上完成任務。它們無法回答使用者是否偏好該輸出。只有受控的線上實驗能回答，而且前提是實驗有足夠的檢定力、控制了非決定性，並對多重比較做了校正。

## 核心概念 (The Concept)

### Evals vs A/B 測試

**Evals**：離線、有標註的集合、評審（rubric 或 LLM-as-judge 或人工）。回答：「輸出在這個固定分布上是否正確 / 有幫助 / 安全？」

**A/B 測試**：線上、真實使用者、隨機分派。回答：「新變體是否改變了重要的使用者層級指標？」

兩者都需要。Evals 在曝光前抓出退化；A/B 在曝光後確認產品影響。

### 要測什麼

1. **Prompt engineering**：措辭、system-prompt 結構、範例。指標：任務成功、使用者留存、每個 request 的成本。
2. **模型選擇**：GPT-4 vs GPT-3.5-Turbo vs Llama-OSS。指標：準確度（任務）+ 每個 request 成本 + latency P99。多目標。
3. **生成參數**：temperature、top-p、max_tokens。指標：因任務而異（輸出多樣性 vs 決定性）。

### CUPED：降低變異

Controlled-experiments Using Pre-Experiment Data。在比較實驗後期間之前，先把實驗前期間的變異回歸掉。典型的變異降低：30-70%。有效樣本數免費提升。

實作：Statsig 與 GrowthBook 都有實作。

### Sequential testing

傳統 A/B 假設固定樣本數。Sequential 測試（「偷看再決定」）在重複查看下仍能控制偽陽性率。永遠有效的 sequential 程序（mSPRT、Howard 的 confidence sequences）讓你能在明顯的贏家出現時提早停止。

### 多重比較校正

以 95% 信賴度跑 20 個 A/B 測試，偶然就會產生一個偽陽性。Bonferroni 校正會收緊每個測試的 α；Benjamini-Hochberg 控制的是錯誤發現率。GrowthBook 兩者都有實作。

### SRM：sample ratio mismatch

分派 hash 會把使用者隨機分到變體。如果 50/50 的切分變成 47/53，就是哪裡壞了，SRM 檢查會標記它。兩個平台都有實作。

### Statsig vs GrowthBook

**Statsig**：
- 被 OpenAI 以 $1.1B 收購（2025 年 9 月）。託管、SaaS。
- Sequential testing、CUPED、held-out 族群。
- 一站式：feature flags + 實驗 + observability。
- 最適合：已經想要打包產品、不在意 OpenAI 擁有權的團隊。

**GrowthBook**：
- 開源 (MIT)；warehouse-native（直接從 Snowflake/BigQuery/Redshift 讀取）。
- 多種引擎：Bayesian、Frequentist、Sequential。
- CUPED、SRM、Bonferroni、BH 校正。
- 可自架或使用託管雲端。
- 最適合：以 warehouse-SQL 為主、資料團隊掌控指標層、想要 OSS。

### 非決定性讓檢定力複雜化

同一個 prompt 會產生不同的輸出。傳統的檢定力計算假設觀測值是 IID。在 LLM 非決定性下，有效樣本數低於名目值。所需樣本數請乘上約 ~1.3-1.5x 作為安全邊際。

### 真實案例結果

- 聊天機器人 reward model 變體：+70% 對話長度、+30% 留存。
- Nextdoor 主旨行：reward 函式調整後 +1% CTR。
- Khan Academy Khanmigo：latency 對數學準確度的反覆取捨。

### 反模式：憑感覺上線

每位資深工程師都能舉出一個因為「感覺比較好」而沒做 A/B 就上線的功能。其中大多數讓團隊數月都沒注意到的產品指標退步了。A/B 是強制手段。

### 該記住的數字

- Statsig 被 OpenAI 收購：$1.1B，2025 年 9 月。
- GrowthBook：開源 MIT；Bayesian + Frequentist + Sequential。
- CUPED 變異降低：30-70%。
- LLM 非決定性 → 樣本數緩衝 +30-50%。

```figure
mx-sequential-test
```

## 實際使用 (Use It)

`code/main.py` 以固定邊界與 sequential 邊界模擬 sequential A/B 測試。展示 sequential 如何讓你提早停止。

## 交付 (Ship It)

這堂課產出 `outputs/skill-ab-plan.md`。給定功能變更、工作負載、基準，挑選平台、關卡、樣本數。

## 練習 (Exercises)

1. 執行 `code/main.py`。對基準轉換率 3%、預期 5% 提升，達到 80% 檢定力需要多少樣本數？
2. 為一位受醫療法規限制的 on-prem 客戶，在 Statsig 與 GrowthBook 之間挑一個。
3. 設計一個以每解決一張工單的成本來測試 GPT-4 vs GPT-3.5 的 A/B。主要指標、guardrail 指標、次要指標各是什麼？
4. 你的 canary 通過了，但 A/B 顯示轉換率 -1.2%。要上線嗎？寫出升級準則。
5. 以一個能解釋實驗後期間 60% 變異的實驗前期間共變數（相關係數平方 0.6）套用 CUPED。計算有效樣本數的提升。

## 關鍵術語 (Key Terms)

| 術語 | 一般說法 | 實際意義 |
|------|----------------|------------------------|
| Eval | 「離線測試」 | 對模型能力做有標註集合的評估 |
| A/B test | 「實驗」 | 對使用者的即時隨機比較 |
| CUPED | 「變異降低」 | 以實驗前期間的回歸降低變異 |
| Sequential test | 「可偷看的測試」 | 允許提早停止的永遠有效程序 |
| Multiple comparison | 「家族錯誤」 | 執行許多測試會使偽陽性膨脹 |
| Bonferroni | 「嚴格校正」 | 把 α 除以測試數量 |
| Benjamini-Hochberg | 「BH FDR」 | 錯誤發現率控制，較不保守 |
| SRM | 「壞掉的切分」 | Sample ratio mismatch；分派的 bug |
| Statsig | 「OpenAI 擁有」 | 商業一站式平台，2025 年被收購 |
| GrowthBook | 「OSS 那個」 | MIT 的 warehouse-native 平台 |
| mSPRT | 「sequential probability ratio test」 | 經典的 sequential 程序 |

## 延伸閱讀 (Further Reading)

- [GrowthBook — How to A/B Test AI](https://blog.growthbook.io/how-to-a-b-test-ai-a-practical-guide/)
- [Statsig — Beyond Prompts: Data-Driven LLM Optimization](https://www.statsig.com/blog/llm-optimization-online-experimentation)
- [Statsig vs GrowthBook comparison](https://www.statsig.com/perspectives/ab-testing-feature-flags-comparison-tools)
- [Deng et al. — CUPED](https://www.exp-platform.com/Documents/2013-02-CUPED-ImprovingSensitivityOfControlledExperiments.pdf)
- [Howard — Confidence Sequences](https://arxiv.org/abs/1810.08240)
