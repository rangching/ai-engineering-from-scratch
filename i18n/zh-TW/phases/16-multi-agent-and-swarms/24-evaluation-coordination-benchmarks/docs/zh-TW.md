# 評估與協調基準 (Evaluation and Coordination Benchmarks)

> 五個 2025-2026 年的基準涵蓋了 multi-agent 評估的空間。**MultiAgentBench / MARBLE**（ACL 2025，arXiv:2503.01935）以 milestone KPI 評估 star/chain/tree/graph 這幾種 topology；**graph 最適合研究型任務**，認知規劃（cognitive planning）能多帶來約 3% 的 milestone 達成率。**COMMA** 評估多模態、資訊不對稱的協調；包括 GPT-4o 在內的最先進模型，連隨機基準都很難打敗。**MedAgentBoard**（arXiv:2505.12371）涵蓋四類醫療任務，常發現 multi-agent 並沒有勝過 single-LLM。**AgentArch**（arXiv:2509.10769）對結合 tool-use + memory + orchestration 的企業 agent 架構做基準測試。**SWE-bench Pro**（[arXiv:2509.16941](https://arxiv.org/abs/2509.16941)）有 1865 個問題，橫跨 41 個 repo，涵蓋商業應用、B2B 服務和開發者工具；前沿模型在 Pro 上得分約 ~23%，在 Verified 上則是 70% 以上，這是對資料汙染的一次現實檢驗。Claude Opus 4.7（2026 年 4 月）在 Pro 上，搭配明確的 agent-teams 協調，回報為 **64.3%**（目前尚無 Anthropic 的第一手來源，請視為初步數字）；Verdent（agent scaffold）在 Verified 上達到 **76.1% pass@1**（[Verdent technical report](https://www.verdent.ai/blog/swe-bench-verified-technical-report)）。**AAAI 2026 Bridge Program WMAC**（https://multiagents.org/2026/）是 2026 年社群的焦點。這堂課以 MARBLE 的指標為基礎，執行一次 topology 對指標的掃描，並釘下「光是通過 SWE-bench Verified 並不是泛化能力的證據」這條規則。

**Type:** Learn
**Languages:** Python (stdlib)
**Prerequisites:** Phase 16 · 15 (Voting and Debate Topology), Phase 16 · 23 (Failure Modes)
**Time:** ~75 分鐘

## 問題 (Problem)

當一篇論文宣稱「我們的 multi-agent 系統比較好」，問題是：比什麼好、在什麼上、用什麼測量？2023-2024 年的 multi-agent 評估是一片混亂，每個人都挑自己的指標、自己的基準、自己的任務集。2025-2026 年的基準帶來了結構。

沒有共享的基準，你就無法有意義地比較兩個 multi-agent 系統。更糟的是，沒有保留 (hold-out) 的基準，前沿模型可能造成汙染。SWE-bench Verified 到 2025 年中已部分汙染了訓練語料；前沿的分數被膨脹；Pro 就是為了當作未被汙染的現實檢驗而設計的。

這堂課列舉五個 2026 年的標準基準，說明每一個測量什麼，並教你帶著懷疑的眼光閱讀基準的宣稱。

## 核心概念 (Concept)

### MultiAgentBench (MARBLE) — ACL 2025

arXiv:2503.01935。在研究、程式開發與規劃任務上，評估四種協調 topology（star、chain、tree、graph）。以 milestone 為基礎的 KPI 追蹤部分進展，而不是只看最終成功。

測得的結果：

- **Graph** topology 最適合研究情境；支援任意對任意的批評。
- **Chain** 最適合逐步精煉的程式開發。
- **Star** 最適合快速的事實整合。
- 在 graph 上，超過約 4 個 agent 後會出現 **協調稅**。
- **認知規劃** 在各種 topology 上多帶來約 3% 的 milestone 達成率。

何時使用：你想把協調 topology 做公平的對照比較。MARBLE repo (https://github.com/ulab-uiuc/MARBLE) 提供了評估器。

### COMMA — 多模態、資訊不對稱

涵蓋 agent 具有不同觀察模態、必須在不完整分享資訊的情況下協調的任務。回報的結果令人不舒服：包括 GPT-4o 在內的前沿模型，在 COMMA 的 agent 對 agent 協作上，連 **隨機基準** 都很難打敗。這個訊號是，multi-agent 的 modality 訓練不足、評估也不足：LLM 還能合理地處理單一模態的合作；多模態的協調則崩潰。

何時使用：你的系統有多模態或資訊不對稱的協調。COMMA 的負面結果是一個警告：先測量，再宣稱。

### MedAgentBoard — 領域壓力測試

arXiv:2505.12371。四類醫療任務：診斷、治療規劃、報告產生、病患溝通。比較 multi-agent、single-LLM 與傳統的規則式系統。

發現：在大多數類別上，multi-agent 並沒有勝過 single-LLM。Multi-agent 的優勢很窄：當子任務清楚可分離時（診斷 + 治療），任務拆解有幫助；當協調的額外負擔超過專門化的收益時（報告產生），反而有害。

何時使用：你的領域有明確的 single-LLM 基準。如果 MedAgentBoard 的教訓可以推廣，那麼許多被提出的 multi-agent 系統都是過度工程。

### AgentArch — 企業架構

arXiv:2509.10769。企業環境中，tool use、memory 和 orchestration 疊加在一起。這個基準隔離出每一層的貢獻：加入 tool 有多大幫助？加入 memory 呢？加入 multi-agent orchestration 呢？

何時使用：你正在設計企業 agent 堆疊，需要為每一層提出理由。AgentArch 幫助你避免買下你無法衡量價值的功能。

### SWE-bench Pro — 現實檢驗

arXiv:2509.16941。1865 個問題，橫跨 41 個 repo，涵蓋商業應用、B2B 服務和開發者工具。設計目標是對較晚訓練截止日期的模型保持 **未被汙染**。前沿模型在 Pro 上得分約 ~23%，在 Verified 上是 70% 以上。這個差距就是汙染的訊號。

2026 年 4 月的分數：
- Claude Opus 4.7 在 Pro 上：**64.3%**（搭配明確的 agent-teams 協調回報；目前尚無 Anthropic 的第一手來源，請視為初步數字）。
- Verdent（agent scaffold）在 Verified 上：**76.1% pass@1**（[technical report](https://www.verdent.ai/blog/swe-bench-verified-technical-report)）。
- 前沿模型在沒有 agent scaffold 的 Pro 上的原始分數：約 ~23-35%（[SWE-bench Pro paper](https://arxiv.org/abs/2509.16941)）。

重點：「我們打敗了 SWE-bench Verified」已不再是能力的證據。Pro 是目前的把關測試。Agent-team 的 scaffold 在 Pro 上帶來可測量的增益（約 ~30-40 個百分點的差距），這是 2026 年支持 multi-agent 協調最有力的實證論點之一。

### AAAI 2026 WMAC

AAAI 2026 Bridge Program — Workshop on Multi-Agent Coordination (https://multiagents.org/2026/)。2026 年 multi-agent AI 研究的社群焦點。被接受的論文與 workshop 論文集，是評估新方法的標準場合；做正式環境決策時，WMAC 接受的宣稱比 arXiv preprint 更值得採信。

### 帶著懷疑閱讀基準宣稱：2026 年的檢查清單 (Read benchmark claims skeptically — the 2026 checklist)

當有人宣稱一個 multi-agent 的結果時：

1. **哪個基準、哪個 split？** SWE-bench Verified 與 Pro 差很多。在錯誤的 split 上回報的數字毫無價值。
2. **汙染檢查。** 這個基準是在模型的訓練截止日期之後才發布的嗎？如果不是，要謹慎看待。
3. **基準比較。** 對 single-LLM 基準、對隨機、對先前的 multi-agent 研究。不是「對同一個系統未調校的版本」。
4. **統計顯著性。** N 次試驗、p-value、信賴區間。前沿模型的變異大；單次執行會誤導。
5. **任務多樣性。** 一個任務還是很多？泛化對正式環境很重要。
6. **成本揭露。** 每個任務的 token、實際耗時。以 20 倍成本達到 90% 的解法是商業決策，不是能力宣稱。

### 沒有任何基準測得好的東西 (What none of the benchmarks measure well)

- **長視野的協調。** 長達數天的實際互動。目前所有基準跑的時間都很短。
- **對抗性的韌性。** 當其中一個 agent 是惡意或被入侵時，會發生什麼事？
- **部署下的漂移。** 基準是靜態的；正式環境的分布會偏移。
- **成本正規化的表現。** 大多數基準回報的是原始準確度，而不是每一塊錢的準確度。

為你真正在意的那條軸建立自己的內部基準，往往是正確的做法。

```figure
a5-bench-gap
```

## 動手做 (Build It)

`code/main.py` 是非互動式的逐步講解：

- 在一個玩具任務上模擬 3 個 multi-agent 系統。
- 為每個系統計算 MARBLE 風格的 milestone 指標。
- 透過把任務從「訓練」集中保留起來，進行汙染檢查。
- 明確地與隨機基準比較。
- 印出基準宣稱的計分卡。

執行：

```bash
python3 code/main.py
```

預期輸出：系統計分卡，包含原始準確度、milestone 達成率、每個任務的成本、相對於隨機基準的差距，以及一則汙染檢查的註記。

## 實際使用 (Use It)

`outputs/skill-benchmark-reader.md` 讀取任何 multi-agent 基準的宣稱並套用審查檢查清單。輸出：一個評分與注意事項。

## 交付 (Ship It)

正式環境評估的紀律：

- **建立反映你實際正式環境分布的內部基準。** 公開基準可以提供參考，但無法取代。
- **每次比較都包含隨機基準。** 如果在協調任務上你無法大幅打敗隨機，這個任務可能本身就設得不好。
- **準確度之外也要回報成本。** Token 成本與實際耗時。維運團隊兩者都需要。
- **每季重建基準。** 正式環境的分布會偏移；過時的基準會誤導。
- **避免對已發布的基準過擬合。** 如果你的團隊專門為了 SWE-bench Pro 的數字而最佳化，你在正式環境中會退步。

## 練習 (Exercises)

1. 執行 `code/main.py`。找出三個模擬系統中，每個 milestone 成本最佳的是哪一個。它和原始準確度最高的系統一致嗎？
2. 閱讀 MultiAgentBench (arXiv:2503.01935)。針對你自己的任務領域，判斷 MARBLE 會推薦四種 topology 中的哪一種。根據論文的結果說明理由。
3. 閱讀 SWE-bench Pro 論文。具體是什麼讓它能抵抗汙染？同樣的技術能不能用在你在意的其他基準上？
4. 閱讀 COMMA 對多模態協調的發現。設計一個簡單的多模態協調任務，可以加入你的內部基準。什麼會算是有用的訊號？
5. 把基準宣稱的檢查清單套用到最近一篇 multi-agent 論文的標題結果。你會給這個宣稱什麼評分？

## 關鍵術語 (Key Terms)

| 術語 | 一般說法 | 實際意義 |
|------|----------------|------------------------|
| MARBLE | 「MultiAgentBench」 | ACL 2025；帶 milestone KPI 的 star/chain/tree/graph topology。 |
| COMMA | 「多模態基準」 | 多模態、資訊不對稱的協調；前沿模型對上隨機基準也很吃力。 |
| MedAgentBoard | 「領域壓力測試」 | 四個醫療類別；常發現 multi-agent 並沒有勝過 single-LLM。 |
| AgentArch | 「企業基準」 | Tool + memory + orchestration 疊加。 |
| SWE-bench Pro | 「抗汙染」 | 1865 個問題、41 個 repo；~23% 對 Verified 的 70% 以上（汙染的訊號）。 |
| Milestone achievement | 「部分給分」 | 獎勵進展、而不只是最終成功的基準。 |
| Contamination | 「基準洩漏進訓練資料」 | 發布之後，基準會漂進訓練語料；分數被膨脹。 |
| WMAC | 「AAAI 2026 Bridge Program」 | Workshop on Multi-Agent Coordination；社群焦點。 |

## 延伸閱讀 (Further Reading)

- [MultiAgentBench / MARBLE](https://arxiv.org/abs/2503.01935) — 帶 milestone KPI 的 topology 基準
- [MARBLE repository](https://github.com/ulab-uiuc/MARBLE) — 參考實作
- [MedAgentBoard](https://arxiv.org/abs/2505.12371) — 領域壓力測試；multi-agent 常常沒有勝出
- [AgentArch](https://arxiv.org/abs/2509.10769) — 企業 agent 架構
- [SWE-bench leaderboards](https://www.swebench.com/) — 前沿模型的 Verified 與 Pro 分數
- [AAAI 2026 WMAC](https://multiagents.org/2026/) — 2026 年的社群焦點
