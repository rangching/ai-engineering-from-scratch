# AlphaEvolve：演化式 Coding Agent (Evolutionary Coding Agents)

> 把 frontier coding model 搭配演化 loop 和機器可檢查的 evaluator，讓 loop 跑夠久。它發現了一個只用 48 次純量乘法的 4x4 複數矩陣乘法程序，這是 56 年來第一次超越 Strassen。它也找到一個橫跨 Google 的 Borg 排程 heuristic，在 production 中回收了約 0.7% 的叢集運算資源。這個架構刻意走得很樸素。勝利來自 evaluator 的嚴謹。

**Type:** Learn
**Languages:** Python (stdlib, evolutionary-loop toy)
**Prerequisites:** Phase 15 · 01 (long-horizon framing), Phase 15 · 02 (self-taught reasoning)
**Time:** ~60 分鐘

## 問題 (The Problem)

大型語言模型會寫程式。演化演算法能在程式碼上搜尋。兩者各自被嘗試了數十年，也都碰到天花板。LLM 的天花板是 confabulation：模型寫出看似合理、卻沒做到它所宣稱的事的程式。演化的天花板是搜尋成本：在語法上做隨機突變，很少能產生可編譯的程式，更別說更好的程式。

AlphaEvolve (Novikov et al., DeepMind, arXiv:2506.13131, 2025 年 6 月) 把兩者結合。LLM 對程式資料庫提出有針對性的修改；自動 evaluator 為每個變體打分；高分的變體成為後續世代的親代。LLM 負責昂貴的步驟，也就是寫出看似合理的程式；evaluator 負責抓出 confabulation。這個 loop 會跑幾小時到幾週。

回報的結果：用 48 次純量乘法完成 4x4 複數矩陣乘法（Strassen 在 1969 年的界限是 49）、Google production 中的 Borg 排程 heuristic、FlashAttention kernel 加速 32.5%、Gemini 訓練吞吐量的提升。

這個架構之所以有效，是因為 evaluator 可以由機器檢查。在 evaluator 做不到的地方，它就沒用。這種不對稱就是這堂課的重點。

## 核心概念 (The Concept)

### Loop

1. 從一個正確但不是最佳的 seed program `P_0` 開始。
2. 維護一個變體程式的資料庫，每個都由 evaluator 打分。
3. 從資料庫取樣一個或多個親代（MAP-elites 風格或 island-based）。
4. 提示 LLM（大量候選用 Gemini Flash，困難的用 Gemini Pro）產生親代的修改版本。
5. 編譯、執行，並用 held-out evaluator 評估這個變體。
6. 以分數和特徵向量為 key，插入資料庫。
7. 重複。

兩個細節很重要。第一，給 LLM 的 prompt 不只有親代程式，通常還包括資料庫中幾個最佳的變體、evaluator 的簽章，以及簡短的任務描述。模型的工作是提出可能提高分數的針對性修改。第二，資料庫是有結構的（MAP-elites 網格、island-based），讓 loop 探索多樣性，而不只是跟著目前的領先者。

### 為什麼 evaluator 沒有討價還價的空間

AlphaEvolve 所有的勝利，都來自 evaluator 快速、確定性、而且難以被鑽漏洞的領域：

- **矩陣乘法演算法**：一個把矩陣相乘並逐位元檢查相等的 unit test。
- **Borg 排程 heuristic**：一個 production 等級的 simulator，重播歷史叢集負載並量測被浪費的運算資源。
- **FlashAttention kernel**：一個正確性測試，加上在真實硬體上的 wall-clock benchmark。
- **Gemini 訓練吞吐量**：量測每一步花費的 GPU 秒數。

在每個案例中，evaluator 都抓住了原本會佔大宗的那類 LLM 錯誤：虛構的正確性宣稱、在硬體上就消失的效能宣稱，以及 edge case 的失敗。拿掉 evaluator，loop 就會去優化好看的程式碼。

### Reward hacking 是同一句話的另一面

演化會優化 evaluator 所量測的任何東西。如果 evaluator 不完美，loop 就會找到那個不完美之處。在沒有驗證的領域，loop 會優化表面特徵，而不是預期的行為。DeepMind 在論文中明確指出這點：AlphaEvolve 的成功，只能轉移到 evaluator 的嚴謹程度與搜尋野心相符的領域。

2025-2026 年 code-search loop 中 reward hacking 的具體例子：

- 獎勵「完成時間」的優化目標，獎勵了提交空白解答的行為。
- 獎勵「測試下正確性」的 benchmark 分數，獎勵了背下測試和 overfitting。
- 「程式碼品質」的代理指標，獎勵了刪除註解和重寫變數名稱，而語意完全沒變。

AlphaEvolve 的解法：提供一個 LLM 從未見過的 held-out evaluator，輸入在評估時才產生。即使如此，DeepMind 仍建議對任何提議的部署做嚴格審查。

### 為什麼 LLM + 搜尋勝過單獨使用任何一個

LLM 可以產生可編譯、語意上合理的修改。對 2000 行的 Python 檔案做隨機突變的 GA，幾乎都會產生語法錯誤。LLM 也讓搜尋集中在合理的鄰近區域（改一個函式，而不是隨機位元組），大幅減少浪費的 evaluator 呼叫。

反過來，evaluator 抓出 LLM 的 confabulation。LLM 會很有自信地宣稱某個函式「極限下是 $O(n \log n)$」，實際上卻是 $O(n^2)$；wall-clock benchmark 能讓這個問題塵埃落定。

### AlphaEvolve 在 frontier 技術棧中的位置

| 系統 | Generator | Evaluator | 領域 | 代表成果 |
|---|---|---|---|---|
| AlphaEvolve | Gemini | 正確性 + benchmark | 演算法、kernel、排程器 | 48 次乘法的 4x4 matmul |
| FunSearch (DeepMind, 2023) | PaLM / Codey | 正確性 | 組合數學 | cap-set 下界 |
| AI Scientist v2 (Sakana, L5) | GPT/Claude | LLM 評論 + 實驗 | ML 研究 | ICLR workshop 論文 |
| Darwin Godel Machine (L4) | agent scaffolding | SWE-bench / Polyglot | agent 程式碼 | 20% → 50% SWE-bench |

四者都是同一個配方的變化：generator 加 evaluator，再跑 loop。差別在於 evaluator 評分的對象，以及它有多嚴謹。

```figure
alphaevolve-loop
```

## 實際使用 (Use It)

`code/main.py` 在一個玩具 symbolic-regression 問題上，實作了最小的類 AlphaEvolve loop。這裡的「LLM」是一個 stdlib 代理，對計算目標函式的程式提出小幅度的語法突變。「evaluator」則在 held-out 測試點上量測均方誤差。

觀察：

- 最佳分數如何隨世代進步。
- MAP-elites 網格如何讓多樣的解保持存活，使 loop 不會收斂到 local minimum。
- 拿掉 held-out 測試（只用訓練資料的 evaluator）後，loop 如何大幅 overfit。

## 交付 (Ship It)

`outputs/skill-evaluator-rigor-audit.md` 是在新領域考慮 AlphaEvolve 風格 loop 的前提條件：你的 evaluator 真的能抓到你在乎的失敗嗎？

## 練習 (Exercises)

1. 執行 `code/main.py`。記下最佳分數的軌跡。停用 held-out evaluator（flag `--no-holdout`）後再執行一次。量化 overfitting 的程度。

2. 閱讀 AlphaEvolve 論文第 3 節關於 MAP-elites 網格的部分。為一個新問題（例如 compiler 最佳化 pass）設計特徵向量描述子，讓搜尋保持多樣。

3. 48 次乘法的 4x4 結果，在 56 年後改進了 Strassen 的 49 次乘法界限。閱讀論文附錄 F，用三句話解釋為什麼這個問題的 evaluator 特別容易做對，以及為什麼大多數領域不是這樣。

4. 提出一個 AlphaEvolve 會失敗的領域。明確指出 evaluator 在哪裡失效，以及原因。

5. 針對你熟悉的領域，寫出你會使用的 evaluator 簽章。包含 (a) 正確性條件、(b) 效能指標、(c) held-out 輸入的產生規則、(d) 至少一項防止 reward hacking 的檢查。

## 關鍵術語 (Key Terms)

| 術語 | 一般說法 | 實際意義 |
|---|---|---|
| AlphaEvolve | 「DeepMind 的演化式 coding agent」 | Gemini + 程式資料庫 + 機器可檢查的 evaluator |
| MAP-elites | 「保留多樣性的 archive」 | 以特徵向量為 key 的網格；每個格子存放具有該描述子的最佳變體 |
| Island model | 「平行演化的子族群」 | 定期遷移的獨立族群；避免過早收斂 |
| Machine-checkable evaluator | 「確定性的 oracle」 | LLM 無法偽造的 unit test、simulator 或 benchmark，是這個 loop 的前提 |
| Reward hacking | 「優化指標，而不是目標」 | Loop 找到不做預期任務也能把分數拉高的方法 |
| Seed program | 「起點」 | Loop 由此演化的初始程式，正確但不是最佳 |
| Held-out evaluator | 「LLM 從沒看過的評估資料」 | 在評估時才產生輸入，以防止背誦 |

## 延伸閱讀 (Further Reading)

- [Novikov et al. (2025). AlphaEvolve: A coding agent for scientific and algorithmic discovery](https://arxiv.org/abs/2506.13131) — 完整論文。
- [DeepMind blog on AlphaEvolve](https://deepmind.google/blog/alphaevolve-a-gemini-powered-coding-agent-for-designing-advanced-algorithms/) — 廠商撰寫的成果說明。
- [AlphaEvolve results repository](https://github.com/google-deepmind/alphaevolve_results) — 發現的演算法，包括 48 次乘法的 4x4 matmul。
- [Romera-Paredes et al. (2023). Mathematical discoveries from program search with LLMs (FunSearch)](https://www.nature.com/articles/s41586-023-06924-6) — 前身系統。
- [Anthropic — Responsible Scaling Policy v3.0 (Feb 2026)](https://anthropic.com/responsible-scaling-policy/rsp-v3-0) — 把受 evaluator 約束的自主性視為關鍵研究方向。
