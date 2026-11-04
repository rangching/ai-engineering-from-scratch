# Benchmark：SWE-bench、GAIA、AgentBench (Benchmarks: SWE-bench, GAIA, AgentBench)

> 2026 年有三個 benchmark 支撐著 agent 評估。SWE-bench 測試程式碼修補。GAIA 測試通才型的 tool use。AgentBench 測試跨多個環境的推理。要了解它們的組成、它們的資料污染狀況，以及它們沒有衡量什麼。

**Type:** Learn
**Languages:** Python (stdlib)
**Prerequisites:** Phase 14 · 06 (Tool Use)
**Time:** ~60 分鐘

## 學習目標 (Learning Objectives)

- 說出 SWE-bench 的測試 harness（FAIL_TO_PASS），並解釋為什麼它以 unit test 作為閘門。
- 解釋為什麼會有 SWE-bench Verified（OpenAI，500 個 task），以及它移除了什麼。
- 描述 GAIA 的設計：對人類簡單、對 AI 困難；三個難度等級。
- 說出 AgentBench 的八個環境，以及它對開源 LLM 的主要阻礙。
- 總結 SWE-bench+ 的資料污染發現及其意涵。

## 問題 (The Problem)

排行榜告訴你哪個模型在某個 benchmark 上勝出。它們不會告訴你：

- 這個 benchmark 是否受到污染（解答出現在訓練資料中、測試洩漏）。
- 這個 benchmark 是否衡量你在意的東西（程式碼、瀏覽或通才能力）。
- Evaluator 是否穩健（AST 比對、狀態檢查、人工審查）。

在引用某個數字之前，先了解這三個基準 benchmark 和它們的失敗模式。

## 核心概念 (The Concept)

### SWE-bench (Jimenez et al., ICLR 2024 oral)

- 來自 12 個熱門 Python repo 的 2,294 個真實 GitHub issue。
- Agent 取得：修復前 commit 的程式碼庫 + 自然語言的 issue 描述。
- Agent 產出：一個 patch。
- Evaluator：套用 patch，執行該 repo 的測試套件。Patch 必須讓 FAIL_TO_PASS 測試（先前失敗、現在通過）翻轉，同時不破壞 PASS_TO_PASS 測試。

SWE-agent（Yang et al., 2024）在發表時透過強調 agent-computer interface（檔案編輯指令、模型看得懂的搜尋語法），達到 12.5%。

### SWE-bench Verified

OpenAI，2024 年 8 月。經人工整理的 500 個 task 子集。移除了含糊的 issue、不可靠的測試，以及修復方式不明確的 task。是「你的 agent 能出貨真正的 patch 嗎？」的主要 benchmark。

### 資料污染 (Contamination)

- 超過 94% 的 SWE-bench issue 早於大多數模型的訓練截止日期。
- **SWE-bench+** 發現 32.67% 的成功 patch 在 issue 文字中洩漏了解答（模型在描述中看到了修復方式），另外 31.08% 因測試覆蓋率薄弱而可疑。
- Verified 比較乾淨，但並非沒有資料污染。

實務上的意涵：在 SWE-bench 得 50% 的模型，在 SWE-bench+ 可能只得 35%。如果你宣稱 SWE-bench 的表現，務必兩者都回報。

### GAIA (Mialon et al., Nov 2023)

- 466 個問題；其中 300 個保留給 huggingface.co/gaia-benchmark 的私有排行榜。
- 設計哲學："conceptually simple for humans (92%) but hard for AI (GPT-4 with plugins: 15%)."
- 測試推理、多模態、web、tool use。
- 三個難度等級；Level 3 需要橫跨多種模態的長 tool chain。

GAIA 是你用來衡量「通才能力」的 benchmark。不要和專門針對程式碼的 benchmark 混淆。

### AgentBench (Liu et al., ICLR 2024)

- 8 個環境，涵蓋程式碼（Bash、DB、KG）、遊戲（Alfworld、LTP）、web（WebShop、Mind2Web）以及開放式生成。
- 多輪，每個 split 約 4k-13k 個 turn。
- 主要發現：長期推理、決策和遵循指令，是開源 LLM 追上商用模型的阻礙。

### 這些沒有衡量的東西 (What these do not measure)

- 真實世界的營運成本（token、wall-clock）。
- 對抗條件下的安全行為。
- 在你的領域上的表現（用你自己的 eval，Lesson 30）。
- 尾端失敗（benchmark 取平均；正式環境的維運人員在意的是最差的 1%）。

### Benchmark 哪裡會出錯 (Where benchmarking goes wrong)

- **執著於單一數字。** SWE-bench 50% 告訴你的，比不上 P50/P75/P95 的成本與步數分佈。
- **受污染的宣稱。** 回報 SWE-bench 卻不提 Verified 或 SWE-bench+，會造成誤導。
- **把 benchmark 當作開發目標。** 為 benchmark 最佳化，會偏離正式環境的實用性。

```figure
ae-swebench-gate
```

## 動手做 (Build It)

`code/main.py` 實作一個 toy 的類 SWE-bench harness：

- 合成的修 bug task（3 個 task）。
- 一個提出 patch 的腳本化「agent」。
- 一個測試執行器，檢查 FAIL_TO_PASS（bug 現在已修復）和 PASS_TO_PASS（沒有東西被破壞）。
- 一個 GAIA 風格、依問題分解深度判斷難度的分類器。

執行：

```
python3 code/main.py
```

輸出顯示每個 task 與每個難度的解決率，並讓 evaluator 規則變得具體。

## 實際使用 (Use It)

- **SWE-bench Verified** 用於程式碼 agent。一律回報 Verified 分數。
- **GAIA** 用於通才 agent。使用私有排行榜的 split。
- **AgentBench** 用於多環境比較。
- **自訂 eval**（Lesson 30）用於你產品的實際形態。

## 交付 (Ship It)

`outputs/skill-benchmark-harness.md` 為任何程式碼庫-task 配對建立 SWE-bench 風格的 harness，並以 FAIL_TO_PASS / PASS_TO_PASS 作為閘門。

## 練習 (Exercises)

1. 把 toy harness 移植到真正的 repo 上執行（挑你自己的其中一個）。為已知的 bug 寫 3 個 FAIL_TO_PASS 測試。
2. 加一個步數指標。在你的 3 個 task 上，每次解決需要幾個 agent 步驟？
3. 讀 SWE-bench+ 論文。實作解答洩漏檢查（將 issue 文字與 diff 做樣式比對）。
4. 從公開 split 下載一個 GAIA 問題。追蹤 GPT-4 等級的 agent 會怎麼做。它需要哪些 tool？
5. 讀 AgentBench 的各環境細項。哪個環境最貼近你的產品介面？那裡的「SOTA」是什麼樣子？

## 關鍵術語 (Key Terms)

| Term | 一般說法 | 實際意義 |
|------|----------------|------------------------|
| SWE-bench | 「程式碼 agent benchmark」 | 2,294 個 GitHub issue；patch 必須讓 FAIL_TO_PASS 測試翻轉 |
| SWE-bench Verified | 「乾淨版 SWE-bench」 | 500 個經人工整理的 task，OpenAI |
| FAIL_TO_PASS | 「修復閘門」 | 先前失敗、patch 後必須通過的測試 |
| PASS_TO_PASS | 「無退化閘門」 | 原本通過、之後仍須通過的測試 |
| GAIA | 「通才 benchmark」 | 466 個對人類容易 / 對 AI 困難的多 tool 問題 |
| AgentBench | 「多環境 benchmark」 | 8 個環境；長時程多輪 |
| Contamination | 「訓練集洩漏」 | Benchmark 的 task 出現在模型訓練資料中 |
| SWE-bench+ | 「資料污染稽核」 | 在成功的 SWE-bench patch 中發現 32.67% 的解答洩漏 |

## 延伸閱讀 (Further Reading)

- [Jimenez et al., SWE-bench (arXiv:2310.06770)](https://arxiv.org/abs/2310.06770)：最初的 benchmark
- [OpenAI, SWE-bench Verified](https://openai.com/index/introducing-swe-bench-verified/)：經整理的子集
- [Mialon et al., GAIA (arXiv:2311.12983)](https://arxiv.org/abs/2311.12983)：通才 benchmark
- [Liu et al., AgentBench (arXiv:2308.03688)](https://arxiv.org/abs/2308.03688)：多環境套件
