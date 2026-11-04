# 自主 Coding Agent 全景 (The Autonomous Coding Agent Landscape, 2026)

> SWE-bench Verified 在不到三年內，從 4% 升到 80.9%。同一個 Claude Sonnet 4.5，在 SWE-agent v1 上得 43.2%，在 Cline autonomous 上得 59.8%，現在模型外圍的 scaffolding 和模型本身一樣重要。OpenHands（前身為 OpenDevin）是最活躍的 MIT 授權平台，它的 CodeAct loop 直接在 sandbox 中執行 Python action，而不是 JSON tool call。這些亮眼的數字背後藏著一個方法論問題：500 個 SWE-bench Verified 任務中，有 161 個只需要改 1 到 2 行，而 SWE-bench Pro（10 行以上的任務）上，同樣的 frontier model 只有 23–59%。

**Type:** Learn
**Languages:** Python (stdlib, CodeAct vs JSON tool-call comparison)
**Prerequisites:** Phase 14 · 07 (Tool use), Phase 15 · 01 (Long-horizon agents)
**Time:** ~45 分鐘

## 問題 (The Problem)

「哪個 coding agent 最好」是錯的問題。對的問題是：在符合我工作的任務分布上，用我在 production 會跑的 scaffolding，我能得到多少端到端的可靠度？

從 2022 到 2026 年，這個領域學到：scaffolding，也就是檢索層、planner、sandbox、編輯與驗證 loop、回饋格式，是承重結構。Claude Sonnet 4.5 在 SWE-agent v1 上的 SWE-bench Verified 得分是 43.2%；同一個模型放進 Cline 的 autonomous scaffold 得到 59.8%。絕對差距 16.6 個百分點，權重完全相同。Base model 是一個元件；loop 才是產品。

伴隨而來的問題是，benchmark 飽和會隱藏退步。SWE-bench Verified 已接近飽和，簡單任務的長尾（500 個任務中有 161 個只需改動 2 行以內）把頂尖分數拉高。真實世界的品質，比較適合用像 SWE-bench Pro（10 行以上的改動）這樣的分布來衡量，在那裡同樣的領先者仍然只有 23–59%。

## 核心概念 (The Concept)

### 一段話說明 SWE-bench

SWE-bench (Jimenez et al.) 取真實的 GitHub issue 與其正確 patch，要求 agent 產生能讓測試套件通過的 patch。SWE-bench Verified (OpenAI, 2024) 是人工策展的 500 個任務子集，移除了含糊與有問題的任務。SWE-bench Pro 是更難的後繼者，任務需要 10 行以上的改動，目前的 frontier agent 在 23–59% 之間。

### 2022 → 2026 曲線實際顯示了什麼

- **2022**：研究模型在原始 SWE-bench 上約 4%。
- **2024**：GPT-4 + Devin 風格的 scaffolding 約 14%；SWE-agent 約 12%。
- **2025**：Claude 3.5/3.7 Sonnet 搭配 Aider 與 SWE-agent，推進到 40–55% 區間。
- **2026**：Claude Sonnet 4.5 與 frontier 競爭者在 SWE-bench Verified 上達到 70–80% 以上。Epoch AI 的排行榜即時追蹤這個數字。

這條斜率來自三個複利的來源：更好的 base model、更好的 scaffolding（CodeAct、reflection、verifier loop），以及更好的 benchmark（Verified 移除了雜訊）。

### CodeAct vs. JSON tool call

OpenHands (All-Hands-AI, arXiv:2407.16741，前身為 OpenDevin) 做了一個特定的架構賭注：不讓模型輸出 JSON tool call 由主機解碼並執行，而是讓模型輸出 Python 程式碼，由 Jupyter 風格的 kernel 在 sandbox 中執行。Agent 可以在一次 action 內迴圈處理檔案、串接工具，並捕捉自己的例外。

取捨：

- **JSON tool call**：每個 action 是一輪；容易稽核；組合性有限；預設就安全，因為每次呼叫都會通過明確的 validator。
- **CodeAct**：一個 action 可以是一整支程式；組合性強；需要強化的 sandbox（OpenHands 使用 Docker 隔離）；失敗模式包含 sandbox runtime 允許的任何事。

兩種架構都在 production 中。CodeAct 在開放平台（OpenHands、smolagents）中佔主導。JSON tool call 在由供應商控制執行器的託管服務（Anthropic Managed Agents、OpenAI Assistants）中仍然佔主導。

### 2026 年全景中的 scaffold

| Scaffold | 授權 | 執行模型 | 值得注意的特性 |
|---|---|---|---|
| OpenHands (OpenDevin) | MIT | Docker 中的 CodeAct | 最活躍的開放平台；event stream 可重播 |
| SWE-agent | MIT | Agent-Computer Interface (ACI) | 第一個端到端的 SWE-bench scaffold |
| Aider | Apache-2 | 在本地 repo 中以 diff 編輯 | 極簡 scaffold，退步穩定性強 |
| Cline | Apache-2 | 帶有工具政策的 VS Code agent | 在 Sonnet 4.5 上得分最高的開放 scaffold |
| Devin (Cognition) | 專有 | 託管 VM + planner | 第一個「AI 軟體工程師」產品類別 |
| Claude Code | 專有 | 權限模式 + routine | 第 10 課詳細說明 agent loop |

### 為什麼 scaffolding 佔主導

一次 coding 執行是一條 long-horizon 的 trajectory（第 1 課）。可靠度會在各步驟之間複利。Scaffolding 在三個地方能換到分數：

1. **檢索**：找到該讀的檔案，是沉默的瓶頸。SWE-agent 的 ACI、OpenHands 的 file-index 和 Aider 的 repo-map 都在處理這個問題。
2. **Verifier loop**：執行測試、讀 stack trace、重新嘗試，在 SWE-bench 上有 10 個百分點以上的差距。
3. **失敗圍堵**：出錯時能 rollback 的 sandbox，可以防止損害複利擴大。同一個模型有沒有 verifier loop，看起來像兩個不同的產品。

### Benchmark 飽和與真實分布

OpenHands 的作者和 Epoch AI 都指出，SWE-bench Verified 有一條簡單的長尾：500 個任務中有 161 個只需要改 1–2 行。高分有一部分是被這條長尾推高的。SWE-bench Pro 限定在 10 行以上的改動，即使對 frontier 系統，分數也落在 23–59%。你的 production 分布幾乎肯定更接近 Pro，而不是 Verified。

對選擇 agent 的啟示：在你自己 bug backlog 中，挑出類似 Pro 的子集來跑。真正重要的分數，是在能代表你所交付內容的任務上的分數。

```figure
a5-scaffold-delta
```

## 實際使用 (Use It)

`code/main.py` 在固定的小型任務分布上，比較兩種玩具 agent scaffold：

1. 每輪只採取一個 action 的 **JSON tool-call** scaffold。
2. 每個 action 可以輸出一小段 Python 程式碼片段的 **CodeAct** scaffold。

兩者都使用一個 stub「模型」（確定性規則），讓比較能把 scaffold 與模型品質分開。輸出顯示，CodeAct scaffold 用更少的輪數解決更多任務，代價是每個 action 的 blast radius 更大。

## 交付 (Ship It)

`outputs/skill-scaffold-audit.md` 幫你在採用之前，稽核一個提議中的 coding-agent scaffold：檢索品質、verifier 是否存在、sandbox 隔離，以及 benchmark 與你的分布是否吻合。

## 練習 (Exercises)

1. 執行 `code/main.py`。兩種 scaffold 在同一組任務上各花幾輪？各自每個 action 的 blast radius 是什麼？

2. 閱讀 OpenHands 論文 (arXiv:2407.16741)。論文主張 CodeAct 在複雜任務上勝過 JSON tool call。找出論文承認的一個失敗模式，並用一句話寫出該模式在 production 中何時會佔主導。

3. 從你的 bug backlog 挑一個需要跨兩個檔案改動 10 行以上的任務。估計 frontier 模型在 (a) JSON tool call 與 (b) CodeAct 下的端到端成功機率。說明差距的理由。

4. SWE-bench Verified 有 161 個單檔、1–2 行的任務。建構一個排除它們的分數。排行榜如何洗牌？

5. 閱讀「Introducing SWE-bench Verified」(OpenAI)。說明用來移除含糊任務的具體方法論，並指出一個策展會漏掉的類別。

## 關鍵術語 (Key Terms)

| 術語 | 一般說法 | 實際意義 |
|---|---|---|
| SWE-bench | 「Coding benchmark」 | 真實的 GitHub issue，附有正確 patch 與測試套件 |
| SWE-bench Verified | 「清理過的子集」 | 500 個人工策展的任務，仍存在簡單的長尾 |
| SWE-bench Pro | 「較難的子集」 | 10 行以上的改動；frontier 落在 23–59% |
| CodeAct | 「把程式碼當作 action」 | Agent 輸出 Python；Jupyter 風格的 kernel 在 sandbox 中執行 |
| JSON tool call | 「Function calling」 | 每個 action 是結構化的 JSON payload，執行前先驗證 |
| Scaffold | 「Agent 框架」 | 圍繞 base model 的檢索 + planner + 執行器 + verifier loop |
| ACI (Agent-Computer Interface) | 「SWE-agent 的格式」 | 為 LLM 的使用習慣設計的指令集，而不是給人用的 shell |
| Verifier loop | 「測試後重試」 | 執行測試、讀輸出、修改 patch；模型之外最大的可靠度增益 |

## 延伸閱讀 (Further Reading)

- [Jimenez et al. — SWE-bench](https://www.swebench.com/) — 原始 benchmark 與方法論。
- [OpenAI — Introducing SWE-bench Verified](https://openai.com/index/introducing-swe-bench-verified/) — 策展子集是如何建立的。
- [Wang et al. — OpenHands: An Open Platform for AI Software Developers](https://arxiv.org/abs/2407.16741) — CodeAct 架構與 event-stream 設計。
- [Epoch AI — SWE-bench leaderboard](https://epoch.ai/benchmarks) — 即時追蹤的分數。
- [Anthropic — Measuring agent autonomy](https://www.anthropic.com/research/measuring-agent-autonomy) — long-horizon coding agent 可靠度的框架。
