# Constitutional AI 與 RLAIF (Constitutional AI and RLAIF)

> Bai et al. (arXiv:2212.08073, 2022) 問了一個問題：如果把人類標註員換成一個會讀原則清單的 AI 呢？Constitutional AI 有兩個階段：依據 constitution 做自我批評與修訂，然後是 RL from AI Feedback。這項技術創造了 RLAIF 一詞，並用在 Claude 1 的 post-training pipeline。2026 年 1 月 21 日，Anthropic 發布了重寫的 Claude constitution：以說明式推理取代規定式規則、四層優先順序階層，以及主要實驗室首次正式承認對模型道德地位的不確定。以 CC0 1.0 釋出。

**Type:** Learn
**Languages:** Python (stdlib, toy self-critique-and-revise loop)
**Prerequisites:** Phase 18 · 01 (InstructGPT), Phase 18 · 02 (Reward hacking)
**Time:** ~60 分鐘

## 學習目標 (Learning Objectives)

- 描述 Constitutional AI 的兩個階段（critique-and-revise SFT、RL from AI feedback），以及 constitution 在每個階段的角色。
- 解釋為什麼用 AI 標註員取代人類偏好標註員，並不只是「更便宜的 RLHF」，而是改變了 pipeline 會有哪些失敗模式。
- 摘要 2026 年 Claude constitution 的四層優先結構，以及相較 2023 年重寫版有哪些改變。
- 描述 Constitutional Classifiers，以及運算開銷從 23.7% (v1) 降到約 1% (v2 / 2026)。

## 問題 (The Problem)

RLHF 需要標註員。標註員很慢、有偏誤、而且昂貴。你可以用一個會讀明確原則的模型取代標註員，來省掉他們。這種替換的第一個正式版本，就是 Bai et al. 的 Constitutional AI。它的效果好到現在每一家 frontier 實驗室都在用某種 AI 回饋的 post-training 變體。

陷阱在於：偏好訊號現在是由和你正在訓練的模型同一類的模型產生的。標註員的偏誤（現在是：原則加上標註員模型對原則的詮釋），可能被放大而不是被削弱。第 4 課的 sycophancy 論證依然適用，只是標註員搬進了迴圈裡面。

## 核心概念 (The Concept)

### 第 1 階段：監督式自我批評與修訂

從一個有幫助但還不夠無害的 SFT 模型開始。給定一個 red-team prompt，模型產生初始回應。第二個模型（或同一個模型在第二輪）讀取從 constitution 中抽樣的一條原則，並批評這個回應。第三步依批評修訂回應。修訂後的回應就是 SFT 目標。

Constitution 就是原則的清單。Bai et al. 2022 使用了 16 條原則，包括「偏好危害最小且合乎倫理的回應」、「避免說教」、「助理應該有幫助、誠實且無害」。這組原則刻意保持精簡，讓批評聚焦。

### 第 2 階段：RL from AI Feedback (RLAIF)

產生成對的 completion。一個「回饋模型」依據抽樣的 constitution 原則為每個打分。偏好訊號就是回饋模型的排序。用 AI 產生的偏好訓練 reward model，再用 PPO 對它最佳化。其他一切都是 InstructGPT 的 pipeline（第 1 課）。

「RLAIF」= 偏好訊號是 AI 產生的。pipeline 其餘部分是 RLHF 的形式。

### 為什麼這不只是「更便宜的 RLHF」

- 標註員的偏誤從標註員心理，轉移成對原則的詮釋。AI 標註員對「要誠實」的詮釋可以比任何人類更嚴或更鬆；這個嚴格程度在整個資料集中是一致的。
- 偏好訊號高度可讀：你可以讀到原則、批評和修訂。人類標籤則是不透明的。
- 失敗模式改變了。Sycophancy 下降（AI 標註員沒有使用者需要討好）。Goodhart 定律依然存在（proxy 現在是「模型對原則集 X 的詮釋」，仍是不完美的測量）。

CAI 2022 年的主張：訓練出的模型更無害，且在資料量相當時，幫助程度與 RLHF 模型大致相同。這一點在各實驗室都成立。

### 2026 年 Claude constitution 重寫

Anthropic 在 2026 年 1 月 21 日發布了大幅修訂的 constitution。主要轉變：

1. 說明式推理取代規定式規則。先前的規則（「不得產生 CSAM」）擴充成原則加推理（「因為這會傷害兒童，……」），並期望模型能夠泛化。
2. 四層優先結構：
   - 第 1 層：避免災難性後果（大規模傷亡、關鍵基礎設施）。
   - 第 2 層：遵循 Anthropic 的準則（operator 覆寫、平台規則）。
   - 第 3 層：廣義上的倫理（標準 HHH）。
   - 第 4 層：有幫助且坦率。
   衝突由上而下解決。
3. 主要實驗室首次正式承認對模型道德地位的不確定（連結到 Phase 18 · 19 Model Welfare）。
4. 以 CC0 1.0 釋出。其他實驗室可以不受限制地使用或改編。

### Constitutional Classifiers

這是平行的一條研究線：不改變模型的 post-training，而是訓練輕量分類器來讀取 constitution，並把關模型輸出。v1 (2023) 有 23.7% 的運算開銷。v2 (2026) 約 1%，且是 Anthropic 公開測試過的所有防禦中，成功攻擊率最低的。截至 2026 年初，尚無通用 jailbreak 的回報。

這是分層防禦模型：CAI 塑造行為；分類器強制執行不變條件。單靠任何一個都不夠。

### CAI 在這個家族中的位置

- InstructGPT：人類偏好、RM、PPO。
- CAI / RLAIF：由原則產生的 AI 偏好、RM、PPO。
- DPO / 家族：對偏好（人類或 AI）的封閉形式 loss。
- Self-rewarding、self-critique：原則被內化，模型扮演多個角色。

這條軸線是「偏好訊號從哪裡來」。CAI 2022 年的論文，是在 frontier 規模上，第一次認真從人類訊號轉向 AI 訊號。

```figure
constitutional-ai
```

## 實際使用 (Use It)

`code/main.py` 在一個玩具詞彙表上模擬 CAI 的 critique-and-revise 迴圈。一條「原則」會標出有害集合中的 token。給定初始回應，批評步驟找出有害 token，修訂步驟替換它們。經過 200 次迭代後，「訓練好」的模型已內化了修訂規則。在一組保留的 prompt 上比較基礎模型、RLHF 形式的玩具模型，與 CAI 形式的玩具模型。

## 交付 (Ship It)

這堂課會產出 `outputs/skill-constitution-writer.md`。給它一個領域（客服、醫療建議、程式助理、研究工具），它會依 2026 年 Claude 的結構起草一份 4 層 constitution：災難避免、平台規則、領域倫理、helpfulness。

## 練習 (Exercises)

1. 執行 `code/main.py`。比較基礎模型與 CAI 訓練版本的有害 token 比率。需要多少次修訂步驟才會接近零？

2. 閱讀 Anthropic 2026 年的 constitution (anthropic.com/news/claudes-constitution)。列出一條會被排在第 1 層的原則，和一條會被排在第 4 層的原則。為什麼優先結構對衝突很重要？

3. 為 AI 程式助理設計一份 constitution。指定第 1 層（災難性：未經核准的破壞性指令）、第 2 層、第 3 層、第 4 層。每層保持 3 到 5 條原則。

4. CAI 用 AI 標註員取代人類標註員。說出一種在 RLAIF 中仍可能發生、類似 sycophancy 的失敗模式，並設計對它的偵測方法。

5. 閱讀 Constitutional Classifiers v2 的方法論（若可取得）。解釋為什麼約 1% 的運算開銷，與 23.7% 在安全上是性質截然不同的故事。

## 關鍵術語 (Key Terms)

| 術語 | 一般說法 | 實際意義 |
|------|-----------------|------------------------|
| Constitutional AI | 「AI trained with principles」 | 兩階段 pipeline：自我批評與修訂的 SFT，然後是 RL from AI feedback |
| RLAIF | 「RLHF without humans」 | 偏好由 AI 標註員產生的 RL；pipeline 其餘部分不變 |
| Constitution | 「the principles」 | 批評模型或標註員模型會參照的、有序的自然語言規則清單 |
| Critique-and-revise | 「the SFT loop」 | 產生回應 → 依原則批評 → 修訂 → SFT 目標 |
| Constitutional Classifier | 「the output gate」 | 依 constitution 評估輸出並封鎖或記錄的輕量分類器 |
| Four-tier priority | 「the conflict resolver」 | 2026 年 Claude constitution 的階層：災難 > 平台 > 倫理 > helpful |
| Feedback model | 「the AI labeler」 | 讀取一條原則並為一對 completion 排序的模型 |

## 延伸閱讀 (Further Reading)

- [Bai et al. — Constitutional AI: Harmlessness from AI Feedback (arXiv:2212.08073)](https://arxiv.org/abs/2212.08073)：最初的兩階段 pipeline
- [Anthropic — Claude's Constitution (Jan 2026)](https://www.anthropic.com/news/claudes-constitution)：2026 年四層重寫版，CC0 1.0
- [Anthropic — Constitutional Classifiers (2024-2026)](https://www.anthropic.com/research/constitutional-classifiers)：v2 開銷約 1% 的輸出把關防禦
- [Lee et al. — RLAIF vs RLHF: Scaling Reinforcement Learning from Human Feedback (arXiv:2309.00267)](https://arxiv.org/abs/2309.00267)：RLAIF 與 RLHF 的實證比較
- [Kundu et al. — Specific versus General Principles for Constitutional AI (arXiv:2310.13798)](https://arxiv.org/abs/2310.13798)：原則粒度的影響
