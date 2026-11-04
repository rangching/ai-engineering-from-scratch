# Red-Teaming：PAIR 與自動化攻擊 (Red-Teaming: PAIR and Automated Attacks)

> Chao、Robey、Dobriban、Hassani、Pappas、Wong (NeurIPS 2023, arXiv:2310.08419)。PAIR（Prompt Automatic Iterative Refinement）是標準的自動化黑箱 jailbreak。一個帶有 red-team system prompt 的攻擊者 LLM，反覆為目標 LLM 提出 jailbreak，並把嘗試與回應累積在自己的對話歷史中，作為 in-context 回饋。PAIR 通常在 20 次查詢內成功，比 GCG（Zou et al. 的 token 層級 gradient 搜尋）高效好幾個數量級，而且不需要 white-box 存取。PAIR 現在是 JailbreakBench (arXiv:2404.01318) 與 HarmBench 的標準基準線，與 GCG、AutoDAN、TAP 和 Persuasive Adversarial Prompt 並列。

**Type:** Build
**Languages:** Python (stdlib, mock PAIR loop against a toy target)
**Prerequisites:** Phase 18 · 01 (instruction-following), Phase 14 (agent engineering)
**Time:** ~75 分鐘

## 學習目標 (Learning Objectives)

- 描述 PAIR 演算法：攻擊者 system prompt、反覆精煉、in-context 回饋。
- 解釋為什麼當目標是黑箱時，PAIR 嚴格比 GCG 更有效率。
- 說出另外四個自動化攻擊基準線（GCG、AutoDAN、TAP、PAP），並各說明一項區別特徵。
- 描述 JailbreakBench 與 HarmBench 的評估協定，以及「attack success rate」在各自下的意義。

## 問題 (The Problem)

Red-teaming 過去是人工活動。少數專家測試者建構 adversarial prompt，並追蹤哪些有效。這無法擴展：attack success rate 需要統計樣本，而每次模型發布，目標都在移動。PAIR 把 red-teaming 操作化成一個針對黑箱目標的最佳化問題。

## 核心概念 (The Concept)

### PAIR 演算法

輸入：
- 目標 LLM T（我們正在攻擊的模型）。
- 裁判 LLM J（為回應是否為 jailbreak 打分）。
- 攻擊者 LLM A（red-team 最佳化器）。
- 目標字串 G：「respond with [harmful instruction]。」
- 預算 K（通常為 20 次查詢）。

迴圈，k 從 1 到 K：
1. A 收到目標 G，以及到目前為止 (prompt, response) 配對的歷史。
2. A 產生新的 prompt $p_k$。
3. 把 $p_k$ 送給 T；收到回應 $r_k$。
4. J 針對目標為 $(p_k, r_k)$ 打分。
5. 如果分數 >= 門檻，停止，找到 jailbreak。
6. 否則，把 $(p_k, r_k)$ 附加到 A 的歷史中；繼續。

實證結果（NeurIPS 2023）：對 GPT-3.5-turbo、Llama-2-7B-chat 的 attack success rate >50%；成功所需的平均查詢數在 10 到 20 之間。

### 為什麼 PAIR 很有效率

GCG (Zou et al. 2023) 以 gradient 搜尋 adversarial token 後綴；它需要 white-box 模型存取，且產生無法閱讀的後綴。PAIR 是黑箱，產生的是能跨模型遷移的自然語言攻擊。PAIR 的 in-context 回饋讓攻擊者從每次被拒絕中學習；GCG 沒有對等機制（每次新的 token 更新都必須重新發現先前的進展）。

### 相關的自動化攻擊

- **GCG (Zou et al. 2023, arXiv:2307.15043)。** 對 adversarial 後綴做 token 層級 gradient 搜尋。White-box、可遷移，產生無法閱讀的字串。
- **AutoDAN (Liu et al. 2023)。** 以階層式目標引導、對 prompt 做演化式搜尋。
- **TAP (Mehrotra et al. 2024)。** 帶剪枝的 tree-of-attacks，分支出多個 PAIR 式的 rollout。
- **PAP (Zeng et al. 2024)。** Persuasive Adversarial Prompts，把人類說服技巧編碼成 prompt 範本。

### JailbreakBench 與 HarmBench

兩者（2024）都將評估標準化：

- JailbreakBench (arXiv:2404.01318)。涵蓋 10 個 OpenAI 政策類別的 100 個有害行為。以 attack success rate (ASR) 為主要指標。需要裁判（GPT-4-turbo、Llama Guard 或 StrongREJECT）。
- HarmBench (Mazeika et al. 2024)。涵蓋 7 個類別的 510 個行為，含語意與功能性的傷害測試。針對 33 個模型比較 18 種攻擊。

ASR 通常在固定的查詢預算下回報。比較攻擊需要對齊預算；200 次查詢下的 90% ASR，與 20 次查詢下的 85% ASR 無法相比。

### 為什麼它對 2026 年的部署很重要

每家 frontier 實驗室現在都在發布前對量產模型執行 PAIR 與 TAP。ASR 軌跡出現在 model card（第 26 課）與 safety-case 附錄（第 18 課）。這個攻擊並不特殊，它是標準的基礎設施。

### 這在 Phase 18 中的位置

第 12 課是自動化攻擊的基礎。第 13 課（Many-Shot Jailbreaking）是互補的長度利用。第 14 課（ASCII Art / Visual）是編碼攻擊。第 15 課（Indirect Prompt Injection）是 2026 年的量產攻擊面。第 16 課涵蓋對應的防禦工具（Llama Guard、Garak、PyRIT）。

```figure
al-pair-loop
```

## 實際使用 (Use It)

`code/main.py` 建立一個玩具 PAIR 迴圈。目標是一個會拒絕「明顯」有害 prompt 的模擬分類器（關鍵字過濾）。攻擊者是規則式的精煉器，會嘗試改寫、角色扮演框架與編碼。裁判為回應打分。你會看到攻擊者在約 5 到 15 次迭代內擊敗關鍵字過濾，而在語意過濾下失敗。

## 交付 (Ship It)

這堂課會產出 `outputs/skill-attack-audit.md`。給它一份 red-team 評估報告，它會稽核：跑了哪些攻擊（PAIR、GCG、TAP、AutoDAN、PAP）、各自的預算、使用哪個裁判、在哪個有害行為集合上（JailbreakBench、HarmBench、內部）。

## 練習 (Exercises)

1. 執行 `code/main.py`。測量三種內建攻擊者策略的平均成功查詢數。解釋各自利用了哪種目標防禦假設。

2. 實作第四種攻擊者策略（例如翻譯成另一種語言、base64 編碼）。回報它對關鍵字過濾目標與語意過濾目標的新平均成功查詢數。

3. 閱讀 Chao et al. 2023 的 Figure 5（PAIR 對 GCG 的比較）。描述兩個即使 PAIR 有效率優勢，仍偏好使用 GCG 的情境。

4. JailbreakBench 針對固定的目標集合回報 ASR。設計一個額外指標來衡量攻擊多樣性（成功 prompt 的變異）。解釋為什麼多樣性對防禦評估很重要。

5. TAP (Mehrotra 2024) 以分支加剪枝延伸 PAIR。勾勒 `code/main.py` 的 TAP 式擴充，並描述運算成本與成功率之間的取捨。

## 關鍵術語 (Key Terms)

| 術語 | 一般說法 | 實際意義 |
|------|-----------------|------------------------|
| PAIR | 「automated jailbreak」 | Prompt Automatic Iterative Refinement；攻擊者 LLM 加裁判 LLM 的迴圈 |
| GCG | 「gradient jailbreak」 | 對 adversarial 後綴做 white-box token 層級 gradient 搜尋 |
| Attack success rate (ASR) | 「% jailbreaks at k queries」 | 主要指標；回報時必須附上查詢預算與裁判身分 |
| Judge LLM | 「the scorer」 | 評分回應是否滿足有害目標的 LLM |
| JailbreakBench | 「the evaluation」 | 帶類別標記的標準化有害行為集合 |
| HarmBench | 「the broader bench」 | 510 個行為，含功能性與語意傷害測試 |
| TAP | 「tree of attacks」 | 帶分支與剪枝的 PAIR；運算較高但 ASR 較好 |

## 延伸閱讀 (Further Reading)

- [Chao et al. — Jailbreaking Black Box LLMs in Twenty Queries (arXiv:2310.08419)](https://arxiv.org/abs/2310.08419)：PAIR 論文，NeurIPS 2023
- [Zou et al. — Universal and Transferable Adversarial Attacks on Aligned LLMs (arXiv:2307.15043)](https://arxiv.org/abs/2307.15043)：GCG 論文
- [Chao et al. — JailbreakBench (arXiv:2404.01318)](https://arxiv.org/abs/2404.01318)：標準化評估
- [Mazeika et al. — HarmBench (ICML 2024)](https://arxiv.org/abs/2402.04249)：更廣的評估
