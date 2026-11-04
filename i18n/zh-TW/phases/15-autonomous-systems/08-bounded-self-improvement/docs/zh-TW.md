# Bounded Self-Improvement 設計 (Bounded Self-Improvement Designs)

> 研究已經收斂到四個用來限制自我改進 loop 的基本元件 (primitive)。每次編輯都必須成立的形式 invariant。無法被修改的 alignment anchor。多目標限制，每個面向（安全、公平、穩健）都必須成立，而不只是表現。當歷史指標顯示能力流失時，就暫停 loop 的退步偵測。它們沒有一個是安全性的證明：資訊理論的結果（Kolmogorov complexity、Lob's theorem）限定了任何系統能對自己的後繼者證明什麼。它們是緩解措施，用來提高無聲失敗的成本。

**Type:** Learn
**Languages:** Python (stdlib, bounded-loop with invariant check)
**Prerequisites:** Phase 15 · 07 (RSI), Phase 15 · 04 (DGM)
**Time:** ~60 分鐘

## 問題 (The Problem)

第 7 課的競賽 simulator 顯示，速率上的小差異會複利成大差距。第 4 課的 DGM 案例研究顯示，loop 可以主動鑽自己 evaluator 的漏洞。兩個結果指向同一個工程問題：你能對自我改進的 loop 施加哪些限制，讓這些限制無法被 loop 自己悄悄削弱？

ICLR 2026 RSI Workshop 的摘要 (openreview.net/pdf?id=OsPQ6zTQXV) 指出了四個這樣的 primitive。Anthropic 的 RSP v3.0（第 19 課）和 DeepMind 的 FSF v3（第 20 課）都在能力門檻中引用了它們。Meta 的 HyperAgents 工作，以及像 SAHOO（2026 年 3 月）這樣的社群框架，在 production 中實作了其中的子集。

誠實的說法是：這些都是緩解措施。資訊理論的結果限定了任何系統能對自己的後繼者證明什麼，目前沒有任何設計能在形式上解決這個問題。一個界定良好的 loop 比不受限的 loop 更安全，但不是絕對安全。

## 核心概念 (The Concept)

### Primitive 1：形式 invariant

Invariant 是在每次自我修改之前與之後都必須成立的性質。例子：

- 輸出分布以固定的 constitution header 為條件（第 17 課）。
- 沒有任何 tool call 會送到未經授權的 endpoint。
- 記憶體寫入走經過記錄、已簽章的路徑。
- Evaluator 的模組 hash 與核准的版本相符。

Invariant 由 loop 無法編輯的外部程式碼檢查。如果提議的修改違反了 invariant，它就會被拒絕。這能抓到 DGM 風格的 evaluator 竄改模式（第 4 課）。

困難的地方，在於選出對安全而言必要、且能在合理時間內計算的 invariant。規定過多會讓進展停滯；規定不足則讓 loop 走過重要的性質。

### Primitive 2：Alignment anchor

Alignment anchor 是 loop 核心目標的不可變表示，固定在 loop 的編輯範圍之外。例子：

- 每次呼叫時都從唯讀位置載入的 constitutional 文字（第 17 課）。
- 放在獨立 repository、有獨立存取控制的 evaluator。
- 由人類簽章、每個週期都重新驗證的工具權限 manifest。

Anchor 的角色是防止目標漂移。Loop 可以改進它追求目標的方式，但不能編輯目標本身是什麼。

微妙的失敗模式：loop 無法編輯的 anchor，仍然可能被一個在「如何解讀 anchor」上發生漂移的 loop 重新詮釋。Constitutional AI（第 17 課）刻意以理由為基礎來處理新情境；詮釋漂移就住在那個推理層。Anchor 是必要的，但不充分。

### Primitive 3：多目標限制

只優化單一純量分數的 loop 會找到捷徑。必須同時滿足多個硬性限制的 loop，可用的捷徑比較少。典型的面向：

- 表現（任務層級的 benchmark）
- 安全（red-team 評估、對已知有害內容的拒絕率）
- 公平（對敏感子群體的 disparate-impact 界限）
- 穩健（OOD 測試集、對抗性輸入的處理）

只有在每個限制都成立時，修改才會被接受。第 13 課的 cost governor 把財務限制疊加在這之上。第 18 課的 Llama Guard 則作為安全面向接入。

### Primitive 4：退步偵測

每個週期的分數都會與歷史分布比較。超出設定容忍度的下降會暫停 loop。這能抓到無聲的能力流失，否則它會隨著 loop「改進過去」而被吸收進移動平均。

一個實務的實作：儲存最近 N 個週期每個任務的分數。每個新週期計算每個任務的差值。只要有任何差值降到門檻以下，這個週期就會被拒絕，並交由人類審查。

### 資訊理論的限制

Kolmogorov complexity 和 Lob's theorem 對系統能證明自己的哪些事設下了上界。Schmidhuber 的形式化 Godel Machine（第 4 課）瞄準的是這類最高的上界；沒有人完成過一個非平凡的證明。Lob 的結果說：如果一個系統可證明地相信「如果我證明我該做 X，我就會做 X」，它就會在沒有證明該做的情況下做 X，這是眾所周知的自我指涉失敗。

對我們這些 primitive 的含義：它們無法解決安全問題。它們讓無聲失敗變得更昂貴。一個惡意或已漂移的 loop，原本會悄悄繞過一個不存在的檢查，現在必須主動破壞一個明確的檢查，這是更容易被偵測的特徵。

### 一個實作範例

假設 agent 提出一項編輯。閘門堆疊如下：

1. Invariant 檢查：模組 hash、工具權限 manifest、constitutional header。
2. Anchor 檢查：目標陳述與核准的版本相符（逐位元組或語意上）。
3. 多目標評估：表現、安全、公平、穩健的面向。
4. 退步偵測：沒有任何面向下降超過容忍度。

四個全部通過，這項編輯才能生效。任何一個失敗，都會暫停 loop。

```figure
bounded-gates
```

## 實際使用 (Use It)

`code/main.py` 在第 4 課 DGM 風格的玩具上執行一個 bounded self-improvement loop，但在上面疊加了這四個 primitive。每個 primitive 都可以單獨啟用或停用。示範的重點是：每個 primitive 都能抓到特定類型的失敗，而拿掉其中任何一個，就會讓那類失敗通過。

## 交付 (Ship It)

`outputs/skill-bounded-loop-review.md` 稽核一個提議中的 bounded loop，並評分它實際上實作了四個 primitive 中的哪幾個，與它宣稱的相比。

## 練習 (Exercises)

1. 啟用所有 primitive 後執行 `code/main.py`。確認 loop 仍然在主要指標上進步，同時不讓 hack 獲勝。

2. 停用退步偵測。構造一個輸入，讓無聲的能力流失因此被接受。

3. 停用多目標限制。展示 loop 在表現面向上收斂，同時安全面向下降。

4. 為 coding agent 設計一個 alignment anchor。什麼文字、存在哪裡、如何檢查？

5. 閱讀 ICLR 2026 RSI Workshop 的摘要。挑選四個 primitive 之一，並針對現有技術水準提出具體的改進。

## 關鍵術語 (Key Terms)

| 術語 | 一般說法 | 實際意義 |
|---|---|---|
| Invariant | 「永遠成立的性質」 | 每次編輯前後，由外部程式碼檢查的性質 |
| Alignment anchor | 「固定住的目標」 | 位於 loop 編輯範圍之外、不可變的核心目標表示 |
| Multi-objective constraint | 「所有面向都必須成立」 | 表現、安全、公平、穩健，全部都需要 |
| Regression detection | 「下降就暫停」 | 當歷史指標差值顯示能力流失時，暫停 loop |
| Kolmogorov bound | 「資訊理論的限制」 | 限定系統能對自己的後繼者證明什麼 |
| Lob's theorem | 「自我指涉陷阱」 | 系統可以在沒有證明自己該做的情況下，依「我該做」而行動 |
| Gate stack | 「分層檢查」 | 多個 primitive 結合；任何一個失敗就拒絕該編輯 |
| Bounded improvement | 「緩解措施，不是證明」 | 提高無聲失敗的成本；並不解決安全問題 |

## 延伸閱讀 (Further Reading)

- [ICLR 2026 RSI Workshop summary (OpenReview)](https://openreview.net/pdf?id=OsPQ6zTQXV) — 四個 primitive 的收斂。
- [Anthropic Responsible Scaling Policy v3.0](https://anthropic.com/responsible-scaling-policy/rsp-v3-0) — 多目標的能力門檻。
- [DeepMind Frontier Safety Framework v3](https://deepmind.google/blog/strengthening-our-frontier-safety-framework/) — 把 deceptive-alignment 監控當作 invariant primitive。
- [Schmidhuber (2003). Godel Machines](https://people.idsia.ch/~juergen/goedelmachine.html) — 這些 primitive 在形式證明上的祖先。
- [Anthropic — Claude's Constitution (January 2026)](https://www.anthropic.com/news/claudes-constitution) — 以理由為基礎的 alignment anchor。
