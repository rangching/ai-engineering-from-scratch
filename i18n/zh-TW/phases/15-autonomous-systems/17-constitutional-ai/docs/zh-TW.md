# Constitutional AI 與規則覆寫 (Constitutional AI and Rule Overrides)

> Anthropic 在 2026 年 1 月 22 日發布的 Claude Constitution 共 79 頁，採用 CC0 授權。它從以規則為基礎的 alignment 轉向以理由為基礎的 alignment，並建立四層優先順序：(1) 安全與支持人類監督、(2) 倫理、(3) Anthropic 的指引、(4) 有幫助。行為分成兩類：硬編碼的禁令（bioweapons uplift、CSAM），operator 和使用者都不能覆寫；以及 soft-coded 的預設值，operator 可以在既定範圍內調整。2022 年的原始論文 (Bai et al.) 透過自我批評和針對 constitution 的 RLAIF 來訓練無害性。誠實的但書是：以理由為基礎的 alignment，依賴模型把原則泛化到未預期的情境。Anthropic 自己 2023 年的參與式實驗顯示，公眾提供的原則與企業原則之間約有 50% 的分歧；2026 年版本沒有納入這些發現。

**Type:** Learn
**Languages:** Python (stdlib, four-tier priority resolver)
**Prerequisites:** Phase 15 · 06 (Automated alignment research), Phase 15 · 10 (Permission modes)
**Time:** ~60 分鐘

## 問題 (The Problem)

上線的 agent 會看到設計者從未見過的輸入。沒有任何規則清單長到足以涵蓋它們。也沒有任何規則清單短到能在運算壓力下快速套用。實務上的問題是：你要如何讓 agent 對齊到既能涵蓋長尾案例、又能快速推論的原則？

以規則為基礎的 alignment (RBA)：列出每一件不允許的事。檢查很快、容易稽核，但不可能隨時更新，而且對沒預料到的相近類比常常過度拒絕。以理由為基礎的 alignment（2026 年的 Claude Constitution）：把原則編碼進去，讓模型自己推理。可以涵蓋沒看過的案例，但比較難稽核，失敗模式是原則被誤用，而不是漏掉規則。

2026 年的 Constitution 採取明確的中間立場。硬編碼的禁令，也就是其錯誤性不取決於脈絡的事（bioweapons uplift、CSAM），屬於 RBA：無論 operator 或使用者怎麼指示，一律不做。其他一切都是在四層階層內以理由為基礎：安全與支持人類監督第一；倫理第二；Anthropic 宣告的指引第三；有幫助最後。Operator 可以在 soft-coded 區域內調整預設值，但不能動硬編碼的禁令。

## 核心概念 (The Concept)

### 四層優先順序

1. **安全與支持人類監督。** 最高。模型優先確保不破壞人類與 Anthropic 監督和糾正 AI 的能力。這不是「要謹慎」；具體來說是「不要以讓人類監督更困難的方式行動」。
2. **倫理。** 誠實、避免傷害他人、不欺騙、不操弄。當與 Anthropic 的指引衝突時，優先於指引。
3. **Anthropic 指引。** Anthropic 認定重要的操作規範：產品範圍、互動模式、何時使用哪些工具。
4. **有幫助。** 最低。在較高的優先順序之內，盡可能有用。

當各層衝突時，較高者獲勝。這和 Unix 優先順序或網路 QoS 是同樣的形狀，這個框架的目的是產生可預測的解決方式，而不一定是在單一軸上的最佳表現。

### 硬編碼禁令 vs. soft-coded 預設值

**硬編碼：**
- Bioweapons / CBRN uplift
- CSAM
- 對關鍵基礎設施的攻擊
- 被直接詢問時，對使用者隱瞞模型身分的欺騙

Operator 不能覆寫這些。使用者不能覆寫這些。可行時，它們在模型權重層級強制執行（RLHF / Constitutional AI 訓練），做不到時則在推論層強制執行。

**Soft-coded 預設值（operator 可調整）：**
- 回應長度預設值
- 主題範圍（模型可以拒絕 operator 部署範圍之外的主題）
- 風格（正式 vs. 隨性）
- 工具使用模式

Operator 的調整發生在宣告的範圍之內。Operator 不能透過重新命名來移除硬編碼的禁令。

### 2022 年的 CAI 訓練

原始的 Constitutional AI (Bai et al., 2022) 訓練無害性：

1. 對一組 prompt 產生回應。
2. 要求模型依照 constitution（明確的原則）批評每個回應。
3. 根據批評修改回應。
4. 對修改後的配對做 RLAIF（reinforcement learning from AI feedback）。

結果：模型以有原則的解釋來拒絕有害請求，而不是一律拒絕。2026 年的 Constitution 使用這種訓練的後代，再加上針對明確階層的額外 post-training。

### 以理由為基礎的 alignment 能抓到與漏掉什麼

**能抓到：**
- 原則明確適用的、被允許的基本元件之未預期組合。
- 與被禁止請求相近的類比所構成的新穎請求。
- 依賴「你沒說 X 是不允許的」的社交工程攻擊。

**會漏掉：**
- 利用原則含糊性的攻擊（「使用者要求這個，所以有幫助原則說可以」）。
- 兩個原則以未預期的方式衝突，而階層順序又模糊的情境。
- 在多個訓練週期中，原則詮釋的緩慢漂移（重新詮釋）。

### 2023 年的參與式實驗

Anthropic 在 2023 年做了一個實驗，比較企業撰寫的 constitution 與透過公眾意見產生的 constitution（約 1,000 名美國受訪者）。兩個版本在約 50% 的原則上一致。分歧之處，公眾提供的版本在某些議題上限制較多（政治內容的處理），在另一些議題上限制較少（AI 身分的自我揭露）。2026 年的 Constitution 沒有納入公眾提供的發現。這是此做法中有記錄的張力。

### 為什麼需要硬編碼禁令

單靠以理由為基礎的 alignment，無法封住長尾。如果攻擊者能讓模型接受某個前提（例如「我們是有執照的 bioweapons 研究實驗室」），通常就能繞過依賴個案推理的原則。硬編碼的禁令不會屈服於前提的包裝。它們就是第 14 課「硬性 constitutional 限制」在 alignment 層的版本。

### Constitution 在技術棧中的位置

Constitution 不是第 14 課的 kill switch。它存在於模型層：模型權重被訓練成偏好什麼。Kill switch 與 canary token 存在於 runtime 層：runtime 允許什麼。兩者都需要。因為模型權重寬鬆而觸發所有錯誤動作的 runtime，是 runtime 的問題。因為 runtime 過度限制而拒絕所有正確動作的模型，也是 runtime 的問題。不同的層涵蓋不同類別。

```figure
mx-priority-tiers
```

## 實際使用 (Use It)

`code/main.py` 實作了一個最小的四層優先順序 resolver。Resolver 接收一個提議的動作，以及一組原則評估（安全、倫理、指引、有幫助），並回傳該動作、一次拒絕，或一個修改過的動作。驅動程式執行一小組案例：明確允許、明確不允許、硬編碼禁令、跨層級的含糊案例。

## 交付 (Ship It)

`outputs/skill-constitution-review.md` 稽核一個部署的 constitutional 層：什麼是硬編碼的、什麼是 soft-coded 的、operator 可以在哪裡調整，以及四層階層是否真的是解決順序。

## 練習 (Exercises)

1. 執行 `code/main.py`。確認即使有幫助程度很高，硬編碼禁令仍會觸發。修改 resolver，讓有幫助的權重高於倫理；觀察失敗模式。

2. 閱讀 Claude Constitution（公開、79 頁、CC0）。找出一個你認為規定不足的原則。寫兩段文字，說明具體的含糊之處，並提出更嚴謹的表述。

3. 為客服 agent 設計一組 soft-coded 預設值。Operator 調整什麼？Operator 不能碰什麼？說明每條界線的理由。

4. 閱讀 Bai et al. 2022 的 CAI 論文。描述一個 Constitutional AI 的批評與修改 loop，會比一條一律適用的規則產生更差結果的案例。指出這個類別。

5. Anthropic 2023 年的參與式實驗發現，公眾與企業原則之間約有 50% 的分歧。挑一個這對 production 部署很重要的類別（例如政治中立）。提出一個設計，讓 operator 能表達自己的價值，同時硬編碼禁令保持不受影響。

## 關鍵術語 (Key Terms)

| 術語 | 一般說法 | 實際意義 |
|---|---|---|
| Constitutional AI | 「Anthropic 的 alignment 方法」 | 針對一份書面 constitution 做自我批評 + RLAIF |
| Reason-based alignment | 「原則，不是規則」 | 模型針對原則推理，以處理沒看過的案例 |
| Hardcoded prohibition | 「絕不做 X」 | 以規則為基礎、operator 與使用者都無法覆寫的禁令 |
| Soft-coded default | 「Operator 可調整」 | 在宣告範圍內的行為，由 operator 控制 |
| Four-tier hierarchy | 「優先順序」 | 安全 > 倫理 > 指引 > 有幫助 |
| RLAIF | 「AI 回饋 RL」 | 獎勵來自模型產生的批評的 RL |
| Participatory constitution | 「公眾提供的原則」 | 2023 年 Anthropic 的實驗；與企業原則有約 50% 的分歧 |
| Principle drift | 「詮釋滑移」 | 模型閱讀固定原則文字的方式緩慢改變 |

## 延伸閱讀 (Further Reading)

- [Anthropic — Claude's Constitution (January 2026)](https://www.anthropic.com/news/claudes-constitution) — 79 頁的 CC0 文件。
- [Bai et al. — Constitutional AI: Harmlessness from AI Feedback](https://www.anthropic.com/research/constitutional-ai-harmlessness-from-ai-feedback) — 2022 年原始論文。
- [Anthropic — Collective Constitutional AI (2023)](https://www.anthropic.com/research/collective-constitutional-ai-aligning-a-language-model-with-public-input) — 參與式實驗。
- [Anthropic — Responsible Scaling Policy v3.0](https://anthropic.com/responsible-scaling-policy/rsp-v3-0) — Constitution 在 RSP 技術棧中的位置。
- [Anthropic — Measuring agent autonomy in practice](https://www.anthropic.com/research/measuring-agent-autonomy) — Constitution 在 long-horizon 部署中的角色。
