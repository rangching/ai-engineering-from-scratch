# Theory of Mind 與湧現式協調 (Theory of Mind and Emergent Coordination)

> Li et al. (arXiv:2310.10701) 顯示，在合作式文字遊戲中的 LLM agent 會表現出 **湧現的高階 Theory of Mind** (ToM)：推理一個 agent 認為第三個 agent 相信什麼；但因為 context 管理和幻想的問題，它們在長期規劃上會失敗。Riedl (arXiv:2510.05174) 在整個群體上測量高階協同效應 (synergy)，發現 **只有** ToM-prompt 條件會產生與身分相連的分化，以及目標導向的互補性；能力較低的 LLM 只呈現虛假的湧現。也就是說，協調的湧現取決於 prompt 和模型，不是免費的。這堂課實作一個最精簡、具備 ToM 的 agent，在有和沒有 ToM prompt 的情況下執行合作任務，並依 Riedl 2025 的 protocol 測量協調的差距。

**Type:** Learn + Build
**Languages:** Python (stdlib)
**Prerequisites:** Phase 16 · 07 (Society of Mind and Debate), Phase 16 · 17 (Generative Agents)
**Time:** ~75 分鐘

## 問題 (Problem)

Multi-agent 的協調常常看起來很神奇：agent 分工、預期彼此的行動、避免重複。通常這種「湧現」是 prompt 工程的產物，有人告訴 agent 要「協調」。拿掉 prompt，協調就消失。

Riedl 在 2025 年的發現更嚴格：在受控的條件下，只有當 agent 被 prompt 去推理 **其他 agent 的心智** (ToM) 時，協調才會湧現。沒有 ToM prompt，即使是強的模型，呈現的協調 pattern 也撐不過統計控制。這對正式環境很重要：團隊上線的「multi-agent 協調」功能依賴 prompt 且很脆弱。

這堂課把 ToM 視為一種特定的能力（推理關於信念的信念），建一個最精簡、具備 ToM 的 agent，並測量真正的協調長什麼樣子，以及 prompt 包裝出來的看起來又是什麼樣子。

## 核心概念 (Concept)

### ToM 的意思 (What ToM means)

發展心理學：三歲的小孩認為每個人的內心世界都和自己一樣。五歲的小孩理解別人有不同的信念。七歲的小孩會推理關於信念的信念（「她認為我認為球在杯子下面」）。這就是零階、一階和二階 ToM。

對 LLM agent 而言，ToM 的階數對應到：

- **零階：** 沒有他人的模型。Agent 只依自己的觀察行動。
- **一階：** Agent 有每個其他 agent 的信念模型。「Alice 相信 X。」
- **二階：** Agent 對遞迴的信念建模。「Alice 相信 Bob 相信 X。」

Li et al. 2023 發現，一階和二階 ToM 會在合作遊戲中的 LLM agent 身上湧現，但在長視野和不可靠的溝通下會退化。

### Sally-Anne 測試，簡述 (The Sally-Anne test, in brief)

1985 年的錯誤信念 (false-belief) 測試：Sally 把一顆彈珠放進籃子 A，然後離開。Anne 把它移到籃子 B。Sally 回來時會去哪裡找？具備一階 ToM 的孩子會說籃子 A（Sally 的信念和現實不同）。沒有的孩子會說籃子 B。

GPT-4 世代的 LLM 在問法直白時能通過 Sally-Anne 類型的測試。當敘事很長、場景變換好幾次，或問題問得很間接時，它們就會失敗。這就是 2026 年正式環境 LLM 在 ToM 上的實際狀態。

### Riedl 的協調測量 (Riedl's coordination measurement)

Riedl (arXiv:2510.05174) 建立了群體規模的測試：N 個 agent、一個合作目標、可變的 prompt 條件。測量：

1. **與身分相連的分化 (Identity-linked differentiation)。** Agent 隨時間發展出穩定的角色區別嗎？
2. **目標導向的互補性 (Goal-directed complementarity)。** Agent 的行動是互補（不同子任務）而不是重複嗎？
3. **高階協同 (Higher-order synergy)。** 一個統計量，衡量群體是否達成了任何子集都無法達成的事。

結果：只有在 ToM prompt 條件下，這三個指標才會產生高於基準的訊號。沒有 ToM prompt，中等能力模型的指標都在機率附近徘徊。大型模型在沒有明確 ToM prompt 時會呈現一些協調，但效果比有明確 prompt 時小。

### 協調的錯覺 (The coordination illusion)

沒有統計控制的話，demo 中的「湧現式協調」常常反映的是：

- 把協調直接寫進去的 prompt 工程（system prompt 說「一起合作」）。
- 觀察者偏誤（我們看到我們期待的模式）。
- 事後挑選成功的執行結果。

行銷「湧現式協調」卻沒有可測量訊號的正式系統，應該被當成行銷話術。先測量，再宣稱。

### 最精簡的具備 ToM 的 agent (A minimal ToM-aware agent)

結構：

```
agent state:
  own_beliefs:    {facts the agent believes}
  other_models:   {other_agent_id -> {beliefs_the_agent_attributes_to_them}}
  actions_last_N: [history of others' actions]

observation update:
  - update own_beliefs from direct observation
  - update other_models[agent_id] from their action + prior beliefs

action selection:
  - enumerate candidate actions
  - for each, predict what each other agent will do next given their modeled beliefs
  - pick action that maximizes joint outcome under those predictions
```

`other_models` 屬性就是 ToM 狀態。一階 ToM 只保留一層。二階再加上 `other_models[i][other_models_of_j]`：我認為 agent i 認為 agent j 相信什麼。

### 為什麼長視野會有害 (Why long-horizon hurts)

Li et al. 記錄到：context 的限制導致 agent 忘記哪個信念屬於誰。幻想把錯誤的信念加進其他 agent 的模型。兩者都會產生「我以為他以為 X」的錯誤，並隨時間累積。

論文以及 2024-2026 年後續研究記錄的緩解方式：

- **在 prompt 中放明確的 ToM 狀態。** 結構化格式：`{agent_id: belief_list}`。強迫檢索保留身分與信念的綁定。
- **較短的推理鏈。** 每輪較少的 ToM 更新，可減少累積的幻想。
- **外部 ToM 儲存。** 把模型維護在 LLM context 之外；每輪只注入相關的部分。

### ToM 在正式環境中失敗的地方 (Where ToM fails in production)

- **對抗性的場景。** ToM 好的 agent 更容易被操弄（你可以建模它們如何建模你，再加以利用）。
- **異質團隊。** 模型不同時，對某個對手有效的 ToM 模型無法推廣。
- **依賴 ground truth 的任務。** ToM 是關於信念；如果正確性取決於事實，ToM 可能是一種干擾。

### 你真正能測量的協調 (The coordination you can actually measure)

三個實用的訊號，用來判斷團隊的協調是真的，而不是 prompt 包裝出來的：

1. **隨時間的互補性。** 在多輪任務中，agent 的行動涵蓋不相交的子任務嗎？
2. **預期。** Agent A 在第 T+1 輪的行動，取決於對 B 在第 T+2 輪行動的預測，而該預測最後是對的嗎？
3. **修正。** 當 A 在第 T 輪誤讀了 B 的信念，A 在第 T+2 輪前有修正嗎？

這些在有記錄的 multi-agent 系統中都是可測量的。它們是「協調」說法的實質版本。

```figure
sw-theory-of-mind
```

## 動手做 (Build It)

`code/main.py` 實作：

- `ToMAgent` — 追蹤自己的信念，以及對每個其他 agent 的信念模型。
- 一個合作任務：三個 agent 必須從三個箱子中收集三個 token；每個箱子只能放一個 token。Agent 不能溝通；它們從彼此的行動推斷意圖。
- 兩種設定：`zeroth_order`（沒有 ToM）和 `first_order`（有一層信念模型的 ToM）。
- 在 200 次隨機試驗上測量：完成率、重複率（兩個 agent 瞄準同一個箱子）、平均完成輪數。

執行：

```
python3 code/main.py
```

預期輸出：零階 agent 以約 35% 的比率重複工作，10 輪內完成約 60% 的試驗。一階 ToM agent 重複率約 5%，完成約 95%。這個差距就是可測量的協調效果。

## 實際使用 (Use It)

`outputs/skill-tom-auditor.md` 是一個 skill，用來稽核 multi-agent 系統對「湧現式協調」的宣稱。檢查是否有 prompt 包裝、相對於對照組的統計顯著性，以及測量到的互補性。

## 交付 (Ship It)

協調宣稱的檢查清單：

- **對照條件。** 你的系統少了協調 prompt 的版本。兩個都測量。
- **統計檢定。** 系統與對照組在你的指標上，差異在 `p < 0.05` 下顯著嗎？
- **互補性的測量。** 隨時間的行動不相交性，而不只是最終的成功。
- **失敗案例的 log。** 當 agent 協調失誤時，ToM 狀態長什麼樣子？
- **模型能力的揭露。** 如果效果在較小的模型上消失，就要說出來。

## 練習 (Exercises)

1. 執行 `code/main.py`。確認一階 ToM 把重複率降低約 7 倍。擴大到 5 個 agent 和 5 個箱子時，差距還在嗎？
2. 實作二階 ToM（agent A 對 B 認為 C 怎麼想建模）。它比一階有改善嗎？在哪些任務上？
3. 對 ToM 狀態注入 **幻想**：每輪隨機翻轉一個信念。這讓一階的表現下降多少？
4. 閱讀 Li et al. (arXiv:2310.10701)。重現「長視野退化」的發現：當輪數從 10 增加到 30，你的一階 ToM 表現有什麼變化？
5. 閱讀 Riedl 2025 (arXiv:2510.05174)。在你的模擬 log 上實作高階協同統計量。沒有 ToM prompt 條件時，這個效果存在嗎？

## 關鍵術語 (Key Terms)

| 術語 | 一般說法 | 實際意義 |
|------|----------------|------------------------|
| Theory of Mind | 「理解他人的心智」 | 對另一個 agent 的信念建模的能力。以階數分級（0、1、2+）。 |
| Sally-Anne test | 「錯誤信念測試」 | 1985 年的發展心理學；LLM 通過直白的版本，在複雜的版本上失敗。 |
| First-order ToM | 「A 相信 X」 | 對另一個人關於事實的信念建模。 |
| Second-order ToM | 「A 相信 B 相信 X」 | 再深一層的遞迴建模。 |
| Identity-linked differentiation | 「隨時間穩定的角色」 | Riedl 的指標：角色持續存在，不是隨機的。 |
| Goal-directed complementarity | 「不相交的行動」 | Agent 瞄準不同的子任務，而不是同一個。 |
| Higher-order synergy | 「群體超越任何子集」 | Riedl 用來衡量真正協調的統計量。 |
| Coordination illusion | 「看起來有協調」 | 用 prompt 包裝出來、沒有可測量訊號的協調外觀。 |

## 延伸閱讀 (Further Reading)

- [Li et al. — Theory of Mind for Multi-Agent Collaboration via Large Language Models](https://arxiv.org/abs/2310.10701) — 合作遊戲中湧現的 ToM；長視野的失敗模式
- [Riedl — Emergent Coordination in Multi-Agent Language Models](https://arxiv.org/abs/2510.05174) — 群體規模的測量；ToM prompting 是承重的條件
- [Premack & Woodruff — Does the chimpanzee have a theory of mind?](https://doi.org/10.1017/S0140525X00076512) — 1978 年 ToM 概念的起源
- [Baron-Cohen, Leslie, Frith — Does the autistic child have a theory of mind?](https://doi.org/10.1016/0010-0277(85)90022-8) — Sally-Anne 論文（1985）
