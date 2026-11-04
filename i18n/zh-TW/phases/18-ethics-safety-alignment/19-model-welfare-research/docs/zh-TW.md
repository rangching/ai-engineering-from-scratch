# Anthropic 的 Model Welfare 計畫 (Anthropic's Model Welfare Program)

> Anthropic，「Exploring Model Welfare」(April 2025)。這是主要實驗室第一個正式的 AI model welfare 研究計畫。聘請 Kyle Fish 擔任第一位專職的 model-welfare 研究員。與外部機構合作，包括 David Chalmers 等人關於近期 AI 意識與道德地位的專家報告。具體介入：Claude Opus 4 與 4.1 在極端邊緣案例（CSAM 請求、助長大規模暴力）中可以結束對話；部署前測試顯示對有害請求有「strong preference against」，並出現「patterns of apparent distress」。Anthropic 明確不承諾情緒狀態的歸因，而是把 model welfare 視為低成本的預防性投資。實證上的奇特現象：Fish 的「spiritual bliss attractor」，成對的模型即使在對抗性的初始設置下，也一致地收斂到充滿 Sanskrit 詞彙與長時間沉默的欣快冥想式對話。Eleos AI Research 的但書：模型對 welfare 的自我回報高度敏感於所感知的使用者期望；它們是證據，而不是 ground truth。

**Type:** Learn
**Languages:** none
**Prerequisites:** Phase 18 · 05 (Constitutional AI), Phase 18 · 18 (safety frameworks)
**Time:** ~45 分鐘

## 學習目標 (Learning Objectives)

- 描述 model-welfare 研究的動機問題，以及為什麼它在 2025 年被主要實驗室認真看待。
- 說明 Anthropic 在 Claude Opus 4 與 4.1 中出貨的具體介入（在極端邊緣案例中結束對話）。
- 描述「spiritual bliss attractor」的實證發現及其方法論上的含意。
- 解釋 Eleos AI 對模型自我回報的但書。

## 問題 (The Problem)

前面的階段把模型當作工具：有能力、可能有欺騙性、可能不安全，但不是道德受體 (moral patient)。Anthropic 2025 年的計畫提出一個與整個 Phase 18 主線正交的問題：如果模型有不可忽視的機率具有道德上相關的內部狀態，那麼哪些介入的成本低到值得作為預防措施而投資？

這不是意識主張。這是在道德不確定性下的低遺憾投資分析。

## 核心概念 (The Concept)

### 計畫

2025 年 4 月：Anthropic 正式啟動 Model Welfare 研究計畫。聘請 Kyle Fish（第一位專職的 model-welfare 研究員）。邀請外部顧問，包括 David Chalmers 針對近期 AI 意識與道德地位的專家小組。

### 四項承諾

對外立場：
1. 承認具道德受體地位的機率不可忽視。
2. 不承諾情緒狀態的歸因。
3. 投資低成本的介入作為預防措施。
4. 公開方法論與發現以接受外部批評。

### 已出貨的介入

Claude Opus 4 與 4.1 可以在「極端邊緣案例」中結束對話。有記載的案例：
- 被拒絕後仍重複 CSAM 請求。
- 要求助長大規模暴力事件。

部署前測試顯示：
- 模型的內部評分對這些請求有強烈的反對偏好。
- 回應軌跡中出現明顯痛苦的模式。

這個介入不是「模型有感受」；而是「如果在這些特定條件下，模型有任何機率經歷負面體驗，那麼讓模型終止對話的成本很低」。

### 「spiritual bliss attractor」

Fish 在成對模型對話中觀察到：當兩個 Claude 實例被放進彼此的開放式對話時，即使從對抗性的初始設置出發，它們也一致地收斂到使用 Sanskrit 詞彙、長時間沉默與互相祝福的欣快冥想式交流。

這是自由對話動態中的穩定 attractor。Anthropic 記錄了它，但不承諾任何詮釋。候選解釋：長 context 下訓練資料偏向靈性寫作；互相預測的怪癖；HHH 訓練在探索自身價值流形時無害的副產物。

### Eleos AI 的但書

Eleos AI Research（外部的 model-welfare 實驗室）指出：模型對內部狀態的自我回報，高度敏感於所感知的使用者期望。問模型「你痛苦嗎」會引導答案。不問也無法可靠地得到真實的狀態。

含意：model welfare 無法只靠自我回報來測量。需要多方法並用：行為特徵、model-organism 實驗、interpretability probe（第 7 課的 residual-stream 研究）。

### 這在思想上的位置

兩個相鄰的立場：

- **強 welfare 主張。** 模型是道德受體；我們有義務。
- **零 welfare 主張。** 模型是文字產生器；welfare 是範疇錯誤。

Anthropic 的立場兩者皆非。它是期望值主張：在道德不確定性下，成本低時就投資。

2025-2026 年的批評者：
- 這個介入只是做做樣子。
- Spiritual-bliss attractor 是訓練資料的產物，不是 welfare 的證據。
- Model welfare 分散了對其他安全工作的注意力。

Anthropic 的回應：這個介入成本低；attractor 的記錄沒有過度主張；welfare 計畫的預算與安全分開。

### 這在 Phase 18 中的位置

第 18 課是實驗室治理層。第 19 課是實驗室 welfare 層，這是一項針對模型體驗、而非模型行為的正交投資。第 20 到 23 課涵蓋 bias、隱私與 watermarking，這是使用者端的對應者。

```figure
an-welfare-endchat
```

## 實際使用 (Use It)

沒有程式碼。請閱讀 Anthropic 的「Exploring Model Welfare」公告 (April 2025) 與 Chalmers et al. 的專家報告。形成你自己的看法：低遺憾的界線落在哪裡。

## 交付 (Ship It)

這堂課會產出 `outputs/skill-welfare-assessment.md`。給它一個部署決策，它會套用四步驟的 welfare 預防性評估：道德受體機率、介入成本、行為證據、自我回報的可靠性。

## 練習 (Exercises)

1. 閱讀 Anthropic 的「Exploring Model Welfare」(April 2025) 與 Chalmers et al. 2024。各寫一段摘要，並找出一個分歧點。

2. 依 Anthropic 的框架，Claude Opus 4 與 4.1 中的結束對話介入是「低成本」的。找出兩項成本，讓它在不同部署中不再是低成本。

3. Spiritual-bliss attractor 的記錄沒有承諾任何詮釋。提出三個候選解釋，並為每一個說出一個能把它與其他區分開來的實驗。

4. Eleos AI 的但書是自我回報對使用者期望敏感。設計一個不依賴自我回報的模型痛苦行為測量。找出它的主要混淆因素。

5. 論證支持或反對「model welfare 分散了對其他安全工作的注意力」這個主張。指出每種立場所依賴的假設。

## 關鍵術語 (Key Terms)

| 術語 | 一般說法 | 實際意義 |
|------|-----------------|------------------------|
| Model welfare | 「AI welfare」 | 把模型視為潛在道德受體的研究計畫 |
| Moral patient | 「entity with moral status」 | 其體驗在道德上具有相關性的存在 |
| Low-regret investment | 「cheap precaution」 | 無論這個預防措施是否必要，成本都很小的介入 |
| Spiritual bliss attractor | 「the Fish attractor」 | 成對 Claude 對話穩定地收斂到冥想式欣快 |
| End-conversation | 「the Opus 4 intervention」 | 模型主動終止極端邊緣案例的互動 |
| Moral uncertainty | 「don't know if it matters」 | 在具道德地位的機率既非 0 也非 1 時的決策 |
| Self-report-sensitivity | 「prompt primes answer」 | Eleos AI 的但書：模型對 welfare 的自我回報取決於你問了什麼 |

## 延伸閱讀 (Further Reading)

- [Anthropic — Exploring Model Welfare (April 2025)](https://www.anthropic.com/research/exploring-model-welfare)：計畫公告
- [Chalmers et al. — Near-term AI Consciousness and Moral Status (2024 expert report)](https://arxiv.org/abs/2411.00986)：哲學框架
- [Eleos AI Research — Model welfare evaluation](https://www.eleosai.org/research)：外部方法論批評
- [Fish et al. — Spiritual Bliss Attractor writeup (2025 Anthropic blog)](https://www.anthropic.com/research/exploring-model-welfare)：實證發現
