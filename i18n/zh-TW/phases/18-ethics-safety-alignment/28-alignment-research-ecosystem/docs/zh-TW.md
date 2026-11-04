# Alignment 研究生態系：MATS、Redwood、Apollo、METR (Alignment Research Ecosystem — MATS, Redwood, Apollo, METR)

> 五個組織定義了 2026 年實驗室之外的 alignment 研究層。MATS (ML Alignment & Theory Scholars)：自 2021 年底以來有 527+ 位研究者、180+ 篇論文、10K+ 引用、h-index 47；2024 年夏季的梯次已登記為 501(c)(3)，約 90 位 scholar 與 40 位 mentor；2025 年以前的校友中 80% 從事安全 / 資安工作，其中 200+ 人在 Anthropic、DeepMind、OpenAI、UK AISI、RAND、Redwood、METR、Apollo。Redwood Research：由 Buck Shlegeris 創立的應用型 alignment 實驗室；提出了 AI Control（第 10 課）；與 UK AISI 合作 control safety case。Apollo Research：為 frontier 實驗室做部署前的 scheming 評估；撰寫 In-Context Scheming（第 8 課）與 Towards Safety Cases for AI Scheming。METR (Model Evaluation and Threat Research)：以任務為基礎的能力評估、自主任務時間範圍的研究；「Common Elements of Frontier AI Safety Policies」比較各實驗室的框架。Eleos AI Research：model-welfare 的部署前評估（第 19 課）；進行了 Claude Opus 4 的 welfare 評估。

**Type:** Learn
**Languages:** none
**Prerequisites:** Phase 18 · 01-27 (prior Phase 18 lessons)
**Time:** ~45 分鐘

## 學習目標 (Learning Objectives)

- 辨識非實驗室 alignment 研究生態系的五個組織及其核心產出。
- 描述 MATS 的規模（scholar、論文、h-index）及其作為人才管道的角色。
- 描述 Redwood 的 AI Control 議程及其與 UK AISI 的合作。
- 描述 METR 以任務為基礎的評估方法論。

## 問題 (The Problem)

Frontier 實驗室（第 18 課）在內部產出安全評估，並發表挑選過的結果。實驗室之外的生態系，才是評估被驗證、新的失敗模式最先被發現、人才被培養的地方。理解這個生態系，有助於判斷哪些研究發現被誰信任。

## 核心概念 (The Concept)

### MATS (ML Alignment & Theory Scholars)

2021 年底開始。研究指導計畫；scholar 花 10 到 12 週，與資深研究者一起處理特定的 alignment 問題。

規模（2026）：
- 自創立以來有 527+ 位研究者。
- 已發表 180+ 篇論文。
- 10K+ 引用。
- h-index 47。
- 2024 年夏季：90 位 scholar + 40 位 mentor；登記為 501(c)(3)。

職涯成果：2025 年以前的校友中，約 80% 從事安全 / 資安工作。200+ 人在 Anthropic、DeepMind、OpenAI、UK AISI、RAND、Redwood、METR、Apollo。

### Redwood Research

應用型 alignment 實驗室。由 Buck Shlegeris 創立。提出 AI Control 議程（第 10 課）。與 UK AISI 合作 control safety case。為 DeepMind 與 Anthropic 提供評估設計的諮詢。

代表性論文：Greenblatt、Shlegeris et al.，「AI Control」(arXiv:2312.06942, ICML 2024)；Alignment Faking (Greenblatt、Denison、Wright et al., arXiv:2412.14093，與 Anthropic 合作)。

風格：具體的威脅模型、最壞情況的對手、可以被壓力測試的具體協定。

### Apollo Research

為 frontier 實驗室做部署前的 scheming 評估。撰寫了 In-Context Scheming（第 8 課，arXiv:2412.04984）。是 2025 年 OpenAI anti-scheming 訓練合作的夥伴。產出 Towards Safety Cases for AI Scheming (2024)。

風格：在欺騙可能浮現的 agentic 情境中做評估；三根支柱的拆解（misalignment、goal-directedness、situational awareness）。

### METR (Model Evaluation and Threat Research)

以任務為基礎的能力評估。自主任務完成時間範圍的研究。「Common Elements of Frontier AI Safety Policies」(metr.org/common-elements, 2025) 比較各實驗室的框架。

與 Apollo 共同撰寫 AI Scheming safety-case 草案。

風格：長時間範圍的任務評估、實證的能力測量、框架綜合。

### Eleos AI Research

Model-welfare 的部署前評估。進行了記載於 system card 第 5.3 節的 Claude Opus 4 welfare 評估。為第 19 課與 welfare 相關的主張提供外部方法論檢核。

### 流動

MATS 訓練研究者。畢業生前往 Anthropic、DeepMind、OpenAI（實驗室安全團隊），或前往 Redwood、Apollo、METR、Eleos（外部評估）。外部評估者與實驗室及 UK AISI / CAISI 合作。出版成果再回饋生態系，供 MATS 的下一個梯次使用。

### 為什麼這一層很重要

單一來源的評估是不可靠的：實驗室評估自己的模型，有結構性的利益衝突。外部評估者可以提出並驗證實驗室可能少報的失敗模式。2024 年的 Sleeper Agents 論文（第 7 課）是 Anthropic + Redwood；Alignment Faking 是 Anthropic + Redwood；In-Context Scheming 是 Apollo；Anti-Scheming 是 Apollo + OpenAI。多組織的結構就是品質控管。

### 這在 Phase 18 中的位置

第 7 到 11 課引用了 Redwood 與 Apollo 的工作；第 18 課引用 METR 的框架比較；第 19 課引用 Eleos。第 28 課是這個 Phase 其餘部分所依賴的生態系的明確組織地圖。

```figure
sae-features
```

## 實際使用 (Use It)

沒有程式碼。閱讀 METR 的「Common Elements of Frontier AI Safety Policies」，作為外部綜合如何為實驗室內部政策工作增加價值的範例。

## 交付 (Ship It)

這堂課會產出 `outputs/skill-ecosystem-map.md`。給它一個 alignment 主張或評估，它會辨識出組織、發表管道與方法論風格，並與已知的對應組織交叉檢核。

## 練習 (Exercises)

1. 從第 7 到 15 課中挑一篇論文，找出涉及的組織。將作者與 MATS 校友及目前的生態系隸屬關係交叉比對。

2. 閱讀 METR 的「Common Elements of Frontier AI Safety Policies」。找出他們強調的三個跨實驗室共通點，以及兩個最大的分歧。

3. MATS 的職涯成果約 80% 是安全 / 資安。論證這種篩選壓力是適應性的（培養這個領域），還是帶偏見的（過濾掉非主流立場）。

4. Redwood 與 Apollo 都做 control/scheming 的工作，但風格不同。挑一個失敗模式，描述兩者各會如何調查。

5. Eleos AI 是唯一純粹做 model-welfare 的組織。設計一個假想的第二個組織，聚焦於不同的 welfare 相關問題（認知自由、機器人具身等），並闡述它的方法論。

## 關鍵術語 (Key Terms)

| 術語 | 一般說法 | 實際意義 |
|------|-----------------|------------------------|
| MATS | 「the mentorship program」 | ML Alignment & Theory Scholars；2021 年以來有 527+ 位研究者 |
| Redwood Research | 「the control lab」 | 應用型 alignment；AI Control 的作者；UK AISI 的夥伴 |
| Apollo Research | 「the scheming evals」 | 為 frontier 實驗室做部署前的 scheming 評估 |
| METR | 「the task-horizon evals」 | 以任務為基礎的能力評估；框架綜合 |
| Eleos AI | 「the welfare lab」 | Model-welfare 的部署前評估 |
| Talent pipeline | 「MATS -> labs」 | MATS 畢業生流向 Anthropic、DM、OpenAI、Redwood、Apollo、METR |
| External evaluation | 「non-lab check」 | 不是由模型生產者進行的評估；增加可信度 |

## 延伸閱讀 (Further Reading)

- [MATS (ML Alignment & Theory Scholars)](https://www.matsprogram.org/)：指導計畫
- [Redwood Research](https://www.redwoodresearch.org/)：AI Control 論文
- [Apollo Research](https://www.apolloresearch.ai/)：scheming 評估
- [METR — Common Elements of Frontier AI Safety Policies](https://metr.org/blog/2025-03-26-common-elements-of-frontier-ai-safety-policies/)：框架比較
- [Eleos AI Research](https://www.eleosai.org/research)：model welfare 方法論
