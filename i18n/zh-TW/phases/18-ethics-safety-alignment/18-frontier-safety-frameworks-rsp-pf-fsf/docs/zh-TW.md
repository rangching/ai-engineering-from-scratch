# Frontier 安全框架：RSP、PF、FSF (Frontier Safety Frameworks — RSP, PF, FSF)

> 三家主要實驗室的框架，定義了 2026 年業界對 frontier 能力的治理。Anthropic Responsible Scaling Policy v3.0 (February 2026) 引入分級的 AI Safety Levels（ASL-1 到 ASL-5+），仿效生物安全等級，ASL-3 在 2025 年 5 月為與 CBRN 相關的模型啟動。OpenAI Preparedness Framework v2 (April 2025) 為被追蹤的能力定義五項準則，並將 Capabilities Reports 與 Safeguards Reports 分開。DeepMind Frontier Safety Framework v3.0 (September 2025) 引入 Critical Capability Levels，包括新的 Harmful Manipulation CCL。三者現在都包含競爭者調整條款，允許在同業實驗室沒有可比防護措施就出貨時延後。跨實驗室的一致性是結構性的，而不是用語上的：「Capability Thresholds」、「High Capability thresholds」和「Critical Capability Levels」指的是類似的概念。

**Type:** Learn
**Languages:** none
**Prerequisites:** Phase 18 · 17 (WMDP), Phase 18 · 07-09 (deception failures)
**Time:** ~75 分鐘

## 學習目標 (Learning Objectives)

- 描述 Anthropic 的 ASL 分級結構，以及是什麼啟動了 ASL-3。
- 說出 OpenAI Preparedness Framework v2 對被追蹤能力的五項準則。
- 描述 DeepMind 的 Critical Capability Level 結構與 Harmful Manipulation CCL。
- 解釋競爭者調整條款，以及它們為什麼對競賽動態很重要。
- 定義 safety case，並描述三根支柱的結構（monitoring、illegibility、incapability）。

## 問題 (The Problem)

第 7 到 17 課確立了欺騙是可能的、雙用途能力存在，而且評估有其限制。擁有 frontier 能力模型的實驗室，需要一套內部治理結構，來：
- 定義何時需要新防護措施的門檻。
- 定義擴大規模前所需的評估。
- 描述 safety case 長什麼樣子。
- 處理競賽動態問題（如果競爭者在沒有防護措施的情況下出貨，你要怎麼辦？）。

這三個 2025-2026 年的框架是目前最先進的水準：不完美、持續演進，而且在各實驗室之間已足夠一致，使得治理的問題現在是這些框架是否足夠，而不是它們是否存在。

## 核心概念 (The Concept)

### Anthropic Responsible Scaling Policy v3.0 (February 2026)

ASL 結構：
- ASL-1：不是 frontier 模型（被比 frontier 更弱的基準線所涵蓋）。
- ASL-2：目前的 frontier 基準線；以一般防護措施部署。
- ASL-3：災難性濫用的風險大幅提高；具有與 CBRN 相關的能力。2025 年 5 月啟動。
- ASL-4：AI R&D-2 跨越門檻；能自動化入門級 AI 研究的模型。
- ASL-5+：進階 AI R&D；能大幅加速有效 scaling 的模型。

v3.0 新增：
- Frontier Safety Roadmaps（以去識別形式公開）。
- Risk Reports（每季一次，部分經外部審查）。
- AI R&D 被拆成 AI R&D-2 與 AI R&D-4。
- 一旦跨越 AI R&D-4，就需要肯定式的 safety case，找出模型追求未對齊目標所帶來的 misalignment 風險。

### OpenAI Preparedness Framework v2 (April 15, 2025)

被追蹤能力的五項準則：
- **Plausible。** 存在合理的威脅模型。
- **Measurable。** 可進行實證評估。
- **Severe。** 傷害很大。
- **Net-new。** 不是既有風險的放大。
- **Instantaneous-or-irremediable。** 傷害發生得快，或無法挽回。

符合全部五項的能力會被追蹤。其他則不追蹤。

PF v2 的其他結構：
- 將 Capabilities Reports（模型能做什麼）與 Safeguards Reports（存在哪些控制）分開。
- Safety Advisory Group 審查。
- 領導階層核准；董事會的 Safety & Security Committee 監督。
- 「Adjustment clause」：若另一家實驗室在沒有可比防護措施的情況下出貨，OpenAI 可以降低要求。

### DeepMind Frontier Safety Framework v3.0 (September 2025)

依領域劃分的 Critical Capability Levels (CCL)：
- Bioweapon Uplift
- Cyber Uplift
- ML R&D Acceleration
- Harmful Manipulation（v3.0 新增）：能在高風險情境中實質改變信念 / 行為的模型。

v2.0 (February 2025) 新增了 Deceptive Alignment 章節，並提高 ML R&D CCL 的安全等級。

### 跨實驗室的一致性

- Anthropic「Capability Thresholds」。
- DeepMind「Critical Capability Levels」。
- OpenAI「High Capability thresholds」。

沒有業界標準用語。結構上一致：三個框架各自定義了分級的能力門檻，並附有公開的評估準則。三者都包含競爭者調整條款（2025 年起）。

### Safety case

Safety case 是一份書面論證，主張在最壞情況假設下，某個部署的安全性可被接受。標準結構針對三根支柱：

- **Monitoring。** 如果發生壞行為，我們能偵測到嗎？
- **Illegibility。** 模型是否缺乏執行連貫計畫以造成傷害的能力？
- **Incapability。** 模型是否缺乏造成該傷害的能力？

不同的 safety case 針對不同的支柱。對 ASL-3 的 CBRN 案例，incapability（透過 unlearning）是主要目標。對 deceptive alignment，monitoring 與 illegibility 是目標。對網路 uplift，三者都相關。

### 競賽動態問題

競爭者調整條款備受爭議。批評者主張它們造成逐底競爭：如果三家實驗室在競爭者背棄時都會降低要求，均衡就會往背棄偏移。辯護者主張，如果背棄的實驗室較不重視安全，那麼另一種做法（單方面採取防護措施）會產生更差的結果。

UK AISI、US CAISI 與 EU AI Office（第 24 課）是外部治理的對應者。實驗室框架是自願性的；監管框架正在浮現。

### 這在 Phase 18 中的位置

第 17 到 18 課是建立在欺騙與 red-team 分析之上的測量與治理層。第 19 到 24 課涵蓋 welfare、bias、隱私、watermarking 與監管結構。第 28 課描繪落實這些評估的研究生態系（MATS、Redwood、Apollo、METR）。

```figure
al-asl-ladder
```

## 實際使用 (Use It)

這堂課沒有程式碼。請閱讀三份原始資料：RSP v3.0、PF v2、FSF v3.0。把每家實驗室的分級結構對應到其他家，並找出每家各自定義、而其他家沒有的一個門檻。

## 交付 (Ship It)

這堂課會產出 `outputs/skill-framework-diff.md`。給它一份安全框架或發布說明，它會把該框架的門檻定義、所需評估與 safety-case 結構，與 RSP v3.0、PF v2、FSF v3.0 比較，並標出跨實驗室的缺口。

## 練習 (Exercises)

1. 閱讀 RSP v3.0、PF v2 和 FSF v3.0。整理一張表，列出各實驗室的 CBRN 門檻、AI R&D 門檻，以及各自要求的部署前評估。

2. 競爭者調整條款存在於三個框架中（2025 年起）。寫一段支持它的論述，再寫一段反對它的論述。指出每種立場所依賴的假設。

3. 為跨越 Anthropic AI R&D-4 門檻的模型設計 safety case。說出三根支柱（monitoring、illegibility、incapability）各自需要的證據。

4. DeepMind 的 FSF v3.0 引入 Harmful Manipulation CCL。提出三項實證測量，能指出模型已跨越這個門檻。

5. 閱讀 METR 的「Common Elements of Frontier AI Safety Policies」(2025)。說出三個最強的跨實驗室共通點，以及兩個最大的分歧。

## 關鍵術語 (Key Terms)

| 術語 | 一般說法 | 實際意義 |
|------|-----------------|------------------------|
| RSP | 「Anthropic's framework」 | Responsible Scaling Policy；ASL 分級；v3.0 為 2026 年 2 月 |
| PF | 「OpenAI's framework」 | Preparedness Framework；五項準則；v2 為 2025 年 4 月 |
| FSF | 「DeepMind's framework」 | Frontier Safety Framework；CCL；v3.0 為 2025 年 9 月 |
| ASL-3 | 「biosafety level 3-analog」 | Anthropic 針對 CBRN 相關能力的等級；2025 年 5 月啟動 |
| CCL | 「critical capability level」 | DeepMind 的門檻概念；依領域劃分 |
| Safety case | 「the formal argument」 | 書面論證，主張部署在最壞情況的 U 下安全性可被接受 |
| Adjustment clause | 「competitor defection allowance」 | 框架中的條款，若競爭者在沒有可比防護措施下出貨，可降低要求 |

## 延伸閱讀 (Further Reading)

- [Anthropic — Responsible Scaling Policy v3.0 (February 2026)](https://www.anthropic.com/responsible-scaling-policy)：ASL 分級、roadmap、AI R&D 拆分
- [OpenAI — Updating the Preparedness Framework (April 15, 2025)](https://openai.com/index/updating-our-preparedness-framework/)：五項準則、adjustment clause
- [DeepMind — Strengthening our Frontier Safety Framework (September 2025)](https://deepmind.google/blog/strengthening-our-frontier-safety-framework/)：CCL v3.0、Harmful Manipulation
- [METR — Common Elements of Frontier AI Safety Policies (2025)](https://metr.org/blog/2025-03-26-common-elements-of-frontier-ai-safety-policies/)：跨實驗室比較
