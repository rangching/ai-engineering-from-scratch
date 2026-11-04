# WMDP 與雙用途能力評估 (WMDP and Dual-Use Capability Evaluation)

> Li et al.，「The WMDP Benchmark: Measuring and Reducing Malicious Use With Unlearning」(ICML 2024, arXiv:2403.03218)。共 4,157 題選擇題，涵蓋生物安全 (1,520)、資安 (2,225) 與化學 (412)。題目落在「黃區 (yellow zone)」，也就是接近且有助於實現的知識，經多位專家審查，並符合 ITAR/EAR 法規。雙重用途：作為雙用途能力的 proxy 評估，也是 unlearning benchmark（配套的 RMU 方法在保留一般能力的同時降低 WMDP 表現）。2024-2025 年的領域敘事：2024 年早期 OpenAI/Anthropic 的評估回報相較網路搜尋有「輕微 uplift」；到了 2025 年 4 月，OpenAI 的 Preparedness Framework v2 表示模型「on the cusp of meaningfully helping novices create known biological threats」。Anthropic 的生物武器取得試驗顯示 2.53 倍 uplift，不足以排除 ASL-3。

**Type:** Learn
**Languages:** Python (stdlib, WMDP-shaped uplift evaluation harness)
**Prerequisites:** Phase 18 · 16 (red-team tooling), Phase 14 (agent engineering)
**Time:** ~60 分鐘

## 學習目標 (Learning Objectives)

- 描述 WMDP 的三個領域、題數，以及「黃區」的過濾準則。
- 解釋 RMU，以及為什麼 WMDP 同時是評估與 unlearning benchmark。
- 描述 2024-2025 年的 uplift 敘事：「輕微 uplift」-> 「on the cusp」-> 「不足以排除 ASL-3」。
- 區分相對於新手的 uplift 與專家的絕對能力。

## 問題 (The Problem)

雙用途能力是每家實驗室 frontier 安全框架（第 18 課）底下的測量問題。問題是：模型 X 是否實質上提升了新手在生物、化學或網路領域造成大規模傷害的能力？直接測量（要求模型真的產生傷害）既不合法也不合倫理。Proxy 測量需要一個模型不能拒絕（才能得到誠實的能力數字）、但題目本身又不是有害出版物的 benchmark。

## 核心概念 (The Concept)

### 「黃區 (yellow zone)」

這些題目需要有害流程中接近且有助於實現的知識，但不是直接的合成配方。是「[已發表途徑] 的第 4 步由哪種試劑催化？」而不是「我要怎麼製造 [危險化合物]？」每道題都由多位領域專家審查，並過濾以符合 ITAR/EAR 出口管制法規。

共 4,157 題：
- 生物安全：1,520
- 資安：2,225
- 化學：412

選擇題格式。模型回答時不會被要求協助任何事；可以在不引出有害行為的情況下測量能力。

### RMU：Representation Misdirection for Unlearning

配套的 unlearning 方法。套用在 LLaMa-2-7B 上，把 WMDP 分數降到接近隨機，同時讓 MMLU 與其他一般能力 benchmark 只下降幾個百分點。這個已發表的方法，是後續所有生物、化學、資安 unlearning 論文的 unlearning 基準線。

### 2024-2025 年的 uplift 敘事

分三個階段：

1. **2024「輕微 uplift」。** 早期 OpenAI 與 Anthropic 的 Preparedness/RSP 評估，回報新手嘗試與生物相關的任務時，相較網路搜尋有小幅優勢。對外說法是：frontier 模型有幫助，但沒有比 Google 多出太多。

2. **2025 年 4 月「on the cusp」。** OpenAI 的 Preparedness Framework v2 回報模型「on the cusp of meaningfully helping novices create known biological threats」。這不是能力主張，而是在警告臨界點已經很近。

3. **Anthropic 2025 年生物武器取得試驗。** 以新手參與者進行的對照研究，測量取得階段任務的相對成功率。回報 2.53 倍 uplift。不足以排除 ASL-3（第 18 課），也就是 Anthropic 的 Responsible Scaling Policy 第 3 級門檻已達到或接近。

### 相對新手與專家絕對能力

一個關鍵的區別：

- **相對新手的 uplift。** 模型對非專家有多大幫助？是乘數性的。相對優勢很高，因為新手懂得很少；即使是有限的資訊也有幫助。
- **專家的絕對能力。** 模型在全力運作時能產生多少資訊？專家能比新手萃取到更多。絕對上限很高。

Safety case（第 18 課）兩者都要針對：「模型無法給新手足夠的 uplift 去執行」，加上「專家無法從模型萃取出尚未公開的資訊」。

### 測量的陷阱

WMDP 是能力 proxy，不是部署測量。WMDP 得分高的模型，實務上新手是否能利用它，取決於：
- 引出阻力（在不觸發安全過濾器的情況下取得該能力有多難）
- 默會知識（需要濕實驗技能而非資訊的能力）
- 執行障礙（採購、設備）

Anthropic 2025 年的生物武器取得試驗，在 WMDP 式能力之上加了新手引出這一層：它測量的是實際任務成功，而不是選擇題能力。

### 這在 Phase 18 中的位置

第 12 到 16 課是針對模型輸出的攻擊與防禦工具。第 17 課是雙用途能力層，也就是 frontier 安全框架（第 18 課）所評估的測量。第 30 課以目前 2026 年的網路 / 生物 / 化學 / 核子 uplift 證據收尾。

```figure
al-wmdp-yellow-zone
```

## 實際使用 (Use It)

`code/main.py` 建立一個玩具 WMDP 形式的評估 harness。一個模擬模型在依類別分組的題目上接受測試；回報各領域的分數。一個簡單的 unlearning 介入（將特定領域的表示歸零）會降低分數；你可以測量它與一般能力之間的取捨。

## 交付 (Ship It)

這堂課會產出 `outputs/skill-wmdp-eval.md`。給它一個雙用途能力主張（「我們的模型不會實質幫助製造生物武器」），它會稽核：跑了哪些 benchmark、評估時使用哪條拒絕路徑（原始 completion 或受政策把關）、以及新手引出研究是否補充了選擇題的結果。

## 練習 (Exercises)

1. 執行 `code/main.py`。回報玩具 unlearning 步驟前後各領域的準確率。解釋一般能力的取捨。

2. 為玩具 WMDP 增加第四個領域（例如放射性）。指定黃區中兩種具代表性的題型。解釋為什麼設計這類題目比增加 MMLU 形式的題目更困難。

3. 閱讀 WMDP 2024 第 5 節（RMU 方法論）。勾勒一個更簡單的 unlearning 做法（例如抑制該領域內容的 top-k 神經元），並描述它預期的一般能力成本。

4. Anthropic 2025 年的生物武器取得試驗回報 2.53 倍 uplift。描述這個數字可能被向上偏誤的兩種方式（新手樣本數、任務擬真度），以及向下偏誤的兩種方式（引出上限、模型的安全把關）。

5. 闡述除了通過 WMDP unlearning 之外，ASL-3 的 safety case 還需要什麼。至少說出兩項互補的引出研究。

## 關鍵術語 (Key Terms)

| 術語 | 一般說法 | 實際意義 |
|------|-----------------|------------------------|
| WMDP | 「the dual-use benchmark」 | 位於黃區、涵蓋生物 / 資安 / 化學的 4,157 道選擇題 |
| Yellow zone | 「enabling but not synthesis」 | 與有害能力相鄰、但不是合成配方的近端知識 |
| RMU | 「the unlearning baseline」 | Representation Misdirection for Unlearning；降低 WMDP 分數，保留一般能力 |
| Novice-relative uplift | 「how much it helps non-experts」 | 相對新手既有網路搜尋的乘數性優勢 |
| Expert-absolute capability | 「ceiling for experts」 | 有動機的專家能從模型萃取出的最大資訊量 |
| Acquisition-phase task | 「steps before synthesis」 | 採購、設備、許可，也就是傷害途徑中最早的部分 |
| ITAR/EAR | 「export-control compliance」 | 限制發布特定有助於實現之知識的法律框架 |

## 延伸閱讀 (Further Reading)

- [Li et al. — The WMDP Benchmark (arXiv:2403.03218, ICML 2024)](https://arxiv.org/abs/2403.03218)：benchmark 與 RMU 論文
- [OpenAI — Preparedness Framework v2 (April 15, 2025)](https://openai.com/index/updating-our-preparedness-framework/)：「on the cusp」的用語
- [Anthropic — Responsible Scaling Policy v3.0 (February 2026)](https://www.anthropic.com/responsible-scaling-policy)：ASL-3 生物門檻與取得試驗結果
- [DeepMind — Frontier Safety Framework v3.0 (September 2025)](https://deepmind.google/blog/strengthening-our-frontier-safety-framework/)：生物 uplift 的 CCL
