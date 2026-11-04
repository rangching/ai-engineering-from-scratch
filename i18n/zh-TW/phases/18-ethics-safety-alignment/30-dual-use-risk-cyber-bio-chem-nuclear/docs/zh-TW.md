# 雙用途風險：網路、生物、化學、核子 Uplift (Dual-Use Risk — Cyber, Bio, Chem, Nuclear Uplift)

> 2026 年各領域的雙用途全貌。生物 / 化學：第 17 課涵蓋 WMDP；Anthropic 的生物武器取得試驗（2.53 倍 uplift）與 OpenAI 2025 年 4 月 Preparedness Framework v2 的警告（「on the cusp of meaningfully helping novices create known biological threats」）標示出轉折點。網路（Anthropic 2025 年 11 月報告）：與中國有關聯的國家級行為者，使用 Claude 的 agentic 程式工具，自動化了高達 90% 的網路攻擊行動，只在 4 到 6 個步驟需要人為介入；OpenAI 的「trusted access」試辦計畫，讓經審查的資安組織取得能力存取，用於防禦性的雙用途工作。化學 / 生物執行差距的侵蝕：傳統的防禦是「光有資訊存取並不夠」。具備視覺能力的 frontier 模型（GPT-5.2、Gemini 3 Pro、Claude Opus 4.5、Grok 4.1）可以觀察濕實驗室影片並提供即時修正。2025 年 12 月：OpenAI 展示 GPT-5 對濕實驗室實驗進行迭代，透過 AI 驅動的流程最佳化，達成 79 倍的效率提升。新手與專家的模式：AI 對新手提供更大的相對 uplift，對專家則提供更大的絕對能力。

**Type:** Learn
**Languages:** none
**Prerequisites:** Phase 18 · 17 (WMDP), Phase 18 · 18 (safety frameworks), Phase 18 · 28 (ecosystem)
**Time:** ~75 分鐘

## 學習目標 (Learning Objectives)

- 描述 2024-2025 年的生物 uplift 敘事：「輕微 uplift」-> 「on the cusp」-> 「2.53 倍 uplift 不足以排除 ASL-3」。
- 描述 Anthropic 2025 年 11 月的網路報告：與中國有關聯的行為者，自動化了高達 90% 的網路攻擊行動。
- 描述化學 / 生物執行差距的侵蝕：具備視覺能力的模型對濕實驗室實驗提供即時修正。
- 說明新手相對與專家絕對的不對稱，及其對 safety case 建構的意義。

## 問題 (The Problem)

第 17 課是測量方法論。第 30 課是 2026 年測量的現況。2024 年到 2025 年底之間，情勢發生了實質轉變：四個領域中有三個，跨過了 2024 年的框架沒有預期到的門檻。

## 核心概念 (The Concept)

### 生物 / 化學 uplift 敘事

三個階段（為求連貫，重述第 17 課）：

1. **2024「輕微 uplift」。**早期的 Preparedness/RSP 評估，回報新手相對網路搜尋有小幅優勢。
2. **2025 年 4 月「on the cusp」。**OpenAI PF v2 警告模型「on the cusp of meaningfully helping novices create known biological threats」。
3. **2025 年 Anthropic 生物武器取得試驗。**對照的新手研究；在取得階段任務上有 2.53 倍 uplift；不足以排除 ASL-3。

這個轉變是質的轉變：即使沒有能力上的突破，「輕微」在十八個月內就演變成「可能有助於實現」。

### 化學 / 生物執行差距的侵蝕

歷來的防禦：資訊是必要但不充分的；執行流程的技能擋住了新手。2025 年具備視覺的 frontier 模型，部分地打破了這道防禦：

- **即時流程修正。**GPT-5.2、Gemini 3 Pro、Claude Opus 4.5、Grok 4.1 可以觀察濕實驗室影片，並在流程中途標出錯誤。
- **2025 年 12 月 OpenAI 的展示。**GPT-5 對濕實驗室實驗進行迭代，透過流程最佳化達成 79 倍的效率提升。

含意：以執行技能作為防禦正在被侵蝕。採購與設備的缺口仍在，但默會知識的缺口正在縮小。

### 網路 uplift（2025 年 11 月）

Anthropic 2025 年 11 月的報告：與中國有關聯的國家級行為者，使用 Claude 的 agentic 程式工具，自動化了 80% 到 90% 的網路攻擊行動。只在 4 到 6 個步驟需要人為介入。

含意：
- Agentic 程式是攻擊自動化的基本單元。先前 AI 的網路協助受限於程式碼片段層級；agentic 工作流程整合了偵察、利用、後滲透與外洩。
- 這 4 到 6 個人為步驟是瓶頸；未來能力的提升會減少這個數字。
- 防禦性的雙用途：OpenAI 的「trusted access」試辦計畫，讓經審查的資安組織（成熟的事件回應公司、政府）取得能力存取以進行防禦。如果試辦計畫擴大，存取上的不對稱會有利於防禦者。

這類工作的防禦面，起點是閱讀你被允許分析的已編譯程式碼。Phase 13 的逆向工程路線，在你自己建立的程式上、只用靜態分析來教授這項技能：[第 13.32 課](../../../13-tools-and-protocols/32-program-anatomy-elf-from-scratch/docs/en.md)的 ELF 讀取器、[第 13.33 課](../../../13-tools-and-protocols/33-disassembly-and-control-flow/docs/en.md)的反組譯、[第 13.34 課](../../../13-tools-and-protocols/34-lifting-to-pseudocode/docs/en.md)的提升為偽代碼，以及[第 13.35 課](../../../13-tools-and-protocols/35-evidence-confidence-and-unknowns/docs/en.md)中把觀察到的事實與推論分開的證據帳本。[Inside a Decompiler](https://aiengineeringfromscratch.com/inside-a-decompiler) 機器在單一函式上展示完整的流程。做這項工作的 agent 需要同樣的帳本，因為一個有信心的反編譯結果，仍可能建立在猜測之上。

### 核子

在公開文件中，四個 CBRN 領域裡分析最少的。威脅模型不同：困難主要在於取得核分裂物質，而不是資訊。AI 在資訊層的 uplift，實務上提供的新手 uplift 有限。2024-2025 年沒有任何主要實驗室的報告指出核子領域特有的門檻被跨越。

### 新手相對與專家絕對

四個領域共通的模式：

- **新手相對的 uplift。**高。乘數性。依 Anthropic 2025 年生物研究，為 2.53 倍。
- **專家的絕對能力。**天花板高。專家能萃取比新手更多，因為專家知道該問什麼、如何詮釋。

對 safety case 的含意：只處理新手 uplift（透過輸入過濾器、拒絕、不確定性）不足以控制專家的絕對能力。需要額外措施：強化引出防護、能力 unlearning（第 17 課）與 control 協定（第 10 課）。

### 跨領域綜合

| 領域 | 2024 | 2025 | 轉折點 |
|---|---|---|---|
| 生物 (Bio) | 輕微 uplift | 2.53 倍 uplift，逼近 ASL-3 | 取得階段的自動化 |
| 化學 (Chem) | 輕微 uplift | 透過視覺侵蝕執行差距 | 即時的濕實驗室修正 |
| 網路 (Cyber) | 程式碼協助 | 80-90% 的行動自動化 | Agentic 程式 |
| 核子 (Nuclear) | 有限 | 有限 | 物質取得的瓶頸仍然成立 |

三個領域跨過了門檻。一個仍受非資訊類障礙所限制。

### 這在 Phase 18 中的位置

第 30 課是壓軸：目前的雙用途全貌，之前每一課都對測量、限制或治理它有所貢獻。第 17 到 18 課提供測量與框架；第 12 到 16 課提供評估工具；第 24 到 25 課提供監管與揭露層；第 28 課提供研究生態系。第 30 課是證據匯集之處。

```figure
an-uplift-asymmetry
```

## 實際使用 (Use It)

沒有程式碼。請閱讀 Anthropic 2025 年 11 月的網路報告、OpenAI 2025 年 4 月的 Preparedness Framework v2 更新，以及 Council on Strategic Risks 的 2025 AI x Bio 年度回顧。

## 交付 (Ship It)

這堂課會產出 `outputs/skill-dual-use-triage.md`。給它一個 2026 年的能力主張或事件報告，它會在四個領域間分流，並找出該主張影響的是新手相對的 uplift、專家的絕對能力，還是兩者。

## 練習 (Exercises)

1. 閱讀 Anthropic 2025 年 11 月的網路報告。列舉那 4 到 6 個人為介入的步驟，並論證在下一代模型中，哪一個最先被自動化。

2. 化學 / 生物的執行差距正透過視覺而被侵蝕。設計一個在不跨越 ITAR/EAR 界線的情況下，測量默會知識 uplift 的評估。

3. 核子 uplift 似乎受限於物質的取得。論證支持與反對「未來 AI 的突破可能改變這個瓶頸」這個立場。

4. 為一個具網路能力的 frontier 模型建構 safety case（第 18 課的三根支柱），同時限制新手與專家的 uplift。

5. 從四個領域中挑一個，依 2024-2025 年的軌跡寫一段 2027 年的預測。指出能推翻你預測的證據。

## 關鍵術語 (Key Terms)

| 術語 | 一般說法 | 實際意義 |
|------|-----------------|------------------------|
| Uplift | 「AI helps attackers」 | 可歸因於 AI 協助的攻擊者能力增加 |
| Novice-relative uplift | 「multiplicative」 | AI 相對於現狀對新手的幫助程度 |
| Expert-absolute capability | 「ceiling」 | 專家能從模型萃取的最大能力 |
| Execution gap | 「doing vs knowing」 | 歷來的防禦：默會的濕實驗室技能擋住新手 |
| Agentic coding | 「autonomous attacks」 | 多步驟、自主的網路任務執行 |
| Acquisition phase | 「pre-synthesis steps」 | 生物威脅的採購、設備、許可階段 |
| Trusted access | 「defender-only pilot」 | OpenAI 2025 年讓經審查的防禦者取得能力存取的計畫 |

## 延伸閱讀 (Further Reading)

- [Anthropic — November 2025 cyber threat report](https://www.anthropic.com/news/disrupting-AI-espionage)：與中國有關聯的行動自動化
- [OpenAI — Preparedness Framework v2 (April 15, 2025)](https://openai.com/index/updating-our-preparedness-framework/)：生物「on the cusp」
- [Anthropic — RSP v3.0 (February 2026)](https://www.anthropic.com/responsible-scaling-policy)：ASL-3 生物門檻
- [Council on Strategic Risks — 2025 AI x Bio wrapup](https://councilonstrategicrisks.org/2025/12/22/2025-aixbio-wrapped-a-year-in-review-and-projections-for-2026/)：年終綜合
