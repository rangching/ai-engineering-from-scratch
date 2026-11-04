# OpenAI Preparedness Framework 與 DeepMind Frontier Safety Framework

> OpenAI Preparedness Framework v2（2025 年 4 月）引入 Research Categories，包含 Long-range Autonomy、Sandbagging、Autonomous Replication and Adaptation、Undermining Safeguards，與 Tracked Categories 區分開來。Tracked Categories 會觸發 Capabilities Report 加上 Safeguards Report，由 Safety Advisory Group 審查。DeepMind 的 FSF v3（2025 年 9 月，2026 年 4 月 17 日新增 Tracked Capability Levels）把自主性併入 ML R&D 和 Cyber 領域（ML R&D autonomy level 1 = 以具競爭力的成本，對照人類 + AI 工具，完全自動化 AI R&D pipeline）。FSF v3 明確處理 deceptive alignment：對 instrumental-reasoning 的濫用做自動化監控。誠實的註記是：PF v2 中的 Research Categories（包含 Long-range Autonomy）不會自動觸發緩解措施；政策用語是「potential」。DeepMind 自己也表示，如果 instrumental reasoning 變強，自動化監控「長期而言將不再足夠」。

**Type:** Learn
**Languages:** Python (stdlib, three-framework decision-table diff tool)
**Prerequisites:** Phase 15 · 19 (Anthropic RSP)
**Time:** ~45 分鐘

## 問題 (The Problem)

第 19 課仔細閱讀了 Anthropic 的擴展政策。這堂課讀 OpenAI 與 DeepMind 的，讓整體圖像完整。這三份文件是堂兄弟般的成品，處理的是同一個問題：frontier 實驗室何時應該暫停或限制一個模型。它們收斂在一小組類別上，並在幾個重要的特定地方出現分歧。

收斂之處：三者都把 long-range autonomy 標示為值得追蹤的能力類別。三者都承認欺騙行為（alignment faking、sandbagging）是一類特定的風險。三者都有內部審查單位。分歧之處：OpenAI 把類別分為「Tracked」（強制緩解）與「Research」（沒有自動觸發）。DeepMind 把自主性併入兩個領域，而不是另外命名。各實驗室使用 Tracked vs Research、Critical vs Moderate，或 Tier-1 vs Tier-2 等名稱；某項能力落在哪個類別，在不同實驗室之間會造成不同的實際後果。

把它們放在一起讀才是有用的練習。同一項能力，在 Anthropic 可能是「強制緩解」，在 OpenAI 是「受監控但不觸發」，在 DeepMind 是「在特定領域中被追蹤」。政策立場很重要。

## 核心概念 (The Concept)

### OpenAI Preparedness Framework v2（2025 年 4 月）

結構：

- **Tracked Categories**：觸發 Capabilities Report（模型能做什麼）加上 Safeguards Report（有哪些緩解措施）。部署前由 Safety Advisory Group 審查。
- **Research Categories**：實驗室正在追蹤、但尚未承諾特定緩解措施的模型能力。包含 Long-range Autonomy、Sandbagging、Autonomous Replication and Adaptation、Undermining Safeguards。

v2 中的關鍵框架：Research Categories 不會自動觸發緩解措施。政策用語是「potential」的緩解措施。這是要留意的具體差異：一個類別位於 Tracked 或 Research，是「被把關」與「被觀察」之間的樞紐。

### DeepMind Frontier Safety Framework v3（2025 年 9 月；2026 年 4 月 17 日新增 Tracked Capability Levels）

結構：

- **Critical Capability Levels (CCLs)**：分為三組的能力門檻：濫用（CBRN、Cyber、Harmful Manipulation）、ML R&D，以及一個探索性的 misalignment 軌道。自主性不是獨立的領域；它被併入 ML R&D 和 Cyber。
- **Tracked Capability Levels**：2026 年 4 月新增的額外粒度。具體例子：ML R&D autonomy level 1 = 以具競爭力的成本，對照人類 + AI 工具，完全自動化 AI R&D pipeline。
- **Deceptive alignment 監控**：明確承諾對 instrumental-reasoning 的濫用做自動化監控。

自主性的框架與 OpenAI 不同。DeepMind 不把「Autonomy」保留為頂層領域；它被併入自主性會造成傷害的領域（ML R&D 與 Cyber）。論點是：沒有領域的自主性只是沒有風險的能力；反駁的論點是：跨領域的自主性是一種 meta 風險，框架應該把它點名出來。

### 三者的共同點

- 內部的 Safety Advisory Group（Anthropic SAG、OpenAI SAG、DeepMind 內部委員會）。對高能力模型在部署前審查。
- 明確提到 deceptive alignment / alignment faking 是一類風險。
- 依宣告頻率發布的常設成品（Anthropic：Frontier Safety Roadmap、Risk Report；OpenAI：Capabilities 與 Safeguards Report；DeepMind：FSF 更新週期）。
- 承認僅靠監控的防禦有上限。DeepMind 說得很明確：「自動化監控長期而言將不再足夠」。

### 分歧之處

- **Anthropic**：v3.0 移除暫停承諾；AI R&D-4 門檻是被點名的下一個關卡。
- **OpenAI**：Tracked 與 Research 的區分；Research Categories（包含 Long-range Autonomy）不會自動把關。
- **DeepMind**：自主性併入其他領域；2026 年 4 月的 Tracked Capability Levels 增加粒度。

### Sandbagging：讓三者都變複雜的特定能力

Sandbagging（模型在評估中策略性地表現不佳）位於 OpenAI 的 Research Categories。Anthropic 的 RSP v3.0 透過評估情境落差（第 1 課）處理它。DeepMind 透過 FSF v3 中的 deceptive alignment 監控處理它。

如果模型在評估中 sandbag，每個框架的能力門檻都會被低估。只有量測有效，框架才有效。這就是為什麼除了實驗室的自我評估之外，外部量測（第 21 課，METR）和對抗性評估也是必要的。

### 閱讀政策的技能

- 定位：你在乎的每項能力都應該能在政策中找到。如果找不到，政策就沒有涵蓋它。
- 分類：它是 Tracked（觸發緩解）還是 Research（被追蹤但不觸發）？OpenAI 明確命名這點；Anthropic 和 DeepMind 有各自的對應物。
- 頻率：政策是依宣告的時程更新，還是只在特定事件之後才更新？宣告頻率比較強。
- 獨立性：外部審查是強制的還是選擇性的？Anthropic 與 Apollo 及美國 AI Safety Institute 合作；OpenAI 與 METR 合作；DeepMind 主要靠內部 SAG。

```figure
a5-tracked-vs-research
```

## 實際使用 (Use It)

`code/main.py` 實作了一個小型的決策表比對工具。給定一項能力（自主性、deceptive alignment、R&D 自動化、cyber uplift 等），它會輸出三份政策各自如何分類這項能力，以及會觸發什麼緩解措施。它是閱讀輔助工具，不是政策工具。

## 交付 (Ship It)

`outputs/skill-cross-policy-diff.md` 以三個框架為參考，針對特定能力產出跨政策的比較。

## 練習 (Exercises)

1. 執行 `code/main.py`。對至少兩項你能對照原始文件驗證的能力，確認比對工具的輸出與政策相符。

2. 完整閱讀 OpenAI Preparedness Framework v2。找出每個 Research Category。針對每一個，用一句話寫出它為什麼在 Research 而不是 Tracked。

3. 完整閱讀 DeepMind FSF v3，以及 2026 年 4 月的 Tracked Capability Levels 更新。找出 ML R&D autonomy level 1 的具體評估標準。你會如何從外部量測它？

4. Sandbagging 位於 OpenAI 的 Research Categories。設計一個評估，迫使 sandbag 的模型揭露它真正的能力。參考第 1 課關於 eval-context gaming 的討論。

5. 針對一項特定能力（由你選擇）比較這三份政策。指出你認為哪一份政策的分類最嚴謹、哪一份最不嚴謹。以原文佐證。

## 關鍵術語 (Key Terms)

| 術語 | 一般說法 | 實際意義 |
|---|---|---|
| Preparedness Framework | 「OpenAI 的擴展政策」 | PF v2（2025 年 4 月）；Tracked vs Research 類別 |
| Tracked Category | 「強制緩解」 | 觸發 Capabilities + Safeguards Report；SAG 審查 |
| Research Category | 「僅監控」 | 被追蹤但沒有自動緩解；包含 Long-range Autonomy |
| Frontier Safety Framework | 「DeepMind 的擴展政策」 | FSF v3（2025 年 9 月）+ Tracked Capability Levels（2026 年 4 月） |
| CCL | 「Critical Capability Level」 | DeepMind 每個領域的門檻（CBRN、Cyber、Harmful Manipulation、ML R&D） |
| ML R&D autonomy level 1 | 「R&D 自動化」 | 以具競爭力的成本完全自動化 AI R&D pipeline |
| Sandbagging | 「策略性表現不佳」 | 模型在評估中表現不佳；位於 OpenAI Research Categories |
| Instrumental reasoning | 「手段與目的推理」 | 關於如何達成目標的推理；DeepMind 監控的目標 |

## 延伸閱讀 (Further Reading)

- [OpenAI — Updating our Preparedness Framework](https://openai.com/index/updating-our-preparedness-framework/) — v2 公告。
- [OpenAI — Preparedness Framework v2 PDF](https://cdn.openai.com/pdf/18a02b5d-6b67-4cec-ab64-68cdfbddebcd/preparedness-framework-v2.pdf) — 完整文件。
- [DeepMind — Strengthening our Frontier Safety Framework](https://deepmind.google/blog/strengthening-our-frontier-safety-framework/) — FSF v3 公告。
- [DeepMind — Updating the Frontier Safety Framework (April 2026)](https://deepmind.google/blog/updating-the-frontier-safety-framework/) — 新增 Tracked Capability Levels。
- [Gemini 3 Pro FSF Report](https://storage.googleapis.com/deepmind-media/gemini/gemini_3_pro_fsf_report.pdf) — FSF 格式 Risk Report 的範例。
