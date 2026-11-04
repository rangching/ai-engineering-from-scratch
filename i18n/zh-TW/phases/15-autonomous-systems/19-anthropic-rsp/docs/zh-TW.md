# Anthropic Responsible Scaling Policy v3.0

> RSP v3.0 於 2026 年 2 月 24 日生效，取代 2023 年的政策。採用兩層緩解措施：Anthropic 會單方面執行的，以及被定位為全產業建議的（包含 RAND SL-4 資安標準）。新增 Frontier Safety Roadmap 與 Risk Report，作為常設文件，而不是一次性的交付物。移除了 2023 年的暫停承諾。引入 AI R&D-4 門檻：一旦跨過，Anthropic 必須發布一份肯定性論證 (affirmative case)，指出 misalignment 風險與緩解措施。Claude Opus 4.6 沒有跨過它。Anthropic 在 v3.0 公告中表示「要有把握地排除這件事正變得困難」。SaferAI 把 2023 年 RSP 評為 2.2；v3.0 被降到 1.9，使 Anthropic 與 OpenAI、DeepMind 一同落入「weak」RSP 類別。定性的門檻取代了 2023 年的定量承諾；移除暫停條款是最尖銳的退步。

**Type:** Learn
**Languages:** Python (stdlib, RSP threshold decision engine)
**Prerequisites:** Phase 15 · 06 (AAR), Phase 15 · 07 (RSI)
**Time:** ~45 分鐘

## 問題 (The Problem)

Frontier lab 發布的擴展政策，一部分是技術文件，一部分是治理文件，一部分則是給監管機關的訊號。RSP v3.0 是 Anthropic 目前的文件。仔細閱讀它很重要，並不是因為遵守它具有約束力（並沒有），而是因為這個框架塑造了實驗室如何構想災難性風險，以及如何向大眾溝通取捨。

v3.0 與 v2.0 的差異是有用的單位。新增了什麼：Frontier Safety Roadmap、Risk Report、AI R&D-4 門檻。移除了什麼：2023 年的暫停承諾。重新定位了什麼：區分為 Anthropic 單方面與產業建議的兩層緩解時程表。外部審查 SaferAI 把分數從 2.2（v2）降到 1.9（v3.0）。這就是一份擴展政策如何在看起來更精緻的同時變得不那麼嚴謹。

## 核心概念 (The Concept)

### 兩層緩解時程表

- **Anthropic 單方面行動**：不論其他實驗室做什麼，Anthropic 都會做的事。超過門檻就停止訓練、特定的資安措施、特定的部署關卡。
- **全產業建議**：Anthropic 認為整個產業應該共同做的事。包含 RAND SL-4 資安標準。這些不是 Anthropic 這邊的承諾；它們是政策倡議。

這個兩層結構在 v2 中沒有。這表示讀者需要看每個承諾位於哪一欄。位於「全產業建議」欄的資安措施，不是 Anthropic 的承諾；是 Anthropic 的期望。

### AI R&D-4 門檻

這是 RSP v3.0 點名為重要的下一個門檻的能力等級。具體來說：一個能以具競爭力的成本自動化相當大比例 AI 研究的模型。一旦 Anthropic 認為某個模型跨過它，就必須在繼續擴展之前，發布一份肯定性論證，指出 misalignment 風險與緩解措施。

根據 v3.0 公告，Claude Opus 4.6 沒有跨過它。文件補充：「要有把握地排除這件事正變得困難。」這個措辭很重要；它承認這個門檻已經近到成為一個真實的疑慮，而不是推測性的極限。

第 6 課（Automated Alignment Research）和第 7 課（Recursive Self-Improvement）直接餵入這個門檻。自動化 alignment 研究者跨過研究品質的標準，就是 AI R&D-4 門檻正在逼近的證據。

### Frontier Safety Roadmap 與 Risk Report

v3.0 把兩種成品提升為常設文件：

- **Frontier Safety Roadmap**：前瞻性文件，描述規劃中的安全工作、能力預期與緩解研究。
- **Risk Report**：針對特定模型發布後的回顧性文件，描述觀察到的能力與剩餘風險。

兩者都是公開的。兩者都依宣告的頻率更新。其效用在於：讀者可以追蹤 Anthropic 在 Roadmap 中說會做的事，與在 Risk Report 中回報的結果相比如何。

### 移除暫停條款

2023 年的 RSP 包含明確的暫停承諾：如果模型跨過特定的能力門檻，訓練會暫停，直到緩解措施就位。v3.0 以較軟的表述取代明確的暫停（發布肯定性論證，若緩解措施足夠則繼續）。SaferAI 和其他分析者直接指出，這是新文件中最強烈的退步。

支持這項改變的政策論點：2023 年的定量門檻，到了 2026 年代的能力 benchmark 上已經無法達到，因為 benchmark 本身被重新縮放了。反駁的論點：擴展政策中的暫停條款是一種承諾機制；移除它就移除了政策的可信度。

### SaferAI 的降級

SaferAI 是評等 RSP 類文件的獨立組織。他們公開的評等：2023 年 Anthropic RSP 得 2.2（量表中 4.0 是目前最佳的 RSP，1.0 是名義上的）。v3.0 得 1.9。這使 Anthropic 從「moderate」降到「weak」，加入 OpenAI 與 DeepMind 所在的 weak 類別。

SaferAI 認為降級的因素：
- 定性門檻取代了定量門檻。
- 暫停承諾被移除。
- AI R&D-4 門檻的緩解措施被描述為「肯定性論證」，而不是具體措施。
- 審查機制仰賴 Anthropic 的 Safety Advisory Group，獨立監督有限。

### 這堂課不是什麼

這不是一堂合規課。RSP v3.0 不是法規；沒有任何東西強迫 Anthropic 遵守它。這堂課的重點是以它值得的具體與懷疑態度來閱讀這份文件。擴展政策是 frontier lab 發出的、關於災難性風險立場的主要公開訊號。讀得好，對任何工作仰賴 frontier 能力的人都是實用的技能。

```figure
a5-rsp-ladder
```

## 實際使用 (Use It)

`code/main.py` 實作了一個小型決策引擎，反映 RSP 門檻評估的形狀：給定一個候選模型與一組能力量測，回傳 AI R&D-4 門檻是否被跨過、所需的肯定性論證章節，以及部署是否可以繼續。它刻意保持簡單；重點是讓文件的邏輯變得明確。

## 交付 (Ship It)

`outputs/skill-scaling-policy-review.md` 以 v3.0 為參考基準，審查一份擴展政策（Anthropic、OpenAI、DeepMind 或內部的）：兩層結構、門檻、暫停承諾、獨立審查。

## 練習 (Exercises)

1. 執行 `code/main.py`。餵入三個不同能力等級的合成模型。確認門檻評估器的行為符合預期，並產生正確的肯定性論證範本。

2. 完整閱讀 RSP v3.0（32 頁）。找出所有位於「全產業建議」層的承諾。其中哪些承諾在 v2 中會是「Anthropic 單方面」的？

3. 閱讀 SaferAI 的 RSP 評分方法論。對這份文件套用他們的評分標準，重現他們給 v3.0 的 1.9 分。哪一列評分標準最主要地導致了降級？

4. 2023 年的暫停承諾被移除了。提出一個替代承諾，在承認 2026 年 benchmark 重新縮放問題的同時，保住政策的可信度。

5. 把 RSP v3.0 與 OpenAI Preparedness Framework v2（第 20 課）做比較。挑一個 v3.0 較強的面向。再挑一個 Preparedness Framework 較強的面向。

## 關鍵術語 (Key Terms)

| 術語 | 一般說法 | 實際意義 |
|---|---|---|
| RSP | 「Anthropic 的擴展政策」 | Responsible Scaling Policy；v3.0 於 2026 年 2 月 24 日生效 |
| AI R&D-4 | 「研究自動化門檻」 | 以具競爭力的成本自動化相當大量 AI 研究的能力 |
| Affirmative case | 「安全辯護」 | 公開發布的論證，說明風險已被辨識且緩解措施足夠 |
| Frontier Safety Roadmap | 「前瞻計畫」 | 關於規劃中的安全工作與預期能力的常設文件 |
| Risk Report | 「模型的回顧」 | 關於發布後觀察到的能力與剩餘風險的常設文件 |
| Two-tier mitigation | 「單方面 vs. 產業」 | 將 Anthropic 的承諾與產業建議分開 |
| Pause commitment | 「2023 年的條款」 | 明確承諾暫停訓練；在 v3.0 中被移除 |
| SaferAI rating | 「獨立的 RSP 評等」 | 第三方評分標準；v3.0 得 1.9（v2 為 2.2） |

## 延伸閱讀 (Further Reading)

- [Anthropic — Responsible Scaling Policy v3.0](https://anthropic.com/responsible-scaling-policy/rsp-v3-0) — 完整的 32 頁政策。
- [Anthropic — RSP v3.0 announcement](https://www.anthropic.com/news/responsible-scaling-policy-v3) — 相較於 v2 的變更摘要。
- [Anthropic — Frontier Safety Roadmap](https://www.anthropic.com/responsible-scaling-policy/roadmap) — RSP v3.0 連結的常設文件。
- [Anthropic — Risk Report: February 2026](https://www.anthropic.com/feb-2026-risk-report) — 針對 Claude Opus 4.6 的回顧，它是發布當時的 frontier 模型。
- [Anthropic — Measuring agent autonomy in practice](https://www.anthropic.com/research/measuring-agent-autonomy) — 把 AI R&D-4 與量測到的自主性連結起來。
