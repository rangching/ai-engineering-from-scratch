# 合規：SOC 2、HIPAA、GDPR、PCI-DSS、EU AI Act、ISO 42001 (Compliance)

> 多框架涵蓋是 2026 年企業交易的基本門檻。**EU AI Act**：自 2024 年 8 月 1 日起生效。多數高風險要求在 2026 年 8 月 2 日開始執行。高風險系統義務的罰款最高 €15M 或全球年營業額的 3% (Art. 99(4))；禁止的 AI 行為最高 €35M 或 7% (Art. 99(3))。只要服務歐盟使用者，就全球適用。**Colorado AI Act**：2026 年 6 月 30 日生效（因 SB25B-004 自 2026 年 2 月延後），針對高風險系統的影響評估、對 AI 決策的申訴權。維吉尼亞州在信用/就業/住房/教育方面有類似規定。**SOC 2 Type II**：事實上的 B2B AI 要求（fintech 要的是 Type II，不是 Type I）。**GDPR**：有紀錄的最大 AI 專屬罰款是針對 Clearview AI 的 €30.5M（荷蘭 DPA，2024 年 9 月）；義大利 Garante 在 2024 年 12 月對 OpenAI 開罰 €15M（後於 2026 年 3 月上訴時被撤銷）。推論時的即時 PII 清除是可辯護的標準；事後處理清理並不夠。**HIPAA**：受醫療限制，沒有 BAA 就不能把 PHI 送到外部 AI 服務。**PCI-DSS**：涵蓋 AI 互動層需要設定加上合約協議，不是自動的。**ISO 42001**：新興的 AI 治理標準，與 ISO 27001 並列成為日益增加的採購要求。參考範例：OpenAI 維持 SOC 2 Type 2、ISO/IEC 27001:2022、ISO/IEC 27701:2019、GDPR/CCPA/HIPAA (BAA)/FERPA，以及 ChatGPT 付款元件的 PCI-DSS。跨框架對映可降低稽核疲勞：存取控制可對應 ISO 27001 A.5.15-5.18、GDPR Art. 32、HIPAA §164.312(a)。

**Type:** Learn
**Languages:** (Python optional — compliance is policy + process, not code)
**Prerequisites:** Phase 17 · 25 (Security), Phase 17 · 13 (Observability)
**Time:** ~60 分鐘

## 學習目標 (Learning Objectives)

- 列舉與 LLM 產品相關的七個 2026 年框架，並把每個對應到客戶區隔。
- 引用 EU AI Act 的執行時程（2024 年 8 月生效；高風險於 2026 年 8 月開始執行）以及兩級罰款上限（高風險義務為 €15M / 3%，禁止行為為 €35M / 7%）。
- 解釋為什麼事後處理的 PII 清理不足以符合 GDPR，並指出推論層的即時清除才是可辯護的標準。
- 描述跨框架的控制對映（例如存取控制對應 ISO 27001 A.5.15-5.18 + GDPR Art. 32 + HIPAA §164.312(a)）。

## 問題 (The Problem)

一位企業客戶的採購單位要求 SOC 2 Type II、GDPR、HIPAA BAA、ISO 27001，以及「EU AI Act 合規聲明」。你的團隊有 SOC 2 Type I。你離 Type II 還有六個月，而且還沒開始做 GDPR Article 30 紀錄。

多框架涵蓋不是 LLM 的問題，而是企業 SaaS 的問題，只是加上 LLM 專屬的覆蓋層。2026 年的採購團隊想要的是一份矩陣，每個框架一列、每個控制一欄，而不是一份 PDF。

## 核心概念 (The Concept)

### 七個框架

| 框架 | 範圍 | LLM 專屬要求 |
|-----------|-------|--------------------------|
| SOC 2 Type II | B2B SaaS 基準 | 流程控制經過 6-12 個月稽核 |
| HIPAA | 美國醫療 | 需要 BAA；沒有簽署協議 PHI 不能離開基礎設施 |
| GDPR | 歐盟使用者 | 即時 PII 清除；資料主體權利；Article 30 紀錄 |
| PCI-DSS | 付款資料 | 涉及付款的 AI 需要設定 + 合約 |
| EU AI Act | 服務歐盟使用者 | 風險等級分類；高風險系統：符合性評估、文件、logging |
| Colorado AI Act | 服務 CO 居民 | 影響評估；申訴權 |
| ISO 42001 | AI 治理 | 新興；與 ISO 27001 搭配 |

### EU AI Act 時程

- 2024 年 8 月 1 日：生效。
- 2025 年 2 月 2 日：禁止的 AI 行為開始執行。
- 2026 年 8 月 2 日：高風險系統開始執行（符合性評估、文件、logging）。
- 2027 年 8 月：受統一法規規範之產品中的高風險系統。

風險等級：不可接受（禁止）、高風險（符合性 + logging）、有限風險（透明度）、最低風險（無限制）。多數 B2B LLM SaaS 屬有限風險；高風險適用於就業、信用、教育、執法、移民、基本服務。

罰款（Article 99）：違反高風險系統義務最高 €15M 或全球年營業額的 3% (Art. 99(4))；禁止的 AI 行為最高 €35M 或 7% (Art. 99(3))；以較高者為準。

### GDPR：即時清除是標準

事後處理的清理（在 LLM 看過之後才遮罩 PII）並不是可辯護的立場，因為模型已經看過資料。推論層的即時清除是 2026 年的標準：

- LLM 呼叫之前先做實體辨識。
- 一致的 tokenization（Mesh 做法）保留語意。
- 只儲存已清除的 prompt + 經同意 opt-in 的原始內容。

近期執法：針對 Clearview AI 的 €30.5M（荷蘭 DPA，2024 年 9 月）是迄今有紀錄的最大 AI 專屬 GDPR 罰款；針對 OpenAI 的 €15M（義大利 Garante，2024 年 12 月）是最大的 LLM 專屬罰款，不過它在 2026 年 3 月上訴時被撤銷，該裁決仍在進一步審查中。事後處理的主張在稽核中都失敗了。

### HIPAA：BAA 不是選配

沒有簽署 Business Associate Agreement，就不能把 PHI 送到外部 AI 服務。三家 hyperscaler 的 LLM 平台 (Bedrock、Azure OpenAI、Vertex) 都提供 BAA。OpenAI 直接 API 提供 BAA。Anthropic 直接 API 提供 BAA。送出 PHI 之前請先確認。

### SOC 2 Type II

Type I：控制已設計並文件化。
Type II：控制在 6-12 個月內有效運作。

2026 年的 B2B 採購預設是 Type II。Type I 是起點；Type II 是門檻。

常見的稽核驅動項目：存取 log（誰看了什麼）、變更管理（如何部署的）、風險評估（每季）、事件應變（有測試過嗎？）。Phase 17 · 25 的 audit log 可直接重用。

### 跨框架對映

一份存取控制政策可滿足多個框架的控制：

| 控制 | 框架 |
|---------|-----------|
| 存取 logging | ISO 27001 A.5.15-5.18, GDPR Art. 32, HIPAA §164.312(a) |
| 變更管理 | ISO 27001 A.8.32, PCI DSS Req. 6, HIPAA breach-notification scope |
| 傳輸中加密 | ISO 27001 A.8.24, GDPR Art. 32, HIPAA §164.312(e) |
| Secrets 管理 | ISO 27001 A.8.19, PCI DSS Req. 8, SOC 2 CC6.1 |

合規工具（Drata、Vanta、Secureframe）會自動化這個對映。規模化後值得這筆花費。

### ISO 42001：新興

2023 年底發布。與 ISO 27001 並列成為日益增加的採購要求。涵蓋風險管理、資料品質、透明度、人類監督的 AI 治理框架。

### OpenAI 的參考範例

OpenAI 維持 SOC 2 Type 2、ISO/IEC 27001:2022、ISO/IEC 27701:2019、GDPR/CCPA/HIPAA (BAA)/FERPA，以及 ChatGPT 付款元件的 PCI-DSS。這大致就是 2026 年企業的基本門檻。

### 該記住的數字

- EU AI Act 罰款：最高 €15M / 3%（高風險義務，Art. 99(4)）；最高 €35M / 7%（禁止行為，Art. 99(3)）。
- EU AI Act 高風險執行：2026 年 8 月 2 日。
- 有紀錄的最大 AI 專屬 GDPR 罰款：€30.5M，Clearview AI（荷蘭 DPA，2024 年 9 月）。
- 最大的 LLM 專屬 GDPR 罰款：€15M，OpenAI（義大利 Garante，2024 年 12 月；2026 年 3 月上訴時被撤銷）。
- SOC 2 Type II 視窗：6-12 個月的控制運作。
- Colorado AI Act 生效日：2026 年 6 月 30 日（因 SB25B-004 自 2026 年 2 月延後）。

```figure
i4-control-matrix
```

## 實際使用 (Use It)

`code/main.py` 是用 Python 寫的合規對映試算表：給定一個控制，列出它滿足的框架。

## 交付 (Ship It)

這堂課產出 `outputs/skill-compliance-matrix.md`。給定客戶區隔與地區，指定所需的框架與控制。

## 練習 (Exercises)

1. 你的第一位企業客戶要求 SOC 2 Type II、HIPAA BAA、EU AI Act 聲明。要贏下這筆交易，最低可行的合規立場是什麼？
2. 把三個假設的 LLM 產品依 EU AI Act 風險等級分類。到了高風險時什麼會改變？
3. 你不小心在沒有 BAA 的情況下把 PHI 送給了供應商。逐步說明事件應變。
4. 辯論 ISO 42001 對中型市場的 AI 廠商而言，在 2026 年是否「必要」。
5. 把你的 LLM audit log 欄位 (Phase 17 · 25) 對映到至少三個框架的控制。

## 關鍵術語 (Key Terms)

| 術語 | 一般說法 | 實際意義 |
|------|----------------|------------------------|
| SOC 2 Type II | 「經稽核的控制」 | 控制在 6-12 個月內運作，並經獨立認證 |
| HIPAA BAA | 「醫療合約」 | Business Associate Agreement；處理 PHI 時必須具備 |
| GDPR | 「歐盟隱私」 | 即時 PII 清除是 2026 年可辯護的標準 |
| EU AI Act | 「歐盟 AI 規則」 | 高風險於 2026 年 8 月開始執行；€15M / 3%（高風險義務）— €35M / 7%（禁止行為） |
| Colorado AI Act | 「美國州級 AI 法」 | 2026 年 6 月 30 日生效（因 SB25B-004 延後）；影響評估 |
| ISO 42001 | 「AI 治理」 | 針對 AI 風險 + 透明度的新興框架 |
| ISO 27001 | 「資安 ISMS」 | 資訊安全管理系統基準 |
| Conformity assessment | 「歐盟 AI 文件包」 | 高風險要求：文件、測試、logging |
| Cross-framework mapping | 「一個控制，多個框架」 | 單一政策滿足多個框架的控制 |

## 延伸閱讀 (Further Reading)

- [OpenAI Security and Privacy](https://openai.com/security-and-privacy/)：參考合規範例。
- [GuardionAI — LLM Compliance 2026: ISO 42001, EU AI Act, SOC 2, GDPR](https://guardion.ai/blog/llm-compliance-guide-iso-42001-eu-ai-act-soc2-gdpr-2026)
- [Dsalta — SOC 2 Type 2 Audit Guide 2026: 10 AI Controls](https://www.dsalta.com/resources/ai-compliance/soc-2-type-2-audit-guide-2026-10-ai-powered-controls-every-saas-team-needs)
- [EU AI Act official text](https://eur-lex.europa.eu/eli/reg/2024/1689/oj)：原始來源。
- [Colorado AI Act](https://leg.colorado.gov/bills/sb24-205)：原始來源。
- [ISO/IEC 42001:2023](https://www.iso.org/standard/81230.html)：AI 管理系統標準。
