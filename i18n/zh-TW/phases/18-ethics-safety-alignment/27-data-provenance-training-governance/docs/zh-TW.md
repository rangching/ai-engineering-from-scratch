# 資料來源追溯與訓練資料治理 (Data Provenance and Training-Data Governance)

> 歐盟 AI Act 要求 GPAI 在 2025 年 8 月前提供機器可讀的 opt-out 標準（透過歐盟著作權指令的 TDM 例外）。California AB 2013（2024 年簽署）：Generative AI 訓練資料透明度要求開發者公布資料集摘要，含 12 個法定欄位。2025 年各資料保護機關 (DPA) 在正當利益上的趨同：愛爾蘭 DPC（2025 年 5 月 21 日）在 EDPB 意見之後，接受 Meta 在有防護措施下，以第一方公開的歐盟 / EEA 成人內容訓練 LLM；科隆高等法院（2025 年 5 月 23 日）駁回禁制令；漢堡 DPA 撤回緊急程序；英國 ICO（2025 年 9 月 23 日）對 LinkedIn 的 AI 訓練防護措施（透明度、簡化的 opt-out、延長的異議期間）發出正面的監管回應並持續監督，這不是正式的許可。巴西 ANPD（2024 年 7 月 2 日）因資訊透明度不足，暫停了 Meta 的處理；在 Meta 提交合規計畫後，該預防性措施於 2024 年 8 月 30 日解除。關鍵的不可逆問題：cookie 同意框架是為即時、可逆的追蹤而設計；一旦資料進入模型權重，就無法外科手術式地抹除，對訓練完成的神經網路沒有實際可行的 GDPR 被遺忘權。合規的時間窗口在收集資料的當下。Data Provenance Initiative（dataprovenance.org，Longpre、Mahari、Lee et al.，「Consent in Crisis」，2024 年 7 月）：大規模稽核顯示，隨著發布者加入 robots.txt 限制，AI 資料公共領域正快速萎縮。

**Type:** Learn
**Languages:** Python (stdlib, 12-field California AB 2013 scaffolding generator)
**Prerequisites:** Phase 18 · 24 (regulatory), Phase 18 · 26 (cards)
**Time:** ~60 分鐘

## 學習目標 (Learning Objectives)

- 描述 California AB 2013 對 Generative AI 訓練資料透明度規定的 12 個欄位。
- 說明 2025 年 DPA 對以正當利益訓練 LLM 的立場（愛爾蘭 DPC、英國 ICO、漢堡、科隆）。
- 描述不可逆問題：為什麼 GDPR 被遺忘權對訓練完成的神經網路沒有實際可行的對應做法。
- 說出 Data Provenance Initiative「Consent in Crisis」的發現。

## 問題 (The Problem)

訓練資料治理是每份 model card（第 26 課）與監管義務（第 24 課）的上游。在 2024-2025 年，監管版圖收斂到三項原則：opt-out 基礎設施、逐資料集揭露，以及對公開可得資料的正當利益通融。沒有在收集當下就合規的提供者，無法在下游補救。

## 核心概念 (The Concept)

### California AB 2013

2024 年簽署。對於 2022 年 1 月 1 日當天或之後發布的系統，文件必須在 2026 年 1 月 1 日當天或之前公布。第 3111(a) 條要求開發者公布訓練所用資料集的高層摘要，含 12 項法定內容：
1. 資料集的來源或擁有者。
2. 說明這些資料集如何促進 AI 系統的預期目的。
3. 資料集中的資料點數量（可接受一般範圍；動態資料集可用估計值）。
4. 資料點類型的說明（有標註資料集為標籤類型；無標註者為一般特徵）。
5. 資料集是否包含受著作權、商標或專利保護的資料，或完全屬於公共領域。
6. 資料集是購買或授權取得的。
7. 資料集是否包含個人資訊（依 Cal. Civ. Code §1798.140(v)）。
8. 資料集是否包含彙總的消費者資訊（依 Cal. Civ. Code §1798.140(b)）。
9. 開發者進行的清理、處理或其他修改，及其預期目的。
10. 資料收集的時間區間，若仍在收集中須註明。
11. 資料集在開發過程中首次被使用的日期。
12. 系統是否使用或持續使用合成資料生成。

第 12 項（合成資料）相對於 Gebru et al. 2018 的 datasheet 是新增的。第 7 項（個人資訊）會觸發 Privacy Rights Act (CPRA) 的義務。該法規豁免資安 / 完整性、航空器操作，以及僅限聯邦的國家安全系統（第 3111(b) 條）。

### 歐盟 AI Act（第 24 課）與 TDM opt-out

歐盟著作權指令的文字與資料探勘例外，允許在權利人未 opt-out 的情況下，以公開可得的內容進行訓練。歐盟 AI Act GPAI Code of Practice 的 Copyright 章節，要求 GPAI 提供者尊重機器可讀的 opt-out 訊號（robots.txt、C2PA 的「No AI Training」聲明等）。

### 2025 年 DPA 在正當利益上的趨同

愛爾蘭 DPC（2025 年 5 月 21 日）：在 EDPB 意見之後，接受 Meta 在有防護措施下，以第一方公開的歐盟 / EEA 成人使用者內容進行訓練的計畫。科隆高等法院（2025 年 5 月 23 日）駁回對 Meta 的禁制令：opt-out 已足夠。漢堡 DPA 為了歐盟層級的一致性，撤回緊急程序。英國 ICO（2025 年 9 月 23 日）對 LinkedIn 在類似防護措施與持續監督下恢復 AI 訓練，發出正面的監管回應，並非正式許可。

趨同的原則：正當利益可以正當化以公開可得的第一方內容進行訓練，但須附有 opt-out。不需要取得同意。

### 巴西 ANPD（2024 年 7 月）

因資訊透明度不足，暫停了 Meta 為 AI 訓練處理巴西使用者資料。與歐盟 DPA 的結果不同，ANPD 把透明度的優先順序排在正當利益的可接受性之前。

### 不可逆問題

Cookie 同意是為即時、可逆的追蹤而設計的。訓練資料則不同：資料一旦進入模型權重，就無法外科手術式地移除。從頭重新訓練是唯一完整的補救，而成本高到難以負擔。

部分補救：
- **Unlearning。**近似移除；以 MIA（第 22 課）衡量。
- **基於 influence function 的定位。**找出受該資料影響最大的權重；選擇性更新。
- **Fine-tune 抑制。**訓練模型拒絕輸出由該資料衍生的內容。

沒有一個能完全解決問題。合規的時間窗口在收集資料的當下。

### Data Provenance Initiative

dataprovenance.org。Longpre、Mahari、Lee et al.「Consent in Crisis」(2024 年 7 月)：對 AI 訓練資料公共領域的大規模稽核。發現：發布者正以加速的速度加入 robots.txt 限制。可公開用於訓練的公共領域正在快速萎縮。2023 到 2024 年間，約 25% 的頂級訓練來源加入了某種限制。含意：未來訓練資料的可得性，取決於新的取得方式（授權、合成生成、激勵參與）。

### 這在 Phase 18 中的位置

第 26 課是模型層級的文件。第 27 課是資料集層級的治理。兩者合起來定義了透明度層。第 28 課描繪處理這些問題的研究生態系。

```figure
an-provenance-oneway
```

## 實際使用 (Use It)

`code/main.py` 為一個玩具資料集產生符合 California AB 2013 的 12 欄位資料集摘要骨架。你可以填入欄位，並觀察哪些會觸發隱私或著作權的後續義務。

## 交付 (Ship It)

這堂課會產出 `outputs/skill-provenance-check.md`。給它一個用於訓練的資料集，它會檢查 AB 2013 12 欄位的涵蓋度、opt-out 基礎設施的合規性、DPA 的一致性，以及不可逆風險評估。

## 練習 (Exercises)

1. 執行 `code/main.py`。為一個玩具資料集產生 12 欄位摘要，並找出哪些欄位說明不足。

2. 歐盟著作權指令的 TDM opt-out 是機器可讀的。提出一種 opt-out 訊號的標準格式，並與 robots.txt 和 C2PA「No AI Training」比較。

3. 閱讀 Data Provenance Initiative 的「Consent in Crisis」(2024 年 7 月)。描述限制速度最快的三個內容類別，並論證一項經濟後果。

4. 2025 年 DPA 的趨同接受以正當利益訓練公開內容。建構一個正當利益不足以成立的情境，並找出提供者改而需要的法律依據。

5. 勾勒一份訓練資料來源 manifest，能與 AB 2013 欄位及每個資料集的 C2PA 簽署來源鏈組合。找出一項技術障礙與一項法律障礙。

## 關鍵術語 (Key Terms)

| 術語 | 一般說法 | 實際意義 |
|------|-----------------|------------------------|
| AB 2013 | 「the California law」 | Generative AI 訓練資料透明度；12 個法定欄位 |
| TDM exception | 「text-and-data-mining」 | 歐盟著作權指令中附有 opt-out 的訓練資料例外 |
| Legitimate interest | 「the EU basis」 | 可能正當化以公開內容訓練的 GDPR 第 6 條依據 |
| Opt-out signal | 「machine-readable no-train」 | robots.txt、C2PA「No AI Training」、TDM.Reservation |
| Irreversibility | 「cannot un-train」 | 模型權重中的資料無法被外科手術式移除 |
| Unlearning | 「approximate removal」 | 訓練後的介入措施，用以降低模型對特定資料的依賴 |
| Consent in Crisis | 「the DPI audit」 | 2024 年 7 月關於 robots.txt 限制加速增加的發現 |

## 延伸閱讀 (Further Reading)

- [California AB 2013](https://leginfo.legislature.ca.gov/faces/billNavClient.xhtml?bill_id=202320240AB2013)：Generative AI 訓練資料透明度法
- [EU AI Act + GPAI Code of Practice (Lesson 24)](https://digital-strategy.ec.europa.eu/en/policies/regulatory-framework-ai)：Copyright 章節
- [Longpre, Mahari, Lee et al. — Consent in Crisis (dataprovenance.org, July 2024)](https://www.dataprovenance.org/consent-in-crisis-paper)：DPI 稽核
- [IAPP — EU Digital Omnibus GDPR amendments (2025)](https://iapp.org/news/a/eu-digital-omnibus-amendments-to-gdpr-to-facilitate-ai-training-miss-the-mark)：監管背景
