# LLM 中的 Bias 與表徵性傷害 (Bias and Representational Harm in LLMs)

> Gallegos、Rossi、Barrow、Tanjim、Kim、Dernoncourt、Yu、Zhang、Ahmed (Computational Linguistics 2024, arXiv:2309.00770)。2024 年的奠基性綜述，區分表徵性傷害 (representational harms，刻板印象、抹除) 與分配性傷害 (allocational harms，資源分配不均)，並將評估指標分為 embedding-based、probability-based 與 generated-text-based。2024-2025 年的實證：An et al. (PNAS Nexus, March 2025) 在針對 20 個入門職位的自動化履歷評估上，測量 GPT-3.5 Turbo、GPT-4o、Gemini 1.5 Flash、Claude 3.5 Sonnet、Llama 3-70B 的性別 x 種族交織性 bias。WinoIdentity (COLM 2025, arXiv:2508.07111) 為交織身分引入以不確定性為基礎的 fairness 評估。Yu & Ananiadou 2025 找出 MLP 層中的 gender neuron；Ahsan & Wallace 2025 用 SAE 揭露臨床上的種族 bias；Zhou et al. 2024 (UniBias) 操縱 attention head 來去除 bias。後設批評 (arXiv:2508.11067)：十年的文獻不成比例地聚焦在二元性別 bias。

**Type:** Build
**Languages:** Python (stdlib, toy embedding-based bias probe)
**Prerequisites:** Phase 05 (word embeddings), Phase 18 · 01 (instruction following)
**Time:** ~60 分鐘

## 學習目標 (Learning Objectives)

- 定義表徵性傷害與分配性傷害，並各舉一個 LLM 部署中的例子。
- 說出 Gallegos et al. 2024 的三種評估指標類別，並各描述一個指標。
- 描述交織性 (intersectionality)，以及為什麼 WinoIdentity 以不確定性為基礎的 fairness 測量，補上了單一軸向 bias 評估的缺口。
- 描述三種處理 bias 的 mechanistic-interpretability 做法（gender neuron、SAE feature、attention-head 操縱）。

## 問題 (The Problem)

前面的課程涵蓋刻意的傷害（jailbreak、scheming）與安全治理。Bias 則是無意間浮現的傷害，來自訓練資料分佈、prompt 框架，以及累積的設計選擇。測量與降低它，是與對抗穩健性截然不同的方法論挑戰。

## 核心概念 (The Concept)

### 表徵性與分配性

- **表徵性傷害。**刻板印象、抹除、貶低性的描繪。一個把護理師描繪為全是女性的 LLM，就是在製造表徵性傷害。
- **分配性傷害。**不平等的實質結果。一個系統性地給黑人申請者履歷較低分的 LLM，就是在製造分配性傷害。

這兩者不一樣。模型可以是「表徵上無 bias」（產生多元的描繪），同時卻「分配上有 bias」（做出不平等的推薦）。評估需要兩者都測量。

### 三種評估指標類別 (Gallegos et al. 2024)

- **Embedding-based。**在 RLHF 之前的 embedding 上做 WEAT 式測試。測量身分詞與屬性詞之間的統計關聯。限制：測量的是表示，不是行為。
- **Probability-based。**符合刻板印象與違反刻板印象的 completion 的 log-likelihood。是 decoder 端的測量。能捕捉一部分行為上的 bias。
- **Generated-text-based。**對生成文字的下游任務測量。履歷評分、推薦撰寫、對話。生態效度最高；最難重現。

### 交織性 (Intersectionality)

只對「性別」做 bias 評估，會漏掉只在（性別, 種族）配對上才觸發的 bias。An et al. 2025 發現 GPT-4o 在履歷評分中對黑人女性的懲罰，比對黑人男性、也比對白人女性分別單獨看都更重。單一軸向的評估無法捕捉這點。

WinoIdentity (COLM 2025) 引入以不確定性為基礎的交織 fairness。它測量模型對結果的不確定性，在不同的交織身分組合之間是否不同，而不只是點預測。這能抓到模型在各群體上同樣錯、但對某些群體更不確定的情況，而這會產生不同的下游分配行為。

### Mechanistic 做法

2024-2025 年的 interpretability 研究，讓 bias 可以做 mechanistic 介入：

- **Gender neuron (Yu & Ananiadou 2025)。**特定的 MLP 神經元與性別相關行為相關。消融這些神經元可降低性別差距指標，且能力成本有限。
- **透過 SAE 的臨床種族 bias (Ahsan & Wallace 2025)。**Sparse autoencoder feature 把內部表示分解成可解釋的維度；與種族相關的 feature 可以被找出並抑制。
- **UniBias (Zhou et al. 2024)。**操縱 attention head 做 zero-shot 去 bias。特定的 head 會放大對身分類別的敏感度；將這些 head 歸零或重新加權，無需 fine-tuning 就能降低 bias。

### 後設批評

十年文獻回顧 (arXiv:2508.11067, 2025) 發現，這個領域不成比例地聚焦在二元性別 bias。其他軸向，如身心障礙、宗教、移民身分、多語言身分，得到的關注少得多。這個後設批評主張，狹隘的聚焦可能因為忽略而傷害邊緣群體：一個在二元性別上去 bias 得很好的模型，可能在沒有人檢查的維度上嚴重偏頗。

### 這在 Phase 18 中的位置

第 20 到 21 課正式涵蓋 bias 與 fairness。第 22 課涵蓋隱私。第 23 課涵蓋 watermarking。這些是補充先前欺騙 / 安全層的使用者傷害層。

```figure
an-bias-two-harms
```

## 實際使用 (Use It)

`code/main.py` 建立一個玩具 embedding-based bias probe：在簡單的共現 embedding 中，測量身分詞與屬性詞之間 WEAT 式的距離。你可以注入一個 bias 並觀察指標觸發；套用簡單的去 bias 操作，並觀察部分恢復。

## 交付 (Ship It)

這堂課會產出 `outputs/skill-bias-eval.md`。給它一份 model card 或 fairness 主張，它會依三種指標類別（embedding、probability、generated-text）、交織性的涵蓋範圍，以及任何去 bias 介入的機制，來稽核這個評估。

## 練習 (Exercises)

1. 執行 `code/main.py`。回報去 bias 步驟前後的 WEAT 式 bias 分數。解釋為什麼指標沒有降到零。

2. 用交織性測試擴充 probe：(性別, 種族) x (職涯, 家庭)。回報跨軸向的 bias 分數。

3. 閱讀 An et al. 2025 (PNAS Nexus)。找出他們回報的兩個、單一軸向性別評估會漏掉的交織效應。

4. Yu & Ananiadou 2025 找出 gender neuron。勾勒一個否證實驗，能區分「這些神經元造成性別 bias」與「這些神經元與性別 bias 相關」。

5. 後設批評主張這個領域過度狹隘地聚焦在二元性別。挑一個研究不足的軸向，描述它的表徵性傷害測量協定。

## 關鍵術語 (Key Terms)

| 術語 | 一般說法 | 實際意義 |
|------|-----------------|------------------------|
| Representational harm | 「stereotypes / erasure」 | 對某群體的帶偏見描繪 |
| Allocational harm | 「unequal decisions」 | 對某群體的帶偏見實質結果 |
| WEAT | 「the embedding test」 | Word Embedding Association Test；以共現為基礎的 bias probe |
| Intersectionality | 「combined identity effects」 | 在多個身分軸向的交集處浮現的 bias |
| Gender neurons | 「MLP bias neurons」 | 激活與性別特定行為相關的特定神經元 |
| SAE feature | 「interpretable dimension」 | Sparse autoencoder 找出的 feature；適用於 mechanistic 的 bias 分析 |
| UniBias | 「attention-head debiasing」 | 透過重新加權 attention head 的 zero-shot 去 bias |

## 延伸閱讀 (Further Reading)

- [Gallegos et al. — Bias and Fairness in LLMs: A Survey (arXiv:2309.00770, Computational Linguistics 2024)](https://arxiv.org/abs/2309.00770)：經典綜述
- [An et al. — Intersectional resume-evaluation bias (PNAS Nexus, March 2025)](https://academic.oup.com/pnasnexus/article/4/3/pgaf089/8111343)：五個模型的交織性研究
- [WinoIdentity — uncertainty-based intersectional fairness (arXiv:2508.07111, COLM 2025)](https://arxiv.org/abs/2508.07111)：新的 benchmark
- [UniBias — attention-head manipulation (Zhou et al. 2024, ACL)](https://arxiv.org/abs/2405.20612)：zero-shot 去 bias
