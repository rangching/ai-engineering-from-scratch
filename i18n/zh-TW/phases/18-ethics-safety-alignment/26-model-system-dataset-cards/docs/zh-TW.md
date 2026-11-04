# Model Card、System Card 與 Dataset Card (Model, System, and Dataset Cards)

> 三種文件格式構成了 AI 透明度的骨幹。Model Card (Mitchell et al. 2019)：模型的營養標示，涵蓋訓練資料、依群體拆解的量化分析、倫理考量、注意事項；Hugging Face 上只有 0.3% 的 model card 記載了倫理考量 (Oreamuno et al. 2023)。Datasheets for Datasets (Gebru et al. 2018, CACM)：動機、組成、收集過程、標註、散布、維護；類比電子元件的 datasheet。Data Cards (Pushkarna et al., Google 2022)：模組化、分層的細節（telescopic、periscopic、microscopic），作為服務多元讀者的邊界物件。2024-2025 年的發展：以 LLM 自動生成 (CardGen, Liu et al. 2024)；model card 的詳細程度與 HF 下載量最多增加 29% 相關 (Liang et al. 2024)；可驗證的證明 (Laminator, Duddu et al. 2024)；加入碳 / 水的永續性報告 (Jouneaux et al. July 2025)；歐盟 / ISO 的監管 card 正在浮現。System Card (Sidhpurwala 2024；Meta 系統層級的透明度；「Blueprints of Trust」arXiv:2509.20394)：端到端的 AI 系統文件，涵蓋資安能力、prompt-injection 防護、資料外洩偵測、與人類價值的對齊。

**Type:** Build
**Languages:** Python (stdlib, model-card + datasheet + system-card generator)
**Prerequisites:** Phase 18 · 18 (safety frameworks), Phase 18 · 24 (regulatory)
**Time:** ~60 分鐘

## 學習目標 (Learning Objectives)

- 描述 Mitchell et al. 2019 最初的 model card 與 Gebru et al. 2018 的 datasheet。
- 描述 Data Card 的 telescopic/periscopic/microscopic 分層。
- 描述 System Card 及其端到端的涵蓋範圍。
- 說出 2024-2025 年的三項發展（自動生成、可驗證的證明、永續性報告）。

## 問題 (The Problem)

監管框架（第 24 課）與實驗室安全政策（第 18 課）都要求文件。文件格式從針對模型 (model card)，演進到針對資料集 (datasheet)，再到針對系統 (system card)。每一種處理的是不同範圍的透明度。2024-2025 年的自動化與可驗證證明工作，處理的是長期存在的採用問題。

## 核心概念 (The Concept)

### Model Card (Mitchell et al. 2019)

章節：
- 模型細節。
- 預期用途。
- 因素（與評估相關的人口統計或環境因素）。
- 指標。
- 評估資料。
- 訓練資料。
- 量化分析（依因素拆解）。
- 倫理考量。
- 注意事項與建議。

採用問題：Oreamuno et al. 2023 對 Hugging Face model card 的稽核發現，只有 0.3% 記載了倫理考量。

### Datasheets for Datasets (Gebru et al. 2018)

類比電子元件的 datasheet。章節：
- 動機（為什麼建立這個資料集）。
- 組成（裡面有什麼）。
- 收集過程（如何組裝）。
- 標註（如適用）。
- 用途（預期、禁止、風險）。
- 散布。
- 維護。

發表於 CACM 2021。Datasheet 是上游文件；model card 依賴 datasheet 的準確性。

### Data Cards (Pushkarna et al., Google 2022)

模組化的分層細節。三個縮放層級：
- **Telescopic。**給非專家的高層摘要。
- **Periscopic。**給 ML 從業者的中層概覽。
- **Microscopic。**給稽核者的詳細特徵層級文件。

邊界物件的框架：不同的讀者從同一份文件中擷取不同的資訊。

### System Card

範圍：端到端的 AI 系統，包括模型、安全堆疊與部署情境。章節通常包括：
- 資安能力。
- Prompt-injection 防護。
- 資料外洩偵測。
- 與所宣稱人類價值的對齊。
- 事件回應。

Sidhpurwala 2024 與 Meta 系統層級的透明度工作。「Blueprints of Trust」(arXiv:2509.20394) 把 System Card 正式定義為 Model Card 在部署層的互補物。

### 2024-2025 年的發展

- **CardGen (Liu et al. 2024)。**以 LLM 自動生成 model card；回報在標準化的 Mitchell 2019 欄位上，客觀性高於許多人工撰寫的 card。
- **下載量相關性 (Liang et al. 2024)。**詳細的 model card 與 HF 上最多高出 29% 的下載率相關，採用的壓力現在是由市場驅動，而不只是由合規驅動。
- **Laminator (Duddu et al. 2024)。**透過硬體 TEE / 密碼學簽章的可驗證證明，讓 model card 能攜帶主張的證據，而不只是主張。
- **永續性 (Jouneaux et al. July 2025)。**加入碳、水與運算能耗足跡；ISO 標準正在浮現。
- **監管 card。**歐盟 AI Act（第 24 課）GPAI Code of Practice 的 Transparency 章節，要求 model card 作為合規成果。

### 這在 Phase 18 中的位置

第 24 到 25 課是監管與 CVE 層。第 26 課是文件層。第 27 課是訓練資料治理，也就是 datasheet 的上游。第 28 課是產出 card 中所引用之評估的研究生態系。

```figure
an-card-scopes
```

## 實際使用 (Use It)

`code/main.py` 為一個玩具部署產生最精簡的 model card、datasheet 和 system card。每一個都遵循標準的章節結構。你可以檢視格式，並比較三者的範圍。

## 交付 (Ship It)

這堂課會產出 `outputs/skill-card-audit.md`。給它一份 model card、datasheet 或 system card，它會稽核章節涵蓋度、數值的拆解，以及是否有可驗證的證明。

## 練習 (Exercises)

1. 執行 `code/main.py`。檢視產生的 card。找出薄弱的章節（只有佔位內容），並指定什麼證據能強化它們。

2. 用橫跨兩個人口統計群體的量化拆解分析（第 20 課）擴充 model card。

3. 閱讀 Oreamuno et al. 2023 關於 0.3% 採用率的研究。對 model card 規格提出一項結構性改變，以提高倫理考量的採用率。

4. Laminator (Duddu et al. 2024) 使用 TEE 做可驗證的證明。設計一個攜帶評估結果之密碼學證明的 model-card 欄位，並描述驗證者的角色。

5. 為你過去的某個專案或一個假想的部署撰寫 System Card（是 System Card，不是 Model Card）。找出對第三方稽核者價值最高的章節。

## 關鍵術語 (Key Terms)

| 術語 | 一般說法 | 實際意義 |
|------|-----------------|------------------------|
| Model Card | 「the Mitchell card」 | Mitchell et al. 2019 的 ML 模型標準文件 |
| Datasheet | 「the Gebru datasheet」 | Gebru et al. 2018 的資料集標準文件 |
| Data Card | 「the Pushkarna card」 | Google 2022 的模組化分層資料文件 |
| System Card | 「the deployment card」 | 包含安全堆疊的端到端 AI 系統文件 |
| Boundary object | 「different readers, one doc」 | Data Card 的框架：同一份文件服務多元的讀者 |
| Verifiable attestation | 「the Laminator attestation」 | 附加在文件主張上的密碼學或 TEE 證明 |
| Sustainability field | 「carbon / water footprint」 | 2025 年新興的環境核算欄位 |

## 延伸閱讀 (Further Reading)

- [Mitchell et al. — Model Cards for Model Reporting (arXiv:1810.03993, FAT* 2019)](https://arxiv.org/abs/1810.03993)：經典的 model card
- [Gebru et al. — Datasheets for Datasets (CACM 2021, arXiv:1803.09010)](https://arxiv.org/abs/1803.09010)：datasheet 論文
- [Pushkarna et al. — Data Cards (Google 2022)](https://arxiv.org/abs/2204.01075)：分層的資料文件
- [Sidhpurwala et al. — Blueprints of Trust (arXiv:2509.20394)](https://arxiv.org/abs/2509.20394)：System Card 的正式定義
