# 監管框架：歐盟、美國、英國、韓國 (Regulatory Frameworks — EU, US, UK, Korea)

> 四個主要監管體制定義了 2026 年的 AI 治理版圖。歐盟 AI Act（2024 年 8 月 1 日生效）：禁止性做法與 AI 素養自 2025 年 2 月 2 日起適用；GPAI 義務自 2025 年 8 月 2 日起；全面適用與第 50 條透明度要求為 2026 年 8 月 2 日；既有 GPAI 與嵌入式高風險系統為 2027 年 8 月 2 日；罰款最高可達 15M EUR 或全球營業額的 3%。GPAI Code of Practice（2025 年 7 月 10 日）：三個章節（透明度、著作權、安全與資安）、12 項承諾；執法自 2026 年 8 月開始。UK AISI -> AI Security Institute（2025 年 2 月）：更名意味著範圍縮小。US AISI -> CAISI（2025 年 6 月）：NIST 之下的 Center for AI Standards and Innovation；轉向支持成長的立場。韓國 AI Framework Act（2024 年 12 月通過，2026 年 1 月生效）：第 12 條在 MSIT 之下設立 AISI；要求外國 AI 公司設立當地代表、風險評估，以及針對高影響與生成式 AI 的安全措施。

**Type:** Learn
**Languages:** none
**Prerequisites:** Phase 18 · 18 (frontier frameworks), Phase 18 · 27 (data governance)
**Time:** ~75 分鐘

## 學習目標 (Learning Objectives)

- 描述歐盟 AI Act 的風險分級（禁止、高風險、通用目的、有限風險），以及 2025 年 8 月 / 2026 年 8 月 / 2027 年 8 月的時程。
- 描述 GPAI Code of Practice 的三個章節，以及各自約束哪些提供者。
- 描述 2025 年的更名：UK AISI -> AI Security Institute；US AISI -> CAISI；各自暗示了什麼政策方向。
- 說明韓國 AI Framework Act 的核心條款。

## 問題 (The Problem)

實驗室框架（第 18 課）是自願性的。監管框架則是強制性的。2024 到 2026 年，第一波全面的 AI 法規開始生效。部署者必須把技術控制對應到監管義務；而對應方式因司法管轄區而異。

## 核心概念 (The Concept)

### 歐盟 AI Act

**2024 年 8 月 1 日生效。**風險分級結構：

- **禁止性做法**（第 5 條）。社會評分、公共場所的即時遠端生物辨識（有執法例外）、對弱勢群體的剝削性操縱。2025 年 2 月 2 日起適用。
- **高風險系統**（Annex III）。就業、教育、信用、執法、司法、移民。需要符合性評估、風險管理、記錄、透明度。
- **通用目的 AI (GPAI) 模型**。2025 年 8 月 2 日起適用。所有 GPAI 提供者都有義務；具系統性風險的 GPAI（訓練運算 >1e25 FLOP）有額外義務。
- **有限風險系統**。第 50 條下的透明度義務（AI 生成內容標示）。2026 年 8 月 2 日起適用。

時程：
- 2025 年 2 月 2 日：禁止性做法 + AI 素養。
- 2025 年 8 月 2 日：GPAI + 治理。
- 2026 年 8 月 2 日：全面適用 + 第 50 條透明度 + 罰款最高 15M EUR / 全球營業額 3%。
- 2027 年 8 月 2 日：既有 GPAI + 嵌入式高風險。

執委會在 2025 年底提議將高風險時程調整 16 個月。

### GPAI Code of Practice

2025 年 7 月 10 日發布。三個章節：

- **透明度。**所有 GPAI 提供者。
- **著作權。**所有 GPAI 提供者。
- **安全與資安。**具系統性風險的 GPAI 提供者（估計有 5 到 15 家公司）。

共 12 項承諾。由 AI Office 主持的 Signatory Taskforce 管理實施。執法自 2026 年 8 月 2 日開始；在那之前，接受善意遵循。

### 第 50 條的 Transparency Code

第一版草案 2025 年 12 月 17 日。第二版草案 2026 年 3 月。最終版 2026 年 6 月。涵蓋 AI 生成內容的標示，包括 deepfake，也就是要求第 23 課 watermarking 技術的監管層。

### UK AI Security Institute（2025 年 2 月）

由 AI Safety Institute 更名而來。更名縮小了範圍：捨棄演算法 bias 與言論自由的框架；聚焦於 frontier 能力的安全。開源了 Inspect 評估工具（2024 年 5 月）。與 Redwood（第 10 課）合作 control safety case。

### US CAISI（2025 年 6 月）

川普政府把 NIST 的 AI Safety Institute 轉型為 Center for AI Standards and Innovation。依副總統 Vance 在巴黎 AI Action Summit 的談話，轉向「支持成長的 AI 政策」。降低對部署前評估的重視；強調標準與創新支持。是對歐盟 AI Act 監管立場的國內制衡。

### 韓國 AI Framework Act

2024 年 12 月通過。2025 年 1 月公布。2026 年 1 月生效。整合了 19 項個別的 AI 法案。

第 12 條在科學技術資訊通信部 (MSIT) 之下設立 AISI。要求：
- 在韓國營運的外國 AI 公司須設立當地代表。
- 對「高影響」AI 系統進行風險評估。
- 針對生成式 AI 與高影響 AI 的安全措施。

這是第一個擁有全面性水平 AI 法規的亞洲司法管轄區。

### 跨司法管轄區的動態

- 歐盟：嚴格、風險分級、重罰。是與隱私相關法規的標竿。
- 美國：偏好創新、分散，由各州（例如 California AB 2013，第 27 課）填補聯邦的空缺。
- 英國：狹窄的安全聚焦、強大的評估基礎設施。
- 韓國：由 MSIT 主導，聚焦外國提供者。

互相競爭的監管哲學。在多個司法管轄區的部署者，必須遵循最嚴格的一個，2026 年通常就是歐盟 AI Act。

### 這在 Phase 18 中的位置

第 18 課是實驗室自願性治理；第 24 課是監管；第 25 課是 AI 系統新興的 CVE 類別；第 26 到 27 課涵蓋文件（card）與訓練資料治理。

```figure
an-eu-act-timeline
```

## 實際使用 (Use It)

沒有程式碼。請閱讀歐盟 AI Act 的原始資料：法規文本、GPAI Code of Practice、UK AISI Inspect 框架。把你的部署對應到每個司法管轄區適用的義務。

## 交付 (Ship It)

這堂課會產出 `outputs/skill-regulatory-map.md`。給它一份部署描述，它會對應出適用的司法管轄區、各自的分級、各司法管轄區的義務，以及期限結構。

## 練習 (Exercises)

1. 閱讀歐盟 AI Act（regulation 2024/1689）與 GPAI Code of Practice（2025 年 7 月 10 日）。找出適用於每個 GPAI 提供者的三項義務，以及只適用於具系統性風險 GPAI 的三項義務。

2. 某個部署由美國公司建置，運行在歐盟基礎設施上，並服務韓國使用者。適用哪三個司法管轄區的規則？對每個實質問題，哪一條規則具有約束力？

3. 英國 AI Security Institute 的更名縮小了範圍。論證支持與反對這個較窄的框架。指出每種立場所依賴的政策假設。

4. CAISI 的「支持成長」框架，偏離了 2022-2024 年 AI 安全機構的模式。找出兩項由此框架而來、可衡量的政策轉變。

5. 韓國 AI Framework Act 要求外國提供者設立當地代表。描述這對一家服務韓國使用者的灣區公司在營運上的影響。

## 關鍵術語 (Key Terms)

| 術語 | 一般說法 | 實際意義 |
|------|-----------------|------------------------|
| EU AI Act | 「the regulation」 | 以風險分級為基礎的水平式 AI 法規；2024 年 8 月生效 |
| GPAI | 「general-purpose AI」 | 大型基礎模型；具系統性風險的子集有額外義務 |
| Article 50 | 「transparency obligations」 | AI 生成內容標示；2026 年 8 月適用 |
| UK AISI | 「AI Security Institute」 | 2025 年 2 月更名；範圍較窄的 frontier 安全聚焦 |
| CAISI | 「US center for AI standards」 | 2025 年 6 月由 AI Safety Institute 更名；支持成長的立場 |
| Korean AI Framework Act | 「MSIT horizontal regulation」 | 第一部亞洲全面性 AI 法律；2026 年 1 月生效 |
| Systemic-risk GPAI | 「the 1e25 FLOP threshold」 | 額外義務的層級；估計有 5 到 15 家公司受約束 |

## 延伸閱讀 (Further Reading)

- [EU AI Act text (Regulation 2024/1689)](https://digital-strategy.ec.europa.eu/en/policies/regulatory-framework-ai)：法規與時程
- [GPAI Code of Practice (10 July 2025)](https://digital-strategy.ec.europa.eu/en/library/final-version-general-purpose-ai-code-practice)：三章節的準則
- [UK AI Security Institute (renamed Feb 2025)](https://www.gov.uk/government/organisations/ai-security-institute)：官方頁面
- [CSET — South Korea AI Framework Act Analysis (2025)](https://cset.georgetown.edu/publication/south-korea-ai-law-2025/)：韓國框架分析
