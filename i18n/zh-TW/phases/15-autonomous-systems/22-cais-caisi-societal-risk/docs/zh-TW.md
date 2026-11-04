# CAIS、CAISI 與社會規模的風險 (CAIS, CAISI, and Societal-Scale Risk)

> Center for AI Safety（CAIS，位於舊金山，2022 年由 Hendrycks 與 Zhang 創立）發表了四種風險的框架：惡意使用 (malicious use)、AI 競賽 (AI races)、組織風險 (organizational risks)、失控 AI (rogue AIs)，以及 2023 年 5 月由數百位教授與公司領導人簽署的滅絕風險聲明。CAIS 2026 年的發表包括：用於 frontier 模型評估的 AI Dashboard、Remote Labor Index（與 Scale AI 合作）、Superintelligence Strategy Paper，以及 AI Frontiers 電子報。另一個不同的機構：NIST Center for AI Standards and Innovation (CAISI)，面向美國政府的自願性協議，以及聚焦於網路、生物與化學武器風險的非機密能力評估。CAIS 把組織風險列為四大頂層風險之一：安全文化、嚴謹的稽核、多層防禦與資訊安全是基礎，卻經常為了部署速度而被拿來取捨。加州 SB-53 如果簽署，將成為美國第一個州層級的災難性風險法規。

**Type:** Learn
**Languages:** Python (stdlib, four-risk inventory and mitigation matcher)
**Prerequisites:** Phase 15 · 19 (RSP), Phase 15 · 20 (PF + FSF)
**Time:** ~45 分鐘

## 問題 (The Problem)

第 19 與 20 課涵蓋實驗室內部的擴展政策。第 21 課涵蓋獨立的能力評估。這堂課涵蓋第三個視角：形塑災難性 AI 風險的公共討論與法規基準的公民社會與政府組織。

兩個不同的機構很重要。CAIS 是非營利研究組織，發表思考 AI 風險的框架，並協調公開聲明。CAISI 是 NIST 內的美國政府中心，與實驗室簽訂自願性協議，並執行非機密的能力評估。名稱押韻，但任務並不重疊。實務工作者應該兩者都認識。

實務上的內容：CAIS 的四種風險框架，是文獻中被引用最廣的社會規模風險分類法。安全文化與組織風險是這四者之一，而這是最直接受實務工作者掌控的一項。SB-53（加州）如果簽署，將是美國第一個州層級的災難性風險法規；這項法案的框架很重要，因為在美國科技政策中，州層級的法規歷來領先聯邦行動。

## 核心概念 (The Concept)

### CAIS：Center for AI Safety

- 成立：2022 年於舊金山，由 Dan Hendrycks 與 Oliver Zhang 創立。Hendrycks 是 Executive and Research Director；Zhang 是 Managing Director。
- 地位：501(c)(3) 非營利組織。
- 2023 年值得注意的產出：滅絕風險聲明，由數百位研究者與 CEO 共同簽署。聲明寫道：「降低 AI 造成滅絕的風險，應該與疫情和核戰等其他社會規模的風險並列為全球優先事項。」
- 2026 年的產出：用於 frontier 模型評估的 AI Dashboard、Remote Labor Index（與 Scale AI 合作）、Superintelligence Strategy Paper、AI Frontiers 電子報。

### 四種風險的框架

CAIS 的框架把災難性 AI 風險分成四個頂層類別：

1. **惡意使用 (Malicious use)**：壞人利用 AI 造成傷害（bioweapons 合成、假訊息、網路攻擊）。
2. **AI 競賽 (AI races)**：實驗室、公司或國家之間的競爭壓力，把部署推過安全的界線。
3. **組織風險 (Organizational risks)**：實驗室內部的動態（安全文化失敗、稽核不足、資安資源不足）造成不良的部署。
4. **失控 AI (Rogue AIs)**：能力夠強的 AI 追求與人類福祉衝突的目標。

這不是唯一的分類法；是被引用最多的。這些類別並不互斥：一個由為了速度而犧牲稽核的組織，在競賽中產出的失控 AI，同時涵蓋了這四者。

### 組織風險在哪裡

在這四個類別中，組織風險對實務工作者來說最可行動。實驗室的安全文化、稽核的嚴謹度、防禦的分層與資訊安全，決定它的模型是否真的帶著第 10–18 課的控制措施出貨，還是這些控制措施只是沒人驗證過的檢查清單項目。

具體的組織風險槓桿：

- **安全文化**：團隊成員是否能在不付出職涯代價的情況下提出疑慮？CAIS 的調查發現，這是其他槓桿的強力預測因子。
- **嚴謹的稽核**：外部與內部。只有內部的稽核會產出樂觀的報告。
- **多層防禦**：沒有任何單一層是足夠的（Phase 15 一貫的主題）。
- **資訊安全**：模型權重外洩、eval 資料外洩、繞過 monitor 的技巧外洩。第 19 課的 RAND SL-4 是一個特定的標準。

### CAISI：Center for AI Standards and Innovation

- 隸屬於 NIST。
- 與 frontier 實驗室簽訂自願性協議。
- 發表聚焦於網路、生物與化學武器風險的非機密能力評估。
- 與 CAIS 不同；縮寫相撞；檢查 URL（nist.gov）以確認你在讀的是哪一個。

CAISI 的角色，是 METR 與實驗室私下合作（第 21 課）的公開、面向政府的對應。CAISI 的報告是非機密的；METR 的報告常常受 NDA 限制。兩者都讀的實務工作者，會得到更完整的圖像。

### 加州 SB-53

加州參議院法案（2025–2026 會期）處理 frontier 模型的災難性風險。草案中的主要條款：

- 觸發州層級義務的特定能力門檻。
- 對 AI 實驗室員工的吹哨者保護。
- 對災難性失敗的事件通報要求。

如果簽署，它將是美國第一個州層級的災難性風險法規。不論簽署與否，這項法案的框架都會影響其他州議會如何處理這個問題。加州的實務工作者應該追蹤這項法案的狀態；其他地方的實務工作者應該閱讀它，以了解美國州層級的法規可能長什麼樣子。

### 社會規模的風險不是單層問題

Phase 15 一貫的主題，縱深防禦，在社會層面同樣適用。沒有任何單一組織、法規或框架能封住災難性風險。只有當以下條件都成立，這個生態系才能運作：

- 實驗室發布擴展政策（第 19、20 課）。
- 外部評估者產出量測（第 21 課）。
- 公民社會追蹤並公開（CAIS）。
- 政府執行自願性計畫與基準法規（CAISI、SB-53）。
- 實務工作者建立多層控制（第 10–18 課）。

這是本階段的最終綜合：先前的每一課都是堆疊中的一層，而這個堆疊的完整性，比任何單一層的強度都重要。

```figure
a5-four-risks
```

## 實際使用 (Use It)

`code/main.py` 實作了一個小型風險盤點工具。給定一個提議的部署，它會把該部署對照四種風險類別加上標籤，並回傳緩解措施的檢查清單。它是框架的閱讀輔助工具，不能取代人的判斷。

## 交付 (Ship It)

`outputs/skill-societal-risk-review.md` 審查一個部署的社會規模風險立場：它碰到四個類別中的哪幾個、有哪些緩解措施就位、組織風險的暴露程度如何。

## 練習 (Exercises)

1. 執行 `code/main.py`。餵入三個不同規模的合成部署。確認四種風險的標籤符合你的預期；找出一個工具標籤不足或過度標記的案例。

2. 完整閱讀 CAIS 的四種風險論文。挑一個風險類別，寫兩段文字說明你認為該類別在 2026 年最重要的發展。

3. 閱讀加州 SB-53 目前的草案。找出一項你認為加強了災難性風險立場的條款，以及一項你認為削弱的條款。為兩者說明理由。

4. 挑一個你熟悉的 production AI 部署（你自己的或已公開的）。依組織風險的子槓桿為它評分：安全文化、稽核嚴謹度、多層防禦、資訊安全。哪一項最弱？要把它提升到同等水準要花多少成本？

5. 草擬一個 2028 年版本的四種風險框架，反映多一年的能力與多一年的部署經驗。你會新增、移除或重新分組什麼？

## 關鍵術語 (Key Terms)

| 術語 | 一般說法 | 實際意義 |
|---|---|---|
| CAIS | 「Center for AI Safety」 | 非營利組織；四種風險框架；2023 年的滅絕聲明 |
| CAISI | 「美國政府的 AI 安全」 | NIST 中心；自願性協議；非機密評估 |
| Four-risk framework | 「CAIS 的分類法」 | 惡意使用、AI 競賽、組織風險、失控 AI |
| Malicious use | 「壞人使用 AI」 | Bioweapons、假訊息、網路攻擊 |
| AI races | 「競爭壓力」 | 實驗室/公司/國家把部署推過安全的界線 |
| Organizational risk | 「實驗室內部失敗」 | 安全文化、稽核、防禦、資訊安全 |
| Rogue AI | 「Misaligned 的 agent」 | 有能力的 AI 追求與人類福祉衝突的目標 |
| California SB-53 | 「州層級法規」 | 2025–2026 年的法案；如果簽署，將是美國第一個州層級的災難性風險法規 |

## 延伸閱讀 (Further Reading)

- [Center for AI Safety](https://safe.ai/) — 四種風險框架的機構首頁。
- [CAIS — AI Risks that Could Lead to Catastrophe](https://safe.ai/ai-risk) — 四種風險的論文。
- [CAIS — May 2023 statement on extinction risk](https://safe.ai/statement-on-ai-risk) — 簡短的聯合聲明。
- [NIST CAISI](https://www.nist.gov/caisi) — 面向政府的 AI 標準與創新中心。
- [Anthropic — Measuring agent autonomy in practice](https://www.anthropic.com/research/measuring-agent-autonomy) — 把實驗室層級的承諾與社會規模的框架連結起來。
