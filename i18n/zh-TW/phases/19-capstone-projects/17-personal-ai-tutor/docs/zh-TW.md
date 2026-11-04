# Capstone 17 — 個人 AI 家教 (Personal AI Tutor，適性、多模態、具記憶)

> Khanmigo (Khan Academy)、Duolingo Max、Google LearnLM / Gemini for Education、Quizlet Q-Chat 和 Synthesis Tutor，在 2026 年都大規模推出了適性的多模態家教。共同的形態是：Socratic 策略 (絕不直接丟出答案)、每次互動後更新的學習者模型 (Bayesian knowledge tracing 風格)、語音 + 文字 + 拍照解數學的輸入、課程 graph 檢索、間隔重複排程，以及針對適齡內容的嚴格安全過濾。這個 capstone 要交付一個特定科目的家教 (K-12 代數或入門 Python)，與 10 位學習者進行為期兩週的成效研究，並通過內容安全稽核。

**Type:** Capstone
**Languages:** Python (backend, learner model), TypeScript (web app), SQL (curriculum graph via Postgres + Neo4j)
**Prerequisites:** Phase 5 (NLP), Phase 6 (speech), Phase 11 (LLM engineering), Phase 12 (multimodal), Phase 14 (agents), Phase 17 (infrastructure), Phase 18 (safety)
**Phases exercised:** P5 · P6 · P11 · P12 · P14 · P17 · P18
**Time:** 30 小時

## 問題 (Problem)

適性教學過去是教育科技的研究利基。到了 2026 年，它已是消費性產品。Khanmigo 部署在大多數美國學區。Duolingo Max 達到數千萬月活躍使用者。Google 的 LearnLM / Gemini for Education 支撐 Google Classroom 中的教學。Quizlet Q-Chat 與字卡並列。Synthesis Tutor 以給好奇孩子的家教爆紅。共同要素是：多模態輸入 (打字、說話、拍攝方程式)、Socratic 教學法 (先問、後解釋)、每次互動後更新的學習者模型，以及嚴格的適齡安全。

你要為特定族群打造其中一個。衡量標準是一場真實的成效研究：10 位學習者、兩週內的前測與後測分數。語音迴圈必須感覺自然 (capstone 03 的子堆疊)。記憶必須尊重隱私。安全過濾必須通過針對 K-12 的 COPPA 意識 red-team。

## 核心概念 (Concept)

四個元件。**家教策略** 是 Socratic 迴圈：學習者要求答案時，策略會提出引導性的問題；答對時，進入下一個概念；卡住時，提供有鷹架的提示。**學習者模型** 是 Bayesian knowledge tracing (或簡單變體)，在每次互動後更新每個課程節點的精熟機率。**課程 graph** 是以 Neo4j 儲存、帶有先備邊的概念圖；策略走訪此圖來挑選下一個概念。**記憶** 是情節 + 語意儲存 (agentmemory 風格)，保存過往互動、錯誤與偏好。

UX 是多模態的。打字答案用文字輸入。語音輸入透過 LiveKit + Whisper (重用 capstone 03)。數學題的照片輸入透過 dots.ocr 或 PaliGemma 2。語音輸出透過 Cartesia Sonic-2。安全使用 Llama Guard 4 加上適齡過濾器 (阻擋成人內容、暴力、自傷)，以及具 COPPA 意識的記憶保留政策。

成效研究就是交付成果。10 位學習者、前測與後測、兩週。回報學習增益差異與信賴區間。與非適性基準線 (相同內容、線性呈現、沒有家教策略) 比較。

## 架構 (Architecture)

```
learner device
  |
  +-- text         -> web app
  +-- voice        -> LiveKit Agents (ASR + TTS)
  +-- photo math   -> dots.ocr / PaliGemma 2
       |
       v
  tutor policy (LangGraph)
       - Socratic decision head
       - next-concept chooser (curriculum graph walk)
       - hint scaffolder
       - mastery update
       |
       v
  learner model (BKT / item-response theory)
       - per-concept mastery probability
       - spaced-repetition scheduler (SM-2 or FSRS)
       |
       v
  memory (agentmemory-style)
       - episodic: every interaction
       - semantic: learned mistakes, preferences
       - retention policy: COPPA / GDPR aware
       |
       v
  curriculum graph (Neo4j)
       - prerequisite edges
       - OER content attached
       |
       v
  safety:
    Llama Guard 4 + age-appropriate filter
    memory access guarded by learner ID scope
```

## 技術堆疊 (Stack)

- 科目選擇：K-12 代數或入門 Python (擇一以求深入)
- 家教策略：在 Claude Sonnet 4.7 之上的 LangGraph (啟用 prompt caching)
- 學習者模型：Bayesian knowledge tracing (經典) 或用於間隔的 FSRS
- 課程 graph：Neo4j 儲存概念 + 先備邊 + OER 內容
- 記憶：agentmemory 風格的持久化向量 + 情節 + 語意儲存
- 語音：LiveKit Agents 1.0 + Cartesia Sonic-2 (重用 capstone 03 子堆疊)
- 拍照解數學：dots.ocr 或 PaliGemma 2 做方程式辨識
- 安全：Llama Guard 4 + 自訂適齡過濾器
- 評測：Bloom 層級的題目產生、前後測 harness、成效研究工具

```figure
cf-tutor-loop
```

## 動手做 (Build It)

1. **課程 graph。** 建立包含 50-150 個概念節點的 Neo4j (例如從「數線」到「二次公式」的 K-12 代數)，附先備邊。為每個節點附上 OER 內容 (Open Textbook、OpenStax)。

2. **學習者模型。** 用先驗初始化 Bayesian knowledge tracing：guess、slip、learn-rate。每次互動後更新每個概念的精熟度。依學習者持久化。

3. **家教策略。** 具備下列節點的 LangGraph：`read_signal` (學習者的答案是正確 / 部分 / 卡住？)、`select_concept` (走訪課程 graph，挑選最高優先順序的概念)、`scaffold` (Socratic prompt)、`update_mastery`。

4. **記憶。** 每次互動都寫入情節儲存。錯誤與偏好晉升為語意記憶。具 COPPA 意識的保留政策：1 年後自動刪除，家長可存取。

5. **語音路徑。** 附加到家教策略的 LiveKit Agents worker。ASR 用 Whisper-v3-turbo。TTS 用 Cartesia Sonic-2。支援 barge-in (重用 capstone 03 的機制)。

6. **拍照解數學路徑。** 上傳或拍攝圖片；執行 dots.ocr 或 PaliGemma 2 辨識方程式；以結構化輸入送給家教。

7. **安全。** 每個模型輸出都通過 Llama Guard 4 + 適齡過濾器 (阻擋自傷、成人內容、暴力)。記憶存取以學習者 ID 界定範圍；提供家長刪除用的存取入口。

8. **成效研究。** 10 位學習者、前測 (標準化的 30 題基準)、兩週的家教互動 (每週 3 次)、後測。在相同內容上，與 10 位學習者組成的非適性基準線組比較。

9. **每週進度報告。** 對每位學習者，自動產生 PDF 摘要，包含探索過的主題、精熟度軌跡與建議的下一步。

## 實際使用 (Use It)

```
learner: "I don't understand why 3x + 6 = 12 means x = 2"
[signal]   stuck
[concept]  'isolating variables' (prerequisite: addition-subtraction-equality)
[scaffold] "what number would you subtract from both sides to start?"
learner: "6"
[signal]   correct
[mastery]  addition-subtraction-equality: 0.62 -> 0.77
[concept]  continue 'isolating variables'
[scaffold] "great. now what is 3x / 3 equal to?"
```

## 交付 (Ship It)

`outputs/skill-ai-tutor.md` 是交付成果。一個特定科目的適性家教，具備多模態輸入、學習者模型、記憶、安全，以及可衡量的成效。

| 權重 | 評分項目 | 衡量方式 |
|:-:|---|---|
| 25 | 學習增益差異 | 10 位學習者、兩週研究中的前後測差異 |
| 20 | Socratic 忠實度 | 對逐字稿樣本的評分標準分數 |
| 20 | 多模態 UX | 語音 + 照片 + 文字的端到端一致性 |
| 20 | 安全 + 隱私立場 | Llama Guard 4 通過率 + 具 COPPA 意識的保留 |
| 15 | 課程廣度與 graph 品質 | 概念涵蓋度 + 先備 graph 一致性 |
| **100** | | |

## 練習 (Exercises)

1. 在有與沒有適性學習者模型 (隨機概念順序) 的情況下執行成效研究。回報差異。預期適性版本會勝出，但差距的大小才是有趣的數字。

2. 加入多模態探測：同一個概念問題分別以文字、語音和照片傳遞。衡量學習者使用偏好的模態時是否收斂得更快。

3. 建立家長 dashboard：練習過的主題、精熟度軌跡、即將學的概念、安全事件 (任何護欄命中)。符合 COPPA。

4. 加入語言切換模式：家教接受西班牙語輸入並以西班牙語教學。衡量 X-Guard 的涵蓋度。

5. 對記憶隱私做壓力測試：驗證學習者 A 即使透過語音片段重新匯入攻擊，也看不到學習者 B 的資料。記錄嘗試存取的行為並發出警示。

## 關鍵術語 (Key Terms)

| 術語 | 一般說法 | 實際意義 |
|------|-----------------|------------------------|
| Socratic policy | 「問，不要直接給」 | 家教提出引導性問題，而不是直接給答案 |
| Bayesian knowledge tracing | 「BKT」 | 計算每個概念精熟機率的經典學習者模型方程式 |
| FSRS | 「Free Spaced Repetition Scheduler」 | 2024 年的間隔重複排程器，比 SM-2 更好 |
| Curriculum graph | 「概念 DAG」 | 帶有先備邊的概念 Neo4j |
| Episodic memory | 「每次互動紀錄」 | 儲存每次互動以供日後檢索 |
| Semantic memory | 「學到的模式儲存」 | 從情節記憶中晉升、經壓縮的錯誤與偏好 |
| COPPA | 「兒童隱私法」 | 限制向 13 歲以下兒童蒐集資料的美國法律 |

## 延伸閱讀 (Further Reading)

- [Khanmigo (Khan Academy)](https://www.khanmigo.ai) — 面向消費者的 K-12 家教參考
- [Duolingo Max](https://blog.duolingo.com/duolingo-max/) — 語言學習家教參考
- [Google LearnLM / Gemini for Education](https://blog.google/products-and-platforms/products/education/google-learnlm-gemini-generative-ai/) — 託管的參考模型
- [Quizlet Q-Chat](https://quizlet.com) — 另一個參考
- [Synthesis Tutor](https://www.synthesis.com) — 新創公司參考
- [FSRS algorithm](https://github.com/open-spaced-repetition/fsrs4anki) — 間隔重複排程器
- [Bayesian Knowledge Tracing](https://en.wikipedia.org/wiki/Bayesian_knowledge_tracing) — 學習者模型經典
- [LiveKit Agents](https://github.com/livekit/agents) — 語音堆疊
