# 內容審核系統：OpenAI、Perspective、Llama Guard (Moderation Systems — OpenAI, Perspective, Llama Guard)

> 量產的內容審核系統，把第 12 到 16 課定義的安全政策付諸實作。OpenAI Moderation API：`omni-moderation-latest`（2024）建立在 GPT-4o 之上，一次呼叫即可分類文字加影像；在多語言測試集上比前一版好 42%；回應結構回傳 13 個類別布林值：harassment、harassment/threatening、hate、hate/threatening、illicit、illicit/violent、self-harm、self-harm/intent、self-harm/instructions、sexual、sexual/minors、violence、violence/graphic；對多數開發者免費。分層模式：Input moderation（生成前）、Output moderation（生成後）、Custom moderation（領域規則）。非同步平行呼叫能隱藏延遲；在 input moderation 完成的期間，可以用佔位回應涵蓋等待。Llama Guard 3/4（第 16 課）：14 個 MLCommons 危害類別、Code Interpreter Abuse、8 種語言（v3）、多影像（v4）。Perspective API (Google Jigsaw)：早於 LLM-as-moderator 浪潮的毒性評分；主要是單一維度的 toxicity，帶有 severe-toxicity/insult/profanity 變體；是內容審核研究的基準線。淘汰：Azure Content Moderator 於 2024 年 2 月被標示為棄用，2027 年 2 月停用，由 Azure AI Content Safety 取代。

**Type:** Build
**Languages:** Python (stdlib, three-layer moderation harness)
**Prerequisites:** Phase 18 · 16 (Llama Guard / Garak / PyRIT)
**Time:** ~60 分鐘

## 學習目標 (Learning Objectives)

- 描述 OpenAI Moderation API 的類別分類法，以及它與 Llama Guard 3 的 MLCommons 集合有何不同。
- 描述三層審核模式（input、output、custom），並各說出一種失敗模式。
- 描述 Perspective API 作為 LLM 時代之前基準線的定位，以及它為什麼仍被用於研究。
- 說明 Azure 的棄用時程。

## 問題 (The Problem)

第 12 到 16 課描述攻擊與防禦工具。第 29 課涵蓋已部署的審核系統，它們在使用者接觸產品的介面上，把防禦付諸實作。三層模式是 2026 年的預設配置。

## 核心概念 (The Concept)

### OpenAI Moderation API

`omni-moderation-latest`（2024）。建立在 GPT-4o 之上。一次呼叫即可分類文字加影像。對多數開發者免費。

類別（回應結構中的 13 個布林值）：
- harassment、harassment/threatening
- hate、hate/threatening
- self-harm、self-harm/intent、self-harm/instructions
- sexual、sexual/minors
- violence、violence/graphic
- illicit、illicit/violent

多模態支援適用於 `violence`、`self-harm` 與 `sexual`，但不包含 `sexual/minors`；其餘為純文字。

在 `code/main.py` 的程式 harness 中，為了教學簡化，我們把 `/threatening`、`/intent`、`/instructions` 和 `/graphic` 子類別合併到它們的頂層父類別。量產程式碼應使用完整的 13 類別結構。

在多語言測試集上比前一代的審核端點好 42%。每個類別都有分數；應用程式自行設定門檻。

### Llama Guard 3/4

已在第 16 課涵蓋。14 個 MLCommons 危害類別（組織方式與 OpenAI 的 13 個回應結構布林值不同）。支援 8 種語言（v3）。Llama Guard 4（2025 年 4 月）是原生多模態，12B。

OpenAI 與 Llama Guard 的分類法有重疊但也有分歧。OpenAI 把「illicit」作為廣泛的類別；Llama Guard 則把「violent crimes」和「non-violent crimes」分開。部署者依自己的政策分類法的契合度來選擇。

### Perspective API (Google Jigsaw)

早於 LLM-as-moderator 浪潮（2020 年之前）的毒性評分系統。類別：TOXICITY、SEVERE_TOXICITY、INSULT、PROFANITY、THREAT、IDENTITY_ATTACK。以單一維度為主要分數（TOXICITY），並有子維度變體。

因為 API 穩定、有文件、並且有多年的校準資料，被廣泛用作內容審核研究的基準線。對於現代與 LLM 相鄰的使用情境，Llama Guard 或 OpenAI Moderation 通常更合適。

### 三層模式

1. **Input moderation。**在生成前分類使用者的 prompt。被標記就拒絕。延遲：一次分類器呼叫。
2. **Output moderation。**在交付前分類模型的輸出。被標記就換成拒絕訊息。延遲：生成後一次分類器呼叫。
3. **Custom moderation。**領域專屬規則（regex、allowlist、商業政策）。可在 input 或 output 執行。

這三層在設計上是循序的：input moderation 必須在生成前完成，output moderation 在生成後執行。平行性適用於單一層內：對同一段文字同時執行多個分類器（例如 OpenAI Moderation + Llama Guard + Perspective），可隱藏各分類器的延遲。作為選用的最佳化，可以在 input moderation 完成期間顯示佔位回應（「稍等，正在檢查……」），並延後第一個 token 的串流。被標記時的行為可設定：拒絕、清理、升級給人工審查。

### 失敗模式

- **只有 input。**抓不到 output 的幻覺（第 12 到 14 課的編碼攻擊能繞過 input 分類器）。
- **只有 output。**任何輸入都能到達模型；增加成本；向攻擊者暴露內部推理。
- **只有 custom。**對各類別不夠穩健；regex 很脆弱。

預設是分層。腰帶加吊帶。

### Azure 的棄用

Azure Content Moderator：2024 年 2 月棄用，2027 年 2 月停用。由 Azure AI Content Safety 取代，後者以 LLM 為基礎，並與 Azure OpenAI 整合。這項遷移是 Azure 部署在 2024 到 2027 年間，欄位層級的專案。

### 這在 Phase 18 中的位置

第 16 課在 red-team 的脈絡中涵蓋審核工具。第 29 課涵蓋已部署的審核。第 30 課以目前的雙用途能力證據收尾。

```figure
an-moderation-layers
```

## 實際使用 (Use It)

`code/main.py` 建立一個三層審核 harness：input 審核器（關鍵字加類別分數）、output 審核器（對輸出使用相同分類器）、custom 審核器（領域規則）。你可以讓輸入通過這三層，並觀察哪一層攔下了什麼。

## 交付 (Ship It)

這堂課會產出 `outputs/skill-moderation-stack.md`。給它一個部署，它會推薦審核堆疊的配置：input 用哪個分類器、output 用哪個、哪些 custom 規則，以及邊緣案例用什麼裁判。

## 練習 (Exercises)

1. 執行 `code/main.py`。讓無害、邊界與有害的輸入各通過全部三層。回報每個輸入是哪一層觸發。

2. 用 Perspective API 風格的毒性評分，針對某個特定類別擴充 harness。比較它的門檻行為與類別分數。

3. 閱讀 OpenAI Moderation API 文件與 Llama Guard 3 的類別清單。把每個 OpenAI 類別對應到最接近的 Llama Guard 類別。找出三個無法乾淨對應的類別。

4. 為程式碼助理部署（例如 GitHub Copilot）設計審核堆疊。找出最相關與最不相關的類別，並提出 custom 規則。

5. Azure Content Moderator 在 2027 年 2 月停用。規劃遷移到 Azure AI Content Safety。找出遷移中風險最高的環節。

## 關鍵術語 (Key Terms)

| 術語 | 一般說法 | 實際意義 |
|------|-----------------|------------------------|
| OpenAI Moderation | 「omni-moderation-latest」 | 以 GPT-4o 為基礎的 13 類別（文字）分類器，部分支援多模態 |
| Perspective API | 「Google Jigsaw toxicity」 | LLM 時代之前的毒性評分基準線 |
| Llama Guard | 「MLCommons 14-category」 | Meta 的危害分類器（v3：8B 文字、8 種語言；v4：12B 多模態） |
| Input moderation | 「pre-generation filter」 | 在呼叫模型前，對使用者 prompt 的分類器 |
| Output moderation | 「post-generation filter」 | 在交付前，對模型輸出的分類器 |
| Custom moderation | 「domain rules」 | 部署專屬的規則（regex、allowlist、政策） |
| Layered moderation | 「all three layers」 | 標準的量產部署模式 |

## 延伸閱讀 (Further Reading)

- [OpenAI Moderation API docs](https://platform.openai.com/docs/api-reference/moderations)：omni-moderation 端點
- [Meta PurpleLlama + Llama Guard](https://github.com/meta-llama/PurpleLlama)：Llama Guard repo
- [Google Jigsaw Perspective API](https://perspectiveapi.com/)：毒性評分
- [Azure AI Content Safety](https://learn.microsoft.com/en-us/azure/ai-services/content-safety/)：Azure 的替代方案
