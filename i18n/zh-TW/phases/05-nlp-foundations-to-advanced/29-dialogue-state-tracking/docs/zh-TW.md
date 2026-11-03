# 對話狀態追蹤 (Dialogue State Tracking)

> 「I want a cheap restaurant in the north... actually make it moderate... and add Italian.」三輪對話，三次狀態更新。DST 讓 slot-value 字典保持同步，訂位才能成功。

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 5 · 17 (Chatbots), Phase 5 · 20 (Structured Outputs)
**Time:** ~75 分鐘

## 問題 (The Problem)

在任務導向的對話系統中，使用者的目標被編碼為一組 slot-value 配對：`{cuisine: italian, area: north, price: moderate}`。使用者的每一輪都可能新增、變更或移除一個 slot。系統必須讀完整段對話，並正確輸出目前的狀態。

只要一個 slot 錯了，系統就會訂錯餐廳、排錯班機，或刷錯卡。DST 是使用者說了什麼與後端執行什麼之間的樞紐。

為什麼儘管有 LLM，它在 2026 年仍然重要：

- 對合規敏感的領域（銀行、醫療、航空訂位）要求確定性的 slot 值，而不是自由格式的生成。
- 使用工具的 agent 在呼叫 API 之前，仍然需要 slot 解析。
- 多輪的修正比看起來更難："actually no, make it Thursday."

現代 pipeline：傳統 DST 概念，加上 LLM 抽取器，再加上結構化輸出的 guardrail。

## 核心概念 (The Concept)

![DST: dialog history → slot-value state](../assets/dst.svg)

**任務結構。** Schema 定義領域 (domain)（餐廳、飯店、計程車）及其 slot（cuisine、area、price、people）。每個 slot 可以是空的、填入封閉集合中的值（price: {cheap, moderate, expensive}），或是自由格式的值（name: "The Copper Kettle"）。

**兩種 DST 的表述方式。**

- **分類。** 對每一組 (slot, candidate_value) 配對，預測是或否。適用於封閉詞彙的 slot。2020 年前的標準做法。
- **生成。** 給定對話，把 slot 值生成為自由文字。適用於開放詞彙的 slot。現代的預設。

**指標。** Joint Goal Accuracy (JGA)：*所有* slot 都正確的輪次所占比例。全有或全無。2026 年，MultiWOZ 2.4 排行榜的最高約為 83%。

**架構。**

1. **Rule-based（slot regex 加關鍵字）。** 窄領域的強力 baseline。好除錯。
2. **TripPy / BERT-DST。** 以 BERT 編碼、基於複製 (copy) 的生成。LLM 之前的標準。
3. **LDST（LLaMA + LoRA）。** 以領域-slot prompt 做 instruction tuning 的 LLM。在 MultiWOZ 2.4 上達到 ChatGPT 等級的品質。
4. **Ontology-free（2024 到 2026）。** 跳過 schema；直接生成 slot 名稱和值。能處理開放領域。
5. **Prompt 加結構化輸出（2024 到 2026）。** 搭配 Pydantic schema 與 constrained decoding 的 LLM。5 行程式碼，可用於正式環境。

### 經典的失敗模式

- **跨輪次的 Co-reference。** "Let's stay with the first option." 需要解析是哪一個選項。
- **覆寫 vs 附加。** 使用者說 "add Italian."，你要取代 cuisine 還是附加？
- **隱含的確認。** "OK cool"，這算接受了提供的訂位嗎？
- **修正。** "Actually make it 7 pm." 必須更新時間，而不清除其他 slot。
- **對先前系統發言的 coreference。** "Yes, that one." 哪個 "that"？

```figure
n5-slot-tracker
```

## 動手做 (Build It)

### 步驟 1：rule-based slot 抽取器

見 `code/main.py`。Regex 加同義詞字典，可涵蓋窄領域中 70% 的標準語句：

```python
CUISINE_SYNONYMS = {
    "italian": ["italian", "pasta", "pizza", "italy"],
    "chinese": ["chinese", "chow mein", "noodles"],
}


def extract_cuisine(utterance):
    for canonical, synonyms in CUISINE_SYNONYMS.items():
        if any(syn in utterance.lower() for syn in synonyms):
            return canonical
    return None
```

超出標準詞彙就很脆弱。適合確定性的 slot 確認。

### 步驟 2：狀態更新迴圈

```python
def update_state(state, utterance):
    new_state = dict(state)
    for slot, extractor in SLOT_EXTRACTORS.items():
        value = extractor(utterance)
        if value is not None:
            new_state[slot] = value
    for slot in NEGATION_CLEARS:
        if is_negated(utterance, slot):
            new_state[slot] = None
    return new_state
```

三個不變式：

- 絕不重設使用者沒碰的 slot。
- 明確的否定（"never mind the cuisine"）必須清除。
- 使用者的修正（"actually..."）必須覆寫，而不是附加。

### 步驟 3：以 LLM 驅動、搭配結構化輸出的 DST

```python
from pydantic import BaseModel
from typing import Literal, Optional
import instructor

class RestaurantState(BaseModel):
    cuisine: Optional[Literal["italian", "chinese", "indian", "thai", "any"]] = None
    area: Optional[Literal["north", "south", "east", "west", "center"]] = None
    price: Optional[Literal["cheap", "moderate", "expensive"]] = None
    people: Optional[int] = None
    day: Optional[str] = None


def llm_dst(history, llm):
    prompt = f"""You track the slot values of a restaurant booking across turns.
Dialogue so far:
{render(history)}

Update the state based on the latest user turn. Output only the JSON state."""
    return llm(prompt, response_model=RestaurantState)
```

Instructor 加 Pydantic 保證得到有效的狀態物件。沒有 regex，沒有 schema 不匹配，沒有幻覺出來的 slot。

### 步驟 4：JGA 評估

```python
def joint_goal_accuracy(predicted_states, gold_states):
    correct = sum(1 for p, g in zip(predicted_states, gold_states) if p == g)
    return correct / len(predicted_states)
```

校準：系統有多少比例的輪次能把所有 slot 都做對？在 MultiWOZ 2.4 上，2026 年頂尖系統約為 80 到 83%。你的領域內系統在你窄小的詞彙上應該超過這個數字，否則 LLM baseline 就會打敗你。

### 步驟 5：處理修正

```python
CORRECTION_CUES = {"actually", "no wait", "on second thought", "change that to"}


def is_correction(utterance):
    return any(cue in utterance.lower() for cue in CORRECTION_CUES)
```

偵測到修正時，覆寫最後更新的 slot，而不是附加。沒有 LLM 的協助很難做對。現代模式：一律讓 LLM 從歷史中重新生成整個狀態，而不是增量更新，這自然就能處理修正。

## 陷阱

- **整段歷史重新生成的成本。** 讓 LLM 每輪重新生成狀態，總 token 數是 $O(n^2)$。要限制歷史長度，或摘要較舊的輪次。
- **Schema 漂移。** 事後加入新 slot 會破壞舊的訓練資料。要為你的 schema 做版本控制。
- **大小寫敏感。** "Italian" vs "italian" vs "ITALIAN"，到處都要正規化。
- **隱含的繼承。** 如果使用者先前指定了 "for 4 people"，之後針對不同時間的新請求不該清除人數。永遠傳入完整的歷史。
- **自由格式 vs 封閉集合。** 名稱、時間和地址需要自由格式的 slot；cuisine 和 area 則是封閉的。在 schema 中兩者混用。

## 實際使用 (Use It)

2026 年的技術棧：

| 情境 | 做法 |
|-----------|----------|
| 窄領域（一兩個 intent） | Rule-based 加 regex |
| 寬領域，有標註資料 | LDST（在 MultiWOZ 風格資料上用 LLaMA + LoRA） |
| 寬領域，沒有標籤，可上線 | LLM 加 Instructor 加 Pydantic schema |
| 口語 / 語音 | ASR 加正規化器加 LLM-DST |
| 多領域訂位流程 | 每個領域各有 Pydantic 模型、由 schema 引導的 LLM |
| 對合規敏感 | Rule-based 為主，LLM 為 fallback，並搭配確認流程 |

## 交付 (Ship It)

存成 `outputs/skill-dst-designer.md`：

```markdown
---
name: dst-designer
description: Design a dialogue state tracker — schema, extractor, update policy, evaluation.
version: 1.0.0
phase: 5
lesson: 29
tags: [nlp, dialogue, task-oriented]
---

Given a use case (domain, languages, vocab openness, compliance needs), output:

1. Schema. Domain list, slots per domain, open vs closed vocabulary per slot.
2. Extractor. Rule-based / seq2seq / LLM-with-Pydantic. Reason.
3. Update policy. Regenerate-whole-state / incremental; correction handling; negation handling.
4. Evaluation. Joint Goal Accuracy on a held-out dialogue set, slot-level precision/recall, confusion on the hardest slot.
5. Confirmation flow. When to explicitly ask the user to confirm (destructive actions, low-confidence extractions).

Refuse LLM-only DST for compliance-sensitive slots without a rule-based secondary check. Refuse any DST that cannot roll back a slot on user correction. Flag schemas without version tags.
```

## 練習 (Exercises)

1. **簡單。** 為 3 個 slot（cuisine、area、price）建立 `code/main.py` 中的 rule-based 狀態追蹤器。在 10 段手寫的對話上測試。測量 JGA。
2. **中等。** 用 Instructor 加 Pydantic 加小型 LLM 處理同一個資料集。比較 JGA。檢視最難的幾輪。
3. **困難。** 兩者都實作並做路由：rule-based 為主，當 rule-based 輸出少於 2 個有信心的 slot 時改用 LLM fallback。測量合併後的 JGA，以及每輪的推論成本。

## 關鍵術語 (Key Terms)

| 術語 | 一般人怎麼說 | 實際意義 |
|------|-----------------|-----------------------|
| DST | 對話狀態追蹤 | 在對話輪次之間維護 slot-value 字典。 |
| Slot | 使用者意圖的單位 | 後端需要的具名參數（cuisine、date）。 |
| Domain | 任務領域 | 餐廳、飯店、計程車，各有一組 slot。 |
| JGA | Joint Goal Accuracy | 所有 slot 都正確的輪次所占比例。全有或全無。 |
| MultiWOZ | 那個 benchmark | 多領域 WOZ 資料集；標準的 DST 評估。 |
| Ontology-free DST | 沒有 schema | 直接生成 slot 名稱和值，沒有固定清單。 |
| Correction | 「Actually...」 | 覆寫先前已填 slot 的那一輪。 |

## 延伸閱讀 (Further Reading)

- [Budzianowski et al. (2018). MultiWOZ — A Large-Scale Multi-Domain Wizard-of-Oz](https://arxiv.org/abs/1810.00278) — 標準 benchmark。
- [Feng et al. (2023). Towards LLM-driven Dialogue State Tracking (LDST)](https://arxiv.org/abs/2310.14970) — 用 LLaMA + LoRA 做 DST 的 instruction tuning。
- [Heck et al. (2020). TripPy — A Triple Copy Strategy for Value Independent Neural Dialog State Tracking](https://arxiv.org/abs/2005.02877) — 基於複製的 DST 主力。
- [King, Flanigan (2024). Unsupervised End-to-End Task-Oriented Dialogue with LLMs](https://arxiv.org/abs/2404.10753) — 以 EM 為基礎的非監督 TOD。
- [MultiWOZ leaderboard](https://github.com/budzianowski/multiwoz) — 標準的 DST 結果。
