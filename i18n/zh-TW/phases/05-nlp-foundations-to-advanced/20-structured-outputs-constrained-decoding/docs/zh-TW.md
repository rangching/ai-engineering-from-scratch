# 結構化輸出與 Constrained Decoding (Structured Outputs & Constrained Decoding)

> 要 LLM 給你 JSON，大多數時候你會拿到 JSON。在正式環境中，「大多數」就是問題。Constrained decoding 在取樣前編輯 logits，把「大多數」變成「永遠」。

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 5 · 17 (Chatbots), Phase 5 · 19 (Subword Tokenization)
**Time:** ~60 分鐘

## 問題 (The Problem)

某個分類器提示 LLM：「Return one of {positive, negative, neutral}.」模型卻回傳「The sentiment is positive — this review is overwhelmingly favorable because the customer explicitly states that they ...」。你的 parser 當掉，分類器的 F1 是 0.0。

自由格式的生成不是契約，只是建議。正式系統需要的是契約。

2026 年有三個層次。

1. **Prompting。** 好好地請求。「Return only the JSON object.」在前沿模型上約 80% 有效，較小的模型更低。
2. **原生結構化輸出 API。** OpenAI 的 `response_format`、Anthropic 的 tool use、Gemini 的 JSON mode。在支援的 schema 上可靠，但綁定廠商。
3. **Constrained decoding。** 在每個生成步驟修改 logits，讓模型*不可能*輸出無效的 token。依設計 100% 有效。適用於任何本地模型。

這堂課會為這三者建立直覺，並指出何時該用哪一個。

## 核心概念 (The Concept)

![Constrained decoding masking invalid tokens at each step](../assets/constrained-decoding.svg)

**Constrained decoding 如何運作。** 在每個生成步驟，LLM 會對完整詞彙表（約 10 萬個 token）產生 logit 向量。一個 *logit processor* 位於模型和 sampler 之間。它計算在目標文法（JSON Schema、regex、context-free grammar）中的目前位置，哪些 token 是有效的，並把所有無效 token 的 logit 設為負無限大。對剩餘 logit 做 softmax，機率質量就只落在有效的後續上。

2026 年的實作：

- **Outlines。** 把 JSON Schema 或 regex 編譯成有限狀態機 (finite-state machine)。每個 token 都有 $O(1)$ 的有效下一 token 查詢。以 FSM 為基礎，所以遞迴 schema 需要攤平。
- **XGrammar / llguidance。** Context-free grammar 引擎。能處理遞迴的 JSON Schema。解碼額外負擔接近零。OpenAI 在其 2025 年的結構化輸出實作中提到了 llguidance。
- **vLLM guided decoding。** 內建 `guided_json`、`guided_regex`、`guided_choice`、`guided_grammar`，後端可用 Outlines、XGrammar 或 lm-format-enforcer。
- **Instructor。** 以 Pydantic 為基礎、包裝任何 LLM 的封裝。驗證失敗時重試。跨供應商，但不修改 logits，它依賴重試加上了解結構化輸出的 prompt。

### 違反直覺的結果

Constrained decoding 常常比不受限的生成*更快*。有兩個原因。第一，它縮小了下一個 token 的搜尋空間。第二，聰明的實作會對被強制決定的 token 完全跳過 token 生成（像 `{"name": "` 這種骨架，每個 byte 都已確定）。

### 會讓你付出代價的陷阱

欄位順序很重要。把 `answer` 放在 `reasoning` 之前，模型會在思考之前就先對答案做出承諾。JSON 是有效的，答案卻是錯的。沒有任何驗證抓得到。

```json
// BAD
{"answer": "yes", "reasoning": "because ..."}

// GOOD
{"reasoning": "... therefore ...", "answer": "yes"}
```

Schema 的欄位順序是邏輯，不是排版。

```figure
constrained-decoder
```

## 動手做 (Build It)

### 步驟 1：從零實作 regex 約束的生成

獨立的 FSM 實作見 `code/main.py`。核心想法用 30 行表達：

```python
def mask_logits(logits, valid_token_ids):
    mask = [float("-inf")] * len(logits)
    for tid in valid_token_ids:
        mask[tid] = logits[tid]
    return mask


def generate_constrained(model, tokenizer, prompt, fsm):
    ids = tokenizer.encode(prompt)
    state = fsm.initial_state
    while not fsm.is_accept(state):
        logits = model.next_token_logits(ids)
        valid = fsm.valid_tokens(state, tokenizer)
        logits = mask_logits(logits, valid)
        tok = sample(logits)
        ids.append(tok)
        state = fsm.transition(state, tok)
    return tokenizer.decode(ids)
```

FSM 追蹤我們到目前為止滿足了文法的哪些部分。`valid_tokens(state, tokenizer)` 計算哪些詞彙表 token 可以推進 FSM，而不離開可接受的路徑。

### 步驟 2：用 Outlines 處理 JSON Schema

```python
from pydantic import BaseModel
from typing import Literal
import outlines


class Review(BaseModel):
    sentiment: Literal["positive", "negative", "neutral"]
    confidence: float
    evidence_span: str


model = outlines.models.transformers("meta-llama/Llama-3.2-3B-Instruct")
generator = outlines.generate.json(model, Review)

result = generator("Classify: 'The wait staff was attentive and the food arrived hot.'")
print(result)
# Review(sentiment='positive', confidence=0.93, evidence_span='attentive ... hot')
```

零驗證錯誤，永遠如此。FSM 讓無效輸出變得不可達。

### 步驟 3：用 Instructor 處理與供應商無關的 Pydantic

```python
import instructor
from anthropic import Anthropic
from pydantic import BaseModel, Field


class Invoice(BaseModel):
    vendor: str
    total_usd: float = Field(ge=0)
    line_items: list[str]


client = instructor.from_anthropic(Anthropic())
invoice = client.messages.create(
    model="claude-opus-4-7",
    max_tokens=1024,
    response_model=Invoice,
    messages=[{"role": "user", "content": "Extract from: 'Acme Corp $420. Widget, Gizmo.'"}],
)
```

機制不同。Instructor 不碰 logits。它把 schema 格式化進 prompt，解析輸出，並在驗證失敗時重試（預設 3 次）。適用於任何供應商。重試會增加延遲與成本。跨供應商的可攜性是它的賣點。

### 步驟 4：原生廠商 API

```python
from openai import OpenAI

client = OpenAI()
response = client.responses.create(
    model="gpt-5",
    input=[{"role": "user", "content": "Classify: 'The food was cold.'"}],
    text={"format": {"type": "json_schema", "name": "sentiment",
          "schema": {"type": "object", "required": ["sentiment"],
                     "properties": {"sentiment": {"type": "string",
                                                  "enum": ["positive", "negative", "neutral"]}}}}},
)
print(response.output_parsed)
```

伺服器端的 constrained decoding。對支援的 schema，可靠度與 Outlines 相當。不需要管理本地模型。但會把你鎖定在該廠商。

## 陷阱

- **遞迴 schema。** Outlines 會把遞迴攤平到固定深度。樹狀結構的輸出（巢狀留言、AST）需要 XGrammar 或 llguidance（以 CFG 為基礎）。
- **巨大的 enum。** 有 1 萬個選項的 enum 編譯很慢或逾時。改用 retriever：先預測前 k 個候選，再把約束限制在這些候選上。
- **文法太嚴格。** 強制 `date: "YYYY-MM-DD"` 的 regex，模型就無法為缺少的日期輸出 `"unknown"`。模型會靠捏造日期來補償。請允許 `null` 或哨兵值。
- **過早承諾。** 見上面的欄位順序陷阱。永遠把 reasoning 放在前面。
- **沒有 schema 的廠商 JSON mode。** 單純的 JSON mode 只保證 JSON 有效，不保證*對你的使用情境*有效。一定要提供完整的 schema。

## 實際使用 (Use It)

2026 年的技術棧：

| 情境 | 選擇 |
|-----------|------|
| OpenAI/Anthropic/Google 模型，簡單的 schema | 原生廠商結構化輸出 |
| 任何供應商，Pydantic 工作流程，可容忍重試 | Instructor |
| 本地模型，需要 100% 有效，扁平 schema | Outlines (FSM) |
| 本地模型，遞迴 schema | XGrammar 或 llguidance |
| 自行託管的推論伺服器 | vLLM guided decoding |
| 批次處理，可接受重試 | Instructor 加最便宜的模型 |

## 交付 (Ship It)

存成 `outputs/skill-structured-output-picker.md`：

```markdown
---
name: structured-output-picker
description: Choose a structured output approach, schema design, and validation plan.
version: 1.0.0
phase: 5
lesson: 20
tags: [nlp, llm, structured-output]
---

Given a use case (provider, latency budget, schema complexity, failure tolerance), output:

1. Mechanism. Native vendor structured output, Instructor retries, Outlines FSM, or XGrammar CFG. One-sentence reason.
2. Schema design. Field order (reasoning first, answer last), nullable fields for "unknown", enum vs regex, required fields.
3. Failure strategy. Max retries, fallback model, graceful `null` handling, out-of-distribution refusal.
4. Validation plan. Schema compliance rate (target 100%), semantic validity (LLM-judge), field-coverage rate, latency p50/p99.

Refuse any design that puts `answer` or `decision` before reasoning fields. Refuse to use bare JSON mode without a schema. Flag recursive schemas behind an FSM-only library.
```

## 練習 (Exercises)

1. **簡單。** 不用 constrained decoding，提示一個小型 open-weights 模型（例如 Llama-3.2-3B）產生 `Review(sentiment, confidence, evidence_span)`。在 100 則評論上，測量能解析成有效 JSON 的比例。
2. **中等。** 用 Outlines JSON mode 處理同一份語料庫。比較合規率、延遲與語意準確率。
3. **困難。** 從零實作針對電話號碼 (`\d{3}-\d{3}-\d{4}`) 的 regex 約束 decoder。驗證 1000 個樣本中有 0 個無效輸出。

## 關鍵術語 (Key Terms)

| 術語 | 一般人怎麼說 | 實際意義 |
|------|-----------------|-----------------------|
| Constrained decoding | 強制有效輸出 | 在每個生成步驟遮蔽無效 token 的 logit。 |
| Logit processor | 負責約束的東西 | 函式：`(logits, state) -> masked_logits`。 |
| FSM | 有限狀態機 | 編譯後的文法表示；$O(1)$ 的有效下一 token 查詢。 |
| CFG | Context-free grammar | 能處理遞迴的文法；比 FSM 慢，但表達力更強。 |
| Schema field order | 有差嗎？ | 有，第一個欄位就是承諾；永遠把 reasoning 放在 answer 前面。 |
| Guided decoding | vLLM 的稱呼 | 同樣的概念，整合進推論伺服器。 |
| JSON mode | OpenAI 的早期版本 | 保證 JSON 語法正確；不保證符合 schema。 |

## 延伸閱讀 (Further Reading)

- [Willard, Louf (2023). Efficient Guided Generation for LLMs](https://arxiv.org/abs/2307.09702) — Outlines 論文。
- [XGrammar paper (2024)](https://arxiv.org/abs/2411.15100) — 快速的、以 CFG 為基礎的 constrained decoding。
- [vLLM — Structured Outputs](https://docs.vllm.ai/en/latest/features/structured_outputs.html) — 推論伺服器整合。
- [OpenAI — Structured Outputs guide](https://platform.openai.com/docs/guides/structured-outputs) — API 參考加注意事項。
- [Instructor library](https://python.useinstructor.com/) — 跨供應商的 Pydantic 加重試。
- [JSONSchemaBench (2025)](https://arxiv.org/abs/2501.10868) — 對 6 個 constrained decoding 框架做 benchmark。
