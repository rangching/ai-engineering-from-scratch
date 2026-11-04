# Structured Output：JSON Schema、Pydantic、Zod、Constrained Decoding (Structured Output — JSON Schema, Pydantic, Zod, Constrained Decoding)

> 「客氣地請模型回傳 JSON」，即使在前沿模型上也有 5% 到 15% 的機率失敗。Structured output 用 constrained decoding 補上這個缺口：模型根本無法輸出會違反 schema 的 token。OpenAI 的 strict mode、Anthropic 的 schema 型別 tool use、Gemini 的 `responseSchema`、Pydantic AI 的 `output_type`，以及 Zod 的 `.parse`，是同一個概念的五種表面形式。這堂課會建立 schema validator 和 strict-mode 合約，之後每一條 production 抽取流程都會用到。

**Type:** Build
**Languages:** Python (stdlib, JSON Schema 2020-12 subset)
**Prerequisites:** Phase 13 · 02 (function calling deep dive)
**Time:** ~75 分鐘

## 學習目標 (Learning Objectives)

- 用正確的限制條件（enum、min/max、required、pattern），為一個抽取目標寫出 JSON Schema 2020-12。
- 說明為什麼 strict mode 與 constrained decoding 提供的保證，和「產生後再驗證」不同。
- 區分三種失敗模式：parse error、schema violation、model refusal。
- 交付一條具備型別化 repair 與型別化 refusal 處理的抽取流程。

## 問題 (The Problem)

一個讀採購單 email 的 agent，需要把自由文字轉成 `{customer, line_items, total_usd}`。有三種做法。

**做法一：用 prompt 要求 JSON。**「用 JSON 回覆，欄位為 customer、line_items、total_usd。」在前沿模型上，85% 到 95% 的時候有效。會以六種方式失敗：少了大括號、多餘的逗號、錯誤的型別、幻想出來的欄位、在 token 上限被截斷，以及像「Here is your JSON:」這種漏出來的散文。

**做法二：產生後再驗證。** 自由產生、解析、對照 schema 驗證，失敗就重試。可靠，但很貴：每次重試都要付費，而截斷的 bug 每發生一次，就多花一個 turn。

**做法三：constrained decoding。** Provider 在 decode 時就強制 schema。無效的 token 會從 sampling 分布中被遮掉。輸出保證能被解析，也保證能通過驗證。失敗只剩下一種模式：refusal（模型判斷輸入不符合 schema）。

2026 年每家前沿 provider 都提供某種形式的做法三。

- **OpenAI。** `response_format: {type: "json_schema", strict: true}`，若模型拒絕，回應中會有 `refusal`。
- **Anthropic。** 對 `tool_use` 的 input 做 schema 強制；沒有 `stop_reason: "refusal"` 這種東西，但沒有 tool call 的 `end_turn` 就是訊號。
- **Gemini。** 請求層級的 `responseSchema`；2026 年 Gemini 針對部分型別提供 token 層級的文法限制。
- **Pydantic AI。** `output_type=InvoiceModel` 會輸出型別為 `InvoiceModel` 的結構化 `RunResult`。
- **Zod (TypeScript)。** Runtime parser，依 Zod schema 驗證 provider 的輸出；可搭配 OpenAI 的 `beta.chat.completions.parse`。

共通點是：schema 只宣告一次，然後從頭到尾強制執行。

## 核心概念 (The Concept)

### JSON Schema 2020-12：通用語言

每家 provider 都接受 JSON Schema 2020-12。你最常用到的語法：

- `type`：`object`、`array`、`string`、`number`、`integer`、`boolean`、`null` 之一。
- `properties`：欄位名稱對應子 schema 的 map。
- `required`：必須出現的欄位名稱清單。
- `enum`：封閉的允許值集合。
- `minimum` / `maximum`（數字）、`minLength` / `maxLength` / `pattern`（字串）。
- `items`：套用到每個陣列元素的子 schema。
- `additionalProperties`：`false` 禁止額外欄位（預設值依模式而異）。

OpenAI strict mode 增加三項要求：每個 property 都必須列在 `required`、到處都要有 `additionalProperties: false`，以及不能有未解析的 `$ref`。如果違反，API 會在請求時回傳 400。

### Pydantic：Python 的綁定

Pydantic v2 透過 `model_json_schema()`，從 dataclass 形狀的 model 產生 JSON Schema。Pydantic AI 把它包起來，所以你這樣寫：

```python
class Invoice(BaseModel):
    customer: str
    line_items: list[LineItem]
    total_usd: Decimal
```

Agent 框架會在邊緣把 schema 翻譯成 OpenAI strict mode、Anthropic 的 `input_schema` 或 Gemini 的 `responseSchema`。模型的輸出回來時，是一個有型別的 `Invoice` 實例。驗證錯誤會丟出 `ValidationError`，並帶有型別化的錯誤路徑。

### Zod：TypeScript 的綁定

Zod（`z.object({customer: z.string(), ...})`）是 TS 的對應工具。OpenAI 的 Node SDK 提供 `zodResponseFormat(Invoice)`，會翻譯成 API 的 JSON Schema payload。

### Refusals

Strict mode 無法強迫模型作答。如果輸入無法符合 schema（「這封 email 是一首詩，不是發票」），模型會輸出 `refusal` 欄位，裡面是原因。你的程式必須把這當作一等結果來處理，而不是失敗。Refusal 也可以當作安全訊號：被要求從受保護內容的 email 中擷取信用卡號的模型，會回傳附上安全理由的 refusal。

### 開放生態中的 constrained decoding

Open-weights 的實作用三種技術。

1. **以文法為基礎的 decoding**（`outlines`、`guidance`、`lm-format-enforcer`）：從 schema 建立一個 deterministic finite automaton；每一步都遮掉會違反 FSM 的 token 的 logit。
2. **用 JSON parser 做 logit masking**：讓 streaming JSON parser 與模型同步執行；每一步計算有效的下一個 token 集合。
3. **搭配 verifier 的 speculative decoding**：便宜的 draft model 提議 token，verifier 強制 schema。

商業 provider 會在背後挑其中一種。2026 年的最佳水準，在短的結構化輸出上比一般產生更快，在長輸出上速度大致相同。

### 三種失敗模式 (The three failure modes)

1. **Parse error。** 輸出不是有效的 JSON。在 strict mode 下不會發生，在 non-strict 的 provider 上仍然可能發生。
2. **Schema violation。** 輸出可以解析，但違反 schema。在 strict mode 下不會發生，之外很常見。
3. **Refusal。** 模型拒絕。必須當作型別化的結果來處理。

### Retry 策略

當你不在 strict mode 時（Anthropic tool use、non-strict 的 OpenAI、較舊的 Gemini），復原的模式是：

```
generate -> parse -> validate -> if fail, inject error and retry, max 3x
```

通常重試一次就夠了。重試三次可以抓到弱模型的偶發錯誤。超過三次，代表 schema 有問題：模型對某些輸入無法滿足它，需要修正 prompt 或 schema。

### 小模型的支援

Constrained decoding 在小模型上也有效。在結構化任務上，一個 3B 參數、加上文法強制的開放模型，表現優於一個只靠原始 prompt 的 70B 參數模型。這是 structured output 對 production 重要的主要原因：它讓可靠度與模型大小脫鉤。

```figure
constrained-decoding
```

## 實際使用 (Use It)

`code/main.py` 提供一個用 stdlib 寫成的最小 JSON Schema 2020-12 validator（types、required、enum、min/max、pattern、items、additionalProperties）。它包裝一個 `Invoice` schema，並讓假的 LLM 輸出通過 validator，示範 parse error、schema violation 和 refusal 三條路徑。在 production 中，把假輸出換成任何 provider 的真實回應。

觀察重點：

- Validator 回傳型別化的 `[ValidationError]` 清單，包含 path 和 message。這就是你要放進 retry prompt 的形狀。
- Refusal 分支不會重試。它記錄後回傳型別化的 refusal。Phase 14 · 09 把 refusal 當作安全訊號。
- `additionalProperties: false` 檢查在對抗性測試輸入上觸發，說明為什麼 strict mode 能擋住幻想出來的欄位。

## 交付 (Ship It)

這堂課會產出 `outputs/skill-structured-output-designer.md`。給定一個自由文字的抽取目標（發票、客服工單、履歷等），這個 skill 會產出與 strict-mode 相容的 JSON Schema 2020-12，以及與之對應的 Pydantic model，並預先搭好型別化 refusal 與 retry 處理的雛形。

## 練習 (Exercises)

1. 執行 `code/main.py`。新增第四個測試案例，讓 `total_usd` 為負數。確認 validator 會以 `minimum` 限制的 path 拒絕它。

2. 擴充 validator，支援帶 discriminator 的 `oneOf`。常見情境：`line_item` 不是 product 就是 service，由 `kind` 標記。Strict mode 在這裡有些微妙的規則，請查閱 OpenAI 的 structured outputs 指南。

3. 把同一份 Invoice schema 寫成 Pydantic BaseModel，並比較 `model_json_schema()` 的輸出與你手寫的 schema。找出 Pydantic 預設會設定、但手寫版本省略的那一個欄位。

4. 測量 refusal 比率。建構十個不該被抽取的輸入（歌詞、數學證明、空白 email），用真實 provider 搭配 strict mode 執行。計算 refusal 與幻想輸出的數量。這是你做 refusal-aware retry 的基準真相。

5. 從頭到尾讀完 OpenAI 的 structured outputs 指南。找出它在 strict mode 中明確禁止、但一般 JSON Schema 允許的那一個語法。接著設計一個非必要使用該禁用語法的 schema，並重構成與 strict 相容。

## 關鍵術語 (Key Terms)

| 術語 | 一般說法 | 實際意義 |
|------|----------------|------------------------|
| JSON Schema 2020-12 | 「Schema 規格」 | 每家現代 provider 都使用的 IETF-draft schema 方言 |
| Strict mode | 「保證 schema」 | OpenAI 的旗標，透過 constrained decoding 強制 schema |
| Constrained decoding | 「Logit masking」 | Decode 時遮掉無效下一個 token 的強制機制 |
| Refusal | 「模型拒絕」 | 輸入無法符合 schema 時的型別化結果 |
| Parse error | 「無效 JSON」 | 輸出無法解析成 JSON；strict 下不可能發生 |
| Schema violation | 「形狀錯誤」 | 可以解析，但違反 types / required / enum / range |
| `additionalProperties: false` | 「不允許額外欄位」 | 禁止未知欄位；OpenAI strict 必備 |
| Pydantic BaseModel | 「型別化輸出」 | 會輸出並驗證 JSON Schema 的 Python class |
| Zod schema | 「TypeScript 輸出型別」 | 用來驗證 provider 輸出的 TS runtime schema |
| Grammar enforcement | 「開放權重的 constrained decode」 | 以 FSM 為基礎的 logit masking，如 outlines / guidance |

## 延伸閱讀 (Further Reading)

- [OpenAI — Structured outputs](https://platform.openai.com/docs/guides/structured-outputs) — strict mode、refusal 與 schema 要求
- [OpenAI — Introducing structured outputs](https://openai.com/index/introducing-structured-outputs-in-the-api/) — 2024 年 8 月的發表文章，說明 decoding 保證
- [Pydantic AI — Output](https://ai.pydantic.dev/output/) — 可序列化到各家 provider 的型別化 output_type 綁定
- [JSON Schema — 2020-12 release notes](https://json-schema.org/draft/2020-12/release-notes) — 標準規格
- [Microsoft — Structured outputs in Azure OpenAI](https://learn.microsoft.com/en-us/azure/foundry/openai/how-to/structured-outputs) — 企業部署筆記與 strict-mode 注意事項
