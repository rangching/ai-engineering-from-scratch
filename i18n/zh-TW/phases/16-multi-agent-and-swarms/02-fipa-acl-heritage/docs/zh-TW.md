# FIPA-ACL 與 Speech Act 的淵源 (Heritage of FIPA-ACL and Speech Acts)

> 在 MCP、A2A 之前，就有 FIPA-ACL。2000 年，Foundation for Intelligent Physical Agents（FIPA，2005 年成為 IEEE Computer Society 的標準委員會）批准了一套 agent communication language，包含二十個 performative、兩種 content language，以及一組 interaction protocol：contract net、subscribe/notify、request-when。它後來在業界淡出，因為 ontology 的負擔對網路來說太重；但 LLM 帶動的 multi-agent 復興，正悄悄重新實作同樣的想法，只是少了形式語意：JSON 契約取代 performative，自然語言取代 ontology。這堂課認真讀一遍 FIPA-ACL，讓你看清楚 2026 年的哪些 protocol 決策只是重新發明、哪些是真的創新，以及這一波浪潮會在哪裡重新發現 2000 年代就已經解決過的問題。

**Type:** Learn
**Languages:** Python (stdlib)
**Prerequisites:** Phase 16 · 01 (Why Multi-Agent)
**Time:** ~60 分鐘

## 問題 (Problem)

2026 年的 agent-protocol 生態很熱鬧：MCP 管 tool、A2A 管 agent、ACP 管企業稽核、ANP 管去中心化信任、NLIP 管自然語言內容，再加上 CA-MCP 和二十幾個研究提案。每份規格都宣稱自己是基礎。

老實說，它們大多是在重新發現一棵二十年前就有的決策樹。Austin（1962）和 Searle（1969）的 speech-act 理論告訴我們「說話就是行動」。KQML（1993）把它變成 wire protocol。FIPA-ACL（2000 年批准）則產出參考標準：二十個 performative、SL0/SL1 content language，以及 contract-net 和 subscribe-notify 的 interaction protocol。JADE 和 JACK 是當時的 Java 參考平台。這項努力在 2010 年前後淡出，因為 ontology 的負擔太重，而且網路正在勝出。

當你看 MCP 的 `tools/call`、A2A 的 task lifecycle，或 CA-MCP 的 shared context store，你看到的是 FIPA 決策的一個較軟、原生 JSON 的翻版。了解這段淵源能告訴你兩件事：哪些新的「創新」其實是重新發明，以及新規格會重新踩到哪些舊的失敗模式。

## 核心概念 (Concept)

### 一段話講完 Speech Act (Speech acts, in one paragraph)

Austin 注意到有些句子不是在描述世界，而是在改變世界。「我保證。」「我請求。」「我宣布。」他把這些稱為 performative utterance。Searle 把它們正式分成五類：assertive、directive、commissive、expressive、declarative。KQML（Finin et al., 1993）讓這套東西能用在軟體 agent 上：一則訊息是一個 performative（行動）加上 content（行動的對象）。FIPA-ACL 補上 KQML 的缺口，並以二十個 performative 為中心做標準化。

### 二十個 FIPA Performative（部分清單）(The twenty FIPA performatives)

| Performative | 意圖 |
|---|---|
| `inform` | 「我告訴你 P 為真」 |
| `request` | 「我請你做 X」 |
| `query-if` | 「P 為真嗎？」 |
| `query-ref` | 「X 的值是什麼？」 |
| `propose` | 「我提議我們做 X」 |
| `accept-proposal` | 「我接受這個提案」 |
| `reject-proposal` | 「我拒絕這個提案」 |
| `agree` | 「我同意做 X」 |
| `refuse` | 「我拒絕做 X」 |
| `confirm` | 「我確認 P 為真」 |
| `disconfirm` | 「我否認 P」 |
| `not-understood` | 「你的訊息無法解析」 |
| `cfp` | 「徵求 X 的提案」 |
| `subscribe` | 「X 改變時通知我」 |
| `cancel` | 「取消正在進行的 X」 |
| `failure` | 「我試了 X 但失敗了」 |

完整清單在 `fipa00037.pdf`（FIPA ACL Message Structure）。重點不是背下來，而是這裡每一個都對應到某個原語，而 LLM protocol 最後都會把它再加回去。

### 標準的 FIPA-ACL 訊息 (Canonical FIPA-ACL message)

```
(inform
  :sender       agent1@platform
  :receiver     agent2@platform
  :content      "((price IBM 83))"
  :language     SL0
  :ontology     finance
  :protocol     fipa-request
  :conversation-id   conv-42
  :reply-with   msg-17
)
```

七個欄位承載 protocol 的外層封包；一個欄位（`content`）承載 payload。其餘欄位，正是你每次在 JSON protocol 上外掛重試、串接（threading）和 ontology 時，一再重新發明的東西。

### 兩個舊平台 (The two legacy platforms)

**JADE**（Java Agent DEvelopment framework，1999–2020s）是使用最廣的 FIPA 相容 runtime。Agent 繼承一個基底類別、交換 ACL 訊息、在 container 內執行，並用「behavior」協調。它的 interaction-protocol 函式庫內建 contract-net、subscribe-notify、request-when 和 propose-accept。

**JACK**（Agent Oriented Software，商業產品）強調在 FIPA 訊息之上做 BDI（Belief-Desire-Intention）推理。更形式化，採用者較少。

網路技術堆疊吃掉 multi-agent 的應用場景之後，兩者都衰退了。MCP 和 A2A 就是 2026 年的 runtime「container」。

### FIPA 為什麼淡出 (Why FIPA faded)

- **Ontology 的負擔。** FIPA 需要共享的 ontology 才能解析 `content`。要在 ontology 上達成共識，是一個動輒數年的標準化流程。網路則直接用 HTTP + JSON。
- **沒人用的形式語意。** SL（Semantic Language）提供嚴謹的真值條件，但大多數正式環境的系統用自由格式的 content，直接忽略形式化的部分。
- **工具被綁死。** JADE 只支援 Java，JACK 是商業產品。多語言團隊都繞過這兩者。
- **網際網路贏得了技術堆疊。** REST、接著 JSON-RPC、再來 gRPC，取代了 ACL 的傳輸層。

### LLM 的復興是精簡版 FIPA (The LLM revival is FIPA-lite)

比較 FIPA 的 `request` 與 MCP 的 `tools/call`：

```
(request                                {
  :sender  agent1                         "jsonrpc": "2.0",
  :receiver tool-server                   "method":  "tools/call",
  :content "(lookup stock IBM)"           "params":  {"name":"lookup_stock",
  :ontology finance                                   "arguments":{"symbol":"IBM"}},
  :conversation-id c42                    "id": 42
)                                        }
```

同樣的封包，不同的語法。兩者都帶有：誰、對誰、意圖、payload、correlation id。沒有哪一個比另一個更革命，它們只是在同一個設計上做不同的取捨。

Liu 等人 2025 年的綜述（"A Survey of Agent Interoperability Protocols: MCP, ACP, A2A, ANP", arXiv:2505.02279）把這條脈絡講得很明白：MCP 對應 tool-use 的 speech act，A2A 對應 agent-peer 的 speech act，ACP 對應稽核軌跡的 speech act，ANP 對應去中心化身分的延伸。新規格是 ACL 的後代，只是改用 JSON 語法，語意也較寬鬆。

### 取捨，直接講白 (The trade-off, stated plainly)

**FIPA 給了你、但現代規格放掉的：**

- 形式語意：你可以證明 `inform` 蘊含發送者相信該 content。
- 一份標準的 performative 目錄：你不必再爭論「我們要不要有 `cancel`？」。
- 數十年累積的 interaction-protocol pattern：contract-net、subscribe-notify、propose-accept，而且它們的正確性質都是已知的。

**現代規格給了你、但 FIPA 沒有的：**

- 原生 JSON 的 payload，與所有現代工具相容。
- LLM 不需要手寫 ontology 就能解讀的自然語言 content。
- 網路技術堆疊的傳輸方式（HTTP、SSE、WebSocket）。
- 透過即時的 MCP `server/discover` 和 A2A Agent Card 做 capability discovery。

用較鬆散的意圖語意，換取更容易實作。這就是確切的取捨。

### 值得移植的 Interaction Protocol (Interaction protocols worth porting)

FIPA 出了大約 15 個 interaction protocol。其中三個值得帶進 LLM multi-agent 系統：

1. **Contract Net Protocol (CNP)。** Manager 發出 `cfp`（徵求提案）；投標者回應 `propose`；manager 接受或拒絕。這是標準的任務市場 pattern（Phase 16 · 16 Negotiation）。
2. **Subscribe/Notify。** 訂閱者送出 `subscribe`；發布者在主題改變時送出 `inform`。這就是 2026 年每一個 event bus。
3. **Request-When。** 「條件 Y 成立時，做 X。」帶有前置條件的延遲動作。2026 年的對應物，是 durable workflow engine 裡的 deferred task（Phase 16 · 22 Production Scaling）。

每一個都能乾淨地對應到現代的 message queue、HTTP + polling，或 SSE streaming。

### 拿掉 Ontology 會壞掉什麼 (What breaks when you drop the ontology)

沒有共享的 ontology，agent 只能從自然語言 content 推斷意義。2026 年有紀錄的失敗模式是 **semantic drift**：兩個 agent 用同一個詞（`"customer"`）指稍微不同的概念，接收方的 agent 依錯誤的詮釋行動，沒有任何 schema validator 攔得到。FIPA 的 ontology 要求會在解析階段就拒絕這則訊息。

不必走到完整 ontology 的緩解方式：

- 對 `content` 使用 JSON Schema：在傳輸階段就擋掉結構錯誤。
- 有型別的 artifact（A2A）：擋掉錯誤的 modality。
- 在封包中明確標出 performative：即使 content 是自然語言，意圖也不會含糊。

### 2026 年的規格，對照 Speech-Act 淵源 (The 2026 specs, mapped to speech-act heritage)

| 現代規格 | FIPA 對應物 | 保留什麼 | 放掉什麼 |
|---|---|---|---|
| MCP `tools/call` | `request` | 明確的意圖、correlation id | 形式語意、ontology |
| MCP `resources/read` | `query-ref` | 明確的意圖、correlation id | 形式語意 |
| A2A Task lifecycle | contract-net + request-when | 非同步生命週期、狀態轉換 | 形式化的完備性保證 |
| A2A streaming events | subscribe/notify | 非同步推送 | 有型別的 predicate 訂閱 |
| CA-MCP shared context | blackboard (Hayes-Roth 1985) | 多寫入者共享記憶體 | 邏輯一致性模型 |
| NLIP | natural-language content | 對 LLM 原生 | schema |

由上往下讀這張表，規律是：保留結構性的原語，放掉形式化的部分，讓 LLM 去彌補含糊之處。

```figure
sw-contract-net
```

## 動手做 (Build It)

`code/main.py` 實作了一個只用標準函式庫的 FIPA-ACL 轉換器。它能編碼與解碼標準的 ACL 封包，並展示每一種 MCP / A2A 訊息形狀，都能化約成同樣的七個欄位。這個 demo 會：

- 把五則 MCP 風格與 A2A 風格的訊息編碼成 FIPA-ACL。
- 把 FIPA-ACL 解碼回現代的對應格式。
- 用 `cfp`、`propose`、`accept-proposal`、`reject-proposal`，跑一場一個 manager 對三個投標者的簡易 Contract Net 協商。

執行：

```
python3 code/main.py
```

輸出是並排的追蹤紀錄，每則現代訊息都同時以 2026 年的 JSON 形式與 FIPA-ACL 形式呈現，接著是一輪 contract-net 投標的來回轉換。來回轉換之後，同樣的 protocol 原語都保留下來，只有語法不同。

## 實際使用 (Use It)

`outputs/skill-fipa-mapper.md` 是一個 skill，它會讀任何 agent-protocol 規格，並產出 FIPA-ACL 的對照。採用新 protocol 之前先用它，回答：「這真的是新東西，還是換成 JSON 語法的 `inform`？」

## 交付 (Ship It)

不要把 FIPA-ACL 找回來。把它的檢查清單帶回來：

- 每則訊息的意圖原語（performative）是什麼？
- request-response 和取消是否有 correlation id？
- 是否有明確的 content language（JSON-RPC、純文字、有型別的結構化 artifact）？
- Interaction protocol 是一級公民，還是你在從頭重新實作 contract-net？
- 兩個 agent 對 content 的意義有分歧時（semantic drift），會發生什麼事？

任何新 protocol 在上線到正式環境之前，都先把這五個問題寫成文件。

## 練習 (Exercises)

1. 執行 `code/main.py`。觀察來回轉換的編碼。找出哪個 FIPA performative 對應 `tools/call`、`resources/read`，以及 A2A 的 task 建立。
2. 在 contract-net demo 加上 `cancel` performative，讓 manager 能在投標中途撤回任務。`cancel` 解決了哪種光靠重試解決不了的失敗情境？
3. 閱讀 FIPA ACL Message Structure (http://www.fipa.org/specs/fipa00037/) 的 4.1–4.3 節。挑一個本課沒有涵蓋的 performative，描述它在現代 JSON-RPC 中的對應物。
4. 閱讀 Liu et al., arXiv:2505.02279。針對 MCP、A2A、ACP、ANP，分別列出它們保留與放掉的 FIPA performative 家族。
5. 為你自己系統裡 `request` performative 的 `content` 欄位，設計一份最精簡的 JSON-Schema。這份 schema 帶給你什麼是純自然語言沒有的？代價又是什麼？

## 關鍵術語 (Key Terms)

| 術語 | 一般說法 | 實際意義 |
|------|----------------|------------------------|
| Speech act | 「會做事的一句話」 | Austin/Searle：把說話視為行動。ACL 在理論上的源頭。 |
| FIPA | 「那個老舊的 XML 東西」 | Foundation for Intelligent Physical Agents。2000 年標準化 ACL；2005 年成為 IEEE Computer Society 的標準委員會。 |
| ACL | 「Agent Communication Language」 | FIPA 的封包格式：performative + content + metadata。 |
| Performative | 「動詞」 | 訊息的意圖類別：`inform`、`request`、`propose`、`cfp` 等。 |
| KQML | 「FIPA 的前身」 | Knowledge Query and Manipulation Language（1993）。更簡單、範圍更窄。 |
| Ontology | 「共享詞彙」 | 對 content language 所談論的概念，給出的形式化定義。 |
| SL0 / SL1 | 「FIPA 的 content language」 | Semantic Language 第 0 與第 1 層，屬於形式化 content language 家族。 |
| Contract Net | 「任務市場」 | Manager 發出 cfp；投標者 propose；manager 接受。標準的 interaction protocol。 |
| Interaction protocol | 「訊息的模式」 | 一串正確性已知的 performative 序列：request-when、subscribe-notify 等。 |

## 延伸閱讀 (Further Reading)

- [Liu et al. — A Survey of Agent Interoperability Protocols: MCP, ACP, A2A, ANP](https://arxiv.org/html/2505.02279v1) — 2025 年把現代規格連結到 FIPA 淵源的標準綜述
- [FIPA ACL Message Structure Specification (fipa00037)](http://www.fipa.org/specs/fipa00037/) — 2000 年批准的封包格式
- [FIPA Communicative Act Library Specification (fipa00037)](http://www.fipa.org/specs/fipa00037/) — 完整的 performative 目錄
- [MCP specification 2026-07-28](https://modelcontextprotocol.io/specification/2026-07-28) — 目前 stateless 的 tool-use，相當於 `request`/`query-ref`
- [A2A specification](https://a2a-protocol.org/latest/specification/) — 現代 agent-peer 對 contract-net 與 subscribe-notify 的對應物
