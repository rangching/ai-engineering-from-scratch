# 聊天機器人 (Chatbots) — 從 Rule-Based 到 Neural 再到 LLM Agent

> ELIZA 用模式比對來回應。DialogFlow 對應 intent。GPT 從權重回答。Claude 執行工具並驗證。每個時代都解決了前一個時代最糟的失敗。

**Type:** Learn
**Languages:** Python
**Prerequisites:** Phase 5 · 13 (Question Answering), Phase 5 · 14 (Information Retrieval)
**Time:** ~75 分鐘

## 問題 (The Problem)

使用者說「I want to change my flight.」。系統得弄清楚他們想要什麼、缺少什麼資訊、怎麼取得這些資訊，以及怎麼完成這個動作。接著使用者說「wait, what if I cancel instead?」，系統得記住 context、切換任務，並保留狀態。

對話對 ML 系統來說很難。輸入是開放式的。輸出必須在許多輪次中保持連貫。系統可能需要對世界採取行動（改班機、刷信用卡）。每一個錯誤的步驟，使用者都看得到。

聊天機器人的架構經歷了四種典範，每一種的出現，都是因為前一種的失敗太明顯。這堂課依序走過它們。2026 年的正式環境版圖，是最後兩種的混合。

## 核心概念 (The Concept)

![Chatbot evolution: rule-based → retrieval → neural → agent](../assets/chatbot.svg)

### 腳本化的半個世紀，1950 到 2001 年

第一種典範撐的不是五年，而是五十年。了解它的軌跡很重要，因為其中每個系統都是同一台機器：比對輸入、吐出罐頭回應、稍微更新一點狀態。對這台機器加了五十年的規則，也從未產生通用的情況。這個天花板就是第二到第四種典範存在的原因。

**1950。** Turing 迴避了「機器能思考嗎？」這個問題，提出一個操作性的替代方案：如果詢問者透過電傳打字機無法分辨機器和人，那麼這個哲學問題就沒有意義了。在這個領域還沒有名字之前，對話就成了它的 benchmark。

**1956。** 名字誕生了。一場在 Dartmouth 的夏季工作坊，基於「智慧的每個特徵原則上都能被精確描述，使機器可以模擬它」這個推測，創造了「人工智慧」一詞。提案預留了兩個月，期待有實質進展。

**1966。** ELIZA 推出了你會在步驟 1 建立的反射技巧：分解規則從輸入中取出片段，重組規則再把它們當成問題回聲出去。總共約 200 個模式，零狀態，零理解，使用者卻依然向它傾訴心事。Weizenbaum 在他往後的職業生涯中，一直為這麼少的機制就能辦到這件事而感到警覺。

**1972。** PARRY 是在 Stanford 為了模擬妄想症而建立的，它加上了 ELIZA 缺少的部分：內部狀態。代表恐懼、憤怒與不信任的數值變數，每一輪都會更新，並決定下一個要觸發哪個腳本，所以相同的輸入會依到目前為止的對話而產生不同的回應。在盲測的對話紀錄測驗中，精神科醫師區分 PARRY 與人類病患的表現只相當於隨機猜測。它是 persona conditioning 的直系祖先：用三個浮點數實作的 system prompt。同一年，這兩個機器人透過 ARPANET 被連到彼此：一個治療師腳本在訪談一個妄想症狀態機，這是網路上第一次機器人對機器人的對話。

**1995。** ALICE 用 AIML 擴大了 ELIZA 的做法，AIML 是為模式-模板配對設計的 XML 方言。約 4 萬個手寫類別，三次贏得 Loebner Prize。它證明了 rule-based 系統的擴展定律：更多規則換來涵蓋範圍，卻永遠換不到通用性。每條規則都是有人必須維護的負債。

**2001。** SmarterChild 把這套做法送到 3,000 萬名即時通訊使用者面前，並加上後端查詢，天氣、股票、電影時刻，拼接進模板。瞇起眼睛看，它就是穿著 2001 年服裝的 tool calling：解析 intent、呼叫服務、把結果渲染進回覆。

五十年，一種機制，規則數量不斷上升。這個典範的結束，不是因為有人證明它是錯的，而是因為手寫狀態機的維護成本隨涵蓋範圍線性成長，而使用者的期待卻隨著他們上週看到的東西而成長。

```figure
chatbot-lineage
```

**Rule-based (ELIZA, AIML, DialogFlow)。** 手工撰寫的模式比對使用者輸入並產生回應。Intent 分類器把請求導向預先定義的流程。Slot-filling 狀態機收集必要資訊。在它被設計的狹窄範圍內表現出色，一旦超出範圍就立刻失敗。在不能容忍幻覺的安全關鍵領域（銀行身分驗證、航空訂位），仍然有在使用。

**Retrieval-based。** FAQ 風格的系統。編碼每一組（utterance, response）配對。執行時，編碼使用者的訊息，並檢索最近的已儲存回應。想想 Zendesk 經典的「類似文章」功能。處理改寫比規則更好。不做生成，所以沒有幻覺。

**Neural (seq2seq)。** 在對話紀錄上訓練的 encoder-decoder。從零開始生成回應。流暢，但容易產生泛泛的輸出（"I don't know"）和事實漂移。從來無法穩定地切題。這也是 Google、Facebook 和 Microsoft 在 2016 到 2019 年的聊天機器人都令人失望的原因。

**LLM agents。** 包在迴圈中的語言模型，會規劃、呼叫工具、驗證結果。不是帶著很長 prompt 的聊天機器人，而是 agent 迴圈：規劃 → 呼叫工具 → 觀察結果 → 決定下一步。Retrieval-first 的 grounding (RAG) 避免它產生幻覺。工具呼叫讓它真的能做事。這是 2026 年的架構。

這四種典範不是依序取代彼此。2026 年的正式聊天機器人會把請求導向這四種：身分驗證和破壞性動作用 rule-based，FAQ 用 retrieval，自然措辭用 neural generation，模糊的開放式查詢用 LLM agent。

## 動手做 (Build It)

### 步驟 1：rule-based 模式比對

```python
import re


class RulePattern:
    def __init__(self, pattern, response_template):
        self.regex = re.compile(pattern, re.IGNORECASE)
        self.template = response_template


PATTERNS = [
    RulePattern(r"my name is (\w+)", "Nice to meet you, {0}."),
    RulePattern(r"i (need|want) (.+)", "Why do you {0} {1}?"),
    RulePattern(r"i feel (.+)", "Why do you feel {0}?"),
    RulePattern(r"(.*)", "Tell me more about that."),
]


def rule_based_respond(user_input):
    for pattern in PATTERNS:
        m = pattern.regex.match(user_input.strip())
        if m:
            return pattern.template.format(*m.groups())
    return "I don't understand."
```

20 行的 ELIZA。反射技巧（"I feel sad" → "Why do you feel sad"）是 Weizenbaum 1966 年經典的心理治療師示範，至今仍有啟發性。

### 步驟 2：retrieval-based (FAQ)

這個示意用的程式片段需要 `pip install sentence-transformers`（它會帶入 torch）。這堂課可執行的 `code/main.py` 改用標準函式庫的 Jaccard 相似度，所以這堂課不需要外部相依套件就能執行。

```python
from sentence_transformers import SentenceTransformer
import numpy as np


FAQ = [
    ("how do i reset my password", "Go to Settings > Security > Reset Password."),
    ("how do i cancel my order", "Go to Orders, find the order, click Cancel."),
    ("what is your return policy", "30-day returns on unused items, original packaging."),
]


encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
faq_questions = [q for q, _ in FAQ]
faq_embeddings = encoder.encode(faq_questions, normalize_embeddings=True)


def faq_respond(user_input, threshold=0.5):
    q_emb = encoder.encode([user_input], normalize_embeddings=True)[0]
    sims = faq_embeddings @ q_emb
    best = int(np.argmax(sims))
    if sims[best] < threshold:
        return None
    return FAQ[best][1]
```

以門檻為基礎的拒答，是關鍵的設計選擇。如果最佳匹配不夠接近，就回傳 `None`，讓系統升級處理。

### 步驟 3：neural generation（baseline）

使用小型 instruction-tuned 的 encoder-decoder (FLAN-T5)，或經過 fine-tune 的對話模型。在 2026 年單獨使用無法上線（自相矛盾、離題漂移、事實胡扯），但會在混合系統內用來產生自然的措辭。DialoGPT 風格的 decoder-only 模型需要明確的輪次分隔符與 EOS 處理才能產生連貫的回覆；FLAN-T5 的 text2text pipeline 則開箱即用，適合作為教學範例。

```python
from transformers import pipeline

chatbot = pipeline("text2text-generation", model="google/flan-t5-small")

response = chatbot("Respond politely to: Hi there!", max_new_tokens=40)
print(response[0]["generated_text"])
```

### 步驟 4：LLM agent 迴圈

2026 年正式環境的樣貌：

```python
def agent_loop(user_message, tools, llm, max_steps=5):
    history = [{"role": "user", "content": user_message}]
    for _ in range(max_steps):
        response = llm(history, tools=tools)
        tool_call = response.get("tool_call")
        if tool_call:
            tool_name = tool_call.get("name")
            args = tool_call.get("arguments")
            if not isinstance(tool_name, str) or tool_name not in tools:
                history.append({"role": "assistant", "tool_call": tool_call})
                history.append({"role": "tool", "name": str(tool_name), "content": f"error: unknown tool {tool_name!r}"})
                continue
            if not isinstance(args, dict):
                history.append({"role": "assistant", "tool_call": tool_call})
                history.append({"role": "tool", "name": tool_name, "content": f"error: arguments must be a dict, got {type(args).__name__}"})
                continue
            fn = tools[tool_name]
            result = fn(**args)
            history.append({"role": "assistant", "tool_call": tool_call})
            history.append({"role": "tool", "name": tool_name, "content": result})
        else:
            return response["content"]
    return "I could not complete the task in the step budget."
```

有三件事要點名。工具是 LLM 可以呼叫的函式。當 LLM 回傳最終答案而不是工具呼叫時，迴圈就終止。步數預算防止在模糊任務上陷入無限迴圈。

真正的正式環境還會加上：retrieval-first 的 grounding（在每次 LLM 呼叫前注入相關文件）、guardrail（沒有確認就拒絕破壞性動作）、可觀測性（記錄每一步），以及評估（自動檢查 agent 行為維持在規格內）。

### 步驟 5：hybrid routing

```python
def hybrid_chat(user_input):
    if is_destructive_action(user_input):
        return structured_flow(user_input)

    faq_answer = faq_respond(user_input, threshold=0.6)
    if faq_answer:
        return faq_answer

    return agent_loop(user_input, tools, llm)


def is_destructive_action(text):
    danger_words = ["delete", "cancel", "charge", "refund", "transfer"]
    return any(w in text.lower() for w in danger_words)
```

模式是：任何破壞性的事用確定性規則，罐頭 FAQ 用 retrieval，其他一切交給 LLM agent。這就是 2026 年客服系統實際出貨的樣子。

## 實際使用 (Use It)

2026 年的技術棧：

| 使用情境 | 架構 |
|---------|---------------|
| 訂位、付款、身分驗證 | Rule-based 狀態機加 slot filling |
| 客服 FAQ | 在精選答案上做 retrieval |
| 開放式的協助聊天 | 搭配 RAG 與工具呼叫的 LLM agent |
| 內部工具 / IDE 助理 | 搭配工具呼叫（搜尋、讀取、寫入）的 LLM agent |
| 陪伴 / 角色聊天機器人 | 搭配 persona system prompt 的調校 LLM，對知識做 retrieval |

正式環境一律使用 hybrid routing。沒有單一架構能把每種請求都處理好。路由層本身通常是一個小型的 intent 分類器。

## 仍然會上線的失敗模式

- **自信滿滿的捏造。** LLM agent 宣稱它完成了一個其實沒做的動作。緩解方式：驗證結果、記錄工具呼叫，絕不讓 LLM 在沒有成功的工具回傳時，宣稱它做了某件事。
- **Prompt injection。** 使用者插入覆蓋 system prompt 的文字。在 OWASP Top 10 for LLM Applications 2025 中排名 LLM01。有兩種型態：直接注入（貼在聊天中）與間接注入（藏在 agent 讀取的文件、email 或工具輸出裡）。

  攻擊率因情境而異。在一般的工具使用和程式碼 benchmark 中，前沿模型被量測到的成功率約在 0.5 到 8.5% 之間。特定的高風險設定（針對 AI 程式碼 agent 的適應性攻擊、有漏洞的 orchestration）已達到約 84%。正式環境的 CVE 包括 EchoLeak (CVE-2025-32711, CVSS 9.3)，這是 Microsoft 365 Copilot 中由攻擊者控制的 email 觸發的 zero-click 資料外洩漏洞。

  緩解方式：在整個迴圈中把使用者輸入視為不可信；在工具呼叫前先清理；把工具輸出與主要 prompt 隔離；使用 Plan-Verify-Execute (PVE) 模式，讓 agent 先規劃，再在執行前對照計畫驗證每個動作（這能阻止工具結果注入未經規劃的新動作）；破壞性動作需要使用者確認；對工具範圍套用最小權限原則。

  再多的 prompt engineering 都無法完全消除這個風險。需要外部的執行期防禦層（LLM Guard、allowlist 驗證、語意異常偵測）。
- **範圍蔓延 (Scope creep)。** 因為工具呼叫回傳了略微相關的資訊，agent 跑去做不相干的事。緩解方式：收窄工具契約；讓 system prompt 保持專注；加入離題率的評估。
- **無限迴圈。** Agent 不斷呼叫同一個工具。緩解方式：步數預算、工具呼叫去重，以及由 LLM judge 判斷「我們有在進展嗎」。
- **Context window 耗盡。** 長對話把最早的輪次擠出 context。緩解方式：摘要較舊的輪次、依相似度檢索相關的過往輪次，或使用長 context 模型。

## 交付 (Ship It)

存成 `outputs/skill-chatbot-architect.md`：

```markdown
---
name: chatbot-architect
description: Design a chatbot stack for a given use case.
version: 1.0.0
phase: 5
lesson: 17
tags: [nlp, agents, chatbot]
---

Given a product context (user need, compliance constraints, available tools, data volume), output:

1. Architecture. Rule-based, retrieval, neural, LLM agent, or hybrid (specify which paths go where).
2. LLM choice if applicable. Name the model family (Claude, GPT-4, Llama-3.1, Mixtral). Match to tool-use quality and cost.
3. Grounding strategy. RAG sources, retrieval method (see lesson 14), tool contracts.
4. Evaluation plan. Task success rate, tool-call correctness, off-task rate, hallucination rate on held-out dialogs.

Refuse to recommend a pure-LLM agent for any destructive action (payments, account deletion, data modification) without a structured confirmation flow. Refuse to skip the prompt-injection audit if the agent has write access to anything.
```

## 練習 (Exercises)

1. **簡單。** 用 10 個模式，為咖啡店點餐機器人實作上面的 rule-based respond。測試邊界情況：重複下單、修改、取消、意圖不明。
2. **中等。** 建立 hybrid FAQ 加 LLM fallback。為某個 SaaS 產品準備 50 筆罐頭 FAQ，LLM fallback 對文件網站做 retrieval。在 100 個真實客服問題上測量拒答率與準確率。
3. **困難。** 用三個工具（search、read-user-data、send-email）實作上面的 agent 迴圈。用 50 個測試情境（包含 prompt injection 嘗試）執行評估。回報離題率、任務失敗率，以及任何 injection 成功的情況。

## 關鍵術語 (Key Terms)

| 術語 | 一般人怎麼說 | 實際意義 |
|------|-----------------|-----------------------|
| Intent | 使用者想要什麼 | 類別標籤（book_flight、reset_password）。導向某個 handler。 |
| Slot | 一項資訊 | 機器人需要的參數（日期、目的地）。Slot filling 是一連串的詢問。 |
| RAG | 檢索加生成 | 檢索相關文件，再讓 LLM 的回應紮根於其上。 |
| Tool call | 函式呼叫 | LLM 發出帶有名稱和參數的結構化呼叫。Runtime 執行並回傳結果。 |
| Agent loop | 規劃、行動、驗證 | 控制器交錯地執行 LLM 呼叫與工具呼叫，直到任務完成。 |
| Prompt injection | 使用者攻擊 prompt | 試圖覆蓋 system prompt 的惡意輸入。 |

## 延伸閱讀 (Further Reading)

- [Turing (1950). Computing Machinery and Intelligence](https://academic.oup.com/mind/article/LIX/236/433/986238) — 讓對話成為這個領域 benchmark 的論文。
- [Weizenbaum (1966). ELIZA — A Computer Program For the Study of Natural Language Communication](https://web.stanford.edu/class/cs124/p36-weizenabaum.pdf) — 原始的 rule-based 聊天機器人論文。
- [Colby, Weber, Hilf (1971). Artificial Paranoia](https://doi.org/10.1016/0004-3702(71)90002-6) — PARRY 的情感變數架構，第一個有狀態的聊天機器人。
- [Thoppilan et al. (2022). LaMDA: Language Models for Dialog Applications](https://arxiv.org/abs/2201.08239) — Google 後期的神經網路聊天機器人論文，就在 LLM agent 接手之前。
- [Yao et al. (2022). ReAct: Synergizing Reasoning and Acting in Language Models](https://arxiv.org/abs/2210.03629) — 命名 agent 迴圈模式的論文。
- [Anthropic's guide on building effective agents](https://www.anthropic.com/research/building-effective-agents) — 2024 年的正式環境指引，到 2026 年仍然適用。
- [Greshake et al. (2023). Not what you've signed up for: Compromising Real-World LLM-Integrated Applications with Indirect Prompt Injection](https://arxiv.org/abs/2302.12173) — prompt injection 論文。
- [OWASP Top 10 for LLM Applications 2025 — LLM01 Prompt Injection](https://genai.owasp.org/llmrisk/llm01-prompt-injection/) — 讓 prompt injection 成為首要安全疑慮的排名。
- [AWS — Securing Amazon Bedrock Agents against Indirect Prompt Injections](https://aws.amazon.com/blogs/machine-learning/securing-amazon-bedrock-agents-a-guide-to-safeguarding-against-indirect-prompt-injections/) — 實用的 orchestration 層防禦，包括 Plan-Verify-Execute 與使用者確認流程。
- [EchoLeak (CVE-2025-32711)](https://www.vectra.ai/topics/prompt-injection) — 來自間接 prompt injection 的標準 zero-click 資料外洩 CVE。說明為什麼有寫入權限的 agent 需要執行期防禦的參考案例。
