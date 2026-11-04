# Group Chat 與 Speaker Selection (Group Chat and Speaker Selection)

> 共享對話式的 orchestration，把 N 個 agent 放進同一場對話；由 selector 函式（LLM、round-robin 或自訂）挑出下一個發言者。這是湧現式 multi-agent 對話的原型：agent 不知道自己在靜態圖中的角色，只是對共享的 pool 做出反應。AutoGen GroupChat 和 AG2 GroupChat 是參考實作：AutoGen v0.2 的 GroupChat 語意保留在 AG2 fork 中；AutoGen v0.4 則把它重寫成事件驅動的 actor 模型。Microsoft 在 2026 年 2 月把 AutoGen 轉入維護模式，並與 Semantic Kernel 合併成 Microsoft Agent Framework（2026 年 2 月 RC）。GroupChat 這個原語在 AG2 和 Microsoft Agent Framework 中都留了下來，學一次，到處都能用。

**Type:** Learn + Build
**Languages:** Python (stdlib)
**Prerequisites:** Phase 16 · 04 (Primitive Model)
**Time:** ~60 分鐘

## 問題 (Problem)

當工作流程已知時，靜態圖（LangGraph）很棒。真實的對話不是靜態的：有時 coder 問 reviewer，有時問 researcher，有時問 writer。把每個可能的 handoff 都寫死，會造成邊的爆炸。你想要的是 *agent 對共享 pool 做出反應*，再由某個函式決定下一個誰說話。

這正是 AutoGen GroupChat 做的事。

## 核心概念 (Concept)

### 形狀 (The shape)

```
              ┌─── shared pool ────┐
              │   m1  m2  m3  ...  │
              └─────────┬──────────┘
                        │ (everyone reads all)
      ┌───────┬─────────┼─────────┬───────┐
      ▼       ▼         ▼         ▼       ▼
    Agent A  Agent B  Agent C  Agent D  Selector
                                           │
                                           ▼
                                  "next speaker = C"
```

每個 agent 都看得到每則訊息。每一輪都會呼叫 selector 函式，挑出下一個發言者。

### 三種 Selector (The three selector flavors)

**Round-robin。** 固定循環。確定性。隨 N 線性擴展，但忽略 context，即使話題是法務審查，coder 也會輪到發言。

**LLM 選擇 (LLM-selected)。** 呼叫一次 LLM，讀取最近的 pool 並回傳最適合的下一位發言者。能感知 context 但很慢：每一輪都多一次 LLM 呼叫。這是 AutoGen 的預設。

**自訂 (Custom)。** 一個 Python 函式，邏輯隨你寫。典型做法：LLM 選擇，再加上備援規則（例如「coder 之後一定輪到 verifier」）。

### ConversableAgent API

```
agent = ConversableAgent(
    name="coder",
    system_message="You write Python.",
    llm_config={...},
)
chat = GroupChat(agents=[coder, reviewer, tester], messages=[])
manager = GroupChatManager(groupchat=chat, llm_config={...})
```

`GroupChatManager` 持有 selector。當一個 agent 完成一輪，manager 呼叫 selector，由它回傳下一個 agent。迴圈持續到終止條件成立。

### 終止 (Termination)

三種常見 pattern：

- **最大輪數。** 對總輪數的硬性上限。
- **「TERMINATE」token。** Agent 可以送出哨兵訊息；manager 看到時就停止。
- **目標達成檢查。** 一個輕量的 verifier 每輪執行，完成時停止對話。

### 淵源：fork 與合併 (Lineage: forks and mergers)

2025 年初，Microsoft 開始圍繞事件驅動的 actor 模型，對 AutoGen (v0.4) 進行大幅重寫。社群把 AutoGen v0.2 的 GroupChat 語意 fork 成 AG2，保留早期採用者已經整合的 API。

2026 年 2 月，Microsoft 宣布 AutoGen 將進入維護模式，事件驅動的 actor 模型併入 **Microsoft Agent Framework**（2026 年 2 月 RC，現已與 Semantic Kernel 合併）。GroupChat 的概念在兩條路線中都留下來；實作細節不同。AG2 是與 v0.2 相容程式碼的首選上游。

### GroupChat 適合的時候 (When GroupChat fits)

- **湧現式對話。** 你不想預先接好每一個可能的下一位發言者。
- **角色混合的任務。** Coder 問 researcher，researcher 問 archivist，archivist 又回頭問 coder。流程不是 DAG。
- **探索式的問題解決。** 想想「腦力激盪會議」，而不是「組裝線」。

### 它失敗的時候 (When it fails)

- **嚴格的確定性。** LLM selector 可能不一致。同樣的 prompt，不同次執行，下一位發言者不同。
- **阿諛串聯 (Sycophancy cascades)。** Agent 順從說話最有信心的人。要明確地用 prompt 反制。
- **Context 膨脹。** 每個 agent 都讀每則訊息；10 輪之後 context 就很大。用 projection（Lesson 15）限定視圖範圍。
- **熱門發言者 (Hot speakers)。** 因為 selector 偏好它的專長，一個 agent 主導了整場對話。把發言平衡作為 selector 的一個特徵引入。

### Group chat 與 supervisor (Group chat vs supervisor)

同樣的原語，不同的預設：

- Supervisor：一個 agent 規劃，其他 agent 執行。Selector 是「問 planner 該做什麼」。
- Group chat：所有 agent 是平等的 peer；selector 是作用在共享 pool 上的函式。

兩者都使用 Lesson 04 的四個原語。Group chat 預設使用 LLM 選擇的 orchestration 和完整 pool 的 shared state。

```figure
swarm-speaker
```

## 動手做 (Build It)

`code/main.py` 用標準函式庫從零實作 GroupChat。三個 agent（coder、reviewer、manager）、round-robin 與 LLM 選擇兩種變體，並以 `TERMINATE` token 終止。

Demo 會印出對話紀錄，以及兩種變體下 selector 的決策追蹤。

執行：

```
python3 code/main.py
```

## 實際使用 (Use It)

`outputs/skill-groupchat-selector.md` 為指定的任務設定 GroupChat selector：round-robin、LLM 選擇或自訂，以及要使用哪些 selector 輸入（最近的訊息、agent 專長、輪次計數）。

## 交付 (Ship It)

檢查清單：

- **最大輪數上限。** 一定要有。一般任務 10-20。
- **發言平衡指標。** 追蹤每個 agent 的發言輪數；失衡超過門檻時發出警報。
- **終止 token。** `TERMINATE` 或專用的 verifier agent。
- **Projection 或限定範圍的記憶體。** 大約 10 則訊息之後，考慮只給每個 agent 限定範圍的視圖，以避免 context 膨脹。
- **Selector 紀錄。** 對 LLM 選擇的變體，同時記錄 selector 的輸入與它的選擇。否則無法除錯。

## 練習 (Exercises)

1. 執行 `code/main.py`。比較 round-robin 與 LLM 選擇下的對話。在每種情況下，哪個 agent 佔主導？
2. 在 selector 中加入「每個 agent 最多發言次數」規則。它如何影響對話紀錄？
3. 實作目標達成的終止條件：當 reviewer 回傳「approved」時停止。它在輪數上限之前觸發的頻率有多高？
4. 閱讀 AutoGen stable 文件中關於 GroupChat 的部分 (https://microsoft.github.io/autogen/stable/user-guide/core-user-guide/design-patterns/group-chat.html)。找出 `GroupChatManager` 使用的預設 selector。
5. 閱讀 AG2 repo (https://github.com/ag2ai/ag2)，比較它的 v0.2 GroupChat 和 v0.4 事件驅動版本。v0.4 增加了哪個具體特性（吞吐量、容錯、可組合性）？

## 關鍵術語 (Key Terms)

| 術語 | 一般說法 | 實際意義 |
|------|----------------|------------------------|
| GroupChat | 「Agent 在同一間聊天室」 | 共享訊息 pool + selector 函式。AutoGen / AG2 的原語。 |
| Speaker selection | 「下一個誰說話」 | 挑出下一個 agent 的函式。Round-robin、LLM 選擇或自訂。 |
| GroupChatManager | 「會議主持人」 | 持有 selector 並逐輪迴圈的 AutoGen 元件。 |
| ConversableAgent | 「基底 agent」 | AutoGen 的基底類別；能傳送與接收訊息的 agent。 |
| Termination token | 「『停止』的暗號」 | 結束對話的哨兵字串（通常是 `TERMINATE`）。 |
| Hot speaker | 「一個 agent 獨佔」 | 失敗模式：selector 一直挑同一個 agent。 |
| Context bloat | 「Pool 無限增長」 | 每個 agent 都讀每則先前的訊息；context 隨輪次增長。 |
| Projection | 「限定範圍的視圖」 | 對共享 pool 依角色做的視圖，避免 context 膨脹。 |

## 延伸閱讀 (Further Reading)

- [AutoGen group chat docs](https://microsoft.github.io/autogen/stable/user-guide/core-user-guide/design-patterns/group-chat.html) — 參考實作
- [AG2 repo](https://github.com/ag2ai/ag2) — 社群延續的 AutoGen v0.2
- [Microsoft Agent Framework docs](https://learn.microsoft.com/en-us/agent-framework/) — 合併後的繼任者，2026 年 2 月 RC
- [AutoGen v0.4 release notes](https://microsoft.github.io/autogen/stable/) — 事件驅動 actor 模型重寫的細節
