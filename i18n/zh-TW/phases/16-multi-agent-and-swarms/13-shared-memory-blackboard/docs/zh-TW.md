# Shared Memory 與 Blackboard Pattern (Shared Memory and Blackboard Patterns)

> 2026 年的 multi-agent 系統中有兩種做法並存：**message pool**（每個人都看得到所有人的訊息，如 AutoGen GroupChat 或 MetaGPT），以及 **帶訂閱的 blackboard**（agent 訂閱相關事件，如 Context-Aware MCP 或 Matrix 框架）。兩者都是 multi-agent 系統中唯一有狀態的部分，也就是有趣的 bug 所在之處。典型的失敗模式是 **memory poisoning**：一個 agent 幻想出一個「事實」，其他 agent 把它當成已驗證的，準確度就這樣逐漸衰退，比立即當機難除錯得多。這堂課用標準函式庫建出這兩種結構，注入一次 poisoning 攻擊，並展示在正式環境中真正有效的三種緩解方式。

**Type:** Learn + Build
**Languages:** Python (stdlib, `threading`)
**Prerequisites:** Phase 16 · 04 (Primitive Model), Phase 16 · 09 (Parallel Swarm Networks)
**Time:** ~75 分鐘

## 問題 (Problem)

Multi-agent 系統需要一個讓 agent 分享事實的地方。字面上的選項是「所有東西都用訊息傳」，但這只是用額外的複製重新發明 shared state。另一個是「給每個人一份全域 log」，但全域 log 會無限增長，而且容易被汙染。第三個是「為每個 agent 投影一個視圖」，可擴展，但 schema 負擔很重。

當其中一個 agent 產生幻想並把幻想寫進 shared state，每個讀取該 state 的下游 agent 都會把這個幻想當成事實。等到人類發現時，推理鏈已經深到五步，而根本原因是寫下的第三則訊息。除錯 multi-agent 的準確度衰退，比除錯當機更難。

這就是 memory poisoning。它是 MAST 分類法中記錄第二多的失敗家族 (Cemri et al., arXiv:2503.13657)，而且是結構性的：任何沒有 provenance 和不可寫入的 verifier 的 shared-memory 設計，遲早都會出現這個問題。

## 核心概念 (Concept)

### 兩種主要的拓撲 (The two main topologies)

**完整的 message pool。** 每個 agent 讀每一則訊息。AutoGen GroupChat 和 MetaGPT 使用這種方式。簡單、透明、可檢視，但擴展不到約 10 個 agent 以上，因為每個 agent 的 context 會被其他 agent 的工作塞滿。

```
agent-A ──write──▶ ┌────────────────┐ ◀──read── agent-D
                   │ message pool   │
agent-B ──write──▶ │                │ ◀──read── agent-E
                   │ (global log)   │
agent-C ──write──▶ └────────────────┘ ◀──read── agent-F
```

**帶訂閱的 blackboard。** Agent 宣告對某些 topic 感興趣；底層基礎只路由相關的訊息。CA-MCP (arXiv:2601.11595) 和 Matrix 去中心化框架 (arXiv:2511.21686) 使用這種方式。擴展得更遠，但需要事先設計 schema，訂閱才有意義。

```
                   ┌─ topic: prices ──┐
agent-A ──pub────▶ │                  │ ──▶ agent-D (subscribed)
                   ├─ topic: orders ──┤
agent-B ──pub────▶ │                  │ ──▶ agent-E (subscribed)
                   ├─ topic: alerts ──┤
agent-C ──pub────▶ │                  │ ──▶ agent-F (subscribed)
                   └──────────────────┘
```

### 各自勝出的時候 (When each wins)

- **完整 pool** 勝出的條件：agent 很少（< 10 個）、彼此異質，且對話是短期的。當每個人都看到全部，要推理誰說了什麼就很容易。
- **Blackboard** 勝出的條件：agent 很多、角色同質但實例很多（swarm），且對話長時間進行。路由可以節省 token 成本和 context 汙染。

正式環境的系統常常混用：頂層是小型的完整 pool（規劃層），底下是 blackboard（worker 層）。

### 用一個情境講 Memory Poisoning (Memory poisoning, in one scenario)

三個 agent 做一項研究任務。Agent A 是檢索 agent。Agent B 是摘要者。Agent C 是分析師。

1. A 抓取一個頁面，並對 shared state 寫入一則訊息：「該研究報告了 42% 的準確度提升。」
2. 被抓取的頁面其實寫的是「4.2% 的提升」。A 幻想出一個小數點。
3. B 讀取 shared state，寫下：「報告了 42% 的大幅準確度提升（來源：A）。」
4. C 讀取 shared state，寫下：「建議採用，42% 的提升是變革性的。」
5. 最終報告引用了一個從未存在過的 42% 數字。

沒有 agent 當機。沒有測試失敗。系統「運作」得好好的。這個幻想透過 shared state，從一個 agent 的 context 跨進每一個下游 agent 的推理。

### 為什麼這是結構性的 (Why this is structural)

沒有 shared state 的話，agent A 的幻想會留在 A 的 context 裡。下游 agent 會重新抓取或重新推導，也許會抓到錯誤。有了天真的 shared state，A 的 context 就變成每個人的 context，幻想就被洗成了事實。

問題不在 shared state 本身，而在於 **沒有 provenance、也沒有獨立 verifier 的** shared state。有三種緩解方式處理這件事：

1. **每次寫入都標註 provenance。** Shared state 中的每個項目都記錄誰寫的、何時、在什麼 prompt 之下，以及（如適用）agent 引用了什麼來源。下游 agent 依 provenance 抱持相應的懷疑來讀取。
2. **寫入要有版本，並視為 append-only。** 更正是一個取代舊項目的新項目，而不是原地更新。稽核軌跡得以保留。
3. **至少保留一個不能寫入 shared state 的 agent。** 唯讀的 verifier agent 抽樣檢查項目、重新抓取來源，並標出不一致之處。它仍然讀取 pool，所以可能被錯誤的項目騙到，但它會對照原始來源檢查每個項目，又不能寫入，所以它的判斷永遠不會把被汙染的事實餵回 shared state。

### Blackboard 的先例（Hayes-Roth, 1985）(Blackboard precedent (Hayes-Roth, 1985))

Blackboard pattern 比 LLM agent 早了四十年。Hayes-Roth（1985，"A Blackboard Architecture for Control"）描述了專家式的 Knowledge Source，它們觀察全域的 blackboard、貢獻部分解答，並觸發其他 source。2026 年的 blackboard（CA-MCP、Matrix）是同樣的 pattern，只是 Knowledge Source 換成 LLM agent，部分解答換成 JSON blob。舊文獻已經記錄了寫入競爭、機會式控制 (opportunistic control) 與一致性的解法，現代系統正在重新發現它們。

### Projection 與完整視圖 (Projection vs full view)

純粹的 blackboard 給每個訂閱者相同的 projection（以 topic 為範圍）。更積極的設計是 **每個 agent 各自的 projection**：每個 agent 得到針對其角色客製的視圖。LangGraph 的 state reducer 是 2026 年的標準實作：reducer 函式把全域 state 折疊成依角色切出的片段。

每個 agent 各自的 projection 能擴展得更遠，但需要 schema。沒有的話，你就得在每個 agent 的 prompt 裡重建臨時拼湊的 projection。

### 寫入競爭 pattern (Write-contention patterns)

多個 agent 同時寫入是並行問題，不只是 LLM 的問題。有三種 pattern 有效：

- **循序寫入者（單一 producer）。** 所有寫入都經過一個協調者 agent 序列化。簡單，但是瓶頸。
- **帶版本的樂觀並行 (optimistic concurrency)。** 每個項目有版本；版本不符時寫入者失敗並重試。經典的資料庫技術。
- **Topic 分區。** 不同的 agent 擁有不同的 topic。沒有跨 topic 的競爭。需要設計好分區邊界。

2026 年大多數框架預設使用循序寫入者，因為 LLM 呼叫夠慢，競爭很少發生，瓶頸也不會造成傷害。

### 不可寫入的 Verifier (The unwritable verifier)

最承重的緩解方式是唯讀 verifier。實作規則：

- Verifier 與團隊共享 state（讀取 blackboard 或 pool）。
- Verifier 對 shared state 沒有寫入 handle，只能寫入一個獨立的驗證頻道。
- Verifier 獨立抓取寫入中引用的來源。標出不一致。
- Verifier 自己的輸出被送到人類或獨立的決策 agent，絕不回灌到 pool 裡。

沒有這種分離，verifier 的輸出會變成 pool 中的新項目，也就是說被汙染的 pool 會汙染 verifier，進而汙染它的驗證結果。

```figure
swarm-blackboard
```

## 動手做 (Build It)

`code/main.py` 用標準函式庫 Python 實作這兩種拓撲，外加一個玩具版的 poisoning 攻擊與三種緩解方式。

- `MessagePool` — 執行緒安全、append-only 的 log，可完整讀出。
- `Blackboard` — 以 topic 為鍵的 pub/sub，每個 agent 有各自的訂閱。
- `ProvenanceEntry` — 每次寫入都記錄 (writer, timestamp, prompt_hash, source_uri)。
- `PoisoningScenario` — 執行一個三 agent 的研究任務，其中 agent A 幻想出一個小數點。印出最終報告。
- `Verifier` — 唯讀的 agent，重新抓取來源並標出不一致。在 verifier 在場的情況下執行同一個情境。

執行：

```
python3 code/main.py
```

預期輸出：
- 第 1 次執行（沒有 verifier）：幻想出來的 42% 傳播到最終報告。
- 第 2 次執行（有 verifier）：verifier 標出不一致，pool 被標記為「flagged」，最終報告包含撤回聲明。

## 實際使用 (Use It)

`outputs/skill-memory-auditor.md` 是一個 skill，用來稽核任何 multi-agent 系統的 shared-memory 設計是否具備 provenance、版本控制與 verifier 分離。在新的 multi-agent 架構上線之前先跑一遍。

## 交付 (Ship It)

對任何 shared-memory 設計：

- 每次寫入都記錄 provenance：`(writer, timestamp, prompt_hash, tool_calls_cited, source_uri)`。
- 讓 log 保持 append-only。更正是引用被取代項目的新項目。
- 部署至少一個唯讀的 verifier agent，並讓它有獨立的來源存取權。
- 把 verifier 的輸出送到獨立的頻道，不要回灌到 shared pool。
- 記錄屬於取代（supersession）的寫入比例，這個比例上升是幻想模式的早期證據。

## 練習 (Exercises)

1. 執行 `code/main.py`。確認第 1 次執行傳播了幻想，而第 2 次執行抓到了它。
2. 加入第二個幻想：agent B 捏造一個資料集大小。Verifier 應該在不為任一者手動調整的情況下，兩個都抓到。
3. 把完整 pool 換成帶 topic 分區（`prices`、`summaries`、`analyses`）的 blackboard。Topic 分區讓哪些 poisoning 情境更難得逞？又對哪些沒有幫助？
4. 閱讀 Hayes-Roth（1985，"A Blackboard Architecture for Control"）。找出論文中兩個本課沒有討論、而 2026 年的系統會受益的控制 pattern。
5. 閱讀 CA-MCP (arXiv:2601.11595)。把它的 Shared Context Store 對應到 `code/main.py` 中的 MessagePool 或 Blackboard 類別。CA-MCP 在上面額外加了哪些原語？

## 關鍵術語 (Key Terms)

| 術語 | 一般說法 | 實際意義 |
|------|----------------|------------------------|
| Message pool | 「共享聊天歷史」 | 每個 agent 都會讀的 append-only log。完全透明，擴展性差。 |
| Blackboard | 「共享工作區」 | 以 topic 為鍵的 pub/sub。Agent 訂閱相關的 topic。擴展得更遠。 |
| Provenance | 「誰寫了什麼」 | 每次寫入的 metadata：寫入者、時間戳記、prompt、來源。 |
| Memory poisoning | 「幻想擴散」 | 一個 agent 的錯誤進入 shared state，下游 agent 把它當成事實。 |
| Append-only | 「不做原地更新」 | 更正是取代舊項目的新項目。保留稽核軌跡。 |
| Unwritable verifier | 「獨立稽核者」 | 唯讀的 agent，重新抓取來源並標出不一致。 |
| Projection | 「限定範圍的視圖」 | 從全域 state 計算出的每個 agent 的視圖。LangGraph reducer 是標準案例。 |
| Knowledge Source | 「專家 agent」 | Hayes-Roth 在 1985 年對 blackboard 參與者的稱呼。 |

## 延伸閱讀 (Further Reading)

- [Cemri et al. — Why Do Multi-Agent LLM Systems Fail?](https://arxiv.org/abs/2503.13657) — MAST 分類法；memory poisoning 是協調失敗的子家族
- [CA-MCP — Context-Aware Multi-Server MCP](https://arxiv.org/abs/2601.11595) — 用於協調多個 MCP server 的 Shared Context Store
- [Matrix — decentralized multi-agent framework](https://arxiv.org/abs/2511.21686) — 沒有中央 orchestrator、以 message queue 為基礎的 blackboard
- [LangGraph state and reducers](https://docs.langchain.com/oss/python/langgraph/workflows-agents) — 正式環境中每個 agent 各自的 projection pattern
- [Anthropic — How we built our multi-agent research system](https://www.anthropic.com/engineering/multi-agent-research-system) — 正式部署中關於 provenance 與驗證的筆記
