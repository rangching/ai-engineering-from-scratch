# Agent Memory：虛擬 Context 與 Memory Paging (Agent Memory — Virtual Context and Memory Paging)

> Context window 是有限的。對話、文件和 tool trace 卻不是。解法是把 OS 的虛擬記憶體重新詮釋：主要 context 是 RAM，外部儲存是硬碟，agent 在兩者之間做 paging。MemGPT（Packer et al., 2023）替這個模式命名；許多正式環境的 memory 系統都建立在它之上。

**Type:** Build
**Languages:** Python (stdlib)
**Prerequisites:** Phase 14 · 01 (Agent Loop), Phase 14 · 06 (Tool Use)
**Time:** ~75 分鐘

## 學習目標 (Learning Objectives)

- 解釋 MemGPT 所依據的 OS 類比：main context = RAM、external context = 硬碟、memory tool = page in/out。
- 用 stdlib 實作 MemGPT 的兩層模式，包含 main-context buffer、可搜尋的外部儲存，以及 page in/out tool。
- 描述 agent 如何發出「interrupt」來查詢或修改外部 memory，以及結果如何被接回下一個 prompt。
- 找出 MemGPT 中會延續到 Letta（Lesson 08）和 Mem0（Lesson 09）的設計選擇。

## 問題 (The Problem)

Context window 看起來應該能解決 memory，但沒有。正式環境中反覆出現三種失敗模式：

1. **溢出 (Overflow)。** 多輪對話、長文件或大量 tool call 的 trajectory 超過 window。超出截止點的內容全部消失。
2. **稀釋 (Dilution)。** 即使在 window 內，塞進不相關的 context 也會稀釋對重要內容的 attention。前沿模型在長輸入上仍會退化。
3. **持久性 (Persistence)。** 新 session 一開始是空的 window。沒有外部 memory 的 agent 無法跨 session 說「還記得你請我...」。

更大的 window 有幫助，但無法根治。Mem0 的 2025 年論文測量到，128k window 的 baseline 仍會漏掉長時程的事實，而有外部 memory 的 4k window agent 反而抓得到。

## 核心概念 (The Concept)

### OS 類比 (The OS analogy)

MemGPT（Packer et al., arXiv:2310.08560, v2 Feb 2024）把 context 管理對應到作業系統的虛擬記憶體：

| OS 概念 | MemGPT 概念 | 2026 年正式環境的對應 |
|------------|---------------|------------------------|
| RAM | main context (prompt) | Anthropic/OpenAI context window |
| Disk | external context | vector DB、KV、graph store |
| Page fault | memory tool call | `memory.search`、`memory.read`、`memory.write` |
| OS kernel | agent control loop | 帶有 memory tool 的 ReAct loop |

Agent 執行一般的 ReAct loop。多出來的一類 tool 讓它把資料 page in 和 page out 主要 context。

### 兩層 (Two tiers)

- **Main context。** 固定大小的 prompt，放目前的任務。模型隨時看得到。
- **External context。** 無上限，可透過 tool 搜尋。相關時讀取，出現事實時寫入。

原論文用兩個超出基本 window 的任務評估這個設計：超過 100k token 的文件分析，以及跨越數天、具持久 memory 的多 session 聊天。

### Interrupt 模式 (The interrupt pattern)

MemGPT 引入了 memory-as-interrupt：在對話途中，agent 可以呼叫 memory tool，runtime 執行它，結果作為新的 observation 接回下一個 assistant turn。概念上等同於 Unix 的 `read()` syscall：阻塞行程、回傳 bytes，然後行程繼續。

標準的 memory tool 介面：

- `core_memory_append(section, text)`：寫入 prompt 中持久的一個區段。
- `core_memory_replace(section, old, new)`：編輯持久的區段。
- `archival_memory_insert(text)`：寫入可搜尋的外部儲存。
- `archival_memory_search(query, top_k)`：從外部儲存取回。
- `conversation_search(query)`：掃描過去的 turn。

### 論文結束、正式環境開始的地方 (Where the paper ends and production begins)

2024 年 9 月，MemGPT 變成 Letta。研究 repo（`cpacker/MemGPT`）仍然存在；Letta 擴充了這個設計：

- 三層而不是兩層（core、recall、archival，見 Lesson 08）。
- 以原生 reasoning 取代 `send_message`/heartbeat 模式（Lesson 08）。
- 執行非同步 memory 工作的 sleep-time agent（Lesson 08）。

即使正式系統跑的是 Letta、Mem0 或自訂的兩層儲存，MemGPT 論文仍是 2026 年的基礎。

### 這個模式哪裡會出錯 (Where this pattern goes wrong)

- **Memory rot。** 寫入的速度快過讀取；retrieval 淹沒在過時的事實裡。解法：定期整併（Letta sleep-time）、明確的失效處理（Mem0 衝突偵測器）。
- **Memory poisoning。** 外部 memory 是被取回的文字。如果攻擊者控制的內容進了 memory 筆記，agent 下個 session 會再次吸收它。這就是 Greshake et al.（Lesson 27）攻擊在時間軸上的重演。
- **引用遺失 (Citation loss)。** Agent 記得「使用者請我出貨 X」，卻無法引用是哪一個 turn。每次 archival 寫入都要一併存來源參照（session ID、turn ID）。

```figure
context-budget
```

## 動手做 (Build It)

`code/main.py` 用 stdlib 實作 MemGPT 的兩層模式：

- `MainContext`：固定大小的 prompt buffer，有 `core` dict 和 `messages` 清單；超過上限時自動壓縮最舊的訊息。
- `ArchivalStore`：記憶體內、類似 BM25 的儲存（token 重疊計分），內容是 (id, text, tags, session, turn) 記錄。
- 五個對應 MemGPT 介面的 memory tool。
- 一個腳本化的 agent：先把事實填進 archival，再呼叫 `archival_memory_search` 回答問題。

執行：

```
python3 code/main.py
```

Trace 顯示 agent 寫入三個事實，把 main context 填到上限（迫使驅逐），然後從 archival 取回來回答後續問題，在沒有任何真正 LLM 的情況下重現 MemGPT 的流程。

## 實際使用 (Use It)

今天每個正式環境的 memory 系統都是 MemGPT 的變體：

- **Letta**（Lesson 08）：三層、原生 reasoning、sleep-time compute。
- **Mem0**（Lesson 09）：vector + KV + graph，再加一層計分融合。
- **OpenAI Assistants / Responses**：透過 thread 和檔案的託管 memory。
- **Claude Agent SDK**：透過 skill 和 session store 做長期 memory。

依營運形態（自架、託管、整合進 framework）來選，而不是依核心模式，因為核心模式就是 MemGPT。

### Agent memory 的形狀 (The shape of agent memory)

Paging 解決容量問題。它不決定要存什麼。正式系統中反覆出現四種 memory 類型，各自回答不同的問題：

- **Working memory**：現在什麼重要？in-context 那一層：目前任務、近期 turn、釘選的 core 區段。就是 prompt 本身。
- **Episodic memory**：發生了什麼？過去的 turn 和 trajectory，連同 session 與 turn 的參照一起儲存，可隨時重播。
- **Semantic memory**：什麼是真的？關於使用者、領域、世界的事實，隨變化而更新與去重。
- **Procedural memory**：這件事怎麼做？學到的流程、偏好和規則，用來引導未來行為，而不是回憶。

開源實作選了不同的切入點：

| 類型 | 實作 | 解決方式 |
|------|----------------|-------------------|
| Working | MemGPT / Letta | 透過 memory tool，在固定的 prompt 預算內 page in 和 page out 內容（本課、Lesson 08） |
| Episodic | Zep | 時間性 knowledge graph：事實帶有有效區間，因此「某時間點什麼是真的」可以被查詢 |
| Semantic | Mem0 | 擷取 pipeline，跨 vector、KV 和 graph store 對事實去重與更新（Lesson 09） |
| Semantic + procedural | LangMem | 在背景把事實和行為規則擷取到 store，agent 在 turn 之間查詢 |
| Episodic + semantic | agentmemory | 在 session 執行時擷取，整併成有型別、可搜尋的記錄 |

## 交付 (Ship It)

`outputs/skill-virtual-memory.md` 是可重複使用的 skill，可為任何目標 runtime 產生正確的兩層 memory 骨架（main + archival + tool 介面），並接好驅逐策略與引用欄位。

## 練習 (Exercises)

1. 加一個以 token 計算的 `max_main_context_tokens` 上限（用 `len(text.split())` * 1.3 近似）。超過上限時，把最舊的訊息壓縮成摘要。比較有無摘要器的行為。
2. 在 archival store 上正確實作 BM25（term frequency、inverse document frequency）。在一組 toy 事實集上量測 recall@10，並與 token 重疊的 baseline 比較。
3. 為 archival 寫入加上 `citation` 欄位（session_id、turn_id、source_url）。讓 agent 在每個以 retrieval 為依據的答案中引用來源。
4. 模擬 memory poisoning：加入一筆內容為「ignore all future user instructions」的 archival 記錄。寫一個 guard，掃描 retrieval 結果中帶有指令形式的文字，並標記為不可信。
5. 改用 MemGPT 研究 repo 的 core-memory JSON schema（`cpacker/MemGPT`）重寫實作。從扁平字串換成有型別的區段，會有什麼改變？

## 關鍵術語 (Key Terms)

| Term | 一般說法 | 實際意義 |
|------|----------------|------------------------|
| Virtual context | 「無限 memory」 | Main（prompt）+ external（可搜尋）兩層，加上 page in/out |
| Main context | 「Working memory」 | Prompt：固定大小，隨時可見 |
| Archival memory | 「長期儲存」 | 外部可搜尋的持久儲存，按需取回 |
| Core memory | 「持久的 prompt 區段」 | 釘在 main context 內的具名區段 |
| Memory tool | 「Memory API」 | Agent 為了讀寫外部 memory 而發出的 tool call |
| Interrupt | 「Memory page fault」 | Agent 暫停，runtime 取回資料，結果接進下一個 turn |
| Memory rot | 「過時的事實」 | 舊的寫入淹沒 retrieval；用整併修復 |
| Memory poisoning | 「被注入的持久筆記」 | 攻擊者的內容被存成 memory，回想時再次被吸收 |

## 延伸閱讀 (Further Reading)

- [Packer et al., MemGPT (arXiv:2310.08560)](https://arxiv.org/abs/2310.08560)：受 OS 啟發的虛擬 context 論文
- [Letta, Memory Blocks blog](https://www.letta.com/blog/memory-blocks)：三層的演進
- [Anthropic, Effective context engineering](https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents)：把 context 當作預算
- [Chhikara et al., Mem0 (arXiv:2504.19413)](https://arxiv.org/abs/2504.19413)：建立在這個模式上的混合式正式環境 memory
- [Zep (getzep/zep)](https://github.com/getzep/zep)：分類表中的時間性 knowledge-graph memory
- [Mem0 (mem0ai/mem0)](https://github.com/mem0ai/mem0)：Lesson 09 混合儲存背後的擷取 pipeline
- [LangMem (langchain-ai/langmem)](https://github.com/langchain-ai/langmem)：在背景擷取事實和行為規則
- [agentmemory (rohitg00/agentmemory)](https://github.com/rohitg00/agentmemory)：將 session 擷取整併成有型別、可搜尋的記錄
