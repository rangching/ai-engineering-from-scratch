# Memory Block 與 Sleep-Time Compute (Memory Blocks and Sleep-Time Compute)

> 模型可以直接編輯的離散功能型 memory block，以及在主要 agent 閒置時非同步整併 memory 的 sleep-time agent。這兩個想法，是讓 memory 擴展到超過單一對話的方法。

**Type:** Build
**Languages:** Python (stdlib)
**Prerequisites:** Phase 14 · 07 (MemGPT)
**Time:** ~75 分鐘

## 學習目標 (Learning Objectives)

- 說出 Letta 使用的三個 memory 層級（core、recall、archival）以及各自的角色。
- 解釋 memory-block 模式：Human block、Persona block，以及作為一級有型別物件的使用者自訂 block。
- 描述 sleep-time compute 是什麼、為什麼它不在關鍵路徑上，以及為什麼它可以跑比主要 agent 更強的模型。
- 實作腳本化的雙 agent loop：主要 agent 負責回應，sleep-time agent 在 turn 之間整併 block。

## 問題 (The Problem)

MemGPT（Lesson 07）解決了虛擬記憶體的控制流程。正式環境出現了三個問題：

1. **延遲 (Latency)。** 每個 memory 操作都在關鍵路徑上。如果 agent 得在使用者等待時修剪、摘要或調和，尾端延遲就會暴增。
2. **Memory rot。** 寫入不斷累積。被推翻的事實仍留著。Retrieval 淹沒在過時內容裡。
3. **結構遺失 (Structure loss)。** 扁平的 archival store 無法表達「Human block 永遠在 prompt 裡；Persona block 永遠在 prompt 裡；Task block 每個 session 輪替」。

Letta（letta.com）是原 MemGPT 專案在 2024 年採用的平台名稱（論文的模式仍沿用 MemGPT 這個名字），而 2026 年的 Letta V1 重寫是之後獨立的一步。Memory block 讓結構變明確；sleep-time compute 把整併移出關鍵路徑。

## 核心概念 (The Concept)

### 三層 (Three tiers)

| 層級 | 範圍 | 存放位置 | 寫入者 |
|------|-------|----------------|------------|
| Core | 隨時可見 | 在主要 prompt 內 | Agent tool call + sleep-time 重寫 |
| Recall | 對話歷史 | 可取回 | 自動記錄 turn |
| Archival | 任意事實 | Vector + KV + graph | Agent tool call + sleep-time 匯入 |

Core 就是 MemGPT 的 core。Recall 是對話 buffer 加上被驅逐的尾端。Archival 是外部儲存。這個切分整理了 MemGPT 兩層設計中職責重疊的問題。

### Memory block

Block 是 core 層中有型別、持久、可編輯的區段。原 MemGPT 論文定義了兩個：

- **Human block**：關於使用者的事實（姓名、角色、偏好、目標）。
- **Persona block**：agent 的自我概念（身分、語氣、限制）。

Letta 把它一般化成任意的使用者自訂 block：目前目標用 `Task` block、程式碼庫事實用 `Project` block、硬性限制用 `Safety` block。每個 block 有 `id`、`label`、`value`、`limit`（字元上限）、`description`（讓模型知道何時該編輯它）。

Block 可透過 tool 介面編輯：

- `block_append(label, text)`
- `block_replace(label, old, new)`
- `block_read(label)`
- `block_summarize(label)`：壓縮接近上限的 block。

### Sleep-time compute

2025 年 Letta 的新增功能：在背景、關鍵路徑之外，執行第二個 agent。Sleep-time agent 處理對話 transcript 和程式碼庫 context，把 `learned_context` 寫進共用 block，並整併或使 archival 記錄失效。

由此自然得到的特性：

- **沒有延遲成本。** 主要回應不必等 memory 操作。
- **允許更強的模型。** Sleep-time agent 可以是更貴、更慢的模型，因為它不受延遲限制。
- **自然的整併時間窗。** 在使用者沒在等待時，進行去重、摘要、讓被推翻的事實失效。

這個形態和人類的運作方式吻合：你做完任務，睡一覺，長期記憶在夜裡沉澱下來。

### 原生 reasoning (Native reasoning)

Letta V1（`letta_v1_agent`，2026）棄用 `send_message`/heartbeat 和行內 `Thought:` token，改用原生 reasoning。Responses API（OpenAI）和帶有 extended thinking 的 Messages API（Anthropic）會在獨立的 channel 輸出 reasoning，並跨 turn 傳遞（正式環境中跨 provider 時會加密）。控制 loop 仍然是 ReAct。思考 trace 是結構性的，不是 prompt 形式的。

### 這個模式哪裡會出錯 (Where this pattern goes wrong)

- **Block 膨脹 (Block bloat)。** 無限的 `block_append` 很快就碰到上限。在超過上限的那次寫入之前，接上 block 摘要器。
- **無聲漂移 (Silent drift)。** Sleep-time agent 重寫了 block，主要 agent 卻毫無察覺。為 block 做版本管理，並在 trace 中顯示 diff。
- **被污染的整併 (Poisoned consolidation)。** Sleep-time agent 把攻擊者可觸及的內容處理進 core。Lesson 27 也適用於 sleep-time 這個攻擊面。

```figure
memory-blocks
```

## 動手做 (Build It)

`code/main.py` 實作：

- `Block`：id、label、value、limit、description。
- `BlockStore`：CRUD + `near_limit(label)` 輔助函式。
- 兩個腳本化的 agent：`PrimaryAgent` 處理一個 turn，`SleepTimeAgent` 在 turn 之間整併。
- 一份 trace，顯示三個 turn 的對話與 block 寫入，再加上一次 sleep-time 處理，摘要一個 block 並使一筆過時事實失效。

執行：

```
python3 code/main.py
```

Transcript 顯示這個切分：主要 turn 很快，產生原始寫入；sleep 階段負責壓縮與清理。

## 實際使用 (Use It)

- **Letta**（letta.com）：參考實作。可自架或使用託管雲端。
- **Claude Agent SDK skill**：作為 block 形式的知識，skill 是具名、有版本、可取回的指令 block，agent 按需載入。
- **自行打造**：適合想控制儲存後端的團隊。沿用 Letta API 的合約，之後才能遷移。

## 交付 (Ship It)

`outputs/skill-memory-blocks.md` 為任何 runtime 產生 Letta 形狀的 block 系統與 sleep-time hook，包含安全規則與引用接線。

## 練習 (Exercises)

1. 加一個 `block_summarize` tool，當 `near_limit` 回傳 true 時，用模型產生的摘要取代 block 的 value。哪個觸發門檻能讓摘要呼叫次數和 block 溢出次數都最少？
2. 在 archival 上實作 sleep-time 去重：文字 token 重疊超過 90% 的兩筆記錄合併成一筆。只在 sleep 階段做，絕不放在關鍵路徑上。
3. 為 block 做版本管理。每次寫入都記錄舊值與 diff。提供 `block_history(label)`，讓維運人員能 debug「為什麼 agent 忘了 X」。
4. 把 sleep-time agent 視為不可信的寫入者。當它們碰到 Persona 或 Safety block 時，要求第二個 agent 審查後才能提交。
5. 把範例改用 Letta API（`letta_v1_agent`）重寫。Block schema 有什麼改變？原生 reasoning 如何改變 trace 的形狀？

## 關鍵術語 (Key Terms)

| Term | 一般說法 | 實際意義 |
|------|----------------|------------------------|
| Memory block | 「可編輯的 prompt 區段」 | core memory 中有型別、持久、LLM 可編輯的片段 |
| Human block | 「使用者 memory」 | 關於使用者的事實，釘在 core |
| Persona block | 「Agent 身分」 | 自我概念、語氣、限制，釘在 core |
| Sleep-time compute | 「非同步 memory 工作」 | 第二個 agent 在關鍵路徑之外做整併 |
| Core / Recall / Archival | 「層級」 | 三層 memory 切分：隨時可見 / 對話 / 外部 |
| Block limit | 「上限」 | 每個 block 的字元上限；迫使摘要 |
| Native reasoning | 「Thinking channel」 | Provider 層級的 reasoning 輸出，不是 prompt 層級的 `Thought:` |
| Learned context | 「Sleep 的輸出」 | Sleep-time agent 寫進共用 block 的事實 |

## 延伸閱讀 (Further Reading)

- [Letta, Memory Blocks blog](https://www.letta.com/blog/memory-blocks)：block 模式
- [Letta, Sleep-time Compute blog](https://www.letta.com/blog/sleep-time-compute)：非同步整併
- [Letta, Rearchitecting the Agent Loop](https://www.letta.com/blog/letta-v1-agent)：原生 reasoning 重寫
- [Packer et al., MemGPT (arXiv:2310.08560)](https://arxiv.org/abs/2310.08560)：起源
