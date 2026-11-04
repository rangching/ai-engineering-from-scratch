# 角色專門化：Planner、Critic、Executor、Verifier (Role Specialization — Planner, Critic, Executor, Verifier)

> 2026 年最常見的 multi-agent 拆法：一個 agent 規劃，一個執行，一個批評或驗證。MetaGPT (arXiv:2308.00352) 把這件事形式化為編碼進角色 prompt 的 SOP：Product Manager、Architect、Project Manager、Engineer、QA Engineer，遵循 `Code = SOP(Team)`。ChatDev (arXiv:2307.07924) 透過「chat chain」把 designer、programmer、reviewer、tester 串起來，並搭配「communicative dehallucination」（agent 明確要求補上缺少的細節）。Verifier 是承重的角色：Cemri 等人 (MAST, arXiv:2503.13657) 顯示，每一個 multi-agent 失敗都可以追溯到缺少或壞掉的驗證。PwC 回報，在 CrewAI 中加入結構化的驗證迴圈，準確度提升 7×（10% → 70%）。

**Type:** Learn + Build
**Languages:** Python (stdlib)
**Prerequisites:** Phase 16 · 04 (Primitive Model), Phase 16 · 05 (Supervisor)
**Time:** ~60 分鐘

## 問題 (Problem)

通用的 multi-agent 系統產生通用的輸出。三個 coder 在群組聊天裡，寫出同一份平庸程式碼的三種口味。你可以加更多 agent、加更多輪，仍然跨不過品質門檻。

解法不是更多 agent，而是 *不同的* agent。指派不同的角色。給 critic 一些 planner 沒有的 tool。給 verifier 一套客觀的測試。這時系統有的是內部分歧加上有根據的修正，而不只是平行猜測。

## 核心概念 (Concept)

### 四個標準角色 (The four canonical roles)

**Planner。** 讀目標，產生步驟清單或規格。Tool：知識檢索、文件。輸出：結構化的計畫。

**Executor。** 一次讀一個計畫步驟，產生 artifact。Tool：實際工作用的 tool（程式碼編譯器、shell、API client）。輸出：artifact。

**Critic。** 對照 planner 的意圖，讀 executor 的輸出。Tool：對 artifact 的唯讀存取、靜態分析。輸出：接受／拒絕與理由。

**Verifier。** 讀 artifact 並執行確定性的檢查。Tool：測試執行器、型別檢查器、schema validator。輸出：通過／失敗與證據。

Critic 是主觀的、有主見的，通常基於 LLM。Verifier 是客觀的、確定性的，通常基於程式碼。它們不是同一個角色。

### MetaGPT 的 SOP Pattern (MetaGPT's SOP pattern)

MetaGPT (arXiv:2308.00352) 把軟體工程的 SOP 編碼成角色 prompt：

- **Product Manager** 撰寫 PRD。
- **Architect** 產出系統設計。
- **Project Manager** 拆分任務。
- **Engineer** 實作。
- **QA Engineer** 執行測試。

每個角色都有嚴格的輸入／輸出 schema。角色 prompt 說明這個角色 *是什麼*，以及它 *必須產出什麼*。`Code = SOP(Team)` 這個公式：確定性的 SOP 把一群 LLM 變成可預測的 pipeline。

### ChatDev 的 Communicative Dehallucination (ChatDev's communicative dehallucination)

ChatDev 加了一個關鍵動作：當 executor 需要計畫裡沒有的具體細節時，會在繼續之前明確地問 designer。這防止了典型的 LLM 失敗：看似合理地編造那個細節。

實作方式：角色 prompt 包含「當你需要沒有被提供的具體資訊時，在產生輸出之前，以名稱詢問相關的角色。」

### 為什麼 Verifier 最重要 (Why verifier matters most)

Cemri 等人 (MAST) 追蹤了 1642 次 multi-agent 執行失敗。其中 21.3% 是驗證缺口：系統交出了一個沒人檢查過的答案。其餘 79% 往往能追溯到「有一個檢查默默失敗，或根本沒執行」。驗證是承重的角色。

PwC 回報（CrewAI 部署，2025）加入結構化的驗證迴圈，準確度從 10% 提升到 70%。一個角色就帶來 7× 的增益。

### Critic 與 Verifier (Critic vs verifier)

- Critic 是 LLM，審查 artifact 的品質。主觀。可能被看似合理的文字騙過。
- Verifier 是對 artifact 執行的確定性程式。客觀。給出帶證據的通過／失敗。

兩者都要用。Critic 抓 verifier 說不出口的品味問題。Verifier 抓 critic 看不見、只在執行期才出現的 bug。

### 反模式 (The anti-pattern)

系統中每個角色都是 LLM，而且每個角色的輸出都是「我看起來不錯」。典型的 MAST 失敗模式。至少加一個 verifier，它的通過／失敗由程式碼決定，而不是由 LLM 決定。

### 框架對應 (Framework mappings)

- **CrewAI** — `Agent(role, goal, backstory)` 是教科書式的專門化介面。
- **LangGraph** — node 可以有專門化的 prompt；edge 強制執行 pipeline。
- **AutoGen** — 在 GroupChat 中使用角色專屬、名字只有一個字的 ConversableAgent。
- **OpenAI Agents SDK** — 在角色專門化的 Agent 之間使用 handoff tool。

```figure
swarm-roles
```

## 動手做 (Build It)

`code/main.py` 實作一條 4 角色的 pipeline，用來建出一個簡單的 Python 函式：

- **Planner** 產生規格。
- **Executor** 產生程式碼字串。
- **Critic**（LLM 模擬）標出明顯的問題。
- **Verifier** 在沙箱中（`exec`）用一個測試案例執行產生的程式碼。

Demo 執行兩次：一次 executor 產出正確的程式碼（critic 和 verifier 都通過），另一次 executor 產出不符規格的程式碼（critic 沒發現 bug，因為它看起來合理；verifier 抓到了，因為測試失敗）。

執行：

```
python3 code/main.py
```

## 實際使用 (Use It)

`outputs/skill-role-designer.md` 接收一個任務，產出角色名單（3-5 個角色）、每個角色的輸入／輸出 schema，以及 verifier 的檢查。在把 agent 接進框架之前先用它。

## 交付 (Ship It)

檢查清單：

- **至少一個確定性的 verifier。** 絕對不要全部都是 LLM。
- **每個角色有明確的 I/O schema。** Planner 回傳的是規格，不是散文；executor 讀的是那個 schema。
- **Communicative dehallucination。** 資訊缺少時，executor 必須問 planner；絕不編造。
- **Critic／verifier 的順序。** 先跑 critic（便宜，抓設計問題），再跑 verifier（慢，抓 bug）。
- **迴圈預算。** 在升級給人類處理之前，critic-executor 的修訂最多 2 輪。

## 練習 (Exercises)

1. 執行 `code/main.py`，觀察 verifier 如何抓到 critic 漏掉的 bug。再加一個靜態分析檢查（計算 `return` 出現的次數）作為額外的 verifier。它抓到了哪些執行期測試漏掉的東西？
2. 加入第 5 個角色：「requirements analyst」，把使用者的願望轉成 planner 可用的規格。有哪些 communicative dehallucination 的請求應該往上流向它？
3. 閱讀 MetaGPT 第 3 節（"Agents"）。列出 MetaGPT 5 個角色各自的輸入／輸出 schema。
4. 閱讀 ChatDev 的 chat-chain 圖 (arXiv:2307.07924 Figure 3)。找出 communicative dehallucination 在哪裡打斷了原本會無限進行的迴圈。
5. PwC 的 7× 準確度增益來自驗證迴圈。請假設三個加上 verifier 也沒有幫助的任務，也就是正確性無法以確定性方式檢查，或檢查成本高得離譜。

## 關鍵術語 (Key Terms)

| 術語 | 一般說法 | 實際意義 |
|------|----------------|------------------------|
| Role specialization | 「不同 agent，不同工作」 | 為 planner/executor/critic/verifier 角色調整過的不同 system prompt。 |
| SOP pattern | 「編碼的標準作業程序」 | MetaGPT 的框架：每個角色嚴格的 I/O schema，把團隊變成 pipeline。 |
| Communicative dehallucination | 「編造之前先問」 | ChatDev pattern：遇到缺少的細節時，executor 問 planner，而不是自己編一個。 |
| Critic | 「LLM reviewer」 | 主觀、有主見的審查者。抓品味問題。可能被看似合理的文字騙過。 |
| Verifier | 「確定性的檢查」 | 基於程式碼的通過／失敗。測試執行器、型別檢查器、schema validator。騙不了。 |
| Verification gap | 「沒人檢查」 | MAST 失敗的 21.3%。答案交出時沒有能抓到 bug 的檢查。 |
| Revision loop | 「Critic 把它退回」 | Critic 拒絕會觸發 executor 帶著回饋重新執行。需要預算。 |
| All-LLM anti-pattern | 「我看起來不錯」 | 每個角色都是 LLM，沒有確定性的檢查。典型的 MAST 失敗。 |

## 延伸閱讀 (Further Reading)

- [Hong et al. — MetaGPT: Meta Programming for Multi-Agent Collaboration](https://arxiv.org/abs/2308.00352) — SOP 作為角色 prompt 的參考論文
- [Qian et al. — Communicative Agents for Software Development (ChatDev)](https://arxiv.org/abs/2307.07924) — chat chain + communicative dehallucination
- [Cemri et al. — Why Do Multi-Agent LLM Systems Fail?](https://arxiv.org/abs/2503.13657) — MAST 分類法；驗證缺口佔失敗的 21.3%
- [CrewAI docs — Agent roles](https://docs.crewai.com/en/introduction) — 正式環境的角色規格介面
