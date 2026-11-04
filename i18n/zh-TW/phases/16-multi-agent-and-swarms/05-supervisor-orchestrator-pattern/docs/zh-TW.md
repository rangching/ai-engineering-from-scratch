# Supervisor / Orchestrator-Worker Pattern

> 一個 lead agent 負責規劃與委派；專門的 worker 在平行的 context 中執行並回報。這就是 Anthropic Research 系統背後的 pattern（Claude Opus 4 當 lead，Sonnet 4 當 subagent），在內部研究評測上比 single-agent Opus 4 高出 +90.2%。Anthropic 的工程文章指出，BrowseComp 上 80% 的變異光靠 token 用量就能解釋，multi-agent 會贏，很大程度是因為每個 subagent 都有全新的 context window。這堂課從原語開始建出 supervisor pattern，並涵蓋 2026 年正式部署累積的工程教訓。

**Type:** Learn + Build
**Languages:** Python (stdlib, `threading`)
**Prerequisites:** Phase 16 · 04 (Primitive Model)
**Time:** ~75 分鐘

## 問題 (Problem)

研究是 single-agent 系統失敗的典型任務。你問「2023 到 2026 年間 multi-agent 系統有什麼改變？」Single agent 依序讀五篇論文，context 被它們的文字塞掉一半，接著還得把它們全部放在一起推理。讀到第五篇時，它已經忘了第一篇。它也無法平行化。

Supervisor pattern 修正這個問題：一個 lead agent 規劃搜尋、把每個子問題委派給一個 worker，再做綜合。每個 worker 為一個狹窄的問題獲得自己的 200k-token window。Lead 從不看原始論文，只看 worker 的摘要。

Anthropic 的正式 Research 系統，在內部研究評測上比單一 Opus 4 高出 +90.2%。同一篇文章指出，BrowseComp 80% 的變異可由 *token 用量本身* 解釋。每個 subagent 有全新的 context，是主要機制。

## 核心概念 (Concept)

### Pattern 本身 (The pattern)

```
                 ┌──────────────┐
                 │   Lead       │  plans, decomposes,
                 │  (Opus 4)    │  synthesizes
                 └──┬────┬───┬──┘
                    │    │   │
            ┌───────┘    │   └───────┐
            ▼            ▼           ▼
      ┌─────────┐  ┌─────────┐  ┌─────────┐
      │ Worker1 │  │ Worker2 │  │ Worker3 │
      │(Sonnet) │  │(Sonnet) │  │(Sonnet) │
      └─────────┘  └─────────┘  └─────────┘
         fresh       fresh        fresh
         context     context      context
```

Lead 從不讀原始素材。在 lead 綜合之前，worker 永遠看不到彼此的工作。每個箭頭都是一次 handoff，傳遞的是狹窄的 artifact。

### 為什麼會贏 (Why it wins)

三個機制：

1. **每個 subagent 都有全新的 context。** 探索「FIPA-ACL 淵源」的 worker，不必揹著 lead 在規劃上花掉的 40k token。它為一個問題拿到 200k 的 window。
2. **透過 prompt 專門化。** Lead 的 prompt 是「拆解並綜合」，不是「研究」。每個 worker 的 prompt 都很窄：「找出 X 有什麼改變」。聚焦的 prompt 產生聚焦的輸出。
3. **平行化。** Worker 同時執行。實際耗時大約是 `max(worker_times) + plan + synthesis`，而不是 `sum(worker_times)`。

### 工程教訓（Anthropic 2025）(Engineering lessons (Anthropic 2025))

Anthropic 的文章列出幾項正式環境的教訓，到 2026 年仍然適用：

- **依查詢複雜度調整投入。** 簡單查詢：一個 agent，3-10 次 tool call。複雜查詢：10 個以上的 agent。這要由 lead 來估計，而不是呼叫者。
- **先廣後窄。** 先拆成廣泛的子問題，如果答案值得深入，再為每個子問題產生更多 worker。
- **Rainbow deployment。** Agent 是長時間執行且有狀態的。傳統的 blue-green 行不通。Anthropic 使用 rainbow：新版本逐步推出，舊版本逐步排空。
- **Token 用量佔主導。** Multi-agent 大約是 single-agent 的 ~15× token。只有任務價值足以支撐成本時才使用。

### 圖原生的轉向 (The graph-native turn)

LangGraph 最初推出了 `langgraph-supervisor` 函式庫，附帶高階的 `create_supervisor` 輔助函式。2025 年，LangChain 把建議改為直接透過 tool-calling 實作 supervisor pattern，因為 tool call 能更精細地控制 *supervisor 看到什麼*（context engineering）。這個函式庫仍然可用；文件現在則建議使用 tool-calling 的形式。

### 失敗模式 (The failure modes)

- **Lead 幻想出計畫。** 如果 lead 產生的子問題沒有拆解真正的問題，worker 就會對錯誤的目標做精準的研究。
- **Worker 過度探索。** 沒有明確的範圍邊界，worker 會偏離被指派的子問題，汙染綜合步驟。
- **綜合時的衝突。** 兩個 worker 回傳互相矛盾的事實。Lead 必須重新詢問（多加一輪），或明確註記分歧。默默選邊是最糟的失敗：使用者永遠不知道曾發生過分歧。

### 什麼時候 Supervisor 是錯的 (When supervisor is wrong)

- **循序任務。** 如果第 2 步真的需要第 1 步的輸出，平行化什麼也換不到。改用 pipeline（CrewAI Sequential、LangGraph 線性圖）。
- **簡單查詢。** Single agent 處理得更快、更便宜。產生 worker 之前，先用 lead 的「調整投入」檢查。
- **嚴格的確定性。** Supervisor 使用由 LLM 選擇的委派。當稽核／重播比適應性更重要時，靜態圖比較好。

```figure
supervisor-hierarchy
```

## 動手做 (Build It)

`code/main.py` 用 `threading` 實作一個管理三個平行 worker 的 supervisor。Lead 把查詢拆成子問題，worker 對每個子問題同時執行，再由 lead 綜合。沒有真正的 LLM，worker 是寫好腳本、用來模擬抓取與摘要的。

關鍵結構：

- `Lead.plan(query)` 把查詢拆成 3 個子問題。
- `Worker.run(sub_q)` 回傳一份假的摘要（正式環境中可以是任何會用 tool 的 agent）。
- `Lead.run(query)` 在執行緒中啟動 worker、join，然後綜合。

執行：

```
python3 code/main.py
```

輸出會顯示計畫、帶有開始／結束時間戳記的平行 worker 追蹤，以及最終的綜合結果。你可以看到實際耗時上的優勢：三個 0.3 秒的 worker 約 0.35 秒就跑完，而不是 0.9 秒。

## 實際使用 (Use It)

`outputs/skill-supervisor-designer.md` 接收使用者的查詢，產出一份 supervisor pattern 的設計：lead 的 system prompt、worker 角色、子問題拆解規則，以及綜合模板。在建立新的研究型 agent 系統之前先用它。

## 交付 (Ship It)

部署 supervisor pattern 之前的檢查清單：

- **模型搭配。** Lead 用推理等級的模型（Opus 級、`o3` 級）。Worker 用更快、更便宜的模型（Sonnet、`o4-mini`）。
- **Worker 逾時。** 任何超過中位數執行時間 2× 的 worker 會被終止；lead 要嘛用更窄的範圍重新產生，要嘛不等它就繼續。
- **每個 worker 的 token 上限。** 硬性限制（例如預期綜合輸入的 10×），防止失控的 worker 燒光預算。
- **可觀測性。** 追蹤 lead 的計畫、每個 worker 的 tool call，以及綜合過程。這是任何事後除錯的基礎。
- **Rainbow rollout。** 有狀態、長時間執行的 agent 需要逐步的版本轉換，而不是熱替換。

## 練習 (Exercises)

1. 執行 `code/main.py`，然後修改 lead，讓它產生 5 個 worker 而不是 3 個。觀察對實際耗時的影響。在這個 demo 中，worker 數量到多少時，產生的額外負擔會超過平行化省下的時間？
2. 實作 worker 逾時：終止任何執行超過 0.5 秒的 worker，並讓 lead 綜合剩餘的結果。你需要什麼可觀測性，才能知道某個 worker 被中斷了？
3. 在 lead 的綜合中加入衝突偵測步驟：如果兩個 worker 回傳互相矛盾的答案，lead 要註記分歧而不是選一邊。不呼叫 LLM 的話，你要怎麼偵測矛盾？
4. 閱讀 Anthropic 的 Research 系統工程文章。列出三項這個玩具 demo 要上線到正式環境時需要採用的做法。
5. 比較 LangGraph 的 `create_supervisor`（舊版）和新的 tool-calling 建議。哪一個對 supervisor 看到什麼提供更好的控制？為什麼 Anthropic 明確只把子答案，而不是 worker 的原始 context，傳進綜合步驟？

## 關鍵術語 (Key Terms)

| 術語 | 一般說法 | 實際意義 |
|------|----------------|------------------------|
| Supervisor | 「Lead agent」 | 負責規劃、委派和綜合的 orchestrator agent。自己不做實際的工作。 |
| Worker | 「Subagent」 | 由 supervisor 叫用的聚焦 agent，範圍狹窄，擁有自己的 context window。 |
| Orchestrator-worker | 「Supervisor pattern」 | 同一件事，不同名稱。2026 年的文獻兩種都用。 |
| Fresh context | 「乾淨的 window」 | Worker 的 context 從它的 system prompt 和被指派的問題開始，而不是 lead 的歷史。 |
| Rainbow deployment | 「逐步推出」 | 有狀態、長時間執行的 agent 需要有版本的排空與替換，而不是 blue-green。 |
| Token dominance | 「Context 才是變數」 | 依 Anthropic 的說法，研究評測 80% 的變異來自使用的 token 總量，而不是模型的選擇。 |
| Scale effort | 「讓 agent 數量與複雜度相符」 | Lead 估計查詢難度，據此產生 1 個或 10 個以上的 worker。 |
| Synthesis conflict | 「Worker 意見不一」 | 兩個 worker 回傳矛盾的事實；lead 必須揭露分歧，而不是默默選一邊。 |

## 延伸閱讀 (Further Reading)

- [Anthropic engineering — How we built our multi-agent research system](https://www.anthropic.com/engineering/multi-agent-research-system) — supervisor pattern 的正式環境參考
- [LangGraph workflows and agents](https://docs.langchain.com/oss/python/langgraph/workflows-agents) — tool-calling 的 supervisor 現在是建議的形式
- [LangGraph supervisor reference](https://reference.langchain.com/python/langgraph-supervisor) — 舊版輔助函式，2026 年仍用於正式環境
- [OpenAI cookbook — Orchestrating Agents: Routines and Handoffs](https://developers.openai.com/cookbook/examples/orchestrating_agents) — 以 handoff 為基礎的 supervisor 變體
