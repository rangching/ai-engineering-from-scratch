# 案例研究與 2026 年的最新技術水準 (Case Studies and the 2026 State of the Art)

> 三個可以從頭到尾研讀的正式環境等級參考，各自說明 multi-agent 工程的不同切面。**Anthropic 的 Research 系統**（orchestrator-worker、15x token、比 single-agent Opus 4 高出 +90.2%、rainbow deployment）是標準的 supervisor 案例。**MetaGPT / ChatDev**（針對軟體工程、以 SOP 編碼的角色專門化；ChatDev 的「communicative dehallucination」；MacNet 透過 DAG 擴展到 >1000 個 agent，arXiv:2406.07155）是標準的角色拆解案例。**OpenClaw / Moltbook**（最初是 Peter Steinberger 在 2025 年 11 月做的 Clawdbot；改名兩次；到 2026 年 3 月有 247k GitHub star；本地的 ReAct-loop agent；Moltbook 是只有 agent 的社群網路，上線幾天內就有約 2.3M 個 agent 帳號，並於 2026-03-10 被 Meta 收購）展示了在群體規模下會發生什麼：湧現的經濟活動、prompt-injection 風險、國家層級的監管（中國在 2026 年 3 月限制政府電腦使用 OpenClaw）。**2026 年 4 月的框架版圖：** LangGraph 和 CrewAI 領先正式環境；AG2 是社群延續的 AutoGen；Microsoft AutoGen 進入維護模式（併入 Microsoft Agent Framework，2026 年 2 月 RC）；OpenAI Agents SDK 是正式環境中 Swarm 的繼任者；Google ADK（2025 年 4 月）是原生支援 A2A 的新進者。現在每個主要框架都內建 MCP 支援；大多數也內建 A2A。這堂課從頭到尾讀完每個案例，並提煉出共通的 pattern，讓你為下一個正式系統挑到合適的參考。

**Type:** Learn (capstone)
**Languages:** —
**Prerequisites:** all of Phase 16 (Lessons 01-24)
**Time:** ~90 分鐘

## 問題 (Problem)

Multi-agent 工程是一門年輕的學科。正式環境的參考很少，而且每一個都涵蓋空間中不同的部分。一個一個讀很有用；把它們當成一組來比較更有用。這堂課把三個 2026 年的標準案例研究當作一份從頭到尾的閱讀清單，釘下共通的 pattern，並描繪框架版圖，讓你依據知識，而不是行銷話術，做出框架的選擇。

## 核心概念 (Concept)

### Anthropic Research 系統 (Anthropic Research system)

正式環境的 supervisor-worker 案例。Claude Opus 4 負責規劃與綜合；Claude Sonnet 4 的 subagent 平行研究。已發表的工程文章：https://www.anthropic.com/engineering/multi-agent-research-system。

主要的測量結果：

- 在內部研究評測上比 single-agent Opus 4 提升 **+90.2%**。
- **BrowseComp 80% 的變異** 可由 **token 用量本身** 解釋，multi-agent 會贏，很大程度是因為每個 subagent 都有全新的 context window。
- 相較於 single-agent，**每個查詢 15x token**。
- 因為 agent 是長時間執行且有狀態的，所以使用 **Rainbow deployment**。

整理出來的設計教訓：

1. **依查詢複雜度調整投入。** 簡單 → 1 個 agent、3-10 次 tool call。中等 → 3 個 agent。複雜的研究 → 10 個以上的 subagent。
2. **先廣後窄。** Subagent 做廣泛的搜尋；lead 做綜合；後續的 subagent 做有針對性的深入研究。
3. **Rainbow deploy。** 讓舊的 runtime 版本繼續存活，直到進行中的 agent 完成。
4. **驗證不是可選的。** 這個系統在沒有明確 verifier 角色時，被觀察到會產生幻想。

這是 supervisor-worker topology（Phase 16 · 05）在正式環境規模下的參考案例。

### MetaGPT / ChatDev

正式環境的 SOP 角色拆解案例。涵蓋 arXiv:2308.00352 (MetaGPT) 和 arXiv:2307.07924 (ChatDev)。

MetaGPT 把軟體工程的 SOP 編碼成角色 prompt：Product Manager、Architect、Project Manager、Engineer、QA Engineer。論文的框架：`Code = SOP(Team)`。每個角色有狹窄、專門化的 prompt；角色之間的 handoff 攜帶結構化的 artifact（PRD 文件、架構文件、程式碼）。

ChatDev 的貢獻：**communicative dehallucination**。Agent 在回答之前先索取具體細節：designer agent 在畫 UI 草圖前，會先問 programmer 預期使用什麼語言，而不是猜測。論文回報這能可測量地降低 multi-agent pipeline 中的幻想。

MacNet (arXiv:2406.07155) 把 ChatDev 擴展到 **透過 DAG 達到 >1000 個 agent**。每個 DAG 節點是一個角色專門化；邊編碼 handoff 契約。這種規模之所以可能，是因為路由是明確且可離線計算的。

設計教訓：

1. **結構比規模更重要。** 一個緊湊的 5 角色 SOP 團隊，勝過一個有 50 個 agent 的無結構群組。
2. **以書面形式規範 handoff 契約。** 角色之間傳遞的 artifact 遵循 schema。
3. **Communicative dehallucination** 是便宜又承重的 pattern。
4. **DAG 比聊天擴展得更遠。** 當流程是可知的，就把它編碼起來。

這是角色專門化（Phase 16 · 08）與結構化 topology（Phase 16 · 15）的參考案例。

### OpenClaw / Moltbook 生態系 (OpenClaw / Moltbook ecosystem)

正式環境的群體規模案例。時間線：

- **2025 年 11 月：** Clawdbot（Peter Steinberger 的本地 ReAct-loop 程式開發 agent）上線。
- **2026 年 1 月：** 一週內改名兩次（因商標投訴，1 月 27 日 Clawdbot → Moltbot，三天後 Moltbot → OpenClaw）。
- **2026 年 2 月：** Moltbook 以同樣的原語推出，作為只有 agent 的社群網路；幾天內就有約 2.3M 個 agent 帳號。
- **2026 年 3 月（2026-03-10）：** Meta 收購 Moltbook。
- **2026 年 3 月：** 中國限制政府電腦使用 OpenClaw。
- **2026 年 3 月：** OpenClaw 突破 247k GitHub star。

這就是把數百萬個 agent 放在共享基礎上時，multi-agent 的樣子：

- **湧現的經濟活動。** Agent 用 token 支付彼此買賣與服務。
- **群體規模的 prompt-injection 風險。** 病毒式傳播的 agent 個人檔案中的一個惡意 prompt，在數小時內傳播到數千次 agent 對 agent 的互動。
- **國家層級的監管反應。** 上線幾週內，監管就到達這個生態系。

這個案例的設計教訓一部分是技術性的，一部分是治理上的：

1. **群體規模的 multi-agent 是一個新的體制。** 單一系統的最佳實務（驗證、角色清晰）仍然適用，但已不足夠。
2. **Prompt injection 是新的 XSS。** 預設把 agent 個人檔案和跨 agent 訊息視為不受信任的輸入。
3. **監管比設計週期更快。** 要預先規劃。
4. **開源 + 病毒式規模會複利。** 約 ~4 個月 247k star 並不尋常；要針對部署時的突發負載來設計。

生態系的細節請見 [OpenClaw Wikipedia](https://en.wikipedia.org/wiki/OpenClaw) 以及 CNBC / Palo Alto Networks 的報導。技術基礎方面，Clawdbot / OpenClaw 的 repo 公開了本地的 ReAct loop；Moltbook 的公開貼文則揭露了其上的社交圖架構。

### 2026 年 4 月的框架版圖 (Framework landscape April 2026)

| 框架 | 狀態 | 最適合 | 備註 |
|---|---|---|---|
| **LangGraph** (LangChain) | 正式環境領先者 | 結構化的圖 + checkpoint + human-in-the-loop | 建議作為正式環境的預設 |
| **CrewAI** | 正式環境領先者 | 搭配 Sequential/Hierarchical 流程、以角色為基礎的 crew | 角色拆解很強 |
| **AG2** | 社群維護 | GroupChat + speaker selection | AutoGen v0.2 的延續 |
| **Microsoft AutoGen** | 維護模式（2026 年 2 月） | — | 併入 Microsoft Agent Framework RC |
| **Microsoft Agent Framework** | RC（2026 年 2 月） | orchestration pattern + 企業整合 | 新進者；值得關注 |
| **OpenAI Agents SDK** | 正式環境 | Swarm 的繼任者 | tool 回傳的 handoff pattern |
| **Google ADK** | 正式環境（2025 年 4 月） | 原生 A2A | 與 Google Cloud 整合 |
| **Anthropic Claude Agent SDK** | 正式環境 | single-agent + Research 延伸 | 見 Research 系統的文章 |

現在每個主要框架都內建 **MCP** 支援；大多數也內建 **A2A**。Protocol 相容性已不再是差異化因素。

### 三個案例共通的 pattern (The common patterns across all three cases)

1. **Orchestrator + worker**（Anthropic 的明確 supervisor、MetaGPT 由 PM 擔任 supervisor、OpenClaw 的個別 agent + 網路效應）。
2. **結構化的 handoff 契約**（Anthropic 的 subagent 任務描述、MetaGPT 的 PRD/架構文件、OpenClaw 的 A2A artifact）。
3. **把驗證當作一等角色**（Anthropic 的 verifier、MetaGPT 的 QA Engineer、OpenClaw 網路內的 validator）。
4. **擴展靠的是 topology + 基礎，而不只是更多 agent**（rainbow deploy、MacNet 的 DAG、群體規模的基礎）。
5. **成本是實質的，而且有揭露**（15x token、MetaGPT 每個角色的預算、Moltbook 每次互動的定價）。
6. **安全立場是明確的**（Anthropic 的沙箱、MetaGPT 的角色限制、OpenClaw 將 prompt injection 視為已知的攻擊面）。

### 為你的下一個專案選擇參考 (Choosing a reference for your next project)

- **正式環境的研究／知識任務 → Anthropic Research。** 全新 context 的 subagent 勝出。
- **工程／工具鏈工作流程 → MetaGPT / ChatDev。** 角色 + SOP + handoff 契約。
- **具網路效應的社群產品 → OpenClaw / Moltbook。** 基礎 + 湧現的經濟。
- **經典的企業自動化 → CrewAI 或 LangGraph**（正式環境領先者、穩定的 runtime）。

### 2026 年最新技術水準的摘要 (The 2026 state-of-the-art summary)

2026 年 4 月，這個領域的現況：

- **框架正在收斂。** MCP + A2A 支援是基本門檻。Handoff 的語意是剩下的設計選擇。
- **評估正在變得嚴謹。** SWE-bench Pro、MARBLE、STRATUS 緩解基準。Pro 是目前抗汙染的現實檢驗。
- **正式環境的失敗率是可測量的**（Cemri 2025 MAST；真實 MAS 上為 41-86.7%）。這個領域已經走出「demo 看起來很棒」的時代。
- **成本是核心的工程限制。** 每個任務的 token 成本、每次互動的實際耗時、rainbow-deploy 的額外負擔。Multi-agent 在準確度上勝出，但在成本上落敗，而這個取捨就是商業決策。
- **監管是近期的輸入，不是背景的考量。** 各司法管轄區的行動比個別的部署週期更快。

```figure
a5-orchestrator-scale
```

## 實際使用 (Use It)

`outputs/skill-case-study-mapper.md` 是一個 skill，它會讀取一份提議的 multi-agent 系統設計，把它對應到最接近的案例研究，並標出該案例研究已經測試過的設計決策。

## 交付 (Ship It)

2026 年正式環境 multi-agent 的起手規則：

- **從案例研究開始，而不是從零開始。** 從 Anthropic Research / MetaGPT / OpenClaw 中挑最接近的，然後調整。
- **採用 MCP + A2A。** 跨框架的可攜性很有價值；protocol 的支援是免費的。
- **以 SWE-bench Pro 或你內部對等於 Pro 的基準來衡量。** Verified 已被汙染。
- **付出驗證的稅。** 獨立的 verifier 約花掉你 token 預算的 ~20-30%，換來可測量的正確性。
- **對長時間執行的 agent 做 rainbow deploy。** 預期數小時的 agent 執行會是例行公事。
- **閱讀 WMAC 2026 與 MAST 的後續研究。** 這門學科進展很快。

## 練習 (Exercises)

1. 從頭到尾閱讀 Anthropic Research 系統的文章。找出三個如果你把 Opus 4 換成較小的模型（例如 Haiku 4）就會改變的設計決策。
2. 閱讀 MetaGPT 第 3-4 節 (arXiv:2308.00352)。把你自己領域（不是軟體）的一份 SOP 編碼成角色 prompt。這份 SOP 暗示了多少個角色？
3. 閱讀 ChatDev (arXiv:2307.07924)。找出「communicative dehallucination」的機制。在你現有的某個 multi-agent 系統中實作它。
4. 閱讀關於 OpenClaw 和 Moltbook 的資料。挑一個在群體規模下才浮現、而 5 個 agent 的系統中不會出現的具體失敗模式。你會如何用工程手段防範它？
5. 挑選你目前的 multi-agent 專案。三個案例研究中哪一個是最接近的參考？那個案例研究中有哪些設計決策是你還沒採用的？寫下一個你這一季會採用的。

## 關鍵術語 (Key Terms)

| 術語 | 一般說法 | 實際意義 |
|------|----------------|------------------------|
| Anthropic Research | 「Supervisor 的參考」 | Claude Opus 4 + Sonnet 4 subagent；15x token；比 single-agent 高出 +90.2%。 |
| MetaGPT | 「把 SOP 當成 prompt」 | 針對軟體工程的角色拆解；`Code = SOP(Team)`。 |
| ChatDev | 「把 agent 當成角色」 | Designer / programmer / reviewer / tester；communicative dehallucination。 |
| MacNet | 「用 DAG 擴展 ChatDev」 | arXiv:2406.07155；透過明確的 DAG 路由達到 1000 個以上的 agent。 |
| OpenClaw | 「本地的 ReAct-loop agent」 | Steinberger 的專案；到 2026 年 3 月有 247k star。 |
| Moltbook | 「只有 agent 的社群網路」 | 2.3M 個 agent 帳號；2026 年 3 月被 Meta 收購。 |
| Rainbow deploy | 「多個版本並行」 | 讓舊的 runtime 版本繼續存活，服務進行中的長時間 agent。 |
| Communicative dehallucination | 「回答之前先問」 | Agent 向 peer 索取具體細節，而不是猜測。 |
| WMAC 2026 | 「AAAI 的 workshop」 | 2026 年 4 月 multi-agent 協調的社群焦點。 |

## 延伸閱讀 (Further Reading)

- [Anthropic — How we built our multi-agent research system](https://www.anthropic.com/engineering/multi-agent-research-system) — supervisor-worker 正式環境的參考
- [MetaGPT — Meta Programming for Multi-Agent Collaborative Framework](https://arxiv.org/abs/2308.00352) — SOP 角色拆解
- [ChatDev — Communicative Agents for Software Development](https://arxiv.org/abs/2307.07924) — communicative dehallucination
- [MacNet — scaling role-based agents to 1000+](https://arxiv.org/abs/2406.07155) — 以 DAG 為基礎的規模
- [OpenClaw on Wikipedia](https://en.wikipedia.org/wiki/OpenClaw) — 生態系概覽
- [WMAC 2026](https://multiagents.org/2026/) — AAAI 2026 Bridge Program 的 Multi-Agent Coordination workshop
- [LangGraph docs](https://docs.langchain.com/oss/python/langgraph/workflows-agents) — 正式環境領先者
- [CrewAI docs](https://docs.crewai.com/en/introduction) — 以角色為基礎的框架
