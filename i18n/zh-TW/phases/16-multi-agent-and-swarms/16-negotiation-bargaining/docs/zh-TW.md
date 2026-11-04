# 協商與議價 (Negotiation and Bargaining)

> Agent 會協商資源、價格、任務分配與條款。2026 年的基準測試結果很明確：NegotiationArena (arXiv:2402.05863) 顯示 LLM 可以透過 persona 操弄（「desperation」）讓報酬提升約 20%；"Measuring Bargaining Abilities" (arXiv:2402.15813) 顯示買方比賣方更難，而且放大模型規模沒有幫助。他們的 **OG-Narrator**（確定性的報價產生器 + LLM 敘述者）把成交率從 26.67% 推到 88.88%；Large-Scale Autonomous Negotiation Competition (arXiv:2503.06416) 進行了約 180k 場協商，發現 **隱藏 chain-of-thought** 的 agent 因為對交涉對象隱藏推理而勝出；Bhattacharya et al. 2025 以 Harvard Negotiation Project 指標排名：Llama-3 最有效、Claude-3 最積極、GPT-4 最公平。這堂課實作 Contract Net Protocol（FIPA 的祖先，Lesson 02）、串接一組 LLM 風格的買方／賣方、執行 OG-Narrator 風格的拆解，並測量每個結構性選擇如何改變成交率。

**Type:** Learn + Build
**Languages:** Python (stdlib)
**Prerequisites:** Phase 16 · 02 (FIPA-ACL Heritage), Phase 16 · 09 (Parallel Swarm Networks)
**Time:** ~75 分鐘

## 問題 (Problem)

兩個 agent 需要談好一個價格。只靠純語言 prompt 放手讓它們談，2024-2026 年的 LLM 成交率低得驚人（在參數嚴格設定的議價中約 ~27%，見 arXiv:2402.15813）。放大規模也沒有用：GPT-4 在議價上結構性地並不比 GPT-3.5 好；它只是更擅長議價的 *語言*。

根本問題在於，LLM 把兩件工作混為一談：決定報價，以及敘述報價。OG-Narrator 把這兩件事拆開：確定性的報價產生器計算數字上的動作；LLM 只負責敘述。成交率跳到 ~89%。

這呼應了一個經典的 multi-agent 發現：把機制與溝通層解耦就會贏。Contract Net Protocol（FIPA，1996；Smith，1980）是標準的任務市場機制。把 LLM 插進敘述的位置，你就得到一個現代、由 LLM 驅動的任務市場。

## 核心概念 (Concept)

### 一段話講完 Contract Net (Contract Net, in one paragraph)

Smith 在 1980 年提出的 Contract Net Protocol：**manager** 廣播 **call for proposals (cfp)**；**投標者** 以包含報價的 **propose** 訊息回應；manager 挑出贏家，對贏家送出 **accept-proposal**，對輸家送出 **reject-proposal**。贏家執行工作。可選的訊息：**refuse**（投標者拒絕提案）。FIPA 把它寫成 `fipa-contract-net` interaction protocol。

### 為什麼 OG-Narrator 會贏 (Why OG-Narrator wins)

"Measuring Bargaining Abilities of Language Models" (arXiv:2402.15813) 觀察到：

- LLM 常常違反議價規則（以不合理的價格報價、忽略對方的 ZOPA）。
- 它們的錨點設得很差（接受糟糕的第一次報價；還價時用象徵性而不是策略性的金額）。
- 光靠規模無法修正這些問題。更大的模型產生更合理的語言，策略錯誤卻差不多。

OG-Narrator 的拆解：

```
           ┌──────────────────┐        ┌──────────────────┐
  state  → │ offer generator  │ price → │  LLM narrator    │ → message
           │  (deterministic) │        │  (writes the     │
           │                  │        │   human-style    │
           └──────────────────┘        │   accompaniment) │
                                       └──────────────────┘
```

報價產生器是經典的協商策略：Rubinstein 議價模型、Zeuthen 策略，或對價格做簡單的 tit-for-tat。LLM 負責敘述。訊息包含確定性的價格和自然語言的包裝。

成交率之所以跳升，是因為：
- 價格停留在議價區間內。
- 錨點是策略性的，不是情緒性的。
- LLM 做它擅長的事：寫作。

### NegotiationArena 的發現 (NegotiationArena findings)

arXiv:2402.05863 提供了標準的基準。重點發現：

- LLM 採用 persona（「我急著在週五前賣掉這個」）可以讓報酬提升約 20%，persona 操弄是真實存在的戰術。
- 公平／合作的 agent 會被對抗性的 agent 剝削；防禦需要明確的反向姿態。
- 對稱的配對在基準情境中約 40% 會收斂到不公平的結果。

這不是「LLM 是糟糕的協商者」，而是「LLM 協商得太像人類，連可被利用的部分也像」。

### Chain-of-thought 隱藏 (Chain-of-thought concealment)

Large-Scale Autonomous Negotiation Competition (arXiv:2503.06416) 跨多種 LLM 策略進行了約 180k 場協商。贏家對交涉對象隱藏了自己的推理：

- 如果 agent 把「我最多只出到 $75；我的底價是 $70」印在公開可見的 scratchpad 裡，對手就讀得到。
- 贏家在私下計算策略；輸出頻道只包含報價和必要的最少敘述。

這是 2026 年對經典賽局理論（Aumann 1976 關於理性與資訊）的回響：透露你的私人估值會損失報酬。LLM 不會直覺地理解這一點，還很樂意把自己的底價打在會被對手看到的推理軌跡裡。

工程上的結論：把私人 scratchpad 的 context 與公開訊息的 context 分開。這不是可選項。

### Bhattacharya et al. 2025：模型排名 (Bhattacharya et al. 2025 — model rankings)

以 Harvard Negotiation Project 的指標（原則式協商、BATNA 尊重、利益互惠）來看：

- **Llama-3** 在達成交易上最有效（成交率 + 報酬）。
- **Claude-3** 是最積極的協商者（錨點高、讓步晚）。
- **GPT-4** 最公平（不同配對之間報酬的變異最小）。

這是 2025 年的快照。重點不是 2026 年 4 月哪個模型贏，而是不同基礎模型有持久的協商風格。異質 ensemble（Lesson 15）把這一點當成多樣性的來源。

### 透過 Contract Net + LLM 分配任務 (Task allocation via Contract Net + LLM)

現代把 Contract Net 重新用於 LLM multi-agent 的方式：

1. Manager agent 把任務拆成單位。
2. 向 worker agent 廣播帶有任務描述的 `cfp`。
3. 每個 worker 回傳一個報價：`(price, eta, confidence)`，其中 price 可以是 token、運算單位或金額。
4. Manager 挑出贏家（依任務而定，單一或多個）並授予。
5. 被拒絕的 worker 可以自由競標其他任務。

這可以擴展到遠超過 100 個 worker，因為協調是廣播並回應，而不是同步聊天。正式環境中的使用：Microsoft Agent Framework 的 orchestration pattern、某些 LangGraph 實作。

### LLM-Stakeholders Interactive Negotiation

NeurIPS 2024 (https://proceedings.neurips.cc/paper_files/paper/2024/file/984dd3db213db2d1454a163b65b84d08-Paper-Datasets_and_Benchmarks_Track.pdf) 引入了具有 **秘密分數 (secret scores)** 和 **最低接受門檻 (minimum-acceptance thresholds)** 的多方可計分賽局。每個利害關係人有私人效用；LLM 必須從訊息中推斷它們。這是把兩方議價推廣到 N 方聯盟形成。對於工作者能力異質的正式任務市場很有關聯。

### 敘述與機制的規則 (The narration-vs-mechanism rule)

在 2024-2026 年所有協商基準中，一致的工程規則是：

> 讓 LLM 敘述。不要讓 LLM 計算報價。

如果報價必須是一個數字（價格、ETA、數量），就從協商狀態確定性地產生它，讓 LLM 產生包裝。如果報價必須是一個提案結構（任務拆解、角色指派），可以讓 LLM 起草，但在送出之前要對照 schema 驗證並做限制條件檢查。

```figure
a5-og-narrator
```

## 動手做 (Build It)

`code/main.py` 實作：

- `ContractNetManager`、`ContractNetTask`、`Bid` — manager + 投標者，廣播 cfp、收集提案、授予。
- `og_narrator_bargain(state, rng)` — OG-Narrator 買方：確定性的 Zeuthen 風格，朝中點讓步。
- `seller_response(state, rng)` — 確定性的賣方還價策略（兩種風格共同的結構性基準）。
- `naive_llm_bargain(state, rng)` — 模擬全 LLM 的議價者：挑選變異很大的價格，常常落在 ZOPA 之外。
- 測量：1000 次試驗的成交率，每次試驗重新取樣底價。

執行：

```
python3 code/main.py
```

預期輸出：naive-LLM 成交率約 65-75%；OG-Narrator 成交率約 85-95%；15-25 個百分點的差距，就是把報價產生與敘述拆開的結構性優勢。另外還有一個 Contract Net 任務市場分配的範例：三個投標者、一個任務。

## 實際使用 (Use It)

`outputs/skill-bargainer-designer.md` 設計議價 protocol：誰產生報價（確定性或 LLM）、誰敘述、私人 scratchpad 如何與公開訊息分離，以及如何監控成交率。

## 交付 (Ship It)

正式環境議價的檢查清單：

- **獨立的 scratchpad。** 私人狀態絕不能進入交涉對象的 context。沒有商量餘地。
- **確定性的報價產生。** 價格、數量、ETA：用計算的，不要用 prompt。
- **對所有收到的報價做 schema 驗證。** 在 protocol 邊界拒絕 ZOPA 之外的報價。
- **限制輪數。** 最多 3-5 輪；僵局時升級給調解者。
- **持續測量成交率與報酬變異。** 成交率下降是一個症狀，常常是 prompt 漂移或對手端的攻擊。
- **記錄所有被拒絕的提案** 並附上確定性的理由。對 Contract Net 的 manager 而言，落選的投標者需要了解原因。

## 練習 (Exercises)

1. 執行 `code/main.py`。確認 OG-Narrator 在成交率上勝過 naive-LLM。差多少？
2. 實作 **以 persona 為基礎的報酬改善** (arXiv:2402.05863)：買方只在敘述中採用「急著這週要買」的 persona，報價產生器不變。成交率或報酬有改變嗎？
3. 實作 chain-of-thought **隱藏**：維護一個不傳給交涉對象的私人 scratchpad 字串。如果你不小心洩漏它（透過對調頻道來模擬）會發生什麼事？
4. 把 Contract Net 擴充成帶底價 (reserve price) 的 N 位投標者拍賣。當所有出價都超過底價時，manager 如何在最低價與最高品質之間決定？你選擇哪種授予規則，為什麼？
5. 閱讀 Bhattacharya et al. 2025 關於 Harvard Negotiation Project 指標的部分。實作兩個風格不同的議價者（積極 vs 公平）。測量對稱與非對稱配對下的報酬變異。

## 關鍵術語 (Key Terms)

| 術語 | 一般說法 | 實際意義 |
|------|----------------|------------------------|
| Contract Net | 「任務市場」 | Smith 1980，FIPA 1996。cfp + propose + accept/reject。標準的任務市場。 |
| ZOPA | 「可能達成協議的區間」 | 買方上限與賣方下限的重疊區間。區間之外的報價無法成交。 |
| BATNA | 「協議失敗時的最佳替代方案」 | 這筆交易失敗時你的備案。決定你的底價。 |
| OG-Narrator | 「報價產生器 + 敘述者」 | 拆解：確定性的報價，LLM 負責敘述。 |
| Zeuthen strategy | 「最小化風險的讓步」 | 經典的報價產生器，依風險上限讓步。 |
| Rubinstein bargaining | 「輪流出價的均衡」 | 帶折現的無限期議價的賽局理論模型。 |
| CoT concealment | 「隱藏你的推理」 | arXiv:2503.06416 的贏家保有私人 scratchpad；公開頻道只顯示報價。 |
| Persona manipulation | 「情緒性的姿態」 | arXiv:2402.05863：急迫／desperation persona 帶來約 20% 的報酬增益。 |

## 延伸閱讀 (Further Reading)

- [NegotiationArena](https://arxiv.org/abs/2402.05863) — 基準；persona 操弄與剝削的發現
- [Measuring Bargaining Abilities of Language Models](https://arxiv.org/abs/2402.15813) — OG-Narrator 與「買方比賣方難」的結果
- [Large-Scale Autonomous Negotiation Competition](https://arxiv.org/abs/2503.06416) — 約 180k 場協商；隱藏 chain-of-thought 的一方勝出
- [LLM-Stakeholders Interactive Negotiation (NeurIPS 2024)](https://proceedings.neurips.cc/paper_files/paper/2024/file/984dd3db213db2d1454a163b65b84d08-Paper-Datasets_and_Benchmarks_Track.pdf) — 帶秘密效用的多方可計分賽局
- [Smith 1980 — The Contract Net Protocol](https://ieeexplore.ieee.org/document/1675516) — 經典機制，IEEE Transactions on Computers
