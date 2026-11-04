# Generative Agents 與湧現式模擬 (Generative Agents and Emergent Simulation)

> Park et al. 2023 (UIST '23, arXiv:2304.03442) 建立了 **Smallville**，一個有 25 個 agent 的沙盒，採用三部分的架構：**memory stream**（自然語言 log）、**reflection**（agent 對自己的 stream 所產生的更高層次的綜合）、以及 **plan**（一天層級的行為，再往下拆成子計畫）。最具標誌性的結果是情人節派對的湧現：一個 agent 被植入「想辦一場情人節派對」，在沒有進一步寫腳本的情況下，邀請在群體中傳開、日期被協調好，派對真的舉辦了，而另外 24 個 agent 一開始完全不知道這件事。Ablation 顯示三個元件都是可信度所必需的。有紀錄的失敗是空間規範的錯誤（走進打烊的商店、共用單人廁所）。這是 2026 年 agent 模擬與 multi-agent 社會性評估的參考架構。

**Type:** Learn + Build
**Languages:** Python (stdlib)
**Prerequisites:** Phase 16 · 04 (Primitive Model), Phase 16 · 13 (Shared Memory)
**Time:** ~75 分鐘

## 問題 (Problem)

大多數 multi-agent 系統是腳本寫得很緊的團隊：planner 規劃、coder 寫程式、reviewer 審查。這對定義清楚的任務很有效。但它捕捉不到當 agent 擁有記憶、優先順序和開放世界時，會出現的湧現式、沒有腳本的行為。研究、社會模擬，以及越來越多的遊戲 AI 都需要這第二種。

Smallville 架構就是它的基準。在 Park 2023 之前，最好的 agent 模擬只是淺層的腳本遵循者；之後，這個 pattern 就成為開放世界中 generative agent 的預設。如果你在 2026 年建立 agent 模擬，你要嘛用 Smallville 的三個元件，要嘛明確說明為什麼不用。

## 核心概念 (Concept)

### 三個元件 (The three components)

**Memory stream。** 觀察、行動、reflection 和 plan 的 append-only log。每個項目有時間戳記、類型、描述（自然語言），以及衍生的 metadata：**recency**（新近度）、**importance**（重要性，由 agent 自評 1-10）和 **relevance**（相關性，與目前查詢的 cosine similarity）。

```
[2026-02-14 09:12:03] observation: Isabella Rodriguez asked me if I like jazz
[2026-02-14 09:14:22] reflection:   I enjoy long conversations about music
[2026-02-14 10:05:00] plan:         Attend Isabella's Valentine's Day party tonight
```

記憶檢索結合這三個分數：`score = w_recency * e^(-decay * age) + w_importance * importance + w_relevance * cos_sim`。前 k 個項目進入目前的 prompt。

**Reflection。** 定期（每 N 則記憶，或遇到重要事件時），agent 從最近的記憶產生更高階的綜合。Reflection 項目會回到 stream 裡，像任何其他記憶一樣可被檢索。這就是 agent 建立「理解」的方式，也就是這個架構中對長期信念的對應物。

**Plan。** 由上而下的拆解。首先是大方向的一天層級計畫（「去上班，和 Klaus 吃晚餐」）。接著是小時層級的計畫。再來是行動層級的計畫。計畫是可修訂的：當觀察與計畫矛盾時，agent 只重新規劃受影響的那一段。

### 為什麼三個都重要 (Ablation) (Why all three matter (ablation))

Park et al. 做了 ablation，分別拿掉觀察、reflection 和 plan。每個 ablation 都會損害可信度：

- 沒有 **observation**，agent 會錯過 context，依過時的信念行動。
- 沒有 **reflection**，agent 無法形成更高階的信念；互動停留在表面。
- 沒有 **plan**，行為變成反應式的雜訊；目標消散。

人類評分者給的可信度分數在三者齊備時最高；拿掉任何一個都會造成可測量的退步。

### 情人節的湧現 (The Valentine's Day emergence)

一個 agent，Isabella Rodriguez，被植入目標「想在 2 月 14 日下午 5 點在 Hobbs Cafe 辦一場情人節派對」。另外 24 個 agent 沒有收到這樣的種子。在模擬的幾天裡：

1. Isabella 的計畫包含邀請人們。
2. 每一次邀請都成為鄰居 memory stream 中的一則觀察。
3. 那位鄰居的 reflection 產生信念：「Isabella 要辦派對。」
4. 鄰居的計畫納入「2 月 14 日參加派對」。
5. 鄰居告訴其他鄰居。邀請在沒有中央協調的情況下傳開。
6. 2 月 14 日下午 5 點，好幾個 agent 聚集在 Hobbs Cafe。

這是技術意義上的湧現：系統層級的行為（一場派對），從局部的互動（雙邊邀請 + 個別規劃）中產生，沒有中央 orchestrator。

### 有紀錄的失敗模式 (The documented failure modes)

Park et al. 明確記錄了：

- **空間規範錯誤。** Agent 走進打烊的商店。Agent 試圖使用同一間單人廁所。Agent 在不該吃東西的房間吃東西。模型無法光從環境推斷出社會與物理規範。
- **記憶體溢位。** 深入的模擬執行會讓記憶檢索成本增加。實際的補救方式：定期的記憶壓縮（摘要並修剪），以及對低重要性項目做衰減。
- **Reflection 幻想。** Reflection 可能捏造 memory stream 中不存在的關係。緩解：在 reflection 的 prompt 中包含來源記憶的 id，並在檢索時驗證。

這些是與正式環境相關的失敗模式：任何 2026 年的 agent 模擬都會繼承它們。

### 三元件的實作規則 (Three-component implementation rules)

1. **記憶是 append-only。** 絕不修改記憶項目。更正是新的項目。
2. **重要性分數很便宜。** 寫入時呼叫 LLM 評 1-10 的重要性。快取分數。
3. **檢索是排名，不是過濾。** 依綜合分數取前 k 個；不要使用硬性過濾（會丟失 context）。
4. **Reflection 定期執行。** 當未處理記憶的重要性總和超過閾值（例如 150）時觸發。
5. **計畫可修訂。** 當新的觀察與計畫矛盾時，只重新產生受影響的那一段，而不是整份計畫。

### Smallville 之外的 Generative Agent (Generative agents beyond Smallville)

2024-2026 年的後續文獻擴展了這個架構：

- **用於政策／市場研究的 multi-agent 社會模擬。** 類似 Smallville 的群體模擬使用者對功能的行為反應。比 A/B 測試快；準確度仍有爭議。
- **遊戲的 NPC AI。** 使用 Smallville agent 的 RPG 產生湧現的故事線，而不是腳本化的任務。
- **Generative-agent 評估基準。** 指標不再是任務準確度，而是長時間執行下行為的可信度 + 連貫性。

這個架構是參考。擴展版本會替換元件（記憶用 vector store、檢索增強的 reflection、neurosymbolic 的 plan），但保留三部分的結構。

### 為什麼這對 multi-agent 工程很重要 (Why this matters for multi-agent engineering)

Smallville 證明了：當元件正確時，multi-agent 的湧現成本很低。這個架構現在已在開源模型上重現（較小的 LLM 可信度是平緩下降，而不是急遽下降）。任何需要 **湧現式社會行為** 的正式系統都使用這種形狀。任何需要 **緊湊任務執行** 的系統，則使用本階段前面介紹的 supervisor / roles / primitives pattern。

```figure
a5-memory-reflection
```

## 動手做 (Build It)

`code/main.py` 用標準函式庫 Python 與寫好腳本的 agent policy（沒有真正的 LLM）實作這三個元件。這個 demo 以縮小版重現情人節派對的湧現：

- `MemoryStream` — 帶有 recency/importance/relevance 檢索的 append-only log。
- `reflect(stream)` — 對最近高重要性記憶做寫好腳本的 reflection。
- `plan(agent_state)` — 依目前信念產生的一天層級與小時層級計畫。
- 情境：5 個 agent。Agent 1 一開始就有「下午 5 點辦派對」。在模擬的 tick 中，邀請傳開，agent 聚集。

執行：

```
python3 code/main.py
```

預期輸出：逐 tick 的追蹤。到最後一個 tick，5 個 agent 中至少有 3 個在計畫中有這場派對，並聚集在派對地點。單一的種子在沒有任何 orchestrator 的情況下，產生了協調一致的抵達。

## 實際使用 (Use It)

`outputs/skill-simulation-designer.md` 設計 generative-agent 模擬：agent 數量、記憶 schema、reflection 頻率、計畫視野（horizon）和評估指標。

## 交付 (Ship It)

正式環境模擬的規則：

- **記憶就是資料庫。** 規模大時選擇真正的儲存方式（vector DB、Postgres）。記憶體內的標準函式庫只用於原型。
- **記錄檢索軌跡。** 每個行動都記錄驅動它的前 k 個記憶。這是你的除錯能力。
- **為每個 agent 編列 token 預算。** 每個 agent 每個 tick 的檢索 + reflection + 計畫是 $O(k)$ 次 LLM 呼叫。N 個 agent × T 個 tick × 每個 tick 的呼叫數，可能遠遠超過你的預算。
- **定期壓縮記憶。** 對低重要性項目摘要並修剪。保留政策是設計決策，不是細節。
- **明確偵測空間／社會規範的違反。** 這個架構學不會它們。

## 練習 (Exercises)

1. 執行 `code/main.py`。確認有 3 個以上的 agent 聚集在派對。把 agent 增加到 10 個，湧現還會發生嗎？
2. 移除 reflection 步驟。行為看起來如何？對應到 Park 2023 的 ablation 發現。
3. 引入一個競爭的種子目標（「Klaus 想在下午 5 點辦一場研究演講」）。Agent 會分流，還是由其中一個目標佔上風？是什麼決定的？
4. 加入空間限制：Hobbs Cafe 最多容納 4 個 agent。模擬能優雅地處理溢出嗎，還是會碰上「單人廁所」的失敗模式？
5. 閱讀 Park et al. (arXiv:2304.03442) 第 6 節（湧現行為實驗）。找出一個你的縮小版重現不了的行為。你需要強化架構的哪個元件？

## 關鍵術語 (Key Terms)

| 術語 | 一般說法 | 實際意義 |
|------|----------------|------------------------|
| Memory stream | 「Agent 的日記」 | 觀察、行動、reflection、plan 的 append-only log。 |
| Recency | 「記憶有多新」 | 依時間以指數衰減的分數。 |
| Importance | 「Agent 有多在意」 | 寫入時自評 1-10。有快取。 |
| Relevance | 「與目前查詢多相關」 | Cosine similarity（以 embedding 為基礎）。 |
| Reflection | 「更高階的信念」 | 從最近記憶產生的綜合，重新寫回成為新記憶。 |
| Plan | 「天／小時／行動的拆解」 | 由上而下的計畫樹。觀察與之矛盾時可修訂。 |
| Smallville | 「Park 2023 的沙盒」 | 產生情人節湧現的 25 個 agent 模擬。 |
| Believability | 「品質指標」 | 人類評分者對行為看起來是否像合理 agent 的評分。 |

## 延伸閱讀 (Further Reading)

- [Park et al. — Generative Agents: Interactive Simulacra of Human Behavior](https://arxiv.org/abs/2304.03442) — 參考架構
- [UIST '23 paper page](https://dl.acm.org/doi/10.1145/3586183.3606763) — 發表場合
- [Smallville code release](https://github.com/joonspk-research/generative_agents) — 參考的 Python 實作
- [Hayes-Roth 1985 — A Blackboard Architecture for Control](https://www.sciencedirect.com/science/article/abs/pii/0004370285900639) — 結構化記憶 agent 的先前研究
