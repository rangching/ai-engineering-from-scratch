# 失敗模式：MAST、Groupthink、Monoculture、連鎖錯誤 (Failure Modes — MAST, Groupthink, Monoculture, Cascading Errors)

> 2026 年的參考分類法是 **MAST**（Cemri et al., NeurIPS 2025, arXiv:2503.13657），由 7 個最先進開源 MAS 的 1642 條執行軌跡推導而來，顯示 **41–86.7% 的失敗率**。三大根本類別：**Specification Problems**（41.77%）：角色含糊、任務定義不清；**Coordination Failures**（36.94%）：溝通中斷、狀態不同步；**Verification Gaps**（21.30%）：缺少驗證、沒有品質檢查。**Groupthink** 家族 (arXiv:2508.05687) 另外加入：monoculture collapse（同一個基礎模型 → 相關的失敗）、從眾偏誤（agent 互相強化彼此的錯誤）、不足的 theory of mind、混合動機的動態，以及連鎖的可靠性失敗。連鎖失敗的例子：retry storm，付款失敗觸發訂單重試，進而觸發庫存重試，壓垮庫存服務（幾秒內負載 10 倍，需要 circuit breaker）。Memory poisoning：一個 agent 的幻想進入 shared memory，下游 agent 把它當成事實；準確度逐漸衰退，使根本原因的診斷很痛苦。**STRATUS**（NeurIPS 2025）回報透過專門的偵測／診斷／驗證 agent，緩解成功率提升 1.5 倍。這堂課把失敗模式當成一等的工程目標。

**Type:** Learn
**Languages:** Python (stdlib)
**Prerequisites:** Phase 16 · 13 (Shared Memory), Phase 16 · 14 (Consensus and BFT), Phase 16 · 15 (Voting and Debate Topology)
**Time:** ~75 分鐘

## 問題 (Problem)

Multi-agent 系統在真實任務上有 41-86.7% 的時間會失敗（Cemri et al. 2025 在 7 個開源 MAS 上測得）。這不是「多加幾個 agent」就能除錯的。這些失敗有結構性的原因。MAST 分類法給了你類別。這堂課把每個類別對應到具體的偵測、診斷與緩解 pattern，讓這些數字不再顯得隨意。

2026 年的正式環境實務，是把失敗模式當作設計的輸入。除非你能指出每個 MAST 類別，並說出你部署了什麼緩解措施，否則你的架構還稱不上「夠好」。

## 核心概念 (Concept)

### MAST 類別 (MAST categories)

**Specification Problems（佔失敗的 41.77%）。** Agent 的任務沒有定義得夠緊。例如：

- 角色含糊：兩個 agent 都以為自己是 reviewer。
- 任務規格不足：使用者想要特定角度，卻只說「摘要這個」。
- 成功標準是隱含的：agent 無法判斷自己是否成功。

緩解：
- 寫出明確的角色契約。每個 agent 的 prompt 說明它做什麼，*以及它不做什麼*。
- 每個任務有驗收測試。在 agent 開始之前，定義「完成長得像 X」。
- 事前的規格檢查：由另一個 agent 在派送之前審查任務定義。

**Coordination Failures（36.94%）。** 溝通或狀態的中斷。

例如：
- 兩個 agent 在沒有同步的情況下更新 shared state。
- Agent 之間的訊息遺失（queue 失敗、逾時）。
- 狀態漂移：agent A 認為任務完成了；agent B 仍在執行。

緩解：
- 搭配樂觀並行控制的有版本 shared state。
- 對關鍵訊息的明確確認（重試直到收到 ack）。
- 定期的狀態同步 checkpoint；及早偵測漂移。

**Verification Gaps（21.30%）。** 對輸出沒有獨立的檢查。

例如：
- 一個 agent 宣稱成功；沒有人驗證。
- 一串 agent 各自信任前一個的輸出。
- 對湧現出來的組合行為缺少測試覆蓋。

緩解：
- 獨立的 verifier agent（Lesson 13）。唯讀，並有獨立的來源存取權。
- 明確的 handoff 契約：「A 的輸出必須通過檢查器 C，B 才能開始。」
- 為事後分析記錄結果。

### Groupthink 家族 (arXiv:2508.05687) (Groupthink family)

當 agent 同質化或互相模仿時，有五種相關的失敗：

**Monoculture collapse。** 同一個基礎模型或訓練資料 → 相關的錯誤。當三個 agent 共用一個 LLM，它們就共用它的幻想。

**從眾偏誤 (Conformity bias)。** Agent 向最大聲或最有信心的 peer 靠攏，即使它是錯的。

**不足的 ToM。** Agent 無法為彼此的信念建模；協調瓦解（Lesson 18）。

**混合動機的動態 (Mixed-motive dynamics)。** 誘因只部分一致的 agent，會漂向折衷的中間點，結果誰都不滿意。

**連鎖的可靠性失敗 (Cascading reliability failures)。** 一個元件的錯誤模式，觸發相依元件的錯誤模式。

### 連鎖的例子：retry storm (Cascading example — the retry storm)

2026 年一種典型的事故模式：

```
payment service fails 10% of requests
   ↓
order agent retries payment (exponential backoff but naive)
   ↓
each retry is a new order-inventory check
   ↓
inventory service sees 2x normal load
   ↓
inventory service starts timing out
   ↓
every order retries inventory check
   ↓
inventory service sees 10x normal load
   ↓
cluster goes down
```

解法是經典的：**circuit breaker**。當下游錯誤率超過閾值時，就用快取或預設結果短路。再加上每個請求有上限的重試預算。

Circuit breaker 是少數可以不經修改、直接從分散式系統借用的 multi-agent 失敗緩解措施之一。

### Memory poisoning（回顧）(Memory poisoning (revisited))

來自 Lesson 13：一個 agent 的幻想變成 shared-memory 的事實；下游 agent 根據被汙染的事實推理。以 MAST 的說法，這是 shared-memory 層的驗證缺口。

症狀是準確度逐漸衰退。你不會遇到當機；你得到的是緩慢的漂移，很難找出根本原因。

緩解：append-only log、provenance、不可寫入的 verifier。Lesson 13 已經涵蓋。

### STRATUS：用於偵測失敗的專門 agent (STRATUS — specialized agents for failure detection)

STRATUS（NeurIPS 2025）回報，當你部署以下角色時，緩解成功率提升 1.5 倍：

- **偵測 agent。** 監看症狀模式（高度分歧、重試尖峰、準確度漂移）。
- **診斷 agent。** 根據症狀，從 MAST 分類法中推斷可能的根本原因。
- **驗證 agent。** 套用緩解措施之後，檢查症狀是否消除。

這是 SRE 風格的事故應變，套用到 agent 系統上。這三個角色都可以是帶專門 prompt 的 LLM agent。

### 失敗模式稽核 (The failure-mode audit)

2026 年的最佳實務是每季（以及每次重大發布）做一次失敗模式稽核：

1. **軌跡抽樣。** 收集約 ~1000 條真實的執行軌跡。
2. **分類。** 對每條軌跡中的失敗，對應到 MAST + Groupthink 類別。
3. **計算各類別的失敗率。** 哪些類別在你的系統中佔主導？
4. **為緩解措施排序。** 哪個修正能消除最多的失敗？
5. **挑 2-3 個緩解措施。** 實作；下一季重新稽核。

紀律比具體的選擇更重要。沒有稽核，失敗就會混進雜訊，永遠得不到系統性的處理。

### 系統無聲地失敗時 (When systems fail silently)

最危險的失敗類別是無聲的正確性失敗。大聲失敗的系統（當機、例外、警報）可以被監控。產生看似合理卻錯誤輸出的系統，無法靠例外 log 偵測。這就是為什麼驗證缺口雖然以數量計只有 21.30%，卻是每次失敗成本最高的類別。

投資於：
- 以抽樣為基礎的人工審查。
- Golden-dataset 的回歸測試。
- 對重要輸出做跨 agent 的交叉檢查。

### 失敗與緩慢失敗 (Failure vs slow failure)

有些失敗是立即的；有些是緩慢的。立即的失敗（逾時、schema 不符、驗證錯誤）偵測起來很便宜。緩慢的失敗（memory poisoning、monoculture 漂移、角色含糊）要偵測和預防都很昂貴。

2026 年的工程動作：為緩慢失敗裝上代理指標，讓你能在漂移變成可見的錯誤之前抓到它。一致率、重試率、輸出長度分布，以及連續 agent 版本之間的編輯距離，都是有用的代理指標。

```figure
a5-retry-cascade
```

## 動手做 (Build It)

`code/main.py` 實作：

- `FailureTaxonomy` — 把模擬的事故分類到 MAST + Groupthink 類別。
- `CircuitBreaker` — 經典 pattern；錯誤率超過閾值時打開。
- `RetryStormSimulator` — 展示連鎖失敗；可開關 circuit breaker。
- `DetectionAgent` — 寫好腳本的 STRATUS 風格症狀比對器。

執行：

```
python3 code/main.py
```

預期輸出：
- 沒有 circuit breaker 的 retry storm：庫存錯誤暴增（模擬）。
- 有 circuit breaker：限制在閾值；提供降級模式的回應。
- 偵測 agent 標出這個模式，並指名 MAST 類別。

## 實際使用 (Use It)

`outputs/skill-mast-auditor.md` 對 multi-agent 系統執行 MAST 風格的失敗模式稽核。軌跡 → 分類 → 緩解措施排序。

## 交付 (Ship It)

正式環境中的失敗模式紀律：

- **每季一次 MAST 稽核。** 不是每年。隨著系統成長，類別會改變。
- **到處都用 circuit breaker。** 對每個相依服務的每個對外呼叫。預設在 5-10% 的錯誤率打開。
- **Golden dataset。** 小而高品質、經過人工稽核。每週對它們做回歸測試。
- **STRATUS 三人組。** 偵測 + 診斷 + 驗證 agent 監控正式環境。先只從偵測 agent 開始；等症狀有雜訊時再加上診斷。
- **失敗預算。** 為各類別的失敗率設定明確的 SLO。超出預算就觸發停止出貨的討論。

## 練習 (Exercises)

1. 執行 `code/main.py`。確認 circuit breaker 限制住了 retry storm。調整失敗閾值並觀察取捨。
2. 實作 **緩慢失敗的代理指標**：3 個平行 agent 之間的一致率。當它急遽下降時觸發警報。透過逐漸讓 agent 輸出變得相關，模擬 monoculture 漂移。
3. 閱讀 Cemri et al. (arXiv:2503.13657)。挑選他們 7 個 MAS 系統之一，對應它的前 3 個失敗類別。這和 MAST 預測的相比如何？
4. 閱讀 Groupthink 論文 (arXiv:2508.05687)。找出五種 pattern 中，哪一種在正式環境中最難偵測。提出一個代理指標。
5. 為你熟悉的某個特定 multi-agent 系統，設計 STRATUS 風格的偵測-診斷-驗證三人組。偵測要監看哪些症狀？診斷建議哪些緩解措施？驗證如何確認它們有效？

## 關鍵術語 (Key Terms)

| 術語 | 一般說法 | 實際意義 |
|------|----------------|------------------------|
| MAST | 「2026 年的分類法」 | Cemri 2025；3 個根本類別 + 14 個失敗子類型。 |
| Specification Problem | 「角色含糊」 | 任務或角色定義不足；agent 不知道該做什麼。 |
| Coordination Failure | 「狀態漂移」 | Agent 之間溝通或同步的中斷。 |
| Verification Gap | 「沒人檢查」 | 輸出未經獨立驗證就被接受。 |
| Groupthink family | 「同質性失敗」 | Monoculture、從眾、不足的 ToM、混合動機、連鎖。 |
| Monoculture collapse | 「同一個模型，同樣的幻想」 | 共用基礎模型或訓練資料所造成的相關錯誤。 |
| Retry storm | 「連鎖的錯誤放大」 | 一個失敗觸發重試，進而放大下游的負載。 |
| Circuit breaker | 「依錯誤率快速失敗」 | 錯誤率超過閾值時打開；以預設值短路。 |
| STRATUS | 「事故應變三人組」 | 偵測 + 診斷 + 驗證 agent。緩解成功率 1.5 倍。 |
| Memory poisoning | 「幻想會傳播」 | Shared-memory 的事實被汙染；下游 agent 根據毒素推理。 |

## 延伸閱讀 (Further Reading)

- [Cemri et al. — Why Do Multi-Agent LLM Systems Fail?](https://arxiv.org/abs/2503.13657) — MAST 分類法，NeurIPS 2025
- [Groupthink failures in multi-agent LLMs](https://arxiv.org/abs/2508.05687) — monoculture、從眾，以及五個家族的分類法
- [STRATUS — specialized agents for MAS incident response](https://neurips.cc/) — NeurIPS 2025 論文集條目（偵測 + 診斷 + 驗證）
- [Release It! — stability patterns (Nygard)](https://pragprog.com/titles/mnee2/release-it-second-edition/) — circuit breaker 的標準參考
- [Anthropic — Multi-agent research system](https://www.anthropic.com/engineering/multi-agent-research-system) — 正式環境的失敗模式筆記
