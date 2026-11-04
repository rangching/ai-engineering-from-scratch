# Agent 經濟、Token 誘因與信譽 (Agent Economies, Token Incentives, Reputation)

> 長視野的自主 agent（METR 的 1 小時到 8 小時工作曲線）需要經濟上的行動力。正在成形的 **5 層堆疊** 是：**DePIN**（實體運算）→ **Identity**（W3C DID + 信譽資本）→ **Cognition**（RAG + MCP）→ **Settlement**（account abstraction）→ **Governance**（Agentic DAO）。正式環境的 agent 誘因網路包括 **Bittensor**（TAO subnet 獎勵特定任務的模型）、**Fetch.ai / ASI Alliance**（ASI-1 Mini LLM + FET token），以及 **Gonka**（以 transformer 為基礎的 PoW，把運算重新分配給有生產力的 AI 任務）。學術工作方面：AAMAS 2025 的去中心化 LaMAS 使用 **Shapley-value 貢獻歸屬** 來公平獎勵有貢獻的 agent；Google Research 的 "Mechanism design for large language models" 提出在單調聚合下、採第二價格支付的 **token 拍賣**。這堂課建立一個最精簡的 agent 市場，對 multi-agent pipeline 套用 Shapley-value 貢獻歸屬，並執行第二價格的 token 拍賣，讓賽局理論的機制具體落地。

**Type:** Learn
**Languages:** Python (stdlib)
**Prerequisites:** Phase 16 · 16 (Negotiation and Bargaining), Phase 16 · 09 (Parallel Swarm Networks)
**Time:** ~75 分鐘

## 問題 (Problem)

當 agent 共同創造價值，卻需要被個別獎勵時，multi-agent 系統就會變複雜。經典的機制，例如平分、最後貢獻者全拿，不是不公平，就是容易被操弄。透過 Shapley value 的聯盟式獎勵在結構上是公平的，但計算成本高。2025-2026 年的文獻推出了有用的近似方法：Shapley sampling、單調聚合拍賣，以及由已確認的貢獻累積而成的鏈上信譽。

除了貢獻歸屬，這個領域已經轉向真正的經濟 agent：Bittensor TAO 獎勵挖礦運算來微調特定 subnet 的模型，Fetch.ai/ASI 用 FET token 獎勵 ASI-1 Mini LLM 的使用，Gonka 把 transformer 的 proof-of-work 重新導向有生產力的 AI 任務。能自主交易的 agent 今天就已經存在；問題是如何讓誘因對齊。

這堂課把 agent 經濟視為一個特定的問題家族：貢獻歸屬、機制設計與信譽，並用最少的數學把每一個建出來，讓這些想法留得住。

## 核心概念 (Concept)

### 5 層的 agent 經濟堆疊 (The 5-layer agent-economy stack)

1. **DePIN（實體運算）。** 租用 GPU、儲存、頻寬的去中心化基礎設施。Bittensor subnet、Render Network、Akash。並非專屬於 agent；agent 使用它。
2. **Identity。** W3C Decentralized Identifier (DID) 給每個 agent 一個不依賴任何平台的持久 ID。信譽累積在 DID 上。Agent Network Protocol (ANP) 使用 DID 作為 discovery 層。
3. **Cognition。** Agent 的推理迴圈：LLM + RAG + MCP。這是其他階段在建的東西。
4. **Settlement。** Account abstraction (ERC-4337) 讓 agent 用自己的餘額支付 gas，不需要持有 ETH。Agent 可以支付服務、彼此之間或運算的費用。
5. **Governance。** Agentic DAO：人類 *和* agent 都對 protocol 變更投票的治理結構，投票權與信譽綁定。

不是每個正式系統都會用到全部五層。Bittensor 用 1、2，部分的 3、部分的 4，不用 5。OpenAI 的 agent 除了 3 之外都不用。這個堆疊是一張參考地圖，不是要求。

### Bittensor、Fetch.ai、Gonka：實際運行的是什麼 (Bittensor, Fetch.ai, Gonka — what runs)

**Bittensor (TAO)。** Subnet 是專門的任務（語言建模、圖像生成、預測）。Miner 提交模型輸出。Validator 為它們排名；以質押加權的評分分配 TAO 獎勵。每個 subnet 有自己的評估。經濟上的教訓：為特定任務的輸出品質付費，而不是為用掉的運算付費。

**Fetch.ai / ASI Alliance。** ASI-1 Mini LLM 跑在 Fetch.ai 的網路上；使用者用 FET token 支付推論費用。「agent 是 peer」的敘事在這裡更強：Fetch 上的 agent 可以為了一項任務呼叫另一個 agent，並用 FET 付款。

**Gonka。** Transformer proof-of-work：「工作」是 transformer 的前向傳遞。Miner 靠執行有已知正確輸出（來自訓練資料）的推論任務賺取報酬。這是有生產力的 PoW，而不是以雜湊為基礎的 PoW。

截至 2026 年 4 月，三者都達到正式環境等級。報酬的分配方式不同。Bittensor 獎勵相對於 subnet validator 的品質；Fetch 獎勵由付費使用者衡量的效用；Gonka 獎勵可驗證的推論工作。

### Shapley-value 貢獻歸屬 (Shapley-value credit attribution)

三個 agent 合作完成一項任務。輸出得分 0.8。誰貢獻了什麼？

Shapley value：滿足四個公理（效率、對稱、線性、null）的唯一貢獻分配。對 agent `i`：

```
shapley(i) = (1/N!) * sum over all orderings O of (v(S_i_O ∪ {i}) - v(S_i_O))
```

其中 `S_i_O` 是在排序 `O` 中排在 `i` 之前的 agent 集合。實務上：列舉所有排列，記錄每個 agent 在每個排列中的邊際貢獻，再取平均。

N=3 個 agent 有 6 種排列。N=10 就有 3.6M 種，所以實務上是對排序做取樣，而不是列舉。

### 用於聚合的第二價格拍賣 (Second-price auction for aggregation)

Google Research（"Mechanism design for large language models"）提出用第二價格的 token 拍賣來聚合 LLM 的輸出。設定：N 個 agent 各自提出一個 completion；每個 agent 對於被選中有一個私人價值。拍賣者挑出價值最高的提案，並支付 *第二高* 的價值。在單調聚合下（價值取決於選了哪個提案，而不是有多少人出價），這是真實的（truthful），agent 會出價自己的真實價值。

這對 LLM 系統為什麼重要：你可以把 completion 任務外包給多個定價不同的 agent；拍賣挑出最好的並公平付費，而 agent 沒有誘因謊報。

### 信譽資本 (Reputation capital)

綁定 DID 的信譽分數由已確認的貢獻累積而來。一個簡單的更新規則：

```math
\text{rep}(i, t+1) = \alpha \cdot \text{rep}(i, t) + (1 - \alpha) \cdot \text{contribution\_quality}(i, t)
```

衰減係數 `alpha` 接近 1。信譽：

- 讀取便宜，可用於路由決策（「把困難任務送給高信譽的 agent」）。
- 偽造昂貴（隨時間累積，綁定 DID）。
- 可以被懲罰（slash）：驗證失敗的貢獻會扣分。

### AAMAS 2025 去中心化 LaMAS (AAMAS 2025 decentralized LaMAS)

LaMAS 的提案（AAMAS 2025）結合了：DID 身分、Shapley-value 貢獻歸屬，以及簡單的拍賣機制。核心主張：把貢獻歸屬這一步去中心化，讓系統可稽核，並免疫於單點操弄。

### 經濟學失效的地方 (Where the economics falls apart)

- **價格 oracle 操弄。** 如果貢獻函式可以被鑽漏洞，agent 就會去鑽。每個機制都需要對抗性測試。
- **Sybil 攻擊。** 一個營運者啟動 N 個假 agent 來膨脹自己的貢獻。DID 會減慢但無法阻止；偽造信譽的成本是緩解方式。
- **驗證成本。** 貢獻歸屬的公平程度只取決於驗證者。如果驗證便宜（小型 LLM），可能被鑽漏洞；如果昂貴（人類評審團），系統就無法擴展。
- **監管陰影。** Agent 經濟與金融監管交會。截至 2026 年，Bittensor、Fetch 和 Gonka 在某些司法管轄區都處於法律灰色地帶。

### 什麼時候 agent 經濟有意義 (When agent economies make sense)

- **有異質營運者的開放網路。** 沒有單一團隊掌控所有 agent。
- **可驗證的輸出。** 沒有驗證，貢獻歸屬只是猜測。
- **長視野的工作流程。** 一次性的任務無法從信譽累積中受益。
- **在你的司法管轄區內，token 化支付在法律上可行。**

在封閉的企業系統中，經濟學讓位給更簡單的分配方式（主管分派工作，指標是內部的）。經濟學的文獻主要適用於開放網路。

```figure
swarm-auction
```

## 動手做 (Build It)

`code/main.py` 實作：

- `shapley(value_fn, agents)` — 對小型 N 以列舉方式精確計算 Shapley。
- `second_price_auction(bids)` — 真實機制；贏家支付第二高的出價。
- `Reputation` — 綁定 DID、帶指數衰減與懲罰的信譽。
- Demo 1：三個 agent 合作，用精確的 Shapley 歸屬貢獻。
- Demo 2：五個 agent 競標一個任務名額；第二價格拍賣挑出贏家 + 付款。
- Demo 3：對信譽異質的 agent 進行 100 輪任務分配；依信譽加權的路由勝過隨機。

執行：

```
python3 code/main.py
```

預期輸出：每個 agent 的 Shapley value；顯示真實出價均衡的拍賣結果；依信譽加權的路由在暖機之後，品質比隨機高出 10-20%。

## 實際使用 (Use It)

`outputs/skill-economy-designer.md` 設計最精簡的 agent 經濟：身分層的選擇、貢獻歸屬機制、支付機制、信譽規則。

## 交付 (Ship It)

在 2026 年營運 agent 經濟：

- **從信譽開始，不要從 token 開始。** 信譽實作便宜，單獨就有價值；token 會增加法律與經濟上的複雜度。
- **先驗證，再獎勵。** 絕不在沒有獨立驗證步驟的情況下分配貢獻。自我回報的品質會招來 sybil 遊戲。
- **用 Shapley 取樣，不要用 Shapley 精確計算。** 取樣 100-1000 個排序；精確列舉無法擴展。
- **限制衰減係數，並設定信譽下限。** 無限制的衰減會抹去正當的貢獻者；衰減太慢則獎勵過時的高信譽 agent。
- **以對抗方式稽核機制。** 在開放網路之前執行紅隊情境。每個機制都有它的賽局理論；你要找到的是漏洞，而不是攻擊者。

## 練習 (Exercises)

1. 執行 `code/main.py`。確認 Shapley value 加總等於總價值（效率公理）。改變價值函式；Shapley 的分配有朝預期方向改變嗎？
2. 實作 Shapley *取樣*（對 K 個排序做 Monte Carlo）。K 如何影響近似準確度？在 N=4 時與精確值比較。
3. 在拍賣之前實作聯盟形成的步驟：agent 可以合併成團隊，並作為一個單位出價。會形成哪些聯盟？結果在 Pareto 意義上比個別出價好嗎？
4. 閱讀 Google Research 的機制設計文章。找出一個一旦被違反就會破壞真實性 (truthfulness) 的假設。這種失敗模式在 LLM 的情境中看起來像什麼？
5. 閱讀 AAMAS 2025 的去中心化 LaMAS 論文。在一個合成任務上，對 10 個 agent 實作他們的 Shapley 步驟。精確計算要花多久？用 100 次抽樣能逼近到什麼程度？

## 關鍵術語 (Key Terms)

| 術語 | 一般說法 | 實際意義 |
|------|----------------|------------------------|
| DePIN | 「去中心化實體基礎設施」 | 以 token 激勵的運算／儲存／頻寬。Bittensor、Akash、Render。 |
| DID | 「去中心化識別碼」 | 可攜式 ID 的 W3C 規格。Agent 信譽綁定在 DID 上，而不是平台。 |
| ERC-4337 | 「Account abstraction」 | 可以代付 gas 的合約帳戶，使 agent 能進行支付。 |
| Shapley value | 「公平的貢獻歸屬」 | 滿足效率、對稱、線性、null 的唯一分配。 |
| Second-price auction | 「Vickrey 拍賣」 | 真實機制：贏家支付第二高的出價。與單調聚合相容。 |
| Reputation capital | 「累積的品質分數」 | 由已確認的貢獻累積、綁定 DID 的分數；隨時間衰減。 |
| Agentic DAO | 「Agent 與人類共同治理」 | 把 agent 投票者視為一等公民、投票權與信譽綁定的 DAO。 |
| TAO / FET / GPU credits | 「Token 的面額」 | Bittensor TAO、Fetch.ai FET、各種 DePIN token。 |

## 延伸閱讀 (Further Reading)

- [The Agent Economy](https://arxiv.org/abs/2602.14219) — 2026 年對 5 層 agent 經濟堆疊的綜述
- [Google Research — Mechanism design for large language models](https://research.google/blog/mechanism-design-for-large-language-models/) — 單調聚合下的 token 拍賣
- [AAMAS 2025 — decentralized LaMAS](https://www.ifaamas.org/Proceedings/aamas2025/pdfs/p2896.pdf) — Shapley-value 貢獻歸屬
- [Bittensor TAO documentation](https://docs.bittensor.com/) — subnet 結構與獎勵分配
- [Fetch.ai / ASI Alliance](https://fetch.ai/) — ASI-1 Mini LLM 與 FET token
- [W3C Decentralized Identifiers (DIDs) spec](https://www.w3.org/TR/did-core/) — 身分的基礎
