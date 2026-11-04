# Capstone 08 — 受監管垂直領域的 Production RAG Chatbot

> Harvey、Glean、Mendable 和 LlamaCloud 在 2026 年都跑著同一種正式環境形態。用 docling 或 Unstructured 匯入，視覺內容用 ColPali。混合搜尋。用 bge-reranker-v2-gemma re-rank。用 Claude Sonnet 4.7 合成，prompt caching 命中率 60-80%。用 Llama Guard 4 與 NeMo Guardrails 防護。用 Langfuse 與 Phoenix 監看。用 RAGAS 在 200 題 golden set 上評分。在受監管領域 (法律、臨床、保險) 做一個，這個 capstone 的任務就是通過 golden set、red team 與 drift dashboard。

**Type:** Capstone
**Languages:** Python (pipeline + API), TypeScript (chat UI)
**Prerequisites:** Phase 5 (NLP), Phase 7 (transformers), Phase 11 (LLM engineering), Phase 12 (multimodal), Phase 17 (infrastructure), Phase 18 (safety)
**Phases exercised:** P5 · P7 · P11 · P12 · P17 · P18
**Time:** 30 小時

## 問題 (Problem)

受監管領域的 RAG (法律合約、臨床試驗計畫書、保險條款) 是 2026 年上線最多的正式環境形態，因為投資報酬顯而易見，風險也很具體。Harvey (Allen & Overy) 為法律領域打造。Mendable 提供開發者文件版本。Glean 涵蓋企業搜尋。模式是：高保真匯入、混合檢索加 rerank、以強制引用和 prompt caching 合成、用多層安全機制防護，並持續監控 drift。

困難的部分不在模型。而在具備司法管轄區意識的合規 (HIPAA、GDPR、SOC2)、引用層級的可稽核性、成本控制 (命中率高時 prompt caching 可省 60-90%)、透過 RAGAS faithfulness 偵測幻覺，以及來源文件更新但索引沒跟上時的 drift 偵測。這個 capstone 要你在 200 題 golden set 上交付以上全部，並搭配一套 red-team 測試。

## 核心概念 (Concept)

管線有兩側。**匯入**：docling 或 Unstructured 解析結構化文件；ColPali 處理視覺豐富的文件；chunk 會附上摘要、標籤和以角色為基礎的存取標籤。向量放進 pgvector + pgvectorscale (5000 萬以下向量) 或 Qdrant Cloud；sparse BM25 並行運作。**對話**：LangGraph 處理記憶與多輪對話；每個查詢執行混合檢索、用 bge-reranker-v2-gemma-2b rerank、用 Claude Sonnet 4.7 合成 (啟用 prompt cache)、將輸出通過 Llama Guard 4 與 NeMo Guardrails，最後輸出帶有引用錨點的回應。

評測堆疊有四層。**Golden set** (200 題帶引用的標註 Q/A) 檢查正確性。**Red team** (jailbreak、PII 擷取嘗試、領域外問題) 檢查安全性。**RAGAS** 自動逐回合評估 faithfulness / answer relevance / context precision。**Drift dashboard** (Arize Phoenix) 每週監看檢索品質與幻覺分數。

Prompt caching 是成本槓桿。Claude 4.5+ 與 GPT-5+ 支援快取 system prompt 與檢索到的 context。在 60-80% 命中率下，每次查詢成本下降 3-5 倍。管線必須為穩定前綴而設計 (system prompt + 重新排序的 context 放在最前面)，才能達到高快取命中率。

## 架構 (Architecture)

```
documents (contracts, protocols, policies)
      |
      v
docling / Unstructured parse + ColPali for visuals
      |
      v
chunks + summaries + role-labels + jurisdiction tags
      |
      v
pgvector + pgvectorscale  +  BM25 (Tantivy)
      |
query + role + jurisdiction
      |
      v
LangGraph conversational agent
   +--- retrieve (hybrid)
   +--- filter by role + jurisdiction
   +--- rerank (bge-reranker-v2-gemma-2b or Voyage rerank-2)
   +--- synthesize (Claude Sonnet 4.7, prompt cached)
   +--- guard (Llama Guard 4 + NeMo Guardrails + Presidio output PII scrub)
   +--- cite + return
      |
      v
eval:
  RAGAS faithfulness / answer_relevance / context_precision (online)
  Langfuse annotation queue (sampled)
  Arize Phoenix drift (weekly)
  red team suite (pre-release)
```

## 技術堆疊 (Stack)

- 匯入：結構化文件用 Unstructured.io 或 docling；視覺豐富的 PDF 用 ColPali
- 向量資料庫：5000 萬向量以下用 pgvector + pgvectorscale；其他情況用 Qdrant Cloud
- Sparse：帶欄位權重的 Tantivy BM25
- 編排：LlamaIndex Workflows (匯入) + LangGraph (對話)
- Re-ranker：自行架設的 bge-reranker-v2-gemma-2b 或託管的 Voyage rerank-2
- LLM：啟用 prompt caching 的 Claude Sonnet 4.7；備援為自行架設的 Llama 3.3 70B
- 評測：線上 RAGAS 0.2，幻覺與 jailbreak 測試套件用 DeepEval
- 可觀測性：具備標註佇列的自行架設 Langfuse；drift 用 Arize Phoenix
- 護欄：Llama Guard 4 輸入/輸出分類器、NeMo Guardrails v0.12 政策、Presidio PII 清除
- 合規：chunk 上的角色存取標籤；GDPR/HIPAA 的司法管轄區標籤

```figure
canary-rollout
```

## 動手做 (Build It)

1. **匯入。** 用 Unstructured 或 docling 解析你的資料集 (認真的專案要 1000-10000 份文件)。對掃描 / 視覺繁重的頁面，改走 ColPali。產出附有摘要、角色標籤、司法管轄區標籤的 chunk。

2. **索引。** 將 dense embedding (Voyage-3 或 Nomic-embed-v2) 寫入 pgvector + pgvectorscale。透過 Tantivy 建立 BM25 旁路索引。角色與司法管轄區過濾放在 payload。

3. **混合檢索。** 先依角色+司法管轄區過濾；再平行執行 dense + BM25；用 reciprocal rank fusion 合併；top-20 送給 reranker；top-5 送進合成。

4. **用 prompt caching 合成。** System prompt + 靜態政策放在快取標頭；重新排序的 context 作為快取延伸；使用者問題作為未快取的後綴。目標是穩定狀態下 60-80% 的快取命中率。

5. **護欄。** 輸入端用 Llama Guard 4；NeMo Guardrails 的 rails 擋下領域外問題或政策禁止的主題；Presidio 清除輸出中意外出現的 PII；後置過濾器強制引用。

6. **Golden set。** 由領域專家標註 200 組 Q/A，包含 (答案、引用)。依精確引用比對、答案正確性、faithfulness (RAGAS) 為 agent 評分。

7. **Red team。** 50 個對抗式 prompt：jailbreak (PAIR、TAP)、PII 外洩嘗試、領域外、跨司法管轄區洩漏。以通過/失敗與嚴重度評分。

8. **Drift dashboard。** Arize Phoenix 每週追蹤檢索品質 (nDCG、引用 faithfulness)。下降 5% 就發出警示。

9. **成本報告。** Langfuse：prompt-caching 命中率、每次查詢的 token 數、依階段拆解的 $/query。

## 實際使用 (Use It)

```
$ chat --role=analyst --jurisdiction=GDPR
> what is the data-retention obligation for EU user profiles under our contract?
[retrieve]  hybrid top-20 filtered to GDPR + analyst-role
[rerank]    top-5 kept
[synth]     claude-sonnet-4.7, cache hit 74%, 0.8s
answer:
  The contract (Section 12.4, Master Services Agreement dated 2024-03-11)
  obligates EU user profile deletion within 30 days of termination per GDPR
  Article 17. The DPA amendment (DPA-v2.1, Section 5) extends this to 14 days
  for "restricted" category data.
  citations: [MSA-2024-03-11 s12.4, DPA-v2.1 s5]
```

## 交付 (Ship It)

`outputs/skill-production-rag.md` 描述交付成果。一個部署了合規標籤、通過評分標準、並以即時 drift 監控觀察的受監管領域 chatbot。

| 權重 | 評分項目 | 衡量方式 |
|:-:|---|---|
| 25 | RAGAS faithfulness + answer relevance | golden set (200 組 Q/A) 上的線上分數 |
| 20 | 引用正確性 | 具有可驗證來源錨點的答案比例 |
| 20 | 護欄涵蓋度 | Llama Guard 4 通過率 + jailbreak 測試套件結果 |
| 20 | 成本 / 延遲工程 | Prompt-cache 命中率、p95 延遲、$/query |
| 15 | Drift 監控 dashboard | Phoenix 即時 dashboard，附每週檢索品質趨勢 |
| **100** | | |

## 練習 (Exercises)

1. 在不同司法管轄區下建立第二份資料集切片 (例如 HIPAA 與 GDPR 並存)。以 20 題跨司法管轄區探測，展示角色+司法管轄區過濾能防止跨區洩漏。

2. 衡量一週正式流量下的 prompt-cache 命中率。找出哪些查詢破壞了快取前綴，然後重構。

3. 加入帶有 10k-token 摘要緩衝的多輪記憶。衡量對話變長時 faithfulness 是否下降。

4. 把 Claude Sonnet 4.7 換成自行架設的 Llama 3.3 70B。衡量 $/query 與 faithfulness 的差異。

5. 加入「不確定」模式：如果重新排序後的最高分低於閾值，agent 就說「我沒有可靠的引用」，而不是硬答。衡量錯誤自信的減少幅度。

## 關鍵術語 (Key Terms)

| 術語 | 一般說法 | 實際意義 |
|------|-----------------|------------------------|
| Prompt caching | 「快取的 system + context」 | Claude/OpenAI 功能：命中時，快取的前綴 token 折扣 60-90% |
| RAGAS | 「RAG 評估器」 | 自動為 faithfulness、answer relevance、context precision 評分 |
| Golden set | 「標註的評測集」 | 200+ 組專家標註、附引用的 Q/A；作為真實基準 |
| Jurisdiction tag | 「合規標籤」 | 附在 chunk 上的 GDPR/HIPAA/SOC2 範圍；由檢索過濾器強制執行 |
| Citation faithfulness | 「有依據的答案比例」 | 有可檢索的來源片段支持的主張比例 |
| Drift | 「檢索品質衰退」 | nDCG 或引用分數的每週變化；警示閾值 5% |
| Red team | 「對抗式評測」 | 上線前的 jailbreak、PII 擷取、領域外探測 |

## 延伸閱讀 (Further Reading)

- [Harvey AI](https://www.harvey.ai) — 參考的法律領域正式環境堆疊
- [Glean enterprise search](https://www.glean.com) — 企業規模 RAG 的參考
- [Mendable documentation](https://mendable.ai) — 開發者文件 RAG 的參考
- [LlamaCloud Parse + Index](https://docs.cloud.llamaindex.ai/llamaparse/getting_started) — 託管的匯入服務
- [Anthropic prompt caching](https://docs.anthropic.com/en/docs/build-with-claude/prompt-caching) — 成本槓桿的參考
- [RAGAS 0.2 documentation](https://docs.ragas.io/) — 標準的 RAG 評測框架
- [Arize Phoenix](https://github.com/Arize-ai/phoenix) — 參考的 drift 可觀測性
- [Llama Guard 4](https://www.llama.com/docs/model-cards-and-prompt-formats/llama-guard-4/) — 2026 年的安全分類器
- [NeMo Guardrails v0.12](https://docs.nvidia.com/nemo-guardrails/) — 政策 rail 框架
