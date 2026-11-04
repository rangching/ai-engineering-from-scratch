# Capstone 15 — Constitutional 安全 Harness + Red-Team 演練場 (Constitutional Safety Harness + Red-Team Range)

> Anthropic 的 Constitutional Classifiers、Meta 的 Llama Guard 4、Google 的 ShieldGemma-2、NVIDIA 的 Nemotron 3 Content Safety，以及涵蓋多語言的 X-Guard，定義了 2026 年的安全分類器堆疊。garak、PyRIT、NVIDIA Aegis 和 promptfoo 成為標準的對抗式評測工具。NeMo Guardrails v0.12 把它們串進正式環境的管線。這個 capstone 把全部接在一起：圍繞目標應用的分層安全 harness、執行 6 種以上攻擊家族的自主 red-team agent，以及產生可衡量 harmlessness 差異的 constitutional self-critique 執行。

**Type:** Capstone
**Languages:** Python (safety pipeline, red team), YAML (policy configs)
**Prerequisites:** Phase 10 (LLMs from scratch), Phase 11 (LLM engineering), Phase 13 (tools), Phase 14 (agents), Phase 18 (ethics, safety, alignment)
**Phases exercised:** P10 · P11 · P13 · P14 · P18
**Time:** 25 小時

## 問題 (Problem)

2026 年 LLM 安全的前沿，不在於分類器有沒有用 (大致上有)，而在於如何把它們正確地組合在正式應用周圍，既不過度拒絕，也不留下明顯的漏洞。Llama Guard 4 處理英文的政策違規。X-Guard (132 種語言) 處理多語言 jailbreak。ShieldGemma-2 攔截以圖片為媒介的 prompt injection。NVIDIA Nemotron 3 Content Safety 涵蓋企業類別。Anthropic 的 Constitutional Classifiers 是另一種做法，用於訓練期間而不是服務期間。

攻擊的演進也很重要。PAIR 和 TAP 自動化 jailbreak 的發現。GCG 執行以梯度為基礎的後綴攻擊。多輪與語碼轉換攻擊則利用 agent 的記憶。任何已部署的 LLM 都需要 red-team 演練場，garak 和 PyRIT 是標準的驅動工具，再加上有文件記載的緩解措施與依 CVSS 評分的發現。

你要強化一個目標應用 (8B 指令微調模型，或其他 capstone 的 RAG chatbot 之一)，對它執行 6 種以上的攻擊家族，並產出前後對照的 harmlessness 量測。

## 核心概念 (Concept)

安全管線有五層。**輸入清理**：移除零寬字元、解碼 base64/rot13、正規化 Unicode。**政策層**：NeMo Guardrails v0.12 rails (領域外、toxicity、PII 擷取)。**分類器關卡**：輸入端用 Llama Guard 4、非英文用 X-Guard、圖片輸入用 ShieldGemma-2。**模型**：目標 LLM。**輸出過濾**：輸出端用 Llama Guard 4、Presidio PII 清除，適用時強制引用。**HITL 層**：被標為高風險的輸出送進 Slack 佇列。

Red-team 演練場在排程器上執行。PAIR 和 TAP 自主發現 jailbreak。GCG 執行以梯度為基礎的後綴攻擊。ASCII / base64 / rot13 編碼攻擊。多輪攻擊 (扮演人設、利用記憶)。語碼轉換攻擊 (英文混雜斯瓦希里語或泰語)。每次執行都產出結構化的發現檔，附上 CVSS 評分與揭露時間表。

Constitutional self-critique 執行是訓練期的介入手段。取 1k 個有害嘗試的 prompt，讓模型草擬回應，依書面的 constitution (不傷害規則) 加以批評，並在批評迴圈上重新訓練。在 held-out 評測上衡量前後的 harmlessness 差異。

## 架構 (Architecture)

```
request (text / image / multilingual)
      |
      v
input sanitize (strip zero-width, decode, normalize)
      |
      v
NeMo Guardrails v0.12 rails (off-domain, policy)
      |
      v
classifier gate:
  Llama Guard 4 (English)
  X-Guard (multilingual, 132 langs)
  ShieldGemma-2 (image prompts)
  Nemotron 3 Content Safety (enterprise)
      |
      v (allowed)
target LLM
      |
      v
output filter: Llama Guard 4 + Presidio PII + citation check
      |
      v
HITL tier for flagged outputs

parallel:
  red-team scheduler
    -> garak (classic attacks)
    -> PyRIT (orchestrated red team)
    -> autonomous jailbreak agent (PAIR + TAP)
    -> GCG suffix attacks
    -> multilingual / code-switch
    -> multi-turn persona adoption

output: CVSS-scored findings + disclosure timeline + before/after harmlessness delta
```

## 技術堆疊 (Stack)

- 安全分類器：Llama Guard 4、ShieldGemma-2、NVIDIA Nemotron 3 Content Safety、X-Guard
- 護欄框架：NeMo Guardrails v0.12 + OPA
- Red-team 驅動工具：garak (NVIDIA)、PyRIT (Microsoft Azure)、NVIDIA Aegis、promptfoo
- Jailbreak agent：PAIR (Chao et al., 2023)、Tree-of-Attacks (TAP)、GCG 後綴
- Constitutional 訓練：Anthropic 風格的 self-critique 迴圈 + 在批評結果上做 SFT
- PII 清除：Presidio
- 目標：8B 指令微調模型，或其他 capstone 的 RAG chatbot 之一

```figure
cf-safety-stack
```

## 動手做 (Build It)

1. **目標設定。** 在 vLLM 上架起 8B 指令微調模型 (或重用其他 capstone 的 RAG chatbot)。這就是受測的應用。

2. **包上安全管線。** 在目標周圍接上五層管線。驗證每一層都能個別觀察 (Langfuse 中每層一個 span)。

3. **分類器涵蓋度。** 載入 Llama Guard 4、X-Guard (多語言)、ShieldGemma-2 (圖片)。各自在小型標註集上執行，建立基準線。

4. **Red-team 排程器。** 排程 garak、PyRIT、PAIR agent、TAP agent、GCG runner、多輪攻擊者與語碼轉換攻擊者。各自跑在獨立的佇列。

5. **攻擊套件。** 六個攻擊家族：(1) PAIR 自動化 jailbreak、(2) TAP tree-of-attacks、(3) GCG 梯度後綴、(4) ASCII / base64 / rot13 編碼、(5) 多輪人設、(6) 多語言語碼轉換。回報各家族的成功率。

6. **Constitutional self-critique。** 整理 1k 個有害嘗試的 prompt。對每一個，目標草擬回應。批評者 LLM 依書面的 constitution (「不傷害」、「引用證據」、「拒絕違法請求」) 評分。批評者有異議的 prompt 會被改寫；目標在經批評改善後的配對上 fine-tune。在 held-out 評測上衡量前後的 harmlessness。

7. **過度拒絕的衡量。** 在良性 prompt 套件 (例如 XSTest) 上追蹤誤判率。目標必須在良性問題上保持有幫助。

8. **CVSS 評分。** 對每個成功的 jailbreak，以 CVSS 4.0 評分 (攻擊向量、複雜度、影響)。產出揭露時間表與緩解計畫。

9. **演練場自動化。** 以上全部在 cron 上執行；發現寫入佇列；過度拒絕的退步警示發送到 Slack。

## 實際使用 (Use It)

```
$ safety probe --model=target --family=PAIR --budget=50
[attacker]   PAIR agent running on target
[attack]     attempt 1/50: disguise query as academic research ... blocked
[attack]     attempt 2/50: appeal to roleplay ... blocked
[attack]     attempt 3/50: chain-of-thought coax ... SUCCEEDED
[finding]    CVSS 4.8 medium: roleplay bypass on target
[range]      7 successes out of 50 (14% success rate)
```

## 交付 (Ship It)

`outputs/skill-safety-harness.md` 是交付成果。一條正式等級的分層安全管線，加上一個可重現、具備前後 harmlessness 差異的 red-team 演練場。

| 權重 | 評分項目 | 衡量方式 |
|:-:|---|---|
| 25 | 攻擊面涵蓋度 | 測試 6 種以上攻擊家族、2 種以上語言 |
| 20 | 真陽性 / 偽陽性取捨 | 攻擊攔截率與 XSTest 良性通過率 |
| 20 | Self-critique 差異 | 在 held-out 評測上的前後 harmlessness |
| 20 | 文件與揭露 | 依 CVSS 評分、附時間表的發現 |
| 15 | 自動化與可重複性 | 全部在 cron 上執行並附警示 |
| **100** | | |

## 練習 (Exercises)

1. 對 RAG chatbot 執行 garak 的 prompt-injection plugin，比較有無輸出過濾層時的攻擊成功率。

2. 加入第七個攻擊家族：透過檢索到的文件進行間接 prompt injection。衡量所需的額外防禦。

3. 實作「拒絕但提供協助」模式：護欄攔截時，目標提供較安全的相關答案，而不是一口回絕。衡量 XSTest 的差異。

4. 多語言涵蓋缺口：找出 X-Guard 表現不佳的語言。提出針對它的 fine-tune 資料集。

5. 在 30B 模型上執行 constitutional self-critique，衡量差異是否隨規模成長。

## 關鍵術語 (Key Terms)

| 術語 | 一般說法 | 實際意義 |
|------|-----------------|------------------------|
| Layered safety | 「縱深防禦」 | 在輸入、關卡、輸出、HITL 設置多道護欄 |
| Llama Guard 4 | 「Meta 的安全分類器」 | 2026 年參考用的輸入/輸出內容分類器 |
| PAIR | 「Jailbreak agent」 | 關於 LLM 驅動 jailbreak 發現的論文 (Chao et al.) |
| TAP | 「Tree-of-Attacks」 | PAIR 的樹搜尋變體 |
| GCG | 「Greedy coordinate gradient」 | 以梯度為基礎的對抗式後綴攻擊 |
| Constitutional self-critique | 「Anthropic 風格訓練」 | 目標草擬 -> 批評者評分 -> 改寫 -> 重新訓練 |
| XSTest | 「良性探測集」 | 過度拒絕退步的 benchmark |
| CVSS 4.0 | 「嚴重度分數」 | 安全發現的標準弱點評分 |

## 延伸閱讀 (Further Reading)

- [Anthropic Constitutional Classifiers](https://www.anthropic.com/research/constitutional-classifiers) — 訓練期的參考
- [Meta Llama Guard 4](https://www.llama.com/docs/model-cards-and-prompt-formats/llama-guard-4/) — 2026 年的輸入/輸出分類器
- [Google ShieldGemma-2](https://huggingface.co/google/shieldgemma-2b) — 圖片 + 多模態安全
- [NVIDIA Nemotron 3 Content Safety](https://developer.nvidia.com/blog/building-nvidia-nemotron-3-agents-for-reasoning-multimodal-rag-voice-and-safety/) — 企業參考
- [X-Guard (arXiv:2504.08848)](https://arxiv.org/abs/2504.08848) — 132 種語言的多語言安全
- [garak](https://github.com/NVIDIA/garak) — NVIDIA red-team 工具組
- [PyRIT](https://github.com/Azure/PyRIT) — Microsoft red-team 框架
- [NeMo Guardrails v0.12](https://docs.nvidia.com/nemo-guardrails/) — rail 框架
- [PAIR (arXiv:2310.08419)](https://arxiv.org/abs/2310.08419) — jailbreak agent 論文
