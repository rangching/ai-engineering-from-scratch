# LLM 評估 (LLM Evaluation) — RAGAS、DeepEval、G-Eval

> Exact-match 和 F1 抓不到語意等價。人工審查無法擴展。LLM-as-judge 是正式環境的答案，但要有足夠的校準，才能信任那個數字。

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 5 · 13 (Question Answering), Phase 5 · 14 (Information Retrieval)
**Time:** ~75 分鐘

## 問題 (The Problem)

你的 RAG 系統回答："June 29th, 2007."
標準答案是："June 29, 2007."
Exact Match 得 0 分。F1 約 75%。人類會給 100%。

現在乘上 1 萬個測試案例。再乘上 retriever、chunking、prompt 或模型的每一次變更。你需要一個能理解語意、能便宜地大規模執行、不會在退步上說謊，而且能呈現正確失敗模式的評估器。

2026 年有三個框架掌握了這個問題。

- **RAGAS。** Retrieval-Augmented Generation ASsessment。四個 RAG 指標（faithfulness、answer-relevance、context-precision、context-recall），後端是 NLI 加 LLM-judge。有研究支持，輕量。
- **DeepEval。** 給 LLM 用的 Pytest。G-Eval、task-completion、hallucination、bias 指標。原生支援 CI/CD。
- **G-Eval。** 一種方法（也是一個 DeepEval 指標）：帶有 chain-of-thought 的 LLM-as-judge，自訂標準，0 到 1 分。

三者都依賴 LLM-as-judge。這堂課為這個方法，以及圍繞它的信任層，建立直覺。

## 核心概念 (The Concept)

![Four evaluation dimensions, LLM-as-judge architecture](../assets/llm-evaluation.svg)

**LLM-as-judge。** 用一個依評分標準 (rubric) 為輸出打分的 LLM，取代靜態指標。給定 `(query, context, answer)`，提示 judge LLM：「Score 0-1 on faithfulness.」並回傳分數。

為什麼有效：LLM 以極小的成本近似人類的判斷。以約每個評分案例 0.003 美元計，GPT-4o-mini 讓 1000 個樣本的退步評估，成本低於 5 美元。

為什麼會無聲地失敗：

1. **Judge 偏誤。** Judge 偏好較長的答案、來自它自己模型家族的答案、符合 prompt 風格的答案。
2. **JSON 解析失敗。** 壞掉的 JSON → NaN 分數 → 在彙整時被無聲排除。RAGAS 使用者都知道這種痛。要用 try/except 加明確的失敗模式來把關。
3. **隨模型版本漂移。** 升級 judge 會改變每個指標。要凍結 judge 模型加版本。

**RAG 四指標。**

| 指標 | 問題 | 後端 |
|--------|----------|---------|
| Faithfulness | 答案中的每個主張都來自檢索到的 context 嗎？ | 以 NLI 為基礎的 entailment |
| Answer relevance | 答案有回應問題嗎？ | 從答案產生假設性問題；與真實問題比較 |
| Context precision | 在檢索到的 chunk 中，有多少比例相關？ | LLM-judge |
| Context recall | 檢索有回傳所有需要的東西嗎？ | 對照標準答案的 LLM-judge |

**G-Eval。** 定義一個自訂標準："Did the answer cite the correct source?" 框架會自動把它展開成 chain-of-thought 的評估步驟，再打 0 到 1 分。適合 RAGAS 沒涵蓋的特定領域品質面向。

**校準 (Calibration)。** 在沒有對照人工標籤算出相關性之前，絕不要相信原始的 judge 分數。跑 100 個手動標註的範例。畫出 judge 對人類的圖。計算 Spearman rho。如果 rho < 0.7，你的 judge 評分標準就需要改進。

```figure
n5-judge-gauge
```

## 動手做 (Build It)

### 步驟 1：用 NLI 做 faithfulness（RAGAS 風格）

```python
from typing import Callable
from transformers import pipeline

nli = pipeline("text-classification",
               model="MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli",
               top_k=None)

# `llm` is any callable: prompt str -> generated str.
# Example: llm = lambda p: client.messages.create(model="claude-haiku-4-5", ...).content[0].text
LLM = Callable[[str], str]


def atomic_claims(answer: str, llm: LLM) -> list[str]:
    prompt = f"""Break this answer into simple factual claims (one per line):
{answer}
"""
    return llm(prompt).splitlines()


def faithfulness(answer: str, context: str, llm: LLM) -> float:
    claims = atomic_claims(answer, llm)
    if not claims:
        return 0.0
    supported = 0
    for claim in claims:
        result = nli({"text": context, "text_pair": claim})[0]
        entail = next((s for s in result if s["label"] == "entailment"), None)
        if entail and entail["score"] > 0.5:
            supported += 1
    return supported / len(claims)
```

把答案拆解成原子主張。對照檢索到的 context，用 NLI 檢查每個主張。Faithfulness = 被支持的比例。

### 步驟 2：answer relevance

```python
import numpy as np
from sentence_transformers import SentenceTransformer

# encoder: any model implementing .encode(texts, normalize_embeddings=True) -> ndarray
# e.g., encoder = SentenceTransformer("BAAI/bge-small-en-v1.5")

def answer_relevance(question: str, answer: str, encoder, llm: LLM, n: int = 3) -> float:
    prompt = f"Write {n} questions this answer could be the answer to:\n{answer}"
    generated = [line for line in llm(prompt).splitlines() if line.strip()][:n]
    if not generated:
        return 0.0
    q_emb = np.asarray(encoder.encode([question], normalize_embeddings=True)[0])
    g_embs = np.asarray(encoder.encode(generated, normalize_embeddings=True))
    sims = [float(q_emb @ g_emb) for g_emb in g_embs]
    return sum(sims) / len(sims)
```

如果答案隱含的問題與被問的問題不同，relevance 就會下降。

### 步驟 3：G-Eval 自訂指標

```python
from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCaseParams, LLMTestCase

metric = GEval(
    name="Correctness",
    criteria="The answer should be factually accurate and match the expected output.",
    evaluation_steps=[
        "Read the expected output.",
        "Read the actual output.",
        "List factual claims in the actual output.",
        "For each claim, mark supported or unsupported by the expected output.",
        "Return score = fraction supported.",
    ],
    evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT, LLMTestCaseParams.EXPECTED_OUTPUT],
)

test = LLMTestCase(input="When was the first iPhone released?",
                   actual_output="June 29th, 2007.",
                   expected_output="June 29, 2007.")
metric.measure(test)
print(metric.score, metric.reason)
```

評估步驟就是評分標準。明確的步驟比隱含的「score 0-1」prompt 更穩定。

### 步驟 4：CI 關卡

```python
import deepeval
from deepeval.metrics import FaithfulnessMetric, ContextualRelevancyMetric


def test_rag_system():
    cases = load_regression_cases()
    faith = FaithfulnessMetric(threshold=0.85)
    rel = ContextualRelevancyMetric(threshold=0.7)
    for case in cases:
        faith.measure(case)
        assert faith.score >= 0.85, f"faithfulness regression on {case.id}"
        rel.measure(case)
        assert rel.score >= 0.7, f"relevancy regression on {case.id}"
```

以 pytest 檔案的形式出貨。每個 PR 都執行。遇到退步就擋下合併。

### 步驟 5：從零實作玩具評估

見 `code/main.py`。僅用標準函式庫，近似 faithfulness（答案主張與 context 的重疊）和 relevance（答案 token 與問題 token 的重疊）。不是正式版本，只是展示形狀。

## 陷阱

- **沒有校準。** 與人工標籤的相關性只有 0.3 的 judge 就是雜訊。出貨前要求做一次校準。
- **自我評估。** 用同一個 LLM 來生成和評判，會讓分數虛高 10 到 20%。Judge 要用不同的模型家族。
- **成對評判中的位置偏誤。** Judge 偏好先呈現的選項。一定要隨機化順序，兩種順序都跑。
- **原始彙整隱藏失敗。** 平均分數 0.85 常常隱藏 5% 的災難性失敗。一定要檢查最低分位數。
- **黃金資料集腐化。** 沒有版本控制、隨時間漂移的評估集，會破壞縱向比較。每次變更都要為資料集打標籤。
- **LLM 成本。** 大規模時，judge 呼叫主導成本。使用能達到校準門檻的最便宜模型。GPT-4o-mini、Claude Haiku、Mistral-small。

## 實際使用 (Use It)

2026 年的技術棧：

| 使用情境 | 框架 |
|---------|-----------|
| RAG 品質監控 | RAGAS（4 個指標） |
| CI/CD 退步關卡 | DeepEval 加 pytest |
| 自訂領域標準 | DeepEval 內的 G-Eval |
| 線上即時流量監控 | 使用不需參考答案模式的 RAGAS |
| 有人在迴圈中的抽查 | 具標註介面的 LangSmith 或 Phoenix |
| Red-teaming / 安全評估 | Promptfoo 加 DeepEval |

典型的技術棧：RAGAS 做監控，DeepEval 做 CI，G-Eval 處理新穎的面向。三者都跑；它們的分歧有其用處。

## 交付 (Ship It)

存成 `outputs/skill-eval-architect.md`：

```markdown
---
name: eval-architect
description: Design an LLM evaluation plan with calibrated judge and CI gates.
version: 1.0.0
phase: 5
lesson: 27
tags: [nlp, evaluation, rag]
---

Given a use case (RAG / agent / generative task), output:

1. Metrics. Faithfulness / relevance / context-precision / context-recall + any custom G-Eval metrics with criteria.
2. Judge model. Named model + version, rationale for cost vs accuracy.
3. Calibration. Hand-labeled set size, target Spearman rho vs human > 0.7.
4. Dataset versioning. Tag strategy, change log, stratification.
5. CI gate. Thresholds per metric, regression-window logic, bottom-quantile alert.

Refuse to rely on a judge untested against ≥50 human-labeled examples. Refuse self-evaluation (same model generates + judges). Refuse aggregate-only reporting without bottom-10% surfacing. Flag any pipeline where judge upgrade lands without parallel baseline eval.
```

## 練習 (Exercises)

1. **簡單。** 在 10 個已知含幻覺的 RAG 範例上使用 RAGAS。驗證 faithfulness 指標能抓到每一個。
2. **中等。** 手動為 50 個 QA 答案的正確性標註 0 到 1。用 G-Eval 打分。測量 judge 與人類之間的 Spearman rho。
3. **困難。** 用 DeepEval 建立 pytest CI 關卡。刻意讓 retriever 退步，驗證關卡會失敗。透過對最低 10% 做門檻檢查，加入最低分位數的警報。

## 關鍵術語 (Key Terms)

| 術語 | 一般人怎麼說 | 實際意義 |
|------|-----------------|-----------------------|
| LLM-as-judge | 用 LLM 打分 | 提示 judge 模型依評分標準為輸出打 0 到 1 分。 |
| RAGAS | RAG 指標函式庫 | 開源的評估框架，有 4 個不需參考答案的 RAG 指標。 |
| Faithfulness | 答案有根據嗎？ | 被檢索到的 context 所蘊含的答案主張比例。 |
| Context precision | 檢索到的 chunk 相關嗎？ | 前 K 個 chunk 中，真正重要的比例。 |
| Context recall | 檢索有找到所有東西嗎？ | 標準答案主張中，被檢索到的 chunk 所支持的比例。 |
| G-Eval | 自訂 LLM judge | 評分標準加 chain-of-thought 評估步驟加 0 到 1 分。 |
| Calibration | 信任但要驗證 | judge 分數與人類分數之間的 Spearman 相關性。 |

## 延伸閱讀 (Further Reading)

- [Es et al. (2023). RAGAS: Automated Evaluation of Retrieval Augmented Generation](https://arxiv.org/abs/2309.15217) — RAGAS 論文。
- [Liu et al. (2023). G-Eval: NLG Evaluation using GPT-4 with Better Human Alignment](https://arxiv.org/abs/2303.16634) — G-Eval 論文。
- [DeepEval docs](https://deepeval.com/docs/metrics-introduction) — 開放的正式環境技術棧。
- [Zheng et al. (2023). Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena](https://arxiv.org/abs/2306.05685) — 偏誤、校準、限制。
- [MLflow GenAI Scorer](https://mlflow.org/blog/third-party-scorers) — 整合 RAGAS、DeepEval、Phoenix 的統一框架。
