# 長 Context 評估 (Long-Context Evaluation) — NIAH、RULER、LongBench、MRCR

> Gemini 3 Pro 宣稱有 1,000 萬 token 的 context。在 100 萬 token 時，8-needle MRCR 掉到 26.3%。宣稱的 ≠ 可用的。長 context 評估告訴你，你所部署的模型實際的容量是多少。

**Type:** Learn
**Languages:** Python
**Prerequisites:** Phase 5 · 13 (Question Answering), Phase 5 · 23 (Chunking Strategies)
**Time:** ~60 分鐘

## 問題 (The Problem)

你有一份 200 頁的合約。模型宣稱有 100 萬 token 的 context。你把合約貼進去問：「What is the termination clause?」模型回答了，但它是從封面頁回答的，因為終止條款位於 12 萬 token 深的地方，超出了模型實際注意得到的範圍。

這就是 2026 年的 context 容量落差。規格表寫 100 萬或 1,000 萬。現實是其中只有 60 到 70% 可用，而「可用」取決於任務。

- **檢索（大海撈針中的單一根針）：** 在前沿模型上，直到宣稱的最大值都近乎完美。
- **多跳 / 彙整：** 在大多數模型上，超過約 128k 就急遽退化。
- **對分散的事實做推理：** 第一個失敗的任務。

長 context 評估量測這些軸。這堂課會點出各個 benchmark 實際上量測什麼，以及怎麼為你的領域建立自訂的 needle 測試。

## 核心概念 (The Concept)

![NIAH baseline, RULER multi-task, LongBench holistic](../assets/long-context-eval.svg)

**Needle-in-a-Haystack（NIAH，2023）。** 把一個事實（"the magic word is pineapple"）放在長 context 中受控制的深度。要求模型把它檢索出來。掃過深度 × 長度。最初的長 context benchmark。前沿模型現在已經在這個上面飽和；它是必要但不充分的 baseline。

**RULER（Nvidia，2024）。** 4 個類別共 13 種任務類型：檢索（單一 / 多 key / 多 value）、多跳追蹤（變數追蹤）、彙整（常見字頻率）、QA。可設定 context 長度（4k 到 128k 以上）。揭露那些在 NIAH 上飽和、卻在多跳上失敗的模型。2024 年發布時，宣稱有 32k 以上 context 的 17 個模型中，只有一半在 32k 時維持住品質。

**LongBench v2（2024）。** 503 道選擇題，context 長度 8k 到 200 萬字，六個任務類別：單文件 QA、多文件 QA、長 in-context learning、長對話、程式碼 repo、長結構化資料。是真實世界長 context 行為的正式 benchmark。

**MRCR（Multi-Round Coreference Resolution）。** 大規模的多輪 coreference。有 8-needle、24-needle、100-needle 的變體。揭露模型在 attention 退化之前，能同時掌握多少事實。

**NoLiMa。** 「Non-lexical needle」。針與查詢沒有任何字面上的重疊；檢索需要一步語意推理。比 NIAH 更難。

**HELMET。** 串接許多文件，從其中任何一份提問。測試選擇性注意力。

**BABILong。** 把 bAbI 推理鏈嵌入不相關的乾草堆中。測試在乾草堆裡推理，而不只是檢索。

### 實際上該回報什麼

- **宣稱的 context window。** 規格表上的數字。
- **有效檢索長度。** NIAH 在某個門檻（例如 90%）下通過的長度。
- **有效推理長度。** 多跳或彙整在該門檻下通過的長度。
- **退化曲線。** 準確率對 context 長度，依任務類型畫出。

你的規格表要有兩個數字：檢索有效長度與推理有效長度。通常推理有效長度是宣稱 window 的 25 到 50%。

```figure
gx-niah-decay
```

## 動手做 (Build It)

### 步驟 1：為你的領域建立自訂 NIAH

見 `code/main.py`。骨架如下：

```python
def build_haystack(filler_text, needle, depth_ratio, total_tokens):
    if not (0.0 <= depth_ratio <= 1.0):
        raise ValueError(f"depth_ratio must be in [0, 1], got {depth_ratio}")
    if total_tokens <= 0:
        raise ValueError(f"total_tokens must be positive, got {total_tokens}")

    filler_tokens = tokenize(filler_text)
    needle_tokens = tokenize(needle)
    if not filler_tokens:
        raise ValueError("filler_text produced no tokens")

    # Repeat filler until long enough to fill the haystack body.
    body_len = max(total_tokens - len(needle_tokens), 0)
    while len(filler_tokens) < body_len:
        filler_tokens = filler_tokens + filler_tokens
    filler_tokens = filler_tokens[:body_len]

    insert_at = min(int(body_len * depth_ratio), body_len)
    haystack = filler_tokens[:insert_at] + needle_tokens + filler_tokens[insert_at:]
    return " ".join(haystack)


def score_niah(model, haystack, question, expected):
    answer = model.complete(f"Context: {haystack}\nQ: {question}\nA:", max_tokens=50)
    return 1 if expected.lower() in answer.lower() else 0
```

掃過 `depth_ratio` ∈ {0, 0.25, 0.5, 0.75, 1.0} × `total_tokens` ∈ {1k, 4k, 16k, 64k}。畫出熱力圖。那就是你目標模型的 NIAH 卡片。

### 步驟 2：多針變體

```python
def build_multi_needle(filler, needles, total_tokens):
    depths = [0.1, 0.4, 0.7]
    chunks = [filler[:int(total_tokens * 0.1)]]
    for depth, needle in zip(depths, needles):
        chunks.append(needle)
        next_chunk = filler[int(total_tokens * depth): int(total_tokens * (depth + 0.3))]
        chunks.append(next_chunk)
    return " ".join(chunks)
```

像「What are the three magic words?」這樣的問題，需要把三個都檢索出來。單針成功並不能預測多針成功。

### 步驟 3：多跳變數追蹤（RULER 風格）

```python
haystack = """X1 = 42. ... (filler) ... X2 = X1 + 10. ... (filler) ... X3 = X2 * 2."""
question = "What is X3?"
```

答案需要串接三個指派。前沿模型在 128k 時，這裡的準確率常常掉到 50 到 70%。

### 步驟 4：在你的技術棧上跑 LongBench v2

```python
from datasets import load_dataset
longbench = load_dataset("THUDM/LongBench-v2")

def eval_model_on_longbench(model, subset="single-doc-qa"):
    tasks = [x for x in longbench["test"] if x["task"] == subset]
    correct = 0
    for x in tasks:
        answer = model.complete(x["context"] + "\n\nQ: " + x["question"], max_tokens=20)
        if normalize(answer) == normalize(x["answer"]):
            correct += 1
    return correct / len(tasks)
```

回報各類別的準確率。彙總分數會隱藏任務層級的巨大差異。

## 陷阱

- **只做 NIAH 評估。** 在 100 萬 token 通過 NIAH，完全不能說明多跳。一定要跑 RULER 或自訂的多跳測試。
- **均勻的深度取樣。** 許多實作只測 depth=0.5。請測 depth=0、0.25、0.5、0.75、1.0，「lost in the middle」效應是真的。
- **與填充文字有字面重疊。** 如果針與填充文字共享關鍵字，檢索就變得微不足道。請使用 NoLiMa 風格、沒有重疊的針。
- **忽略延遲。** 100 萬 token 的 prompt，prefill 要 30 到 120 秒。除了準確率，也要量測 time-to-first-token。
- **廠商自行回報的數字。** OpenAI、Google、Anthropic 都發布自己的分數。一定要針對你的使用情境獨立重跑。

## 實際使用 (Use It)

2026 年的技術棧：

| 情境 | Benchmark |
|-----------|-----------|
| 快速的健全性檢查 | 3 種深度 × 3 種長度的自訂 NIAH |
| 正式環境的模型選擇 | 在你目標長度上的 RULER（13 個任務） |
| 真實世界的 QA 品質 | LongBench v2 的單文件 QA 子集 |
| 多跳推理 | BABILong 或自訂的變數追蹤 |
| 對話式 | 在你目標長度上的 MRCR 8-needle |
| 模型升級的退步檢查 | 固定的內部 NIAH 加 RULER harness，每個新模型都跑 |

正式環境的經驗法則：在你預期的長度上，沒有 NIAH 加一個推理任務之前，永遠不要相信某個 context window。

## 交付 (Ship It)

存成 `outputs/skill-long-context-eval.md`：

```markdown
---
name: long-context-eval
description: Design a long-context evaluation battery for a given model and use case.
version: 1.0.0
phase: 5
lesson: 28
tags: [nlp, long-context, evaluation]
---

Given a target model, target context length, and use case, output:

1. Tests. NIAH depth × length grid; RULER multi-hop; custom domain task.
2. Sampling. Depths 0, 0.25, 0.5, 0.75, 1.0 at each length.
3. Metrics. Retrieval pass rate; reasoning pass rate; time-to-first-token; cost-per-query.
4. Cutoff. Effective retrieval length (90% pass) and effective reasoning length (70% pass). Report both.
5. Regression. Fixed harness, rerun on every model upgrade, surface deltas.

Refuse to trust a context window from the model card alone. Refuse NIAH-only evaluation for any multi-hop workload. Refuse vendor self-reported long-context scores as independent evidence.
```

## 練習 (Exercises)

1. **簡單。** 建立 3 種深度 (0.25, 0.5, 0.75) × 3 種長度 (1k, 4k, 16k) 的 NIAH。在任何模型上執行。把通過率畫成 3×3 熱力圖。
2. **中等。** 加入 3 針變體。測量每種長度下三根針全部檢索到的比例。與同長度的單針通過率比較。
3. **困難。** 建構一個嵌入 64k 填充文字中的變數追蹤任務（X1 → X2 → X3，3 跳）。測量 3 個前沿模型的準確率。回報每個模型的有效推理長度。

## 關鍵術語 (Key Terms)

| 術語 | 一般人怎麼說 | 實際意義 |
|------|-----------------|-----------------------|
| NIAH | 大海撈針 | 在填充文字中放一個事實，要求模型把它檢索出來。 |
| RULER | 加強版 NIAH | 涵蓋檢索 / 多跳 / 彙整 / QA 的 13 種任務類型。 |
| Effective context | 真實的容量 | 準確率仍維持在門檻以上的長度。 |
| Lost in the middle | 深度偏誤 | 模型對長輸入中間的內容注意不足。 |
| Multi-needle | 同時多個事實 | 多根針；測試 attention 的同時處理能力，而不只是檢索。 |
| MRCR | 多輪 coref | 8、24 或 100 針的 coreference；揭露 attention 飽和。 |
| NoLiMa | Non-lexical needle | 針與查詢沒有共同的字面 token；需要推理。 |

## 延伸閱讀 (Further Reading)

- [Kamradt (2023). Needle in a Haystack analysis](https://github.com/gkamradt/LLMTest_NeedleInAHaystack) — 最初的 NIAH repo。
- [Hsieh et al. (2024). RULER: What's the Real Context Size of Your Long-Context LMs?](https://arxiv.org/abs/2404.06654) — 多任務 benchmark。
- [Bai et al. (2024). LongBench v2](https://arxiv.org/abs/2412.15204) — 真實世界的長 context 評估。
- [Modarressi et al. (2024). NoLiMa: Non-lexical needles](https://arxiv.org/abs/2404.06666) — 更難的針。
- [Kuratov et al. (2024). BABILong](https://arxiv.org/abs/2406.10149) — 在乾草堆中推理。
- [Liu et al. (2024). Lost in the Middle: How Language Models Use Long Contexts](https://arxiv.org/abs/2307.03172) — 深度偏誤論文。
