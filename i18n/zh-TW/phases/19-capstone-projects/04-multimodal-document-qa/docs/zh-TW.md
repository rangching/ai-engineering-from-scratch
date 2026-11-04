# Capstone 04 — Multimodal Document QA (以視覺為先的 PDF、表格、圖表)

> 2026 年的文件問答前沿，已經從「先 OCR 再處理文字」轉向「以視覺為先的 late interaction」。ColPali、ColQwen2.5 和 ColQwen3-omni 把每一頁 PDF 當成圖片，用 multi-vector late interaction 做 embedding，讓查詢直接對 patch 做 attention。在財報 10-K、科學論文和手寫筆記上，這種做法大幅勝過 OCR 優先。把整條管線在 10k 頁上做完，並發佈與「OCR 再轉文字」的並排比較。

**Type:** Capstone
**Languages:** Python (pipeline), TypeScript (viewer UI)
**Prerequisites:** Phase 4 (computer vision), Phase 5 (NLP), Phase 7 (transformers), Phase 11 (LLM engineering), Phase 12 (multimodal), Phase 17 (infrastructure)
**Phases exercised:** P4 · P5 · P7 · P11 · P12 · P17
**Time:** 30 小時

## 問題 (Problem)

企業手上有一堆被 OCR 管線搞壞的 PDF：表格旋轉的掃描版 10-K、滿是公式的科學論文、只有看圖才有意義的圖表，以及手寫註記。把這些當成文字優先處理，就會失去一半的訊號。2026 年的答案，是對原始頁面圖片做 late-interaction multi-vector 檢索。ColPali (Illuin Tech) 開創了這個做法；ColQwen2.5-v0.2 和 ColQwen3-omni 進一步提升準確度。在 ViDoRe v3 上，視覺為先的檢索分數明顯高於 OCR 再轉文字，而且在圖表、表格和手寫內容上差距更大。

代價是儲存空間與延遲。一個 ColQwen embedding 每頁約有 2048 個 patch 向量，而不是單一個 1024 維向量。原始儲存量會暴增。DocPruner (2026) 可以在準確度沒有明顯損失的情況下剪掉 50%。你要為 10k 頁建索引、衡量 ViDoRe v3 nDCG@5、在 2 秒內提供答案，並直接與 OCR 再轉文字的基準線比較。

## 核心概念 (Concept)

Late interaction 的意思是：每個 query token 對每個 patch token 評分，然後把每個 query token 的最高分加總。你得到細粒度的比對，不需要單一的 pooled 向量。Multi-vector 索引 (Vespa、Qdrant multi-vector 或 AstraDB) 儲存每個 patch 的 embedding，並在檢索時執行 MaxSim。

回答者是一個 vision-language model，輸入查詢加上 top-k 檢索到的頁面圖片，寫出附有證據區域 (bounding box 或頁面參照) 的答案。Qwen3-VL-30B、Gemini 2.5 Pro 和 InternVL3 是 2026 年的前沿選擇。遇到公式與科學符號時，會插入 OCR 備援 (Nougat、dots.ocr) 作為可選的文字通道。

評測是二維矩陣。一個軸是內容類型 (純文字段落、密集表格、長條/折線圖、手寫筆記、公式)。另一個軸是檢索方式 (視覺為先的 late interaction、OCR 再轉文字、混合)。每個格子都有 nDCG@5 與答案準確度。這份報告就是交付成果。

## 架構 (Architecture)

```
PDFs -> page renderer (PyMuPDF, 180 DPI)
           |
           v
  ColQwen2.5-v0.2 embed (multi-vector per page, ~2048 patches)
           |
           +------> DocPruner 50% compression
           |
           v
   multi-vector index (Vespa or Qdrant multi-vector)
           |
query ----+----> retrieve top-k pages (MaxSim)
           |
           v
  VLM answerer: Qwen3-VL-30B | Gemini 2.5 Pro | InternVL3
    inputs: query + top-k page images + optional OCR text
           |
           v
  answer with cited page numbers + evidence regions
           |
           v
  Streamlit / Next.js viewer: highlighted boxes on source page
```

## 技術堆疊 (Stack)

- 頁面渲染：PyMuPDF (fitz)，180 DPI，統一為直式
- Late-interaction 模型：ColQwen2.5-v0.2 或 ColQwen3-omni (Hugging Face 上的 vidore 團隊)
- 索引：具備 multi-vector 欄位的 Vespa，或 Qdrant multi-vector，或支援 MaxSim 的 AstraDB
- 剪枝：DocPruner 2026 策略 (保留高變異的 patch，壓縮 50%，準確度損失 < 0.5%)
- OCR 備援 (公式 / 密集表格)：dots.ocr 或 Nougat
- VLM 回答者：自行架設的 Qwen3-VL-30B 或託管的 Gemini 2.5 Pro；InternVL3 作為備援
- 評測：ViDoRe v3 benchmark，多頁推理用 M3DocVQA
- Viewer UI：Next.js 15，用 canvas 疊加證據區域

```figure
ce-late-interaction
```

## 動手做 (Build It)

1. **匯入。** 走訪 10k 頁 PDF 的資料集，包含 10-K、科學論文與掃描文件。把每頁渲染成 1536x2048 的 PNG。保存 `{doc_id, page_num, image_path}`。

2. **Embed。** 對每張頁面圖片執行 ColQwen2.5-v0.2。輸出形狀約為 2048 個維度 128 的 patch embedding。套用 DocPruner，保留訊號最高的一半。寫入 Vespa multi-vector 欄位或 Qdrant multi-vector。

3. **查詢。** 對每個進來的查詢，用 query tower 做 embedding (token 層級的 embedding)。對索引執行 MaxSim：對每個 query token，取其與頁面 patch embedding 的最大內積，然後加總。回傳 top-k 頁面。

4. **合成。** 用查詢與 top-5 頁面圖片呼叫 Qwen3-VL-30B。Prompt：「只使用提供的頁面回答。以 (doc_id, page) 引用每個主張，並指出區域 (圖、表、段落)。」

5. **證據區域。** 對答案做後處理，擷取被引用的區域。如果 VLM 會輸出 bounding box (Qwen3-VL 會)，就在 viewer 中以疊加層呈現。

6. **OCR 備援。** 對被判定為公式密集的頁面 (用圖片變異數的啟發式)，執行 Nougat 或 dots.ocr，並把 OCR 文字當作額外通道，與圖片一起傳入。

7. **評測。** 執行 ViDoRe v3 (檢索 nDCG@5) 與 M3DocVQA (多頁 QA 準確度)。同時用相同的合成器，在同一份資料集上跑 OCR 再轉文字的管線。產出「內容類型 × 方式」矩陣。

8. **UI。** 先做 Streamlit 原型；正式版用 Next.js 15 viewer，逐頁疊加證據區域。

## 實際使用 (Use It)

```
$ doc-qa ask "what was the 2024 operating margin change for segment EMEA?"
[retrieve]   top-5 pages in 320ms (ColQwen2.5, MaxSim, Vespa)
[synth]      qwen3-vl-30b, 1.4s, cited (form-10k-2024, p. 88) + (..., p. 92)
answer:
  EMEA operating margin moved from 18.2% to 16.8%, a 140bp decline.
  cited: 10-K-2024.pdf p.88 (Table 4, Segment Operating Margin)
         10-K-2024.pdf p.92 (MD&A, Operating Performance)
[viewer]     open with highlighted bounding boxes overlaid on p.88 Table 4
```

## 交付 (Ship It)

`outputs/skill-doc-qa.md` 描述交付成果：一套針對特定資料集調校、以視覺為先的多模態文件問答系統，並在 ViDoRe v3 上與 OCR 再轉文字的基準線比較。

| 權重 | 評分項目 | 衡量方式 |
|:-:|---|---|
| 25 | ViDoRe v3 / M3DocVQA 準確度 | benchmark 數字與 OCR 文字基準線及公開排行榜比較 |
| 20 | 證據區域的依據性 | 被引用的區域中，真正包含答案片段的比例 |
| 20 | 儲存與延遲工程 | DocPruner 壓縮比、索引 p95、答案 p95 |
| 20 | 多頁推理 | 在人工標註的 100 題多頁集合上的準確度 |
| 15 | 來源檢視 UX | viewer 清晰度、疊加層保真度、並排比較工具 |
| **100** | | |

## 練習 (Exercises)

1. 在同一份資料集上比較 ColQwen2.5-v0.2 與 ColQwen3-omni。哪些頁面一個答對、另一個漏掉？在索引加上「內容類別」標籤，依類型路由。

2. 積極剪枝 embedding (75%、90%)。找出壓縮的懸崖：ViDoRe nDCG@5 掉到 OCR 基準線以下的那一點。

3. 做一個混合方案：平行執行 OCR 再轉文字與 ColQwen，用 RRF 融合，再用 cross-encoder rerank。混合方案是否勝過任何單一方式？在哪裡幫助最大？

4. 把 Qwen3-VL-30B 換成較小的 VLM (Qwen2.5-VL-7B)。衡量準確度對成本的曲線。

5. 加入手寫筆記支援。渲染手寫資料集，用 ColQwen 做 embedding，衡量檢索表現。與手寫 OCR 管線比較。

## 關鍵術語 (Key Terms)

| 術語 | 一般說法 | 實際意義 |
|------|-----------------|------------------------|
| Late interaction | 「ColPali 式檢索」 | query token 各自對頁面 patch 評分；由 MaxSim 彙整 |
| Multi-vector | 「每個 patch 一個 embedding」 | 每份文件有許多向量，而不是一個 pooled 向量 |
| MaxSim | 「Late-interaction 評分」 | 對每個 query token，取其與文件向量的最大相似度，再加總 |
| DocPruner | 「Patch 壓縮」 | 2026 年的剪枝方法，保留 50% 的 patch，準確度損失可忽略 |
| ViDoRe v3 | 「文件檢索 benchmark」 | 2026 年衡量視覺文件檢索的標準 |
| Evidence region | 「被引用的 bounding box」 | 來源頁面上標出答案片段位置的 bbox |
| OCR fallback | 「公式通道」 | 與視覺並用的文字管線，處理公式或表格密集的頁面 |

## 延伸閱讀 (Further Reading)

- [ColPali (Illuin Tech) repository](https://github.com/illuin-tech/colpali) — 參考用的 late-interaction 文件檢索
- [ColPali paper (arXiv:2407.01449)](https://arxiv.org/abs/2407.01449) — 奠基的方法論文
- [ColQwen family on Hugging Face](https://huggingface.co/vidore) — 可用於正式環境的 checkpoint
- [M3DocRAG (Adobe)](https://arxiv.org/abs/2411.04952) — 多頁多模態 RAG 基準線
- [Vespa multi-vector tutorial](https://docs.vespa.ai/en/colpali.html) — 參考的服務堆疊
- [Qdrant multi-vector support](https://qdrant.tech/documentation/concepts/vectors/#multivectors) — 替代索引
- [AstraDB multi-vector](https://docs.datastax.com/en/astra-db-serverless/databases/vector-search.html) — 另一個託管索引
- [Nougat OCR](https://github.com/facebookresearch/nougat) — 支援公式的 OCR 備援
