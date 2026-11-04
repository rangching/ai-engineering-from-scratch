# Capstone 12 — 影片理解管線 (Video Understanding Pipeline，場景、問答、搜尋)

> Twelve Labs 把 Marengo + Pegasus 產品化。VideoDB 推出了影片版的 CRUD API。AI2 的 Molmo 2 發佈了開放的 VLM checkpoint。Gemini 的長 context 原生就能處理數小時的影片。TimeLens-100K 定義了大規模的時間定位 (temporal grounding)。2026 年的管線已經底定：場景切分、每個場景的 caption + embedding、逐字稿對齊、multi-vector 索引，以及以 (start, end) 時間戳記加畫面預覽回答的查詢。這個 capstone 要匯入 100 小時的影片、達成公開 benchmark 的成績，並衡量在計數與動作問題上的幻覺。

**Type:** Capstone
**Languages:** Python (pipeline), TypeScript (UI)
**Prerequisites:** Phase 4 (CV), Phase 6 (speech), Phase 7 (transformers), Phase 11 (LLM engineering), Phase 12 (multimodal), Phase 17 (infrastructure)
**Phases exercised:** P4 · P6 · P7 · P11 · P12 · P17
**Time:** 30 小時

## 問題 (Problem)

長影片問答是 2026 年規模下最耗頻寬的多模態問題。Gemini 2.5 Pro 能原生讀取 2 小時的影片，但要把 100 小時的影片匯入成可查詢的資料集，仍然需要場景層級的索引。正式環境的形態結合了場景切分 (TransNetV2 或 PySceneDetect)、用 VLM 對每個場景產生 caption (Gemini 2.5、Qwen3-VL-Max 或 Molmo 2)、逐字稿對齊 (帶字詞時間戳記的 Whisper-v3-turbo)，以及並排儲存 caption、畫面 embedding 與逐字稿的 multi-vector 索引。查詢管線以 (start, end) 時間戳記加畫面預覽回答。

Benchmark 是公開的 (ActivityNet-QA、NeXT-GQA)，再加上你自己的 100 題自訂集合。計數與動作類問題上的幻覺是已知最難的失敗類別；這個 capstone 明確要求衡量它。

## 核心概念 (Concept)

匯入時有三條管線並行。**場景切分** 把影片切成場景。**VLM captioning** 為每個場景產生 caption，並從關鍵影格產生畫面 embedding。**ASR 對齊** 產生字詞層級的時間戳記。三條串流以 (scene_id, 時間範圍) 合併。每個場景在 multi-vector 索引 (Qdrant) 中取得三種向量類型：caption embedding、關鍵影格 embedding、逐字稿 embedding。

查詢時，自然語言問題對三種向量同時發出；結果用 RRF 合併；時間定位轉接器 (TimeLens 風格) 在最高分場景內細化 (start, end) 視窗。VLM 合成器 (Gemini 2.5 Pro 或 Qwen3-VL-Max) 接收查詢 + 最高分場景 + 裁切後的畫面，並以附有引用時間戳記與畫面預覽的方式回答。

幻覺的衡量很重要。計數 (「有多少人走進房間？」) 與動作類 (「廚師是先倒再攪拌嗎？」) 問題出了名地不可靠。要把準確度與描述性問題分開回報。

## 架構 (Architecture)

```
video file / URL
      |
      v
PySceneDetect / TransNetV2  (scene segmentation)
      |
      +--- per-scene keyframe --- VLM caption + frame embedding
      |                            (Gemini 2.5 Pro / Qwen3-VL-Max / Molmo 2)
      |
      +--- audio channel --- Whisper-v3-turbo ASR + word timestamps
      |
      v
multi-vector Qdrant: {caption_emb, keyframe_emb, transcript_emb}
      |
query:
  dense queries against all three -> RRF merge -> top-k scenes
      |
      v
TimeLens / VideoITG temporal grounding (refine start/end within scene)
      |
      v
VLM synth: query + top scenes + frame previews
      |
      v
answer + (start, end) timestamps + frame thumbs + citations
```

## 技術堆疊 (Stack)

- 場景切分：TransNetV2 (2024-26 年最先進) 或 PySceneDetect
- ASR：透過 faster-whisper、帶字詞時間戳記的 Whisper-v3-turbo
- VLM captioner + 回答者：Gemini 2.5 Pro 或 Qwen3-VL-Max 或 Molmo 2
- 時間定位：以 TimeLens-100K 訓練的轉接器或 VideoITG
- 索引：支援 multi-vector 的 Qdrant (caption / 畫面 / 逐字稿)
- UI：Next.js 15，搭配 HTML5 影片播放器與場景縮圖
- 評測：ActivityNet-QA、NeXT-GQA、自訂的 100 題人工標註集合
- 幻覺 benchmark：計數與動作類子集，附人工標籤

```figure
cf-scene-index
```

## 動手做 (Build It)

1. **匯入 walker。** 接受 YouTube URL 或本機 MP4。必要時降到 720p。保存 `{video_id, file_path}`。

2. **場景切分。** 執行 TransNetV2 或 PySceneDetect，產生 `[{scene_id, start_ms, end_ms, keyframe_path}]`。目標 100 小時：約 6k-8k 個場景。

3. **ASR 階段。** 對音訊執行 Whisper-v3-turbo；匯出字詞層級時間戳記；切成每個場景的逐字稿片段。

4. **VLM captioning。** 對每個場景，用關鍵影格與簡短的 caption 範本呼叫 Gemini 2.5 Pro (或 Qwen3-VL-Max)。產生 caption + 畫面 embedding。

5. **Multi-vector 索引。** 具備三個具名向量的 Qdrant collection。Payload：`{video_id, scene_id, start_ms, end_ms, keyframe_url}`。

6. **查詢。** 自然語言問題發出三個 dense 查詢；用 reciprocal rank fusion 合併；top-k=5 個場景。

7. **時間定位。** 對最高分場景執行 TimeLens 風格的轉接器，在場景內細化 (start, end) 視窗。

8. **VLM 合成。** 用查詢 + top-3 場景片段 (以圖片或短片段形式) + 逐字稿呼叫 Gemini 2.5 Pro。要求 `(video_id, start_ms, end_ms)` 引用。

9. **評測。** 執行 ActivityNet-QA 與 NeXT-GQA。建立 100 題自訂集合。回報整體準確度 + 各類別拆解 (計數、動作、描述)。

## 實際使用 (Use It)

```
$ video-qa ask --url=https://youtube.com/watch?v=X "how many cars pass the intersection in the first minute?"
[scene]    23 scenes detected
[asr]      transcript complete, 4m12s
[index]    69 vectors written (23 scenes x 3)
[query]    top scene: scene 3 [01:32-01:54], confidence 0.84
[ground]   refined window: [00:12-00:58]
[synth]    gemini 2.5 pro, 1.4s
answer:    5 cars pass the intersection between 00:12 and 00:58.
citations: [scene 3: 00:12-00:58]
          [frame preview at 00:14, 00:27, 00:44, 00:51, 00:57]
```

## 交付 (Ship It)

`outputs/skill-video-qa.md` 是交付成果。給定 YouTube URL 或上傳的影片，管線會為場景建立索引，並以附有時間戳記的引用回答問題。

| 權重 | 評分項目 | 衡量方式 |
|:-:|---|---|
| 25 | 時間定位 IoU | 在 held-out 定位集合上的 intersection-over-union |
| 20 | 問答準確度 | NeXT-GQA 與自訂的 100 題 |
| 20 | 匯入吞吐量 | 每花一塊錢能處理的影片小時數 |
| 20 | UI 與引用 UX | 時間戳記連結、縮圖列、跳到畫面 |
| 15 | 幻覺率 | 計數與動作類準確度分開計算 |
| **100** | | |

## 練習 (Exercises)

1. 在 captioning 階段把 Gemini 2.5 Pro 換成 Qwen3-VL-Max。在人工評分的 50 個場景樣本上，回報 caption 品質差異。

2. 把每個場景的畫面 embedding 縮減為單一 pooled 向量，而不是 multi-vector。衡量檢索的退步。

3. 建立「嚴格計數」模式：合成器擷取每一個被計數的實例並附上時間戳記，使用者點擊即可驗證。衡量使用者驗證是否能降低幻覺。

4. 對匯入成本做 benchmark：三種 VLM 選擇下每塊錢的影片小時數。挑出甜蜜點。

5. 加入說話者分離的逐字稿：對音訊執行 pyannote 說話者分離 (diarization)，並對每位說話者的逐字稿做 embedding。展示「Alice 說了什麼關於 X 的事？」這類查詢。

## 關鍵術語 (Key Terms)

| 術語 | 一般說法 | 實際意義 |
|------|-----------------|------------------------|
| Scene segmentation | 「鏡頭偵測」 | 在鏡頭邊界把影片切成場景 |
| Multi-vector index | 「Caption + 畫面 + 逐字稿」 | 每種表示都有具名向量的 Qdrant collection |
| Temporal grounding | 「到底何時發生」 | 為查詢答案細化 (start, end) 視窗 |
| Frame embedding | 「視覺表示」 | 關鍵影格的向量 embedding；用於場景視覺相似度 |
| RRF fusion | 「Reciprocal rank fusion」 | 跨多個排序清單的合併策略；經典的混合檢索技巧 |
| Counting hallucination | 「數錯」 | VLM 在「有多少個 X」問題上的已知失敗模式 |
| ActivityNet-QA | 「影片問答 benchmark」 | 長影片問答準確度的 benchmark |

## 延伸閱讀 (Further Reading)

- [AI2 Molmo 2](https://allenai.org/blog/molmo2) — 開放的 VLM checkpoint
- [TimeLens (CVPR 2026)](https://github.com/TencentARC/TimeLens) — 大規模時間定位
- [Gemini Video long-context](https://deepmind.google/technologies/gemini) — 託管的參考
- [VideoDB](https://videodb.io) — 影片版 CRUD API 的參考
- [Twelve Labs Marengo + Pegasus](https://www.twelvelabs.io) — 商用參考
- [TransNetV2](https://github.com/soCzech/TransNetV2) — 場景切分模型
- [PySceneDetect](https://github.com/Breakthrough/PySceneDetect) — 經典的開源替代方案
- [ActivityNet-QA](https://arxiv.org/abs/1906.02467) — 參考評測 benchmark
