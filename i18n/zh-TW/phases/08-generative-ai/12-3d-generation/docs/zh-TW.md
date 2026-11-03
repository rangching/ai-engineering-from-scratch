# 3D 生成 (3D Generation)

> 3D 是 2D 到 3D 的槓桿效果最強的模態。2023 年的突破是 3D Gaussian Splatting。2024-2026 年的生成式進展，則是在上面疊加 multi-view diffusion 與 3D 重建，從單一 prompt 或照片產生物件與場景。

**Type:** Learn
**Languages:** Python
**Prerequisites:** Phase 4 (Vision), Phase 8 · 07 (Latent Diffusion)
**Time:** ~45 分鐘

## 問題 (The Problem)

3D 內容很痛苦：

- **表示方式。** Mesh、點雲、voxel 網格、signed distance field (SDF)、neural radiance field (NeRF)、3D Gaussian。每一種都有取捨。
- **資料稀少。** ImageNet 有 1400 萬張圖像。最大的乾淨 3D 資料集（Objaverse-XL, 2023）約有 1000 萬個物件，多數品質不高。
- **記憶體。** 512³ 的 voxel 網格有 1.28 億個 voxel；一個實用的場景 NeRF 每條光線需要 1M 個樣本。生成比重建更困難。
- **監督訊號。** 對 2D 圖像你有像素。對 3D 你通常只有少數幾個 2D 視角，必須把它們提升到 3D。

2026 年的技術組合把兩個問題分開。第一步，用 diffusion model 生成 *2D 多視角圖像*。第二步，把 *3D 表示*（通常是 Gaussian splatting）擬合到這些圖像上。

## 核心概念 (The Concept)

![3D 生成：multi-view diffusion + 3D 重建](../assets/3d-generation.svg)

### 表示方式：3D Gaussian Splatting (Kerbl et al., 2023)

把場景表示成約 1M 個 3D Gaussian 的雲。每個有 59 個參數：位置 (3)、共變異數 (6，或四元數 4 + scale 3)、不透明度 (1)、球諧函數 (spherical harmonics) 顏色（degree 3 時 48 個，degree 0 時 3 個）。

渲染 = 投影 + alpha-compositing。很快（在 4090 上 1080p 約 100 fps）。可微分。對真實照片做梯度下降來擬合。在消費級 GPU 上，一個場景 5-30 分鐘就能擬合完成。

在此之上，2023-2024 年有兩項創新：
- **生成式 Gaussian splat。** LGM、LRM、InstantMesh 等模型直接從一張或少數幾張圖像預測 Gaussian 雲。
- **4D Gaussian Splatting。** 針對動態場景，讓 Gaussian 帶有逐格的偏移量。

### Multi-view diffusion

Fine-tune 一個預訓練的圖像 diffusion model，讓它根據文字 prompt 或單張圖像，生成同一個物件的多個一致視角。Zero123 (Liu et al., 2023)、MVDream (Shi et al., 2023)、SV3D (Stability, 2024)、CAT3D (Google, 2024)。通常輸出物件周圍 4-16 個視角，再透過 Gaussian splatting 或 NeRF 提升到 3D。

### Text-to-3D 管線

| 模型 | 輸入 | 輸出 | 時間 |
|------|------|------|------|
| DreamFusion (2022) | 文字 | 透過 SDS 得到 NeRF | 每個資產約 1 小時 |
| Magic3D | 文字 | mesh + 紋理 | ~40 分鐘 |
| Shap-E (OpenAI, 2023) | 文字 | 隱式 3D | ~1 分鐘 |
| SJC / ProlificDreamer | 文字 | NeRF / mesh | ~30 分鐘 |
| LRM (Meta, 2023) | 圖像 | triplane | ~5 s |
| InstantMesh (2024) | 圖像 | mesh | ~10 s |
| SV3D (Stability, 2024) | 圖像 | 新視角 | ~2 分鐘 |
| CAT3D (Google, 2024) | 1-64 張圖像 | 3D NeRF | ~1 分鐘 |
| TripoSR (2024) | 圖像 | mesh | ~1 s |
| Meshy 4 (2025) | 文字 + 圖像 | PBR mesh | ~30 s |
| Rodin Gen-1.5 (2025) | 文字 + 圖像 | PBR mesh | ~60 s |
| Tencent Hunyuan3D 2.0 (2025) | 圖像 | mesh | ~30 s |

2025-2026 年的方向：直接 text-to-mesh、帶 PBR 材質、適合遊戲引擎的模型。對一般物件而言，multi-view diffusion 作為中間步驟，仍是表現最好的配方。

### NeRF（作為背景知識）

Neural Radiance Field (Mildenhall et al., 2020)。一個小型 MLP 接收 `(x, y, z, 視線方向)`，輸出 `(顏色, 密度)`。沿著光線積分來渲染。在新視角合成的品質上勝過基於 mesh 的方法，但渲染慢 100 到 1000 倍。在大多數即時應用中已被 Gaussian splatting 取代，但在研究中仍是主流。

```figure
v4-3d-multiview
```

## 動手做 (Build It)

`code/main.py` 實作了一個玩具版的 2D「Gaussian splatting」擬合：把一張合成的目標圖像（平滑漸層）表示成 2D Gaussian splat 的總和。用梯度下降最佳化位置、顏色和共變異數，使其吻合目標。你會看到兩個核心操作：forward 渲染（splat + alpha-composite）以及用梯度下降擬合。

### 步驟 1：2D Gaussian splat

```python
def gaussian_at(x, y, gaussian):
    px, py = gaussian["pos"]
    sigma = gaussian["sigma"]
    d2 = (x - px) ** 2 + (y - py) ** 2
    return math.exp(-d2 / (2 * sigma * sigma))
```

### 步驟 2：把 splat 加總來渲染

```python
def render(image_size, gaussians):
    img = [[0.0] * image_size for _ in range(image_size)]
    for g in gaussians:
        for y in range(image_size):
            for x in range(image_size):
                img[y][x] += g["color"] * gaussian_at(x, y, g)
    return img
```

真正的 3D Gaussian splatting 會依深度排序 Gaussian，並按順序做 alpha-composite。我們的 2D 玩具只做加總。

### 步驟 3：用梯度下降擬合

```python
for step in range(steps):
    pred = render(size, gaussians)
    loss = mse(pred, target)
    gradients = compute_grads(pred, target, gaussians)
    update(gaussians, gradients, lr)
```

## 常見陷阱 (Pitfalls)

- **視角不一致。** 如果你獨立生成 4 個視角，而它們對物件結構的看法不一，3D 擬合就會模糊。解法：使用共享 attention 的 multi-view diffusion。
- **背面幻覺。** 單張圖像 → 3D 必須自己編造看不到的那一面。品質差異很大。
- **Gaussian splat 爆炸。** 不加約束的訓練會長到 1000 萬個 splat 並過度擬合。Densification + pruning 的啟發式方法（來自 3D-GS 原始論文）是必要的。
- **拓撲問題。** 從隱式場 (SDF) 得到的 mesh 常有破洞或自我交叉。上線前先跑 remesher（例如 blender 的 voxel remesh）。
- **訓練資料的授權。** Objaverse 的授權混雜；商業使用依模型而異。

## 實際使用 (Use It)

| 任務 | 2026 年的選擇 |
|------|-------------|
| 從照片重建場景 | Gaussian splatting (3DGS, Gsplat, Scaniverse) |
| 供遊戲使用的 text-to-3D 物件 | Meshy 4 或 Rodin Gen-1.5（PBR 輸出） |
| Image-to-3D | Hunyuan3D 2.0、TripoSR、InstantMesh |
| 從少數圖像做新視角合成 | CAT3D、SV3D |
| 動態場景重建 | 4D Gaussian Splatting |
| Avatar / 穿衣的人體 | Gaussian Avatar、HUGS |
| 研究 / SOTA | 上週剛發表的那個 |

要把 3D 上線到遊戲或電商管線：Meshy 4 或 Rodin Gen-1.5 輸出的 PBR mesh 可以直接放進 Unity / Unreal。

## 交付 (Ship It)

存成 `outputs/skill-3d-pipeline.md`。這個 skill 接收 3D 簡報（輸入：文字 / 一張圖像 / 少數圖像；輸出：mesh / splat / NeRF；用途：渲染 / 遊戲 / VR），輸出：管線（multi-view diffusion + 擬合，或直接 mesh 模型）、基礎模型、迭代預算、拓撲後處理，以及需要的材質通道。

## 練習 (Exercises)

1. **簡單。** 用 4、16、64 個 Gaussian 執行 `code/main.py`。回報相對於目標的最終 MSE。
2. **中等。** 擴充成有顏色的 Gaussian (RGB)。確認重建結果吻合目標的顏色模式。
3. **困難。** 使用 gsplat 或 Nerfstudio，從 50 張照片的拍攝中重建一個真實物件。回報擬合時間，以及在保留視角上的最終 SSIM。

## 關鍵術語 (Key Terms)

| 術語 | 一般說法 | 實際意義 |
|------|---------|---------|
| 3D Gaussian Splatting | 「3DGS」 | 場景表示成 3D Gaussian 的雲；可微分的 alpha-composite 渲染。 |
| NeRF | 「Neural radiance field」 | 在 3D 點上輸出顏色 + 密度的 MLP；以光線積分渲染。 |
| Triplane | 「三個 2-D 平面」 | 將 3D 分解成三個與座標軸對齊的 2-D 特徵網格；比體積表示便宜。 |
| SDS | 「Score distillation sampling」 | 用 2D-diffusion 的 score 當作偽梯度來訓練 3D 模型。 |
| Multi-view diffusion | 「一次產生很多視角」 | 輸出一批一致的相機視角的 diffusion model。 |
| PBR | 「Physically-based rendering」 | 具有 albedo、roughness、metallic、normal 通道的材質。 |
| Densification | 「讓 splat 長出來」 | 3DGS 的訓練啟發式方法：在高梯度區域分裂 / 複製 splat。 |

## 實務筆記 (Production note)：3D 還沒有共同的底座

不像圖像（latent diffusion + DiT）和影片（時空 DiT），3D 在 2026 年沒有單一主導的 runtime。正式環境的決策樹會依表示方式分岔：

- **NeRF / triplane。** 推論是 ray-marching + 每個樣本一次 MLP forward。一張 512² 的渲染需要數百萬次 MLP forward。要積極地 batch 光線樣本；SDPA/xformers 適用。
- **Multi-view diffusion + LRM 重建。** 兩階段管線。第 1 階段（multi-view DiT）就是跟第 07 課一樣的 diffusion 伺服器。第 2 階段（LRM transformer）是對各視角做一次 forward pass。整體的延遲輪廓是「diffusion + 一次到位」，所以要依階段選擇對應的服務基本元件。
- **SDS / DreamFusion。** 是逐資產的最佳化，不是推論。要建成 job，而不是 request handler。

對 2026 年的大多數產品來說，正確答案是：「收到請求時跑 multi-view diffusion 模型，非同步地重建成 3DGS，再提供 3DGS 做即時檢視」。這樣把工作負載乾淨地切成 GPU 推論伺服器（快）與離線最佳化器（慢）。

## 延伸閱讀 (Further Reading)

- [Mildenhall et al. (2020). NeRF: Representing Scenes as Neural Radiance Fields](https://arxiv.org/abs/2003.08934) — NeRF。
- [Kerbl et al. (2023). 3D Gaussian Splatting for Real-Time Radiance Field Rendering](https://arxiv.org/abs/2308.04079) — 3DGS。
- [Poole et al. (2022). DreamFusion: Text-to-3D using 2D Diffusion](https://arxiv.org/abs/2209.14988) — SDS。
- [Liu et al. (2023). Zero-1-to-3: Zero-shot One Image to 3D Object](https://arxiv.org/abs/2303.11328) — Zero123。
- [Shi et al. (2023). MVDream](https://arxiv.org/abs/2308.16512) — multi-view diffusion。
- [Hong et al. (2023). LRM: Large Reconstruction Model for Single Image to 3D](https://arxiv.org/abs/2311.04400) — LRM。
- [Gao et al. (2024). CAT3D: Create Anything in 3D with Multi-View Diffusion Models](https://arxiv.org/abs/2405.10314) — CAT3D。
- [Stability AI (2024). Stable Video 3D (SV3D)](https://stability.ai/research/sv3d-novel-multi-view-synthesis-and-3d-generation-from-a-single-image-using-latent-video-diffusion) — SV3D。
