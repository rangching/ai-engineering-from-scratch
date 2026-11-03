# StyleGAN

> 大多數 generator 會同時把 $z$ 攪進每一層。StyleGAN 把它拆開：先把 $z$ 映射成中間的 $w$，再透過 AdaIN 在每個解析度層級 *注入* $w$。光是這個改變，就解開了 latent 空間的糾纏，讓擬真人臉成為一個連續七年都被視為已解決的問題。

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 8 · 03 (GANs), Phase 4 · 08 (Normalization), Phase 3 · 07 (CNNs)
**Time:** ~45 分鐘

## 問題 (The Problem)

DCGAN 透過一疊 transposed convolution，把 $z$ 映射成圖像。問題是：$z$ 控制了一切，包括姿態、光線、身分、背景，全部糾纏在一起。沿著 $z$ 的某一個軸移動，四件事都會變。你沒辦法對模型說「同一個人，不同姿態」，因為表示方式並不是這樣分解的。

Karras et al. (2019, NVIDIA) 提出：不要再把 $z$ 直接餵進 conv 層。改成把一個常數的 `4×4×512` tensor 當作網路輸入。學一個 8 層的 MLP，把 $z \in Z \to w \in W$。透過 *adaptive instance normalization* (AdaIN) 在每個解析度注入 $w$：先把每個 conv feature map 正規化，再用 $w$ 的 affine 投影做縮放與平移。每一層再加上雜訊，產生隨機的細節（皮膚毛孔、髮絲）。

結果：$W$ 對於「高層風格」（姿態、身分）與「細部風格」（光線、顏色）有大致正交的軸。你可以交換兩張圖像的風格：低解析度層級用圖像 A 的 $w$，高解析度層級用圖像 B 的 $w$。這解鎖了編輯、跨領域風格化，以及整條「StyleGAN inversion」的研究路線。

## 核心概念 (The Concept)

![StyleGAN：mapping network + AdaIN + 逐層雜訊](../assets/stylegan.svg)

**Mapping network。** $f: Z \to W$，一個 8 層的 MLP。$Z = \mathcal{N}(0, I)^{512}$。$W$ 不被強迫成為 Gaussian，它會學出貼合資料的形狀。

**Synthesis network。** 從一個學出來的常數 `4×4×512` 開始。每個解析度區塊：`upsample → conv → AdaIN(w_i) → noise → conv → AdaIN(w_i) → noise`。解析度逐步加倍：4、8、16、32、64、128、256、512、1024。

**AdaIN。**

```math
\mathop{\mathrm{AdaIN}}(x, y) = y_{\text{scale}} \cdot \frac{x - \mathop{\mathrm{mean}}(x)}{\mathop{\mathrm{std}}(x)} + y_{\text{bias}}
```

其中 $y_{\text{scale}}$ 和 $y_{\text{bias}}$ 來自 $w$ 的 affine 投影。先對每個 feature map 正規化，再重新賦予風格。這裡的「風格」指的是 feature map 的一階與二階統計量。

**逐層雜訊。** 把單通道的 Gaussian 雜訊加到每個 feature map 上，並乘上學出來的逐通道係數。它控制隨機細節，卻不影響整體結構。

**Truncation trick。** 推論時，抽樣 $z$、計算 $w = \text{mapping}(z)$，然後 $w' = \hat{w} + \psi \cdot (w - \hat{w})$，其中 $\hat{w}$ 是許多樣本上 $w$ 的平均。$\psi < 1$ 以多樣性換取品質。幾乎每個 StyleGAN demo 都用 $\psi \approx 0.7$。

## StyleGAN 1 → 2 → 3

| 版本 | 年份 | 創新 |
|------|------|------|
| StyleGAN | 2019 | Mapping network + AdaIN + 雜訊 + progressive growing。 |
| StyleGAN2 | 2020 | 以 weight demodulation 取代 AdaIN（修正水滴瑕疵）；skip/residual 架構；path-length 正則化。 |
| StyleGAN3 | 2021 | 無 aliasing 的卷積 + equivariant kernel；消除紋理黏在像素網格上的問題。 |
| StyleGAN-XL | 2022 | 類別條件式、1024²、ImageNet。 |
| R3GAN | 2024 | 以更強的正則化重新包裝；用少 20 倍的參數，在 FFHQ-1024 上拉近與 diffusion 的差距。 |

2026 年，StyleGAN3 仍是下列情況的預設選擇：(a) 要高 FPS 的窄領域擬真度，(b) few-shot 領域適應（用 100 張圖像訓練新資料集，並凍結 mapping），(c) 基於 inversion 的編輯（找出能重建真實照片的 $w$，再編輯那個 $w$）。對於開放領域的 text-to-image，它不是合適的工具，diffusion 才是。

```figure
gx-stylegan-mapping
```

## 動手做 (Build It)

`code/main.py` 在 1-D 中實作一個玩具版的「style-GAN lite」：一個 mapping MLP、一個合成函數（取一個學出來的常數向量，再用 $w$ 導出的 scale/bias 去調變它），以及逐層雜訊。它顯示，透過 affine 調變注入 $w$，效果等於或優於把 $z$ 串接到 generator 的輸入。

### 步驟 1：mapping network

```python
def mapping(z, M):
    h = z
    for i in range(num_layers):
        h = leaky_relu(add(matmul(M[f"W{i}"], h), M[f"b{i}"]))
    return h
```

### 步驟 2：adaptive instance normalization

```python
def adain(x, w_scale, w_bias):
    mu = mean(x)
    sd = std(x)
    x_norm = [(xi - mu) / (sd + 1e-8) for xi in x]
    return [w_scale * xi + w_bias for xi in x_norm]
```

每個 feature map 的 scale 和 bias，是由 $w$ 經線性投影得到的。

### 步驟 3：逐層雜訊

```python
def add_noise(x, sigma, rng):
    return [xi + sigma * rng.gauss(0, 1) for xi in x]
```

每個通道的 sigma 是可學習的。

## 常見陷阱 (Pitfalls)

- **水滴瑕疵 (Droplet artifacts)。** StyleGAN 1 在 feature map 中會產生一個斑塊狀的水滴，因為 AdaIN 把平均值歸零。StyleGAN 2 的 weight demodulation 改成縮放卷積權重，修正了這點。
- **紋理黏著 (Texture sticking)。** StyleGAN 1 和 2 的紋理跟著像素座標走，而不是跟著物體座標走（內插時看得出來）。StyleGAN 3 的 alias-free 卷積用 windowed sinc filter 修正這點。
- **Mode 覆蓋率。** Truncation `ψ < 0.7` 看起來乾淨，但樣本來自很窄的錐形範圍；需要多樣性時請用 `ψ = 1.0`。
- **Inversion 有損失。** 把真實照片 invert 進 $W$，通常透過最佳化或 encoder（e4e、ReStyle、HyperStyle）完成。迭代很多次之後，結果會漂移。

## 實際使用 (Use It)

| 使用情境 | 做法 |
|---------|------|
| 擬真人臉（動漫、商品、窄領域） | StyleGAN3 FFHQ / 自訂 fine-tune |
| 從照片編輯人臉 | e4e inversion + StyleSpace / InterFaceGAN 方向 |
| 換臉 / 表情重演 | StyleGAN + encoder + blending |
| Avatar 管線 | 用 ADA 做低資料量 fine-tune 的 StyleGAN3 |
| 用少量圖像做領域適應 | 凍結 mapping network，fine-tune synthesis |
| 多模態或文字條件式生成 | 別用，改用 diffusion |

對於答案是「某個人臉的照片」的產品級 demo，在同樣的品質門檻下，StyleGAN 在推論成本（單次 forward pass，在 4090 上 <10ms）與銳利度上勝過 diffusion。

## 交付 (Ship It)

存成 `outputs/skill-stylegan-inversion.md`。這個 skill 接收一張真實照片，輸出：inversion 方法（e4e / ReStyle / HyperStyle）、預期的 latent 損失、編輯預算（在出現瑕疵之前，你能在 $W$ 中移動多遠），以及一份已知可用的編輯方向清單（年齡、表情、姿態）。

## 練習 (Exercises)

1. **簡單。** 分別用 `adain_on=True` 和 `adain_on=False` 執行 `code/main.py`。比較固定 latent 與受擾動 latent 下，輸出的分散程度。
2. **中等。** 實作 mixing regularization：對一個訓練 batch，計算 $w_a$、$w_b$，並在合成的前半段使用 $w_a$、後半段使用 $w_b$。Decoder 會學到 disentangled 的風格嗎？
3. **困難。** 取一個預訓練的 StyleGAN3 FFHQ 模型 (ffhq-1024.pkl)。透過在有標註的樣本上訓練 SVM，找出控制「微笑」的 $w$ 方向；回報在身分開始漂移之前，你能推多遠。

## 關鍵術語 (Key Terms)

| 術語 | 一般說法 | 實際意義 |
|------|---------|---------|
| Mapping network | 「那個 MLP」 | $f: Z \to W$，8 層，將 latent 幾何與資料統計量解耦。 |
| W space | 「風格空間」 | Mapping network 的輸出；大致上是 disentangled 的。 |
| AdaIN | 「Adaptive instance norm」 | 先正規化 feature map，再用 $w$ 的投影做縮放 + 平移。 |
| Truncation trick | 「Psi」 | $w = \text{mean} + \psi \cdot (w - \text{mean})$，$\psi<1$ 以多樣性換品質。 |
| Path-length regularization | 「PL reg」 | 懲罰 $w$ 每單位變化所造成的大幅圖像變化；讓 $W$ 更平滑。 |
| Weight demodulation | 「StyleGAN2 的修正」 | 正規化 conv 權重而不是 activation；消除水滴瑕疵。 |
| Alias-free | 「StyleGAN3 的技巧」 | Windowed sinc filter；消除紋理黏在像素網格上。 |
| Inversion | 「為真實圖像找出 w」 | 透過最佳化或編碼 $x \to w$，使 $G(w) \approx x$。 |

## 實務筆記 (Production note)：為什麼 StyleGAN 在 2026 年仍然上線

StyleGAN3 在 4090 上生成一張 1024² 的 FFHQ 人臉不到 10 ms：`num_steps = 1`，沒有 VAE decode，沒有 cross-attention。以正式環境的角度，這是任何圖像生成器的延遲下限。同樣解析度下，50 步的 SDXL + VAE-decode 管線約 3 秒。這是 **300 倍的差距**，對窄領域的產品（avatar 服務、證件文件管線、圖庫人臉生成）而言，總持有成本（TCO）上是勝出的。

兩個營運上的結果：

- **不需要 scheduler，不需要 batcher。** 在目標占用率下用靜態 batch 就是最佳的。Continuous batching（對 LLM 和 diffusion 很重要）在這裡毫無助益，因為每個請求花的 FLOPs 都一樣。
- **Truncation `ψ` 是安全旋鈕。** `ψ < 0.7` 從 mapping network 範圍中很窄的錐形範圍取樣。這是服務層唯一能用來控制樣本變異度的槓桿。尖峰時段調低 `ψ`，對付費用戶則調高。

## 延伸閱讀 (Further Reading)

- [Karras et al. (2019). A Style-Based Generator Architecture for GANs](https://arxiv.org/abs/1812.04948) — StyleGAN。
- [Karras et al. (2020). Analyzing and Improving the Image Quality of StyleGAN](https://arxiv.org/abs/1912.04958) — StyleGAN2。
- [Karras et al. (2021). Alias-Free Generative Adversarial Networks](https://arxiv.org/abs/2106.12423) — StyleGAN3。
- [Tov et al. (2021). Designing an Encoder for StyleGAN Image Manipulation](https://arxiv.org/abs/2102.02766) — e4e inversion。
- [Sauer et al. (2022). StyleGAN-XL: Scaling StyleGAN to Large Diverse Datasets](https://arxiv.org/abs/2202.00273) — StyleGAN-XL。
- [Huang et al. (2024). R3GAN: The GAN is dead; long live the GAN!](https://arxiv.org/abs/2501.05441) — 現代極簡 GAN 配方。
