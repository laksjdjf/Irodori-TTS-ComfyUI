# Irodori-TTS ComfyUI 実装設計書 v3

> v2 からの変更点:
> - `cfg_min_t` / `cfg_max_t` を IrodoriCFGGuider の入力ピンとして公開
> - IrodoriVAEDecode カスタムノード廃止 → 公式 `VAEDecodeAudio` を使用
> - 公式 `VAEEncodeAudio` も使用 → IrodoriVAEWrapper が `encode()` / `decode()` を実装
> - TextEncode の ref_latent 入力を LATENT 型（VAEEncodeAudio の出力）に変更
> - TextEncode に MODEL ピンを追加（encode_conditions 実行時のデバイス管理のため）
> - カスタムノードが 4 個に整理

---

## 1. ファイル構成

```
Irodori-TTS-ComfyUI/
├── __init__.py
├── core/
│   ├── __init__.py
│   ├── irodori_import.py      # irodori_tts パッケージのパス解決
│   ├── model_wrapper.py       # IrodoriModelSampling / IrodoriLatentFormat / IrodoriModelWrapper
│   └── loader.py              # チェックポイント読み込み + IrodoriVAEWrapper
└── nodes/
    ├── __init__.py
    ├── checkpoint_loader.py   # IrodoriCheckpointLoader
    ├── text_encode.py         # IrodoriTextEncode
    ├── guider.py              # IrodoriCFGGuider ノード + IrodoriGuider クラス
    └── latent.py              # IrodoriEmptyLatent
```

### カスタムノード 4 個

| ノード | 役割 |
|---|---|
| `IrodoriCheckpointLoader` | MODEL + CLIP + VAE を出す |
| `IrodoriTextEncode` | テキスト＋ref latent → 4種 CONDITIONING |
| `IrodoriCFGGuider` | N-way CFG の GUIDER を作る |
| `IrodoriEmptyLatent` | duration 計算 → ゼロ LATENT |

公式ノード（追加不要）: `LoadAudio` / `VAEEncodeAudio` / `VAEDecodeAudio` / `SaveAudio` / `RandomNoise` / `KSamplerSelect` / `BasicScheduler` / `SamplerCustomAdvanced`

---

## 2. 型一覧

| 型名 | Python 実体 | 説明 |
|---|---|---|
| `MODEL` | `ModelPatcher` | 公式型 |
| `CLIP` | `dict` | tokenizer + model_ref + cfg + duration info |
| `VAE` | `IrodoriVAEWrapper` | 公式型名。encode/decode を実装 |
| `CONDITIONING` | `list[list]` | 公式型 |
| `LATENT` | `{"samples": Tensor}` | 公式型。**2 種の shape が流れる（後述）** |
| `AUDIO` | `{"waveform": Tensor, "sample_rate": int}` | 公式型 |
| `GUIDER` | `IrodoriGuider` | 公式型名 |
| `SIGMAS` | `Tensor` | 公式型 |

---

## 3. LATENT の 2 種類の shape

ワークフロー内で LATENT 型のテンソルが 2 種類流れるが、**接続先が異なるため混在しない**。

| 用途 | shape | 生成元 | 接続先 |
|---|---|---|---|
| **ref latent**（話者条件用） | `(B, T_lat, 32)` ← 3D | `VAEEncodeAudio` | `IrodoriTextEncode` |
| **サンプリング latent** | `(B, D, 1, T)` ← 4D | `IrodoriEmptyLatent` → `SamplerCustomAdvanced` | `VAEDecodeAudio` |

4D の `(B, D, 1, T)` の規約:
- `D = patched_latent_dim = latent_dim × latent_patch_size`
- `IrodoriLatentFormat.latent_channels = D` にすることで `fix_empty_latent_channels` がスキップされる

---

## 4. IrodoriVAEWrapper（core/loader.py に実装）

`VAEEncodeAudio` / `VAEDecodeAudio` が呼ぶインターフェースを実装する。

```python
class IrodoriVAEWrapper:
    def __init__(self, codec: DACVAECodec, model_cfg: ModelConfig):
        self.codec = codec
        self.model_cfg = model_cfg
        self.audio_sample_rate = codec.sample_rate         # VAEEncodeAudio が参照
        self.audio_sample_rate_output = codec.sample_rate  # VAEDecodeAudio が参照

    def encode(self, waveform: Tensor) -> Tensor:
        """
        VAEEncodeAudio が呼ぶ。
        waveform: (B, T_audio, C)  ← vae.encode(waveform.movedim(1,-1)) で変換済み
        return:   (B, T_lat, 32)   ← raw DACVAE latent (3D)
        """
        # (B, T, C) → (B, C, T) に戻してから DACVAE encode
        wav = waveform.movedim(-1, 1)  # (B, C, T)
        return self.codec.encode_waveform(wav, self.codec.sample_rate)  # (B, T_lat, 32)

    def decode(self, latent: Tensor) -> Tensor:
        """
        VAEDecodeAudio が呼ぶ。decode 結果に .movedim(-1, 1) が適用される。
        latent: (B, D, 1, T_patched)  ← SamplerCustomAdvanced の出力（4D ComfyUI 形式）
        return: (B, T_audio, 1)       ← .movedim(-1,1) で (B,1,T_audio) の AUDIO になる
        """
        # (B, D, 1, T) → (B, T, D) → unpatchify → (B, T_raw, 32) → decode → (B, 1, T_audio)
        x = latent.squeeze(2).transpose(1, 2)          # (B, T_patched, D)
        x = unpatchify_latent(x,
                              self.model_cfg.latent_patch_size,
                              self.codec.latent_dim)    # (B, T_raw, 32)
        audio = self.codec.decode_latent(x)             # (B, 1, T_audio)
        return audio.movedim(1, -1)                     # (B, T_audio, 1)
```

> **注意**: `vae_decode_audio` は decode 後に標準偏差で正規化する処理を行う。
> DACVAE 出力は Stable Audio 等とスケールが異なる可能性があるが、まず動かして確認する。

---

## 5. 各ノード詳細

### 5-1. IrodoriCheckpointLoader

**入力**

| ピン | 型 | 説明 |
|---|---|---|
| `ckpt_name` | STRING (combo) | `models/checkpoints/` 以下 |
| `codec_repo` | STRING | デフォルト `"Aratako/Semantic-DACVAE-Japanese-32dim"` |
| `device` | STRING combo | `"cuda"` / `"cpu"` |
| `dtype` | STRING combo | `"bf16"` / `"fp32"` |

**出力**: MODEL / CLIP / VAE

```python
# CLIP の中身
clip = {
    "tokenizer": PretrainedTextTokenizer,
    "caption_tokenizer": PretrainedTextTokenizer | None,  # use_caption_condition 時のみ
    "model": TextToLatentRFDiT,
    "cfg": ModelConfig,
    "has_duration_predictor": bool,
}
# VAE
vae = IrodoriVAEWrapper(DACVAECodec.load(...), model_cfg)
```

---

### 5-2. IrodoriTextEncode

**入力**

| ピン | 型 | 必須 | 説明 |
|---|---|---|---|
| `model` | MODEL | ✅ | encode_conditions 実行時のデバイス管理 |
| `clip` | CLIP | ✅ | tokenizer + model |
| `text` | STRING (multiline) | ✅ | 読み上げテキスト |
| `ref_latent` | LATENT | ❌ | VAEEncodeAudio の出力。`samples` が `(B, T_lat, 32)` の 3D tensor |
| `caption` | STRING | ❌ | VoiceDesign 用 |
| `speaker_uncond_mode` | combo | ✅ | `"mask"` / `"noise"` |

**出力**: cond / text_uncond / speaker_uncond / caption_uncond（全て CONDITIONING）

**処理の流れ**:
```python
# 1. デバイスにモデルをロード
comfy.model_management.load_models_gpu([model])
device = model.load_device

# 2. テキストトークナイズ
text_ids, text_mask = clip["tokenizer"].batch_encode([text])

# 3. ref_latent の取得（オプション）
if ref_latent is not None:
    ref_lat = ref_latent["samples"]          # (B, T_lat, 32)  ← 3D
    ref_mask = torch.ones(B, T_lat, dtype=torch.bool)
else:
    ref_lat, ref_mask = None, None

# 4. encode_conditions（1回呼ぶだけ）
model_obj = clip["model"]  # TextToLatentRFDiT
with torch.no_grad():
    text_s_c, text_m_c, spk_s_c, spk_m_c, cap_s_c, cap_m_c = model_obj.encode_conditions(
        text_input_ids=text_ids.to(device),
        text_mask=text_mask.to(device),
        ref_latent=ref_lat.to(device) if ref_lat is not None else None,
        ref_mask=ref_mask.to(device) if ref_mask is not None else None,
        caption_input_ids=cap_ids.to(device) if caption else None,
        caption_mask=cap_mask.to(device) if caption else None,
        speaker_uncond_mode=speaker_uncond_mode,
    )

# 5. uncond テンソル生成
text_s_u = torch.zeros_like(text_s_c)
text_m_u = torch.zeros_like(text_m_c)
# speaker uncond
if spk_s_c is not None:
    if speaker_uncond_mode == "noise":
        spk_s_u = torch.randn_like(spk_s_c) * spk_s_c.std().clamp_min(1e-6)
        spk_m_u = torch.ones_like(spk_m_c)
    else:
        spk_s_u = torch.zeros_like(spk_s_c)
        spk_m_u = torch.zeros_like(spk_m_c)
# caption uncond
if cap_s_c is not None:
    cap_s_u = torch.zeros_like(cap_s_c)
    cap_m_u = torch.zeros_like(cap_m_c)

# 6. CONDITIONING にパック
def _pack(ts, tm, ss, sm, cs, cm):
    return [[None, {"text_state": ts, "text_mask": tm,
                    "speaker_state": ss, "speaker_mask": sm,
                    "caption_state": cs, "caption_mask": cm}]]

cond           = _pack(text_s_c, text_m_c, spk_s_c, spk_m_c, cap_s_c, cap_m_c)
text_uncond    = _pack(text_s_u, text_m_u, spk_s_c, spk_m_c, cap_s_c, cap_m_c)
speaker_uncond = _pack(text_s_c, text_m_c, spk_s_u, spk_m_u, cap_s_c, cap_m_c)
caption_uncond = _pack(text_s_c, text_m_c, spk_s_c, spk_m_c, cap_s_u, cap_m_u)
```

---

### 5-3. IrodoriCFGGuider（ノード）

**入力**

| ピン | 型 | 必須 | デフォルト |
|---|---|---|---|
| `model` | MODEL | ✅ | |
| `cond` | CONDITIONING | ✅ | |
| `uncond_1` | CONDITIONING | ❌ | |
| `scale_1` | FLOAT | ❌ | 3.0 |
| `uncond_2` | CONDITIONING | ❌ | |
| `scale_2` | FLOAT | ❌ | 5.0 |
| `uncond_3` | CONDITIONING | ❌ | |
| `scale_3` | FLOAT | ❌ | 3.0 |
| `cfg_min_t` | FLOAT | ✅ | 0.5 |
| `cfg_max_t` | FLOAT | ✅ | 1.0 |

**出力**: GUIDER

---

### 5-4. IrodoriGuider クラス（実装の核心）

#### `sample()` の流れ

```python
def sample(self, noise, latent_image, sampler, sigmas, ...):
    device = comfy.model_management.get_torch_device()
    comfy.model_management.load_models_gpu([self.model_patcher])
    self.inner_model = self.model_patcher.model          # IrodoriModelWrapper
    irodori = self.inner_model.model                     # TextToLatentRFDiT

    # 条件をデバイスへ移動
    self._cond_dev    = _cond_to(self.cond,    device, self.inner_model.dtype)
    self._unconds_dev = [_cond_to(u, device, self.inner_model.dtype) for u in self.unconds]

    # ノイズスケーリング
    max_denoise = math.isclose(float(self.inner_model.model_sampling.sigma_max),
                               float(sigmas[0]), rel_tol=1e-5)
    x_start = self.inner_model.model_sampling.noise_scaling(
        sigmas[0], noise.to(device), latent_image.to(device), max_denoise
    )

    # sampler に渡す（self が callable として使われる）
    samples = sampler.sample(
        self, sigmas.to(device),
        {"model_options": {}, "seed": seed},
        callback, x_start, latent_image.to(device), denoise_mask, disable_pbar,
    )

    samples = self.inner_model.model_sampling.inverse_noise_scaling(sigmas[-1], samples)
    output = self.inner_model.process_latent_out(samples.float())
    comfy.model_management.cleanup_models()
    return output
```

#### `predict_noise()` の流れ（per step）

```python
def predict_noise(self, x, sigma, model_options={}, seed=None):
    """
    x:     (B, D, 1, T)  4D ComfyUI latent
    sigma: Tensor  ← t ∈ [0,1]（sigma == t）
    return:(B, D, 1, T)  x0 (denoised)
    """
    t_val = float(sigma.flatten()[0])
    irodori = self.inner_model.model

    # (B,D,1,T) → (B,T,D)
    x_lit = x.squeeze(2).transpose(1, 2).to(self.inner_model.dtype)
    t_lit = sigma[:x_lit.shape[0]].to(x_lit.dtype)

    use_cfg = bool(self._unconds_dev) and (self.cfg_min_t <= t_val <= self.cfg_max_t)

    if not use_cfg:
        v = irodori.forward_with_encoded_conditions(
            x_t=x_lit, t=t_lit, **_unpack(self._cond_dev)
        )  # (B, T, D)
    else:
        all_bundles = [self._cond_dev] + self._unconds_dev
        N = len(all_bundles)
        # バッチ化して 1 回の forward で全部出す
        v_out = irodori.forward_with_encoded_conditions(
            x_t=x_lit.repeat(N, 1, 1),
            t=t_lit.repeat(N),
            **_batch_conds(all_bundles),
        )  # (B*N, T, D)

        chunks = v_out.chunk(N, dim=0)
        v = chunks[0]
        for i, scale in enumerate(self.scales):
            v = v + scale * (chunks[0] - chunks[i + 1])

    # (B,T,D) → (B,D,1,T)、x0 を返す
    v_comfy = v.float().transpose(1, 2).unsqueeze(2)
    return self.inner_model.model_sampling.calculate_denoised(sigma, v_comfy, x)
```

> **KV cache は後回し**。追加時は `forward_with_encoded_conditions` に `context_kv_cache=...` を渡すだけ。

---

### 5-5. IrodoriEmptyLatent

**入力**

| ピン | 型 | 必須 | 説明 |
|---|---|---|---|
| `model` | MODEL | ✅ | model_cfg から patched_latent_dim を取得 |
| `clip` | CLIP | ✅ | tokenizer（token count 用）+ duration predictor |
| `vae` | VAE | ✅ | sample_rate から秒↔フレーム換算 |
| `text` | STRING | ✅ | duration predictor に渡すテキスト |
| `seconds` | FLOAT | ✅ | 0 = 自動（duration predictor）、>0 = 手動 |
| `duration_scale` | FLOAT | ✅ | 自動推定値への乗数（デフォルト 1.0） |
| `batch_size` | INT | ✅ | |

**出力**: `LATENT {"samples": zeros(B, D, 1, T), "sample_rate": int}`

> `sample_rate` を LATENT に入れておくと、`vae_decode_audio` が `samples["sample_rate"]` を使ってくれる。

---

## 6. core/model_wrapper.py の IrodoriModelSampling

```python
class IrodoriModelSampling(nn.Module):
    # sigma == t ∈ [0, 1]（RF: Rectified Flow）
    def timestep(self, sigma): return sigma          # ×1000 しない
    def percent_to_sigma(self, p): ...               # BasicScheduler 互換
    def calculate_input(self, sigma, x): return x    # x_T = z そのまま
    def calculate_denoised(self, s, v, x): return x - v * reshape(s, v.ndim)  # x0 = x_t - t*v
    def noise_scaling(self, s, n, lat, _): return reshape(s)*n + (1-reshape(s))*lat
    def inverse_noise_scaling(self, s, x): return x
```

**k-euler との整合**:
```
denoised = x - v * sigma  →  d = (x - denoised)/sigma = v
x_next = x + v * (sigma_next - sigma)  ←  RF Euler と一致 ✅
```

---

## 7. 完成ワークフロー

```
[IrodoriCheckpointLoader]
    → MODEL / CLIP / VAE

[LoadAudio] → AUDIO(ref)

[VAEEncodeAudio]  AUDIO(ref) + VAE
    → LATENT(ref) ← samples shape: (B, T_lat, 32) 3D

[IrodoriTextEncode]
    MODEL + CLIP + text + LATENT(ref) + caption
    → cond / text_uncond / speaker_uncond / caption_uncond

[IrodoriEmptyLatent]
    MODEL + CLIP + VAE + text + seconds=0
    → LATENT(empty) ← samples shape: (B, D, 1, T) 4D

[IrodoriCFGGuider]
    MODEL + cond
    + uncond_1=text_uncond    / scale_1=3.0
    + uncond_2=speaker_uncond / scale_2=5.0
    + cfg_min_t=0.5 / cfg_max_t=1.0
    → GUIDER

[RandomNoise]      seed → NOISE
[KSamplerSelect]   euler → SAMPLER
[BasicScheduler]   MODEL + steps=40 → SIGMAS

[SamplerCustomAdvanced]
    GUIDER + NOISE + SAMPLER + SIGMAS + LATENT(empty)
    → LATENT(out) ← samples shape: (B, D, 1, T) 4D

[VAEDecodeAudio]  LATENT(out) + VAE → AUDIO
[SaveAudio]       AUDIO → file
```

---

## 8. 未確定・要確認事項

| # | 事項 | 確認方法 |
|---|---|---|
| 1 | base v3 の `latent_patch_size` 値 | チェックポイント読み込み後 `model_cfg.latent_patch_size` を print |
| 2 | DACVAE の hop_length / frame_rate（秒↔フレーム換算） | `inference_runtime.py` の duration 計算部分 |
| 3 | `vae_decode_audio` の正規化（std×5 除算）が DACVAE 出力の音量に悪影響を与えないか | 実際に動かして確認 |
| 4 | `CONDITIONING [[None, {...}]]` 形式で ComfyUI の `convert_cond` が crash しないか | `comfy/sampler_helpers.py` を確認 |
| 5 | `encode_conditions` はモデルがデバイス上にある必要があるか（CPU でも動くか） | 動かして確認 |
| 6 | VoiceClone 時に `ref_latent=None` で `encode_conditions` を呼ぶとエラーか（`use_speaker_condition_resolved=True` 時） | `model.py` L1500 付近のチェック確認 |

---

## 9. 実装順序

1. `core/irodori_import.py`
2. `core/loader.py`（IrodoriVAEWrapper 含む）
3. `core/model_wrapper.py`（IrodoriModelSampling / IrodoriModelWrapper）
4. `nodes/checkpoint_loader.py`
5. `nodes/text_encode.py`
6. **`nodes/guider.py`**（IrodoriGuider + ノード）
7. `nodes/latent.py`
8. `__init__.py`
