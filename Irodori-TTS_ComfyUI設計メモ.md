# Irodori-TTS を ComfyUI 公式 KSampler / SamplerCustomAdvanced に乗せる設計メモ

> セッション引き継ぎ用。Irodori-TTS（RF-DiT ベースの日本語 TTS）を「ラッパーで全部隠す」のではなく、
> **ComfyUI の公式サンプリング基盤（SamplerCustomAdvanced）に正規ルートで乗せる**ための設計議論まとめ。

---

## 0. ゴールと基本方針

- **目標**: 独自の生成ループをブラックボックス化したラッパーを作るのではなく、**ComfyUI 公式の `SamplerCustomAdvanced`（Guider / Sampler / Sigmas / Noise 分割）に乗せる**。
- **ユーザーの前提**: 普段から KSampler ではなく `SamplerCustomAdvanced` しか使っていない（cond が2つの普通の生成でも）。→ 今回の Irodori 複数条件 CFG にそのまま地続きで行ける。
- **黄金ルール**: 「**繋ぎ口（型・ソケット）は公式準拠、中身は自作**」。
  - `MODEL` 型 → `comfy.model_base.BaseModel` を継承した**契約付きラッパー**（必須）
  - `CLIP` / `VAE` → ただのアダプター・ラッパーで OK（出力の型だけ公式準拠）
  - CFG 合成 → `CFGGuider` を継承した自作 Guider。ただの **N-way 線形 CFG**（cond + uncond×N + scale×N、mode 切替なし）

### 設計シンプル化の方針（ユーザー確定）
- **エンコードは1ノードに集約**：speaker encode / caption encode / uncond maker は作らない。
  `Irodori Text Encode` 1ノードに全エンコーダ（CLIP）を入れ、cond と各 uncond を一括出力する。
  → 実物の `encode_conditions`（text/speaker/caption を1関数で一括エンコード）に忠実。
- **CFGGuider はモード切替を持たない**。`independent` 相当の N-way 線形結合のみ。uncond 端子は**省略可**（繋がない＝その項スキップ）。
  → 「本数可変」問題は Guider 内ロジックではなく**配線（何本繋ぐか）で自然に解決**。Irodori 非依存の汎用 Guider になる。

---

## 1. ラッパーの種類（重要な区別）

| 対象 | ラッパー可否 | 理由 |
|---|---|---|
| **MODEL（RF-DiT 本体）** | ✅ ただし `BaseModel` 継承＋`model_sampling` 必須 | KSampler ループ内で毎 step `apply_model(x, t, cond)` の**契約**で呼ばれる。sigma↔t 変換を ComfyUI に肩代わりさせるため正規ルート必須 |
| **CLIP（text encoder）** | ✅ 素のアダプターで可 | 生成前に1回だけ呼ばれるただの前処理。出力を `CONDITIONING` 型に着地させればよい |
| **VAE（DACVAE）** | ✅ 素のアダプターで可 | `encode`/`decode` を呼ぶだけ。入出力の型（LATENT / 波形）を整える |

- 「MODEL もラッパーでしょ？」→ **半分正しい**。Irodori を書き直すのではなく被せる意味ではラッパー。ただし
  **`BaseModel` 継承＋`ModelSamplingFlow`（flow 用 sigma↔t 変換）を仕込んだ "契約付きラッパー"** でないと、
  sigma↔t 変換が宙に浮いて自前ループ送り＝当初目的が崩壊する。
- Rectified Flow なので `ModelType.FLOW` + `ModelSamplingDiscreteFlow`（Flux/SD3 と同じ系譜）を使う。
  SD3/Flux のモデル実装（`comfy/model_base.py`）がほぼお手本。

---

## 2. Irodori-TTS の実装（リポジトリ確認済みの事実）

リポジトリ: https://github.com/Aratako/Irodori-TTS （`main` = v3 系）

### アーキテクチャ
- **Rectified Flow Diffusion Transformer (RF-DiT)** over continuous **DACVAE** latents（Echo-TTS ベース）。
- 構成要素:
  1. **Text Encoder**: 事前学習 LLM のトークン埋め込み起点 + Self-Attn + SwiGLU + RoPE
  2. **Condition Encoder**: base = reference latent encoder / VoiceDesign = caption encoder
  3. **Diffusion Transformer**: Joint-attention DiT + **Low-Rank AdaLN**（timestep 条件の adaptive LN）+ half-RoPE + SwiGLU
  4. **Duration Predictor**: v3 base に統合（`log1p(num_frames)` を Huber 回帰、token-sum 型）
- コーデック: **Semantic-DACVAE-Japanese-32dim**（48kHz 再構成、`patched_latent_dim`）。

### チェックポイント2系統
- **base（Irodori-TTS-500M-v3）**: text encoder + reference latent encoder + DiT + duration predictor。speaker/style 条件あり。
- **VoiceDesign（500M-v2-VoiceDesign）**: text encoder + caption encoder + DiT。**speaker/reference は無効化**。caption 条件あり。

### 主要ファイル
- `irodori_tts/model.py` … `TextToLatentRFDiT` 本体
- `irodori_tts/rf.py` … **Rectified Flow ユーティリティ & Euler CFG サンプリング（最重要）**
- `irodori_tts/codec.py` … DACVAE ラッパー
- `irodori_tts/tokenizer.py` … 事前学習 LLM トークナイザラッパー（emoji もここで処理）
- `irodori_tts/inference_runtime.py` … キャッシュ付きスレッドセーフ推論ランタイム

---

## 3. CFG の確定仕様（`rf.py: sample_euler_rf_cfg`）

### 3モードある（`cfg_guidance_mode`）
- **`independent`（デフォルト・本命）**: 各有効条件を「1個だけ uncond」にした版を作り、full との差分を各 scale で足す。
- `joint`: 全条件まとめて uncond にした1本との差分（旧 `--cfg-scale` 単一指定用）。
- `alternating`: step ごとに drop 対象を回す（毎 step 2本で省メモリ）。

### independent の合成式（写経用・符号そのまま）
```python
v = v_full
for name in enabled_cfg_names:           # "text" / "speaker" / "caption"
    v = v + cfg_scales[name] * (v_full - v_<name>drop)
```
- `v_full` = 全条件入りの予測
- `v_<name>drop` = その条件だけ uncond にした予測
- **基準は v_full**。`v_full + w*(v_full - v_drop)`。

### enabled 本数は「可変」（4本決め打ちではない）
実行時に以下で本数が変わる:
- `has_text_cfg = cfg_scale_text > 0`
- `has_speaker_cfg = cfg_scale_speaker > 0`（`use_speaker_condition_resolved` が False なら speaker=0 強制）
- `has_caption_cfg = use_caption_condition and cfg_scale_caption > 0 and caption_mask.any()`

組み合わせ例:
- text + speaker + caption 全部 → **4本**
- text + speaker（VoiceClone）→ 3本
- text + caption（VoiceDesign、speaker 無効）→ 3本
- text のみ → 2本（＝普通の CFG）

### uncond の作り方（条件ごとに非自明・ノード外出しの根拠）
```python
text_state_uncond    = torch.zeros_like(text_state_cond)        # text: ゼロ
caption_state_uncond = torch.zeros_like(caption_state_cond)     # caption: ゼロ
# speaker は2モード！
if speaker_uncond_mode == "noise":
    speaker_state_uncond = noise * speaker_state_cond.std()     # ノイズ版
else:  # "mask"
    speaker_state_uncond = torch.zeros_like(speaker_state_cond) # ゼロ版
```
→ **speaker uncond だけ `mask`/`noise` の2モード**がある。これをノード入力として見せられるのが「uncond 化を外のノードに出す」設計の価値。

### デフォルト scale（参考）
`cfg_scale_text=3.0` / `cfg_scale_caption=3.0` / `cfg_scale_speaker=5.0`、`cfg_min_t=0.5` / `cfg_max_t=1.0`（CFG を適用する t 範囲）。

### サンプリングの骨格
- Euler over RF ODE。`x_t = x_t + v * (t_next - t)`。
- t スケジュール: `linear` または `sway`（F5-TTS 風 Sway Sampling、`sway_coeff` デフォルト -1.0）。`t_schedule = (1-u) * 0.999`、狭義単調減少が必須。
- 速度の定義: 直線補間 `x_t=(1-t)x0 + t·z`、`velocity = z - x0`、`x0 = x_t - t·v`。

### KV cache（実在した）
- `model.build_context_kv_cache(text_state, speaker_state, caption_state)` で **text/speaker/caption の K/V をループ外で1回だけ計算してキャッシュ**。
- 各 step は `forward_with_encoded_conditions(..., context_kv_cache=...)` で使い回す。
- speaker K/V だけ別途スケール可能（`scale_speaker_kv_cache`、`speaker_kv_scale` / `speaker_kv_min_t` 等）。
- → 「KV cache は MODEL 内部（apply_model 側）に押し込む」が正しい。conditioning ノード側に持たせない。

### 条件は完全に別経路（concat ではない）
`forward_with_encoded_conditions` は6引数を別々に受ける:
```python
model.forward_with_encoded_conditions(
    x_t, t,
    text_state=..., text_mask=...,
    speaker_state=..., speaker_mask=...,
    caption_state=..., caption_mask=...,
    context_kv_cache=...,
)
```
- Joint-attention + Low-Rank AdaLN で各条件を別注入。
- → conditioning は「別 key 格納（pooling 的）」で正しい。各条件は `(state, mask)` ペアで保持。

### ref 音声の前処理
- README 明記: 「reference latent encoder **consumes patched DACVAE latents** from reference audio」。
- → 参照音声は **DACVAE で latent 化してから** speaker branch へ。生波形直ではない。
- `encode_conditions(..., ref_latent, ref_mask, ...)` で裏取り済み。
- なお `--ref-embed`（Speaker Inversion 学習済み埋め込み）や `--no-ref` も存在。

### text encoder / tokenizer
- 事前学習 LLM 由来の独自トークナイザ。emoji もここで処理。
- → 公式 `CLIPTextEncode` は使えない。**自作して出力を `CONDITIONING` 型に着地**（`text_state`, `text_mask` を extra dict に）。

---

## 4. 想定 ComfyUI ワークフロー（確定版・シンプル化後）

```
[CheckpointLoader (Irodori)]
   → MODEL(RF-DiT) / CLIP(全エンコーダ: text + speaker(ref) + caption) / VAE(DACVAE)
   → DURATION_PREDICTOR（Empty Latent 用に出す）

[VAE Encode]            ref_sound → ref_latent           ※DACVAE latent化(確定)・Text Encode の前段

[Irodori Text Encode]   CLIP + text + ref_latent + caption
                        → cond
                          text_uncond
                          speaker_uncond
                          caption_uncond
                          full_uncond        ※A案で残す（independent では未使用、joint 用の予備端子）
                        ※公式CLIP不可・自作。encode_conditions を1ノードに対応させる。
                        ※uncond 生成（speaker の mask/noise 含む）もこのノード内で完結。

[Irodori CFGGuider]     MODEL
                        + cond
                        + uncond_1 / uncond_2 / uncond_3   ※各省略可（繋がなければスキップ）
                        + scale_1 / scale_2 / scale_3
                        → GUIDER
                        （ただの N-way 線形 CFG。mode 切替なし）

[Irodori Empty Latent]  DURATION_PREDICTOR + text → latent（長さN自動／手動上書き）
[RandomNoise]           seed → noise
[KSamplerSelect]        euler → sampler
[BasicScheduler / 自作] MODEL + num_steps (+ sway系) → sigmas
                        （ModelSamplingFlow を仕込めば BasicScheduler 流用可）

[SamplerCustomAdvanced] guider + noise + sampler + sigmas → latent
[DACVAE Decode]         latent → sound
[SaveAudio]             → 出力
```

### 配線でのつなぎ方（典型ケース）
- **VoiceClone（base, text+speaker, caption 無し）**:
  `uncond_1=text_uncond / scale_1=cfg_text`、`uncond_2=speaker_uncond / scale_2=cfg_speaker`。uncond_3 は未接続。
- **VoiceDesign（text+caption, speaker 無効）**:
  `uncond_1=text_uncond`、`uncond_2=caption_uncond`。uncond_3 未接続。
- **text-only**: `uncond_1=text_uncond` のみ（＝普通の CFG）。
- → 何本繋ぐかで本数が決まるので Guider 側に可変ロジック不要。

### `full_uncond` の扱い（A案 採用）
- `independent`（= 今回のN-way CFG）では**使わない**。全条件まとめて uncond にした1本で、`joint` モード専用。
- 出力端子としては**残す**（`encode_conditions` の副産物でコスト無し、将来 joint を試すとき配線するだけ）。
- 不要なら出力から削ってよい（B案）。

### duration の鉄則
- Duration Predictor は **KSampler ループの外**。`[Text]→[DurationPredictor]→N→[EmptyAudioLatent]` で長さ確定してからサンプリング。
- auto（duration 予測）/ manual（秒数指定→フレーム換算）の2モードが親切。`--duration-scale` 相当も。

---

## 5. ユーザー想像の答え合わせ（最終採点 95点）

| 想像 | 判定 |
|---|---|
| CFG は3条件を1個ずつ uncond した全4本 | ✅ 正解（`independent`・デフォルト）。ただし本数は可変 |
| 合成式 `v_full + w*(v_full - v_drop)` | ✅ 符号まで完全一致 |
| speaker/caption は別 key（pooling 的） | ✅ 正解。6引数で完全分離、KV cache 実在 |
| ref_latent（VAE Encode 通す） | ✅ 正解。patched DACVAE latent |
| CLIP 公式は諦め | ✅ 正解。LLM 由来の独自 tokenizer |
| uncond 化を見せる | ✅ 正解。ただし別ノードにせず Text Encode の出力端子として見せる（speaker は mask/noise 2モード） |
| 出力 = full 及び「全 uncond」 | ⚠️ N-way CFG では「各条件ごとの uncond」を使う。full_uncond は joint 用予備（A案で端子は残す） |
| Text Encode 1ノードに全エンコーダ集約 | ✅ encode_conditions が一括処理なので実物に忠実。speaker/caption encode・uncond maker は不要 |
| CFGGuider は cond+uncond×N+scale だけ | ✅ mode 切替不要。本数は配線で決まる汎用 N-way 線形 CFG |
| SamplerCustomAdvanced 軸の骨格 | ✅ 完璧 |
| Empty Latent が duration+text | ✅ 理想形（KSampler の外で N 確定） |

---

## 6. 次セッションでやること（TODO・シンプル化後）

1. **`Irodori Text Encode` ノード（最重要・入口を集約）**
   - 入力: CLIP（全エンコーダ）, text, ref_latent, caption。
   - 出力: cond / text_uncond / speaker_uncond / caption_uncond /（full_uncond）。
   - 中身は `rf.py` 前半の `encode_conditions(...)` ＋ uncond 生成（text/caption=ゼロ、speaker=mask or noise）を1ノードに移植。
   - 出力は ComfyUI `CONDITIONING` 互換の形（各 state/mask を extra dict に格納）に着地させる。
2. **`IrodoriCFGGuider`（汎用 N-way 線形 CFG）**
   - `comfy.samplers.CFGGuider` を継承し `predict_noise(x, timestep, ...)` を override。
   - ロジックはシンプル：`v = v_cond + Σ scale_i * (v_cond - v_uncond_i)`。uncond_i 未接続ならスキップ。
   - mode 切替・可変本数ロジックは持たない（本数は配線で決まる）。
   - drop 版を batch 化 → 1回 `forward_with_encoded_conditions` → chunk して合成、が効率的。
3. **MODEL ラッパー（`BaseModel` 継承）**
   - `forward_with_encoded_conditions` を `apply_model` にマッピング。
   - `ModelType.FLOW` + `ModelSamplingDiscreteFlow` 相当を設定（sigma↔t、BasicScheduler 整合）。
   - **KV cache の置き場所が最大の設計どころ**：`build_context_kv_cache` をループ外で1回計算し各 step で使い回すのを、
     SamplerCustomAdvanced の流れ（Guider/transformer_options 等）のどこに保持するか。
4. **出口/その他の自作ノード**
   - VAE Encode/Decode（音声用、画像前提スライス回避）、Empty Audio Latent（duration 連携）、Save Audio。
   - ※ speaker encode / caption encode / uncond maker は**不要**（Text Encode に統合済み）。
5. **要再確認（実装時）**
   - `model.py` の `forward_with_encoded_conditions` / `encode_conditions` / `build_context_kv_cache` の正確なシグネチャと戻り値形状。
   - `config.py` の `patched_latent_dim`、`use_speaker_condition_resolved`、`use_caption_condition` 等のフラグ。
   - `codec.py` の DACVAE encode/decode I/O（patch 化の単位、フレーム↔秒の換算）。
   - duration predictor の入出力（text のみか、speaker も見るか）。

---

## 7. 参考リンク
- リポジトリ: https://github.com/Aratako/Irodori-TTS
- rf.py（CFG サンプリング本体）: https://github.com/Aratako/Irodori-TTS/blob/main/irodori_tts/rf.py
- base モデル: https://huggingface.co/Aratako/Irodori-TTS-500M-v3
- VoiceDesign モデル: https://huggingface.co/Aratako/Irodori-TTS-500M-v2-VoiceDesign
- パラメータガイド: https://github.com/Aratako/Irodori-TTS/blob/main/docs/parameters.md
- ComfyUI ModelSamplingFlux 参考: https://comfyui.dev/docs/guides/nodes/modelsamplingflux/
- 参考: ComfyUI-CapitanFlowMatch（rectified flow 用 sampler/scheduler）: https://github.com/capitan01R/ComfyUI-CapitanFlowMatch
