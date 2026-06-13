# Irodori-TTS-ComfyUI

[Irodori-TTS](https://github.com/Aratako/Irodori-TTS)（RF-DiT 日本語TTS）を ComfyUI の
`SamplerCustomAdvanced` パイプラインに接続するカスタムノード集。

## セットアップ

1. `irodori_tts` パッケージを使えるようにする（下記 2 方式のどちらか）
2. チェックポイント（例: `Irodori-TTS-V3.safetensors`）を `ComfyUI/models/checkpoints/` に置く
3. ComfyUI を再起動

### irodori_tts の解決方式

インポートは「① pip インストール済み → ② `IRODORI_TTS_PATH` → ③ `~/Irodori-TTS` →
④ custom_nodes 隣接ディレクトリ」の順で解決される。

**方式A（推奨）: リポジトリを置いてパス参照**

```bash
git clone https://github.com/Aratako/Irodori-TTS.git ~/Irodori-TTS
# または任意の場所に置いて IRODORI_TTS_PATH=/path/to/Irodori-TTS
```

ComfyUI の Python 環境には触れない。実行に必要な依存（peft / safetensors /
transformers / sentencepiece 等）が足りない場合は個別に pip install する。

**方式B: pip インストール**

```bash
pip install git+https://github.com/Aratako/Irodori-TTS.git
```

> ⚠️ Irodori-TTS は `torch>=2.10` を要求し、wandb / gradio / datasets など
> 学習用の重い依存も含む。**ComfyUI の venv に入れると torch が強制
> アップグレードされ環境が壊れる可能性がある**ため、依存バージョンを
> 確認できる場合のみ推奨。同じ理由で本リポジトリは requirements.txt での
> 自動インストールを意図的にしていない。

コーデック（デフォルト: `Aratako/Semantic-DACVAE-Japanese-32dim`）は初回実行時に
Hugging Face から自動ダウンロードされる。

## ノード一覧

| ノード | 役割 |
|---|---|
| `IrodoriCheckpointLoader` | チェックポイント読み込み → MODEL + VAE |
| `IrodoriLoraLoader` | PEFT 形式 LoRA をディレクトリ直指定で適用（変換不要） |
| `IrodoriSpeakerEmbedLoader` | speaker inversion 埋め込み（`models/speaker_embeddings/*.safetensors`）→ SPEAKER_EMBED |
| `IrodoriTextEncode` | テキスト＋話者条件 → 4種 CONDITIONING（cond / text_uncond / speaker_uncond / caption_uncond） |
| `IrodoriCFGGuider` | N-way CFG の GUIDER を作成（cfg_min_t〜cfg_max_t の範囲でのみCFG適用） |
| `IrodoriEmptyLatent` | duration 予測（seconds=0 で自動）→ ゼロ LATENT |

話者条件は `ref_latent`（参照音声）か `speaker_embed`（inversion 埋め込み）の**どちらか一方**を
`IrodoriTextEncode` に接続する（両方接続するとエラー）。`speaker_embed` は
`IrodoriEmptyLatent` にも接続でき、duration 予測の精度が上がる。

残りは公式ノードを使う: `LoadAudio` / `VAEEncodeAudio` / `VAEDecodeAudio` /
`SaveAudio` / `RandomNoise` / `KSamplerSelect` / `BasicScheduler` / `SamplerCustomAdvanced`

## ワークフロー

```
[IrodoriCheckpointLoader] → MODEL / VAE

[LoadAudio] → [VAEEncodeAudio] → LATENT(ref)        ← 話者参照（VoiceClone時）
[IrodoriSpeakerEmbedLoader] → SPEAKER_EMBED          ← inversion埋め込み使用時（refと排他）

[IrodoriTextEncode] MODEL + text + (LATENT(ref) | SPEAKER_EMBED) + caption
    → cond / text_uncond / speaker_uncond / caption_uncond

[IrodoriEmptyLatent] MODEL + VAE + text + seconds=0 (+ SPEAKER_EMBED) → LATENT(empty)

[IrodoriCFGGuider] MODEL + cond
    + uncond_1=text_uncond    / scale_1=3.0
    + uncond_2=speaker_uncond / scale_2=5.0
    → GUIDER

[RandomNoise] / [KSamplerSelect euler] / [BasicScheduler steps=40]

[SamplerCustomAdvanced] → [VAEDecodeAudio] → [SaveAudio]
```

### 標準ノードとの互換性（責任分解）

MODEL は ComfyUI の BaseModel 互換インターフェース
（`extra_conds` / `apply_model` / `memory_required`）を実装しているため、
ガイダンス・サンプリングループ・cond のバッチング・メモリ管理は ComfyUI 標準
スタックに委譲される。カスタム側の責務は **Irodori の forward と条件エンコードのみ**。

使えるガイダンス構成:

| 構成 | 用途 |
|---|---|
| 素の `CFGGuider`（positive + negative + cfg） | 通常の 2-way CFG |
| `BasicGuider`（positive のみ） | CFG なし高速生成 |
| 通常の `KSampler` | positive/negative を直接接続する最短構成 |
| `IrodoriCFGGuider` | text / speaker / caption の uncond に**別々のスケール**を掛ける N-way CFG |

条件側の KV射影（各ブロックの wk/wv）はモデルラッパー側でキャッシュされ、
ランの最初の step で1回だけ計算して全 step で再利用する（`build_context_kv_cache`）。
**どのガイダンス構成でも同様に効く**ため、経路による速度差はない。
キャッシュはモデルロード（LoRAパッチ適用）後に構築され、ラン毎に破棄されるので
LoRA の付け外し・strength 変更とも正しく整合する
（hooks による step 途中の重み変更がある場合のみ自動でキャッシュを使わない）。

CFG を特定の時間範囲だけ有効にする（旧 `cfg_min_t` / `cfg_max_t` 相当）には、
negative に標準の `ConditioningSetTimestepRange` を挟む:

```
[IrodoriTextEncode] text_uncond → [ConditioningSetTimestepRange start=0.0 end=0.5]
    → [CFGGuider] negative      ← sigma ∈ [0.5, 1.0] の間だけ CFG が効く
```

cond と uncond は ComfyUI が自動で 1 回の forward にバッチするため、
標準経路でも速度ペナルティはない。

### LoRA

**方法1（推奨）: `IrodoriLoraLoader` で PEFT チェックポイントを直接指定（変換不要）**

```
[IrodoriCheckpointLoader] → MODEL → [Irodori LoRA Loader (PEFT)] → 以降のノードへ
    lora_path: ~/Irodori-TTS/outputs/<name>/checkpoint_final   ← ディレクトリかファイルを直接指定
    strength:  1.0
```

キーリネームと alpha（`adapter_config.json` から読む）の変換はメモリ内で行われ、
適用自体は ComfyUI 標準の `ModelPatcher.add_patches` 経由なので、
他の LoRA との重ねがけ・strength 調整・lowvram は公式ローダーと同じ挙動。

**方法2: 変換スクリプト + 公式 `LoraLoaderModelOnly`**

```bash
python convert_peft_lora.py ~/Irodori-TTS/outputs/<name>/checkpoint_final
# → ComfyUI/models/loras/<name>.safetensors に出力
```

models/loras のドロップダウンで選びたい場合はこちら。

> 素の PEFT ファイルを公式ローダーに直接入れることはできない。
> PEFT はスケール（alpha/r）を adapter_config.json に持ちテンソルファイル内に
> 情報が無いため強度が狂い、modules_to_save（duration_predictor のフル重み）も
> 公式の認識形式でないため捨てられる。

変換内容（両方式共通）:
- `lora_A` / `lora_B` をキーリネーム + alpha テンソル付与（PEFT と同じ alpha/r スケール）
- `modules_to_save`（duration_predictor 等のフル重み）は `set_weight` パッチとして変換
  （strength の影響を受けず常に完全適用される点は PEFT と同様の挙動）

### audio2audio（img2img相当）

`VAEEncodeAudio` の出力はサンプリング形式（4D LATENT）なので、そのまま
`SamplerCustomAdvanced` の `latent_image` に接続できる。
`BasicScheduler` の `denoise` を 1.0 未満（例: 0.5）にすると、入力音声の
構造を保ったまま部分的にリサンプリングされる。

```
[LoadAudio] → [VAEEncodeAudio] → LATENT(init) → [SamplerCustomAdvanced]
[BasicScheduler denoise=0.5] → SIGMAS
```

API形式のサンプルワークフロー: [irodori-tts.json](irodori-tts.json)

詳細設計は [Irodori-TTS_ComfyUI実装設計書.md](Irodori-TTS_ComfyUI実装設計書.md) を参照。

## 構成

ノードは ComfyUI の **V3 スキーマ**（`comfy_api.latest` の `io.ComfyNode` /
`define_schema` / `comfy_entrypoint`）で定義されている。

```
__init__.py                # ComfyExtension + comfy_entrypoint（V3登録）
convert_peft_lora.py       # PEFT LoRA → 公式ローダー用ファイル変換 CLI
core/
├── irodori_import.py      # irodori_tts パッケージのパス解決
├── latents.py             # (B,D,1,T) ⇔ (B,T,D) 変換
├── conditioning.py        # CONDITIONING パック / encode_conditions 共通処理
├── peft_lora.py           # PEFT → ComfyUI LoRA 形式変換
├── model_wrapper.py       # IrodoriModelSampling / IrodoriLatentFormat / IrodoriModelWrapper
└── loader.py              # チェックポイント読み込み + IrodoriVAEWrapper
nodes/
├── types.py                 # SPEAKER_EMBED カスタム型
├── checkpoint_loader.py     # IrodoriCheckpointLoader
├── lora_loader.py           # IrodoriLoraLoader (PEFT直接適用)
├── speaker_embed_loader.py  # IrodoriSpeakerEmbedLoader
├── text_encode.py           # IrodoriTextEncode
├── guider.py                # IrodoriCFGGuider + IrodoriGuider
└── latent.py                # IrodoriEmptyLatent
```
## テスト

モデル・GPU不要のスタブ回帰テスト（ComfyUI checkout と Python 環境のみ必要）:

```bash
python custom_nodes/Irodori-TTS-ComfyUI/tests/run_tests.py
```

## ライセンス

MIT License（Irodori-TTS 本体と同じ）。
