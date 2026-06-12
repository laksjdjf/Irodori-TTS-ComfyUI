"""
Load Irodori-TTS checkpoint → (MODEL, VAE) for ComfyUI.

Tokenizer, config, max_text_len are stored on IrodoriModelWrapper
so all nodes only need MODEL (+ VAE where codec info is required).
"""
from __future__ import annotations

from pathlib import Path

import torch

import comfy.model_management
import comfy.model_patcher

from .latents import comfy_to_irodori, irodori_to_comfy
from .model_wrapper import IrodoriModelWrapper


def _resolve_dtype(dtype_str: str, device: torch.device) -> torch.dtype:
    if dtype_str == "bf16":
        if device.type == "cuda" and torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return torch.float32
    if dtype_str == "fp16":
        return torch.float16
    return torch.float32


class IrodoriVAEWrapper:
    """
    Adapts DACVAECodec to the interface expected by VAEEncodeAudio / VAEDecodeAudio.

    Both encode() and decode() use the ComfyUI 4D sampling format
    (B, D, 1, T_patched) with D = latent_dim * latent_patch_size, so the
    encoded LATENT can go directly into SamplerCustomAdvanced (audio2audio)
    or into IrodoriTextEncode's ref_latent.

    VAEEncodeAudio calls:  vae.encode(waveform.movedim(1, -1))
      waveform: (B, C, T) → movedim → (B, T, C) passed to encode()
      returns:  (B, D, 1, T_patched)

    VAEDecodeAudio calls:  vae.decode(samples["samples"]).movedim(-1, 1)
      latent:  (B, D, 1, T_patched) ← ComfyUI LATENT
      returns: (B, T_audio, 1)  → movedim → (B, 1, T_audio) = AUDIO
    """

    def __init__(self, codec, model_cfg) -> None:
        self.codec = codec
        self.model_cfg = model_cfg
        self.audio_sample_rate = codec.sample_rate
        self.audio_sample_rate_output = codec.sample_rate

    def encode(self, waveform: torch.Tensor) -> torch.Tensor:
        from irodori_tts.codec import patchify_latent

        wav = waveform.movedim(-1, 1).float()           # (B, T, C) → (B, C, T)
        lat = self.codec.encode_waveform(wav, self.codec.sample_rate)   # (B, T_lat, latent_dim)
        lat = patchify_latent(lat, self.model_cfg.latent_patch_size)    # (B, T_p, D)
        return irodori_to_comfy(lat)                                    # (B, D, 1, T_p)

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        from irodori_tts.codec import unpatchify_latent

        x = comfy_to_irodori(latent)                                 # (B, D, 1, T) → (B, T, D)
        x = unpatchify_latent(x, self.model_cfg.latent_patch_size, self.codec.latent_dim)
        audio = self.codec.decode_latent(x)                          # → (B, 1, T_audio)
        return audio.movedim(1, -1)                                  # → (B, T_audio, 1)


def load_irodori_checkpoint(
    ckpt_path: str,
    codec_repo: str,
    device_str: str,
    dtype_str: str,
) -> tuple:
    """Returns (model_patcher, vae_wrapper)."""
    from irodori_tts.inference_runtime import _load_checkpoint_for_inference
    from irodori_tts.config import ModelConfig
    from irodori_tts.model import TextToLatentRFDiT
    from irodori_tts.tokenizer import PretrainedTextTokenizer
    from irodori_tts.codec import DACVAECodec

    if device_str == "auto":
        load_device = comfy.model_management.get_torch_device()
    else:
        load_device = torch.device(device_str)
    offload_device = comfy.model_management.unet_offload_device()
    dtype = _resolve_dtype(dtype_str, load_device)

    # --- model ---
    model_state, model_cfg_dict, train_cfg = _load_checkpoint_for_inference(Path(ckpt_path))
    model_cfg = ModelConfig(**model_cfg_dict)

    irodori_model = TextToLatentRFDiT(model_cfg)
    irodori_model.load_state_dict(model_state)
    irodori_model = irodori_model.to(device=load_device, dtype=dtype)
    irodori_model.eval()

    # --- tokenizers ---
    tokenizer = PretrainedTextTokenizer.from_pretrained(
        repo_id=model_cfg.text_tokenizer_repo,
        add_bos=bool(model_cfg.text_add_bos),
    )
    caption_tokenizer = None
    if model_cfg.use_caption_condition:
        caption_tokenizer = PretrainedTextTokenizer.from_pretrained(
            repo_id=model_cfg.caption_tokenizer_repo_resolved,
            add_bos=model_cfg.caption_add_bos_resolved,
        )

    max_text_len = 256
    if isinstance(train_cfg, dict):
        v = train_cfg.get("max_text_len")
        if isinstance(v, int) and v > 0:
            max_text_len = v

    wrapper = IrodoriModelWrapper(
        irodori_model=irodori_model,
        model_cfg=model_cfg,
        tokenizer=tokenizer,
        caption_tokenizer=caption_tokenizer,
        max_text_len=max_text_len,
        device=load_device,
        dtype=dtype,
    )
    size = comfy.model_management.module_size(irodori_model)
    model_patcher = comfy.model_patcher.ModelPatcher(
        model=wrapper,
        load_device=load_device,
        offload_device=offload_device,
        size=size,
    )

    # --- codec / VAE ---
    codec = DACVAECodec.load(
        repo_id=codec_repo,
        device=str(load_device),
        dtype=dtype,
    )
    if model_cfg.latent_dim != codec.latent_dim:
        raise ValueError(
            f"Latent dim mismatch: checkpoint={model_cfg.latent_dim}, codec={codec.latent_dim}"
        )
    vae = IrodoriVAEWrapper(codec, model_cfg)

    return model_patcher, vae
