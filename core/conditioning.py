"""
Shared conditioning helpers used by IrodoriTextEncode and IrodoriEmptyLatent.

CONDITIONING payload format (one dict per ComfyUI cond entry):
  {"text_state", "text_mask", "speaker_state", "speaker_mask",
   "caption_state", "caption_mask"}
States are float tensors; masks are bool tensors and must stay bool
(SDPA treats float masks as additive bias, not as padding masks).
"""
from __future__ import annotations

from typing import NamedTuple

import torch

import comfy.model_management

from .latents import comfy_to_irodori

COND_KEYS = (
    "text_state", "text_mask",
    "speaker_state", "speaker_mask",
    "caption_state", "caption_mask",
)


def pack_cond(text_state, text_mask, speaker_state, speaker_mask,
              caption_state, caption_mask) -> list:
    return [[None, {
        "text_state": text_state,
        "text_mask": text_mask,
        "speaker_state": speaker_state,
        "speaker_mask": speaker_mask,
        "caption_state": caption_state,
        "caption_mask": caption_mask,
    }]]


def unpack_cond(cond: list) -> dict:
    return cond[0][1]


def empty_ref(model_cfg, device: torch.device, dtype: torch.dtype):
    """Zero ref latent + all-False mask — same as Irodori's no_ref=True."""
    ref_len = max(1, int(model_cfg.speaker_patch_size))
    ref_lat = torch.zeros(
        1, ref_len, model_cfg.latent_dim * model_cfg.latent_patch_size,
        device=device, dtype=dtype,
    )
    ref_msk = torch.zeros(1, ref_len, dtype=torch.bool, device=device)
    return ref_lat, ref_msk


def empty_caption_tokens(device: torch.device):
    """Dummy caption tokens — the model requires them even when caption is empty."""
    cap_ids = torch.zeros(1, 1, dtype=torch.long, device=device)
    cap_msk = torch.zeros(1, 1, dtype=torch.bool, device=device)
    return cap_ids, cap_msk


class EncodedConditions(NamedTuple):
    text_state: torch.Tensor
    text_mask: torch.Tensor
    speaker_state: torch.Tensor | None
    speaker_mask: torch.Tensor | None
    caption_state: torch.Tensor | None
    caption_mask: torch.Tensor | None
    has_real_speaker: bool  # speaker condition backed by a ref audio / inversion


def ref_latent_to_irodori(ref_latent: dict, model_cfg, device, dtype):
    """ComfyUI LATENT dict → (latent in encode_conditions space, mask)."""
    from irodori_tts.codec import patchify_latent

    ref = ref_latent["samples"]
    # encode_conditions expects latent-patched space: (B, T, latent_dim*latent_patch_size)
    if ref.ndim == 4:
        # ComfyUI sampling format (B, D, 1, T_p) — already latent-patched
        ref = comfy_to_irodori(ref)
    else:
        # raw codec latent (B, T_lat, latent_dim)
        ref = patchify_latent(ref, model_cfg.latent_patch_size)
    ref = ref.to(device=device, dtype=dtype)
    msk = torch.ones(ref.shape[0], ref.shape[1], dtype=torch.bool, device=device)
    return ref, msk


def encode_speaker_from_latent(model, ref_latent: dict) -> torch.Tensor:
    """
    Run the speaker encoder on a reference LATENT and return the speaker_state
    (tokens, speaker_dim) — after speaker_norm + the prepended mean token, i.e.
    the exact tensor the DiT consumes. This is a drop-in SPEAKER_EMBED: it is
    fed back through speaker_state_override, which skips encoder/norm/prepend,
    so the result is identical to conditioning directly on the reference latent.
    """
    from irodori_tts.model import patch_sequence_with_mask

    comfy.model_management.load_models_gpu([model])
    w = model.model            # IrodoriModelWrapper
    device = model.load_device
    irodori = w.diffusion_model  # TextToLatentRFDiT
    model_cfg = w.model_cfg

    if not model_cfg.use_speaker_condition_resolved:
        raise ValueError("This checkpoint has speaker conditioning disabled.")
    if getattr(irodori, "speaker_encoder", None) is None:
        raise ValueError("This checkpoint has no speaker encoder (speaker-inversion only).")

    ref, msk = ref_latent_to_irodori(ref_latent, model_cfg, device, w.dtype)

    with torch.no_grad():
        ref_p, msk_p = patch_sequence_with_mask(ref, msk, model_cfg.speaker_patch_size)
        state = irodori.speaker_encoder(ref_p, msk_p)
        state = irodori.speaker_norm(state)
        state, _ = irodori._prepend_masked_mean_token(state, msk_p)

    return state[0].contiguous()  # (tokens, speaker_dim) — first reference item


def encode_text_conditions(
    model,
    text: str,
    *,
    speaker_embed: dict | None = None,
    caption: str = "",
    speaker_uncond_mode: str = "mask",
) -> EncodedConditions:
    """
    Normalize + tokenize + encode_conditions on a loaded MODEL (ModelPatcher).

    speaker_embed: SPEAKER_EMBED dict {"embedding": Tensor (tokens, speaker_dim)}
                   from IrodoriSpeakerEmbedLoader / IrodoriSpeakerEncode /
                   IrodoriSpeakerEmbedMerge. Passed via speaker_state_override.
                   None → no reference (empty-ref / no_ref fallback).
    """
    from irodori_tts.text_normalization import normalize_text

    comfy.model_management.load_models_gpu([model])
    w = model.model            # IrodoriModelWrapper
    device = model.load_device
    irodori = w.diffusion_model  # TextToLatentRFDiT
    model_cfg = w.model_cfg

    text = normalize_text(text)
    text_ids, text_mask = w.tokenizer.batch_encode([text], max_length=w.max_text_len)
    text_ids = text_ids.to(device)
    text_mask = text_mask.to(device)

    # --- speaker condition: a single SPEAKER_EMBED, fed via override ---
    spk_override = None
    if speaker_embed is not None:
        if not model_cfg.use_speaker_condition_resolved:
            raise ValueError(
                "This checkpoint has speaker conditioning disabled; disconnect speaker_embed."
            )
        # (tokens, speaker_dim) — batch expansion, dtype cast and dim check
        # happen inside encode_conditions
        spk_override = speaker_embed["embedding"].to(device=device)

    # When speaker conditioning is enabled but no embed is given (and the model
    # has no baked-in inversion), pass the empty/no_ref reference.
    ref_lat = ref_msk = None
    has_speaker_inversion = getattr(irodori, "speaker_inversion", None) is not None
    if model_cfg.use_speaker_condition_resolved and not has_speaker_inversion and spk_override is None:
        ref_lat, ref_msk = empty_ref(model_cfg, device, w.dtype)

    # --- caption tokens ---
    cap_ids = cap_msk = None
    if model_cfg.use_caption_condition:
        caption_text = caption.strip()
        if caption_text and w.caption_tokenizer is not None:
            cap_ids, cap_msk = w.caption_tokenizer.batch_encode(
                [caption_text], max_length=w.max_text_len)
            cap_ids = cap_ids.to(device)
            cap_msk = cap_msk.to(device)
        else:
            cap_ids, cap_msk = empty_caption_tokens(device)

    with torch.no_grad():
        encoded = irodori.encode_conditions(
            text_input_ids=text_ids,
            text_mask=text_mask,
            ref_latent=ref_lat,
            ref_mask=ref_msk,
            caption_input_ids=cap_ids,
            caption_mask=cap_msk,
            speaker_state_override=spk_override,
            speaker_mask_override=None,
            speaker_uncond_mode=speaker_uncond_mode,
        )

    has_real_speaker = model_cfg.use_speaker_condition_resolved and (
        has_speaker_inversion or speaker_embed is not None
    )
    return EncodedConditions(*encoded, has_real_speaker=has_real_speaker)
