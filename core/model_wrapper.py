"""
ComfyUI-compatible wrappers around Irodori-TTS internals.

IrodoriModelSampling – ModelSamplingDiscreteFlow + CONST (same pattern as Flux)
IrodoriLatentFormat  – minimal LatentFormat for audio latents (B, D, 1, T)
IrodoriModelWrapper  – BaseModel-compatible nn.Module that ModelPatcher can manage

IrodoriModelWrapper implements the interface ComfyUI's standard sampling stack
expects from a model (extra_conds / extra_conds_shapes / apply_model /
memory_required), so the stock CFGGuider, BasicGuider, KSampler and
SamplerCustomAdvanced all work. ComfyUI owns guidance, cond batching,
timestep ranges and memory management; this wrapper only owns the
Irodori forward pass.
"""
from __future__ import annotations

import torch
import torch.nn as nn

import comfy.conds
import comfy.model_management
from comfy.model_sampling import ModelSamplingDiscreteFlow, CONST

from .conditioning import COND_KEYS
from .latents import comfy_to_irodori, irodori_to_comfy


class IrodoriModelSampling(ModelSamplingDiscreteFlow, CONST):
    """
    RF (Rectified Flow) model sampling — identical pattern to Flux.

    ModelSamplingDiscreteFlow provides:
      - sigmas buffer (ascending, sigma_min→sigma_max) → simple_scheduler works correctly
      - sigma_min / sigma_max properties
      - percent_to_sigma  (with shift=1.0: linear, 0%→1.0, 100%→0.0)

    CONST provides:
      - calculate_denoised(sigma, v, x) = x - v*sigma  (RF velocity)
      - noise_scaling(sigma, n, lat)   = sigma*n + (1-sigma)*lat
      - inverse_noise_scaling          = lat / (1-sigma)
    """

    def __init__(self) -> None:
        super().__init__()
        # shift=1.0 → linear schedule (no SNR shift), multiplier=1000 for timestep embedding
        self.set_parameters(shift=1.0, multiplier=1000)


class IrodoriLatentFormat:
    """
    Minimal LatentFormat for Irodori audio latents stored as (B, D, 1, T).

    latent_channels is set at runtime from the checkpoint's patched_latent_dim.
    """

    latent_dimensions: int = 2
    spacial_downscale_ratio: float = 1.0
    temporal_downscale_ratio: float = 1.0
    taesd_decoder_name = None
    latent_rgb_factors = None
    latent_rgb_factors_bias = None

    def __init__(self, patched_latent_dim: int) -> None:
        self.latent_channels = patched_latent_dim

    def process_in(self, latent: torch.Tensor) -> torch.Tensor:
        return latent

    def process_out(self, latent: torch.Tensor) -> torch.Tensor:
        return latent


class _IrodoriModelConfig:
    """Minimal model_config stub so comfy.lora.model_lora_keys_unet works."""
    unet_config: dict = {}
    custom_operations = None


class IrodoriModelWrapper(nn.Module):
    """
    Thin nn.Module wrapper so ComfyUI's ModelPatcher can manage Irodori.

    The inner model is exposed as `diffusion_model` so ComfyUI's generic
    LoRA key mapping (diffusion_model.* state-dict keys) applies and the
    stock LoraLoaderModelOnly works.
    """

    def __init__(
        self,
        irodori_model: nn.Module,
        model_cfg,
        tokenizer,
        caption_tokenizer,
        max_text_len: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        super().__init__()
        self.diffusion_model = irodori_model
        self.model_cfg = model_cfg
        self.model_config = _IrodoriModelConfig()
        self.tokenizer = tokenizer
        self.caption_tokenizer = caption_tokenizer
        self.max_text_len = max_text_len
        self.device = device
        self.dtype = dtype

        self.model_sampling = IrodoriModelSampling()
        self.latent_format = IrodoriLatentFormat(model_cfg.patched_latent_dim)

        self.model_loaded_weight_memory: int = 0
        self.lowvram_patch_counter: int = 0
        self.model_lowvram: bool = False
        self.current_weight_patches_uuid = None
        self.model_offload_buffer_memory: int = 0
        self._context_kv_cache: dict = {}
        self.current_patcher = None

    # ModelPatcher sets current_patcher in pre_run() (every sampling run) and
    # clears it in cleanup(); piggyback on that to invalidate the context KV
    # cache so it never survives weight repatching (LoRA strength changes etc.).
    @property
    def current_patcher(self):
        return self._current_patcher

    @current_patcher.setter
    def current_patcher(self, value):
        self._current_patcher = value
        self._context_kv_cache.clear()

    # ---- context KV cache (t-independent condition projections) ----

    def get_context_kv(self, key, text_state, speaker_state=None, caption_state=None):
        """Build the per-block context K/V once per (run, conditioning) and reuse."""
        entry = self._context_kv_cache.get(key)
        if entry is None:
            entry = self.diffusion_model.build_context_kv_cache(
                text_state, speaker_state, caption_state)
            self._context_kv_cache[key] = entry
        return entry

    def clear_context_kv_cache(self) -> None:
        self._context_kv_cache.clear()

    def get_model_object(self, name: str):
        if name == "latent_format":
            return self.latent_format
        return getattr(self, name, None)

    def get_dtype(self) -> torch.dtype:
        return self.dtype

    def get_dtype_inference(self) -> torch.dtype:
        return self.dtype

    def process_latent_in(self, latent: torch.Tensor) -> torch.Tensor:
        return latent

    def process_latent_out(self, latent: torch.Tensor) -> torch.Tensor:
        return latent

    def memory_required(self, input_shape=None, cond_shapes=None, **kwargs) -> int:
        return comfy.model_management.module_size(self.diffusion_model)

    def scale_latent_inpaint(self, **kwargs) -> torch.Tensor:
        raise NotImplementedError("Inpainting not supported for Irodori-TTS")

    # ---- BaseModel-compatible conditioning interface ----
    # process_conds() calls extra_conds() with the raw CONDITIONING payload;
    # wrapping tensors in CONDRegular lets ComfyUI handle device moves,
    # batch repetition and cond/uncond batching (calc_cond_batch).

    def extra_conds(self, **kwargs) -> dict:
        out = {}
        for k in COND_KEYS:
            v = kwargs.get(k)
            if isinstance(v, torch.Tensor):
                out[k] = comfy.conds.CONDRegular(v)
        return out

    def extra_conds_shapes(self, **kwargs) -> dict:
        return {}

    def apply_model(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        text_state: torch.Tensor | None = None,
        text_mask: torch.Tensor | None = None,
        speaker_state: torch.Tensor | None = None,
        speaker_mask: torch.Tensor | None = None,
        caption_state: torch.Tensor | None = None,
        caption_mask: torch.Tensor | None = None,
        transformer_options={},
        **kwargs,
    ) -> torch.Tensor:
        """
        Standard ComfyUI entry point (calc_cond_batch).

        x: (B, D, 1, T) latent, t: sigma (B,) with t ∈ [0, 1].
        Returns the denoised estimate x0 = x - t*v.
        """
        if text_state is None:
            raise RuntimeError(
                "Irodori MODEL requires CONDITIONING from IrodoriTextEncode "
                "(text_state is missing — standard text encoders cannot be used)."
            )

        def _state(s):
            return s.to(dtype=self.dtype) if s is not None else None

        def _mask(m):
            # SDPA treats float masks as additive bias — masks must be bool
            return m.to(dtype=torch.bool) if m is not None else None

        x_lit = comfy_to_irodori(x).to(dtype=self.dtype)
        t_lit = t.to(dtype=self.dtype)
        ts, ss, cs = _state(text_state), _state(speaker_state), _state(caption_state)

        # Reuse the t-independent context K/V across steps. The cond uuids
        # identify the batched conditioning within a run (conds are processed
        # once per run); hooks can repatch weights mid-run, so skip then.
        context_kv = None
        uuids = transformer_options.get("uuids")
        hooks_active = getattr(self._current_patcher, "current_hooks", None)
        if uuids is not None and not hooks_active:
            key = (
                tuple(uuids),
                tuple(ts.shape),
                tuple(ss.shape) if ss is not None else None,
                tuple(cs.shape) if cs is not None else None,
            )
            context_kv = self.get_context_kv(key, ts, ss, cs)

        with torch.no_grad():
            v = self.diffusion_model.forward_with_encoded_conditions(
                x_t=x_lit,
                t=t_lit,
                text_state=ts,
                text_mask=_mask(text_mask),
                speaker_state=ss,
                speaker_mask=_mask(speaker_mask),
                caption_state=cs,
                caption_mask=_mask(caption_mask),
                context_kv_cache=context_kv,
            )

        v_comfy = irodori_to_comfy(v.float())
        return self.model_sampling.calculate_denoised(t, v_comfy, x).float()
