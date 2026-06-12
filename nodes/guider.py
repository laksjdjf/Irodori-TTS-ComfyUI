"""
IrodoriCFGGuider node + IrodoriGuider class.

IrodoriGuider is a standalone guider compatible with SamplerCustomAdvanced.

Call chain (from SamplerCustomAdvanced):
  1. guider.sample(noise, latent, sampler, sigmas, ...)
  2.   sampler.sample(guider_as_model_wrap, sigmas, ...)   ← KSAMPLER does noise_scaling
  3.     KSamplerX0Inpaint(guider)(x, sigma, ...)
  4.       guider(x, sigma, ...)  ←  __call__
  5.         guider.predict_noise(x, sigma, ...)  → x0

KSAMPLER accesses guider.inner_model.model_sampling, so inner_model must be set
before sampler.sample() is called.
"""
from __future__ import annotations

import torch

import comfy.model_management
from comfy_api.latest import io

from ..core.conditioning import unpack_cond
from ..core.latents import comfy_to_irodori, irodori_to_comfy


def _prepare_cond(cond: list, device: torch.device, dtype: torch.dtype, batch_size: int) -> list:
    """
    Move tensors inside a CONDITIONING list to device, cast float states to dtype
    (bool masks must stay bool — SDPA treats float masks as additive bias),
    and tile batch-1 conditions up to the latent batch size.
    """
    result = []
    for item in cond:
        moved = {}
        for k, v in item[1].items():
            if isinstance(v, torch.Tensor):
                v = v.to(device=device, dtype=dtype) if v.is_floating_point() else v.to(device=device)
                if batch_size > 1 and v.shape[0] == 1:
                    v = v.repeat(batch_size, *([1] * (v.ndim - 1)))
            moved[k] = v
        result.append([item[0], moved])
    return result


def _batch_conds(cond_list: list[list]) -> dict:
    """Concatenate N CONDITIONING dicts along dim-0."""
    dicts = [unpack_cond(c) for c in cond_list]
    merged = {}
    for k in dicts[0].keys():
        values = [d.get(k) for d in dicts]
        non_none = [v for v in values if v is not None]
        if not non_none:
            merged[k] = None
            continue
        ref = non_none[0]
        merged[k] = torch.cat(
            [v if v is not None else torch.zeros_like(ref) for v in values], dim=0
        )
    return merged


class IrodoriGuider:
    """
    Drop-in GUIDER for SamplerCustomAdvanced, wired for Irodori RF-DiT N-way CFG.
    """

    def __init__(self, model_patcher):
        self.model_patcher = model_patcher
        # Set by sample() before KSAMPLER sees it
        self.inner_model = None  # IrodoriModelWrapper — KSAMPLER reads .inner_model.model_sampling

        self.cond: list | None = None
        self.unconds: list[list] = []
        self.scales: list[float] = []
        self.cfg_min_t: float = 0.5
        self.cfg_max_t: float = 1.0

        # Prepared in sample(), fixed for the whole run
        self._single: dict | None = None      # cond-only payload
        self._merged: dict | None = None      # [cond + unconds] batched payload

    # ---- configuration ----

    def set_conds(self, cond):
        self.cond = cond

    def add_uncond(self, uncond, scale: float):
        self.unconds.append(uncond)
        self.scales.append(float(scale))

    def set_cfg(self, cfg_min_t: float, cfg_max_t: float):
        self.cfg_min_t = float(cfg_min_t)
        self.cfg_max_t = float(cfg_max_t)

    # ---- callable interface (KSamplerX0Inpaint calls guider(x, sigma)) ----

    def __call__(self, x, sigma, model_options=None, seed=None):
        return self.predict_noise(x, sigma, model_options or {}, seed)

    # ---- predict_noise: called once per diffusion step ----

    def predict_noise(self, x, sigma, model_options=None, seed=None):
        """
        x:     (B, D, 1, T)   4D ComfyUI latent
        sigma: Tensor          t ∈ [0, 1]
        returns (B, D, 1, T)  estimated x0 (denoised)
        """
        t_val = float(sigma.flatten()[0])
        model_sampling = self.inner_model.model_sampling
        irodori = self.inner_model.diffusion_model  # TextToLatentRFDiT
        dtype = self.inner_model.dtype

        x_lit = comfy_to_irodori(x).to(dtype=dtype)  # (B, D, 1, T) → (B, T, D)
        t_lit = sigma[:x_lit.shape[0]].to(dtype=dtype)

        use_cfg = self._merged is not None and (self.cfg_min_t <= t_val <= self.cfg_max_t)

        with torch.no_grad():
            if not use_cfg:
                kw = self._single
                context_kv = self.inner_model.get_context_kv(
                    ("irodori_guider", "single"),
                    kw["text_state"], kw.get("speaker_state"), kw.get("caption_state"))
                v = irodori.forward_with_encoded_conditions(
                    x_t=x_lit, t=t_lit,
                    text_state=kw["text_state"],
                    text_mask=kw["text_mask"],
                    speaker_state=kw.get("speaker_state"),
                    speaker_mask=kw.get("speaker_mask"),
                    caption_state=kw.get("caption_state"),
                    caption_mask=kw.get("caption_mask"),
                    context_kv_cache=context_kv,
                ).float()
            else:
                kw = self._merged
                N = 1 + len(self.unconds)
                context_kv = self.inner_model.get_context_kv(
                    ("irodori_guider", "merged"),
                    kw["text_state"], kw.get("speaker_state"), kw.get("caption_state"))
                v_out = irodori.forward_with_encoded_conditions(
                    x_t=x_lit.repeat(N, 1, 1),
                    t=t_lit.repeat(N),
                    text_state=kw["text_state"],
                    text_mask=kw["text_mask"],
                    speaker_state=kw.get("speaker_state"),
                    speaker_mask=kw.get("speaker_mask"),
                    caption_state=kw.get("caption_state"),
                    caption_mask=kw.get("caption_mask"),
                    context_kv_cache=context_kv,
                )  # (B*N, T, D)
                chunks = v_out.chunk(N, dim=0)
                v = chunks[0].float()
                for scale, chunk in zip(self.scales, chunks[1:]):
                    v = v + scale * (chunks[0].float() - chunk.float())

        v_comfy = irodori_to_comfy(v)  # (B, T, D) → (B, D, 1, T)
        # x0 = x_t - t * v  (RF denoised estimate)
        return model_sampling.calculate_denoised(sigma, v_comfy, x)

    # ---- sample: entry point called by SamplerCustomAdvanced ----

    def sample(self, noise, latent_image, sampler, sigmas, denoise_mask=None,
               callback=None, disable_pbar=False, seed=None):
        device = comfy.model_management.get_torch_device()
        comfy.model_management.load_models_gpu([self.model_patcher])

        # Expose inner_model so KSAMPLER can access .inner_model.model_sampling
        self.inner_model = self.model_patcher.model  # IrodoriModelWrapper

        dtype = self.inner_model.dtype
        batch_size = noise.shape[0]
        cond_dev = _prepare_cond(self.cond, device, dtype, batch_size)
        unconds_dev = [_prepare_cond(u, device, dtype, batch_size) for u in self.unconds]

        # Conditioning is static for the whole run: batch the CFG bundle once.
        # Context KV is cached on the wrapper (shared with the standard path);
        # this guider doesn't go through ModelPatcher.pre_run, so clear manually.
        self._single = unpack_cond(cond_dev)
        self._merged = _batch_conds([cond_dev] + unconds_dev) if unconds_dev else None
        self.inner_model.clear_context_kv_cache()

        sigmas = sigmas.to(device)
        noise = noise.to(device)
        latent_image = latent_image.to(device)
        extra_args = {"model_options": {}, "seed": seed}

        # KSAMPLER.sample(model_wrap=self, ...) handles noise_scaling internally
        samples = sampler.sample(
            self,
            sigmas,
            extra_args,
            callback,
            noise,
            latent_image,
            denoise_mask,
            disable_pbar,
        )

        out = self.inner_model.process_latent_out(samples.float())
        self.inner_model.clear_context_kv_cache()
        self.inner_model = None
        self._single = None
        self._merged = None
        return out


# ---------------------------------------------------------------------------
# ComfyUI node
# ---------------------------------------------------------------------------

class IrodoriCFGGuider(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        scale_kw = {"min": 0.0, "max": 100.0, "step": 0.01}
        t_kw = {"min": 0.0, "max": 1.0, "step": 0.01}
        return io.Schema(
            node_id="IrodoriCFGGuider",
            display_name="Irodori CFG Guider",
            category="Irodori-TTS",
            description="N-way CFG guider: applies separate scales per uncond, only within [cfg_min_t, cfg_max_t].",
            inputs=[
                io.Model.Input("model"),
                io.Conditioning.Input("cond"),
                io.Float.Input("cfg_min_t", default=0.5, **t_kw),
                io.Float.Input("cfg_max_t", default=1.0, **t_kw),
                io.Conditioning.Input("uncond_1", optional=True),
                io.Float.Input("scale_1", default=3.0, optional=True, **scale_kw),
                io.Conditioning.Input("uncond_2", optional=True),
                io.Float.Input("scale_2", default=5.0, optional=True, **scale_kw),
                io.Conditioning.Input("uncond_3", optional=True),
                io.Float.Input("scale_3", default=3.0, optional=True, **scale_kw),
            ],
            outputs=[
                io.Guider.Output(display_name="guider"),
            ],
        )

    @classmethod
    def execute(
        cls,
        model,
        cond,
        cfg_min_t: float,
        cfg_max_t: float,
        uncond_1=None, scale_1: float = 3.0,
        uncond_2=None, scale_2: float = 5.0,
        uncond_3=None, scale_3: float = 3.0,
    ) -> io.NodeOutput:
        guider = IrodoriGuider(model)
        guider.set_conds(cond)
        guider.set_cfg(cfg_min_t, cfg_max_t)

        for uncond, scale in [(uncond_1, scale_1), (uncond_2, scale_2), (uncond_3, scale_3)]:
            if uncond is not None and scale > 0.0:
                guider.add_uncond(uncond, scale)

        return io.NodeOutput(guider)
