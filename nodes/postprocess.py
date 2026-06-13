"""
Post-processing nodes matching the official inference pipeline:

IrodoriTrimTail  — cut the flat "finished speaking" tail from a sampled latent
                   (official trim_tail / find_flattening_point, on by default upstream)
IrodoriWatermark — embed the SilentCipher watermark in generated audio
                   (the official pipeline always applies this; keep it in your
                   workflow so outputs stay machine-identifiable as AI-generated)
"""
from __future__ import annotations

import math

import torch
from comfy_api.latest import io

from ..core.latents import comfy_to_irodori, irodori_to_comfy


class IrodoriTrimTail(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="IrodoriTrimTail",
            display_name="Irodori Trim Tail",
            category="Irodori-TTS",
            description="Detect where the model finished speaking (flat latent tail) and crop the latent there. Place between the sampler and VAEDecodeAudio.",
            inputs=[
                io.Vae.Input("vae"),
                io.Latent.Input("samples"),
                io.Int.Input("window_size", default=20, min=1, max=200),
                io.Float.Input("std_threshold", default=0.05, min=0.0, max=1.0, step=0.005),
                io.Float.Input("mean_threshold", default=0.1, min=0.0, max=1.0, step=0.005),
            ],
            outputs=[
                io.Latent.Output(display_name="latent"),
            ],
        )

    @classmethod
    def execute(cls, vae, samples: dict, window_size: int,
                std_threshold: float, mean_threshold: float) -> io.NodeOutput:
        from irodori_tts.inference_runtime import find_flattening_point
        from irodori_tts.codec import patchify_latent, unpatchify_latent

        lat = samples["samples"]  # (B, D, 1, T_p)
        patch = vae.model_cfg.latent_patch_size
        latent_dim = vae.codec.latent_dim

        z = unpatchify_latent(comfy_to_irodori(lat), patch, latent_dim)  # (B, T, latent_dim)
        total = z.shape[1]

        # batch latents must stay uniform length — trim to the latest flattening point
        cut = 0
        for i in range(z.shape[0]):
            cut = max(cut, find_flattening_point(
                z[i].float().cpu(),
                window_size=window_size,
                std_threshold=std_threshold,
                mean_threshold=mean_threshold,
            ))

        if cut <= 0 or cut >= total:
            return io.NodeOutput(samples)

        # keep patch alignment by extending the cut with original frames
        # (the extra <patch frames are already past the speech end)
        keep = min(total, math.ceil(cut / patch) * patch)
        z = z[:, :keep]
        print(f"[Irodori-TTS] trim tail: {total} -> {keep} latent frames")

        out = dict(samples)
        out["samples"] = irodori_to_comfy(patchify_latent(z, patch))
        return io.NodeOutput(out)


# module-level singleton — ComfyUI locks node classes during execution,
# so the cache cannot live in a class attribute
_WATERMARKER = None


def _get_watermarker():
    global _WATERMARKER
    if _WATERMARKER is None:
        import comfy.model_management
        from irodori_tts.watermark import SilentCipherWatermarker

        device = comfy.model_management.get_torch_device()
        _WATERMARKER = SilentCipherWatermarker(device=str(device))
    return _WATERMARKER


class IrodoriWatermark(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="IrodoriWatermark",
            display_name="Irodori Watermark (SilentCipher)",
            category="Irodori-TTS",
            description="Embed the inaudible SilentCipher watermark, matching the official pipeline (which always applies it). Place before SaveAudio. Passes audio through with a warning if silentcipher is not installed.",
            inputs=[
                io.Audio.Input("audio"),
            ],
            outputs=[
                io.Audio.Output(display_name="audio"),
            ],
        )

    @classmethod
    def execute(cls, audio: dict) -> io.NodeOutput:
        wm = _get_watermarker()
        if not wm.ready:
            print("[Irodori-TTS] warning: SilentCipher is unavailable; "
                  "generated audio was NOT watermarked.")
            return io.NodeOutput(audio)

        waveform = audio["waveform"].detach().to(device="cpu", dtype=torch.float32)  # (B, C, T)
        sample_rate = int(audio["sample_rate"])
        encoded = wm.encode_batch(
            [waveform[i] for i in range(waveform.shape[0])],
            sample_rate=sample_rate,
        )
        out = torch.stack([e.to(torch.float32) for e in encoded], dim=0)
        return io.NodeOutput({"waveform": out, "sample_rate": sample_rate})
