"""
IrodoriTrimTail — cut the flat "finished speaking" tail from a sampled latent
                  (official trim_tail / find_flattening_point, on by default
                  upstream). Place between the sampler and VAEDecodeAudio.

The SilentCipher watermark is NOT a node — to match the official pipeline it is
applied unconditionally inside IrodoriVAEWrapper.decode() (see core/watermark.py).
"""
from __future__ import annotations

import math

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
