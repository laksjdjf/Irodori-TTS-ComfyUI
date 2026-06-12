"""
IrodoriCheckpointLoader — loads a checkpoint and outputs MODEL + VAE.
"""
from __future__ import annotations

import folder_paths
from comfy_api.latest import io

from ..core.loader import load_irodori_checkpoint


class IrodoriCheckpointLoader(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="IrodoriCheckpointLoader",
            display_name="Load Irodori Checkpoint",
            category="Irodori-TTS",
            description="Load an Irodori-TTS (RF-DiT) checkpoint as MODEL + audio VAE (DACVAE codec).",
            inputs=[
                io.Combo.Input("ckpt_name", options=folder_paths.get_filename_list("checkpoints")),
                io.String.Input("codec_repo", default="Aratako/Semantic-DACVAE-Japanese-32dim"),
                io.Combo.Input("device", options=["auto", "cuda", "cpu"]),
                io.Combo.Input("dtype", options=["bf16", "fp16", "fp32"]),
            ],
            outputs=[
                io.Model.Output(display_name="model"),
                io.Vae.Output(display_name="vae"),
            ],
        )

    @classmethod
    def execute(cls, ckpt_name: str, codec_repo: str, device: str, dtype: str) -> io.NodeOutput:
        ckpt_path = folder_paths.get_full_path("checkpoints", ckpt_name)
        model_patcher, vae = load_irodori_checkpoint(
            ckpt_path=ckpt_path,
            codec_repo=codec_repo,
            device_str=device,
            dtype_str=dtype,
        )
        return io.NodeOutput(model_patcher, vae)
