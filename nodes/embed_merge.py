"""
IrodoriSpeakerEmbedMerge — combine multiple speaker inversion embeddings.

Speaker embeddings are (tokens, speaker_dim) tensors consumed as cross-attention
context (K/V) in the DiT. Attention is permutation-invariant and variable-length
over context tokens, so concatenating along the token dim is the natural way to
mix speakers: each embedding's learned tokens are preserved and the model attends
over the union.

  concat:  (16,768) + (16,768) -> (32,768)   — keeps both, model blends
  average: mean over inputs (needs equal token counts) — morph toward the middle
"""
from __future__ import annotations

import torch
from comfy_api.latest import io

from .types import SpeakerEmbed


class IrodoriSpeakerEmbedMerge(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="IrodoriSpeakerEmbedMerge",
            display_name="Irodori Speaker Embedding Merge",
            category="Irodori-TTS",
            description="Merge speaker inversion embeddings. concat = token-dim join (keeps both, recommended); average = mean (needs equal token counts).",
            inputs=[
                SpeakerEmbed.Input("speaker_embed_a"),
                SpeakerEmbed.Input("speaker_embed_b"),
                io.Combo.Input("mode", options=["concat", "average"]),
                SpeakerEmbed.Input("speaker_embed_c", optional=True),
                SpeakerEmbed.Input("speaker_embed_d", optional=True),
            ],
            outputs=[
                SpeakerEmbed.Output(display_name="speaker_embed"),
            ],
        )

    @classmethod
    def execute(cls, speaker_embed_a: dict, speaker_embed_b: dict, mode: str,
                speaker_embed_c: dict | None = None,
                speaker_embed_d: dict | None = None) -> io.NodeOutput:
        embeds = [e["embedding"] for e in
                  (speaker_embed_a, speaker_embed_b, speaker_embed_c, speaker_embed_d)
                  if e is not None]

        dims = {e.shape[-1] for e in embeds}
        if len(dims) != 1:
            raise ValueError(f"speaker_dim mismatch across embeddings: {sorted(dims)}")

        if mode == "concat":
            merged = torch.cat(embeds, dim=0)  # (sum_tokens, speaker_dim)
        else:  # average
            tokens = {e.shape[0] for e in embeds}
            if len(tokens) != 1:
                raise ValueError(
                    f"average requires equal token counts, got {sorted(tokens)}; use concat instead."
                )
            merged = torch.stack(embeds, dim=0).mean(dim=0)  # (tokens, speaker_dim)

        return io.NodeOutput({"embedding": merged.contiguous()})
