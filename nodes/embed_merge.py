"""
IrodoriSpeakerEmbedMerge — combine multiple speaker inversion embeddings, with
optional per-input weights.

Speaker embeddings are (tokens, speaker_dim) tensors consumed as cross-attention
context (K/V) in the DiT. The key projection is RMSNorm'd (magnitude removed),
so:
  - average:  (w_a*e_a + w_b*e_b + ...) / sum(w)  — true convex blend; the weight
              is a voice-interpolation slider. Needs equal token counts.
  - concat:   cat([w_i * e_i]) along the token dim — each embedding's tokens are
              kept; because K is RMSNorm'd, scaling values leaves attention
              weights unchanged and scales each voice's V contribution linearly,
              so the weight acts as a per-voice contribution gain. Only needs
              equal speaker_dim.

All weights default to 1.0, so the unweighted result matches plain concat/mean.
"""
from __future__ import annotations

import torch
from comfy_api.latest import io

from .types import SpeakerEmbed

_W = {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.05}


class IrodoriSpeakerEmbedMerge(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="IrodoriSpeakerEmbedMerge",
            display_name="Irodori Speaker Embedding Merge",
            category="Irodori-TTS",
            description="Merge speaker inversion embeddings with optional weights. concat = token-dim join (weight = per-voice contribution gain, recommended); average = weighted convex blend (needs equal token counts).",
            inputs=[
                SpeakerEmbed.Input("speaker_embed_a"),
                SpeakerEmbed.Input("speaker_embed_b"),
                io.Combo.Input("mode", options=["concat", "average"]),
                io.Float.Input("weight_a", optional=True, **_W),
                io.Float.Input("weight_b", optional=True, **_W),
                SpeakerEmbed.Input("speaker_embed_c", optional=True),
                io.Float.Input("weight_c", optional=True, **_W),
                SpeakerEmbed.Input("speaker_embed_d", optional=True),
                io.Float.Input("weight_d", optional=True, **_W),
            ],
            outputs=[
                SpeakerEmbed.Output(display_name="speaker_embed"),
            ],
        )

    @classmethod
    def execute(cls, speaker_embed_a: dict, speaker_embed_b: dict, mode: str,
                weight_a: float = 1.0, weight_b: float = 1.0,
                speaker_embed_c: dict | None = None, weight_c: float = 1.0,
                speaker_embed_d: dict | None = None, weight_d: float = 1.0) -> io.NodeOutput:
        pairs = [
            (speaker_embed_a, weight_a),
            (speaker_embed_b, weight_b),
            (speaker_embed_c, weight_c),
            (speaker_embed_d, weight_d),
        ]
        items = [(e["embedding"], float(w)) for e, w in pairs if e is not None]
        embeds = [e for e, _ in items]
        weights = [w for _, w in items]

        dims = {e.shape[-1] for e in embeds}
        if len(dims) != 1:
            raise ValueError(f"speaker_dim mismatch across embeddings: {sorted(dims)}")

        if mode == "concat":
            merged = torch.cat([w * e for e, w in items], dim=0)  # (sum_tokens, speaker_dim)
        else:  # average — weighted convex blend
            tokens = {e.shape[0] for e in embeds}
            if len(tokens) != 1:
                raise ValueError(
                    f"average requires equal token counts, got {sorted(tokens)}; use concat instead."
                )
            total = sum(weights)
            if total <= 1e-8:
                raise ValueError("average weights sum to zero; set at least one weight > 0.")
            merged = sum(w * e for e, w in items) / total  # (tokens, speaker_dim)

        return io.NodeOutput({"embedding": merged.contiguous()})
