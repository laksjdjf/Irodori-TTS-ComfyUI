"""
IrodoriSpeakerEmbedSampleTokens — randomly select a subset of tokens from a
SPEAKER_EMBED.

Unlike average pooling (which blends adjacent tokens and shrinks their magnitude
after speaker_norm), random subsampling keeps real encoder tokens intact — full
magnitude, on-distribution. Useful for cutting token count (e.g. balancing a
concat blend) without washing the voice out, or for picking a random facet of a
long reference embedding. Attention over speaker context is permutation-
invariant, so sampled token order does not matter; indices are kept sorted for
tidiness.
"""
from __future__ import annotations

import torch
from comfy_api.latest import io

from .types import SpeakerEmbed


class IrodoriSpeakerEmbedSampleTokens(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="IrodoriSpeakerEmbedSampleTokens",
            display_name="Irodori Speaker Embedding Sample Tokens",
            category="Irodori-TTS",
            description="Randomly keep N tokens from a SPEAKER_EMBED (without replacement). Keeps real tokens intact (no averaging/washout). No-op if num_tokens >= the current count.",
            inputs=[
                SpeakerEmbed.Input("speaker_embed"),
                io.Int.Input("num_tokens", default=16, min=1, max=4096),
                io.Int.Input(
                    "seed", default=0, min=0, max=0xFFFFFFFFFFFFFFFF,
                    control_after_generate=True,
                    tooltip="Selection seed (reproducible).",
                ),
                io.Boolean.Input(
                    "keep_first_token", default=False,
                    tooltip="Always keep token 0 (the prepended mean/summary token of encoder-derived embeddings) and sample the rest.",
                ),
            ],
            outputs=[
                SpeakerEmbed.Output(display_name="speaker_embed"),
            ],
        )

    @classmethod
    def execute(cls, speaker_embed: dict, num_tokens: int, seed: int,
                keep_first_token: bool) -> io.NodeOutput:
        emb = speaker_embed["embedding"].to(device="cpu", dtype=torch.float32)
        s = emb.shape[0]

        if num_tokens >= s:
            return io.NodeOutput({"embedding": emb.contiguous()})

        g = torch.Generator().manual_seed(int(seed))
        if keep_first_token and s >= 2:
            rest = 1 + torch.randperm(s - 1, generator=g)[: max(0, num_tokens - 1)]
            idx = torch.cat([torch.tensor([0]), rest])
        else:
            idx = torch.randperm(s, generator=g)[:num_tokens]
        idx = idx.sort().values

        return io.NodeOutput({"embedding": emb[idx].contiguous()})
