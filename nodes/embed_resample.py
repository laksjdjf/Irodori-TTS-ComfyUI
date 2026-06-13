"""
IrodoriSpeakerEmbedResample — adaptive-average-pool a SPEAKER_EMBED along the
token axis to a fixed token count.

Reference-audio embeddings have ~25 tokens/sec, so a few seconds gives hundreds
of tokens. In a joint-softmax cross-attention the aggregate attention mass each
source gets is roughly proportional to its token count, so concatenating a
315-token reference with a 16-token inversion embedding lets the reference drown
the other out. Pooling each source to a common token count equalizes the blend
(and shrinks bloated reference embeddings). Speaker identity is quasi-stationary
in time, so averaging adjacent tokens preserves timbre while dropping only fine
temporal detail.

No value normalization here on purpose: the encoder output is already RMSNorm'd
with a learned per-dim weight (speaker_norm), so forcing unit RMS would undo that
scaling and push the embedding off-distribution.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from comfy_api.latest import io

from .types import SpeakerEmbed


def adaptive_pool_tokens(emb: torch.Tensor, target: int) -> torch.Tensor:
    """(S, dim) -> (target, dim) by adaptive average pooling over the token axis.
    No-op when target >= S (can't synthesize new tokens)."""
    s = emb.shape[0]
    if target >= s:
        return emb
    x = emb.transpose(0, 1).unsqueeze(0)          # (1, dim, S)
    x = F.adaptive_avg_pool1d(x, target)          # (1, dim, target)
    return x.squeeze(0).transpose(0, 1).contiguous()  # (target, dim)


class IrodoriSpeakerEmbedResample(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="IrodoriSpeakerEmbedResample",
            display_name="Irodori Speaker Embedding Resample",
            category="Irodori-TTS",
            description="Adaptive-average-pool a SPEAKER_EMBED to a fixed token count. Use before Merge to equalize token counts (balanced blend) or to shrink long reference-audio embeddings.",
            inputs=[
                SpeakerEmbed.Input("speaker_embed"),
                io.Int.Input(
                    "target_tokens", default=16, min=1, max=4096,
                    tooltip="Output token count. No-op if >= the current count.",
                ),
                io.Boolean.Input(
                    "keep_first_token", default=False,
                    tooltip="Protect token 0 (the prepended mean/summary token of encoder-derived embeddings) and pool only the rest. Leave off for loaded inversion embeddings.",
                ),
            ],
            outputs=[
                SpeakerEmbed.Output(display_name="speaker_embed"),
            ],
        )

    @classmethod
    def execute(cls, speaker_embed: dict, target_tokens: int,
                keep_first_token: bool) -> io.NodeOutput:
        emb = speaker_embed["embedding"].to(device="cpu", dtype=torch.float32)

        if keep_first_token and emb.shape[0] >= 2:
            head = emb[:1]
            body = adaptive_pool_tokens(emb[1:], max(1, target_tokens - 1))
            out = torch.cat([head, body], dim=0)
        else:
            out = adaptive_pool_tokens(emb, target_tokens)

        return io.NodeOutput({"embedding": out.contiguous()})
