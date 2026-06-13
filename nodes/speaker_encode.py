"""
IrodoriSpeakerEncode — reference LATENT → SPEAKER_EMBED.

Runs the checkpoint's speaker encoder on a reference latent (from VAEEncodeAudio)
and outputs the resulting speaker_state as a SPEAKER_EMBED. This makes
reference-audio voice cloning a first-class SPEAKER_EMBED source: it can be
saved, merged (IrodoriSpeakerEmbedMerge), and fed to IrodoriTextEncode /
IrodoriEmptyLatent exactly like a loaded inversion embedding.

Token count scales with reference length (~25 tokens/sec for the V3 codec),
unlike the fixed-size inversion embeddings.
"""
from __future__ import annotations

from comfy_api.latest import io

from ..core.conditioning import encode_speaker_from_latent
from .types import SpeakerEmbed


class IrodoriSpeakerEncode(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="IrodoriSpeakerEncode",
            display_name="Irodori Speaker Encode",
            category="Irodori-TTS",
            description="Encode a reference LATENT (from VAEEncodeAudio) into a SPEAKER_EMBED via the speaker encoder. Same result as conditioning on the reference directly, but composable/saveable.",
            inputs=[
                io.Model.Input("model"),
                io.Latent.Input("ref_latent"),
            ],
            outputs=[
                SpeakerEmbed.Output(display_name="speaker_embed"),
            ],
        )

    @classmethod
    def execute(cls, model, ref_latent: dict) -> io.NodeOutput:
        embedding = encode_speaker_from_latent(model, ref_latent)  # (tokens, speaker_dim)
        return io.NodeOutput({"embedding": embedding})
