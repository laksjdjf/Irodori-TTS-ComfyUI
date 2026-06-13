"""
IrodoriSpeakerEmbedLoader — loads a speaker inversion embedding.

Any .safetensors in models/speaker_embeddings/ is accepted; validity is
checked by content (the 'speaker_embedding' key), not by filename suffix.

The embedding bypasses the reference-latent speaker encoder: it is passed to
encode_conditions(speaker_state_override=...) as-is.
"""
from __future__ import annotations

import folder_paths
import torch
from comfy_api.latest import io

from .types import SpeakerEmbed


class IrodoriSpeakerEmbedLoader(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="IrodoriSpeakerEmbedLoader",
            display_name="Load Irodori Speaker Embedding",
            category="Irodori-TTS",
            description="Load a speaker inversion embedding from models/speaker_embeddings/.",
            inputs=[
                io.Combo.Input(
                    "embed_name",
                    options=folder_paths.get_filename_list("speaker_embeddings"),
                    tooltip="models/speaker_embeddings/*.safetensors",
                ),
            ],
            outputs=[
                SpeakerEmbed.Output(display_name="speaker_embed"),
            ],
        )

    @classmethod
    def execute(cls, embed_name: str) -> io.NodeOutput:
        from safetensors.torch import load_file
        from irodori_tts.speaker_inversion import normalize_speaker_inversion_payload

        path = folder_paths.get_full_path("speaker_embeddings", embed_name)
        raw = load_file(path, device="cpu")
        embedding = normalize_speaker_inversion_payload(raw)["speaker_embedding"]  # (tokens, speaker_dim)
        # canonical SPEAKER_EMBED storage: CPU float32 (see core/conditioning.py)
        return io.NodeOutput({"embedding": embedding.to(device="cpu", dtype=torch.float32)})
