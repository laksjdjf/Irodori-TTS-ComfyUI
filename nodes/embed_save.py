"""
IrodoriSpeakerEmbedSave — save a SPEAKER_EMBED to models/speaker_embeddings/.

Writes the canonical {"speaker_embedding": (tokens, dim)} safetensors that
IrodoriSpeakerEmbedLoader reads, so a reference-encoded / merged / sampled
embedding can be persisted and reused. Output node, but also passes the
embedding through so it can be saved inline mid-graph.
"""
from __future__ import annotations

import os

import folder_paths
import torch
from comfy_api.latest import io
from irodori_tts.speaker_inversion import SPEAKER_EMBEDDING_KEY

from .types import SpeakerEmbed


def resolve_save_path(filename: str) -> str:
    """Sanitize → models/speaker_embeddings/<name>.safetensors (no path traversal)."""
    name = os.path.basename(filename.strip()) or "speaker_embed"
    if not name.endswith(".safetensors"):
        name += ".safetensors"
    save_dir = folder_paths.get_folder_paths("speaker_embeddings")[0]
    return os.path.join(save_dir, name)


class IrodoriSpeakerEmbedSave(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="IrodoriSpeakerEmbedSave",
            display_name="Save Irodori Speaker Embedding",
            category="Irodori-TTS",
            description="Save a SPEAKER_EMBED to models/speaker_embeddings/ (reloadable with IrodoriSpeakerEmbedLoader). Overwrites an existing file with the same name.",
            is_output_node=True,
            inputs=[
                SpeakerEmbed.Input("speaker_embed"),
                io.String.Input("filename", default="speaker_embed"),
            ],
            outputs=[
                SpeakerEmbed.Output(display_name="speaker_embed"),
            ],
        )

    @classmethod
    def execute(cls, speaker_embed: dict, filename: str) -> io.NodeOutput:
        from safetensors.torch import save_file

        emb = speaker_embed["embedding"].to(device="cpu", dtype=torch.float32).contiguous()
        path = resolve_save_path(filename)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        save_file({SPEAKER_EMBEDDING_KEY: emb}, path)
        print(f"[Irodori-TTS] saved speaker embedding: {path}  shape={tuple(emb.shape)}")
        return io.NodeOutput(speaker_embed)
