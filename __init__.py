"""
Irodori-TTS ComfyUI integration.

Registers 10 custom nodes (V3 schema) that connect Irodori-TTS
(RF-DiT Japanese TTS) to ComfyUI's sampling pipeline.
"""
import os

import folder_paths
from typing_extensions import override
from comfy_api.latest import ComfyExtension, io

from .core.irodori_import import ensure_irodori_importable

ensure_irodori_importable()

# models/speaker_embeddings/*.safetensors (speaker inversion embeddings)
_speaker_embed_dir = os.path.join(folder_paths.models_dir, "speaker_embeddings")
os.makedirs(_speaker_embed_dir, exist_ok=True)
folder_paths.add_model_folder_path("speaker_embeddings", _speaker_embed_dir, is_default=True)
folder_paths.folder_names_and_paths["speaker_embeddings"][1].add(".safetensors")

from .nodes.checkpoint_loader import IrodoriCheckpointLoader  # noqa: E402
from .nodes.lora_loader import IrodoriLoraLoader  # noqa: E402
from .nodes.speaker_embed_loader import IrodoriSpeakerEmbedLoader  # noqa: E402
from .nodes.speaker_encode import IrodoriSpeakerEncode  # noqa: E402
from .nodes.embed_merge import IrodoriSpeakerEmbedMerge  # noqa: E402
from .nodes.text_encode import IrodoriTextEncode  # noqa: E402
from .nodes.guider import IrodoriCFGGuider  # noqa: E402
from .nodes.latent import IrodoriEmptyLatent  # noqa: E402
from .nodes.scheduler import IrodoriSwayScheduler  # noqa: E402
from .nodes.postprocess import IrodoriTrimTail  # noqa: E402


class IrodoriTTSExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [
            IrodoriCheckpointLoader,
            IrodoriLoraLoader,
            IrodoriSpeakerEmbedLoader,
            IrodoriSpeakerEncode,
            IrodoriSpeakerEmbedMerge,
            IrodoriTextEncode,
            IrodoriCFGGuider,
            IrodoriEmptyLatent,
            IrodoriSwayScheduler,
            IrodoriTrimTail,
        ]


async def comfy_entrypoint() -> IrodoriTTSExtension:
    return IrodoriTTSExtension()
