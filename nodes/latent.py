"""
IrodoriEmptyLatent — creates a zero-filled LATENT in (B, D, 1, T) shape.
"""
from __future__ import annotations

import math

import torch
from comfy_api.latest import io

from ..core.conditioning import encode_text_conditions
from .types import SpeakerEmbed

# Used when seconds=0 and the checkpoint has no duration predictor.
FALLBACK_SECONDS = 30.0


class IrodoriEmptyLatent(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="IrodoriEmptyLatent",
            display_name="Irodori Empty Latent",
            category="Irodori-TTS",
            description="Create an empty audio latent; duration is predicted from text when seconds=0.",
            inputs=[
                io.Model.Input("model"),
                io.Vae.Input("vae"),
                io.String.Input("text", multiline=True, default=""),
                io.Float.Input(
                    "seconds",
                    default=0.0, min=0.0, max=300.0, step=0.1,
                    tooltip="0 = auto-predict duration from text",
                ),
                io.Float.Input("duration_scale", default=1.0, min=0.1, max=5.0, step=0.05),
                io.Int.Input("batch_size", default=1, min=1, max=16),
                SpeakerEmbed.Input(
                    "speaker_embed", optional=True,
                    tooltip="Improves duration prediction when using a speaker inversion embedding",
                ),
            ],
            outputs=[
                io.Latent.Output(display_name="latent"),
            ],
        )

    @classmethod
    def execute(cls, model, vae, text: str, seconds: float, duration_scale: float,
                batch_size: int, speaker_embed: dict | None = None) -> io.NodeOutput:
        from irodori_tts.text_normalization import normalize_text
        text = normalize_text(text)

        w = model.model        # IrodoriModelWrapper
        model_cfg = w.model_cfg
        codec = vae.codec
        hop_length = int(codec.model.hop_length)
        sample_rate = codec.sample_rate
        patch_size = model_cfg.latent_patch_size
        patched_dim = model_cfg.patched_latent_dim

        has_duration_predictor = getattr(w.diffusion_model, "duration_predictor", None) is not None

        if seconds > 0.0:
            latent_frames = math.ceil(seconds * sample_rate / hop_length)
        elif has_duration_predictor:
            latent_frames = cls._predict_duration(model, text, duration_scale, speaker_embed)
        else:
            latent_frames = math.ceil(FALLBACK_SECONDS * sample_rate / hop_length)

        patched_steps = math.ceil(latent_frames / patch_size)
        latent = torch.zeros(batch_size, patched_dim, 1, patched_steps, dtype=torch.float32)
        return io.NodeOutput({"samples": latent, "sample_rate": sample_rate})

    @classmethod
    def _predict_duration(cls, model, text: str, duration_scale: float,
                          speaker_embed: dict | None = None) -> int:
        from irodori_tts.duration import build_duration_features

        w = model.model        # IrodoriModelWrapper
        device = model.load_device
        irodori = w.diffusion_model  # TextToLatentRFDiT
        model_cfg = w.model_cfg

        enc = encode_text_conditions(model, text, speaker_embed=speaker_embed)

        with torch.no_grad():
            duration_features = build_duration_features(
                [text],
                token_counts=enc.text_mask.sum(dim=1),
                max_text_len=w.max_text_len,
                has_speaker=[enc.has_real_speaker],
            ).to(device)
            pred_log_frames = irodori.predict_duration_log_frames(
                text_state=enc.text_state, text_mask=enc.text_mask,
                speaker_state=enc.speaker_state, speaker_mask=enc.speaker_mask,
                caption_state=enc.caption_state, caption_mask=enc.caption_mask,
                duration_features=duration_features,
                has_speaker=torch.tensor([enc.has_real_speaker], dtype=torch.bool, device=device),
                has_caption=torch.tensor(
                    [model_cfg.use_caption_condition], dtype=torch.bool, device=device
                ) if model_cfg.use_caption_condition else None,
            )

        pred_frames = torch.expm1(pred_log_frames).float().mean().item()
        return max(1, int(round(pred_frames * duration_scale)))
