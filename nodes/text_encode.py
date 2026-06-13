"""
IrodoriTextEncode — tokenizes text + speaker condition → 4 × CONDITIONING outputs.
"""
from __future__ import annotations

import torch
from comfy_api.latest import io

from ..core.conditioning import encode_text_conditions, pack_cond
from .types import SpeakerEmbed


class IrodoriTextEncode(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="IrodoriTextEncode",
            display_name="Irodori Text Encode",
            category="Irodori-TTS",
            description="Encode text (+ speaker embedding + caption) into cond and uncond CONDITIONINGs.",
            inputs=[
                io.Model.Input("model"),
                io.String.Input("text", multiline=True, default=""),
                io.Combo.Input("speaker_uncond_mode", options=["mask", "noise"]),
                SpeakerEmbed.Input(
                    "speaker_embed", optional=True,
                    tooltip="Speaker embedding (IrodoriSpeakerEncode from reference audio, a loaded inversion embedding, or a merge). Leave unconnected for no speaker reference.",
                ),
                io.String.Input("caption", multiline=True, default="", optional=True),
            ],
            outputs=[
                io.Conditioning.Output("cond", display_name="cond"),
                io.Conditioning.Output("text_uncond", display_name="text_uncond"),
                io.Conditioning.Output("speaker_uncond", display_name="speaker_uncond"),
                io.Conditioning.Output("caption_uncond", display_name="caption_uncond"),
            ],
        )

    @classmethod
    def execute(
        cls,
        model,
        text: str,
        speaker_uncond_mode: str,
        speaker_embed: dict | None = None,
        caption: str = "",
    ) -> io.NodeOutput:
        enc = encode_text_conditions(
            model, text,
            speaker_embed=speaker_embed,
            caption=caption,
            speaker_uncond_mode=speaker_uncond_mode,
        )
        ts_c, tm_c = enc.text_state, enc.text_mask
        ss_c, sm_c = enc.speaker_state, enc.speaker_mask
        cs_c, cm_c = enc.caption_state, enc.caption_mask

        # --- uncond tensors ---
        ts_u = torch.zeros_like(ts_c)
        tm_u = torch.zeros_like(tm_c)

        if ss_c is not None:
            if speaker_uncond_mode == "noise":
                ss_u = torch.randn_like(ss_c) * ss_c.std().clamp_min(1e-6)
                sm_u = torch.ones_like(sm_c)
            else:
                ss_u = torch.zeros_like(ss_c)
                sm_u = torch.zeros_like(sm_c)
        else:
            ss_u, sm_u = ss_c, sm_c

        if cs_c is not None:
            cs_u = torch.zeros_like(cs_c)
            cm_u = torch.zeros_like(cm_c)
        else:
            cs_u, cm_u = cs_c, cm_c

        cond           = pack_cond(ts_c, tm_c, ss_c, sm_c, cs_c, cm_c)
        text_uncond    = pack_cond(ts_u, tm_u, ss_c, sm_c, cs_c, cm_c)
        speaker_uncond = pack_cond(ts_c, tm_c, ss_u, sm_u, cs_c, cm_c)
        caption_uncond = pack_cond(ts_c, tm_c, ss_c, sm_c, cs_u, cm_u)

        return io.NodeOutput(cond, text_uncond, speaker_uncond, caption_uncond)
