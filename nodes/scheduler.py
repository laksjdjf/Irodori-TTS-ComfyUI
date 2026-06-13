"""
IrodoriSwayScheduler — F5-TTS-style Sway Sampling t-schedule as SIGMAS.

Same formula as the official sampler (irodori_tts.rf, t_schedule_mode="sway"):
    u' = u + c * (cos(pi/2 * u) + u - 1),  sigmas = (1 - u') * denoise
Negative sway_coeff densifies the noise side (early steps), where prosody and
phoneme layout are decided. sway_coeff=0 reproduces the linear schedule.
Drop-in replacement for BasicScheduler in front of SamplerCustomAdvanced.
"""
from __future__ import annotations

import math

import torch
from comfy_api.latest import io


def sway_sigmas(steps: int, sway_coeff: float, denoise: float = 1.0) -> torch.Tensor:
    u = torch.linspace(0.0, 1.0, steps + 1, dtype=torch.float32)
    u = u + float(sway_coeff) * (torch.cos(0.5 * math.pi * u) + u - 1.0)
    u = u.clamp(0.0, 1.0)
    sigmas = (1.0 - u) * float(denoise)
    if not bool(torch.all(sigmas[:-1] > sigmas[1:]).item()):
        raise ValueError(
            "sway schedule must be strictly decreasing; adjust steps or sway_coeff."
        )
    return sigmas


class IrodoriSwayScheduler(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="IrodoriSwayScheduler",
            display_name="Irodori Sway Scheduler",
            category="Irodori-TTS",
            description="F5-TTS-style Sway Sampling schedule. Negative sway_coeff spends more steps on the noise side (early structure); 0 = linear.",
            inputs=[
                io.Int.Input("steps", default=20, min=1, max=1000),
                io.Float.Input(
                    "sway_coeff", default=-1.0, min=-1.0, max=1.0, step=0.01,
                    tooltip="Negative densifies early (noise-side) steps; 0 = linear schedule",
                ),
                io.Float.Input("denoise", default=1.0, min=0.0, max=1.0, step=0.01),
            ],
            outputs=[
                io.Sigmas.Output(display_name="sigmas"),
            ],
        )

    @classmethod
    def execute(cls, steps: int, sway_coeff: float, denoise: float) -> io.NodeOutput:
        if denoise <= 0.0:
            return io.NodeOutput(torch.FloatTensor([]))
        return io.NodeOutput(sway_sigmas(steps, sway_coeff, denoise))
