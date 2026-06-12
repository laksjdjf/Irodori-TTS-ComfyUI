"""
Shape conversion between ComfyUI latents and Irodori latents.

ComfyUI LATENT:  (B, D, 1, T)  — 4D, channels-first
Irodori latent:  (B, T, D)     — 3D, time-major
"""
from __future__ import annotations

import torch


def comfy_to_irodori(latent: torch.Tensor) -> torch.Tensor:
    """(B, D, 1, T) → (B, T, D)"""
    return latent.squeeze(2).transpose(1, 2).contiguous()


def irodori_to_comfy(latent: torch.Tensor) -> torch.Tensor:
    """(B, T, D) → (B, D, 1, T)"""
    return latent.transpose(1, 2).unsqueeze(2).contiguous()
