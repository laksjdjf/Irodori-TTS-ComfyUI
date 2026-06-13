"""
SilentCipher watermarking, applied during decode to match the official pipeline.

The official inference_runtime watermarks every decoded waveform with no opt-out
(it only skips, with a warning, when the silentcipher package is missing). We
mirror that here: IrodoriVAEWrapper.decode() always runs apply_watermark().

The SilentCipher model is loaded lazily and cached per device.
"""
from __future__ import annotations

import torch

_WATERMARKERS: dict[str, object] = {}
_WARNED = False


def _get_watermarker(device: str):
    wm = _WATERMARKERS.get(device)
    if wm is None:
        from irodori_tts.watermark import SilentCipherWatermarker

        wm = SilentCipherWatermarker(device=device)
        _WATERMARKERS[device] = wm
    return wm


def apply_watermark(audio: torch.Tensor, codec) -> torch.Tensor:
    """
    audio: (B, 1, T_audio) on the codec device.
    Returns the same shape, watermarked (or unchanged + a one-time warning
    when silentcipher is unavailable — matching the official behavior).
    """
    global _WARNED
    wm = _get_watermarker(str(codec.device))
    if not wm.ready:
        if not _WARNED:
            print("[Irodori-TTS] warning: SilentCipher is unavailable; "
                  "generated audio is NOT watermarked. "
                  "Install with: pip install git+https://github.com/SesameAILabs/silentcipher.git")
            _WARNED = True
        return audio

    items = [audio[i].detach().to(device="cpu", dtype=torch.float32)
             for i in range(audio.shape[0])]
    encoded = wm.encode_batch(items, sample_rate=int(codec.sample_rate))  # list of (1, T) cpu f32
    out = torch.stack(encoded, dim=0)
    return out.to(device=audio.device, dtype=audio.dtype)
