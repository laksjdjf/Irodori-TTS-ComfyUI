"""
PEFT LoRA checkpoint → ComfyUI patch-format conversion.

Used by the IrodoriLoraLoader node (in-memory, no conversion step) and by
the convert_peft_lora.py CLI (writes a file for the stock LoraLoaderModelOnly).

Key conversion:
  base_model.model.<mod>.lora_A.weight → diffusion_model.<mod>.lora_A.weight
  base_model.model.<mod>.lora_B.weight → diffusion_model.<mod>.lora_B.weight
  + diffusion_model.<mod>.alpha = lora_alpha   (ComfyUI scales alpha/r like PEFT)

modules_to_save (full weight replacements, e.g. duration_predictor):
  base_model.model.<name> → diffusion_model.<name minus .weight>.set_weight
"""
from __future__ import annotations

import json
from pathlib import Path

import torch

PEFT_PREFIX = "base_model.model."
OUT_PREFIX = "diffusion_model."
ADAPTER_STATE_NAME = "adapter_model.safetensors"
ADAPTER_CONFIG_NAME = "adapter_config.json"


def resolve_adapter_paths(path: str | Path) -> tuple[Path, Path | None]:
    """Accepts a PEFT checkpoint dir or a .safetensors file path."""
    src = Path(path).expanduser()
    if src.is_dir():
        adapter = src / ADAPTER_STATE_NAME
        config = src / ADAPTER_CONFIG_NAME
    else:
        adapter = src
        config = src.parent / ADAPTER_CONFIG_NAME
    if not adapter.is_file():
        raise FileNotFoundError(f"PEFT adapter not found: {adapter}")
    return adapter, (config if config.is_file() else None)


def read_lora_alpha(config_path: Path | None) -> float | None:
    if config_path is None:
        return None
    cfg = json.loads(config_path.read_text())
    alpha = cfg.get("lora_alpha")
    return float(alpha) if alpha is not None else None


def convert_peft_state_dict(
    raw: dict[str, torch.Tensor],
    lora_alpha: float | None,
) -> dict[str, torch.Tensor]:
    """
    Convert a raw PEFT state dict to ComfyUI LoRA format.
    Already-converted dicts (diffusion_model.* keys) pass through unchanged.

    lora_alpha=None omits alpha tensors → ComfyUI uses scale 1.0
    (correct only when alpha == r; pass the real value whenever known).
    """
    if any(k.startswith(OUT_PREFIX) for k in raw):
        return raw

    out: dict[str, torch.Tensor] = {}
    for key, tensor in raw.items():
        if not key.startswith(PEFT_PREFIX):
            continue
        name = key[len(PEFT_PREFIX):]

        if name.endswith(".lora_A.weight") or name.endswith(".lora_B.weight"):
            out[OUT_PREFIX + name] = tensor
            if name.endswith(".lora_A.weight") and lora_alpha is not None:
                module = name[: -len(".lora_A.weight")]
                out[f"{OUT_PREFIX}{module}.alpha"] = torch.tensor(float(lora_alpha))
        else:
            # modules_to_save: full weight replacement
            x = name[: -len(".weight")] if name.endswith(".weight") else name
            out[f"{OUT_PREFIX}{x}.set_weight"] = tensor
    return out


def load_peft_lora(path: str | Path) -> dict[str, torch.Tensor]:
    """Load a PEFT checkpoint (dir or file) as a ComfyUI-format LoRA dict."""
    from safetensors.torch import load_file

    adapter_path, config_path = resolve_adapter_paths(path)
    lora_alpha = read_lora_alpha(config_path)
    if config_path is None:
        print(
            f"[Irodori-TTS] warning: {ADAPTER_CONFIG_NAME} not found next to "
            f"{adapter_path.name}; assuming lora_alpha == r (scale 1.0)"
        )
    raw = load_file(adapter_path, device="cpu")
    converted = convert_peft_state_dict(raw, lora_alpha)
    if not converted:
        raise ValueError(f"No convertible LoRA keys found in {adapter_path}")
    return converted
