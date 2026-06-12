#!/usr/bin/env python3
"""
Convert an Irodori-TTS PEFT LoRA checkpoint into a ComfyUI-loadable LoRA file
(for the stock LoraLoaderModelOnly node).

Note: the IrodoriLoraLoader node can apply PEFT checkpoints directly without
this conversion — use this script only if you prefer the stock loader / the
models/loras dropdown.

Usage:
  python convert_peft_lora.py <peft_checkpoint_dir | adapter.safetensors> [output.safetensors]

Default output: <ComfyUI>/models/loras/<checkpoint_dir_name>.safetensors when run
inside ComfyUI/custom_nodes, otherwise ./<name>.safetensors.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from core.peft_lora import read_lora_alpha, resolve_adapter_paths, convert_peft_state_dict  # noqa: E402


def main() -> None:
    if len(sys.argv) < 2:
        sys.exit(__doc__)

    from safetensors.torch import load_file, save_file

    src = Path(sys.argv[1]).expanduser()
    adapter_path, config_path = resolve_adapter_paths(src)
    lora_alpha = read_lora_alpha(config_path)
    if lora_alpha is None:
        print("warning: adapter_config.json not found, assuming alpha = rank (scale 1.0)")
    else:
        print(f"adapter_config: lora_alpha={lora_alpha}")

    if src.is_dir():
        default_name = src.parent.name if src.name.startswith("checkpoint") else src.name
    else:
        default_name = src.stem

    if len(sys.argv) >= 3:
        out_path = Path(sys.argv[2]).expanduser()
    else:
        loras_dir = Path(__file__).resolve().parent.parent.parent / "models" / "loras"
        out_dir = loras_dir if loras_dir.is_dir() else Path.cwd()
        out_path = out_dir / f"{default_name}.safetensors"

    raw = load_file(adapter_path, device="cpu")
    tensors = convert_peft_state_dict(raw, lora_alpha)
    if not tensors:
        sys.exit("error: nothing to convert")

    n_lora = sum(1 for k in tensors if k.endswith(".lora_A.weight"))
    n_set = sum(1 for k in tensors if k.endswith(".set_weight"))
    print(f"converted: {n_lora} LoRA modules, {n_set} full-weight (set) tensors")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(out_path), metadata={"source": "irodori-peft", "lora_alpha": str(lora_alpha or "")})
    print(f"saved: {out_path}")


if __name__ == "__main__":
    main()
