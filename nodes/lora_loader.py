"""
IrodoriLoraLoader — applies a PEFT LoRA checkpoint directly (no conversion step).

Point lora_path at the PEFT output directory (the one containing
adapter_model.safetensors + adapter_config.json) or at the safetensors file.
Keys and alpha are converted in memory, then applied through ComfyUI's
standard ModelPatcher.add_patches — stacking with other LoRAs, strength,
and lowvram handling all behave like the stock loader.
"""
from __future__ import annotations

import os

import comfy.lora
from comfy_api.latest import io

from ..core.peft_lora import load_peft_lora, resolve_adapter_paths


class IrodoriLoraLoader(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="IrodoriLoraLoader",
            display_name="Irodori LoRA Loader (PEFT)",
            category="Irodori-TTS",
            description="Apply an Irodori-TTS PEFT LoRA checkpoint directly (dir or safetensors), no conversion needed.",
            inputs=[
                io.Model.Input("model"),
                io.String.Input(
                    "lora_path",
                    default="",
                    tooltip="PEFT checkpoint dir (with adapter_model.safetensors) or a safetensors file path",
                ),
                io.Float.Input(
                    "strength",
                    default=1.0, min=-10.0, max=10.0, step=0.01,
                    tooltip="LoRA strength (full-weight set patches like duration_predictor are always applied as-is)",
                ),
            ],
            outputs=[
                io.Model.Output(display_name="model"),
            ],
        )

    @classmethod
    def fingerprint_inputs(cls, model, lora_path: str, strength: float):
        try:
            adapter_path, config_path = resolve_adapter_paths(lora_path.strip())
            stamp = [str(adapter_path), os.path.getmtime(adapter_path), strength]
            if config_path is not None:
                stamp.append(os.path.getmtime(config_path))
            return tuple(stamp)
        except Exception:
            return (lora_path, strength)

    @classmethod
    def execute(cls, model, lora_path: str, strength: float) -> io.NodeOutput:
        lora_path = lora_path.strip()
        if not lora_path or strength == 0.0:
            return io.NodeOutput(model)

        lora_sd = load_peft_lora(lora_path)

        key_map = comfy.lora.model_lora_keys_unet(model.model, {})
        patches = comfy.lora.load_lora(lora_sd, key_map)

        new_model = model.clone()
        applied = new_model.add_patches(patches, strength)
        print(f"[Irodori-TTS] LoRA applied: {len(applied)}/{len(patches)} patches from {lora_path}")
        return io.NodeOutput(new_model)
