#!/usr/bin/env python3
"""
Stub-based regression tests for Irodori-TTS-ComfyUI.

Requires a ComfyUI checkout (this repo must live in ComfyUI/custom_nodes/)
and its Python environment, but no models, GPU, or running server.

Run from anywhere:
    python custom_nodes/Irodori-TTS-ComfyUI/tests/run_tests.py
"""
from __future__ import annotations

import asyncio
import importlib
import sys
import traceback
from pathlib import Path

COMFY_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(COMFY_ROOT))

import folder_paths  # noqa: F401,E402  (comfy bootstrap)
import torch  # noqa: E402

PKG = "custom_nodes.Irodori-TTS-ComfyUI"
D, T = 8, 16


def _mod(name: str):
    return importlib.import_module(f"{PKG}.{name}")


class StubRFDiT(torch.nn.Module):
    """Minimal stand-in for TextToLatentRFDiT."""

    def __init__(self):
        super().__init__()
        self.p = torch.nn.Parameter(torch.zeros(1))
        self.cache_builds = 0
        self.fwd_total = 0
        self.fwd_cached = 0

    def build_context_kv_cache(self, text_state, speaker_state, caption_state=None):
        self.cache_builds += 1
        return [("k", "v")] * 4

    def forward_with_encoded_conditions(self, x_t, t, text_state, text_mask,
                                        speaker_state=None, speaker_mask=None,
                                        caption_state=None, caption_mask=None,
                                        context_kv_cache=None, **kw):
        assert text_mask.dtype == torch.bool, "masks must stay bool (SDPA additive-bias pitfall)"
        assert text_state.shape[0] == x_t.shape[0], "cond batch must match x batch"
        self.fwd_total += 1
        if context_kv_cache is not None:
            self.fwd_cached += 1
        return x_t * 0.1


def make_wrapper(stub=None):
    mw = _mod("core.model_wrapper")
    stub = stub or StubRFDiT()
    cfg = type("Cfg", (), {"patched_latent_dim": D})()
    w = mw.IrodoriModelWrapper(stub, cfg, None, None, 256, torch.device("cpu"), torch.float32)
    import comfy.model_patcher
    patcher = comfy.model_patcher.ModelPatcher(
        w, load_device=torch.device("cpu"), offload_device=torch.device("cpu"), size=4)
    return stub, w, patcher


def make_conds():
    cd = _mod("core.conditioning")
    ts = torch.randn(1, 5, 32)
    tm = torch.ones(1, 5, dtype=torch.bool)
    pos = cd.pack_cond(ts, tm, None, None, None, None)
    neg = cd.pack_cond(torch.zeros_like(ts), torch.zeros_like(tm), None, None, None, None)
    return pos, neg


# ---------------------------------------------------------------------------

def test_latents_roundtrip():
    L = _mod("core.latents")
    x = torch.randn(2, 16, 1, 7)
    assert torch.equal(L.irodori_to_comfy(L.comfy_to_irodori(x)), x)


def test_prepare_cond_keeps_bool_masks_and_tiles_batch():
    g = _mod("nodes.guider")
    cd = _mod("core.conditioning")
    pos, _ = make_conds()
    prep = g._prepare_cond(pos, torch.device("cpu"), torch.bfloat16, 2)
    d = cd.unpack_cond(prep)
    assert d["text_state"].dtype == torch.bfloat16 and d["text_state"].shape[0] == 2
    assert d["text_mask"].dtype == torch.bool and d["text_mask"].shape[0] == 2
    assert d["speaker_state"] is None


def test_batch_conds_handles_mixed_none():
    g = _mod("nodes.guider")
    cd = _mod("core.conditioning")
    s = torch.randn(1, 5, 8)
    m = torch.ones(1, 5, dtype=torch.bool)
    c1 = g._prepare_cond(cd.pack_cond(s, m, None, None, s.clone(), m.clone()),
                         torch.device("cpu"), torch.float32, 1)
    c2 = g._prepare_cond(cd.pack_cond(s, m, s.clone(), m.clone(), None, None),
                         torch.device("cpu"), torch.float32, 1)
    merged = g._batch_conds([c1, c2])
    assert merged["text_state"].shape[0] == 2
    assert merged["speaker_state"][:1].abs().sum() == 0  # zero-filled for c1
    assert merged["caption_mask"].dtype == torch.bool


def test_stock_cfg_guider_e2e():
    """Standard CFGGuider: timestep-range gating, KV cache, run isolation."""
    import comfy.samplers
    stub, w, patcher = make_wrapper()
    pos, neg = make_conds()
    neg[0][1]["start_percent"] = 0.0
    neg[0][1]["end_percent"] = 0.5

    gd = comfy.samplers.CFGGuider(patcher)
    gd.set_conds(pos, neg)
    gd.set_cfg(4.0)
    sig = comfy.samplers.calculate_sigmas(w.model_sampling, "simple", 8)
    out = gd.sample(torch.randn(1, D, 1, T), torch.zeros(1, D, 1, T),
                    comfy.samplers.sampler_object("euler"), sig, seed=0)
    assert out.shape == (1, D, 1, T) and torch.isfinite(out).all()
    assert stub.cache_builds == 2, f"expected 2 cache builds, got {stub.cache_builds}"
    assert stub.fwd_cached == stub.fwd_total == 8

    gd.sample(torch.randn(1, D, 1, T), torch.zeros(1, D, 1, T),
              comfy.samplers.sampler_object("euler"), sig, seed=1)
    assert stub.cache_builds == 4, "cache must be invalidated between runs (pre_run)"


def test_custom_guider_e2e():
    import comfy.samplers
    g = _mod("nodes.guider")
    stub, w, patcher = make_wrapper()
    pos, neg = make_conds()

    ig = g.IrodoriGuider(patcher)
    ig.set_conds(pos)
    ig.add_uncond(neg, 3.0)
    ig.set_cfg(0.5, 1.0)
    sig = comfy.samplers.calculate_sigmas(w.model_sampling, "simple", 8)
    out = ig.sample(torch.randn(2, D, 1, T), torch.zeros(2, D, 1, T),
                    comfy.samplers.sampler_object("euler"), sig, seed=0)
    assert out.shape == (2, D, 1, T) and torch.isfinite(out).all()
    assert stub.cache_builds == 2
    assert stub.fwd_cached == stub.fwd_total == 8


def test_peft_lora_conversion_and_patching():
    """Synthetic PEFT dict → ComfyUI format → key mapping → weight patching."""
    import comfy.lora
    pl = _mod("core.peft_lora")

    raw = {
        "base_model.model.blocks.0.attention.wq.lora_A.weight": torch.randn(4, 32),
        "base_model.model.blocks.0.attention.wq.lora_B.weight": torch.randn(32, 4),
        "base_model.model.blocks.1.mlp.w1.lora_A.weight": torch.randn(4, 32),
        "base_model.model.blocks.1.mlp.w1.lora_B.weight": torch.randn(64, 4),
        "base_model.model.duration_predictor.proj.weight": torch.randn(8, 8),
        "base_model.model.duration_predictor.proj.bias": torch.randn(8),
    }
    sd = pl.convert_peft_state_dict(raw, lora_alpha=8.0)
    assert "diffusion_model.blocks.0.attention.wq.lora_A.weight" in sd
    assert float(sd["diffusion_model.blocks.0.attention.wq.alpha"]) == 8.0
    assert "diffusion_model.duration_predictor.proj.set_weight" in sd
    assert "diffusion_model.duration_predictor.proj.bias.set_weight" in sd
    # already-converted dicts pass through
    assert pl.convert_peft_state_dict(sd, 8.0) is sd

    class Holder(torch.nn.Module):
        pass

    root = Holder()

    def add_param(full, t):
        m = root
        parts = full.split(".")
        for p in parts[:-1]:
            if p not in m._modules:
                m.add_module(p, Holder())
            m = m._modules[p]
        m.register_parameter(parts[-1], torch.nn.Parameter(torch.zeros_like(t)))

    add_param("blocks.0.attention.wq.weight", torch.zeros(32, 32))
    add_param("blocks.1.mlp.w1.weight", torch.zeros(64, 32))
    add_param("duration_predictor.proj.weight", torch.zeros(8, 8))
    add_param("duration_predictor.proj.bias", torch.zeros(8))

    stub, w, patcher = make_wrapper(stub=None)
    w.diffusion_model = root  # swap in the shaped stub

    key_map = comfy.lora.model_lora_keys_unet(w, {})
    patches = comfy.lora.load_lora(sd, key_map, log_missing=False)
    assert len(patches) == 4, f"expected 4 patches, got {len(patches)}"
    model_sd = w.state_dict()
    assert all(k in model_sd for k in patches)

    applied = patcher.add_patches(patches, 1.0)
    assert len(applied) == 4
    patcher.patch_model()
    params = dict(w.named_parameters())
    assert params["diffusion_model.blocks.0.attention.wq.weight"].abs().max() > 0
    assert params["diffusion_model.duration_predictor.proj.weight"].abs().max() > 0


def test_v3_schemas():
    pkg = importlib.import_module(PKG)
    ext = asyncio.run(pkg.comfy_entrypoint())
    nodes = asyncio.run(ext.get_node_list())
    assert len(nodes) == 11
    for n in nodes:
        schema = n.GET_SCHEMA()  # validates input/output id uniqueness
        assert schema.display_name
        n.INPUT_TYPES()          # V1 compatibility shim


def test_sway_scheduler():
    sch = _mod("nodes.scheduler")

    s = sch.sway_sigmas(20, -1.0)
    assert s.shape == (21,)
    assert float(s[0]) == 1.0 and float(s[-1]) == 0.0
    assert bool(torch.all(s[:-1] > s[1:]))
    # negative coeff densifies the noise side: first step smaller than linear's
    assert float(s[0] - s[1]) < 1.0 / 20

    # coeff 0 == linear schedule
    lin = sch.sway_sigmas(10, 0.0)
    assert torch.allclose(lin, torch.linspace(1.0, 0.0, 11))

    # denoise scales the start point (audio2audio)
    half = sch.sway_sigmas(10, -1.0, denoise=0.5)
    assert abs(float(half[0]) - 0.5) < 1e-6 and float(half[-1]) == 0.0


def test_encode_speaker_from_latent():
    """ref LATENT → speaker encoder → (tokens, speaker_dim), via the real
    patch_sequence_with_mask + prepend-mean-token plumbing with a stub encoder."""
    cd = _mod("core.conditioning")
    import comfy.model_patcher

    D, ldim, spk_dim = 8, 8, 16  # latent_patch_size=1 so D==ldim
    T = 5

    class StubEncoder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = torch.nn.Linear(D, spk_dim)

        def forward(self, latent, mask):
            return self.proj(latent)  # (B, S, spk_dim), preserves length

    class StubRFDiT(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.speaker_encoder = StubEncoder()
            self.speaker_norm = torch.nn.Identity()

        @staticmethod
        def _prepend_masked_mean_token(state, mask):
            mean = state.mean(dim=1, keepdim=True)
            return torch.cat([mean, state], dim=1), mask

    cfg = type("Cfg", (), {
        "latent_dim": ldim, "latent_patch_size": 1, "speaker_patch_size": 1,
        "patched_latent_dim": D, "use_speaker_condition_resolved": True,
    })()
    mw = _mod("core.model_wrapper")
    w = mw.IrodoriModelWrapper(StubRFDiT(), cfg, None, None, 256,
                               torch.device("cpu"), torch.float32)
    patcher = comfy.model_patcher.ModelPatcher(
        w, torch.device("cpu"), torch.device("cpu"), size=4)

    ref = {"samples": torch.randn(1, D, 1, T)}  # 4D sampling format
    emb = cd.encode_speaker_from_latent(patcher, ref)
    # T tokens + 1 prepended mean token
    assert emb.shape == (T + 1, spk_dim), emb.shape
    # canonical storage: CPU float32 (so it merges with loader-sourced embeds)
    assert emb.device.type == "cpu" and emb.dtype == torch.float32


def test_merge_mixed_device_dtype():
    """Encoder output (e.g. cuda/bf16) + loader output (cpu/f32) must merge."""
    em = _mod("nodes.embed_merge")
    a = {"embedding": torch.randn(16, 768, dtype=torch.bfloat16)}  # encoder-like
    b = {"embedding": torch.randn(16, 768, dtype=torch.float32)}   # loader-like
    out = em.IrodoriSpeakerEmbedMerge.execute(a, b, "concat").args[0]
    assert out["embedding"].shape == (32, 768)
    assert out["embedding"].dtype == torch.float32
    avg = em.IrodoriSpeakerEmbedMerge.execute(a, b, "average").args[0]
    assert avg["embedding"].dtype == torch.float32


def test_speaker_embed_sample_tokens():
    sm = _mod("nodes.embed_sample")
    N = sm.IrodoriSpeakerEmbedSampleTokens
    emb = torch.arange(20, dtype=torch.float32)[:, None].repeat(1, 4)  # token i = i

    out = N.execute({"embedding": emb}, 5, 0, False).args[0]["embedding"]
    assert out.shape == (5, 4)
    # every kept token is a real, intact source token
    kept = {int(v) for v in out[:, 0]}
    assert kept.issubset(set(range(20))) and len(kept) == 5
    # indices kept sorted
    assert torch.all(out[:, 0].diff() > 0)

    # deterministic per seed; different seeds differ
    a = N.execute({"embedding": emb}, 5, 7, False).args[0]["embedding"]
    a2 = N.execute({"embedding": emb}, 5, 7, False).args[0]["embedding"]
    b = N.execute({"embedding": emb}, 5, 8, False).args[0]["embedding"]
    assert torch.allclose(a, a2) and not torch.allclose(a, b)

    # keep_first_token always includes token 0
    k = N.execute({"embedding": emb}, 5, 3, True).args[0]["embedding"]
    assert k.shape == (5, 4) and float(k[0, 0]) == 0.0

    # num_tokens >= count -> no-op
    same = N.execute({"embedding": emb}, 50, 0, False).args[0]["embedding"]
    assert torch.allclose(same, emb)


def test_speaker_embed_merge():
    em = _mod("nodes.embed_merge")
    a = {"embedding": torch.randn(16, 768)}
    b = {"embedding": torch.randn(16, 768)}
    c = {"embedding": torch.randn(8, 768)}

    # concat keeps all tokens
    out = em.IrodoriSpeakerEmbedMerge.execute(a, b, "concat").args[0]
    assert out["embedding"].shape == (32, 768)
    out3 = em.IrodoriSpeakerEmbedMerge.execute(a, b, "concat", speaker_embed_c=c).args[0]
    assert out3["embedding"].shape == (40, 768)  # 16+16+8
    # concat works with differing token counts
    assert em.IrodoriSpeakerEmbedMerge.execute(a, c, "concat").args[0]["embedding"].shape == (24, 768)

    # average requires equal token counts
    avg = em.IrodoriSpeakerEmbedMerge.execute(a, b, "average").args[0]
    assert avg["embedding"].shape == (16, 768)
    assert torch.allclose(avg["embedding"], (a["embedding"] + b["embedding"]) / 2)

    # weighted average = normalized convex blend
    wavg = em.IrodoriSpeakerEmbedMerge.execute(a, b, "average", weight_a=3.0, weight_b=1.0).args[0]
    expected = (3.0 * a["embedding"] + 1.0 * b["embedding"]) / 4.0
    assert torch.allclose(wavg["embedding"], expected)
    # weight 1/0 -> just the first embedding
    only_a = em.IrodoriSpeakerEmbedMerge.execute(a, b, "average", weight_a=1.0, weight_b=0.0).args[0]
    assert torch.allclose(only_a["embedding"], a["embedding"])

    # weighted concat scales each block (V-contribution gain)
    wc = em.IrodoriSpeakerEmbedMerge.execute(a, b, "concat", weight_a=2.0, weight_b=0.5).args[0]
    assert wc["embedding"].shape == (32, 768)
    assert torch.allclose(wc["embedding"][:16], 2.0 * a["embedding"])
    assert torch.allclose(wc["embedding"][16:], 0.5 * b["embedding"])

    # all-zero average weights raise
    try:
        em.IrodoriSpeakerEmbedMerge.execute(a, b, "average", weight_a=0.0, weight_b=0.0)
        assert False, "zero-sum average weights must raise"
    except ValueError:
        pass

    try:
        em.IrodoriSpeakerEmbedMerge.execute(a, c, "average")
        assert False, "average with mismatched token counts must raise"
    except ValueError:
        pass

    # speaker_dim mismatch always raises
    try:
        em.IrodoriSpeakerEmbedMerge.execute(a, {"embedding": torch.randn(16, 512)}, "concat")
        assert False, "speaker_dim mismatch must raise"
    except ValueError:
        pass


def test_trim_tail():
    pp = _mod("nodes.postprocess")
    patch, ldim = 2, 4

    class FakeCodec:
        latent_dim = ldim

    class FakeVae:
        codec = FakeCodec()
        model_cfg = type("C", (), {"latent_patch_size": patch})()

    # 10 frames of speech-like noise, 10 flat zero frames
    z = torch.cat([torch.randn(1, 10, ldim), torch.zeros(1, 10, ldim)], dim=1)
    from irodori_tts.codec import patchify_latent
    lat = {"samples": patchify_latent(z, patch).transpose(1, 2).unsqueeze(2), "sample_rate": 48000}

    out = pp.IrodoriTrimTail.execute(FakeVae(), lat, window_size=5,
                                     std_threshold=0.05, mean_threshold=0.1).args[0]
    t_out = out["samples"].shape[-1]
    assert t_out == 5, f"expected 5 patched steps (10 frames), got {t_out}"
    assert out["sample_rate"] == 48000

    # no flat tail -> unchanged
    z2 = torch.randn(1, 20, ldim)
    lat2 = {"samples": patchify_latent(z2, patch).transpose(1, 2).unsqueeze(2)}
    out2 = pp.IrodoriTrimTail.execute(FakeVae(), lat2, window_size=5,
                                      std_threshold=0.05, mean_threshold=0.1).args[0]
    assert out2["samples"].shape[-1] == 10


# ---------------------------------------------------------------------------

def main() -> int:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"PASS {t.__name__}")
        except Exception:
            failed += 1
            print(f"FAIL {t.__name__}")
            traceback.print_exc()
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
