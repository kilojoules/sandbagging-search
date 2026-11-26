"""Tests for sbx.noise (CPU only, < 90 s).

Mechanics are tested on the tiny random Llama of sbx.testing.tiny_ctx (outputs are gibberish by design): restoration is checked
bit for bit, noise statistics against their definition, determinism by seed, hook cleanup, refusal paths. The tiny adapter's
parser is replaced by a deterministic pseudo-answer (crc32 of the generated text) so that per-question correctness actually
depends on the perturbation. Statistics are tested on planted synthetic arrays with hand-computed expectations.
Run:  USE_TF=0 USE_TORCH=1 python3 -m pytest sbx/tests/test_noise.py -q
"""
from __future__ import annotations

import copy
import gc
import json
import math
import types
import weakref
import zlib

import numpy as np
import pytest
import torch

from sbx import core
from sbx import noise as nz
from sbx.core import Ctx
from sbx.framings import FRAMINGS
from sbx.testing import TinyAdapter, tiny_ctx, toy_questions

NOLOG = lambda *a, **k: None  # noqa: E731
N = FRAMINGS["N"]


# ----------------------------------------------------------------------------------------------- fixtures / helpers
def _crc_parse(final, nchoices=4):
    """Deterministic pseudo-answer from the generated text: makes corr depend on the model's output."""
    return zlib.crc32(final.encode("utf-8")) % nchoices


@pytest.fixture(scope="module")
def base():
    return tiny_ctx(layers=4, hidden=64)


@pytest.fixture
def ctx(base):
    """Fresh Ctx with a private deep copy of the tiny model (a failing test can never leak weights into another one)."""
    ad = TinyAdapter("direct", "system")
    ad.parse_letter = _crc_parse
    return Ctx(ad, base.tok, copy.deepcopy(base.model), log=NOLOG)


@pytest.fixture
def qs():
    return toy_questions(6)


def snap(model):
    return {k: v.detach().clone() for k, v in model.state_dict().items()}


def same_state(model, ref):
    cur = model.state_dict()
    return set(cur) == set(ref) and all(torch.equal(cur[k], ref[k]) for k in ref)


def last_logits(ctx, qs, specs=None):
    enc = ctx.adapter.encode(ctx.tok, ctx.prompts(qs)).to(ctx.model.device)
    with torch.no_grad(), core.Hooks(ctx, specs):
        return ctx.model(**enc).logits[:, -1].float().clone()


def n_hooks(ctx):
    return sum(len(m._forward_hooks) for m in ctx.model.modules())


def make_U(ctx, value=2.0):
    return {L: value for L in range(1, ctx.nl)}


def fake_ctx(model, layers_fn):
    """Minimal stand-in for a Ctx (WeightNoise / select_params only use model, adapter.layers and log)."""
    return types.SimpleNamespace(model=model, adapter=types.SimpleNamespace(layers=layers_fn), log=NOLOG)


# ================================================================================================ parameter selection
def test_default_selection_tiny(ctx):
    names = [n for n, _ in nz.select_params(ctx)]
    expect = [f"model.layers.{i}.{m}.weight" for i in range(4)
              for m in ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
                        "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj")]
    assert names == expect
    assert not any(k in n for n in names for k in ("embed", "lm_head", "norm"))


def _tiny_arch(name):
    """(model, layers_fn, expected number of default-selected tensors) for tiny offline instances of real HF architectures."""
    import transformers as T
    common = dict(hidden_size=64, intermediate_size=128, num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                  vocab_size=300, pad_token_id=0)
    flat = lambda m: m.model.layers  # noqa: E731
    mk = lambda cfg: T.AutoModelForCausalLM.from_config(cfg)  # noqa: E731
    if name == "llama":
        return mk(T.LlamaConfig(**common)), flat, 14
    if name == "qwen3":
        return mk(T.Qwen3Config(**common, head_dim=16)), flat, 14
    if name == "qwen3moe":
        return mk(T.Qwen3MoeConfig(**common, head_dim=16, moe_intermediate_size=32, num_experts=4, num_experts_per_tok=2,
                                   decoder_sparse_step=1)), flat, 32
    if name == "phi3":
        return mk(T.Phi3Config(**common)), flat, 8                       # fused qkv_proj, o_proj, gate_up_proj, down_proj
    if name == "mistral":
        return mk(T.MistralConfig(**common)), flat, 14
    if name == "olmo3":
        return mk(T.Olmo3Config(**common)), flat, 14
    if name == "seed_oss":
        return mk(T.SeedOssConfig(**common, head_dim=16)), flat, 14       # q/k/v biases are NOT selected
    if name == "gptoss":
        return mk(T.GptOssConfig(**{**common, "intermediate_size": 64}, head_dim=16, num_local_experts=4,
                                 num_experts_per_tok=2)), flat, 12        # 4 attn + 3D experts gate_up_proj/down_proj per layer
    if name == "glm4":
        return mk(T.Glm4Config(**common, head_dim=16)), flat, 12
    if name == "glm4moe":
        return mk(T.Glm4MoeConfig(**common, head_dim=16, moe_intermediate_size=32, n_routed_experts=4, n_shared_experts=1,
                                  num_experts_per_tok=2, first_k_dense_replace=1)), flat, 26
    if name == "mixtral":
        return mk(T.MixtralConfig(**common, num_local_experts=4, num_experts_per_tok=2)), flat, 32
    if name == "gpt2":
        return mk(T.GPT2Config(n_embd=64, n_layer=2, n_head=4, vocab_size=300)), (lambda m: m.transformer.h), 8
    if name == "gemma3_mm":
        t = T.Gemma3TextConfig(**common, head_dim=16, sliding_window=8)
        v = T.SiglipVisionConfig(hidden_size=32, intermediate_size=64, num_hidden_layers=1, num_attention_heads=2,
                                 image_size=28, patch_size=14)
        return mk(T.Gemma3Config(text_config=t, vision_config=v, mm_tokens_per_image=1)), \
            (lambda m: m.model.language_model.layers), 14
    if name == "mistral3_mm":
        t = T.MistralConfig(**common)
        v = T.PixtralVisionConfig(hidden_size=32, intermediate_size=64, num_hidden_layers=1, num_attention_heads=2,
                                  image_size=28, patch_size=14)
        return T.AutoModelForImageTextToText.from_config(T.Mistral3Config(text_config=t, vision_config=v)), \
            (lambda m: m.model.language_model.layers), 14
    raise KeyError(name)


ARCHS = ["llama", "qwen3", "qwen3moe", "phi3", "mistral", "olmo3", "seed_oss", "gptoss", "glm4", "glm4moe", "mixtral", "gpt2",
         "gemma3_mm", "mistral3_mm"]
FORBIDDEN = ("embed", "lm_head", "norm", "router", "bias", "sinks", "vision", "multi_modal", "wte", "wpe", "ln_")


@pytest.mark.parametrize("arch", ARCHS)
def test_default_selection_across_architectures(arch):
    try:
        model, layers_fn, expected = _tiny_arch(arch)
    except (ImportError, AttributeError) as e:     # architecture not in this transformers version
        pytest.skip(f"{arch}: {e}")
    c = fake_ctx(model, layers_fn)
    for restrict in (True, False):                 # regex alone must already be right; the layer restriction is a second guard
        sel = nz.select_params(c, restrict_to_layers=restrict)
        names = [n for n, _ in sel]
        assert len(names) == expected, (arch, restrict, names)
        assert not [n for n in names if any(f in n for f in FORBIDDEN)], names
        assert all(p.dtype == torch.float32 and p.ndim >= 2 for _, p in sel)
    # attention + MLP presets partition the default selection
    attn = {n for n, _ in nz.select_params(c, nz.INCLUDE_ATTN)}
    mlp = {n for n, _ in nz.select_params(c, nz.INCLUDE_MLP)}
    assert attn | mlp == {n for n, _ in nz.select_params(c)} and not attn & mlp


def test_default_regexes_on_known_name_lists():
    """Name-level checks for families that cannot be instantiated offline (ChatGLM remote code, mxfp4 gpt-oss, qwen3.5-style)."""
    inc, exc = nz._as_pattern(nz.DEFAULT_INCLUDE, "i"), nz._as_pattern(nz.DEFAULT_EXCLUDE, "e")
    sel = lambda n: bool(inc.search(n)) and not exc.search(n)  # noqa: E731
    yes = ["transformer.encoder.layers.3.self_attention.query_key_value.weight", "transformer.encoder.layers.3.self_attention.dense.weight",
           "transformer.encoder.layers.3.mlp.dense_h_to_4h.weight", "transformer.encoder.layers.3.mlp.dense_4h_to_h.weight",
           "model.layers.0.linear_attn.in_proj_qkvz.weight", "model.layers.0.linear_attn.out_proj.weight",
           "model.layers.0.self_attn.kv_a_proj_with_mqa.weight", "model.layers.0.self_attn.q_b_proj.weight",
           # mxfp4 expert storage must be SEEN (and then refused loudly), never silently skipped
           "model.layers.0.mlp.experts.gate_up_proj_blocks", "model.layers.0.mlp.experts.down_proj_scales"]
    no = ["transformer.embedding.word_embeddings.weight", "transformer.output_layer.weight", "transformer.encoder.final_layernorm.weight",
          "transformer.encoder.layers.3.input_layernorm.weight", "transformer.encoder.layers.3.self_attention.query_key_value.bias",
          "model.layers.0.linear_attn.conv1d.weight", "model.layers.0.linear_attn.A_log", "model.layers.0.linear_attn.norm.weight",
          "model.layers.0.mlp.experts.gate_up_proj_bias", "model.layers.0.mlp.router.weight", "model.layers.0.mlp.gate.weight",
          "model.layers.0.mlp.shared_expert_gate.weight", "model.layers.0.mlp.gate.e_score_correction_bias",
          "model.layers.0.self_attn.sinks", "model.embed_tokens.weight", "lm_head.weight", "model.norm.weight",
          "model.vision_tower.transformer.layers.0.attention.q_proj.weight", "model.multi_modal_projector.linear_1.weight",
          "model.layers.0.self_attn.q_proj.qweight"]                 # GPTQ/AWQ packed tensors never match
    assert [n for n in yes if not sel(n)] == []
    assert [n for n in no if sel(n)] == []


def test_layers_restriction_is_one_based(ctx):
    names = [n for n, _ in nz.select_params(ctx, layers=[2, 4])]
    assert len(names) == 14 and all(n.startswith(("model.layers.1.", "model.layers.3.")) for n in names)
    with pytest.raises(ValueError):
        nz.select_params(ctx, layers=[0])
    with pytest.raises(ValueError):
        nz.select_params(ctx, layers=[5])


def test_empty_selection_raises_and_nothing_changes(ctx):
    ref = snap(ctx.model)
    with pytest.raises(ValueError, match="no parameters"):
        with nz.WeightNoise(ctx, 0.1, 0, include=r"does_not_exist"):
            pass
    assert same_state(ctx.model, ref)
    assert getattr(ctx.model, nz._ACTIVE_ATTR, None) is None


def test_paper_preset_perturbs_every_parameter(ctx):
    ref = snap(ctx.model)
    with nz.WeightNoise(ctx, 0.05, 0, include=nz.PAPER_INCLUDE, exclude=nz.PAPER_EXCLUDE, restrict_to_layers=False) as wn:
        cur = snap(ctx.model)
        assert set(wn.report) == set(dict(ctx.model.named_parameters()))
        assert all(not torch.equal(cur[k], ref[k]) for k in ref)                 # embeddings, norms, lm_head too
    assert same_state(ctx.model, ref)


# ================================================================================================ WeightNoise: restore
def test_restore_is_bit_exact(ctx):
    ref = snap(ctx.model)
    with nz.WeightNoise(ctx, 0.1, 3) as wn:
        cur = snap(ctx.model)
        changed = {k for k in ref if not torch.equal(cur[k], ref[k])}
        assert changed == set(wn.report) and len(changed) == 28
    assert same_state(ctx.model, ref)
    assert getattr(ctx.model, nz._ACTIVE_ATTR, None) is None


def test_restore_after_exception_in_block(ctx):
    ref = snap(ctx.model)
    with pytest.raises(KeyError, match="boom"):
        with nz.WeightNoise(ctx, 0.3, 1):
            assert not same_state(ctx.model, ref)
            raise KeyError("boom")
    assert same_state(ctx.model, ref)
    with nz.WeightNoise(ctx, 0.3, 1):          # usable again afterwards
        pass


def test_restore_when_enter_fails_midway(ctx, monkeypatch):
    ref = snap(ctx.model)
    real, calls = nz._perturb_, {"n": 0}

    def flaky(p, std, seed64):
        calls["n"] += 1
        if calls["n"] == 5:
            real(p, std, seed64)                # tensor 5 IS perturbed, then the failure hits
            raise RuntimeError("simulated CUDA OOM")
        return real(p, std, seed64)
    monkeypatch.setattr(nz, "_perturb_", flaky)
    with pytest.raises(RuntimeError, match="simulated"):
        with nz.WeightNoise(ctx, 0.3, 1):
            pytest.fail("must not enter")
    assert calls["n"] == 5 and same_state(ctx.model, ref)
    assert getattr(ctx.model, nz._ACTIVE_ATTR, None) is None
    monkeypatch.setattr(nz, "_perturb_", real)
    with nz.WeightNoise(ctx, 0.3, 1):           # the model is not stuck in a half-active state
        pass


def test_failed_restore_is_never_swallowed(ctx, monkeypatch):
    """If undoing a half-finished __enter__ fails, the caller must hear about it (the model would be silently corrupted)."""
    real_take, real_pert, calls = nz.WeightSnapshot.take, nz._perturb_, {"n": 0}

    def bad_take(self):
        out = real_take(self)
        name, p, buf = self.entries[0]
        self.entries[0] = (name, p, buf[:, :3].clone())                  # not broadcastable: restoring tensor 1 will fail
        return out

    def boom(p, std, seed64):
        calls["n"] += 1
        if calls["n"] == 3:
            raise RuntimeError("perturb failed")
        return real_pert(p, std, seed64)
    monkeypatch.setattr(nz.WeightSnapshot, "take", bad_take)
    monkeypatch.setattr(nz, "_perturb_", boom)
    with pytest.raises(RuntimeError, match="failed to restore") as ei:
        nz.WeightNoise(ctx, 0.1, 0).__enter__()
    assert "perturb failed" in str(ei.value.__cause__)
    assert getattr(ctx.model, nz._ACTIVE_ATTR, None) is None


def test_failed_restore_on_exit_raises_restores_the_rest_and_releases_the_model(ctx):
    ref = snap(ctx.model)
    wn = nz.WeightNoise(ctx, 0.1, 0)
    wn.__enter__()
    name, p, buf = wn._snaps[0]
    wn._snaps[0] = (name, p, buf[:, :3].clone())
    with pytest.raises(RuntimeError, match="failed to restore"):
        wn.__exit__(None, None, None)
    cur = snap(ctx.model)
    assert all(torch.equal(cur[k], ref[k]) for k in ref if k != name)       # everything else is back
    assert not torch.equal(cur[name], ref[name])                            # ... and the failure was real
    assert getattr(ctx.model, nz._ACTIVE_ATTR, None) is None


def test_restore_on_keyboard_interrupt(ctx):
    ref = snap(ctx.model)
    with pytest.raises(KeyboardInterrupt):
        with nz.WeightNoise(ctx, 0.3, 1):
            raise KeyboardInterrupt
    assert same_state(ctx.model, ref)
    assert getattr(ctx.model, nz._ACTIVE_ATTR, None) is None


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_half_precision_restore_and_realised_noise(ctx, dtype):
    ctx.model.to(dtype)
    ref = snap(ctx.model)
    with nz.WeightNoise(ctx, 0.05, 7) as wn:
        assert not same_state(ctx.model, ref)
        assert wn.summary["requested_rel"] == pytest.approx(0.05)
        assert wn.summary["realised_rel"] == pytest.approx(0.05, rel=0.1)     # rounding barely matters at this scale
        assert all(r["dtype"] == str(dtype) for r in wn.report.values())
    assert same_state(ctx.model, ref)
    assert all(v.dtype == dtype for v in ctx.model.state_dict().values())


def test_realised_noise_is_reported_when_rounding_eats_it(ctx):
    ctx.model.to(torch.bfloat16)
    logs = []
    with nz.WeightNoise(ctx, 0.0004, 7, log=logs.append) as wn:       # far below bf16 resolution
        assert wn.summary["realised_rel"] < 0.8 * wn.summary["requested_rel"]
    assert any("WARNING" in str(m) for m in logs)


def test_unselected_params_and_buffers_untouched(ctx):
    ref = snap(ctx.model)
    with nz.WeightNoise(ctx, 0.2, 1) as wn:
        cur = snap(ctx.model)
        for k in ref:
            if k not in wn.report:
                assert torch.equal(cur[k], ref[k]), k
    assert any(not k.startswith("model.layers") or "norm" in k for k in ref)     # the check above was not vacuous


def test_multi_chunk_path(ctx, monkeypatch):
    monkeypatch.setattr(nz, "CHUNK", 1000)          # every tensor is perturbed in several chunks
    ref = snap(ctx.model)
    with nz.WeightNoise(ctx, 0.1, 2) as wn:
        a = snap(ctx.model)
        assert wn.summary["realised_rel"] == pytest.approx(0.1, rel=0.02)
    assert same_state(ctx.model, ref)
    with nz.WeightNoise(ctx, 0.1, 2):
        assert all(torch.equal(v, a[k]) for k, v in snap(ctx.model).items())       # chunked stream is reproducible
    assert same_state(ctx.model, ref)


def test_perturb_and_sumsq_really_work_in_fixed_size_chunks(monkeypatch):
    """CHUNK is part of the reproducibility contract (a CPU/CUDA stream depends on how it is drawn): it must be what is used."""
    monkeypatch.setattr(nz, "CHUNK", 1000)
    sizes, real = [], torch.randn

    def spy(shape, *a, **k):
        sizes.append(math.prod(shape))
        return real(shape, *a, **k)
    monkeypatch.setattr(torch, "randn", spy)
    p = torch.nn.Parameter(torch.arange(4096, dtype=torch.float32).reshape(64, 64) / 4096)
    ss = nz._perturb_(p, 0.1, 5)
    assert sizes == [1000, 1000, 1000, 1000, 96] and ss > 0
    w = real(4096)
    assert nz._sumsq(w) == pytest.approx(float(w.double().pow(2).sum()), rel=1e-6)           # multi-chunk accumulation is exact


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_noise_generation_contract(ctx, dtype):
    """Pins what 'seeded Gaussian noise' means: fp32 N(0,1) from a Generator seeded with _seed64('weight', seed, name), scaled by
    sigma_rel * rms(tensor), ADDED IN FP32 and cast back once. A change here silently changes every reported perturbation."""
    ctx.model.to(dtype)
    name, sigma, seed = "model.layers.1.mlp.up_proj.weight", 0.2, 13
    w = dict(ctx.model.named_parameters())[name].detach().clone()
    g = torch.Generator(device="cpu")
    g.manual_seed(nz._seed64("weight", seed, name))
    z = torch.randn(w.numel(), generator=g, dtype=torch.float32).reshape(w.shape)
    expected = (w.float() + z * (sigma * math.sqrt(nz._sumsq(w) / w.numel()))).to(dtype)
    with nz.WeightNoise(ctx, sigma, seed, include=r"layers\.1\.mlp\.up_proj") as wn:
        assert list(wn.report) == [name]
        got = dict(ctx.model.named_parameters())[name].detach().clone()
    assert torch.equal(got, expected)


def test_noise_of_different_tensors_is_independent(ctx):
    """One generator per (seed, tensor name): same-shaped tensors must not receive the same (or correlated) noise."""
    ref = snap(ctx.model)
    with nz.WeightNoise(ctx, 0.1, 3):
        cur = snap(ctx.model)
    d = lambda n: (cur[n] - ref[n]).flatten()  # noqa: E731
    qd = [d(f"model.layers.{i}.self_attn.q_proj.weight") for i in range(4)]
    for i in range(4):
        for j in range(i + 1, 4):
            assert abs(float(torch.corrcoef(torch.stack([qd[i], qd[j]]))[0, 1])) < 0.06, (i, j)      # SE ~ 0.016 (4096 elements)
    kd = d("model.layers.0.self_attn.k_proj.weight")                                                 # shorter tensor, same layer
    assert abs(float(torch.corrcoef(torch.stack([qd[0][:kd.numel()], kd]))[0, 1])) < 0.1
    kd2 = d("model.layers.1.self_attn.k_proj.weight")
    assert abs(float(torch.corrcoef(torch.stack([kd, kd2]))[0, 1])) < 0.1


# ================================================================================================ WeightNoise: statistics
def test_determinism_by_seed(ctx):
    def perturbed(seed):
        with nz.WeightNoise(ctx, 0.05, seed):
            return snap(ctx.model)
    a1, a2, b = perturbed(1), perturbed(1), perturbed(2)
    assert all(torch.equal(a1[k], a2[k]) for k in a1)
    sel = [n for n, _ in nz.select_params(ctx)]
    assert all(not torch.equal(a1[k], b[k]) for k in sel)


def test_noise_depends_only_on_seed_name_and_rms(ctx):
    """Selecting fewer tensors must not change the noise of the others (generator seeded per (seed, name))."""
    with nz.WeightNoise(ctx, 0.05, 4):
        full = snap(ctx.model)
    with nz.WeightNoise(ctx, 0.05, 4, include=nz.INCLUDE_ATTN):
        part = snap(ctx.model)
    k = "model.layers.2.self_attn.v_proj.weight"
    assert torch.equal(full[k], part[k])
    assert not torch.equal(full["model.layers.2.mlp.up_proj.weight"], part["model.layers.2.mlp.up_proj.weight"])


def test_common_random_numbers_across_sigma(ctx):
    """Same (seed, name) => the noise direction is the same for every sigma: delta(2s) == 2*delta(s) (up to fp32 rounding)."""
    ref = snap(ctx.model)
    deltas = {}
    for s in (0.05, 0.1):
        with nz.WeightNoise(ctx, s, 9):
            cur = snap(ctx.model)
        deltas[s] = {k: cur[k] - ref[k] for k in ref}
    k = "model.layers.1.mlp.down_proj.weight"
    assert torch.allclose(deltas[0.1][k], 2 * deltas[0.05][k], rtol=1e-3, atol=1e-7)


def test_noise_std_matches_sigma_rel_times_rms(ctx):
    sig = 0.1
    ref = snap(ctx.model)
    with nz.WeightNoise(ctx, sig, 5) as wn:
        cur = snap(ctx.model)
    num = den = 0.0
    for name in wn.report:
        w0, d = ref[name], cur[name] - ref[name]
        rms = float(w0.pow(2).mean().sqrt())
        assert float(d.std(unbiased=False)) / (sig * rms) == pytest.approx(1.0, abs=0.1), name
        assert abs(float(d.mean())) < 0.1 * sig * rms, name                       # zero-mean
        assert wn.report[name]["rms"] == pytest.approx(rms, rel=1e-6)
        assert wn.report[name]["std"] == pytest.approx(sig * rms, rel=1e-6)
        num, den = num + float(d.pow(2).sum()), den + float(w0.pow(2).sum())
    assert math.sqrt(num / den) == pytest.approx(sig, rel=0.02)                   # pooled over 147k elements
    assert wn.summary["realised_rel"] == pytest.approx(sig, rel=0.02)
    # Gaussian shape: kurtosis ~ 3, ~68% within one std on the pooled standardised deltas
    z = torch.cat([((cur[n] - ref[n]) / wn.report[n]["std"]).flatten() for n in wn.report])
    assert float(z.pow(4).mean()) == pytest.approx(3.0, abs=0.15)
    assert float((z.abs() < 1).float().mean()) == pytest.approx(0.6827, abs=0.01)


def test_absolute_sigma_mode(ctx):
    ref = snap(ctx.model)
    with nz.WeightNoise(ctx, None, 5, sigma_abs=0.003) as wn:
        cur = snap(ctx.model)
    for name in wn.report:
        d = cur[name] - ref[name]
        assert float(d.std(unbiased=False)) == pytest.approx(0.003, rel=0.1), name
        assert wn.report[name]["std"] == 0.003
    assert same_state(ctx.model, ref)


def test_sigma_zero_is_a_noop(ctx, qs):
    ref = snap(ctx.model)
    base_logits = last_logits(ctx, qs)
    with nz.WeightNoise(ctx, 0.0, 3) as wn:
        assert wn.summary["noop"] and wn._snaps == []
        assert same_state(ctx.model, ref)
        assert torch.equal(last_logits(ctx, qs), base_logits)
    assert same_state(ctx.model, ref)


def test_weight_noise_changes_logits_and_restore_returns_them(ctx, qs):
    base_logits = last_logits(ctx, qs)
    with nz.WeightNoise(ctx, 0.1, 1):
        noisy = last_logits(ctx, qs)
    assert not torch.allclose(noisy, base_logits, atol=1e-4)
    assert torch.equal(last_logits(ctx, qs), base_logits)


def test_weight_noise_leaves_no_hooks(ctx, qs):
    with nz.WeightNoise(ctx, 0.1, 1):
        last_logits(ctx, qs)
        assert n_hooks(ctx) == 0


def test_argument_validation(ctx):
    for bad in (-0.1, float("nan"), float("inf")):
        with pytest.raises(ValueError):
            nz.WeightNoise(ctx, bad, 0)
    with pytest.raises(ValueError, match="exactly one"):
        nz.WeightNoise(ctx, None, 0)
    with pytest.raises(ValueError, match="exactly one"):
        nz.WeightNoise(ctx, 0.1, 0, sigma_abs=0.1)
    with pytest.raises(TypeError):
        nz.WeightNoise(ctx, 0.1, 0, include=5)
    with pytest.raises(Exception):                                  # invalid regex fails at construction, not mid-run
        nz.WeightNoise(ctx, 0.1, 0, include="(")
    with pytest.raises(ValueError):
        nz.WeightNoise(ctx, 0.1, 0, ram_fraction=0)


def test_plan_does_not_touch_the_model(ctx):
    ref = snap(ctx.model)
    plan = nz.WeightNoise(ctx, 0.1, 0).plan()
    assert plan["n_tensors"] == 28 and plan["numel"] == 147456 and plan["nbytes"] == 147456 * 4
    assert same_state(ctx.model, ref)


def test_weight_rms_summary(ctx):
    s = nz.weight_rms_summary(ctx)
    assert s["n_tensors"] == 28 and s["numel"] == 147456
    w = torch.cat([p.detach().flatten() for _, p in nz.select_params(ctx)])
    assert s["rms_pooled"] == pytest.approx(float(w.pow(2).mean().sqrt()), rel=1e-6)
    assert s["rms_min"] <= s["rms_median"] <= s["rms_max"]
    assert 0.01 < s["rms_pooled"] < 0.04                          # HF init std 0.02


# ================================================================================================ WeightNoise: refusals
def test_quantized_params_are_refused_unless_excluded(ctx):
    layer0 = ctx.model.model.layers[0]
    layer0.mlp.gate_proj.weight = torch.nn.Parameter(layer0.mlp.gate_proj.weight.data.to(torch.uint8), requires_grad=False)
    ref = snap(ctx.model)
    with pytest.raises(nz.QuantizedParamError, match=r"model\.layers\.0\.mlp\.gate_proj\.weight.*uint8"):
        with nz.WeightNoise(ctx, 0.1, 0):
            pytest.fail("must not enter")
    assert same_state(ctx.model, ref)                                      # nothing was perturbed before refusing
    assert isinstance(nz.QuantizedParamError("x"), RuntimeError)
    with nz.WeightNoise(ctx, 0.1, 0, include=nz.INCLUDE_ATTN) as wn:       # excluded by include: fine
        assert all("self_attn" in n for n in wn.report) and len(wn.report) == 16
    assert same_state(ctx.model, ref)
    with nz.WeightNoise(ctx, 0.1, 0, exclude=nz.DEFAULT_EXCLUDE + r"|layers\.0\.mlp\.gate_proj"):   # excluded by exclude: fine
        pass
    assert same_state(ctx.model, ref)


@pytest.mark.skipif(not hasattr(torch, "float8_e4m3fn"), reason="no float8 dtype")
def test_float8_params_are_refused(ctx):
    p = ctx.model.model.layers[1].self_attn.q_proj
    p.weight = torch.nn.Parameter(p.weight.data.to(torch.float8_e4m3fn), requires_grad=False)
    with pytest.raises(nz.QuantizedParamError, match="float8"):
        nz.WeightNoise(ctx, 0.1, 0).plan()


def test_mxfp4_like_gptoss_attention_only_workflow():
    """gpt-oss-120b stores its experts in mxfp4: defaults must refuse loudly; INCLUDE_ATTN perturbs only the bf16 attention weights."""
    try:
        model, layers_fn, _ = _tiny_arch("gptoss")
    except (ImportError, AttributeError) as e:
        pytest.skip(str(e))
    for layer in model.model.layers:                                     # mimic packed expert storage
        layer.mlp.experts.gate_up_proj = torch.nn.Parameter(layer.mlp.experts.gate_up_proj.data.to(torch.uint8), requires_grad=False)
        layer.mlp.experts.down_proj = torch.nn.Parameter(layer.mlp.experts.down_proj.data.to(torch.uint8), requires_grad=False)
    c = fake_ctx(model, layers_fn)
    ref = snap(model)
    with pytest.raises(nz.QuantizedParamError, match=r"gate_up_proj.*selection_preset\('attn'\)"):       # the error names the way out
        nz.WeightNoise(c, 0.1, 0).plan()
    with pytest.raises(nz.QuantizedParamError, match="gate_up_proj"):
        nz.WeightNoise(c, 0.1, 0, **nz.selection_preset("default")).plan()
    with nz.WeightNoise(c, 0.1, 0, include=nz.INCLUDE_ATTN) as wn:
        assert len(wn.report) == 8 and all(".self_attn." in n and n.endswith("_proj.weight") for n in wn.report)
        cur = snap(model)
        assert all(torch.equal(cur[k], ref[k]) for k in ref if k not in wn.report)
    assert same_state(model, ref)
    with nz.WeightSnapshot(c, **nz.selection_preset("attn")) as s:                       # the preset also drives a whole sweep's snapshot
        attn = nz.select_params(c, nz.INCLUDE_ATTN)
        assert [n for n, _, _ in s.entries] == [n for n, _ in attn] and s.nbytes == sum(p.numel() * p.element_size() for _, p in attn)
        with s.noise(0.1, 0) as wn2:
            assert set(wn2.report) == set(wn.report)
    assert same_state(model, ref)


def test_ram_check_refuses_before_touching_anything(ctx, monkeypatch):
    ref = snap(ctx.model)
    monkeypatch.setattr(nz, "_available_ram_bytes", lambda: 1000)
    with pytest.raises(MemoryError, match="host RAM"):
        with nz.WeightNoise(ctx, 0.1, 0):
            pytest.fail("must not enter")
    assert same_state(ctx.model, ref) and getattr(ctx.model, nz._ACTIVE_ATTR, None) is None
    with nz.WeightNoise(ctx, 0.1, 0, check_ram=False):                  # explicit opt-out
        pass
    monkeypatch.setattr(nz, "_available_ram_bytes", lambda: 600_000 / 0.8 - 1)     # snapshot is 589824 B: just fits...
    with nz.WeightNoise(ctx, 0.1, 0):
        pass
    monkeypatch.setattr(nz, "_available_ram_bytes", lambda: 589_823 / 0.8)         # ... and just does not fit
    with pytest.raises(MemoryError):
        nz.WeightNoise(ctx, 0.1, 0).__enter__()
    assert same_state(ctx.model, ref)


def test_unknown_ram_is_an_error_not_a_silent_pass(ctx, monkeypatch):
    monkeypatch.setattr(nz, "_available_ram_bytes", lambda: None)
    with pytest.raises(RuntimeError, match="cannot determine"):
        nz.WeightNoise(ctx, 0.1, 0).__enter__()
    with nz.WeightNoise(ctx, 0.1, 0, check_ram=False):
        pass


def test_real_ram_probe_returns_something_sane():
    avail = nz._available_ram_bytes()
    assert avail is None or avail > 0


def test_cgroup_headroom_parsing(tmp_path):
    root = tmp_path / "v2"
    root.mkdir()
    (root / "memory.max").write_text("1000\n")
    (root / "memory.current").write_text("400\n")
    (root / "memory.stat").write_text("anon 1\ninactive_file 100\nactive_file 5\n")
    assert nz._cgroup_available_bytes(str(root)) == 700
    (root / "memory.max").write_text("max\n")
    assert nz._cgroup_available_bytes(str(root)) is None
    v1 = tmp_path / "v1"
    (v1 / "memory").mkdir(parents=True)
    (v1 / "memory" / "memory.limit_in_bytes").write_text("2000\n")
    (v1 / "memory" / "memory.usage_in_bytes").write_text("1500\n")
    (v1 / "memory" / "memory.stat").write_text("total_inactive_file 50\n")
    assert nz._cgroup_available_bytes(str(v1)) == 550
    (v1 / "memory" / "memory.limit_in_bytes").write_text(str(2 ** 63 - 4096) + "\n")      # "unlimited" in cgroup v1
    assert nz._cgroup_available_bytes(str(v1)) is None
    assert nz._cgroup_available_bytes(str(tmp_path / "nothing")) is None


def test_no_nesting_and_no_reentry(ctx):
    ref = snap(ctx.model)
    outer = nz.WeightNoise(ctx, 0.1, 0)
    with outer:
        with pytest.raises(RuntimeError, match="another WeightNoise is active"):
            nz.WeightNoise(ctx, 0.1, 1).__enter__()
        with pytest.raises(RuntimeError, match="already active"):
            outer.__enter__()
    assert same_state(ctx.model, ref)
    with nz.WeightNoise(ctx, 0.1, 1):
        pass
    with outer:                                                            # a finished instance can be used again
        pass
    assert same_state(ctx.model, ref)


def test_tied_weights_are_perturbed_once_and_stay_tied(base):
    import transformers as T
    cfg = T.LlamaConfig(vocab_size=300, hidden_size=64, intermediate_size=128, num_hidden_layers=2, num_attention_heads=4,
                        num_key_value_heads=2, tie_word_embeddings=True)
    model = T.LlamaForCausalLM(cfg).eval()
    assert model.lm_head.weight is model.model.embed_tokens.weight
    c = fake_ctx(model, lambda m: m.model.layers)
    ref = snap(model)
    with nz.WeightNoise(c, 0.1, 0, include=nz.PAPER_INCLUDE, exclude=r"norm", restrict_to_layers=False) as wn:
        assert sum("embed_tokens" in n or "lm_head" in n for n in wn.report) == 1
        assert model.lm_head.weight is model.model.embed_tokens.weight
        std_ratio = float((model.lm_head.weight - ref["lm_head.weight"]).std()) / (0.1 * float(ref["lm_head.weight"].pow(2).mean().sqrt()))
        assert std_ratio == pytest.approx(1.0, rel=0.1)                    # perturbed once, not twice (would be sqrt(2))
    assert same_state(model, ref) and model.lm_head.weight is model.model.embed_tokens.weight


# ================================================================================================ host snapshot (RAM accounting)
# Review finding: the snapshot used page-locked memory for CUDA weights. torch 2.2's CUDA host allocator rounds every pinned allocation
# up to a power of two (+12% .. +77% on real configs) and caches freed blocks for the life of the process, so the RAM pre-check on raw
# bytes understated the footprint and, after cell 1, the cached blocks made the check refuse cell 2 and every later cell. The snapshot is
# now ONE pageable buffer of exactly the selected bytes, taken once per sweep, and freed on release. These tests pin that contract.
def odd_model():
    """Awkward tensor sizes and mixed dtypes (70, 18 and 4 bytes per block): exercises the 64-byte aligned layout and dtype views."""
    class Attn(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.q_proj = torch.nn.Linear(5, 7, bias=False).to(torch.bfloat16)
            self.k_proj = torch.nn.Linear(3, 3, bias=False).to(torch.float16)
            self.o_proj = torch.nn.Linear(1, 1, bias=False)

    class Block(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.self_attn = Attn()

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = torch.nn.ModuleList([Block(), Block()])
    torch.manual_seed(0)
    return Model()


def test_snapshot_layout_is_aligned_exact_and_dtype_safe():
    model = odd_model()
    c = fake_ctx(model, lambda m: m.layers)
    sel = nz.select_params(c)
    assert [n for n, _ in sel] == [f"layers.{i}.self_attn.{m}.weight" for i in range(2) for m in ("q_proj", "k_proj", "o_proj")]
    offsets, total = nz._layout(sel)
    assert offsets == [0, 128, 192, 256, 384, 448] and total == 512          # 70 -> 128, 18 -> 64, 4 -> 64 bytes, per block
    assert nz.WeightNoise(c, 0.1, 0).plan()["nbytes"] == total               # what the RAM check compares is what is allocated
    ref = snap(model)
    with nz.WeightSnapshot(c) as s:
        assert s.nbytes == total == s._flat.numel() and s._flat.dtype == torch.uint8
        base = s._flat.data_ptr()
        for (name, p, view), off in zip(s.entries, offsets):
            assert view.dtype == p.dtype and tuple(view.shape) == tuple(p.shape) and view.is_contiguous()
            assert view.data_ptr() - base == off and off % 64 == 0           # views into the ONE flat buffer, aligned
            assert torch.equal(view, p.data) and view.data_ptr() != p.data_ptr()    # exact, independent copy
        with s.noise(0.2, 1) as wn:
            assert not same_state(model, ref) and len(wn.report) == 6
        assert same_state(model, ref)                                        # bf16 / fp16 / fp32 restored bit for bit
    with nz.WeightNoise(c, 0.2, 1):
        pass
    assert same_state(model, ref)


def test_snapshot_is_one_exact_pageable_allocation(ctx, monkeypatch):
    """The ONLY host allocation is a single uint8 buffer of exactly plan()['nbytes']; nothing is page-locked."""
    need = nz.WeightNoise(ctx, 0.1, 0).plan()["nbytes"]
    assert need == 147456 * 4                                                # fp32 tiny model: no padding, no power-of-two rounding
    calls, pinned = [], []
    real_empty = torch.empty

    def spy(*a, **k):
        calls.append((a, k))
        return real_empty(*a, **k)
    with monkeypatch.context() as mp:
        mp.setattr(torch, "empty", spy)
        mp.setattr(torch.Tensor, "pin_memory", lambda self, *a, **k: pinned.append(1) or self)
        with nz.WeightNoise(ctx, 0.1, 0) as wn:
            assert wn.summary["snapshot_bytes"] == need and wn.summary["snapshot"] == "private"
    assert pinned == [] and not any(k.get("pin_memory") for _, k in calls)
    assert [(a, k.get("dtype")) for a, k in calls if k.get("dtype") == torch.uint8] == [((need,), torch.uint8)]    # exactly one
    assert [k.get("device") for _, k in calls if k.get("dtype") == torch.uint8] == ["cpu"]    # host RAM even if the default device is cuda


def test_private_snapshot_is_freed_when_the_context_exits(ctx, monkeypatch):
    refs = []
    real_take = nz.WeightSnapshot.take

    def tracking_take(self):
        out = real_take(self)
        refs.append(weakref.ref(self._flat))
        return out
    monkeypatch.setattr(nz.WeightSnapshot, "take", tracking_take)
    with nz.WeightNoise(ctx, 0.1, 0):
        gc.collect()
        assert refs[0]() is not None                                         # alive while the perturbation is active
    gc.collect()
    assert refs[0]() is None                                                 # normal exit
    with pytest.raises(KeyError):
        with nz.WeightNoise(ctx, 0.1, 1):
            raise KeyError("boom")
    gc.collect()
    assert refs[1]() is None                                                 # exception in the block
    wn = nz.WeightNoise(ctx, 0.1, 2)
    wn.__enter__()
    wn.__exit__(None, None, None)
    gc.collect()
    assert refs[2]() is None and wn._snap is None and wn._snaps == []        # manual enter / exit too


def test_host_memory_is_not_retained_between_cells(ctx, qs, monkeypatch):
    """The reported failure mode: memory kept after cell 1 shrinks the RAM probe and the check refuses cell 2 and every later cell.
    The probe here reports what the live snapshots hold; room for exactly one snapshot, so any retention fails the next cell."""
    live = []
    real_take = nz.WeightSnapshot.take

    def tracking_take(self):
        out = real_take(self)
        live.append(weakref.ref(self._flat))
        return out

    def held():
        gc.collect()
        return sum(r().numel() for r in live if r() is not None)
    monkeypatch.setattr(nz.WeightSnapshot, "take", tracking_take)
    need = nz.WeightNoise(ctx, 0.1, 0).plan()["nbytes"]
    total = int(need / 0.8 * 1.5)                          # one snapshot fits under the 80% limit, a retained one leaves no room
    monkeypatch.setattr(nz, "_available_ram_bytes", lambda: total - held())
    ref = snap(ctx.model)
    for seed in range(4):                                  # one-shot cells, as run.py makes them
        with nz.WeightNoise(ctx, 0.1, seed):
            assert held() == need                          # exactly one snapshot alive, not one per cell so far
        assert held() == 0 and same_state(ctx.model, ref)
    with pytest.raises(KeyError):
        with nz.WeightNoise(ctx, 0.1, 9):
            raise KeyError("boom")
    assert held() == 0
    nz.noise_sweep(ctx, qs, N, [0.0, 0.1, 0.2], [0, 1], "weight", log=NOLOG)         # a sweep: one snapshot for all its cells
    assert held() == 0 and len(live) == 4 + 1 + 1 and same_state(ctx.model, ref)
    monkeypatch.setattr(nz, "_available_ram_bytes", lambda: total - need)             # control: with the memory held, it IS refused
    with pytest.raises(MemoryError):
        nz.WeightNoise(ctx, 0.1, 0).__enter__()


def test_sweep_copies_weights_and_checks_ram_once(ctx, qs, monkeypatch):
    takes, probes = [], []
    real_take, real_probe = nz.WeightSnapshot.take, nz._available_ram_bytes
    monkeypatch.setattr(nz.WeightSnapshot, "take", lambda self: (takes.append(1), real_take(self))[1])
    monkeypatch.setattr(nz, "_available_ram_bytes", lambda: (probes.append(1), real_probe())[1])
    sw = nz.noise_sweep(ctx, qs, N, [0.0, 0.1, 0.2, 0.4], [0, 1, 2], "weight", log=NOLOG)
    assert len(takes) == 1 and len(probes) == 1                                       # 9 perturbed cells, one copy, one check
    assert sw["meta"]["snapshot_bytes"] == 147456 * 4
    nz.noise_sweep(ctx, qs, N, [0.0], [0, 1], "weight", log=NOLOG)                    # all sigma 0: nothing to restore, no copy
    assert len(takes) == 1
    nz.noise_sweep(ctx, qs, N, [0.1], [0], "act_fixed", layers=[1], log=NOLOG)       # activation noise never touches the weights
    assert len(takes) == 1


def test_sweep_refuses_for_lack_of_ram_before_running_any_cell(ctx, qs, monkeypatch):
    ref, called = snap(ctx.model), []
    with monkeypatch.context() as mp:
        mp.setattr(nz, "_available_ram_bytes", lambda: 1000)
        mp.setattr(ctx, "run", lambda *a, **k: called.append(1))
        with pytest.raises(MemoryError, match="host RAM"):
            nz.noise_sweep(ctx, qs, N, [0.0, 0.1], [0], "weight", log=NOLOG)
    assert called == [] and same_state(ctx.model, ref) and getattr(ctx.model, nz._ACTIVE_ATTR, None) is None
    with monkeypatch.context() as mp:                                                 # the explicit opt-out still works
        mp.setattr(nz, "_available_ram_bytes", lambda: 1000)
        assert nz.noise_sweep(ctx, qs, N, [0.0, 0.1], [0], "weight", log=NOLOG, check_ram=False)["corr"].shape == (2, 1, len(qs))


def test_snapshot_ram_check_is_exact_and_can_be_skipped(ctx, monkeypatch):
    monkeypatch.setattr(nz, "_available_ram_bytes", lambda: 589_824 / 0.8)          # exactly the snapshot size: fits
    with nz.WeightSnapshot(ctx) as s:
        assert s.nbytes == 589_824
    monkeypatch.setattr(nz, "_available_ram_bytes", lambda: 589_823 / 0.8)          # one byte short: refused
    with pytest.raises(MemoryError, match="host RAM"):
        nz.WeightSnapshot(ctx).take()
    with nz.WeightSnapshot(ctx, check_ram=False):
        pass
    monkeypatch.setattr(nz, "_available_ram_bytes", lambda: None)
    with pytest.raises(RuntimeError, match="cannot determine"):
        nz.WeightSnapshot(ctx).take()
    with pytest.raises(ValueError):
        nz.WeightSnapshot(ctx, ram_fraction=1.5)


def test_allocation_failure_is_a_clean_memory_error_and_leaves_the_model_alone(ctx, monkeypatch):
    """The OS can refuse the allocation although the pre-check passed: MemoryError, not a bare torch error and not an OOM kill."""
    ref = snap(ctx.model)
    real_empty = torch.empty

    def refuse(*a, **k):
        if k.get("dtype") == torch.uint8:
            raise RuntimeError("[enforce fail at alloc_cpu.cpp] DefaultCPUAllocator: can't allocate memory")
        return real_empty(*a, **k)
    with monkeypatch.context() as mp:
        mp.setattr(torch, "empty", refuse)
        with pytest.raises(MemoryError, match="could not allocate"):
            with nz.WeightNoise(ctx, 0.1, 0):
                pytest.fail("must not enter")
    assert same_state(ctx.model, ref) and getattr(ctx.model, nz._ACTIVE_ATTR, None) is None
    with nz.WeightNoise(ctx, 0.1, 0):                                                 # the model is not stuck
        pass


def test_shared_snapshot_serves_many_cells_bit_exactly_and_equals_the_private_path(ctx):
    ref = snap(ctx.model)
    states = {}
    with nz.WeightSnapshot(ctx) as s:
        assert s.taken and s.valid and s.nbytes == 147456 * 4
        for sigma, seed in [(0.1, 0), (0.1, 1), (0.3, 0), (0.1, 0), (0.0, 5)]:
            with s.noise(sigma, seed) as wn:
                assert wn.summary["snapshot"] == "shared"
                states.setdefault((sigma, seed), []).append(snap(ctx.model))
                assert same_state(ctx.model, ref) == (sigma == 0.0)
            assert same_state(ctx.model, ref)                                        # restored bit-exactly after EVERY cell
        a, b = states[(0.1, 0)]
        assert all(torch.equal(a[k], b[k]) for k in a)                               # same (sigma, seed): same perturbation each time
    assert not s.taken and s.entries == []
    for sigma, seed in [(0.1, 1), (0.3, 0)]:
        with nz.WeightNoise(ctx, sigma, seed):
            private = snap(ctx.model)
        assert all(torch.equal(private[k], states[(sigma, seed)][0][k]) for k in private)     # shared == private, bit for bit


def test_shared_snapshot_refuses_when_the_weights_changed_behind_its_back(ctx):
    """Restoring a stale snapshot would silently revert someone else's edit: the sum-of-squares fingerprint catches it first."""
    with nz.WeightSnapshot(ctx) as s:
        with s.noise(0.1, 0):
            pass
        with torch.no_grad():
            ctx.model.model.layers[1].mlp.up_proj.weight.mul_(1.01)
        edited = snap(ctx.model)
        with pytest.raises(RuntimeError, match="changed since the snapshot"):
            s.noise(0.1, 0).__enter__()
        assert same_state(ctx.model, edited)                                         # nothing perturbed, nothing reverted
        assert getattr(ctx.model, nz._ACTIVE_ATTR, None) is None


def test_shared_snapshot_misuse_is_an_explicit_error(ctx):
    s = nz.WeightSnapshot(ctx, include=nz.INCLUDE_ATTN)
    with pytest.raises(RuntimeError, match="not taken"):
        s.noise(0.1, 0).__enter__()
    with s:
        with pytest.raises(RuntimeError, match="already taken"):
            s.take()
        with pytest.raises(ValueError, match="differs from the selection"):
            nz.WeightNoise(ctx, 0.1, 0, snapshot=s).__enter__()                      # default selection != the attention-only one
        assert getattr(ctx.model, nz._ACTIVE_ATTR, None) is None
        with s.noise(0.1, 0) as wn:                                                  # its own factory is consistent
            assert len(wn.report) == 16
        wn = s.noise(0.1, 0)                                                         # a failed restore poisons the snapshot
        wn.__enter__()
        name, p, buf = wn._snaps[0]
        wn._snaps[0] = (name, p, buf[:, :3].clone())
        with pytest.raises(RuntimeError, match="failed to restore"):
            wn.__exit__(None, None, None)
        assert s.valid is False
        with pytest.raises(RuntimeError, match="invalid"):
            s.noise(0.1, 0).__enter__()
    s.release()
    s.release()                                                                      # idempotent
    with pytest.raises(ValueError, match="no parameters"):
        nz.WeightSnapshot(ctx, include=r"does_not_exist").take()


def test_shared_snapshot_notices_replaced_parameter_objects(ctx):
    """A snapshot holds the Parameter OBJECTS it copied. If the model got new ones (same names, shapes, values) perturbing the old
    objects would silently perturb nothing: that must be an error, not a no-op."""
    with nz.WeightSnapshot(ctx) as s:
        lin = ctx.model.model.layers[0].mlp.up_proj
        lin.weight = torch.nn.Parameter(lin.weight.data.clone(), requires_grad=False)
        ref = snap(ctx.model)
        with pytest.raises(ValueError, match="differs from the selection"):
            s.noise(0.1, 0).__enter__()
        assert same_state(ctx.model, ref) and getattr(ctx.model, nz._ACTIVE_ATTR, None) is None


def test_snapshot_refuses_quantized_selections_like_weightnoise(ctx):
    ctx.model.model.layers[0].mlp.gate_proj.weight = torch.nn.Parameter(
        ctx.model.model.layers[0].mlp.gate_proj.weight.data.to(torch.uint8), requires_grad=False)
    with pytest.raises(nz.QuantizedParamError):
        nz.WeightSnapshot(ctx).take()
    with nz.WeightSnapshot(ctx, **nz.selection_preset("attn")) as s:                 # the attention-only preset is fine
        assert len(s.entries) == 16


def test_selection_presets(ctx):
    assert set(nz.SELECTION_PRESETS) == {"default", "attn", "mlp", "paper"}
    names = lambda name: [n for n, _ in nz.select_params(ctx, **nz.selection_preset(name))]  # noqa: E731
    assert names("default") == [n for n, _ in nz.select_params(ctx)]
    assert names("attn") == [n for n, _ in nz.select_params(ctx, nz.INCLUDE_ATTN)] and len(names("attn")) == 16
    assert len(names("mlp")) == 12 and set(names("attn")) | set(names("mlp")) == set(names("default"))
    assert set(names("paper")) == set(dict(ctx.model.named_parameters()))            # every parameter, like the paper
    preset = nz.selection_preset("attn")
    preset["include"] = "changed"
    assert nz.SELECTION_PRESETS["attn"]["include"] == nz.INCLUDE_ATTN                # a copy: callers cannot corrupt the table
    with pytest.raises(ValueError, match="unknown weight selection.*attn"):
        nz.selection_preset("gpt-oss")
    with nz.WeightNoise(ctx, 0.1, 0, **nz.selection_preset("attn")) as wn:           # usable as WeightNoise keyword arguments
        assert len(wn.report) == 16
    assert nz.noise_sweep(ctx, toy_questions(3), N, [0.1], [0], "weight", log=NOLOG, **nz.selection_preset("attn"))["meta"]["include"] \
        == nz.INCLUDE_ATTN


# ================================================================================================ activation noise
def test_fixed_mode_same_vector_everywhere(ctx):
    U, sigma = make_U(ctx, 3.0), 0.4
    specs = nz.act_noise_specs(ctx, U, [2], sigma, 7, "fixed")
    assert [L for L, _ in specs] == [2]
    fn = specs[0][1]
    z = torch.zeros(3, 6, ctx.H)
    out = fn(z)
    assert torch.equal(out[0, 0], out[2, 5]) and torch.equal(out[0, 1], out[1, 4])       # all positions and batch rows identical
    assert torch.allclose(out.norm(dim=-1), torch.full((3, 6), sigma * 3.0), rtol=1e-5)  # norm = sigma * U[L]
    assert torch.equal(fn(z), out)                                                       # all calls (decoding steps) identical
    assert torch.equal(fn(torch.zeros(1, 1, ctx.H))[0, 0], out[0, 0])                    # incl. the T=1 decode-step shape
    h = torch.randn(2, 4, ctx.H)
    assert torch.allclose(fn(h) - h, out[:2, :4], atol=1e-6)                             # purely additive


def test_iid_mode_fresh_noise_per_position_and_call(ctx):
    U, sigma = make_U(ctx, 3.0), 0.4
    fn = nz.act_noise_specs(ctx, U, [2], sigma, 7, "iid")[0][1]
    z = torch.zeros(3, 6, ctx.H)
    o1, o2 = fn(z), fn(z)
    assert torch.allclose(o1.norm(dim=-1), torch.full((3, 6), sigma * 3.0), rtol=1e-4)  # same norm as the fixed mode
    flat = o1.reshape(-1, ctx.H)
    cos = torch.nn.functional.normalize(flat, dim=-1) @ torch.nn.functional.normalize(flat, dim=-1).T
    off = cos[~torch.eye(len(flat), dtype=torch.bool)]
    assert off.abs().max() < 0.6 and abs(float(off.mean())) < 0.05                         # positions / rows are different directions
    assert not torch.allclose(o1, o2)                                                      # every call is fresh
    assert float(o1.mean()) == pytest.approx(0.0, abs=0.02)
    assert list(fn._gens) == [z.device] and fn.calls == 2                                  # one generator, on the hidden-state device


def test_iid_is_reproducible_per_seed_and_resettable(ctx):
    U = make_U(ctx)
    z = torch.zeros(2, 5, ctx.H)

    def stream(seed, n=3):
        fn = nz.act_noise_specs(ctx, U, [1], 0.3, seed, "iid")[0][1]
        return [fn(z) for _ in range(n)], fn
    a, fa = stream(5)
    b, _ = stream(5)
    c, _ = stream(6)
    assert all(torch.equal(x, y) for x, y in zip(a, b))
    assert not any(torch.equal(x, y) for x, y in zip(a, c))
    fa.reset()
    assert torch.equal(fa(z), a[0])                                                        # reset restarts the stream


def test_layers_get_independent_noise_and_noise_of_a_layer_does_not_depend_on_the_other_layers(ctx):
    """Seeds are per (seed, layer): single-layer and multi-layer runs use the same noise at a shared layer (first call of the stream)."""
    U = make_U(ctx)
    z = torch.zeros(1, 2, ctx.H)
    for mode in ("fixed", "iid"):
        many = dict(nz.act_noise_specs(ctx, U, [1, 2, 3], 0.3, 4, mode))
        one = dict(nz.act_noise_specs(ctx, U, [2], 0.3, 4, mode))
        o1, o2, o3 = many[1](z), many[2](z), many[3](z)
        assert not torch.equal(o1, o2) and not torch.equal(o2, o3)                         # independent per layer
        assert torch.equal(o2, one[2](z))                                                  # same layer-2 noise either way
    other_seed = dict(nz.act_noise_specs(ctx, U, [2], 0.3, 5, "fixed"))
    assert not torch.equal(other_seed[2](z), one[2](z))


def test_fixed_mode_equals_random_direction_steering(ctx, qs):
    """act 'fixed' at sigma is exactly core.spec_add with a random unit direction at alpha = sigma (the SPEC's random control)."""
    U = make_U(ctx, 1.7)
    L, sigma, seed = 2, 0.35, 11
    d = nz._fixed_vector(ctx.H, seed, L, 1.0, "cpu").numpy()
    assert np.linalg.norm(d) == pytest.approx(1.0, rel=1e-6)
    ref = core.spec_add(ctx, {L: d}, L, sigma, U)
    mine = nz.act_noise_specs(ctx, U, [L], sigma, seed, "fixed")
    h = torch.randn(2, 3, ctx.H)
    assert torch.allclose(ref[0][1](h), mine[0][1](h), atol=1e-6)
    assert torch.allclose(last_logits(ctx, qs, ref), last_logits(ctx, qs, mine), atol=1e-5)


def test_sigma_scales_the_norm_linearly(ctx):
    U = make_U(ctx, 2.5)
    z = torch.zeros(1, 3, ctx.H)
    for mode in ("fixed", "iid"):
        norms = [float(nz.act_noise_specs(ctx, U, [3], s, 1, mode)[0][1](z).norm(dim=-1).mean()) for s in (0.1, 0.2, 0.8)]
        assert norms == pytest.approx([0.25, 0.5, 2.0], rel=1e-4)


@pytest.mark.parametrize("mode", ["fixed", "iid"])
def test_act_noise_changes_logits_sigma_zero_does_not(ctx, qs, mode):
    U = make_U(ctx, 2.0)
    base_logits = last_logits(ctx, qs)
    specs = nz.act_noise_specs(ctx, U, [1, 2, 3], 0.5, 1, mode)
    assert not torch.allclose(last_logits(ctx, qs, specs), base_logits, atol=1e-4)
    assert nz.act_noise_specs(ctx, U, [1, 2, 3], 0.0, 1, mode) == []                       # no hooks at all
    assert torch.equal(last_logits(ctx, qs, nz.act_noise_specs(ctx, U, [1, 2, 3], 0.0, 1, mode)), base_logits)


@pytest.mark.parametrize("mode", ["fixed", "iid"])
def test_hooks_are_removed_after_use(ctx, qs, mode):
    specs = nz.act_noise_specs(ctx, make_U(ctx), [1, 2, 3], 0.3, 1, mode)
    ctx.run(qs, N, specs)
    assert n_hooks(ctx) == 0
    with pytest.raises(RuntimeError, match="boom"):
        with core.Hooks(ctx, specs):
            assert n_hooks(ctx) == 3
            raise RuntimeError("boom")
    assert n_hooks(ctx) == 0


@pytest.mark.parametrize("mode", ["fixed", "iid"])
def test_act_noise_keeps_hidden_dtype(ctx, qs, mode):
    ctx.model.to(torch.bfloat16)
    specs = nz.act_noise_specs(ctx, make_U(ctx), [2], 0.3, 1, mode)
    h = torch.randn(1, 3, ctx.H).to(torch.bfloat16)
    assert specs[0][1](h).dtype == torch.bfloat16
    assert last_logits(ctx, qs, specs).shape[-1] == len(ctx.tok)


def test_act_noise_runs_are_reproducible_end_to_end(ctx, qs):
    U = make_U(ctx, 2.0)
    for mode in ("fixed", "iid"):
        r = [ctx.run(qs, N, nz.act_noise_specs(ctx, U, [1, 2, 3], 0.6, 3, mode)) for _ in range(2)]
        assert r[0].texts == r[1].texts
        l3 = last_logits(ctx, qs, nz.act_noise_specs(ctx, U, [1, 2, 3], 0.6, 3, mode))
        assert not torch.allclose(l3, last_logits(ctx, qs, nz.act_noise_specs(ctx, U, [1, 2, 3], 0.6, 4, mode)), atol=1e-4)  # seeds differ


def test_act_noise_argument_errors(ctx):
    U = make_U(ctx)
    with pytest.raises(ValueError, match="mode"):
        nz.act_noise_specs(ctx, U, [1], 0.1, 0, "gauss")
    for bad in (-1.0, float("nan")):
        with pytest.raises(ValueError):
            nz.act_noise_specs(ctx, U, [1], bad, 0, "fixed")
    with pytest.raises(ValueError):
        nz.act_noise_specs(ctx, U, [], 0.1, 0, "fixed")
    with pytest.raises(ValueError):
        nz.act_noise_specs(ctx, U, None, 0.1, 0, "fixed")
    with pytest.raises(ValueError, match="outside"):
        nz.act_noise_specs(ctx, U, [0], 0.1, 0, "fixed")
    with pytest.raises(ValueError, match="post-final-norm"):
        nz.act_noise_specs(ctx, {**U, ctx.nl: 1.0}, [ctx.nl], 0.1, 0, "fixed")
    assert nz.act_noise_specs(ctx, {**U, ctx.nl: 1.0}, [ctx.nl], 0.1, 0, "fixed", allow_last_layer=True)
    with pytest.raises(KeyError, match="U has no entry"):
        nz.act_noise_specs(ctx, {1: 1.0}, [2], 0.1, 0, "fixed")
    assert nz.default_act_layers(36) == list(range(10, 29))
    assert nz.default_act_layers(4) == [1, 2, 3] and nz.default_act_layers(2) == [1]
    assert all(L < n for n in (2, 3, 4, 12, 36, 80) for L in nz.default_act_layers(n))


def test_recommended_grids_and_per_layer_scaling(ctx):
    assert set(nz.RECOMMENDED_SIGMAS) == set(nz.KINDS)
    for kind, grid in nz.RECOMMENDED_SIGMAS.items():
        assert list(grid) == sorted(grid) and all(b / a == pytest.approx(2.0) for a, b in zip(grid, grid[1:])), kind
    assert max(nz.RECOMMENDED_SIGMAS["act_fixed"]) < 0.8          # 0.8 * U at one layer is the known collapse point (SPEC section 1)
    assert nz.per_layer_sigma(0.4, 16) == pytest.approx(0.1)
    assert nz.per_layer_sigma(0.2, 1) == 0.2 and nz.per_layer_sigma(0.0, 5) == 0.0
    for bad in (0, -1):
        with pytest.raises(ValueError):
            nz.per_layer_sigma(0.1, bad)
    # accumulated injected norm of m independent fixed vectors ~ sigma_single: check the heuristic numerically
    U = {L: 1.0 for L in range(1, ctx.nl)}
    m = 3
    vecs = [fn(torch.zeros(1, 1, ctx.H))[0, 0] for _, fn in nz.act_noise_specs(ctx, U, range(1, 4), nz.per_layer_sigma(0.4, m), 2, "fixed")]
    assert float(torch.stack(vecs).sum(0).norm()) == pytest.approx(0.4, rel=0.5)         # random-direction sums: ~ within 50% at H=64


def test_seed_hash_is_a_stable_contract():
    """Changing the hashing silently changes every noise vector ever reported: pin it."""
    assert nz._seed64("weight", 3, "model.layers.0.self_attn.q_proj.weight") == 111328825200868049
    assert nz._seed64("act_fixed", 1, 2) == 3036243231433046230
    assert nz._seed64("act_iid", 0, 15) == 2644055899022166644
    assert nz._seed64("sweep", 0, repr(0.1)) == 5389579205713036173
    assert all(0 <= nz._seed64("x", i) < 2 ** 63 for i in range(50))
    assert len({nz._seed64("act_fixed", s, L) for s in range(20) for L in range(20)}) == 400


# ================================================================================================ sweeps
@pytest.mark.parametrize("kind", nz.KINDS)
def test_noise_sweep_structure_baseline_and_cleanup(ctx, qs, kind):
    ref = snap(ctx.model)
    plain = ctx.run(qs, N)
    cells = []
    sw = nz.noise_sweep(ctx, qs, N, [0.0, 0.1, 0.4], [0, 1], kind, log=NOLOG, on_cell=lambda s, d, r: cells.append((s, d)),
                        keep_texts=True)
    assert sw["kind"] == kind and sw["sigmas"] == [0.0, 0.1, 0.4] and sw["seeds"] == [0, 1] and sw["n"] == len(qs)
    assert sw["corr"].shape == (3, 2, len(qs)) and sw["parsed"].shape == sw["m_instr"].shape == (3, 2, len(qs))
    assert sw["acc"].shape == sw["parse_rate"].shape == (3, 2)
    assert sw["qids"] == [q["qid"] for q in qs]
    assert np.array_equal(sw["baseline_corr"], plain.corr)                    # sigma = 0 == plain run
    assert np.array_equal(sw["corr"][0, 0], sw["corr"][0, 1]) and sw["texts"][0][0] == plain.texts
    assert cells == [(0.0, 0), (0.1, 0), (0.1, 1), (0.4, 0), (0.4, 1)]        # sigma 0 executed once
    assert np.allclose(sw["acc"], sw["corr"].mean(-1))
    assert same_state(ctx.model, ref) and n_hooks(ctx) == 0
    # noise matters: some texts at sigma > 0 differ from the unperturbed ones
    assert any(sw["texts"][i][j] != plain.texts for i in (1, 2) for j in (0, 1))
    if kind == "weight":
        assert sw["layers"] is None
        assert sw["meta"]["realised_rel"][1][0] == pytest.approx(0.1, rel=0.05) and sw["meta"]["realised_rel"][0] == [0.0, 0.0]
    else:
        assert sw["layers"] == [1, 2, 3] and set(sw["meta"]["U"]) == {1, 2, 3}


@pytest.mark.parametrize("kind", nz.KINDS)
def test_noise_sweep_is_deterministic(ctx, qs, kind):
    kw = dict(log=NOLOG, keep_texts=True)
    a = nz.noise_sweep(ctx, qs, N, [0.0, 0.5], [3, 4], kind, **kw)
    b = nz.noise_sweep(ctx, qs, N, [0.0, 0.5], [3, 4], kind, **kw)
    assert np.array_equal(a["corr"], b["corr"]) and a["texts"] == b["texts"]
    c = nz.noise_sweep(ctx, qs, N, [0.5], [5, 6], kind, **kw)
    assert c["texts"][0] != a["texts"][1]                                      # other seeds, other perturbations


def test_noise_sweep_restores_everything_when_a_cell_fails(ctx, qs, monkeypatch):
    """The failure happens INSIDE generation (hooks registered / weights perturbed), the worst place."""
    ref = snap(ctx.model)
    real = ctx._gen_chunk
    n = {"calls": 0}

    def flaky(*a, **k):
        n["calls"] += 1
        if n["calls"] == 3:
            assert n_hooks(ctx) > 0 or not str(n.get("kind")).startswith("act")       # hooks really were active
            raise RuntimeError("cell failed")
        return real(*a, **k)
    monkeypatch.setattr(ctx, "_gen_chunk", flaky)
    for kind in nz.KINDS:
        n["calls"], n["kind"] = 0, kind
        with pytest.raises(RuntimeError, match="cell failed"):
            nz.noise_sweep(ctx, qs, N, [0.1, 0.3], [0, 1], kind, log=NOLOG)
        assert n["calls"] == 3
        assert same_state(ctx.model, ref) and n_hooks(ctx) == 0
        assert getattr(ctx.model, nz._ACTIVE_ATTR, None) is None


def test_noise_sweep_refuses_bad_selection_before_running_anything(ctx, qs, monkeypatch):
    called = []
    monkeypatch.setattr(ctx, "run", lambda *a, **k: called.append(1))
    ctx.model.model.layers[0].mlp.up_proj.weight = torch.nn.Parameter(
        ctx.model.model.layers[0].mlp.up_proj.weight.data.to(torch.uint8), requires_grad=False)
    with pytest.raises(nz.QuantizedParamError):
        nz.noise_sweep(ctx, qs, N, [0.1], [0], "weight", log=NOLOG)
    assert called == []


def test_noise_sweep_argument_validation(ctx, qs):
    for kw, exc in ((dict(kind="act_gauss"), ValueError), (dict(sigmas=[0.1, 0.1]), ValueError), (dict(seeds=[1, 1]), ValueError),
                    (dict(sigmas=[]), ValueError), (dict(seeds=[]), ValueError), (dict(sigmas=[-0.1]), ValueError),
                    (dict(layers=[ctx.nl]), ValueError), (dict(layers=[0]), ValueError)):
        args = dict(sigmas=[0.1], seeds=[0], kind="act_fixed")
        args.update(kw)
        with pytest.raises(exc):
            nz.noise_sweep(ctx, qs, N, args.pop("sigmas"), args.pop("seeds"), args.pop("kind"), log=NOLOG, **args)
    with pytest.raises(ValueError, match="empty"):
        nz.noise_sweep(ctx, [], N, [0.1], [0], "weight", log=NOLOG)
    with pytest.raises(TypeError, match="core.Framing"):
        nz.noise_sweep(ctx, qs, "N", [0.1], [0], "weight", log=NOLOG)
    assert nz.noise_sweep(ctx, qs, None, [0.1], [0], "weight", log=NOLOG)["framing"] == "N"       # None = neutral


def test_noise_sweep_calibrates_U_like_neutral_stats_unless_given(ctx, qs, monkeypatch):
    sw = nz.noise_sweep(ctx, qs, N, [0.1], [0], "act_fixed", layers=[1, 3], log=NOLOG, n_cal=6)
    U, _ = ctx.neutral_stats(ctx.prompts(qs[:6], core.NEUTRAL), {}, [1, 3])
    assert sw["meta"]["U"] == pytest.approx(U) and sw["layers"] == [1, 3]
    monkeypatch.setattr(ctx, "neutral_stats", lambda *a, **k: pytest.fail("must not calibrate when U is given"))
    sw2 = nz.noise_sweep(ctx, qs, N, [0.1], [0], "act_fixed", layers=[1, 3], U={1: 5.0, 3: 6.0}, log=NOLOG)
    assert sw2["meta"]["U"] == {1: 5.0, 3: 6.0}


def test_noise_sweep_default_layers_and_layers_for_weights(ctx, qs, monkeypatch):
    sw = nz.noise_sweep(ctx, qs, N, [0.2], [0], "act_iid", log=NOLOG)
    assert sw["layers"] == nz.default_act_layers(ctx.nl)
    ref = snap(ctx.model)
    seen, real = [], nz._perturb_

    def spy(p, std, seed64):
        seen.append(p.shape)
        return real(p, std, seed64)
    monkeypatch.setattr(nz, "_perturb_", spy)
    nz.noise_sweep(ctx, qs, N, [0.2], [0], "weight", layers=[2], log=NOLOG)
    assert len(seen) == 7 and same_state(ctx.model, ref)                       # only decoder layer 2 (7 projections)


def test_fresh_per_sigma_changes_the_seed_only_when_asked(ctx, qs, monkeypatch):
    seeds_seen = []
    real = nz.act_noise_specs
    U = {1: 1.0, 2: 1.0, 3: 1.0}

    def spy(c, U_, layers, sg, sd, mode, **k):
        seeds_seen.append((sg, sd))
        return real(c, U_, layers, sg, sd, mode, **k)
    monkeypatch.setattr(nz, "act_noise_specs", spy)
    run = lambda **kw: nz.noise_sweep(ctx, qs[:3], N, [0.1, 0.2], [7, 8], "act_fixed", U=U, log=NOLOG, **kw)  # noqa: E731
    run()
    assert seeds_seen == [(0.1, 7), (0.1, 8), (0.2, 7), (0.2, 8)]              # common random numbers across sigma (run.py semantics)
    seeds_seen.clear()
    run(fresh_per_sigma=True)
    assert len({sd for _, sd in seeds_seen}) == 4                              # paper: a fresh vector per (sigma, trial)
    first = list(seeds_seen)
    seeds_seen.clear()
    run(fresh_per_sigma=True)
    assert seeds_seen == first                                                  # ... and deterministic


def test_noise_sweep_absolute_scale_and_paper_preset(ctx, qs, monkeypatch):
    ref = snap(ctx.model)
    sw = nz.noise_sweep(ctx, qs, N, [0.0, 0.003], [1], "weight", weight_scale="abs", log=NOLOG)
    with nz.WeightNoise(ctx, None, 1, sigma_abs=0.003):
        manual = ctx.run(qs, N)
    assert np.array_equal(sw["corr"][1, 0], manual.corr) and sw["meta"]["weight_scale"] == "abs"
    for bad_kw in (dict(weight_scale="absolute"), dict(weight_scale="abs", kind="act_fixed")):
        with pytest.raises(ValueError):
            nz.noise_sweep(ctx, qs, N, [0.1], [0], bad_kw.pop("kind", "weight"), log=NOLOG, **bad_kw)

    seen = []

    class Rec(nz.WeightNoise):
        def __enter__(self):
            out = super().__enter__()
            seen.append((self.sigma_abs, self.sigma_rel, self.seed, len(self.report)))
            return out
    monkeypatch.setattr(nz, "WeightNoise", Rec)
    paper = nz.noise_sweep(ctx, qs, N, [0.001, 0.002], [0, 1], "weight", log=NOLOG, **nz.PAPER_WEIGHT_KW)
    assert paper["meta"]["weight_scale"] == "abs" and paper["meta"]["fresh_per_sigma"] is True
    assert len(seen) == 4 and {s[0] for s in seen} == {0.001, 0.002} and all(s[1] is None for s in seen)
    assert len({s[2] for s in seen}) == 4                                            # fresh vector per (sigma, trial)
    assert all(s[3] == len(dict(ctx.model.named_parameters())) for s in seen)       # EVERY parameter, embeddings and norms too
    assert same_state(ctx.model, ref)


def test_noise_sweep_matches_manual_cells(ctx, qs):
    """A sweep cell is exactly what run.py computes for the same (sigma, seed)."""
    sw = nz.noise_sweep(ctx, qs, N, [0.3], [2], "weight", log=NOLOG)
    with nz.WeightNoise(ctx, 0.3, 2):
        manual = ctx.run(qs, N)
    assert np.array_equal(sw["corr"][0, 0], manual.corr)
    sw = nz.noise_sweep(ctx, qs, N, [0.3], [2], "act_fixed", layers=[1, 2], U={1: 1.5, 2: 2.5}, log=NOLOG)
    manual = ctx.run(qs, N, nz.act_noise_specs(ctx, {1: 1.5, 2: 2.5}, [1, 2], 0.3, 2, "fixed"))
    assert np.array_equal(sw["corr"][0, 0], manual.corr)


# ================================================================================================ statistics
def planted_sweep(n=200):
    """Baseline acc 0.40. sigma grid [0, .1, .2, .4], 3 seeds, hand-planted flips relative to the baseline."""
    base = np.zeros(n)
    base[:80] = 1
    wrong, right = np.arange(80, n), np.arange(80)

    def flip(up=0, down=0, offset=0):
        c = base.copy()
        c[wrong[offset:offset + up]] = 1
        c[right[offset:offset + down]] = 0
        return c
    corr = np.zeros((4, 3, n))
    corr[0] = base                                                              # sigma 0 == baseline
    corr[1] = [flip(10), flip(5, 0, 20), base]                                  # small gains
    corr[2] = [flip(30), flip(20, 5), flip(10, 10)]                             # the real hit: seed 0 has b=30, c=0
    corr[3] = np.stack([np.where(np.arange(n) < 20, 1.0, 0.0)] * 3)             # collapsed: acc 0.10 < 0.5 * 0.40
    parsed = np.ones((4, 3, n), bool)
    parsed[3] = np.arange(n) < 60                                               # collapse also hits the parse rate
    return base, dict(kind="weight", sigmas=[0.0, 0.1, 0.2, 0.4], seeds=[0, 1, 2], n=n, corr=corr, parsed=parsed,
                      baseline_corr=base.copy())


def test_improvement_stats_planted_values():
    base, sw = planted_sweep()
    st = nz.improvement_stats(base, sw, n_boot=500, seed=1)
    assert st["n"] == 200 and st["n_sigma"] == 4 and st["n_seeds"] == 3 and st["baseline_acc"] == pytest.approx(0.4)
    per = {e["sigma"]: e for e in st["per_sigma"]}
    assert [e["sigma"] for e in st["per_sigma"]] == [0.0, 0.1, 0.2, 0.4]
    # sigma 0: exactly the baseline
    assert per[0.0]["acc_mean"] == pytest.approx(0.4) and per[0.0]["acc_max"] == pytest.approx(0.4)
    assert per[0.0]["delta_mean"] == 0 and per[0.0]["is_zero"]
    # sigma 0.1: accs 0.45, 0.425, 0.40
    assert per[0.1]["acc_by_seed"] == pytest.approx([0.45, 0.425, 0.40])
    assert per[0.1]["acc_mean"] == pytest.approx((0.45 + 0.425 + 0.4) / 3) and per[0.1]["acc_max"] == pytest.approx(0.45)
    assert per[0.1]["best_seed"] == 0 and per[0.1]["delta_max"] == pytest.approx(0.05)
    # sigma 0.2: accs 0.55, 0.475, 0.40 (seed 2: +10 -10)
    assert per[0.2]["acc_by_seed"] == pytest.approx([0.55, 0.475, 0.40]) and per[0.2]["delta_by_seed"] == pytest.approx([0.15, 0.075, 0.0])
    assert per[0.2]["mcnemar_best"] == dict(b=30, c=0, p=pytest.approx(2.0 ** -30), p_bonferroni=pytest.approx(9 * 2.0 ** -30), n_tests=9)
    assert per[0.0]["mcnemar_best"] == dict(b=0, c=0, p=1.0, p_bonferroni=1.0, n_tests=9)       # sigma 0 IS the baseline: nothing to test
    # paired delta of the seed-averaged correctness equals core.paired_delta on the same arrays (same bootstrap seed)
    m, lo, hi = core.paired_delta(sw["corr"][2].mean(0), base, 500, 1)
    assert per[0.2]["delta_mean"] == pytest.approx(m) and per[0.2]["delta_mean_ci"] == pytest.approx([lo, hi])
    assert lo < m < hi and per[0.2]["delta_mean"] == pytest.approx((0.15 + 0.075) / 3)
    # collapse flags and safe sigmas (accuracy AND parse-rate criteria)
    assert [per[s]["collapsed"] for s in (0.0, 0.1, 0.2, 0.4)] == [False, False, False, True]
    assert st["safe_sigmas"] == [0.1, 0.2]
    # cells: 3 sigma > 0 x 3 seeds, uncorrected McNemar
    assert len(st["cells"]) == 9 and {(c["sigma"], c["seed"]) for c in st["cells"]} == {(s, k) for s in (0.1, 0.2, 0.4) for k in (0, 1, 2)}
    c02 = next(c for c in st["cells"] if (c["sigma"], c["seed"]) == (0.2, 0))
    assert (c02["b"], c02["c"], c02["acc"], c02["delta"]) == (30, 0, pytest.approx(0.55), pytest.approx(0.15))
    # headline: phi = m / b, m = best noised accuracy (0.55) -> 1.375; ceiling 1/b = 2.5
    h = st["headline"]
    assert h["baseline_acc"] == pytest.approx(0.4) and h["m"] == pytest.approx(0.55) and h["phi"] == pytest.approx(1.375)
    assert h["phi_ceiling"] == pytest.approx(2.5) and h["n_cells"] == 9
    # per-trial ratios (one per seed): seed 0 -> .55/.4; seed 1 -> .475/.4; seed 2 -> max(.4, .4, .4, .1)/.4 = 1
    assert h["phi_trials"] == pytest.approx([1.375, 1.1875, 1.0]) and h["phi"] == max(h["phi_trials"])
    assert "phi_ci" not in h                                      # a bootstrap of a selected maximum is not a CI (see the null tests)
    b = h["best"]
    assert (b["sigma"], b["seed"], b["b"], b["c"]) == (0.2, 0, 30, 0) and b["p"] == pytest.approx(2.0 ** -30)
    assert b["p_bonferroni"] == pytest.approx(9 * 2.0 ** -30) and b["n_tests"] == 9 and b["delta"] == pytest.approx(0.15)
    assert b["delta_ci"] == pytest.approx(per[0.2]["delta_max_ci"])
    # reproducible
    assert nz.improvement_stats(base, sw, n_boot=500, seed=1) == st


def test_improvement_stats_format_artifact_columns():
    base, sw = planted_sweep()
    bpar = np.ones(200, bool)
    st = nz.improvement_stats(base, sw, baseline_parsed=bpar, n_boot=200)
    per = {e["sigma"]: e for e in st["per_sigma"]}
    assert st["baseline_parse_rate"] == 1.0
    assert per[0.2]["parse_rate_mean"] == 1.0 and per[0.2]["parse_delta_mean"] == 0.0
    assert per[0.2]["delta_both_parsed_mean"] == pytest.approx(per[0.2]["delta_mean"])         # everything parsed in both runs
    assert per[0.4]["parse_rate_mean"] == pytest.approx(0.3) and per[0.4]["collapsed"]
    # a gain that exists only on questions the baseline could not parse is not a gain on commonly-parsed questions
    bpar2 = np.ones(200, bool)
    bpar2[80:] = False                                                                           # baseline parsed only the 80 right ones
    st2 = nz.improvement_stats(base, sw, baseline_parsed=bpar2, n_boot=200)
    p2 = {e["sigma"]: e for e in st2["per_sigma"]}
    assert p2[0.2]["delta_both_parsed_mean"] <= 0.0 + 1e-12 and p2[0.2]["delta_mean"] > 0.05
    assert p2[0.2]["parse_delta_mean"] == pytest.approx(1.0 - 0.4)
    # parse-rate collapse alone (accuracy intact) also marks the sigma as collapsed
    sw3 = dict(sw, parsed=sw["parsed"].copy())
    sw3["parsed"][2] = np.arange(200) < 30                                                     # 15% parsed at sigma .2
    p3 = {e["sigma"]: e for e in nz.improvement_stats(base, sw3, baseline_parsed=bpar, n_boot=50)["per_sigma"]}
    assert p3[0.2]["collapsed"] and not p3[0.1]["collapsed"] and p3[0.2]["acc_mean"] > 0.4
    # no question parsed in both runs: the both-parsed delta is an explicit NaN, not a numpy empty-slice warning
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        p4 = {e["sigma"]: e for e in nz.improvement_stats(base, sw, baseline_parsed=np.zeros(200, bool), n_boot=50)["per_sigma"]}
    assert math.isnan(p4[0.2]["delta_both_parsed_mean"]) and p4[0.2]["parse_delta_mean"] == pytest.approx(1.0)


def test_improvement_stats_uses_the_sweeps_own_baseline_and_handles_edge_cases():
    base, sw = planted_sweep()
    a = nz.improvement_stats(None, sw, n_boot=100)
    b = nz.improvement_stats(base, sw, n_boot=100)
    assert a["headline"]["phi"] == b["headline"]["phi"] and a["baseline_parse_rate"] is None
    sw_nob = {k: v for k, v in sw.items() if k not in ("baseline_corr",)}
    with pytest.raises(ValueError, match="no baseline"):
        nz.improvement_stats(None, sw_nob)
    with pytest.raises(ValueError, match="baseline_corr has shape"):
        nz.improvement_stats(base[:10], sw)
    with pytest.raises(ValueError, match="shape"):
        nz.improvement_stats(base, {**sw, "corr": sw["corr"][:, :2]})
    only0 = dict(sw, sigmas=[0.0], corr=sw["corr"][:1], parsed=sw["parsed"][:1])
    with pytest.raises(ValueError, match="no sigma > 0"):
        nz.improvement_stats(base, only0)
    # one seed: sd undefined, everything else fine
    one = dict(sw, seeds=[0], corr=sw["corr"][:, :1], parsed=sw["parsed"][:, :1])
    st = nz.improvement_stats(base, one, n_boot=100)
    assert math.isnan(st["per_sigma"][1]["acc_sd"]) and st["headline"]["n_cells"] == 3
    # baseline accuracy 0: ratio undefined, nothing crashes
    zero_base = np.zeros(200)
    st0 = nz.improvement_stats(zero_base, dict(sw, corr=sw["corr"] * 0 + 0.0), n_boot=50)
    assert math.isnan(st0["headline"]["phi"]) and math.isnan(st0["headline"]["phi_ceiling"])
    # output is plain python: json-serialisable
    json.dumps(st)
    json.dumps(nz.improvement_stats(base, sw, baseline_parsed=np.ones(200, bool), n_boot=50))


def test_phi_is_at_least_one_and_bonferroni_never_below_raw_p():
    rng = np.random.RandomState(0)
    for trial in range(8):
        n, acc = 120, rng.uniform(0.2, 0.9)
        base = (rng.rand(n) < acc).astype(float)
        corr = (rng.rand(3, 4, n) < acc * rng.uniform(0.3, 1.0)).astype(float)
        sw = dict(sigmas=[0.1, 0.2, 0.3], seeds=[0, 1, 2, 3], corr=corr)
        st = nz.improvement_stats(base, sw, n_boot=100)
        h = st["headline"]
        assert h["phi"] >= 1.0 - 1e-12 and all(t >= 1.0 - 1e-12 for t in h["phi_trials"]) and "phi_ci" not in h
        assert h["phi"] <= h["phi_ceiling"] + 1e-12
        assert h["best"]["p_bonferroni"] >= h["best"]["p"] and h["n_cells"] == 12
        assert h["best"]["p"] == pytest.approx(min(c["p"] for c in st["cells"] if c["acc"] == max(x["acc"] for x in st["cells"])))


def test_null_model_phi_exceeds_one_by_chance_but_the_corrected_pvalue_stays_calibrated():
    """Honest model: noise re-draws every answer from the same marginal accuracy (no hidden capability to reveal). The paper's
    max-statistic phi still exceeds 1 by selection (Table 1: honest phi up to 1.143), so phi is descriptive; the Bonferroni-corrected
    best-cell McNemar p-value keeps its false-positive rate."""
    rng = np.random.RandomState(1)
    n, K, S, reps = 100, 5, 6, 80
    phis, pbonf = [], []
    for _ in range(reps):
        base = (rng.rand(n) < 0.6).astype(float)
        corr = (rng.rand(S, K, n) < 0.6).astype(float)                         # independent draws, same accuracy
        st = nz.improvement_stats(base, dict(sigmas=[0.1 * (i + 1) for i in range(S)], seeds=list(range(K)), corr=corr), n_boot=50)
        phis.append(st["headline"]["phi"])
        pbonf.append(st["headline"]["best"]["p_bonferroni"])
    assert np.mean(np.array(phis) > 1.05) > 0.5                                # selection alone inflates phi
    assert np.mean(np.array(pbonf) < 0.05) <= 0.05 + 0.06                      # ...but the corrected test is not fooled


def honest_null_sweeps(reps, n, K, S, acc, seed):
    """Honest model: the baseline is Bernoulli(acc) and every one of the S*K noised cells is an independent Bernoulli(acc) draw (the
    noise changed nothing systematic). Any 'significant' result on such a sweep is a false positive."""
    rng = np.random.RandomState(seed)
    for _ in range(reps):
        base = (rng.rand(n) < acc).astype(float)
        corr = (rng.rand(S, K, n) < acc).astype(float)
        yield base, dict(sigmas=[0.1 * (i + 1) for i in range(S)], seeds=list(range(K)), corr=corr)


def test_honest_sweeps_do_not_look_significant_once_the_shown_p_values_are_corrected():
    """Review finding 1. The table's per-sigma p was the best of K seeds, uncorrected: below .05 at SOME sigma in about 30% of honest
    sweeps. The headline interval for phi (bootstrap of the maximum: every resample >= 1) excluded 1 for most honest sweeps. Now every
    p that is shown is Bonferroni-corrected over all noised cells and phi has no interval."""
    reps = 150
    raw_any = corrected_any = headline = 0
    for base, sw in honest_null_sweeps(reps, 200, 5, 5, 0.8, seed=11):
        st = nz.improvement_stats(base, sw, n_boot=20)
        assert "phi_ci" not in st["headline"]
        raw_any += any(e["mcnemar_best"]["p"] < 0.05 for e in st["per_sigma"])
        corrected_any += any(e["mcnemar_best"]["p_bonferroni"] < 0.05 for e in st["per_sigma"])
        headline += st["headline"]["best"]["p_bonferroni"] < 0.05
    assert raw_any / reps > 0.15                      # the raw best-of-K p is NOT evidence (nominal 0.05; observed ~0.3): hence the label
    assert corrected_any / reps <= 0.11               # every corrected row together keeps the family-wise error near 0.05 (observed ~0.02)
    assert headline / reps <= 0.11


def test_phi_has_no_interval_anywhere_and_the_table_labels_its_p_values():
    base, sw = planted_sweep()
    st = nz.improvement_stats(base, sw, baseline_parsed=np.ones(200, bool), n_boot=50)
    assert "phi_ci" not in json.dumps(st) and "phi_ci" not in st["headline"]
    n_cells = st["headline"]["n_cells"]
    for e in st["per_sigma"]:
        mb = e["mcnemar_best"]
        assert mb["n_tests"] == n_cells == 9 and mb["p_bonferroni"] == pytest.approx(min(1.0, n_cells * mb["p"]))
        assert mb["p_bonferroni"] >= mb["p"]
    txt = nz.format_stats(st)
    head = next(line for line in txt.splitlines() if line.startswith("headline"))
    assert "[" not in head and "no CI" in head and "p_bonf=" in head                  # a bare phi, never "phi [lo, hi]"
    table = txt.splitlines()[1]
    assert "p_bonf" in table and "p_best" not in table                                 # the column is the corrected one, and says so
    assert any("n_cells = 9" in line and "p_bonf" in line for line in txt.splitlines())  # ... with its family spelled out
    row = next(line for line in txt.splitlines() if line.lstrip().startswith("0.2 "))
    assert f"{st['per_sigma'][2]['mcnemar_best']['p_bonferroni']:8.3g}" in row


def test_mcnemar_exact_values():
    assert nz.mcnemar_one_sided([1, 1, 0, 0], [0, 0, 1, 1]) == (2, 2, pytest.approx(11 / 16))     # P(Bin(4,.5) >= 2)
    assert nz.mcnemar_one_sided([1, 1, 1], [0, 0, 0]) == (3, 0, pytest.approx(0.125))
    assert nz.mcnemar_one_sided([0, 0], [1, 1])[2] == pytest.approx(1.0)                         # worse is never evidence of a gain
    assert nz.mcnemar_one_sided([1, 0], [1, 0]) == (0, 0, 1.0)                                    # no discordant pairs
    with pytest.raises(ValueError):
        nz.mcnemar_one_sided([1, 0], [1])
    scipy_stats = pytest.importorskip("scipy.stats")
    for b, c in ((7, 3), (30, 12), (0, 5), (50, 50), (120, 80)):
        corr = np.r_[np.ones(b), np.zeros(c), np.ones(5)]
        base = np.r_[np.zeros(b), np.ones(c), np.ones(5)]
        assert nz.mcnemar_one_sided(corr, base)[2] == pytest.approx(scipy_stats.binomtest(b, b + c, 0.5, alternative="greater").pvalue,
                                                                    rel=1e-9)


def test_format_stats_renders_a_table():
    base, sw = planted_sweep()
    txt = nz.format_stats(nz.improvement_stats(base, sw, baseline_parsed=np.ones(200, bool), n_boot=50))
    assert "headline phi=1.375" in txt and "COLL" in txt and txt.count("\n") >= 5
    assert "n/a" in nz.format_stats(nz.improvement_stats(base, {k: v for k, v in sw.items() if k != "parsed"}, n_boot=50))


def test_end_to_end_tiny_sweep_then_stats(ctx, qs):
    """noise_sweep output feeds improvement_stats directly (sigma 0 in the sweep provides the baseline)."""
    sw = nz.noise_sweep(ctx, qs, N, [0.0, 0.2, 0.6], [0, 1, 2], "weight", log=NOLOG)
    st = nz.improvement_stats(None, sw, baseline_parsed=sw["baseline_parsed"], n_boot=100)
    assert st["n"] == len(qs) and len(st["per_sigma"]) == 3 and st["headline"]["n_cells"] == 6
    assert st["per_sigma"][0]["delta_mean"] == 0
    assert math.isnan(st["headline"]["phi"]) or st["headline"]["phi"] >= 1.0       # (nan iff the tiny model got 0 right)
    assert "headline" in nz.format_stats(st)
    json.dumps(st)
