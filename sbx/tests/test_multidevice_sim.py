"""Sharded-model device handling WITHOUT a GPU: simdev.py tags tensors with fake cuda devices and raises PyTorch's real cross-device errors, ShardedModel
mimics a model split over two GPUs (accelerate's dispatch).  The entry points that read hidden states must give bit-identical results to the plain
single-device run in every layout; the pre-multi-GPU code raised here (reviewed: hidden_last, massive_mask, neutral_stats)."""
from __future__ import annotations

import numpy as np
import pytest
import torch

import simdev as S
from sbx import core
from sbx.testing import tiny_ctx, toy_questions

LAYERS = [2, 3, 5]
SCENARIOS = [("sim 1-GPU", 99, True), ("sim 2-GPU, outputs moved back to the input device", 3, True), ("sim 2-GPU, per-layer devices", 3, False)]


@pytest.fixture(scope="module")
def base():
    ctx = tiny_ctx(layers=6, hidden=64, seed=3)
    ctx.model.to(torch.bfloat16)
    return ctx


def _mk(base, model):
    return core.Ctx(base.adapter, base.tok, model, log=lambda *a, **k: None)


def _inject(ctx, layer=1, dim=5, val=500.0):
    v = torch.zeros(ctx.H)
    v[dim] = val
    return core.Hooks(ctx, [(layer, core.f_add(v))])


def _same(a, b):
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(_same(a[k], b[k]) for k in a)
    if isinstance(a, tuple):
        return len(a) == len(b) and all(_same(x, y) for x, y in zip(a, b))
    return np.array_equal(np.asarray(a), np.asarray(b))


def _entries(base):
    prompts = base.prompts(toy_questions(7, 4), core.NEUTRAL)
    rng = np.random.RandomState(0)
    dirs = {"d": {L: core.unit(rng.randn(base.H)) for L in LAYERS}, "e": {L: core.unit(rng.randn(base.H)) for L in LAYERS}}
    m = {L: np.zeros(base.H, bool) for L in LAYERS}
    m[2][5] = True
    d_m, _ = core.mask_dirs(dirs, m)
    return {
        "hidden_last": lambda c: c.hidden_last(prompts, cap=3),
        "massive_mask": lambda c: c.massive_mask(prompts, LAYERS, kappa=20.0, cap=3),
        "neutral_stats_masked": lambda c: c.neutral_stats(prompts, d_m, LAYERS, cap=3, mask=m),
        "neutral_stats": lambda c: c.neutral_stats(prompts, dirs, LAYERS, cap=3),
        "letter_logits": lambda c: c.letter_logits(prompts, cap=3),
    }


@pytest.mark.parametrize("name", ["hidden_last", "massive_mask", "neutral_stats_masked", "neutral_stats", "letter_logits"])
@pytest.mark.parametrize("scen,split,same_dev", SCENARIOS, ids=[s[0] for s in SCENARIOS])
def test_sharded_entry_points_equal_the_single_device_result(base, name, scen, split, same_dev):
    fn = _entries(base)[name]
    with _inject(base, 1):
        ref = fn(_mk(base, base.model))
    shard = S.ShardedModel(base.model, split=split, io_same_device=same_dev)
    ctx = _mk(base, shard)
    with S.SimMode():
        with _inject(ctx, 1):
            got = fn(ctx)
    assert _same(got, ref), (name, scen)


def test_massive_mask_with_no_prompts_is_all_false(base):
    masks = base.massive_mask([], LAYERS, kappa=20.0)
    assert all(m.shape == (base.H,) and not m.any() for m in masks.values())


def test_free_bytes_is_min_over_gpus_times_gpu_count(monkeypatch):
    frees = {0: 30e9, 1: 20e9, 2: 25e9}
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 3)
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda i=0: (frees[i], 80e9))
    assert core._free_bytes() == 60e9
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    assert core._free_bytes() == frees[0]                        # one GPU: unchanged behaviour
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert core._free_bytes() == 8e9


def test_oom_in_a_hidden_state_forward_does_not_leave_output_capture_wrappers(base, monkeypatch):
    """transformers' check_model_inputs swaps each layer's forward for a capture closure during output_hidden_states calls and only a normal return
    restores it; core.snapshot_forwards / restore_forwards undo that after a caught OOM so later forwards do not keep appending to a dead buffer."""
    layers = list(base.adapter.layers(base.model))
    before = [m.__dict__.get("forward") for m in layers]
    prompts = base.prompts(toy_questions(2, 4), core.NEUTRAL)
    real_forward = base.model.forward
    state = {"n": 0}

    def flaky(*a, **k):
        state["n"] += 1
        if state["n"] == 1:
            for m in layers:                                    # what the capture decorator leaves behind when the call dies half-way
                m.forward = (lambda orig: (lambda *x, **y: orig(*x, **y)))(m.forward)
            raise torch.cuda.OutOfMemoryError("simulated")
        return real_forward(*a, **k)
    monkeypatch.setattr(base.model, "forward", flaky)
    out = base.hidden_last(prompts)                              # first chunk raises, the handler restores and splits
    assert out.shape[0] == 2 and np.isfinite(out.astype(np.float32)).all()
    assert [m.__dict__.get("forward") for m in layers] == before
