"""Recipe sandbag_cot (directions from the generated reasoning tokens): helper and recipe mechanics on the random tiny model."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from sbx import core
from sbx.dirs import cotdir, get_recipes
from sbx.testing import tiny_ctx, toy_questions


@pytest.fixture(scope="module")
def ctx():
    return tiny_ctx(layers=4, hidden=32, seed=5)


def test_registered_but_not_in_all_recipes():
    assert "sandbag_cot" in get_recipes()
    from sbx import run as R
    assert "sandbag_cot" not in R.RECIPE_ALL


def test_cut_at_end_keeps_the_marker_and_drops_the_padding():
    assert cotdir.cut_at_end("Answer: B<end_of_turn><pad><pad>") == "Answer: B<end_of_turn>"
    assert cotdir.cut_at_end("a<|im_end|>b<end_of_turn>") == "a<|im_end|>"
    assert cotdir.cut_at_end("no marker") == "no marker"
    assert cotdir.cut_at_end("xxEOSyy", extra=("EOS",)) == "xxEOS"


def test_gen_state_means_equals_a_manual_teacher_forced_mean_and_ignores_batching(ctx):
    qs = toy_questions(5, 4)
    prompts = ctx.prompts(qs, core.NEUTRAL)
    texts = ["Answer: A<end_of_turn><pad>", "some reasoning here. Answer: C<end_of_turn>", "", "ok", "long " * 12 + "Answer: D<end_of_turn>"]
    layers = [1, 2, 3]
    a = cotdir.gen_state_means(ctx, prompts, texts, layers, cap=1)
    b = cotdir.gen_state_means(ctx, prompts, texts, layers, cap=3)
    assert a.shape == (5, 3, ctx.H) and a.dtype == np.float32
    assert np.isnan(a[2]).all()                                  # empty generation: nothing to average
    assert np.isfinite(a[[0, 1, 3, 4]]).all()
    ok = [0, 1, 3, 4]
    assert np.abs(a[ok] - b[ok]).max() < 2e-3                    # left padding / batch composition changes nothing (float noise only)
    # manual reference for row 1
    full = prompts[1] + cotdir.cut_at_end(texts[1], (ctx.tok.eos_token or "",))
    ng = max(ctx.ntok(full) - ctx.ntok(prompts[1]), 1)
    enc = ctx.adapter.encode(ctx.tok, [full])
    with torch.no_grad():
        hs = ctx.model(**enc, output_hidden_states=True).hidden_states
    for j, L in enumerate(layers):
        ref = hs[L][0, -ng:, :].float().mean(0).numpy()
        assert np.allclose(a[1, j], ref, atol=2e-3)


def _fake_run(flip_idx, stay_idx):
    """ctx.run stand-in: texts are fixed strings, correctness is controlled per question so that exact questions flip."""
    def run(self, qs, framing=None, specs=None, max_new=None):
        neutral = framing is None or framing.name == "N"
        corr = np.zeros(len(qs))
        for i in range(len(qs)):
            if neutral:
                corr[i] = 1.0 if (i in flip_idx or i in stay_idx) else 0.0
            else:
                corr[i] = 1.0 if i in stay_idx else 0.0
        texts = [("I will answer plainly. " if neutral else "I should hold back here. ") + f"Answer: {'ABCD'[i % 4]}<end_of_turn>" for i in range(len(qs))]
        z = np.zeros(len(qs), bool)
        return core.Result(texts, corr, np.ones(len(qs), bool), z, z.copy(), [""] * len(qs))
    return run


def test_recipe_builds_unit_directions_per_layer_and_caches(ctx, monkeypatch, tmp_path):
    qs = toy_questions(16, 4)
    pool = {"wmdp": qs}
    flip, stay = set(range(0, 7)), set(range(7, 13))
    calls = {"n": 0}
    base = _fake_run(flip, stay)

    def counting(self, *a, **k):
        calls["n"] += 1
        return base(self, *a, **k)
    monkeypatch.setattr(core.Ctx, "run", counting)
    cfg = dict(n_pairs=16, max_new=8, seed=0, cache_dir=str(tmp_path), layers=[1, 2, 3], log=lambda *a, **k: None)
    dirs, meta = cotdir.sandbag_cot(ctx, pool, cfg)
    assert set(dirs) == {"CF", "CS", "CM"}
    for name, d in dirs.items():
        assert sorted(d) == [1, 2, 3]
        for L, v in d.items():
            assert v.shape == (ctx.H,) and np.linalg.norm(v) == pytest.approx(1.0, abs=1e-4)
    assert meta["n_flipped"] == 7 and meta["n_stay"] == 6 and meta["recipe"] == "sandbag_cot" and not meta["omitted"]
    assert meta["neutral_acc"] == pytest.approx(13 / 16) and meta["explicit_acc"] == pytest.approx(6 / 16)
    n_first = calls["n"]
    assert n_first == 2                                          # one generation per framing
    dirs2, _ = cotdir.sandbag_cot(ctx, pool, cfg)
    assert calls["n"] == n_first                                 # second call is a cache hit
    for name in dirs:
        for L in dirs[name]:
            assert np.allclose(dirs[name][L], dirs2[name][L], atol=1e-6)


def test_recipe_omits_flip_directions_when_nothing_flips(ctx, monkeypatch, tmp_path):
    qs = toy_questions(10, 4)
    monkeypatch.setattr(core.Ctx, "run", _fake_run(set(), set(range(10))))
    cfg = dict(n_pairs=10, max_new=8, seed=0, cache_dir=None, layers=[2], log=lambda *a, **k: None)
    dirs, meta = cotdir.sandbag_cot(ctx, {"wmdp": qs}, cfg)
    assert set(dirs) == {"CM"} and sorted(meta["omitted"]) == ["CF", "CS"]
    assert any("flipped" in w for w in meta["warnings"])
