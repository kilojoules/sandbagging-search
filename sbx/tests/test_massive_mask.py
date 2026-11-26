"""Massive-activation masking (Gemma-3 keeps ~99.8% of its residual norm in ONE dimension): Ctx.massive_mask, core.mask_dirs,
Ctx.neutral_stats(mask=...) and the --mask-massive CLI path (random tiny model: only mechanics are checked)."""
from __future__ import annotations

import json
import os

import numpy as np
import pytest
import torch

from sbx import core, run as R
from sbx.testing import tiny_ctx, toy_questions

MASSIVE_DIM, MASSIVE_VAL = 5, 500.0


@pytest.fixture(scope="module")
def ctx():
    return tiny_ctx(layers=4, hidden=64, seed=3)


def _prompts(ctx, n=8):
    return ctx.prompts(toy_questions(n, 4), core.NEUTRAL)


def _inject(ctx, layer=1, dim=MASSIVE_DIM, val=MASSIVE_VAL):
    """Hook that adds a large constant to one residual dimension from `layer` on (what Gemma's massive dimension looks like).  HF records
    hidden_states[layer] before our forward hook edits that layer's output, so the offset is visible from hidden_states[layer + 1] on."""
    v = torch.zeros(ctx.H)
    v[dim] = val
    return core.Hooks(ctx, [(layer, core.f_add(v))])


def test_massive_mask_finds_the_injected_dimension_and_nothing_else(ctx):
    with _inject(ctx):
        masks = ctx.massive_mask(_prompts(ctx), [2, 3], kappa=20.0, cap=2)
    for L in (2, 3):
        assert masks[L].dtype == bool and masks[L].shape == (ctx.H,)
        assert np.flatnonzero(masks[L]).tolist() == [MASSIVE_DIM], (L, np.flatnonzero(masks[L]))


def test_massive_mask_is_empty_without_a_massive_dimension_and_when_off(ctx):
    masks = ctx.massive_mask(_prompts(ctx), [2, 3], kappa=20.0, cap=2)
    assert all(not m.any() for m in masks.values())
    with _inject(ctx):
        off = ctx.massive_mask(_prompts(ctx), [2, 3], kappa=0.0, cap=2)
    assert all(not m.any() for m in off.values())


def test_mask_dirs_zeroes_the_masked_dims_renormalises_and_flags_dead_layers():
    H = 16
    rng = np.random.RandomState(0)
    bulk = rng.randn(H).astype(np.float32)
    bulk[3] = 0
    bulk /= np.linalg.norm(bulk)
    d = 0.1 * bulk.copy()
    d[3] = np.sqrt(1 - 0.01)                                   # 99% of the unit vector's squared norm on dim 3
    d = d / np.linalg.norm(d)
    dirs = {"B": {1: d, 2: d, 7: d}, "dead": {1: np.eye(H, dtype=np.float32)[3], 2: d}}
    mask3 = np.zeros(H, bool)
    mask3[3] = True
    out, kept = core.mask_dirs(dirs, {1: mask3, 2: mask3})
    assert out["B"][1][3] == 0 and np.linalg.norm(out["B"][1]) == pytest.approx(1.0, abs=1e-5)
    assert np.allclose(out["B"][1], bulk, atol=1e-4)           # the bulk part, unit length
    assert kept["B"][1] == pytest.approx(0.1, abs=1e-3)
    assert out["B"][7] is d                                     # a layer without a mask is passed through unchanged
    assert np.all(out["dead"][1] == 0) and kept["dead"][1] == pytest.approx(0.0, abs=1e-6)     # purely massive: no-op direction
    assert np.linalg.norm(out["dead"][2]) == pytest.approx(1.0, abs=1e-5)


def test_neutral_stats_mask_removes_the_massive_dimension_from_U(ctx):
    prompts = _prompts(ctx, 6)
    d = {"d": {L: np.ones(ctx.H, np.float32) / np.sqrt(ctx.H) for L in (2, 3)}}
    with _inject(ctx):
        U_raw, _ = ctx.neutral_stats(prompts, d, [2, 3], cap=2)
        masks = ctx.massive_mask(prompts, [2, 3], kappa=20.0, cap=2)
        U_m, _ = ctx.neutral_stats(prompts, d, [2, 3], cap=2, mask=masks)
        hs = ctx.hidden_last(prompts)
    for L in (2, 3):
        assert U_raw[L] > 10 * U_m[L]                           # the raw norm is the massive dimension
        h = hs[:, L].astype(np.float32).copy()
        h[:, MASSIVE_DIM] = 0
        assert U_m[L] == pytest.approx(float(np.linalg.norm(h, axis=1).mean()), rel=2e-2)
    with _inject(ctx):
        U_none, _ = ctx.neutral_stats(prompts, d, [2, 3], cap=2, mask=None)
    assert all(U_none[L] == pytest.approx(U_raw[L]) for L in (2, 3))


def test_masked_add_displacement_is_on_the_bulk_scale(ctx):
    """alpha x U_masked x d_masked moves the bulk dims by bulk-sized amounts; the raw recipe would have moved them by massive-sized ones."""
    prompts = _prompts(ctx, 6)
    raw = np.zeros(ctx.H, np.float32)
    raw[MASSIVE_DIM] = 0.9
    raw[10:20] = 0.1
    raw /= np.linalg.norm(raw)
    with _inject(ctx):
        masks = ctx.massive_mask(prompts, [2], kappa=20.0, cap=2)
        d_m, kept = core.mask_dirs({"B": {2: raw}}, masks)
        U_raw, _ = ctx.neutral_stats(prompts, {"B": {2: raw}}, [2], cap=2)
        U_m, _ = ctx.neutral_stats(prompts, d_m, [2], cap=2, mask=masks)
    bulk_raw = np.abs(0.2 * U_raw[2] * raw)[10:20].max()
    bulk_m = np.abs(0.2 * U_m[2] * d_m["B"][2])[10:20].max()
    assert bulk_m < bulk_raw / 10, (bulk_m, bulk_raw)
    assert d_m["B"][2][MASSIVE_DIM] == 0 and kept["B"][2] < 0.5


def test_cli_mask_massive_runs_end_to_end_and_records_the_mask(tmp_path):
    out = str(tmp_path / "m")
    R.main(["--model", "tiny", "--out", out, "--answer-mode", "letters", "--stages", "behavior,dirs,steer", "--evals", "wmdp_bio",
            "--splits", "disc", "--max-per-split", "6", "--framings", "N,E", "--recipes", "sandbag_bio", "--dir-names", "M",
            "--interventions", "cap,proj0,add", "--strengths=-0.2", "--layer-sets", "auto1", "--steer-evals", "wmdp_bio",
            "--steer-framings", "E,N", "--n-pairs", "10", "--max-new", "8", "--mask-massive", "20", "--rand-ctrl", "0", "--rand-seeds", "2"])
    calib = json.load(open(os.path.join(out, "calib.json")))
    assert "mask_massive" in calib and calib["mask_massive"]["kappa"] == 20.0
    assert set(calib["mask_massive"]["dims"]) and "U" in calib
    cells = [json.loads(l) for l in open(os.path.join(out, "cells.jsonl"))]
    assert any(c["intervention"] in ("cap", "proj0", "add") for c in cells)
    args = json.load(open(os.path.join(out, "args.json")))
    assert args.get("mask_massive") == 20.0
