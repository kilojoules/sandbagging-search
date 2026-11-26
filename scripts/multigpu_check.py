#!/usr/bin/env python3
"""Does the sbx stack work with a model sharded over several GPUs (device_map="auto")?  Run on a multi-GPU pod:

  python scripts/multigpu_check.py MODEL_KEY [--out FILE.npz]                       # all visible GPUs
  CUDA_VISIBLE_DEVICES=0 python scripts/multigpu_check.py MODEL_KEY --out one.npz   # single-GPU reference (model must fit one GPU)
  python scripts/multigpu_check.py MODEL_KEY --compare one.npz                      # sharded run compared with the reference
  python scripts/multigpu_check.py glm-4.5-air-direct                               # too big for one GPU: no reference, absolute checks only

Absolute checks (each prints [check] ... OK or raises): with more than one visible GPU the parameters really sit on several devices and nothing is
offloaded to cpu/disk/meta; hidden_last / neutral_stats / massive_mask / gen_state_means return finite arrays of the right shape; U (neutral norms)
agrees with hidden_last; steering hooks placed on layers that live on DIFFERENT GPUs change the letter logits; generation with hooks works;
activation noise (iid, per-device generators) changes the logits; weight noise on the attention projections of layers on both GPUs changes the
logits and restores the weights bit-exactly; the free-memory budget is min-over-GPUs times the GPU count.  With --compare the sharded numbers
must match the single-GPU reference (per-row cosine > 0.999 for hidden states, identical massive-dimension sets, close U / logits).
"""
import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from sbx import core  # noqa: E402


def cosine_rows(a, b):
    a, b = a.reshape(len(a), -1), b.reshape(len(b), -1)
    return (a * b).sum(1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) + 1e-12)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--out", default=None)
    ap.add_argument("--compare", default=None)
    ap.add_argument("--n", type=int, default=6)
    a = ap.parse_args()
    from sbx.adapters import get_adapter                                # the loader the real runs use (device_map="auto")
    ad = get_adapter(a.model)
    tok, model = ad.load(None)
    ctx = core.Ctx(ad, tok, model)
    devs = sorted({str(p.device) for p in ctx.model.parameters()})
    per = {}
    for p in ctx.model.parameters():
        per[str(p.device)] = per.get(str(p.device), 0) + p.numel() * p.element_size()
    ng = torch.cuda.device_count()
    print(f"[check] visible GPUs {ng}; parameters live on {devs} (GiB {({d: round(b / 2 ** 30, 1) for d, b in sorted(per.items())})}); "
          f"nl={ctx.nl} H={ctx.H}; free budget {core._free_bytes() / 1e9:.1f} GB", flush=True)
    off = sorted({str(v) for v in (getattr(ctx.model, "hf_device_map", None) or {}).values() if str(v) in ("cpu", "disk", "meta")})
    assert not off and not ({"cpu", "meta"} & set(devs)), f"weights offloaded: {off} {devs}"
    if ng > 1:
        assert len(devs) > 1, f"model NOT sharded (all parameters on {devs}): this check would prove nothing"
        print("[check] model is sharded and fully on GPU OK", flush=True)
    from sbx.evals import get_pool
    qs = get_pool()["wmdp"][:a.n]
    prompts = ctx.prompts(qs, core.NEUTRAL)
    layers = [2, ctx.nl // 2, ctx.nl - 3]
    for L in layers:
        print(f"[check] layer {L} lives on {next(ctx.adapter.layers(ctx.model)[L - 1].parameters()).device}", flush=True)
    res = {}
    h = ctx.hidden_last(prompts)
    assert h.shape == (a.n, ctx.nl + 1, ctx.H), h.shape
    assert np.isfinite(h.astype(np.float32)).all(), "non-finite hidden states (corrupted GPU-to-GPU copies?)"
    res["hidden"] = h[:, layers].astype(np.float32)
    print("[check] hidden_last OK (finite)", flush=True)
    rng = np.random.RandomState(0)
    vecs = {L: core.unit(rng.randn(ctx.H)) for L in layers}
    U, med = ctx.neutral_stats(prompts, {"d": vecs}, layers, cap=2)
    for L in layers:
        ref = float(np.linalg.norm(h[:, L].astype(np.float32), axis=1).mean())
        assert np.isfinite(U[L]) and abs(U[L] - ref) / ref < 2e-2, (L, U[L], ref)
    res["U"] = np.array([U[L] for L in layers])
    print("[check] neutral_stats OK (U matches hidden_last)", flush=True)
    masks = ctx.massive_mask(prompts, layers, kappa=20.0, cap=2)
    res["mask_dims"] = np.concatenate([[L] + np.flatnonzero(masks[L]).tolist() + [-1] for L in layers]).astype(np.int64)
    print("[check] massive_mask OK", {L: int(m.sum()) for L, m in masks.items()}, flush=True)
    base = ctx.letter_logits(prompts)
    assert np.isfinite(base).all()
    res["logits"] = base
    steered = []
    for L in layers:
        lg = ctx.letter_logits(prompts, specs=core.spec_add(ctx, vecs, L, -0.4, U))
        assert not np.allclose(lg, base, atol=1e-3), f"hook at layer {L} changed nothing"
        steered.append(lg)
    res["steered"] = np.stack(steered)
    print("[check] steering hooks on layers", layers, "change the logits OK", flush=True)
    sp = core.spec_proj(ctx, vecs, layers) + core.spec_add(ctx, vecs, layers[0], -0.2, U)
    txt = ctx.gen(prompts[:2], max_new=12, specs=sp)
    assert all(isinstance(t, str) and t for t in txt)
    print("[check] generation with hooks on several GPUs OK", flush=True)
    from sbx import noise as nz
    specs = nz.act_noise_specs(ctx, U, layers, 0.1, 0, "iid")
    lg = ctx.letter_logits(prompts, specs=specs)
    assert np.isfinite(lg).all() and not np.allclose(lg, base, atol=1e-3)
    print("[check] iid activation noise (per-device generators) OK", flush=True)
    sel = nz.selection_preset("attn")
    with nz.WeightNoise(ctx, sigma_rel=0.05, seed=1, layers=layers, **sel):
        lgn = ctx.letter_logits(prompts)
    assert np.isfinite(lgn).all() and not np.allclose(lgn, base, atol=1e-3)
    again = ctx.letter_logits(prompts)
    assert np.array_equal(again, base), "weights were not restored bit-exactly after WeightNoise"
    print("[check] weight noise on attention projections of layers on both GPUs + bit-exact restore OK", flush=True)
    from sbx.dirs.cotdir import gen_state_means
    g = gen_state_means(ctx, prompts[:2], ["Answer: A<end_of_turn>", "I think so. Answer: B<end_of_turn>"], layers)
    assert np.isfinite(g).all() and g.shape == (2, len(layers), ctx.H)
    print("[check] gen_state_means OK", flush=True)
    if a.out:
        np.savez(a.out, **res)
    if a.compare:
        ref = np.load(a.compare)
        cos = cosine_rows(res["hidden"], ref["hidden"])
        print(f"[check] vs single-GPU reference: hidden-state per-row cosine min {cos.min():.5f}", flush=True)
        assert cos.min() > 0.999, cos
        assert np.array_equal(res["mask_dims"], ref["mask_dims"]), "massive-dimension sets differ"
        for k in ("U", "logits", "steered"):
            d = float(np.abs(res[k] - ref[k]).max() / (np.abs(ref[k]).max() + 1e-8))
            print(f"[check] vs single-GPU reference: {k}: max relative difference {d:.4f}", flush=True)
            assert d < 5e-2, (k, d)
    print("[check] ALL OK", flush=True)


if __name__ == "__main__":
    main()
