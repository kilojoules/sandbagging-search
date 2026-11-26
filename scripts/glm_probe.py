#!/usr/bin/env python3
"""ONE-LOAD PROBE for a big sharded model, to run on the pod right after the weights are downloaded and BEFORE any budgeted step.
It measures the three numbers that decide the plan (and prints only counts / numbers, never question or answer text):

  1. load time, where the layers / bytes sit, free memory per GPU                                   -> T_load, the batching budget
  2. seconds per decode step at batch 1 / 32 / 120 (generate 8 vs 40 new tokens)                      -> t_step; is it launch-bound?
  3. generate() calls a 120-question cell needs under N / E / pos_bio_hazard / nat_goal_doc framings,
     with the CURRENT core.batches() budgets and with a patched (tok_budget, kv fraction)             -> minutes per native cell
  4. letter-logit accuracy under N and E (forced choice, no generation) + seconds per forward         -> is the letters mode usable?
  5. massive-activation dims per layer (RMS > kappa x median over non-BOS tokens, as Ctx.massive_mask)    -> is --mask-massive needed?

  python glm_probe.py glm-4.5-air                  # on the pod, from /workspace/sandbagging
  python glm_probe.py --tiny                       # CPU self-test of this script (random tiny Llama)
"""
from __future__ import annotations

import argparse
import collections
import os
import sys
import time
import traceback

import numpy as np
import torch

sys.path.insert(0, os.environ.get("SBX_ROOT") or os.getcwd())
from sbx import core                                   # noqa: E402
from sbx.core import Ctx, NEUTRAL                      # noqa: E402
from sbx.framings import FRAMINGS                      # noqa: E402


def sync():
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            torch.cuda.synchronize(i)


def section(title, fn, *a, **k):
    print(f"\n[probe] ===== {title}", flush=True)
    try:
        return fn(*a, **k)
    except Exception:
        print(f"[probe] SECTION FAILED: {title}", flush=True)
        traceback.print_exc()
        return None


def plan_chunks(lp, max_new, heads, kv_per_tok, free_bytes, cap, tok_budget, kv_frac=0.37, prefill_frac=0.16):
    """Replica of core.batches with the three fractions exposed (checked against core.batches for the default values)."""
    prefill_budget, kv_budget = prefill_frac * free_bytes, kv_frac * free_bytes
    out, i = [], 0
    while i < len(lp):
        bs = 1
        while i + bs < len(lp) and bs < cap:
            p, nb = lp[i + bs][0], bs + 1
            if nb * heads * p * p * 2 > prefill_budget or nb * p > tok_budget or nb * kv_per_tok * (p + max_new) > kv_budget:
                break
            bs += 1
        out.append(bs)
        i += bs
    return out


def report_devices(ctx):
    nbytes, ndev = collections.Counter(), collections.Counter()
    for p in ctx.model.parameters():
        nbytes[str(p.device)] += p.numel() * p.element_size()
    for i, layer in enumerate(ctx.adapter.layers(ctx.model)):
        ndev[str(next(layer.parameters()).device)] += 1
    print("[probe] bytes per device (GB):", {k: round(v / 1e9, 1) for k, v in sorted(nbytes.items())}, flush=True)
    print("[probe] decoder layers per device:", dict(sorted(ndev.items())), flush=True)
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            free, total = torch.cuda.mem_get_info(i)
            print(f"[probe] cuda:{i} {torch.cuda.get_device_name(i)} free {free / 1e9:.1f} GB of {total / 1e9:.1f} GB", flush=True)
    print(f"[probe] core._free_bytes() = {core._free_bytes() / 1e9:.1f} GB (min free x #GPUs); kv_per_tok = {ctx.kv_per_tok} B", flush=True)


def decode_speed(ctx, qs, sizes=(1, 32, 120), short=8, long=40):
    res = {}
    prompts = ctx.prompts(qs, NEUTRAL)
    for B in sizes:
        texts = prompts[:B]
        if len(texts) < B:
            continue
        ts = {}
        for mn in (short, long):
            sync()
            t0 = time.perf_counter()
            ctx._gen_chunk(texts, mn)
            sync()
            ts[mn] = time.perf_counter() - t0
        step = (ts[long] - ts[short]) / (long - short)
        res[B] = step
        peak = {i: round(torch.cuda.max_memory_allocated(i) / 1e9, 1) for i in range(torch.cuda.device_count())} if torch.cuda.is_available() else {}
        print(f"[probe] decode B={B:3d}: {short} new tokens {ts[short]:.1f}s, {long} new tokens {ts[long]:.1f}s -> {step:.3f} s/step "
              f"(prefill+overhead ~{ts[short] - short * step:.1f}s); peak allocated GB per GPU {peak}", flush=True)
    return res


def chunk_plans(ctx, qs, max_new, step_s):
    free = core._free_bytes()
    print(f"[probe] chunk plan for {len(qs)} questions, max_new={max_new}, cap=160; one generate() call = {max_new} steps x {step_s:.2f} s "
          f"= {max_new * step_s / 60:.1f} min", flush=True)
    for name in ("N", "E", "pos_bio_hazard", "nat_goal_doc"):
        fr = FRAMINGS[name]
        lp = sorted((ctx.ntok(p), i, p) for i, p in enumerate(ctx.prompts(qs, fr)))
        lens = [x[0] for x in lp]
        cur = [len(c) for c in core.batches(lp, max_new, ctx.heads, ctx.kv_per_tok, free, 160)]
        assert cur == plan_chunks(lp, max_new, ctx.heads, ctx.kv_per_tok, free, 160, 14000), "replica of core.batches differs"
        pat = plan_chunks(lp, max_new, ctx.heads, ctx.kv_per_tok, free, 160, 50000, kv_frac=0.5)
        print(f"[probe]   {name:15s} prompt tokens mean {np.mean(lens):.0f} max {max(lens)} | CURRENT {len(cur)} calls {cur} = "
              f"{len(cur) * max_new * step_s / 60:.1f} min | PATCHED(tok_budget 50000, kv 0.5) {len(pat)} calls {pat} = "
              f"{len(pat) * max_new * step_s / 60:.1f} min", flush=True)


def letters_check(ctx, qs):
    ad = ctx.adapter
    old = (ad.answer_mode, ad.max_new, ctx.letter_mode)
    ad.answer_mode, ad.max_new, ctx.letter_mode = "direct", 8, True            # exactly what sbx.run --answer-mode letters does
    try:
        accs, corr = {}, {}
        for name in ("N", "E"):
            sync()
            t0 = time.perf_counter()
            r = ctx.run(qs, FRAMINGS[name])
            sync()
            corr[name], accs[name] = r.corr, float(r.acc)
            print(f"[probe] letters {name}: acc {r.acc:.3f} on {len(qs)} questions in {time.perf_counter() - t0:.1f}s "
                  f"(= {(time.perf_counter() - t0) / max(1, -(-len(qs) // 32)):.2f}s per forward of <=32 prompts)", flush=True)
        b = int(((corr["E"] == 0) & (corr["N"] == 1)).sum())
        c = int(((corr["E"] == 1) & (corr["N"] == 0)).sum())
        print(f"[probe] letters E drop {accs['N'] - accs['E']:+.3f} (right->wrong {b}, wrong->right {c}); gate = drop >= 0.15 and p < 0.01", flush=True)
    finally:
        ad.answer_mode, ad.max_new, ctx.letter_mode = old


def massive_stats(ctx, qs, kappa=20.0, n=24, cap=8):
    prompts = ctx.prompts(qs[:n], NEUTRAL)
    lp = sorted((ctx.ntok(p), i, p) for i, p in enumerate(prompts))
    ss, last, cnt = {}, collections.defaultdict(list), 0
    with torch.no_grad():
        for s in range(0, len(lp), cap):
            chunk = lp[s:s + cap]
            enc = ctx.adapter.encode(ctx.tok, [c[2] for c in chunk]).to(ctx.model.device)
            try:
                out = ctx.model(**enc, output_hidden_states=True, logits_to_keep=1)
            except TypeError:
                out = ctx.model(**enc, output_hidden_states=True)
            m = enc["attention_mask"].bool()
            keep = m & ~(m & (m.long().cumsum(1) == 1))
            cnt += int(keep.sum())
            for L in range(1, ctx.nl):
                h = out.hidden_states[L].float()
                if L not in ss:
                    ss[L] = torch.zeros(ctx.H, dtype=torch.float64, device=h.device)
                ss[L] += (h[keep.to(h.device)] ** 2).sum(0).double()
                last[L].append(h[:, -1].double().cpu())
            del out
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    rows, worst = [], 0.0
    for L in range(1, ctx.nl):
        sq = ss[L].cpu().numpy()
        rms = np.sqrt(sq / max(cnt, 1))
        med = max(float(np.median(rms)), 1e-8)
        flagged = rms > kappa * med
        frac_tok = float(sq[flagged].sum() / max(sq.sum(), 1e-30))
        lt = torch.cat(last[L]).numpy()
        sql = (lt ** 2).mean(0)
        frac_last = float(sql[flagged].sum() / max(sql.sum(), 1e-30))
        top = np.argsort(-rms)[:3]
        rows.append((L, int(flagged.sum()), frac_tok, frac_last, [(int(d), round(float(rms[d] / med), 1)) for d in top], med))
        if 4 <= L <= ctx.nl - 6:
            worst = max(worst, frac_tok, frac_last)
    print(f"[probe] kappa={kappa}: layer | #dims RMS>kappa x median | share of squared norm in them (all tokens / last prompt token) | top-3 (dim, RMS/median) | median RMS")
    for r in rows:
        if r[0] in (1, 2, 3, 5, 8, 12, 16, 19, 22, 26, 30, 34, 38, 42, 45) or r[1] > 0:
            print(f"[probe]   L{r[0]:02d} | {r[1]:3d} | {r[2]:.3f} / {r[3]:.3f} | {r[4]} | {r[5]:.2f}", flush=True)
    verdict = ("NEEDED: flagged dims carry >= 50% of the squared norm somewhere in layers 4..nl-6 (Gemma-3: 99.8%) -> pass --mask-massive 20 and "
               "read calib.json mask_massive.kept_norm" if worst >= 0.5 else
               "not needed on this evidence (flagged dims, if any, carry < 50% of the norm): --mask-massive 20 is then a near no-op; still cheap insurance"
               if worst >= 0.0 else "n/a")
    print(f"[probe] massive-dimension verdict (max share over layers 4..{ctx.nl - 6}: {worst:.3f}): {verdict}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model", nargs="?", default="glm-4.5-air")
    ap.add_argument("--tiny", action="store_true", help="CPU self-test with a random tiny Llama")
    ap.add_argument("--n", type=int, default=120)
    ap.add_argument("--max-new", type=int, default=640)
    ap.add_argument("--skip-decode", action="store_true")
    a = ap.parse_args()
    t0 = time.perf_counter()
    if a.tiny:
        from sbx.testing import tiny_ctx, toy_questions
        ctx, qs = tiny_ctx(layers=6, hidden=64, mode="cot"), toy_questions(a.n)
        ctx.adapter.max_new = 640
        sizes, short, long = (1, 8, 24), 4, 12
    else:
        from sbx.adapters import get_adapter
        from sbx.evals import get_eval
        ad = get_adapter(a.model)
        tok, model = ad.load(None)
        ctx, qs = Ctx(ad, tok, model), get_eval("wmdp_bio", "disc")[:a.n]
        sizes, short, long = (1, 32, 120), 8, 40
    print(f"[probe] model {a.model}: load {time.perf_counter() - t0:.0f} s; nl={ctx.nl} H={ctx.H} heads={ctx.heads}; answer_mode={ctx.adapter.answer_mode}", flush=True)
    section("devices and free memory", report_devices, ctx)
    speed = None if a.skip_decode else section("decode speed (generate 8 vs 40 tokens)", decode_speed, ctx, qs, sizes, short, long)
    step = (speed or {}).get(max(speed)) if speed else None
    section("generate() calls per 120-question cell", chunk_plans, ctx, qs, a.max_new, step if step else 1.0)
    section("letters mode sanity (N / E accuracy, no generation)", letters_check, ctx, qs)
    section("massive-activation dimensions", massive_stats, ctx, qs)
    print(f"\n[probe] DONE in {(time.perf_counter() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
