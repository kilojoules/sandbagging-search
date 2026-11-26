#!/usr/bin/env python3
"""
High-power confirmation on gpt-oss-120b (reasoning effort low). Same instrument as gptoss_battery.py, but on LARGE held-out
bio sets disjoint from every direction pool:
  wmdp        : 200 WMDP-bio rows of questions_heldout_eval.jsonl + wmdp_train[680:]   (~593 q)
  mmlu_bio    : MMLU-bio rows of questions_heldout_eval.jsonl                          (200 q)
  mmlupro_bio : data/hard/mmlupro_biology.jsonl[50:]  (first 50 were used for hard-bio directions) (150 q)
Directions: B (WMDP flipped mean) from results/gptoss120b_lean_dirsB.npz, and V/S/H/M from the first battery's
<out>_newdirs.npz (keys like 'V15'). Conditions run set-major so partial results are usable.
Pre-registered readout: one-sided exact McNemar vs N per condition/set, pooled over bio sets; no condition is 'a hit' unless
it also beats the matched random-direction control (N_cap_R / N_sub*_R) and does not help the non-bio control.
"""
import argparse, json, math, os, sys, time, traceback
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gptoss_dir import build, load, boot, short_enough, EXPLICIT  # noqa
from gptoss_battery import (gen_texts, score, calibrate, f_sub, f_proj, SYS, MID)  # noqa


def mcnemar_p(b, c):
    """one-sided exact: P(Bin(b+c, .5) >= b)."""
    n = b + c
    if n == 0:
        return 1.0
    return float(sum(math.comb(n, k) for k in range(b, n + 1)) / 2 ** n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="openai/gpt-oss-120b")
    ap.add_argument("--out", required=True)
    ap.add_argument("--dirs-prefix", default="results/gptoss120b_lean")
    ap.add_argument("--newdirs", default="results/battery_newdirs.npz")
    ap.add_argument("--sets", default="wmdp,mmlu_bio,mmlupro_bio")
    ap.add_argument("--max-new", type=int, default=800)
    ap.add_argument("--only", default=None)
    a = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(a.repo); tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(a.repo, torch_dtype="auto", device_map="auto").eval()
    nl, H = model.config.num_hidden_layers, model.config.hidden_size
    print(f"[load] {a.repo} layers={nl} hidden={H}", flush=True)

    held = load("data/questions_heldout_eval.jsonl")
    ev = {
        "wmdp": short_enough(tok, [q for q in held if q["source"].startswith("wmdp")] + load("data/wmdp_train.jsonl")[680:]),
        "mmlu_bio": short_enough(tok, [q for q in held if q["source"].startswith("mmlu")]),
        "mmlupro_bio": short_enough(tok, load("data/hard/mmlupro_biology.jsonl")[50:]),
    }
    sets = [s for s in a.sets.split(",") if s in ev]
    print("[data] " + str({k: len(v) for k, v in ev.items()}), flush=True)

    zB, zA = np.load(a.dirs_prefix + "_dirsB.npz"), np.load(a.dirs_prefix + "_dirs.npz")
    dirs = {"B": {L: zB[f"d{L}"].astype(np.float32) for L in range(1, nl + 1)}}
    U = {L: float(zA[f"U{L}"]) for L in range(1, nl + 1)}
    if os.path.exists(a.newdirs):
        z = np.load(a.newdirs)
        for k in z.files:
            dirs.setdefault(k[0], {})[int(k[1:])] = z[k].astype(np.float32)
    rng = np.random.RandomState(5)
    dirs["R"] = {L: (lambda r: (r / np.linalg.norm(r)).astype(np.float32))(rng.randn(H)) for L in range(1, nl + 1)}
    print("[dirs] " + str(sorted(dirs)), flush=True)

    calib_p = [build(tok, q, None) for q in short_enough(tok, load("data/wmdp_train.jsonl"))[:40]]
    med = calibrate(tok, model, calib_p, dirs, MID)
    dev = model.device

    def sp_proj(dn, layers, one_sided):
        return [(L, f_proj(torch.tensor(dirs[dn][L], dtype=torch.float32, device=dev), med[dn][L] if one_sided else 0.0, one_sided))
                for L in layers]

    def sp_sub(dn, L, alpha):
        return [(L, f_sub(torch.tensor(-alpha * U[L] * dirs[dn][L], dtype=torch.float32, device=dev)))]

    ALL = ["wmdp", "mmlu_bio", "mmlupro_bio"]
    conds = [  # (name, framing, spec thunk, sets)
        ("N", "N", lambda: None, ALL),
        ("N_cap_B", "N", lambda: sp_proj("B", MID, True), ALL),
        ("N_cap_R", "N", lambda: sp_proj("R", MID, True), ["wmdp"]),
        ("N_sub04_B", "N", lambda: sp_sub("B", 15, 0.4), ALL),
        ("N_sub04_R", "N", lambda: sp_sub("R", 15, 0.4), ["wmdp"]),
        ("N_cap_V", "N", lambda: sp_proj("V", MID, True), ALL),
        ("N_cap_S", "N", lambda: sp_proj("S", MID, True), ALL),
        ("N_proj0_B", "N", lambda: sp_proj("B", MID, False), ALL),
        ("N_sub02_B", "N", lambda: sp_sub("B", 15, 0.2), ALL),
        ("N_sub06_B", "N", lambda: sp_sub("B", 15, 0.6), ALL),
        ("N_sub06_R", "N", lambda: sp_sub("R", 15, 0.6), ["wmdp"]),
        ("C", "C", lambda: None, ALL),
        ("C_cap_B", "C", lambda: sp_proj("B", MID, True), ALL),
        ("E", "E", lambda: None, ["wmdp"]),
        ("E_proj0_B", "E", lambda: sp_proj("B", MID, False), ["wmdp"]),
        ("E_cap_B", "E", lambda: sp_proj("B", MID, True), ["wmdp"]),
    ]
    only = set(a.only.split(",")) if a.only else None
    res = {}
    texts_f = open(a.out + "_texts.jsonl", "a")

    for s in sets:
        for name, fr, thunk, csets in conds:
            if s not in csets or (only and name not in only):
                continue
            if os.path.exists("/workspace/STOP"):
                print("[stop] STOP file", flush=True); np.savez(a.out + "_res.npz", **res); return
            if f"{s}__{name}" in res:
                continue
            try:
                t0 = time.time(); qs = ev[s]
                tx = gen_texts(tok, model, [build(tok, q, SYS[fr]) for q in qs], a.max_new, thunk())
                corr, prate, m1, m2 = score(tx, qs)
                res[f"{s}__{name}"] = corr
                lo, hi = boot(corr)
                print(f"[cond] {name:10s} {s:12s} acc={corr.mean():.3f} [{lo:.3f},{hi:.3f}] n={len(qs)} parse={prate:.2f} ment_instr={m1.mean():.2f} ({(time.time() - t0) / 60:.1f} min)", flush=True)
                if name in ("N", "C"):
                    for q, t, ok in zip(qs, tx, corr):
                        texts_f.write(json.dumps({"cond": name, "set": s, "qid": q.get("qid"), "ok": bool(ok), "text": t[:1800]}) + "\n")
                    texts_f.flush()
                bases = ["N"] + (["C"] if name.startswith("C_") else ["E"] if name.startswith("E_") else [])
                for bname in bases:
                    b = res.get(f"{s}__{bname}")
                    if b is None or bname == name:
                        continue
                    d = corr - b; l2, h2 = boot(d)
                    bb, cc = int(((corr == 1) & (b == 0)).sum()), int(((corr == 0) & (b == 1)).sum())
                    print(f"[delta] {name:10s} {s:12s} vs {bname}: {d.mean():+.3f} [{l2:+.3f},{h2:+.3f}] b={bb} c={cc} p1={mcnemar_p(bb, cc):.3f}", flush=True)
                np.savez(a.out + "_res.npz", **res)
            except Exception:
                print(f"[error] {name} {s}", flush=True); traceback.print_exc(); torch.cuda.empty_cache()

    # pooled over bio sets
    for name, fr, _, csets in conds:
        if name in ("N", "E", "E_proj0_B", "E_cap_B") or not all(f"{s}__{name}" in res for s in ALL if s in csets):
            continue
        ss = [s for s in ALL if s in csets and f"{s}__{name}" in res and f"{s}__N" in res]
        if not ss:
            continue
        d = np.concatenate([res[f"{s}__{name}"] - res[f"{s}__N"] for s in ss]); lo, hi = boot(d)
        bb = int(sum(((res[f"{s}__{name}"] == 1) & (res[f"{s}__N"] == 0)).sum() for s in ss))
        cc = int(sum(((res[f"{s}__{name}"] == 0) & (res[f"{s}__N"] == 1)).sum() for s in ss))
        print(f"[pooled] {name:10s} sets={','.join(ss)} vs N: {d.mean():+.3f} [{lo:+.3f},{hi:+.3f}] b={bb} c={cc} p1={mcnemar_p(bb, cc):.4f}", flush=True)
    np.savez(a.out + "_res.npz", **res)
    print("[done]", flush=True)


if __name__ == "__main__":
    main()
