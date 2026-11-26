#!/usr/bin/env python3
"""Framing-screen summary of the gpt-oss-120b batteries 3-5 (results/pod5): accuracy, paired delta vs N, refusal rate, accuracy among
ANSWERED questions, per eval set.  Local CPU only.

  USE_TF=0 USE_TORCH=1 python3 scripts/summarize_batteries.py [--dir results/pod5] [--out results/pod5/framing_screen_120b.md]
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import re
import sys

import numpy as np

FINAL = "<|channel|>final<|message|>"
REF = re.compile(r"can[’']?t help|cannot help|can[’']?t assist|cannot assist|i[’']?m sorry|unable to|can[’']?t comply|won[’']?t|"
                 r"cannot comply|can[’']?t provide|cannot provide", re.I)


def final(t):
    return t.split(FINAL, 1)[1] if FINAL in t else None


def letter(f):
    m = re.match(r"\s*\(?([A-J])\b(?![’'])", f or "")
    return m.group(1) if m else None


def boot_delta(x, y, n=4000, seed=0):
    rng = np.random.RandomState(seed)
    d = x - y
    idx = rng.randint(0, len(d), size=(n, len(d)))
    m = d[idx].mean(1)
    return float(d.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def p_two(b, c):
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="results/pod5")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    res, texts = {}, {}
    for f in sorted(glob.glob(os.path.join(a.dir, "battery[345]*_res.npz"))):
        z = np.load(f)
        for k in z.files:
            res.setdefault(k, z[k])                     # first file wins; N is identical across files (deterministic greedy)
    for f in sorted(glob.glob(os.path.join(a.dir, "battery[345]*_texts.jsonl"))):
        for l in open(f):
            r = json.loads(l)
            texts.setdefault((r["framing"], r["set"]), []).append(r)
    sets = ["wmdp", "mmlu_bio", "mmlu_nonbio", "wmdp_cyber"]
    names = sorted({k.split("__", 1)[1] for k in res})
    out = ["# gpt-oss-120b (reasoning effort low, greedy) framing screen: batteries 3-5", "",
           "acc = accuracy (refusals and unparsed count as wrong); Δ = paired difference vs N on the same questions [95% bootstrap CI]; "
           "refusal = no letter and the visible answer opens with a refusal phrase; acc|ans = accuracy among questions with a letter answer.", ""]
    for s in sets:
        if f"{s}__N" not in res:
            continue
        base = res[f"{s}__N"]
        out += [f"## {s} (n={len(base)})", "", "| framing | acc | Δ vs N [95% CI] | b/c | p (2-sided exact McNemar) | refusal | acc/answered |", "|---|---|---|---|---|---|---|"]
        rows = []
        for fr in names:
            k = f"{s}__{fr}"
            if k not in res or len(res[k]) != len(base):
                continue
            x = res[k]
            d, lo, hi = boot_delta(x, base)
            b, c = int(((x == 1) & (base == 0)).sum()), int(((x == 0) & (base == 1)).sum())
            T = texts.get((fr, s), [])
            nref = nlet = nok = 0
            for r in T:
                fi = final(r["text"])
                if fi is None:
                    continue
                if letter(fi) is None and REF.search(fi[:200]):
                    nref += 1
                elif letter(fi) is not None:
                    nlet += 1
                    nok += int(r["ok"])
            ref = nref / len(T) if T else float("nan")
            aa = nok / nlet if nlet else float("nan")
            rows.append((d, fr, x.mean(), lo, hi, b, c, p_two(b, c), ref, aa))
        for d, fr, acc, lo, hi, b, c, p, ref, aa in sorted(rows, key=lambda t: t[0]):
            out.append(f"| {fr} | {acc:.3f} | {d:+.3f} [{lo:+.3f},{hi:+.3f}] | {b}/{c} | {p:.4f} | {ref:.3f} | {aa:.3f} |")
        out.append("")
    text = "\n".join(out) + "\n"
    if a.out:
        open(a.out, "w").write(text)
    print(text)


if __name__ == "__main__":
    main()
