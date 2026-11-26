#!/usr/bin/env python3
"""Cross-eval table of the validated instruments (output of scripts/xeval_steer.sh):  per bio eval, what each instrument does to the
instructed-sandbag condition E (restoration) and to the DEFAULT condition N.  Splits (disc + rep) are pooled per eval.

  USE_TF=0 USE_TORCH=1 python3 scripts/xeval_table.py RUN_DIR [RUN_DIR ...] [--out FILE]
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from sbx import core  # noqa: E402
from sbx.run import Store, ckey  # noqa: E402


def pooled(st, ev, framing, d, i, L, s, base):
    """(steered, baseline) per-question arrays concatenated over the splits that have both."""
    xs, ys = [], []
    for split in ("disc", "rep"):
        ks = ckey(ev, split, framing, d, i, L, s, 0) if not base else ckey(ev, split, framing, "none", "none", "-", 0.0, 0)
        kb = ckey(ev, split, framing, "none", "none", "-", 0.0, 0)
        if ks in st.cells and kb in st.cells:
            x, y = st.corr(ks), st.corr(kb)
            if x is not None and y is not None and len(x) == len(y):
                xs.append(x)
                ys.append(y)
    return (np.concatenate(xs), np.concatenate(ys)) if xs else (None, None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="+")
    ap.add_argument("--out", default=None)
    ap.add_argument("--vf", default="E", help="validity framing whose restoration is the second column (E, pos_generic, ...); the default-model column is always N")
    a = ap.parse_args()
    stores = [Store(d) for d in a.dirs if os.path.exists(os.path.join(d, "cells.jsonl"))]
    cells = [(st, r) for st in stores for r in st.cells.values()]
    inst = sorted({(r["direction"], r["intervention"], r["layers"], r["strength"]) for _, r in cells if r["intervention"] != "none" and not r["direction"].startswith("rand")})
    evals = sorted({r["eval"] for _, r in cells})
    L = []
    P = L.append
    P(f"# Cross-eval table ({', '.join(a.dirs)})\n")
    P("Per eval (disc + rep pooled): baseline accuracy N / E; for each instrument the change under the validity framing (restoration) and under N (default), in points, with 95% CI for N.\n")
    for key in inst:
        d, i, ly, s = key
        P(f"\n## instrument: {d} / {i} / layers {ly} / strength {s}\n")
        P(f"| eval | n | N acc | {a.vf} acc | restore under {a.vf} | change under N [95% CI] |")
        P("|---|---|---|---|---|---|")
        rows = []
        for ev in evals:
            for st in stores:
                xe, ye = pooled(st, ev, a.vf, d, i, ly, s, False)
                xn, yn = pooled(st, ev, "N", d, i, ly, s, False)
                if xe is None and xn is None:
                    continue
                en = f"{ye.mean():.3f}" if ye is not None else "-"
                nacc = f"{yn.mean():.3f}" if yn is not None else "-"
                er = f"{(xe.mean() - ye.mean()) * 100:+.1f}" if xe is not None else "-"
                if xn is not None:
                    dn, lo, hi = core.paired_delta(xn, yn)
                    ncell = f"{dn * 100:+.1f} [{lo * 100:+.1f},{hi * 100:+.1f}]"
                    rows.append((ev, dn, lo, hi, (xe.mean() - ye.mean()) if xe is not None else None, len(yn)))
                else:
                    ncell = "-"
                P(f"| {ev} | {len(yn) if yn is not None else len(ye)} | {nacc} | {en} | {er} | {ncell} |")
                break
        valid = [r for r in rows if r[4] is not None and r[4] >= 0.10]
        if valid:
            mx = max(valid, key=lambda r: r[1])
            P(f"\n  -> restores {a.vf} by >= +10 pts on {len(valid)}/{len(rows)} evals; on those, the largest change under N is {mx[1] * 100:+.1f} "
              f"[{mx[2] * 100:+.1f},{mx[3] * 100:+.1f}] ({mx[0]}, n={mx[5]}); evals with N-change CI above 0: "
              f"{sum(1 for r in valid if r[2] > 0)}")
        elif rows:
            P(f"\n  -> restores {a.vf} by >= +10 pts on none of the evals run")
    text = "\n".join(L) + "\n"
    if a.out:
        open(a.out, "w").write(text)
    print(text)


if __name__ == "__main__":
    main()
