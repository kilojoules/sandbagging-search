#!/usr/bin/env python3
"""Pooled (disc + rep) paired accuracy change between prompt framings, per eval, with 95% CIs (unsteered cells only).

  USE_TF=0 USE_TORCH=1 python3 scripts/framing_pooled.py RUN_DIR [--pairs nat_goal_doc:N,neg_goal_doc_inverted:N,nat_goal_doc:neg_goal_doc_inverted]

`A:B` = accuracy(A) - accuracy(B) on the questions both framings were run on (paired bootstrap, sbx.core.paired_delta).
Also prints the difference of differences  (nat - N) on the bio eval minus (nat - N) on the non-bio eval?  No: it only prints the pairs asked for; combine by hand.
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from sbx import core  # noqa: E402
from sbx.run import Store, ckey  # noqa: E402


def pooled(st, ev, fa, fb):
    xs, ys = [], []
    for sp in ("disc", "rep"):
        ka, kb = ckey(ev, sp, fa, "none", "none", "-", 0.0, 0), ckey(ev, sp, fb, "none", "none", "-", 0.0, 0)
        if ka in st.cells and kb in st.cells:
            x, y = st.corr(ka), st.corr(kb)
            if x is not None and y is not None and len(x) == len(y):
                xs.append(x)
                ys.append(y)
    return (np.concatenate(xs), np.concatenate(ys)) if xs else (None, None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--pairs", default="nat_goal_doc:N,neg_goal_doc_inverted:N,nat_goal_doc:neg_goal_doc_inverted")
    a = ap.parse_args()
    st = Store(a.run)
    evals = sorted({r["eval"] for r in st.cells.values()})
    print(f"{a.run}\n{'eval':14s} {'pair':40s} {'n':>5s} {'acc A':>7s} {'acc B':>7s}   change [95% CI] (points)   b/c")
    for ev in evals:
        for pr in a.pairs.split(","):
            fa, fb = pr.split(":")
            x, y = pooled(st, ev, fa, fb)
            if x is None:
                continue
            d, lo, hi = core.paired_delta(x, y)
            print(f"{ev:14s} {fa + ' - ' + fb:40s} {len(x):5d} {x.mean():7.3f} {y.mean():7.3f}   {d * 100:+6.1f} [{lo * 100:+6.1f},{hi * 100:+6.1f}]   {int(((x == 1) & (y == 0)).sum())}/{int(((x == 0) & (y == 1)).sum())}")


if __name__ == "__main__":
    main()
