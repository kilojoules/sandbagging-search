#!/usr/bin/env python3
"""Pooled (disc+rep) paired change of each steered instrument vs its unsteered baseline, per eval and framing, from one run directory.
  USE_TF=0 USE_TORCH=1 python3 scripts/pooled_cells.py RUN_DIR [RUN_DIR ...]"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from sbx import core
from sbx.run import Store, ckey
for d in sys.argv[1:]:
    st = Store(d)
    inst = sorted({(r["framing"], r["eval"], r["direction"], r["intervention"], r["layers"], r["strength"]) for r in st.cells.values() if r["intervention"] != "none"})
    print(f"== {d}")
    for fr, ev, dn, iv, L, s in inst:
        xs, ys = [], []
        for sp in ("disc", "rep"):
            k = ckey(ev, sp, fr, dn, iv, L, s, 0); b = ckey(ev, sp, fr, "none", "none", "-", 0.0, 0)
            if k in st.cells and b in st.cells:
                xs.append(st.corr(k)); ys.append(st.corr(b))
        if not xs:
            continue
        x, y = np.concatenate(xs), np.concatenate(ys)
        dl, lo, hi = core.paired_delta(x, y)
        print(f"  {fr:22s} {ev:12s} {dn:6s} {iv:5s} L{L:6s} n={len(x):4d} base {y.mean():.3f} -> {x.mean():.3f}  {dl*100:+5.1f} [{lo*100:+5.1f},{hi*100:+5.1f}]  b/c={int(((x==1)&(y==0)).sum())}/{int(((x==0)&(y==1)).sum())}")
