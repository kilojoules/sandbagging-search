#!/usr/bin/env python3
"""Baseline (unsteered) cells of one run directory: every framing's accuracy, refusal rate, accuracy among answered and paired change vs N.
  USE_TF=0 USE_TORCH=1 python3 scripts/baselines_table.py RUN_DIR [--eval wmdp_bio --split disc --top 12]"""
import argparse, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from sbx import core
from sbx.run import Store, ckey

ap = argparse.ArgumentParser()
ap.add_argument("run_dir"); ap.add_argument("--eval", default="wmdp_bio"); ap.add_argument("--split", default="disc"); ap.add_argument("--top", type=int, default=12)
a = ap.parse_args()
st = Store(a.run_dir)
base = {r["framing"]: r for r in st.cells.values() if r["intervention"] == "none" and r["eval"] == a.eval and r["split"] == a.split}
N = base.get("N")
if not N:
    sys.exit("no N baseline")
rows = []
for fr, r in base.items():
    x, y = st.corr(r["key"]), st.corr(N["key"])
    if x is None or y is None or len(x) != len(y):
        continue
    d, lo, hi = core.paired_delta(x, y)
    rows.append((d, lo, hi, fr, r))
rows.sort(key=lambda t: t[0])
print(f"{a.run_dir}: {a.eval}/{a.split}  N acc {N['acc']:.3f} (n={N['n']}), {len(rows)} framings")
print(f"{'framing':32s} {'acc':>6s} {'d vs N [95% CI]':>24s} {'refus':>6s} {'acc|ans':>8s} {'parse':>6s}")
for d, lo, hi, fr, r in rows[:a.top]:
    print(f"{fr:32s} {r['acc']:6.3f} {d:+7.3f} [{lo:+.3f},{hi:+.3f}] {r.get('refusal_rate') or 0:6.2f} {r.get('acc_answered') if r.get('acc_answered') is not None else float('nan'):8.3f} {r.get('parse_rate', float('nan')):6.2f}")
print("... top gains:")
for d, lo, hi, fr, r in rows[-4:]:
    print(f"{fr:32s} {r['acc']:6.3f} {d:+7.3f} [{lo:+.3f},{hi:+.3f}] {r.get('refusal_rate') or 0:6.2f} {r.get('acc_answered') if r.get('acc_answered') is not None else float('nan'):8.3f} {r.get('parse_rate', float('nan')):6.2f}")
