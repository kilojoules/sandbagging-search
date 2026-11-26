#!/usr/bin/env python3
"""One-line-per-model overview of the screening results (WMDP-bio, discovery split, unsteered cells only):
instruction gate (drop under E / pos_bio_plain / pos_bio_hazard), natural trigger nat_goal_doc and its sign-reversed twin (change vs N,
refusal rate, accuracy among answered).  Local, CPU.

  USE_TF=0 USE_TORCH=1 python3 scripts/overview_table.py [--out FILE]
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from sbx import core  # noqa: E402
from sbx.run import Store, ckey  # noqa: E402

RUNS = [  # label, run directory
    ("gpt-oss-120b", "results/sbx/gptoss-120b/native"),
    ("GLM-4.5-Air 107B (direct answers)", "results/sbx/glm-4.5-air-direct/full/native"),
    ("GLM-4.5-Air 107B (visible reasoning)", "results/sbx/glm-4.5-air-direct/cot"),
    ("gpt-oss-20b", "results/sbx/gptoss-20b/native"),
    ("Gemma-3-27B", "results/sbx/gemma-3-27b-it-confirm/native"),
    ("Gemma-3-12B", "results/sbx/gemma-3-12b-it-confirm/native"),
    ("Gemma-3-4B", "results/sbx/gemma-3-4b-it/native"),
    ("Qwen3-32B", "results/sbx/qwen3-32b/native"),
    ("Qwen3-14B-think", "results/sbx/qwen3-14b-think/native"),
    ("Qwen3-8B-think", "results/sbx/qwen3-8b-think-a100/native"),
    ("OLMo-3.1-32B-instruct", "results/sbx/olmo-3.1-32b-instruct/native"),
    ("Llama-3.1-8B", "results/sbx/llama-3.1-8b-instruct/native"),
    ("Llama-3.3-70B bf16 (direct answers)", "results/sbx/glm-4.5-air-direct/llama70b-bf16/letters"),
    ("Llama-3.3-70B bf16 (visible reasoning, n=120)", "results/sbx/glm-4.5-air-direct/llama70b-bf16/native"),
    ("Llama-3.3-70B 4-bit (early screen)", "results/sbx/llama-3.3-70b-instruct/native"),
    ("GLM-4-9B", "results/sbx/glm-4-9b-chat-hf/native"),
    ("Mistral-Small-3.2-24B", "results/sbx/mistral-small-3.2-24b/native"),
    ("phi-4", "results/sbx/phi-4/native"),
    ("DeepSeek-R1-distill-8B", "results/sbx/deepseek-r1-distill-llama-8b/native"),
    ("OLMo-3-7B-think", "results/sbx/olmo-3-7b-think/native"),
]


def cell(st, fr, ev="wmdp_bio", sp="disc"):
    return st.cells.get(ckey(ev, sp, fr, "none", "none", "-", 0.0, 0))


def delta(st, a, b):
    x, y = st.corr(a["key"]), st.corr(b["key"])
    if x is None or y is None or len(x) != len(y):
        return None
    return core.paired_delta(x, y)


def fmt(d):
    return "-" if d is None else f"{d[0] * 100:+.1f} [{d[1] * 100:+.1f},{d[2] * 100:+.1f}]"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    L = ["| model | n | N acc | drop under E | drop under pos_bio_plain | drop under pos_bio_hazard | nat_goal_doc vs N [95% CI] | refusal / acc among answered | twin vs N |",
         "|---|---|---|---|---|---|---|---|---|"]
    for label, d in RUNS:
        if not os.path.exists(os.path.join(d, "cells.jsonl")):
            continue
        st = Store(d)
        n = cell(st, "N")
        if not n:
            continue
        row = [label, str(n["n"]), f"{n['acc']:.3f}"]
        for fr in ("E", "pos_bio_plain", "pos_bio_hazard"):
            c = cell(st, fr)
            dd = delta(st, c, n) if c else None
            row.append("-" if dd is None else f"{dd[0] * 100:+.1f}")
        c = cell(st, "nat_goal_doc")
        dd = delta(st, c, n) if c else None
        row.append(fmt(dd))
        row.append("-" if not c else f"{(c.get('refusal_rate') or 0):.2f} / {c['acc_answered']:.3f}" if c.get("acc_answered") is not None else f"{(c.get('refusal_rate') or 0):.2f} / -")
        t = cell(st, "neg_goal_doc_inverted")
        dt = delta(st, t, n) if t else None
        row.append("-" if dt is None else f"{dt[0] * 100:+.1f}")
        L.append("| " + " | ".join(row) + " |")
    text = "\n".join(L) + "\n"
    if a.out:
        open(a.out, "w").write(text)
    print(text)


if __name__ == "__main__":
    main()
