#!/usr/bin/env python3
"""Figures for the README, drawn from the stored per-question results (light and dark variants, PNG).

  USE_TF=0 USE_TORCH=1 python3 scripts/make_figures.py [--out figures]       (run from the repository root; needs matplotlib)

  fig1-flips    questions that flip wrong->right / right->wrong on the DEFAULT prompt, next to the same instrument when the model IS sandbagging
  fig2-restore  every validated instrument: how much it restores the instructed drop (x) vs how much it changes default accuracy (y), with random directions
  fig3-gemma    Gemma natural trigger: accuracy under neutral / trigger / trigger + reasoning-token cap / random caps / sign-reversed twin
  fig4-screen   the screen of all 17 models (Table 1 of results/FINAL_REPORT.md): accuracy change under each sandbagging framing and the natural trigger

Colours: blue and red are the two arms of the diverging pair (fig 1), blue and orange are two categorical slots (figs 2 and 3); each pair was run through the
palette validator (CVD and normal-vision separation, contrast on both surfaces) before use.  The numbers come from scripts/default_flip_counts.py
(fig 1, fig 2) and straight from the stored arrays (fig 3), so the figures and the README tables cannot drift apart.
"""
from __future__ import annotations

import argparse
import logging
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import PathPatch  # noqa: E402
from matplotlib.path import Path  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import default_bounds as db  # noqa: E402
import default_flip_counts as dfc  # noqa: E402
from sbx import core  # noqa: E402
from sbx.run import Store  # noqa: E402

logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)

DPI = 200
PX = 1 / 96  # one CSS pixel in inches (the mark specs are in px)

THEMES = {
    "light": dict(surface="#fcfcfb", ink="#0b0b0b", ink2="#52514e", muted="#898781", grid="#e1e0d9", axis="#c3c2b7",
                  blue="#2a78d6", red="#e34948", orange="#eb6834"),
    "dark": dict(surface="#1a1a19", ink="#f0efec", ink2="#c3c2b7", muted="#898781", grid="#2c2c2a", axis="#383835",
                 blue="#3987e5", red="#e66767", orange="#d95926"),
}

SHORT = {  # row labels of fig 1, keyed by the names in dfc.HEADLINE
    "gpt-oss-120B": "gpt-oss-120B",
    "gpt-oss-20B": "gpt-oss-20B",
    "GLM-4.5-Air 107B (generic instruction as validity framing)": "GLM-4.5-Air 107B (generic instr.)",
    "GLM-4.5-Air 107B (bio-specific instruction E as validity framing)": "GLM-4.5-Air 107B (bio-specific instr.)",
    "Llama-3.3-70B bf16, direct answers": "Llama-3.3-70B, direct answers",
    "Llama-3.3-70B bf16, visible reasoning": "Llama-3.3-70B, visible reasoning",
    "Gemma-3-12B, direct answers": "Gemma-3-12B, direct answers",
    "Gemma-3-12B, visible reasoning †": "Gemma-3-12B, visible reasoning †",
    "Gemma-3-27B, visible reasoning †": "Gemma-3-27B, visible reasoning †",
    "GLM-4-9B": "GLM-4-9B",
    "Llama-3.1-8B": "Llama-3.1-8B",
}


def style(theme):
    t = THEMES[theme]
    plt.rcParams.update({
        "font.family": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
        "font.size": 9,
        "text.color": t["ink"],
        "axes.labelcolor": t["ink2"],
        "axes.edgecolor": t["axis"],
        "axes.linewidth": 0.75,
        "figure.facecolor": t["surface"],
        "axes.facecolor": t["surface"],
        "savefig.facecolor": t["surface"],
        "xtick.color": t["axis"],
        "ytick.color": t["axis"],
        "xtick.labelcolor": t["ink2"],
        "ytick.labelcolor": t["ink"],
    })
    return t


def new_axes(fig, W, H, left, bottom, width, height):
    """Axes placed in inches, so pixel-sized mark specs can be converted to data units."""
    return fig.add_axes([left / W, bottom / H, width / W, height / H])


def quiet_axes(ax, t, x_grid=True, y_grid=False):
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(t["axis"])
    ax.tick_params(axis="both", length=0, pad=5)
    if x_grid:
        ax.xaxis.grid(True, color=t["grid"], linewidth=0.75, linestyle="-")
    if y_grid:
        ax.yaxis.grid(True, color=t["grid"], linewidth=0.75, linestyle="-")
    ax.set_axisbelow(True)


def hbar(ax, x0, x1, yc, th, color, r_in, W, H):
    """Horizontal bar from the baseline x0 to x1: square at the baseline, rounded (4px) at the data end."""
    if x1 == x0:
        return
    pos = ax.get_position()
    xl, yl = ax.get_xlim(), ax.get_ylim()
    per_in_x, per_in_y = abs(xl[1] - xl[0]) / (pos.width * W), abs(yl[1] - yl[0]) / (pos.height * H)
    s = 1 if x1 > x0 else -1
    w = abs(x1 - x0)
    rx, ry = min(r_in * per_in_x, w), min(r_in * per_in_y, th / 2)
    y0, y1 = yc - th / 2, yc + th / 2
    verts = [(x0, y0), (x1 - s * rx, y0), (x1, y0), (x1, y0 + ry), (x1, y1 - ry), (x1, y1), (x1 - s * rx, y1), (x0, y1), (x0, y0)]
    codes = [Path.MOVETO, Path.LINETO, Path.CURVE3, Path.CURVE3, Path.LINETO, Path.CURVE3, Path.CURVE3, Path.LINETO, Path.CLOSEPOLY]
    ax.add_patch(PathPatch(Path(verts, codes), facecolor=color, edgecolor="none", linewidth=0, zorder=3))


# --------------------------------------------------------------------------------------------------------------------- data
def flip_rows(group_res, pooled_res):
    rows = []
    for kind, label, vf, shown in dfc.HEADLINE:
        if kind == "group":
            r = group_res.get((label, vf))
            if r is None or not r["info"]:
                continue
            best = dfc.best_of(r["info"])
            rows.append(dict(name=SHORT[shown], n=r["n"], b=best["b"], c=best["c"], vb=best["vb"], vc=best["vc"], natural=False))
        else:
            p = pooled_res.get(label)
            if p is None or not p["main"]:
                continue
            m = p["main"][0]
            rows.append(dict(name=SHORT[shown], n=m["N"]["n"], b=m["N"]["b"], c=m["N"]["c"], vb=m["VF"]["b"], vc=m["VF"]["c"], natural=True))
    return rows


def random_pairs(dirs, vf, ev="wmdp_bio", split="disc"):
    """(restoration under vf, change under N) in points for every random-direction cell that has both."""
    per = {}
    for st, r in db.load_group(dirs):
        if not r["direction"].startswith("rand") or r["eval"] != ev or r["split"] != split or db.base_of(st, r) is None:
            continue
        d = db.delta_vs_base(st, r)
        if d is not None:
            per.setdefault((r["direction"], r["intervention"], r["layers"], r["strength"]), {})[r["framing"]] = d[0]
    return [(v[vf] * 100, v["N"] * 100) for v in per.values() if vf in v and "N" in v]


def scatter_points(group_res, pooled_res):
    dirs_of = {g.split("=", 1)[0]: g.split("=", 1)[1].split(",") for g in dfc.DEFAULT_GROUPS}
    inst, rand, natural, rand_nat = [], [], [], []
    for kind, label, vf, shown in dfc.HEADLINE:
        if kind != "group" or (label == "GLM-4.5-Air" and vf == "E"):  # one validity framing per model (the E row repeats an instrument of the generic row)
            continue
        r = group_res.get((label, vf))
        if r is None:
            continue
        for v in r["info"]:
            inst.append((v["restore"] * 100, v["nd"] * 100, label, v["key"]))
        rand += [(x, y, label) for x, y in random_pairs(dirs_of[label], vf)]
    for label, p in pooled_res.items():
        for e in p["main"]:
            if e["VF"] is not None:
                natural.append((e["VF"]["d"] * 100, e["N"]["d"] * 100, label))
        for e in p["rand"]:
            if e["VF"] is not None:
                rand_nat.append((e["VF"]["d"] * 100, e["N"]["d"] * 100, label))
    return inst, rand, natural, rand_nat


def pooled_acc(store, framing, direction="-", iv="none", layers="-", ev="wmdp_bio"):
    """Accuracy over discovery + replication with a 95% Wilson interval; the stored per-question correctness arrays are the source."""
    xs = []
    for r in store.cells.values():
        if r["eval"] == ev and r["split"] in ("disc", "rep") and r["framing"] == framing and r["intervention"] == iv and r["layers"] == layers and (iv == "none" or r["direction"] == direction):
            xs.append(store.corr(r["key"]))
    if not xs:
        return None
    x = np.concatenate(xs)
    n, p, z = len(x), float(x.mean()), 1.96
    den = 1 + z * z / n
    mid = (p + z * z / (2 * n)) / den
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return dict(acc=p, lo=mid - half, hi=mid + half, n=n)


def gemma_rows():
    """Rows of fig 3 for each model: (label, kind, [accuracy dicts]) from the confirmation runs."""
    s = lambda d: Store(f"results/sbx/{d}")  # noqa: E731
    out = {}
    # 12B: N and N+cap from cc2, trigger and trigger+cap and random caps from cc1, twin from cc3
    cc1, cc2, cc3 = s("gemma-3-12b-it-cc1/native"), s("gemma-3-12b-it-cc2/native"), s("gemma-3-12b-it-cc3/native")
    out["Gemma-3-12B"] = [
        ("Neutral prompt", "neutral", [pooled_acc(cc2, "N")]),
        ("Neutral prompt + reasoning-token cap", "neutral", [pooled_acc(cc2, "N", "CF", "cap", "13-23")]),
        ("Natural trigger", "trigger", [pooled_acc(cc1, "nat_goal_doc")]),
        ("Trigger + reasoning-token cap", "restored", [pooled_acc(cc1, "nat_goal_doc", "CF", "cap", "13-23")]),
        ("Trigger + random caps (3)", "random", [pooled_acc(cc1, "nat_goal_doc", f"rand{i}", "cap", "13-23") for i in range(3)]),
        ("Sign-reversed twin of the trigger", "twin", [pooled_acc(cc3, "neg_goal_doc_inverted")]),
    ]
    n2, cf, c27 = s("gemma-3-27b-it-cotN2/native"), s("gemma-3-27b-it-cot/confirm"), s("gemma-3-27b-it-confirm/native")
    out["Gemma-3-27B"] = [
        ("Neutral prompt", "neutral", [pooled_acc(n2, "N")]),
        ("Neutral prompt + reasoning-token cap", "neutral", [pooled_acc(n2, "N", "CF", "cap", "17-30")]),
        ("Natural trigger", "trigger", [pooled_acc(cf, "nat_goal_doc")]),
        ("Trigger + reasoning-token cap", "restored", [pooled_acc(cf, "nat_goal_doc", "CF", "cap", "17-30")]),
        ("Trigger + random caps (3)", "random", []),
        ("Sign-reversed twin of the trigger", "twin", [pooled_acc(c27, "neg_goal_doc_inverted")]),
    ]
    return out


# --------------------------------------------------------------------------------------------------------------------- fig 1
def fig_flips(theme, rows, path):
    t = style(theme)
    n = len(rows)
    W, pitch = 9.1, 0.33
    H = 0.85 + n * pitch + 1.0
    fig = plt.figure(figsize=(W, H), dpi=DPI)
    unit = 0.05  # inches per 1% of questions, the same in both panels (so bar lengths compare directly)
    lim1, lim2 = (-10, 10), (-10, 55)
    a1 = new_axes(fig, W, H, 2.55, 1.0, (lim1[1] - lim1[0]) * unit, n * pitch)
    a2 = new_axes(fig, W, H, 4.8, 1.0, (lim2[1] - lim2[0]) * unit, n * pitch)
    th = 14 * PX
    for ax, lim, ticks, key in ((a1, lim1, [-10, -5, 0, 5, 10], ("b", "c")), (a2, lim2, [-10, 0, 10, 20, 30, 40, 50], ("vb", "vc"))):
        ax.set_xlim(*lim)
        ax.set_ylim(n - 0.5, -0.5)
        quiet_axes(ax, t)
        ax.set_xticks(ticks)
        ax.set_xticklabels([str(abs(v)) for v in ticks], fontsize=8.5)
        ax.axvline(0, color=t["axis"], linewidth=1.0, zorder=2)
        ax.set_xlabel("% of questions", fontsize=8.5, color=t["ink2"], labelpad=6)
        th_data = th / (n * pitch) * n  # bar thickness in row units
        for i, r in enumerate(rows):
            good, bad = r[key[0]], r[key[1]]
            nq = r["n"]
            hbar(ax, 0, 100 * good / nq, i, th_data, t["blue"], 4 * PX, W, H)
            hbar(ax, 0, -100 * bad / nq, i, th_data, t["red"], 4 * PX, W, H)
            ax.text(1.06 if ax is a1 else 1.025, i, f"{good} / {bad}", transform=ax.get_yaxis_transform(), va="center", ha="left", fontsize=8.5, color=t["ink2"])
        ax.text(1.06 if ax is a1 else 1.025, -1.15, "wrong→right /\nright→wrong", transform=ax.get_yaxis_transform(), va="center", ha="left", fontsize=7.5, color=t["muted"], linespacing=1.1)
    a1.set_yticks(range(n))
    a1.set_yticklabels([r["name"] for r in rows], fontsize=9)
    a1.tick_params(axis="y", pad=8)
    a2.set_yticks([])
    fig.text(2.55 / W, (H - 0.28) / H, "On the default prompt", fontsize=10.5, fontweight="bold", va="center")
    fig.text(4.8 / W, (H - 0.28) / H, "Same instrument when the model is sandbagging", fontsize=10.5, fontweight="bold", va="center")
    handles = [Line2D([], [], marker="s", linestyle="", markersize=8, color=t["blue"], label="wrong → right (fixed by the steering)"),
               Line2D([], [], marker="s", linestyle="", markersize=8, color=t["red"], label="right → wrong (broken by the steering)")]
    leg = fig.legend(handles=handles, loc="lower left", bbox_to_anchor=(2.45 / W, 0.28 / H), ncol=2, frameon=False, fontsize=8.5, handletextpad=0.3, columnspacing=1.6)
    for txt in leg.get_texts():
        txt.set_color(t["ink2"])
    fig.text(2.55 / W, 0.14 / H, "† the sandbagging condition is the natural trigger (no instruction to underperform); every other row uses an explicit instruction.",
             fontsize=7.5, color=t["muted"], va="center")
    fig.savefig(path, dpi=DPI)
    plt.close(fig)


# --------------------------------------------------------------------------------------------------------------------- fig 2
def fig_restore(theme, pts, path):
    t = style(theme)
    inst, rand, natural, rand_nat = pts
    W, H = 7.4, 4.9
    fig = plt.figure(figsize=(W, H), dpi=DPI)
    ax = new_axes(fig, W, H, 0.85, 0.8, 6.3, 3.35)
    quiet_axes(ax, t, x_grid=True, y_grid=True)
    xs = [p[0] for p in inst + rand + natural + rand_nat]
    ax.set_xlim(min(xs) - 2.5, max(xs) + 3)
    ys = [p[1] for p in inst + rand + natural + rand_nat]
    ax.set_ylim(min(ys) - 1.5, max(ys) + 1.5)
    ax.axhline(0, color=t["axis"], linewidth=1.0, zorder=2)
    ring = dict(edgecolors=t["surface"], linewidths=1.0)
    ax.scatter([p[0] for p in rand + rand_nat], [p[1] for p in rand + rand_nat], s=46, facecolors="none", edgecolors=t["muted"], linewidths=1.3, zorder=3)
    ax.scatter([p[0] for p in inst], [p[1] for p in inst], s=46, color=t["blue"], alpha=0.85, zorder=4, **ring)
    ax.scatter([p[0] for p in natural], [p[1] for p in natural], s=70, color=t["orange"], marker="D", zorder=5, **ring)
    ax.set_xlabel("Restoration of the instructed (or natural-trigger) drop, points of accuracy", fontsize=9, labelpad=8)
    ax.set_ylabel("Change in default-prompt accuracy, points", fontsize=9, labelpad=8)
    ax.tick_params(axis="both", labelsize=8.5)
    handles = [Line2D([], [], marker="o", linestyle="", markersize=7, markerfacecolor=t["blue"], markeredgecolor=t["surface"], color=t["blue"], label=f"validated instrument, explicit instruction ({len(inst)})"),
               Line2D([], [], marker="D", linestyle="", markersize=7, markerfacecolor=t["orange"], markeredgecolor=t["surface"], color=t["orange"], label=f"validated instrument, natural trigger ({len(natural)})"),
               Line2D([], [], marker="o", linestyle="", markersize=7, markerfacecolor="none", markeredgecolor=t["muted"], markeredgewidth=1.3, label=f"random direction ({len(rand) + len(rand_nat)})")]
    leg = fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.75 / W, (H - 0.12) / H), ncol=2, frameon=False, fontsize=8.5, handletextpad=0.2, columnspacing=1.6, labelspacing=0.5)
    for txt in leg.get_texts():
        txt.set_color(t["ink2"])
    fig.savefig(path, dpi=DPI)
    plt.close(fig)


# --------------------------------------------------------------------------------------------------------------------- fig 3
def fig_gemma(theme, data, path):
    t = style(theme)
    W, H = 9.1, 4.25
    fig = plt.figure(figsize=(W, H), dpi=DPI)
    panel_w, gap, left = 2.75, 0.45, 2.75
    colors = dict(neutral=t["ink"], trigger=t["orange"], restored=t["blue"], random=t["muted"], twin=t["ink2"])
    for k, (model, rows) in enumerate(data.items()):
        ax = new_axes(fig, W, H, left + k * (panel_w + gap), 1.0, panel_w, 2.55)
        ax.set_xlim(0.2, 0.85)
        nrow = len(rows)
        ax.set_ylim(nrow - 0.5, -0.5)
        quiet_axes(ax, t)
        ax.set_xticks([0.2, 0.4, 0.6, 0.8])
        ax.set_xticklabels(["20%", "40%", "60%", "80%"], fontsize=8.5)
        base = rows[0][2][0]["acc"]
        ax.axvline(base, color=t["axis"], linewidth=1.0, zorder=2)  # the neutral level, to read "restored" against
        for i, (label, kind, accs) in enumerate(rows):
            if not accs:
                ax.text(0.5, i, "not run at this sample size", fontsize=8, color=t["muted"], va="center", ha="center")
                continue
            for j, a in enumerate(accs):
                hollow = kind == "random"
                yy = i + (j - (len(accs) - 1) / 2) * 0.2 if hollow else i  # the three random caps sit on separate lines of their row
                if not hollow:
                    ax.plot([a["lo"], a["hi"]], [yy, yy], color=colors[kind], linewidth=2, solid_capstyle="round", zorder=3, alpha=0.9)
                ax.scatter([a["acc"]], [yy], s=58, zorder=4, facecolors=t["surface"] if hollow else colors[kind], edgecolors=colors[kind] if hollow else t["surface"], linewidths=1.4 if hollow else 1.0)
            if kind in ("neutral", "trigger", "restored") and label == rows[{"neutral": 0, "trigger": 2, "restored": 3}[kind]][0]:
                a = accs[0]
                ax.text(min(a["hi"] + 0.012, 0.8), i - 0.02, f"{a['acc'] * 100:.0f}%", fontsize=8.5, color=t["ink2"], va="center", ha="left")
        ax.set_yticks(range(nrow))
        if k == 0:
            ax.set_yticklabels([r[0] for r in rows], fontsize=9)
            ax.tick_params(axis="y", pad=8)
        else:
            ax.set_yticks([])
        ax.set_xlabel("accuracy on WMDP-bio (95% interval)", fontsize=8.5, labelpad=6)
        fig.text((left + k * (panel_w + gap)) / W, (H - 0.3) / H, f"{model}, visible reasoning (n = {rows[0][2][0]['n']})", fontsize=10.5, fontweight="bold", va="center")
    fig.text(left / W, 0.28 / H, "Vertical line = accuracy on the neutral prompt.\n"
             "The cap clamps a direction read from the model's own reasoning tokens over a band of layers.\n"
             "Random caps: three random directions capped over the same layers (hollow dots, no interval drawn; each is about ±4 points).",
             fontsize=7.5, color=t["muted"], va="center", linespacing=1.45)
    fig.savefig(path, dpi=DPI)
    plt.close(fig)


# --------------------------------------------------------------------------------------------------------------------- fig 4
SCREEN_SHORT = {  # Table 1 row (start of the model cell) -> (label, counted in the headline table)
    "gpt-oss-120b": ("gpt-oss-120B", True),
    "GLM-4.5-Air 107B MoE, direct": ("GLM-4.5-Air 107B, direct answers", True),
    "GLM-4.5-Air 107B MoE, visible": ("GLM-4.5-Air 107B, visible reasoning", False),
    "gpt-oss-20b": ("gpt-oss-20B", True),
    "Gemma-3-27B": ("Gemma-3-27B", True),
    "Gemma-3-12B": ("Gemma-3-12B", True),
    "Gemma-3-4B": ("Gemma-3-4B", False),
    "Qwen3-32B": ("Qwen3-32B", False),
    "Qwen3-14B-think": ("Qwen3-14B-think", False),
    "Qwen3-8B-think": ("Qwen3-8B-think", False),
    "OLMo-3.1-32B": ("OLMo-3.1-32B", False),
    "Llama-3.1-8B": ("Llama-3.1-8B", True),
    "Llama-3.3-70B bf16, direct": ("Llama-3.3-70B, direct answers", True),
    "Llama-3.3-70B bf16, visible": ("Llama-3.3-70B, visible reasoning", True),
    "GLM-4-9B": ("GLM-4-9B", True),
    "Mistral-Small": ("Mistral-Small-3.2-24B", False),
    "phi-4": ("phi-4", False),
    "DeepSeek-R1": ("DeepSeek-R1-distill-8B", False),
    "OLMo-3-7B": ("OLMo-3-7B-think", False),
}
SCREEN_COLS = ["Explicit\ninstruction", "Bare\ninstruction", "Hazard-\nselective", "Natural\ntrigger", "Sign-reversed\ntwin"]
BINS = [(-1e9, -40, -1.0), (-40, -20, -0.75), (-20, -10, -0.5), (-10, -5, -0.25), (-5, 5, 0.0), (5, 10, 0.25), (10, 1e9, 0.5)]


def _num(cell):
    """First number of a markdown table cell (U+2212 minus); None for an en dash or an empty cell."""
    import re
    c = cell.replace("*", "").replace("−", "-").strip()
    if c.startswith("–") or not c:
        return None
    m = re.search(r"[+-]?\d+(?:\.\d+)?", c)
    return float(m.group(0)) if m else None


def parse_screen(path="results/FINAL_REPORT.md"):
    """Table 1 of the report: accuracy change vs the neutral prompt (points) under each screening framing, and the natural trigger's refusal rate."""
    lines = open(path, encoding="utf-8").read().splitlines()
    i = next(k for k, l in enumerate(lines) if l.startswith("## Table 1"))
    rows = []
    for l in lines[i + 1:]:
        if rows and not l.startswith("|"):
            break
        if not l.startswith("|") or l.startswith("|---") or l.startswith("| model"):
            continue
        cells = [c.strip() for c in l.strip().strip("|").split("|")]
        if len(cells) != 9 or "superseded" in cells[0]:
            continue
        key = next((k for k in SCREEN_SHORT if cells[0].replace("**", "").startswith(k)), None)
        if key is None:
            raise ValueError(f"Table 1 row not in SCREEN_SHORT: {cells[0]}")
        label, counted = SCREEN_SHORT[key]
        vals = [_num(cells[3]), _num(cells[4]), _num(cells[5]), _num(cells[6]), _num(cells[8])]
        rows.append(dict(label=label, counted=counted, vals=vals, refusal=_num(cells[7]) or 0.0))
    assert len(rows) == len(SCREEN_SHORT), (len(rows), len(SCREEN_SHORT))
    return rows


def _blend(a, b, f):
    pa = [int(a[i:i + 2], 16) for i in (1, 3, 5)]
    pb = [int(b[i:i + 2], 16) for i in (1, 3, 5)]
    return "#" + "".join(f"{round(x + (y - x) * f):02x}" for x, y in zip(pa, pb))


def _lum(h):
    c = [int(h[i:i + 2], 16) / 255 for i in (1, 3, 5)]
    c = [x / 12.92 if x <= 0.03928 else ((x + 0.055) / 1.055) ** 2.4 for x in c]
    return 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2]


def rounded_rect(ax, x0, y0, x1, y1, color, r_in, W, H):
    pos = ax.get_position()
    xl, yl = ax.get_xlim(), ax.get_ylim()
    rx = min(r_in * abs(xl[1] - xl[0]) / (pos.width * W), (x1 - x0) / 2)
    ry = min(r_in * abs(yl[1] - yl[0]) / (pos.height * H), (y1 - y0) / 2)
    v = [(x0 + rx, y0), (x1 - rx, y0), (x1, y0), (x1, y0 + ry), (x1, y1 - ry), (x1, y1), (x1 - rx, y1), (x0 + rx, y1), (x0, y1), (x0, y1 - ry), (x0, y0 + ry), (x0, y0), (x0 + rx, y0)]
    c = [Path.MOVETO, Path.LINETO, Path.CURVE3, Path.CURVE3, Path.LINETO, Path.CURVE3, Path.CURVE3, Path.LINETO, Path.CURVE3, Path.CURVE3, Path.LINETO, Path.CURVE3, Path.CURVE3]
    ax.add_patch(PathPatch(Path(v, c), facecolor=color, edgecolor="none", linewidth=0, zorder=3))


def fig_screen(theme, rows, path):
    t = style(theme)
    neutral = "#f0efec" if theme == "light" else "#383835"

    def fill(v):
        if v is None:
            return None
        for lo, hi, f in BINS:
            if lo <= v < hi:
                return _blend(neutral, t["blue"] if f > 0 else t["red"], abs(f))
        return neutral

    groups = [("Steering run on the default prompt (headline table)", [r for r in rows if r["counted"]]),
              ("Screened only (no validated instrument run on the default prompt for this model or mode)", [r for r in rows if not r["counted"]])]
    pitch, cw = 0.275, 0.9
    ncols = len(SCREEN_COLS)
    W = 3.45 + ncols * cw + 0.35
    H = 0.95 + (len(rows) + 2 * 1.35) * pitch + 1.05
    fig = plt.figure(figsize=(W, H), dpi=DPI)
    ax = new_axes(fig, W, H, 3.45, 1.05, ncols * cw, (len(rows) + 2 * 1.35) * pitch)
    total = len(rows) + 2 * 1.35
    ax.set_xlim(0, ncols)
    ax.set_ylim(total, 0)
    ax.axis("off")
    y = 0.0
    gap = 2 * PX
    for gname, grp in groups:
        ax.text(-3.3 / cw, y + 0.55, gname, fontsize=8.5, fontweight="bold", color=t["ink2"], va="center", ha="left")
        y += 1.35
        for r in grp:
            ax.text(-0.1 / cw, y + 0.5, r["label"], fontsize=9, color=t["ink"], va="center", ha="right")
            for j, v in enumerate(r["vals"]):
                f = fill(v)
                if f is None:
                    ax.text(j + 0.5, y + 0.5, "–", fontsize=8.5, color=t["muted"], va="center", ha="center")
                    continue
                gx, gy = gap / (cw), gap / pitch
                rounded_rect(ax, j + gx / 2, y + gy / 2, j + 1 - gx / 2, y + 1 - gy / 2, f, 4 * PX, W, H)
                if abs(v) >= 10:
                    lab = f"{v:+.0f}".replace("-", "−") + (" R" if (j == 3 and r["refusal"] >= 0.3) else "")
                    ax.text(j + 0.5, y + 0.52, lab, fontsize=8, color="#ffffff" if _lum(f) < 0.3 else "#0b0b0b", va="center", ha="center", zorder=4)
            y += 1.0
    for j, name in enumerate(SCREEN_COLS):
        fig.text((3.45 + (j + 0.5) * cw) / W, (H - 0.55) / H, name, fontsize=8.5, color=t["ink2"], ha="center", va="center", linespacing=1.15)
    fig.text(3.45 / W, (H - 0.17) / H, "Accuracy change vs the neutral prompt, WMDP-bio (points)", fontsize=10.5, fontweight="bold", va="center")
    # stepped legend: seven swatches, boundaries printed under them
    leg = [("≤ −40", -1.0), ("−40…−20", -0.75), ("−20…−10", -0.5), ("−10…−5", -0.25), ("±5", 0.0), ("+5…+10", 0.25), ("≥ +10", 0.5)]
    sw, x0 = 0.62, 3.45 - 0.2
    for k, (lab, f) in enumerate(leg):
        col = _blend(neutral, t["blue"] if f > 0 else t["red"], abs(f))
        axl = new_axes(fig, W, H, x0 + k * (sw + 0.04), 0.66, sw, 0.2)
        axl.set_xlim(0, 1)
        axl.set_ylim(1, 0)
        axl.axis("off")
        rounded_rect(axl, 0, 0, 1, 1, col, 4 * PX, W, H)
        fig.text((x0 + k * (sw + 0.04) + sw / 2) / W, 0.5 / H, lab, fontsize=7.5, color=t["ink2"], ha="center", va="center")
    fig.text(0.25 / W, 0.15 / H, "Numbers are printed where the change is 10 points or more. R: the drop is refusals (65% of answers), not wrong answers. Gemma instruction drops come from the n = 120 screen;\n"
             "the other cells use n = 60–291 questions per model (Table 1 of the report).", fontsize=7.2, color=t["muted"], va="center", linespacing=1.45)
    fig.savefig(path, dpi=DPI)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="figures")
    ap.add_argument("--only", default="", help="comma list of fig1,fig2,fig3,fig4")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    only = set(filter(None, a.only.split(",")))
    group_res, pooled_res = dfc.compute()
    rows = flip_rows(group_res, pooled_res)
    pts = scatter_points(group_res, pooled_res)
    gem = gemma_rows()
    screen = parse_screen()
    for theme in THEMES:
        if not only or "fig1" in only:
            fig_flips(theme, rows, os.path.join(a.out, f"fig1-flips-{theme}.png"))
        if not only or "fig2" in only:
            fig_restore(theme, pts, os.path.join(a.out, f"fig2-restore-{theme}.png"))
        if not only or "fig3" in only:
            fig_gemma(theme, gem, os.path.join(a.out, f"fig3-gemma-{theme}.png"))
        if not only or "fig4" in only:
            fig_screen(theme, screen, os.path.join(a.out, f"fig4-screen-{theme}.png"))
    print("wrote", sorted(os.listdir(a.out)))


if __name__ == "__main__":
    main()
