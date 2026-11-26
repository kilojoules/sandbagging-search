"""Recipe sandbag_cot: directions read from the residual stream AT THE GENERATED (reasoning) TOKENS instead of at the last prompt token.

Why.  For a model whose sandbagging is carried out inside visible chain of thought (Gemma-3 under the natural goal + documentation trigger, or
under an explicit instruction in written-answer mode) a direction read at the last prompt token and removed from the prompt positions does not
restore accuracy: the plan is written and executed while the reasoning is generated, and a prompt-end summary direction is not what the
generated tokens consult.  Here the contrast is read where the plan lives: for every direction-pool question the model answers under the
neutral framing N and under the explicit/natural framing X, each (prompt + generated text) is teacher-forced once more and the residual is
averaged over the GENERATED tokens only (text cut after the first end-of-turn marker).

  CF  mean over FLIPPED questions (right under N, wrong under X) of  mean_gen(h_X) - mean_gen(h_N)          same question, same layer
  CS  mean_gen(h_X) of flipped questions minus that of STAY questions (right under both)                   flipped vs not, inside X
  CM  mean over all answered pairs of mean_gen(h_X) - mean_gen(h_N)

Directions point TOWARD sandbagging (X minus N); steer away = negative alpha, exactly like the other sandbag recipes.  They are hooked at every
position (prompt and generated) by sbx.core.Hooks, so at decoding they act on the very tokens they were read from.

Interface: fn(ctx, pool, cfg) -> (dirs, meta); see sbx/dirs/__init__.py.  Opt-in (`--recipes sandbag_cot`), not part of `--recipes all`.
"""
from __future__ import annotations

import hashlib
import json
import os

import numpy as np
import torch

from ..core import restore_forwards, snapshot_forwards
from .sandbag import (_Dirs, _finish_meta, _layers, _logger, _max_new, _take, explicit_framing, explicit_name, neutral_framing)

EOT_MARKERS = ("<end_of_turn>", "<|im_end|>", "<|eot_id|>", "<|end|>", "<|return|>", "<|endoftext|>", "</s>")
MIN_GROUP = 5          # a direction needs at least this many flipped (and stay) questions


def cut_at_end(text, extra=()):
    """Generated text up to and including its first end-of-turn / eos marker (batched generation pads shorter rows with pad tokens)."""
    cuts = [text.find(m) + len(m) for m in EOT_MARKERS + tuple(extra) if m and text.find(m) >= 0]
    return text[:min(cuts)] if cuts else text


def gen_state_means(ctx, prompts, texts, layers, cap=8):
    """float32 [n, len(layers), H]: mean residual over the GENERATED tokens of each (prompt + text), teacher-forced; rows whose text is empty are NaN.
    hidden_states[L] is the output of layer L (index 0 = embeddings), as everywhere else in sbx."""
    tok = ctx.tok
    extra = (getattr(tok, "eos_token", None) or "",)
    texts = [cut_at_end(t, extra) for t in texts]
    n = len(prompts)
    out = np.full((n, len(layers), ctx.H), np.nan, np.float32)
    n_prompt = [ctx.ntok(p) for p in prompts]
    full = [p + t for p, t in zip(prompts, texts)]
    n_full = [ctx.ntok(f) for f in full]
    n_gen = [max(nf - npm, 1) if t else 0 for nf, npm, t in zip(n_full, n_prompt, texts)]      # >= 1 token when there is text (a token may merge across the junction)
    rows = sorted(range(n), key=lambda i: n_full[i])

    def chunk(idx):
        enc = ctx.adapter.encode(tok, [full[i] for i in idx]).to(ctx.model.device)
        saved = snapshot_forwards(ctx.adapter.layers(ctx.model))
        try:
            with torch.no_grad():
                try:
                    o = ctx.model(**enc, output_hidden_states=True, logits_to_keep=1)
                except TypeError:
                    o = ctx.model(**enc, output_hidden_states=True)
                for b, i in enumerate(idx):
                    ng = n_gen[i]
                    if ng <= 0:
                        continue
                    for j, L in enumerate(layers):
                        out[i, j] = o.hidden_states[L][b, -ng:, :].float().mean(0).cpu().numpy()
            return
        except torch.cuda.OutOfMemoryError:
            restore_forwards(saved)                       # see core.snapshot_forwards: an OOM leaves transformers' output-capture wrappers installed
        del enc
        torch.cuda.empty_cache()
        if len(idx) == 1:
            raise RuntimeError("OOM on a single sequence in gen_state_means")
        h = len(idx) // 2
        ctx.log(f"[oom] splitting gen_state_means chunk of {len(idx)}")
        chunk(idx[:h])
        chunk(idx[h:])

    for s in range(0, n, cap):
        chunk(rows[s:s + cap])
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return out


def _key(ctx, qs, fN, fX, max_new, layers, prompts_hash):
    blob = json.dumps({"adapter": ctx.adapter.key, "q": [q["qid"] for q in qs], "N": fN.name, "X": fX.name, "max_new": max_new,
                       "layers": list(layers), "prompts": prompts_hash}, sort_keys=True)
    return hashlib.md5(blob.encode()).hexdigest()[:16]


def _collect(ctx, qs, fN, fX, max_new, layers, cache_dir, log):
    """Per framing: written-answer correctness / parse flags and the per-layer mean residual over the generated tokens (cached)."""
    pN, pX = ctx.prompts(qs, fN), ctx.prompts(qs, fX)
    ph = hashlib.md5("\x1f".join(pN + pX).encode()).hexdigest()[:12]
    path = os.path.join(cache_dir, f"sandbag_cot__{ctx.adapter.key}__{fN.name}__{fX.name}__n{len(qs)}__{_key(ctx, qs, fN, fX, max_new, layers, ph)}.npz") if cache_dir else None
    if path and os.path.exists(path):
        log(f"[sandbag_cot] cache hit {os.path.basename(path)}")
        z = np.load(path)
        return {k: np.array(z[k]) for k in ("corrN", "corrX", "parsedN", "parsedX", "SN", "SX")}
    got = {}
    for tag, fr, pr in (("N", fN, pN), ("X", fX, pX)):
        log(f"[sandbag_cot] framing {fr.name!r}: generating {len(qs)} answers (max_new={max_new})")
        r = ctx.run(qs, fr, max_new=max_new)
        log(f"[sandbag_cot] framing {fr.name!r}: teacher-forcing the generated text for its per-layer mean state")
        got["corr" + tag], got["parsed" + tag] = np.asarray(r.corr, np.float64), np.asarray(r.parsed, bool)
        got["S" + tag] = gen_state_means(ctx, pr, list(r.texts), layers)
    if path:
        os.makedirs(cache_dir, exist_ok=True)
        tmp = path + ".tmp.npz"
        np.savez(tmp, **got)
        os.replace(tmp, path)
    return got


def sandbag_cot(ctx, pool, cfg):
    log, layers = _logger(ctx, cfg), _layers(ctx, cfg)
    qs, max_new = _take(pool, "wmdp", cfg), _max_new(ctx, cfg)
    fN, fX = neutral_framing(), explicit_framing(cfg)
    g = _collect(ctx, qs, fN, fX, max_new, layers, cfg.get("cache_dir"), log)
    cN, cX, pX = g["corrN"], g["corrX"], g["parsedX"]
    SN, SX = g["SN"], g["SX"]
    ok = np.isfinite(SN).all(axis=(1, 2)) & np.isfinite(SX).all(axis=(1, 2))
    flipped = (cN == 1) & (cX == 0) & pX & ok
    stay = (cN == 1) & (cX == 1) & ok
    warns = []
    acc = _Dirs(("CF", "CS", "CM"))
    D = SX - SN
    for j, L in enumerate(layers):
        if ok.any():
            acc.add("CM", L, D[ok, j].mean(0))
        if flipped.sum() >= MIN_GROUP:
            acc.add("CF", L, D[flipped, j].mean(0))
            if stay.sum() >= MIN_GROUP:
                acc.add("CS", L, SX[flipped, j].mean(0) - SX[stay, j].mean(0))
    if flipped.sum() < MIN_GROUP:
        warns.append(f"only {int(flipped.sum())} flipped questions (< {MIN_GROUP}): CF and CS are omitted")
    elif stay.sum() < MIN_GROUP:
        warns.append(f"only {int(stay.sum())} stay questions (< {MIN_GROUP}): CS is omitted")
    dirs = acc.emitted()
    meta = {"recipe": "sandbag_cot", "adapter_key": ctx.adapter.key, "layers": layers, "max_new": max_new,
            "seed": int(cfg.get("seed") or 0), "n_pairs_requested": cfg.get("n_pairs"),
            "sign_convention": "directions point TOWARD sandbagging (generated-token states under X minus under N); steer away = negative alpha",
            "framings": {"neutral": fN.name, "explicit": explicit_name(cfg)}, "n_pairs": int(len(qs)),
            "neutral_acc": float(cN.mean()), "explicit_acc": float(cX.mean()), "n_flipped": int(flipped.sum()), "n_stay": int(stay.sum()),
            "n_usable": int(ok.sum()), "omitted": acc.omitted(), "raw_norm": {n: {str(L): v for L, v in d.items()} for n, d in acc.raw_norm.items()}}
    B = cfg.get("B")
    if B and dirs:
        meta["cos_B"] = {n: {str(L): float(np.dot(v, B[L]) / (np.linalg.norm(v) * np.linalg.norm(B[L]) + 1e-12)) for L, v in d.items() if L in B}
                         for n, d in dirs.items()}
    return dirs, _finish_meta(meta, warns, log)


RECIPES = {"sandbag_cot": sandbag_cot}
