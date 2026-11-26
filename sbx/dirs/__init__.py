"""Direction-recipe registry (auto-discovery).

A recipe module defines  RECIPES = {"recipe_name": fn}  with
    fn(ctx, pool, cfg) -> (dirs, meta)
      ctx  : sbx.core.Ctx (adapter + tokenizer + model)
      pool : dict of question lists for direction construction (never used for evaluation):
             {"wmdp": [...], "hard": [...], "nonbio": [...]}   (see sbx/evals.py: get_pool)
      cfg  : dict(n_pairs=int, max_new=int, seed=int, cache_dir=str, layers=list[int]|None, log=callable);
             optional "B" = {layer: unit vec} (the primary sandbag direction) adds meta["cos_B"] to the non-sandbag recipes
      dirs : {direction_name: {layer(1-based): unit float32 np.ndarray [H]}}
             layers: cfg["layers"] if given (each in 1..ctx.nl, else ValueError), otherwise 1..ctx.nl-1.  Layer ctx.nl is
             post-final-norm in HF's hidden_states, so it is never a default (SPEC section 3: avoid L=nl); every recipe obeys this.
      meta : JSON-serialisable dict (counts, cosines, held-out probe AUROC per layer, anything useful)
Add a module here and it is picked up automatically (no edits to this file).

Meta schema (what a generic consumer has to special-case; per-recipe details are in docs/dirs_*.md).  Every meta is JSON-lossless
(json.loads(json.dumps(meta)) == meta): dict keys are STRINGS ("15", not 15; `dirs` itself keeps int layers), non-finite floats are
null, and meta["layers"] lists the layers the per-layer dicts cover.  All nine recipes write recipe, adapter_key, layers, seed,
n_pairs_requested, sign_convention (str; {direction: str} for honesty / honesty_mcq / persona) and warnings (list of str).  Beyond
that there are two key families:
  sandbag_bio / sandbag_hard / sandbag_generic / sandbag_variants  (sandbag.py, docs/dirs_sandbag.md)
      cos_to_B {direction: {layer: cos}} and cos_to_B_subsets {direction: {"all"|"early"|"mid"|"late": mean cos}} (empty if no WMDP
      question flipped); held-out probes probe_auroc (+ probe_auroc_A, probe_auroc_M in sandbag_bio; transfer_auroc_wmdp in
      sandbag_generic) as {layer: AUROC or null}; behaviour as FLAT top-level keys (n_pairs, neutral_acc, explicit_acc, acc_drop,
      parse_rate_neutral, parse_rate_explicit, mention_rate_* [null when the answers carry no reasoning], n_flipped, n_stay,
      n_reverse, n_both_wrong, n_flipped_unparsed, n_flipped_mention, n_flipped_silent, sign_test_p, instrument_valid);
      direction_instrument_valid {direction: bool}, omitted, neutral_norm {layer: x}, alpha_natural {direction: {layer: x}} and a
      singular `warning` (warnings joined by " | ") only when there is something to warn about.  Exceptions: sandbag_variants has
      no flat explicit-side statistics (explicit_acc, acc_drop, n_flipped, n_stay, ...; they sit per variant in `variants` and
      *_by_variant); sandbag_hard has no probe_auroc.  No `counts`, `auroc` or `cos_B` here.
  refusal / evalaware / honesty / honesty_mcq / persona  (refusal.py, evalaware.py, honesty.py, persona.py; docs/dirs_refusal_evalaware.md,
  docs/dirs_honesty_persona.md)
      cos_B (null unless cfg["B"] is given), auroc, diff_norm and counts (flat {name: int}); no cos_to_B, probe_auroc or flat n_flipped.
      The SHAPE differs by recipe: refusal has flat per-layer dicts, cos_B / auroc / diff_norm = {layer: x} (plus auroc_vs_jbb_benign,
      auroc_chembio_vs_harmless, auroc_chembio_vs_lifesci_benign, each {layer: AUROC} or null); the other four have one more level,
      cos_B = {direction: {layer: cos}} and diff_norm = {direction: {layer: x}}, and auroc = {direction: {layer: AUROC}} for evalaware
      but {direction: {held-out set: {layer: AUROC}}} for honesty / honesty_mcq / persona (sets newq / newq_newpara; val / val_newwrap /
      ood for TRUTH).
"""
import importlib
import pkgutil

_RECIPES = {}
_LOADED = False


def get_recipes():
    global _LOADED
    if not _LOADED:
        for m in pkgutil.iter_modules(__path__):
            mod = importlib.import_module(f"{__name__}.{m.name}")
            for k, fn in getattr(mod, "RECIPES", {}).items():
                if k in _RECIPES:
                    raise KeyError(f"duplicate recipe {k!r} (module {m.name})")
                _RECIPES[k] = fn
        _LOADED = True
    return dict(_RECIPES)
