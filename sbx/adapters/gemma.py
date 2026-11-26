"""Gemma-3 instruction-tuned adapters: gemma-3-27b-it, gemma-3-12b-it, gemma-3-4b-it.

Verified against transformers 4.57.6 (meta-device models built from the real hub configs, real tokenizer, tiny random
Gemma3ForConditionalGeneration for the numerics; see sbx/tests/test_adapters_gemma_llama.py and sbx/models/gemma-3-*.json):

* Architecture. All three checkpoints are multimodal (`architectures = ["Gemma3ForConditionalGeneration"]`: SigLIP vision tower
  + projector + Gemma3 text decoder). In transformers 4.57.6 `AutoModelForCausalLM` maps model_type "gemma3" to
  `Gemma3ForConditionalGeneration` (identical to `AutoModelForImageTextToText`), so the base `load()` already returns the right
  class; text-only use = feed `input_ids` only (the vision tower stays idle; ~0.42 B params / 0.85 GB in bf16). Every checkpoint
  key (read from the hubs' model.safetensors.index.json) maps one-to-one onto this class through `_checkpoint_conversion_mapping`;
  the only "missing" key is the tied `lm_head.weight`, so no weight is randomly initialised.
* Decoder layers: `model.model.language_model.layers` (a ModuleList of `Gemma3DecoderLayer`, `len == text_config.num_hidden_layers`:
  62 / 48 / 34). Older transformers (4.50-4.51, before the VLM refactor) nested them as `model.language_model.model.layers`, and the
  text-only `Gemma3ForCausalLM` uses `model.model.layers`; `layers()` resolves all three and raises if none exists (only the first
  layout can run here; the other two are covered by stub objects in the tests). Layer outputs are 1-tuples `(hidden,)` (core.Hooks
  handles tuples). Hidden size lives in `config.text_config.hidden_size` (base `hidden_size()` already looks there).
* 1-based convention: `hidden_states[L]` == output of `layers[L-1]` for 1 <= L < nl; `hidden_states[nl]` is the post-final-norm
  state (so avoid L = nl for directions). `hidden_states[0]` is the (sqrt(H)-scaled) embedding output.
* Chat template (tokenizer_config.json, identical for the three sizes): starts with `<bos>`; roles `user` / `model`; turn =
  `<start_of_turn>{role}\\n{content|trim}<end_of_turn>\\n`; generation prompt = `<start_of_turn>model\\n`. There is NO system role in
  Gemma's instruction format: the HF template folds a system message into the first user turn as `{system}\\n\\n{user}`. We use
  `sys_role=None`, i.e. the framing text is prepended to the user turn by `Adapter.chat_messages` (byte-identical to what the template
  would produce for a system message, but independent of that template behaviour).
* BOS: the template already emits `<bos>`; `Adapter.encode` uses `add_special_tokens=False`, so BOS appears exactly once.
* Thinking: none (Gemma-3 is not a reasoning model). `answer_mode="cot"`: brief visible reasoning ending in "Answer: X".
* Generation config shipped with the checkpoints: eos = [1 (<eos>), 106 (<end_of_turn>)], pad = 0. `core.Ctx._gen_chunk` passes
  `pad_token_id = tok.pad_token_id or tok.eos_token_id`, which is 1 (`<eos>`) for Gemma because pad id 0 is falsy; harmless, but it means
  finished sequences are padded with `<eos>` in the decoded text, so `split()` cuts at the first `<end_of_turn>` / `<eos>`.
* dtype: bf16 only. fp16 overflows in Gemma-3 (NaN/inf activations), so `load()` refuses fp16.
* 4-bit: `quant="4bit"` works through the base `load()` (bitsandbytes quantises the vision tower too, harmlessly) but is not needed:
  27B in bf16 is 54.9 GB and fits one 80 GB GPU. A pre-quantised repo (e.g. unsloth/gemma-3-27b-it-bnb-4bit: its config.json carries
  a bitsandbytes `quantization_config`, checked on the hub) is loaded with quant=None; transformers then uses the checkpoint's own
  config. The GPU-memory guard follows what will really be loaded: it reads that config.json, and only when bf16 would not fit, and
  applies the 4-bit figure instead of the bf16 one (the load itself is not run here: no bitsandbytes / CUDA locally).
* Memory caveat (arithmetic from the real configs and tokenizers, not run on a GPU): the 27B fits one 80 GB GPU for generation, but
  with the current core.Ctx.hidden_last (full-sequence hidden states + all-position logits + KV cache) the single longest
  direction-pool prompt (labbench_cloning, ~22.8k tokens) needs >= 82 GB; see "hidden_last_long_context" in sbx/models/gemma-3-27b-it.json.
"""
from __future__ import annotations

import re
import warnings

from .base import Adapter, classify_lead

L10 = "ABCDEFGHIJ"

BOS = "<bos>"
GEN_PROMPT = "<start_of_turn>model\n"
_STOPS = ("<end_of_turn>", "<eos>", "<start_of_turn>")   # the model's turn ends at the first of these
_DROP = ("<pad>", BOS)                                    # stripped wherever they occur


# ----------------------------------------------------------------------------------------------- output parsing
# A statement of the final answer: "Answer: B", "**Answer:** B", "Answer: **B**", "the correct answer is (B).",
# "Final answer is: $\boxed{B}$". The keyword is case-insensitive but the LETTER must be upper-case A-J, so
# "the answer is a bit tricky" can never parse as option A; a connector (is / : / = / -) is required, so "Answer A is
# wrong because ..." is not a statement; the letter must not start a word ("Answer: Cannot determine"), and an "A" / "I"
# followed by a lower-case word on the same line ("Answer: A virus ...", "Answer: I think ...") is English, not an option.
_KEY = r"(?:(?i:answer)|(?i:correct\s+(?:option|choice|letter)))"
_CONNECT = r"(?:\s*(?:is|would\s+be|should\s+be|:|=|-|\u2013|\u2014))+"
_WRAP_TOK = r"(?:\*\*|__|\$|\\boxed\{|\\text\{|\\textbf\{)"
# Between the connector and the letter: spaces, then markdown / LaTeX wrappers, then at most one opening bracket:
#     \s* (?:WRAP+ \s*)? (?:[(\[] \s*)?
# Every element starts with a different kind of character, so each position can be matched in exactly one way. The obvious
# spelling `\s* WRAP* \s* [(\[]? \s*` accepts the same strings but has three ADJACENT whitespace-capable groups, which Python's
# `re` (no timeout, no possessive quantifiers before 3.11) backtracks over in O(n^3) when a long whitespace run follows
# "answer:" and no letter does ("Answer:" + 800 newlines + "unclear" took 8.6 s; degenerate generations under steering or noise
# produce such runs): one such reply froze the whole single-threaded run. Pinned by the tests (timing + old-vs-new equivalence).
_PRE = r"\s*(?:" + _WRAP_TOK + r"+\s*)?(?:[\(\[]\s*)?"
_OPT = r"(?:(?i:option|choice|letter)\s+)?"
# "A"/"I" followed by a lower-case word is English ("Answer: A virus ...", "Answer: I think ..."), unless the word is a copula
# ("A is correct", "A was ..."); for the pronoun I the copula "was" is not one ("Answer: I was unable to answer"). A letter followed by
# an apostrophe is a contraction / possessive, never a label ("Answer: I\u2019m sorry, but I can\u2019t help with that.", "Answer: A's ...").
# The pronoun I is a label only before an end of text / line, a closing bracket, quote, asterisk or colon, a "." that does not open an
# ellipsis or an abbreviation, or is/because/since ("Answer: I, as an AI ...", "Answer: I... hmm", "Answer: I (an AI) ..." are refusals).
_I_OK = r"(?:[ \t]*(?:$|\n)|[):*\]}`$\"\u201d\\]|\.(?![.A-Za-z0-9])|[ \t]+(?:is|because|since)\b)"
_TAIL = (r"(?![A-Za-z0-9'\u2019])"
         r"(?!(?<=A)[ \t]+(?!(?:is|was|because|since)\b)[a-z0-9])"
         r"(?:(?<!I)|(?=" + _I_OK + r"))")
_ANS = re.compile(_KEY + _CONNECT + _PRE + _OPT + r"([A-J])" + _TAIL)
_BOXED = re.compile(r"\\boxed\{\s*(?:\\text(?:bf)?\{\s*)?[\(\[]?([A-J])" + _TAIL)
# Fallback for a model that ignores the CoT instruction and replies with just a letter: the (short) reply must START with the
# letter followed by punctuation / end of line ("B", "B.", "(B)", "**B**", "B) fever"). Long outputs never use it, so a
# truncated analysis that starts "A. This option is wrong because ..." is not read as an answer.
# A "." that opens an ellipsis or an abbreviation is not a label ("I... I can't say", "I.e. the answer", "B.C.").
_BARE = re.compile(r"^[\s*_(\[]*([A-J])(?:[ \t]*(?:\.(?![.A-Za-z0-9])|[):\]*_])|[ \t]*(?:\n|$))")
_BARE_MAX_CHARS = 300


def clean_output(text: str) -> str:
    """Raw decoded generation (special tokens kept, finished rows padded) -> the model's visible reply."""
    cut = min((i for i in (text.find(s) for s in _STOPS) if i >= 0), default=len(text))
    text = text[:cut]
    for s in _DROP:
        text = text.replace(s, "")
    return text.strip()


def extract_letter(text: str):
    """Upper-case option letter the reply commits to, or None. The LAST explicit "Answer: X" statement wins; without one,
    a short reply that is only a letter is accepted; anything else (truncated reasoning, refusal, prose) is None."""
    text = clean_output(text)
    hits = [(m.start(), m.group(1)) for m in _ANS.finditer(text)]
    hits += [(m.start(), m.group(1)) for m in _BOXED.finditer(text)]
    if hits:
        return max(hits)[1]
    if len(text) <= _BARE_MAX_CHARS:
        m = _BARE.match(text)
        if m and classify_lead(text, m.end(1), m.group(1)) == "bare":      # not the first item of an option list ("A: Ebola ... B: Influenza ...")
            return m.group(1)
    return None


def decoder_layers(model):
    """ModuleList of Gemma-3 decoder layers for every HF layout (see module docstring); raises if there is none."""
    inner = getattr(model, "model", None)
    lm = getattr(inner, "language_model", None)                 # transformers >= 4.52: ...ForConditionalGeneration.model.language_model
    if lm is not None and hasattr(lm, "layers"):
        return lm.layers
    lm = getattr(model, "language_model", None)                 # transformers 4.50-4.51: .language_model is a Gemma3ForCausalLM
    if lm is not None and hasattr(getattr(lm, "model", None), "layers"):
        return lm.model.layers
    if hasattr(inner, "layers"):                                # text-only Gemma3ForCausalLM
        return inner.layers
    raise AttributeError(f"cannot locate Gemma-3 decoder layers on {type(model).__name__}: tried model.model.language_model.layers, "
                         f"model.language_model.model.layers and model.model.layers")


def gpu_bytes_total():
    """Total memory over all visible CUDA devices, or None without CUDA."""
    import torch
    if not torch.cuda.is_available():
        return None
    return float(sum(torch.cuda.get_device_properties(i).total_memory for i in range(torch.cuda.device_count())))


def require_bitsandbytes():
    """quant='4bit' needs bitsandbytes (even constructing BitsAndBytesConfig looks it up): fail with a clear message."""
    import importlib.metadata
    try:
        importlib.metadata.version("bitsandbytes")
    except importlib.metadata.PackageNotFoundError:
        raise ImportError("quant='4bit' needs bitsandbytes and a CUDA GPU: pip install -U bitsandbytes accelerate") from None


_HUB_KW = ("revision", "token", "cache_dir", "local_files_only")      # from_pretrained kwargs that decide WHICH config.json is read


def checkpoint_precision(repo, kw):
    """Precision of the weights `from_pretrained(repo, **kw)` puts on the GPU when the caller does not ask for quant='4bit':
    "bf16" (nothing is quantised), "4bit" (a pre-quantised 4-bit repo, e.g. a `-bnb-4bit` upload) or None (quantised some
    other way: `weights_gb` has no figure for it). Mirrors transformers 4.57 (quantizers/auto.py get_hf_quantizer, `pre_quantized`):
    a `quantization_config` stored in the checkpoint's config.json is used and wins over one passed by the caller. Reads only
    config.json (no weights); a repo whose config cannot be read raises the hub error here, the same one the loader would raise
    a moment later."""
    from transformers import AutoConfig
    stored = getattr(AutoConfig.from_pretrained(repo, **{k: kw[k] for k in _HUB_KW if k in kw}), "quantization_config", None)
    qc = stored if stored is not None else kw.get("quantization_config")
    if qc is None:
        return "bf16"
    get = qc.get if isinstance(qc, dict) else (lambda name, default=None: getattr(qc, name, default))
    return "4bit" if (get("load_in_4bit") or get("bits") == 4) else None


class Gemma3(Adapter):
    keys = ["gemma-3-27b-it", "gemma-3-12b-it", "gemma-3-4b-it"]
    repos = {"gemma-3-27b-it": "google/gemma-3-27b-it", "gemma-3-12b-it": "google/gemma-3-12b-it",
             "gemma-3-4b-it": "google/gemma-3-4b-it"}
    answer_mode = "cot"
    max_new = 512          # estimate (weights were not available locally to measure); watch parse_rate, raise for 10-option sets
    sys_role = None        # no system role: framing text is prepended to the user turn
    effort = None          # no reasoning-effort knob
    # decimal GB of weights (bf16: sum of safetensors on the hub; 4bit: NF4 + double-quant estimate, see sbx/models/*.json)
    weights_gb = {"gemma-3-27b-it": {"bf16": 54.9, "4bit": 16.3}, "gemma-3-12b-it": {"bf16": 24.4, "4bit": 7.8},
                  "gemma-3-4b-it": {"bf16": 8.6, "4bit": 3.2}}

    # ---- loading -------------------------------------------------------------------------------------------
    def load(self, repo=None, dtype="auto", device_map="auto", quant=None, **kw):
        """(tok, model) via the base loader (AutoModelForCausalLM -> Gemma3ForConditionalGeneration), with explicit guards:
        no fp16, quant in {None, "4bit"}, enough GPU memory for the weights as they will be loaded (instead of silent CPU
        offload; see `_check_gpu_memory`), and a layer-count check after loading."""
        if str(dtype).replace("torch.", "") in ("float16", "fp16", "half"):
            raise ValueError("Gemma-3 must run in bf16 (fp16 overflows); use dtype='auto' or 'bf16'")
        if quant not in (None, "4bit"):
            raise ValueError(f"quant must be None or '4bit', got {quant!r}")
        repo = repo or self.repo
        self._check_gpu_memory(repo, quant, device_map, kw)
        if quant == "4bit":
            require_bitsandbytes()
        tok, model = super().load(repo, dtype, device_map, quant, **kw)
        self.check_model(model)
        return tok, model

    def _check_gpu_memory(self, repo, quant, device_map, kw):
        """MemoryError if the weights, as `from_pretrained` will load them, cannot fit the visible GPUs (the alternative is
        a silent offload to CPU). Skipped without CUDA, with an explicit device_map / max_memory, and for unknown keys.
        quant='4bit' -> the 4-bit figure. Otherwise the checkpoint decides: bf16 weights need the bf16 figure, a pre-quantised
        4-bit repo (config.json carries a quantization_config) the 4-bit figure; config.json is only read when bf16 would NOT
        fit, so the common case makes no extra request."""
        total = gpu_bytes_total()
        if total is None or device_map != "auto" or "max_memory" in kw or self.key not in self.weights_gb:
            return
        gb = self.weights_gb[self.key]
        if quant == "4bit":
            precision = "4bit"
        elif total >= gb["bf16"] * 1e9 * 1.05:
            return
        else:
            precision = checkpoint_precision(repo, kw)
            if precision is None:
                warnings.warn(f"{self.key}: {repo} ships a quantization_config that is not 4-bit and `weights_gb` has no figure "
                              f"for it; the GPU memory check is skipped ({total / 1e9:.0f} GB visible)", stacklevel=3)
                return
        need = gb[precision] * 1e9 * 1.05
        if total < need:
            raise MemoryError(f"{self.key} needs ~{need / 1e9:.0f} GB of GPU memory for the weights ({precision}) but only "
                              f"{total / 1e9:.0f} GB is visible; pass quant='4bit' (or a pre-quantised 4-bit repo) or use a "
                              f"larger / multi-GPU machine")

    def check_model(self, model):
        n, want = self.num_layers(model), getattr(model.config, "text_config", model.config).num_hidden_layers
        if n != want:
            raise RuntimeError(f"{type(model).__name__}: found {n} decoder layers, config says {want}")

    # ---- structure -----------------------------------------------------------------------------------------
    def layers(self, model):
        return decoder_layers(model)

    # ---- prompts -------------------------------------------------------------------------------------------
    def render(self, tok, msgs, effort=None):
        """messages -> '<bos><start_of_turn>user\\n...<end_of_turn>\\n<start_of_turn>model\\n' (`effort` is ignored: no knob)."""
        s = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        if not s.startswith(BOS) or s.startswith(BOS + BOS):
            raise ValueError(f"rendered prompt must start with exactly one {BOS!r}: {s[:40]!r}")
        if not s.endswith(GEN_PROMPT):
            raise ValueError(f"rendered prompt must end with the generation prompt {GEN_PROMPT!r}: {s[-40:]!r}")
        return s

    # ---- outputs -------------------------------------------------------------------------------------------
    def split(self, text):
        """CoT is visible and part of the reply: reasoning = "" and final = reply with end-of-turn / padding tokens removed
        (so `Ctx.score` scans the whole reply for instruction/consequence mentions)."""
        return "", clean_output(text)

    def parse_letter(self, final, nchoices=4):
        letter = extract_letter(final)
        if letter is None:
            return None
        i = L10.index(letter)
        return i if i < nchoices else None
