"""Llama-3.x instruction-tuned adapters: llama-3.1-8b-instruct and llama-3.3-70b-instruct.

Verified against transformers 4.57.6 (meta-device models built from the hub configs, the real Llama-3.1 tokenizer, a tiny random
LlamaForCausalLM for the numerics; see sbx/tests/test_adapters_gemma_llama.py and sbx/models/llama-3.*.json):

* Architecture: plain `LlamaForCausalLM`; decoder layers = `model.model.layers` (`LlamaDecoderLayer`, 32 / 80 layers), hidden size =
  `config.hidden_size` (4096 / 8192). `hidden_states[L]` == output of `layers[L-1]` for 1 <= L < nl; `hidden_states[nl]` is post final
  norm. Layer outputs are plain tensors in transformers 4.57 (`core.Hooks` handles tensors and tuples).
* Chat template (identical string in 3.1-8B and 3.3-70B; same vocab/merges): starts with `<|begin_of_text|>`, ALWAYS emits a system
  header `<|start_header_id|>system<|end_header_id|>\\n\\nCutting Knowledge Date: December 2023\\nToday Date: {date_string}\\n\\n{system}<|eot_id|>`
  even without a system message, then `<|start_header_id|>user<|end_header_id|>\\n\\n{user}<|eot_id|>`; generation prompt =
  `<|start_header_id|>assistant<|end_header_id|>\\n\\n`. The date is a template variable: the 3.1 and official 3.3 templates default to
  the fixed string "26 Jul 2024" (verified), but a template variant that defaults it to `strftime_now(...)` (today's date; I believe
  Llama-3.2-style templates do, not checked here) would make prompts, and therefore results, change from day to day. `render()`
  therefore ALWAYS passes `date_string=DATE_STRING` explicitly, and the tests build exactly such a `strftime_now` variant to prove it.
  Framing text goes in the system role (`sys_role="system"`), after the date lines.
* BOS: the template emits `<|begin_of_text|>`; `Adapter.encode` uses `add_special_tokens=False`, so BOS appears exactly once.
* Padding: Llama-3.1 has no pad token (the base loader would reuse `<|eot_id|>`); `load()` switches to the dedicated
  `<|finetune_right_pad_id|>` (what the official 3.3 tokenizer already uses) so that padding is never confused with end-of-turn.
  Generation stops at eos ids [128001, 128008, 128009] from the checkpoint's generation_config.
* Thinking: none. `answer_mode="cot"`: brief visible reasoning ending in "Answer: X" (Llama also likes "The final answer is:
  $\\boxed{X}$", which `parse_letter` understands).
* 4-bit (needed for the 70B on one 80 GB GPU; bf16 weights are 141 GB): `load(quant="4bit")` goes through the base loader with an
  NF4 + double-quantisation + bf16-compute `BitsAndBytesConfig` (the base default would be FP4; NF4 has lower weight error). Memory:
  ~39.5 GB of weights (68.4 B quantised linear params at ~0.516 B/param + 4.2 GB bf16 embeddings/lm_head), leaving ~35 GB of an
  80 GB card for KV cache and activations (KV is 0.31 MB/token for the 70B). bitsandbytes only swaps the `nn.Linear` children of
  each decoder layer for `Linear4bit`; the `LlamaDecoderLayer` modules, their bf16 outputs and therefore `core.Hooks` forward hooks
  behave exactly as in bf16 (the steering vector is cast to the hidden-state dtype/device inside `f_add`). Not runnable here (no
  CUDA / bitsandbytes locally): this is established from the library code, not from a run.
* Gating: both repos are gated (manual approval). With the token available when this module was written, 3.1-8B was accessible and
  `meta-llama/Llama-3.3-70B-Instruct` was NOT (HTTP 403 GatedRepoError). `unsloth/Llama-3.3-70B-Instruct` is an ungated mirror whose
  30 safetensors shards have byte-identical sha256 to the official repo (`load(repo="unsloth/Llama-3.3-70B-Instruct", quant="4bit")`),
  and `unsloth/Llama-3.3-70B-Instruct-bnb-4bit` (39.5 GB, pre-quantised NF4) avoids the 141 GB download (quantisation details of that
  checkpoint are the uploader's). Load it with `load(repo="unsloth/Llama-3.3-70B-Instruct-bnb-4bit")`, i.e. quant=None: its config.json
  carries a bitsandbytes `quantization_config` (NF4, double-quant, bf16 compute; checked on the hub), transformers then uses that
  config, and the GPU-memory guard follows what will really be loaded: it reads that config.json (only when bf16 would not fit) and
  applies the 4-bit figure (39.5 GB) instead of the 141 GB of the official bf16 repo. `quant="4bit"` with the same repo works too:
  transformers 4.57.6 (quantizers/auto.py: get_hf_quantizer / merge_quantization_configs; library code read, the load itself is
  not run here: no bitsandbytes / CUDA locally) uses the checkpoint's own quantization_config and only warns that the one passed in
  is ignored (a quantization_config of a different class would raise ValueError).
* Memory caveat (arithmetic from the real configs and tokenizers, not run on a GPU): in 4-bit the 70B fits one 80 GB GPU for
  generation, but with the current core.Ctx.hidden_last (full-sequence hidden states + all-position logits + KV cache) the single
  longest direction-pool prompt (labbench_cloning, ~26.5k tokens) needs >= 81 GB; see "hidden_last_long_context" in
  sbx/models/llama-3.3-70b-instruct.json.
"""
from __future__ import annotations

import re
import warnings

from .base import Adapter, classify_lead

L10 = "ABCDEFGHIJ"

BOS = "<|begin_of_text|>"
GEN_PROMPT = "<|start_header_id|>assistant<|end_header_id|>\n\n"
PAD = "<|finetune_right_pad_id|>"
DATE_STRING = "26 Jul 2024"        # the default of the official 3.1 / 3.3 templates; passed explicitly (see module docstring)
_STOPS = ("<|eot_id|>", "<|end_of_text|>", "<|eom_id|>", "<|start_header_id|>")   # the model's turn ends at the first of these
_DROP = (PAD, BOS, "<|python_tag|>")


# ----------------------------------------------------------------------------------------------- output parsing
# Same rules as sbx/adapters/gemma.py (duplicated on purpose so each adapter module stands alone; the test-suite runs one
# battery against both). Statement of the final answer: "Answer: B", "**Answer:** B", "Answer: **B**", "the correct answer is
# (B).", "The final answer is: $\boxed{B}$". Keyword case-insensitive, LETTER upper-case A-J, connector required, letter must
# not start a word, and "A" / "I" followed by a lower-case word ("Answer: A virus ...", "Answer: I think ...") is English.
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
# Fallback for a (short) reply that is only a letter: "B", "B.", "(B)", "**B**", "B) fever".
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


class Llama3(Adapter):
    keys = ["llama-3.1-8b-instruct", "llama-3.3-70b-instruct"]
    repos = {"llama-3.1-8b-instruct": "meta-llama/Llama-3.1-8B-Instruct",
             "llama-3.3-70b-instruct": "meta-llama/Llama-3.3-70B-Instruct"}
    answer_mode = "cot"
    max_new = 512          # estimate (weights were not available locally to measure); watch parse_rate, raise for 10-option sets
    sys_role = "system"    # native system role (placed under the template's date header)
    effort = None          # no reasoning-effort knob
    # decimal GB of weights (bf16: sum of safetensors on the hub; 4bit: NF4 + double-quant estimate, see sbx/models/*.json)
    weights_gb = {"llama-3.1-8b-instruct": {"bf16": 16.1, "4bit": 5.7}, "llama-3.3-70b-instruct": {"bf16": 141.1, "4bit": 39.5}}

    # ---- loading -------------------------------------------------------------------------------------------
    def load(self, repo=None, dtype="auto", device_map="auto", quant=None, **kw):
        """(tok, model) via the base loader. quant="4bit" -> NF4 + double-quant + bf16 compute (override with
        quantization_config=... in **kw). Explicit guards: quant in {None, "4bit"}; enough GPU memory for the weights as they will
        be loaded (a clear MemoryError instead of silent CPU offload, e.g. the 70B in bf16 on one 80 GB card; a pre-quantised 4-bit
        repo is measured at its 4-bit size, see `_check_gpu_memory`); layer-count check after loading.
        The pad token is switched to the dedicated <|finetune_right_pad_id|>."""
        if quant not in (None, "4bit"):
            raise ValueError(f"quant must be None or '4bit', got {quant!r}")
        repo = repo or self.repo
        self._check_gpu_memory(repo, quant, device_map, kw)
        if quant == "4bit":
            require_bitsandbytes()
        if quant == "4bit" and "quantization_config" not in kw:
            import torch
            from transformers import BitsAndBytesConfig
            kw["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                                          bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16)
        tok, model = super().load(repo, dtype, device_map, quant, **kw)
        self.prepare_tokenizer(tok)
        self.check_model(model)
        return tok, model

    @staticmethod
    def prepare_tokenizer(tok):
        """Left padding (last position = last prompt token) and the dedicated pad token (never the end-of-turn token)."""
        tok.padding_side = "left"
        if PAD in tok.get_vocab():
            tok.pad_token = PAD
        elif tok.pad_token is None:
            tok.pad_token = tok.eos_token
        return tok

    def _check_gpu_memory(self, repo, quant, device_map, kw):
        """MemoryError if the weights, as `from_pretrained` will load them, cannot fit the visible GPUs (the alternative is
        a silent offload to CPU). Skipped without CUDA, with an explicit device_map / max_memory, and for unknown keys.
        quant='4bit' -> the 4-bit figure. Otherwise the checkpoint decides: bf16 weights need the bf16 figure, a pre-quantised
        4-bit repo (config.json carries a quantization_config; e.g. unsloth/Llama-3.3-70B-Instruct-bnb-4bit, 39.5 GB) the 4-bit
        figure; config.json is only read when bf16 would NOT fit, so the common case makes no extra request."""
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
        n, want = self.num_layers(model), model.config.num_hidden_layers
        if n != want:
            raise RuntimeError(f"{type(model).__name__}: found {n} decoder layers, config says {want}")

    # ---- prompts -------------------------------------------------------------------------------------------
    def render(self, tok, msgs, effort=None):
        """messages -> '<|begin_of_text|><|start_header_id|>system<|end_header_id|>\\n\\nCutting Knowledge Date: ...Today Date:
        26 Jul 2024\\n\\n{system}<|eot_id|><|start_header_id|>user<|end_header_id|>\\n\\n{user}<|eot_id|>
        <|start_header_id|>assistant<|end_header_id|>\\n\\n' (`effort` is ignored: no knob; the date is always DATE_STRING)."""
        s = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, date_string=DATE_STRING)
        if not s.startswith(BOS) or s.startswith(BOS + BOS):
            raise ValueError(f"rendered prompt must start with exactly one {BOS!r}: {s[:40]!r}")
        if f"Today Date: {DATE_STRING}\n" not in s:
            raise ValueError("chat template did not use the explicit date_string (not a Llama-3.x template?)")
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
