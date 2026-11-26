"""GLM family (Zhipu AI / Z.ai) adapters.

Keys (one registry key per behaviour; the mode is implied by the key, see `_SPEC`):

    glm-4-9b-chat        THUDM/glm-4-9b-chat      ChatGLM remote code (trust_remote_code), non-thinking, answer_mode "cot"
    glm-4-9b-chat-hf     THUDM/glm-4-9b-chat-hf   same tensors, transformers-native GlmForCausalLM (no remote code; the only 9B key on >=5), BUT its
                                                  Hub config has the wrong RoPE base: load() corrects it (10000 -> 5,000,000, see below)
    glm-4.5-air          zai-org/GLM-4.5-Air      Glm4MoeForCausalLM, thinking OFF ("cot")
    glm-4.5-air-think    zai-org/GLM-4.5-Air      thinking ON ("reasoning": reasoning is split off at the closing think tag)
    glm-4.5-air-direct   zai-org/GLM-4.5-Air      thinking OFF, letter-only answers (cheap: ~3 generated tokens per question)
    glm-4.7-flash        zai-org/GLM-4.7-Flash    Glm4MoeLiteForCausalLM (needs transformers>=5.0.0), thinking OFF ("cot")
    glm-4.7-flash-think  zai-org/GLM-4.7-Flash    thinking ON ("reasoning")
    glm-4.7-flash-direct zai-org/GLM-4.7-Flash    thinking OFF, letter-only answers

Prompt formats (all verified against the real tokenizers' chat templates, see sbx/tests/test_adapters_glm.py):

    glm-4-9b-chat(-hf)  [gMASK]<sop><|system|>\\n{sys}<|user|>\\n{user}<|assistant|>
                        (the assistant's reply starts with "\\n"; no thinking mode)
    GLM-4.5-Air         [gMASK]<sop><|system|>\\n{sys}<|user|>\\n{user}[/nothink]<|assistant|>[\\n<think></think> | \\n<think>]
                        thinking OFF: the template appends the soft switch "/nothink" to the user turn and ends the prompt with
                        "<|assistant|>\\n<think></think>".  thinking ON: the template alone ends with "<|assistant|>" (the model
                        decides whether to think); THIS ADAPTER PREFILLS "\\n<think>" so that thinking is always on (the
                        reasoning channel then always exists and the think/no-think decision cannot confound interventions).
    GLM-4.7-Flash       [gMASK]<sop><|system|>{sys}<|user|>{user}<|assistant|>[</think> | <think>]
                        (no newlines after role tags; thinking OFF = "</think>" prefilled by the template with
                        enable_thinking=False, thinking ON = "<think>" prefilled by the template)

The model's generated text therefore looks like "{reasoning}</think>{answer}" for the thinking keys; `split()` cuts at the FIRST
closing tag (no closing tag => truncated reasoning => reasoning=text, final=""), and `parse_letter()` never guesses from reasoning.

Traps handled here (each one was reproduced locally; details in sbx/models/*.json "known_issues"):
  * `tok(text, add_special_tokens=True)` PREPENDS ANOTHER "[gMASK]<sop>" to a rendered prompt for both glm-4-9b-chat tokenizers (remote
    and fast; the 4.5/4.7 tokenizers do not) -> encode() uses add_special_tokens=False and render() asserts exactly one prefix.
  * transformers 4.x tokenizers for GLM-4.5/4.7 return `token_type_ids`, which makes `generate()` raise -> encode() drops them.
  * the ChatGLM remote code breaks `generate()` on transformers 4.57.6 (generate() pre-allocates a DynamicCache; probably since 4.56) ->
    a legacy-cache shim is applied to the loaded model class; it is unusable on transformers>=5 -> explicit error pointing to
    `glm-4-9b-chat-hf`.
  * the ChatGLM remote code's hidden_states[nl] is the last block's output BEFORE the final norm (native HF models: after it); the
    1-based convention (layers()[L-1] output == hidden_states[L]) holds for every 1<=L<nl in all GLM models (L==nl too for ChatGLM).
  * checkpoints of GLM-4.5-Air / GLM-4.7-Flash contain one extra next-token-prediction (MTP) layer that transformers ignores; layers()
    therefore has 46 / 47 entries and is validated against the config, never guessed.
  * THUDM/glm-4-9b-chat-hf ships `rope_theta=10000.0`, but the original model's RoPE base is 10000 * rope_ratio = 5,000,000 (ChatGLM remote
    code: `base = 10000 * config.rope_ratio`, rope_ratio=500 in THUDM/glm-4-9b-chat). The upstream conversion script computes
    `10000.0 * getattr(original_config, "rope_ratio", 1)` on a dict from json.load (getattr never finds a key in a dict -> always 1), so
    EVERY converted Hub config says 10000 (also glm-4-9b-chat-1m-hf, whose original has rope_ratio 10000). Loaded unchanged, the -hf key would
    measure a perturbed model (at 16 tokens distance 11 of the 32 rotary frequencies are off by more than 0.5 rad). The script does no q/k
    permutation (it renames keys and splits the fused qkv), so with the right base the native model equals the remote-code model exactly
    (tiny random model, max|dlogit| = 0; tests). `Glm49ChatHF.load()` therefore loads the config first, sets the base to `rope_base`, passes
    it as `config=` and verifies the rotary `inv_freq` of the loaded model. Note: transformers>=5 silently ignores `rope_theta` as a
    from_pretrained kwarg AND as a plain config attribute; only `config.rope_parameters["rope_theta"]` counts there (set_rope_theta).
  * plain forwards default to use_cache=True, so Ctx.hidden_last / Ctx.neutral_stats / check_convention would allocate a full KV cache for a
    pure prefill (GLM-4.7-Flash on transformers 5.0-5.14: 962,560 B/token, i.e. 13.5 GB for a batch filled to core's 14k-token budget and
    25 GB for one 26k-token prompt, the maximum of the raw LAB-Bench CloningScenarios file).
    `_post_load` sets config.use_cache=False (and pins generation_config.use_cache=True): generate() is unaffected (same texts, still caches).
  * core.kv_bytes_per_token is wrong for these configs (ChatGLM remote 16x too high; GLM-4.7-Flash 4x too low on transformers 5.0-5.14
    where HF caches the expanded per-head K/V, 4.4x too high from 5.15 where it caches the compressed MLA latents);
    `Adapter.kv_bytes_per_token(model)` below probes the real cache, but nothing calls it until core.Ctx does (see the report's
    needs_from_core). Until then batches() mis-sizes batches, and ONLY Ctx._gen_chunk has an OOM split/retry: hidden_last and
    neutral_stats have none, and batches() always tries a prompt alone. use_cache=False cannot help generate(): on GLM-4.7-Flash with
    transformers 5.0-5.14 a prompt of about 20k tokens or more is expected to fail there on an 80 GB GPU (962,560 B/token against ~20 GB
    left beside the 60 GB of weights: RuntimeError 'OOM on a single prompt'; the per-token size is measured, the free memory is an
    estimate, no GPU was available). sbx/evals.py drops questions above MAX_PROMPT_CHARS (7,500 chars at the time of writing) from every
    split and pool, so the registry currently serves at most ~2.8k tokens (measured with this tokenizer: pool 2,175, disc/rep 2,796) and the
    26k-token prompts are out (without a cap the sandbag_hard recipe ran them through ctx.run and ctx.hidden_last). If that cap is raised,
    use transformers>=5.15 (compressed cache, 54,144 B/token) or a larger GPU.
"""
from __future__ import annotations

import logging
import os
import re

from .base import L10, Adapter, classify_lead

log = logging.getLogger("sbx.adapters.glm")

PREFIX = "[gMASK]<sop>"
ASSISTANT = "<|assistant|>"
OPEN, CLOSE = "<think>", "</think>"
NOTHINK = "/nothink"
THINK_PREFILL_AIR = "\n" + OPEN        # GLM-4.5 assistant turns are "<|assistant|>\n<think>...</think>\n{answer}"

# Verified commits of the Hub repos (tokenizer/chat template, config and, for ChatGLM, the remote code were read at these revisions).
REVISIONS = {
    "THUDM/glm-4-9b-chat": "bd8234fe5e0c09c48637a92abb0c797cb5fa0e73",
    "THUDM/glm-4-9b-chat-hf": "8599336fc6c125203efb2360bfaf4c80eef1d1bf",
    "zai-org/GLM-4.5-Air": "a24ceef6ce4f3536971efe9b778bdaa1bab18daa",
    "zai-org/GLM-4.7-Flash": "7dd20894a642a0aa287e9827cb1a1f7f91386b67",
}

# key -> behaviour. mode: "cot" (visible CoT ending 'Answer: X'), "reasoning" (think block + letter), "direct" (letter only).
# max_new are defaults (Ctx.run(max_new=...) overrides): cot 640 / think 2048 / direct 8 generated tokens.
_SPEC = {
    "glm-4-9b-chat": dict(fam="chatglm", repo="THUDM/glm-4-9b-chat", thinking=False, mode="cot", max_new=640),
    "glm-4-9b-chat-hf": dict(fam="glm4hf", repo="THUDM/glm-4-9b-chat-hf", thinking=False, mode="cot", max_new=640),
    "glm-4.5-air": dict(fam="air", repo="zai-org/GLM-4.5-Air", thinking=False, mode="cot", max_new=640),
    "glm-4.5-air-think": dict(fam="air", repo="zai-org/GLM-4.5-Air", thinking=True, mode="reasoning", max_new=2048),
    "glm-4.5-air-direct": dict(fam="air", repo="zai-org/GLM-4.5-Air", thinking=False, mode="direct", max_new=8),
    "glm-4.7-flash": dict(fam="flash", repo="zai-org/GLM-4.7-Flash", thinking=False, mode="cot", max_new=640),
    "glm-4.7-flash-think": dict(fam="flash", repo="zai-org/GLM-4.7-Flash", thinking=True, mode="reasoning", max_new=2048),
    "glm-4.7-flash-direct": dict(fam="flash", repo="zai-org/GLM-4.7-Flash", thinking=False, mode="direct", max_new=8),
}


def _keys(fam):
    return [k for k, s in _SPEC.items() if s["fam"] == fam]


# --------------------------------------------------------------------------------------------------- RoPE base (glm-4-9b-chat-hf)
ROPE_RATIO_9B_CHAT = 500                           # "rope_ratio" in THUDM/glm-4-9b-chat/config.json
ROPE_BASE_9B_CHAT = 10000.0 * ROPE_RATIO_9B_CHAT   # ChatGLM remote RotaryEmbedding: base = 10000 * rope_ratio -> 5,000,000
HUB_ROPE_THETA_9B_HF = 10000.0                     # what THUDM/glm-4-9b-chat-hf/config.json says (upstream conversion bug)
ROPE_CHECK_RTOL = 2e-2                             # inv_freq of base 1e4 vs 5e6 differs by >= 21% from entry 1 on; bf16 rounding is 0.4%
ROPE_CHECK_MIN_FREQ = 1e-3                         # only entries >= this are compared (a half-precision buffer cannot raise a false alarm)


def get_rope_theta(cfg):
    """RoPE base of a native HF config on any transformers version: `rope_parameters["rope_theta"]` (>=5) or `rope_theta` (4.x)."""
    rp = getattr(cfg, "rope_parameters", None)
    if rp:
        if "rope_theta" not in rp:
            raise RuntimeError(f"{type(cfg).__name__}.rope_parameters has no 'rope_theta': {rp!r}")
        return float(rp["rope_theta"])
    theta = getattr(cfg, "rope_theta", None)
    if theta is None:
        raise RuntimeError(f"{type(cfg).__name__} has neither rope_parameters['rope_theta'] nor rope_theta")
    return float(theta)


def set_rope_theta(cfg, theta):
    """Set the RoPE base of a native HF config on any transformers version (in place) and read it back.

    transformers>=5 ignores `rope_theta` as a from_pretrained/AutoConfig kwarg and as a plain config attribute: the rotary module reads
    `config.rope_parameters["rope_theta"]` only, so that is what is written when it exists. transformers 4.x reads `config.rope_theta`.
    (Whether the model really uses the value is checked on the loaded model by `rotary_inv_freq_error`.)"""
    rp = getattr(cfg, "rope_parameters", None)
    if rp:
        rp["rope_theta"] = float(theta)
    else:
        cfg.rope_theta = float(theta)
    if get_rope_theta(cfg) != float(theta):
        raise RuntimeError(f"could not set the RoPE base of {type(cfg).__name__} to {theta:,.0f} (reads back {get_rope_theta(cfg):,.0f})")


def rotary_inv_freq_error(model, base):
    """Largest relative deviation between the loaded native model's rotary `inv_freq` (model.model.rotary_emb) and the one of RoPE `base`.

    The buffer decides what the model computes whatever the config says. Returns None when the buffer lives on the meta device (not
    verifiable). Only entries >= ROPE_CHECK_MIN_FREQ are compared; raises if the layout is not the expected one."""
    import torch
    emb = getattr(getattr(model, "model", None), "rotary_emb", None)
    inv = getattr(emb, "inv_freq", None)
    if inv is None:
        raise RuntimeError(f"cannot find model.model.rotary_emb.inv_freq on {type(model).__name__} to verify the RoPE base")
    if inv.device.type == "meta":
        return None
    dim = 2 * inv.numel()
    want = 1.0 / (float(base) ** (torch.arange(0, dim, 2, dtype=torch.float64) / dim))
    got = inv.detach().float().cpu().double()       # float32 first: the buffer can sit on a device without float64 (mps)
    keep = want >= ROPE_CHECK_MIN_FREQ
    if int(keep.sum()) < 2:
        raise RuntimeError(f"rotary inv_freq has {inv.numel()} entries, too few >= {ROPE_CHECK_MIN_FREQ:g} to verify base {base:,.0f}")
    return float(((got[keep] - want[keep]).abs() / want[keep]).max())


# --------------------------------------------------------------------------------------------------- output parsing
# Special tokens that can surround a decoded answer (decode uses skip_special_tokens=False; the ChatGLM remote tokenizer even
# inserts spaces around them: "B <|endoftext|> <|endoftext|>"). The answer-box tags are removed but their content is kept.
_ROLE_TOKENS = re.compile(r"<\|(?:endoftext|user|assistant|system|observation)\|>|<\|(?:begin|end)_of_box\|>")
_SEP = r"[\s:：=\-–—*_是为]+"
# A..J as a standalone token. "A" and "I" are also English words: reject them when followed by a space and a lowercase word
# ("Answer: A virus ...", "Answer: I think ...") but keep "A", "A.", "A, because", "I" alone.
# The pronoun I counts only before an end of text / line, a closing bracket, quote, asterisk or colon, or a "." that does not open an ellipsis
# or an abbreviation ("Answer: I, as an AI ..." / "Answer: I... hmm" / "Answer: I (an AI) ..." are refusals, not option I).
_I_OK = r"(?:[ \t]*(?:$|\n)|[):*\]}`$\"\u201d\\]|\.(?![.A-Za-z0-9]))"
_LETTER = r"(?:[B-HJ]|A(?!\s+[a-z])|I(?!\s+[a-z])(?=" + _I_OK + r"))\b(?!['\u2019])"      # not "I'm" / "I\u2019m" / "A's": a letter glued to an apostrophe is a contraction
_EXPLICIT = re.compile(
    # "Answer: B", "**Answer:** B", "answer is (B)", "Final Answer: Option B", "答案：B"
    r"(?:(?i:\banswer)\b|答案)" + _SEP + r"(?:(?i:is|would be|should be)\b" + _SEP + r")?"
    r"(?:(?i:option|choice|letter)\s+)?[\(\[]?(?:\*\*)?(" + _LETTER + r")"
    # "the correct option is B"
    r"|(?i:\b(?:correct|right|best)\s+(?:option|choice|letter))\b" + _SEP + r"(?:(?i:is)\b" + _SEP + r")?[\(\[]?(?:\*\*)?(" + _LETTER + r")"
    # "\boxed{B}", "\boxed{\text{B}}"   (no overlapping unbounded quantifiers: linear time; see test_parse_is_total_on_garbage_and_linear_in_time)
    r"|\\boxed\{[\s(\[]*(?:\\(?:text|textbf|mathbf|mathrm)\{[\s(\[]*)?(" + _LETTER + r")"
)
# A reply that BEGINS with the letter: "B", "(B)", "**B**", "B.", "B) text", "Option B", "B\n\nexplanation" (never "A virus ...", "I'm ...",
# "I, as an AI ...", "I... hmm", "I.e.", "C-terminal ..."). After the letter: end / newline, a closing bracket, quote or asterisk, ":", a "."
# that does not start an ellipsis or an abbreviation, a "-" that does not join a word, and "," ";" "!" for every letter but the pronoun I.
_LEAD_END = r"[ \t]*(?:$|\n)|[)\]:*_`\"]|\.(?![.A-Za-z0-9])|-(?![A-Za-z0-9])"
_LEAD = re.compile(
    r"^[\s>*_`\"(\[]*(?:(?i:option|choice|letter)\s+)?("
    r"(?:[B-HJ]|A(?!\s+[a-z]))(?=" + _LEAD_END + r"|[,;!])"
    r"|I(?!\s+[a-z])(?=" + _LEAD_END + r"))"
)
# A reply that is nothing but the letter (the only fallback accepted in "cot" mode, where the reply starts with free text).
_PURE = re.compile(r"^[\s*_`(\[]*([A-J])[\s*_`)\].:]*$")


def clean_final(text):
    """Remove role/EOS/box special tokens from a decoded answer and strip whitespace."""
    return _ROLE_TOKENS.sub("", text).strip()


def _strip_open(text):
    """Drop one leading '<think>' (the Air thinking turn can start with '\\n<think>' when it was not prefilled)."""
    t = text.lstrip()
    return t[len(OPEN):] if t.startswith(OPEN) else text


# ------------------------------------------------------------------------------------------------------- adapter base
class _GlmBase(Adapter):
    """Shared behaviour of the GLM keys. Not registered (keys = []). Subclasses set: keys, repos, family constants."""
    keys = []
    sys_role = "system"
    weights_gib_hint = 0.0              # GiB of weights the placement guard (_check_placement) quotes in its error
    effort = None                       # GLM has no effort knob: Framing.effort is accepted and ignored (documented, by design)

    family = None
    remote_code = False
    model_type = None                   # config model_type the installed transformers must know (None: remote code)
    layer_path = "model.layers"         # attribute path from the loaded model to the decoder ModuleList
    min_transformers = "4.46.0"
    max_transformers_exclusive = None   # str or None
    chat_think_kwarg = False            # True: pass enable_thinking=<bool> to the chat template (GLM-4.5 / 4.7)
    fused_experts_from = None           # transformers version from which MoE experts are fused 3D params (bitsandbytes cannot quantise)
    pin_revision = True

    # ---------------------------------------------------------------------------------------------- construction
    def __init__(self, key=None):
        key = (key or self.keys[0]).lower()
        if key not in self.keys:
            raise ValueError(f"{type(self).__name__} does not serve key {key!r}; its keys: {self.keys}")
        super().__init__(key)
        spec = _SPEC[key]
        self.thinking = spec["thinking"]
        self.answer_mode = spec["mode"]
        self.max_new = spec["max_new"]

    # ---------------------------------------------------------------------------------------------- prompt rendering
    def _prefill(self):
        return ""

    def _ending(self):
        """Exact tail of every rendered prompt for this key (where the assistant's generation starts)."""
        raise NotImplementedError

    def _template_kwargs(self):
        return {"enable_thinking": bool(self.thinking)} if self.chat_think_kwarg else {}

    def render(self, tok, msgs, effort=None):
        """messages -> full prompt string ending exactly where the assistant starts.

        `effort` is ignored (GLM has no effort knob). The thinking mode is fixed by the key (see module docstring)."""
        s = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, **self._template_kwargs())
        if not isinstance(s, str):
            raise TypeError(f"{self.key}: apply_chat_template(tokenize=False) returned {type(s).__name__}, expected str")
        s += self._prefill()
        self._check_prompt(s)
        return s

    def _check_prompt(self, s):
        """Raise (never silently continue) if the chat template changed in a way that breaks the assumptions of this adapter."""
        end = self._ending()
        if not s.startswith(PREFIX):
            raise ValueError(f"{self.key}: rendered prompt must start with {PREFIX!r}, got {s[:30]!r}")
        if s.count("[gMASK]") != 1 or s.count("<sop>") != 1:
            raise ValueError(f"{self.key}: duplicated BOS prefix in rendered prompt: {s[:60]!r}")
        if not s.endswith(end):
            raise ValueError(f"{self.key}: rendered prompt must end with {end!r}, got ...{s[-60:]!r} (chat template changed?)")

    def encode(self, tok, texts):
        """Tokenise rendered prompts: no special tokens added (the template has them) and no token_type_ids (generate() rejects them)."""
        return tok(texts, return_tensors="pt", padding=True, add_special_tokens=False, return_token_type_ids=False)

    # ---------------------------------------------------------------------------------------------- outputs
    def split(self, text):
        """generated text -> (reasoning, final).

        thinking keys:   '{reasoning}</think>{answer}' -> (reasoning, answer); no closing tag (truncated) -> (text, "").
        non-thinking:    no reasoning channel -> ("", cleaned text), unless the model re-opened a think block it never closed.
        A leading '<think>' (Air without prefill) and role/EOS special tokens are removed; both parts are stripped."""
        if CLOSE in text:
            r, f = text.split(CLOSE, 1)
            return _strip_open(r).strip(), clean_final(f)
        if self.thinking:
            return clean_final(_strip_open(text)), ""
        if OPEN in text:
            return clean_final(text.split(OPEN, 1)[1]), ""
        return "", clean_final(text)

    def parse_letter(self, final, nchoices=4):
        """final answer text -> 0-based option index or None (None => unparsed, counted wrong by core).

        cot:             last explicit 'Answer: X' / 'the answer is X' / '\\boxed{X}' / '答案：X'; else the reply must be nothing but the
                         letter (a reply that merely STARTS with 'A. ...' is a truncated option list, not an answer).
        direct/reasoning: a reply that starts with the letter ('B', '(B)', '**B**', 'B. text'), else the last explicit answer phrase.
        Out-of-range letters (E for a 4-option question) are None; 'A virus ...' / 'I think ...' / 'BRCA1' are not letters."""
        t = clean_final(final)
        if not t:
            return None
        letter = None
        if self.answer_mode == "cot":
            ms = list(_EXPLICIT.finditer(t))
            if ms:
                letter = next(g for g in ms[-1].groups() if g)
            else:
                m = _PURE.match(t)
                letter = m.group(1) if m else None
        else:
            m = _LEAD.match(t)
            if m and classify_lead(t, m.end(1), m.group(1)) in ("bare", "copular"):
                letter = m.group(1)         # not "A, B and C ..." (multi) nor "A: Ebola ... B: Influenza ..." (a list of options)
            else:
                ms = list(_EXPLICIT.finditer(t))
                if ms:
                    letter = next(g for g in ms[-1].groups() if g)
        if letter is None:
            return None
        i = L10.index(letter)
        return i if i < nchoices else None

    # ---------------------------------------------------------------------------------------------- structure
    def _config_layers(self, cfg):
        vals = {getattr(cfg, a) for a in ("num_hidden_layers", "num_layers") if getattr(cfg, a, None) is not None}
        if len(vals) != 1:
            raise RuntimeError(f"{self.key}: config has ambiguous/missing layer count {vals}")
        return vals.pop()

    def layers(self, model):
        """The decoder ModuleList at `self.layer_path`, validated against the config (never falls back to another path)."""
        obj = model
        for part in self.layer_path.split("."):
            if not hasattr(obj, part):
                raise RuntimeError(
                    f"{self.key}: expected decoder layers at `model.{self.layer_path}` but {type(obj).__name__} has no attribute "
                    f"{part!r} (model class {type(model).__name__}); refusing to guess a different path")
            obj = getattr(obj, part)
        want = self._config_layers(getattr(model.config, "text_config", model.config))
        if len(obj) != want:
            raise RuntimeError(f"{self.key}: `model.{self.layer_path}` has {len(obj)} layers but the config says {want}")
        return obj

    def kv_bytes_per_token(self, model, bytes_per_elem=None, probe_tokens=16):
        """KV-cache bytes per token (all layers) that HF generate() really allocates for this model.

        core.kv_bytes_per_token(model) is wrong for these configs (ChatGLM remote: 16x too high; GLM-4.7-Flash: 4x too low on
        transformers 5.0-5.14 where HF caches the expanded per-head K/V, 4.4x too high from 5.15 on where it caches the compressed MLA
        latents), so a materialised model is PROBED: one tiny forward with use_cache=True and the cache tensors are counted (exact for
        every transformers version/layout). A meta-device model cannot be probed: then the structural formula is used
        (bytes_per_elem, default 2 = bf16; MLA = expanded layout of transformers<5.15)."""
        if not self._is_meta(model):
            return self._probe_kv_bytes(model, probe_tokens)
        return self._formula_kv_bytes(model, 2 if bytes_per_elem is None else bytes_per_elem)

    @staticmethod
    def _is_meta(model):
        return next(model.parameters()).device.type == "meta"

    @staticmethod
    def _cache_tensors(pkv):
        """All key/value tensors of a returned cache (Cache object with .layers, old key_cache/value_cache lists, or legacy tuples)."""
        if pkv is None:
            raise RuntimeError("model returned no past_key_values with use_cache=True")
        if hasattr(pkv, "layers"):
            ts = [t for layer in pkv.layers for t in (layer.keys, layer.values)]
        elif hasattr(pkv, "key_cache"):
            ts = list(pkv.key_cache) + list(pkv.value_cache)
        else:
            ts = [t for kv in pkv for t in kv]
        return [t for t in ts if t is not None]

    def _probe_kv_bytes(self, model, n):
        import torch
        cfg = getattr(model.config, "text_config", model.config)
        vocab = getattr(cfg, "vocab_size", None) or cfg.padded_vocab_size
        ids = ((torch.arange(n) * 7 + 1000) % min(vocab, 100_000)).unsqueeze(0).to(model.device)
        with torch.no_grad():
            out = model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=True)
        ts = self._cache_tensors(out.past_key_values)
        nbytes = sum(t.numel() * t.element_size() for t in ts)
        if not ts or nbytes % n:
            raise RuntimeError(f"{self.key}: cache probe found {len(ts)} tensors / {nbytes} bytes for {n} tokens (not a multiple)")
        return nbytes // n

    def _formula_kv_bytes(self, model, bytes_per_elem):
        cfg = getattr(model.config, "text_config", model.config)
        nl = self._config_layers(cfg)
        if hasattr(cfg, "kv_lora_rank"):            # MLA, expanded layout: per-head K (nope+rope) and V
            per_layer = cfg.num_key_value_heads * (cfg.qk_nope_head_dim + cfg.qk_rope_head_dim + cfg.v_head_dim)
        elif hasattr(cfg, "multi_query_group_num"):  # ChatGLM remote config: grouped K/V
            per_layer = 2 * cfg.multi_query_group_num * cfg.kv_channels
        else:
            per_layer = 2 * cfg.num_key_value_heads * (getattr(cfg, "head_dim", None) or cfg.hidden_size // cfg.num_attention_heads)
        return nl * per_layer * bytes_per_elem

    # ---------------------------------------------------------------------------------------------- loading
    @staticmethod
    def _release(v):
        """(major, minor, micro) of a version string/Version: pre-release tags are ignored (5.0.0rc3 counts as 5.0.0)."""
        from packaging.version import Version
        return (v if isinstance(v, Version) else Version(v)).release

    def _check_runtime(self):
        """Raise a clear RuntimeError if the installed transformers cannot serve this key; returns its release tuple."""
        import transformers
        found = transformers.__version__
        rel = self._release(found)
        if rel < self._release(self.min_transformers):
            what = f" (config model_type {self.model_type!r})" if self.model_type else ""
            raise RuntimeError(f"{self.key}{what} needs transformers>={self.min_transformers}, found {found}")
        if self.max_transformers_exclusive and rel >= self._release(self.max_transformers_exclusive):
            raise RuntimeError(self._too_new_message(found))
        if self.model_type:
            from transformers.models.auto.configuration_auto import CONFIG_MAPPING_NAMES
            if self.model_type not in CONFIG_MAPPING_NAMES:
                raise RuntimeError(f"{self.key}: transformers {found} does not know model_type {self.model_type!r} "
                                   f"(needs >={self.min_transformers})")
        return rel

    def _too_new_message(self, found):
        return f"{self.key} needs transformers<{self.max_transformers_exclusive}, found {found}"

    @staticmethod
    def _dtype_kwarg(dtype, rel):
        import torch
        if isinstance(dtype, str) and dtype not in ("auto",):
            names = {"bf16": "bfloat16", "bfloat16": "bfloat16", "fp16": "float16", "float16": "float16", "fp32": "float32", "float32": "float32"}
            if dtype not in names:
                raise ValueError(f"unknown dtype {dtype!r}; use 'auto', 'bf16', 'fp16', 'fp32' or a torch.dtype")
            dtype = getattr(torch, names[dtype])
        # `dtype` replaces `torch_dtype` from transformers 4.56 on (the old name only emits a deprecation warning there)
        return {("dtype" if rel >= (4, 56, 0) else "torch_dtype"): dtype}

    def _quant_kwargs(self, quant, rel):
        if quant is None:
            return {}
        import importlib.util
        if quant not in ("4bit", "8bit"):
            raise ValueError(f"unknown quant {quant!r}; use None, '4bit' or '8bit'")
        if self.fused_experts_from and rel >= self._release(self.fused_experts_from):
            raise ValueError(
                f"{self.key}: transformers {'.'.join(map(str, rel))} stores the MoE experts as fused 3-D parameters, which "
                f"bitsandbytes does not quantise (only nn.Linear); quant={quant!r} would not shrink the model. Use bf16 on "
                f"enough GPUs, or transformers<{self.fused_experts_from} (experts are nn.Linear there), or a pre-quantised repo.")
        if importlib.util.find_spec("bitsandbytes") is None:
            raise RuntimeError(f"{self.key}: quant={quant!r} needs the `bitsandbytes` package (pip install bitsandbytes)")
        import torch
        from transformers import BitsAndBytesConfig
        if quant == "4bit":
            cfg = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
                                     bnb_4bit_compute_dtype=torch.bfloat16)
        else:
            cfg = BitsAndBytesConfig(load_in_8bit=True)
        return {"quantization_config": cfg}

    def load(self, repo=None, dtype="auto", device_map="auto", quant=None, revision=None, check=True, **kw):
        """Return (tok, model). left-padded tokenizer, eval mode, greedy generation config, convention verified (check=True).

        Defaults to the verified Hub commit of `self.repo` (REVISIONS) unless `revision`/a local path is given.
        Family hooks: `_config_kwargs` (before loading: e.g. the corrected config of glm-4-9b-chat-hf, skipped when the caller passes
        `config=`), `_post_load` (greedy generation config, no KV cache in plain forwards) and `_verify_model` (raises if the loaded model is
        not the function the key promises; runs even with check=False)."""
        from transformers import AutoModelForCausalLM, AutoTokenizer
        rel = self._check_runtime()
        repo = repo or self.repo
        extra = {}
        if revision is None and self.pin_revision and not os.path.isdir(repo):
            revision = REVISIONS.get(repo)
        if revision:
            extra["revision"] = revision
        tok = AutoTokenizer.from_pretrained(repo, trust_remote_code=self.remote_code, **extra)
        tok.padding_side = "left"
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        kwargs = dict(trust_remote_code=self.remote_code, **extra)
        if device_map is not None:
            kwargs["device_map"] = device_map
        kwargs.update(self._dtype_kwarg(dtype, rel))
        kwargs.update(self._quant_kwargs(quant, rel))
        if "config" not in kw:                   # a caller-supplied config is used as is (and still verified by _verify_model)
            kwargs.update(self._config_kwargs(repo, extra))
        kwargs.update(kw)
        model = AutoModelForCausalLM.from_pretrained(repo, **kwargs).eval()
        self._post_load(model)
        self._verify_model(model)
        if check:
            self.check_convention(tok, model)
        return tok, model

    def _config_kwargs(self, repo, extra):
        """Extra from_pretrained kwargs derived from the repo's config, e.g. {"config": corrected_config} (default: none).
        `extra` holds the pinned `revision`, if any."""
        return {}

    def _verify_model(self, model):
        """Raise if the loaded model does not compute what the key promises (default: nothing to verify; runs even with check=False)."""

    def _check_placement(self, model):
        """A model too big for the visible GPUs is silently spilled to CPU RAM / disk by device_map="auto" (one log line), after which every forward
        streams weights from the host, runs ~100x slower and weight noise hits meta parameters.  Print the weight GiB per device and raise when
        anything is not on a GPU.  No-op without CUDA (CPU unit tests)."""
        import torch
        if not torch.cuda.is_available():
            return
        dm = getattr(model, "hf_device_map", None) or {}
        off = {k: v for k, v in dm.items() if str(v) in ("cpu", "disk", "meta")}
        per = {}
        for prm in model.parameters():
            per[str(prm.device)] = per.get(str(prm.device), 0) + prm.numel() * prm.element_size()
        print("[load] %s: weights GiB per device %s" % (self.key, {d: round(b / 2 ** 30, 1) for d, b in sorted(per.items())}), flush=True)
        if off or {"cpu", "meta"} & set(per):
            raise RuntimeError("%s: weights are not fully on the GPUs (offloaded %s, e.g. %s; parameter devices %s): the model needs about %.0f GiB free "
                               "across the visible GPUs (other processes still holding GPU memory? fewer GPUs than requested?)"
                               % (self.key, sorted({str(v) for v in off.values()}), list(off)[:3], sorted(per), self.weights_gib_hint))

    def _post_load(self, model):
        """Greedy-only generation config (core passes do_sample=False anyway; this removes the sampling-flag warnings) and no KV cache in
        plain forwards.

        Ctx.hidden_last / Ctx.neutral_stats / check_convention call model(**enc, output_hidden_states=True) with the default use_cache, which
        would allocate a full KV cache for a pure prefill (no OOM fallback there; GLM-4.7-Flash on transformers 5.0-5.14: 962,560 B/token).
        config.use_cache=False removes it; generate() passes use_cache=True itself and generation_config.use_cache is pinned True, so
        generation (texts, cache) is unchanged (tests, transformers 4.57.6 / 5.0.0 / 5.24.0, all four families)."""
        gc = getattr(model, "generation_config", None)
        if gc is not None:
            gc.do_sample = False
            for a in ("temperature", "top_p", "top_k"):
                if getattr(gc, a, None) is not None:
                    setattr(gc, a, None)
            gc.use_cache = True
        for cfg in (model.config, getattr(model.config, "text_config", None)):
            if cfg is not None:
                cfg.use_cache = False

    def check_convention(self, tok, model, probe_layers=None):
        """Executable form of the SPEC convention: for L in `probe_layers` (default 1, nl//2, nl-1) the output of layers()[L-1]
        must equal hidden_states[L] of a real forward pass. Raises RuntimeError on any mismatch (wrong path, wrong off-by-one)."""
        import torch
        layers = self.layers(model)
        nl = len(layers)
        probe = set(probe_layers or {1, max(1, nl // 2), max(1, nl - 1)})
        if self.remote_code and probe_layers is None:
            probe.add(nl)          # ChatGLM remote code: hidden_states[nl] is the last block's output (pre final norm)
        probe = sorted(probe)
        if any(not 1 <= L <= nl for L in probe):
            raise ValueError(f"probe layers {probe} outside 1..{nl}")
        cap, hooks = {}, []
        for L in probe:
            hooks.append(layers[L - 1].register_forward_hook(
                lambda m, i, o, L=L: cap.__setitem__(L, (o[0] if isinstance(o, tuple) else o).detach())))
        try:
            enc = self.encode(tok, [self.build_chat(tok, "ping")]).to(model.device)
            with torch.no_grad():
                hs = model(**enc, output_hidden_states=True).hidden_states
        finally:
            for h in hooks:
                h.remove()
        if hs is None or len(hs) != nl + 1:
            raise RuntimeError(f"{self.key}: expected {nl + 1} hidden states (embeddings + {nl} layers), got {None if hs is None else len(hs)}")
        for L in probe:
            a, b = hs[L].float(), cap[L].float().to(hs[L].device)
            rel = ((a - b).abs().max() / (b.abs().max() + 1e-6)).item()
            if a.shape != b.shape or rel > 1e-3:
                raise RuntimeError(f"{self.key}: hidden_states[{L}] != output of layers()[{L - 1}] (shape {tuple(a.shape)} vs "
                                   f"{tuple(b.shape)}, rel.err {rel:.2e}); the 1-based layer convention does not hold for this model")
        return True


# ------------------------------------------------------------------------------------------------ GLM-4-9B (dense)
class _Glm49Base(_GlmBase):
    """glm-4-9b-chat family: no thinking mode; template ends with '<|assistant|>' (the reply starts with '\\n')."""

    def _ending(self):
        return ASSISTANT


class Glm49Chat(_Glm49Base):
    """THUDM/glm-4-9b-chat with the repo's ChatGLM remote code (trust_remote_code).

    Layers: model.transformer.encoder.layers (40 GLMBlocks; blocks take/return [batch, seq, hidden] and return (hidden, kv)).
    hidden_states = (embeddings, out_1 .. out_40) where out_40 is BEFORE the final RMSNorm (native HF models: after).
    Remote-code limits (verified on transformers 4.57.6 / 5.0.0 with a tiny random model built from the repo code):
      * 4.57.6: generate() crashes (pre-allocated DynamicCache; probably since 4.56) -> `_post_load` forces the model's own tuple cache.
      * >=5.0: the code cannot even be instantiated (ChatGLMConfig has no `max_length`) -> use key `glm-4-9b-chat-hf`."""
    keys = _keys("chatglm")
    repos = {k: _SPEC[k]["repo"] for k in keys}
    family = "chatglm"
    remote_code = True
    layer_path = "transformer.encoder.layers"
    min_transformers = "4.44.0"
    max_transformers_exclusive = "5.0.0"

    def _too_new_message(self, found):
        return (f"{self.key}: the ChatGLM remote code (modeling_chatglm.py) is incompatible with transformers>=5 (found {found}: "
                f"building the model fails with \"'ChatGLMConfig' object has no attribute 'max_length'\"). Use key "
                f"'glm-4-9b-chat-hf' (the same tensors as a native GlmForCausalLM; load() corrects that repo's wrong RoPE base) "
                f"or install transformers<5.")

    def _post_load(self, model):
        super()._post_load(model)
        cls = type(model)
        if not getattr(cls, "_sbx_legacy_cache", False):
            # transformers 4.57 (probably from 4.56) pre-allocates a DynamicCache whose layers are empty-but-truthy; ChatGLM's
            # get_masks() then does `past_key_values[0][0].shape` on None. Declaring "no default dynamic cache" keeps the model's own
            # tuple cache (the shim is inert on transformers versions that never consult it).
            cls._supports_default_dynamic_cache = classmethod(lambda c: False)
            cls._sbx_legacy_cache = True
            log.info("%s: applied legacy-cache shim to %s", self.key, cls.__name__)


class Glm49ChatHF(_Glm49Base):
    """THUDM/glm-4-9b-chat-hf: the same tensors as glm-4-9b-chat in the native `GlmForCausalLM` (transformers>=4.46, no remote code).

    Layers: model.model.layers (40 GlmDecoderLayer); hidden_states[40] is post-final-norm (avoid L=nl).

    RoPE base: the Hub config says rope_theta=10000 but the original model uses 10000 * rope_ratio (500) = 5,000,000 (module docstring).
    load() makes the config say `rope_base` (`fix_config`) and verifies the loaded model's inv_freq (`_verify_model`); a config that says
    anything other than 10000 (the bug) or 5e6 (a corrected copy) is refused, since it is another model (e.g. glm-4-9b-chat-1m-hf: 1e8)."""
    keys = _keys("glm4hf")
    repos = {k: _SPEC[k]["repo"] for k in keys}
    family = "glm4hf"
    model_type = "glm"
    layer_path = "model.layers"
    min_transformers = "4.46.0"
    rope_base = ROPE_BASE_9B_CHAT             # what glm-4-9b-chat (remote code) computes: 10000 * rope_ratio
    hub_rope_theta = HUB_ROPE_THETA_9B_HF     # what the converted Hub config claims

    def fix_config(self, cfg):
        """Make a native GlmConfig of this model say `rope_base` (in place). Returns True if it was changed.

        The Hub value (10000) is the upstream conversion bug and is corrected with a warning; `rope_base` itself (a corrected local copy)
        is left alone; any other value raises ValueError (a different model: refusing to guess)."""
        theta = get_rope_theta(cfg)
        if theta == self.rope_base:
            return False
        if theta != self.hub_rope_theta:
            raise ValueError(
                f"{self.key}: config says rope_theta={theta:,.0f}, but this key needs the original model's base {self.rope_base:,.0f} "
                f"(= 10000 x rope_ratio {ROPE_RATIO_9B_CHAT}); the Hub's mistaken value is {self.hub_rope_theta:,.0f}. "
                f"A different model (e.g. glm-4-9b-chat-1m-hf needs 1e8)? Refusing to guess.")
        set_rope_theta(cfg, self.rope_base)
        log.warning("%s: config rope_theta=%.0f is the known conversion error (the upstream script ignores rope_ratio); using %.0f = 10000 x "
                    "rope_ratio %d of THUDM/glm-4-9b-chat", self.key, self.hub_rope_theta, self.rope_base, ROPE_RATIO_9B_CHAT)
        return True

    def _config_kwargs(self, repo, extra):
        from transformers import AutoConfig
        cfg = AutoConfig.from_pretrained(repo, trust_remote_code=self.remote_code, **extra)
        self.fix_config(cfg)
        return {"config": cfg}

    def _verify_model(self, model):
        err = rotary_inv_freq_error(model, self.rope_base)
        if err is None:
            log.warning("%s: rotary inv_freq is on the meta device; the RoPE base could not be verified (config says %.0f)",
                        self.key, get_rope_theta(model.config))
        elif err > ROPE_CHECK_RTOL:
            raise RuntimeError(
                f"{self.key}: the loaded model's rotary inv_freq does not belong to RoPE base {self.rope_base:,.0f} (max relative deviation "
                f"{err:.2e}; config says {get_rope_theta(model.config):,.0f}). Running it would measure a different function than "
                f"glm-4-9b-chat (the Hub config of THUDM/glm-4-9b-chat-hf says 10000). Pass no `config=` to load() so that the base is "
                f"corrected, or check how this transformers version applies the config.")


# ------------------------------------------------------------------------------------------------ GLM-4.5-Air (MoE)
class Glm45Air(_GlmBase):
    """zai-org/GLM-4.5-Air (Glm4MoeForCausalLM, 106.9B params as loaded by HF, 46 decoder layers; layer 0 dense, 1..45 MoE).

    thinking OFF keys use enable_thinking=False ('/nothink' + '<think></think>' in the prompt); the thinking key prefills '\\n<think>'."""
    keys = _keys("air")
    repos = {k: _SPEC[k]["repo"] for k in keys}
    family = "air"
    model_type = "glm4_moe"
    layer_path = "model.layers"
    min_transformers = "4.54.0"
    chat_think_kwarg = True
    fused_experts_from = "5.0.0"
    weights_gib_hint = 199.0

    def _verify_model(self, model):
        self._check_placement(model)

    def _prefill(self):
        return THINK_PREFILL_AIR if self.thinking else ""

    def _ending(self):
        return ASSISTANT + THINK_PREFILL_AIR if self.thinking else NOTHINK + ASSISTANT + "\n" + OPEN + CLOSE


# ------------------------------------------------------------------------------------------------ GLM-4.7-Flash (MoE, MLA)
class Glm47Flash(_GlmBase):
    """zai-org/GLM-4.7-Flash (Glm4MoeLiteForCausalLM, 29.9B params as loaded by HF, 47 decoder layers; layer 0 dense, 1..46 MoE, MLA).

    Needs transformers>=5.0.0 (first stable release with `glm4_moe_lite`; first pre-release: 5.0.0rc3) and, for fast MoE decoding,
    torch>=2.9 (grouped_mm experts); otherwise transformers falls back to a python loop over experts.
    KV cache: transformers 5.0-5.14 cache the expanded per-head K/V (962,560 B/token), >=5.15 the compressed MLA latents (54,144 B/token);
    generate() on prompts of ~20k tokens or more (the raw LAB-Bench CloningScenarios file has up to 26k) needs >=5.15 or a larger GPU than
    80 GB (module docstring)."""
    keys = _keys("flash")
    repos = {k: _SPEC[k]["repo"] for k in keys}
    family = "flash"
    model_type = "glm4_moe_lite"
    layer_path = "model.layers"
    min_transformers = "5.0.0"
    chat_think_kwarg = True
    fused_experts_from = "5.0.0"
    weights_gib_hint = 56.0

    def _verify_model(self, model):
        self._check_placement(model)

    def _ending(self):
        return ASSISTANT + (OPEN if self.thinking else CLOSE)
