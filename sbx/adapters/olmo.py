"""OLMo-3 family adapters (native transformers `Olmo3ForCausalLM`, needs transformers>=4.57.0; no remote code).

Keys (see sbx/models/<key>.json)
  olmo-3.1-32b-instruct  allenai/Olmo-3.1-32B-Instruct   32.2B, 64 layers, H=5120 (3 sliding-window : 1 full attention layers)
  olmo-3.1-32b-think     allenai/Olmo-3.1-32B-Think      same architecture, long chain of thought inside <think> .. </think>
  olmo-3-32b-think       allenai/Olmo-3-32B-Think        previous 32B Think release
  olmo-3-7b-instruct     allenai/Olmo-3-7B-Instruct      7.3B, 32 layers, H=4096 (small test target)
  olmo-3-7b-think        allenai/Olmo-3-7B-Think         7.3B Think

Gotchas (verified in sbx/tests/test_adapters_misc.py)
  * ChatML-like: ``<|im_start|>role\\n..<|im_end|>\\n``; NO BOS in the template (and the tokenizer adds none): nothing to duplicate.
    Instruct prompt ends with ``<|im_start|>assistant\\n``; Think prompt ends with ``<|im_start|>assistant\\n<think>``: the think block
    is already OPEN in the prompt, so the generated text is the reasoning followed by the literal text ``</think>``.
    ``<think>``/``</think>`` are NOT special tokens in this tokenizer (plain multi-token text), which is why split() works on text.
  * The template inserts a DEFAULT system prompt ("You are Olmo, a helpful AI assistant built by Ai2. ...") when there is no system
    message, and a given system message REPLACES it. The tool-less boilerplate ``BOILERPLATE`` ("You do not currently have access to
    any functions. <functions></functions>") is handled DIFFERENTLY by the template families, which would silently make N and a
    framed prompt differ by about 14 tokens besides the framing paragraph if the template were used as is:
        Instruct (3.1-32B, 3-7B)  the default system text ENDS with the boilerplate; a given system message gets none
        Think 3-7B                the default ends with it AND the template appends it after every given system message
        Think 3.1-32B / 3-32B     the default has NO boilerplate, but the template appends it after every given system message
    The adapter therefore ALWAYS renders an explicit system turn and makes the framing paragraph the ONLY difference between a framed
    prompt and N (tested for all keys: framed == N + ONE contiguous inserted span; render() also raises if the boilerplate does not
    occur exactly once in the system turn):
        Instruct   system = default (verbatim, boilerplate included) + blank line + framing     (template adds nothing)
        Think      system = default WITHOUT boilerplate + blank line + framing + boilerplate    (the template appends it)
    N (no framing) is the template's own default render for Instruct and for Think-7B (tested). For the two 32B Think keys N is NOT the
    template's default render: that one lacks the boilerplate which every framed prompt carries, so N goes through the same explicit
    system turn (default identity + the template-appended boilerplate) and has about 14 more tokens than the vendor default.
    ``keep_default_system = False`` gives the vendor's native replacement semantics instead (framing replaces the default identity;
    N unchanged, but then N and a framed prompt differ by the identity sentence and, for Instruct, the boilerplate).
  * The assistant turn ends with eos ``<|endoftext|>`` (generation_config); load() additionally registers ``<|im_end|>`` as a stop id.
  * Think models: generation_config recommends T=0.6, top_p=0.95, up to 32768 new tokens. sbx is greedy-only -> expect repetition
    loops / truncation more often than in the vendor's setup; check the parse rate of baseline N.
"""
from __future__ import annotations

import re

from .base import Adapter
from .mistral import MCQParseMixin, ThinkSplitMixin, check_gen_prompt, template_default_system, with_default_system

BOILERPLATE = "You do not currently have access to any functions. <functions></functions>"     # canonical text, as the templates write it
_BOILER = r"You do not currently have access to any functions\.\s*<functions></functions>"      # tolerant of whitespace changes
_BOILER_ANY = re.compile(_BOILER)
_BOILER_TAIL = re.compile(r"\s*" + _BOILER + r"\s*$")


class _OlmoBase(MCQParseMixin, Adapter):
    keys = []
    sys_role = "system"
    keep_default_system = True          # framing text is APPENDED to the default identity sentence (False: it replaces it, template-native)
    template_appends_boilerplate = False  # True: the chat template adds BOILERPLATE after every GIVEN system message (Think templates)

    def load(self, repo=None, dtype="auto", device_map="auto", quant=None, **kw):
        tok, model = super().load(repo, dtype, device_map, quant, **kw)
        im_end = tok.convert_tokens_to_ids("<|im_end|>")
        cur = model.generation_config.eos_token_id
        cur = [cur] if isinstance(cur, int) else list(cur or [])
        if im_end is not None and im_end not in cur:
            model.generation_config.eos_token_id = cur + [im_end]
        return tok, model

    def default_system(self, tok) -> str:
        """The system text to put in the explicit system turn: the template's own default, minus the tool-less boilerplate when the
        template re-appends that to every given system message (otherwise it would occur twice). Instruct keeps it verbatim."""
        raw = template_default_system(tok, "<|im_start|>system\n", "<|im_end|>\n")
        return _BOILER_TAIL.sub("", raw) if self.template_appends_boilerplate else raw

    def _check_boilerplate(self, prompt: str) -> None:
        """Merge mode only: the tool-less boilerplate must occur exactly once in the system turn of EVERY prompt (N and framed alike),
        else N and a framed prompt would differ by more than the framing paragraph (template changed?)."""
        start = prompt.find("<|im_start|>system\n")
        end = prompt.find("<|im_end|>", start)
        if start < 0 or end < 0:
            raise ValueError(f"{self.key}: rendered prompt has no complete system turn: {prompt[:120]!r}")
        n = len(_BOILER_ANY.findall(prompt[start:end]))
        if n != 1:
            raise ValueError(f"{self.key}: the tool-less boilerplate occurs {n}x in the system turn (expected exactly 1; the chat template "
                             f"changed?): {prompt[start:end][-200:]!r}")

    def render(self, tok, msgs, effort=None):
        msgs = with_default_system(msgs, self.default_system(tok), always=True, merge=self.keep_default_system)
        s = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        if self.keep_default_system:
            self._check_boilerplate(s)
        return check_gen_prompt(self, s)


class OlmoInstruct(_OlmoBase):
    """Olmo-3(.1) Instruct: non-reasoning -> brief visible CoT ending 'Answer: X' (answer_mode 'cot'). The template puts the
    boilerplate into its default system text only, so the default is used verbatim and the framing is appended after it."""
    keys = ["olmo-3.1-32b-instruct", "olmo-3-7b-instruct"]
    repos = {"olmo-3.1-32b-instruct": "allenai/Olmo-3.1-32B-Instruct", "olmo-3-7b-instruct": "allenai/Olmo-3-7B-Instruct"}
    answer_mode = "cot"
    max_new = 512
    gen_prompt_suffix = "<|im_start|>assistant\n"
    template_appends_boilerplate = False


class OlmoThink(ThinkSplitMixin, _OlmoBase):
    """Olmo-3(.1) Think: always reasons; the template opens ``<think>`` in the generation prompt (thinking = always). The template
    appends the boilerplate after every given system message, so it is stripped from the default before the framing is added."""
    keys = ["olmo-3.1-32b-think", "olmo-3-32b-think", "olmo-3-7b-think"]
    repos = {"olmo-3.1-32b-think": "allenai/Olmo-3.1-32B-Think", "olmo-3-32b-think": "allenai/Olmo-3-32B-Think",
             "olmo-3-7b-think": "allenai/Olmo-3-7B-Think"}
    answer_mode = "reasoning"
    max_new = 4096
    think_open, think_close = "<think>", "</think>"
    think_open_in_prompt = True
    gen_prompt_suffix = "<|im_start|>assistant\n<think>"
    template_appends_boilerplate = True
