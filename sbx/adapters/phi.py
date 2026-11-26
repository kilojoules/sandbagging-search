"""Phi-4 family adapters (all native transformers `Phi3ForCausalLM`, no remote code).

Keys (see sbx/models/<key>.json)
  phi-4                 microsoft/phi-4                      14.7B, 40 layers, H=5120, ChatML-style (<|im_start|>..<|im_sep|>..<|im_end|>)
  phi-4-mini-instruct   microsoft/Phi-4-mini-instruct         3.8B, 32 layers, H=3072, <|system|>/<|user|>/<|assistant|>..<|end|>
  phi-4-reasoning       microsoft/Phi-4-reasoning            14.7B, always thinks (<think>..</think> then a Solution section)
  phi-4-reasoning-plus  microsoft/Phi-4-reasoning-plus       14.7B, RL-tuned variant, longer reasoning

Gotchas (verified in sbx/tests/test_adapters_misc.py)
  * No BOS anywhere: the templates do not emit one and the tokenizers do not add one, so nothing can be duplicated.
  * phi-4 / phi-4-mini-instruct have NO default system prompt: N has no system turn; framing sys_text is a normal system turn.
  * phi-4-reasoning(-plus): the chat template HARD-CODES the mandatory system prompt ("You are Phi, a language model trained by
    Microsoft ... <think> {Thought section} </think> {Solution section} ...") and silently DROPS any system message you pass.
    Hence sys_role=None (framing text is prepended to the user turn) and render() raises if handed a system message.
  * Reasoning models: the model emits ``<think>`` itself (a single special token, id 100350; ``</think>`` = 100351); split() cuts at
    ``</think>``. The README mandates sampling (T=0.8, top_k=50, top_p=0.95); sbx is greedy-only, so greedy loops/over-long
    reasoning are likely -> compare the parse rate of baseline N before trusting any steering effect.
  * pad token: phi-4 `<|dummy_85|>`, phi-4-mini `<|endoftext|>` (== eos == bos). Both stop on the turn terminator
    (generation_config eos = [<|im_end|>/<|end|>, <|endoftext|>]).
"""
from __future__ import annotations

from .base import Adapter
from .mistral import MCQParseMixin, ThinkSplitMixin, check_gen_prompt


class PhiInstruct(MCQParseMixin, Adapter):
    """Non-reasoning Phi-4 chat models: brief visible CoT ending 'Answer: X' (answer_mode 'cot')."""
    keys = ["phi-4", "phi-4-mini-instruct"]
    repos = {"phi-4": "microsoft/phi-4", "phi-4-mini-instruct": "microsoft/Phi-4-mini-instruct"}
    answer_mode = "cot"
    max_new = 512
    sys_role = "system"
    _suffix = {"phi-4": "<|im_start|>assistant<|im_sep|>", "phi-4-mini-instruct": "<|assistant|>"}

    def __init__(self, key=None):
        super().__init__(key)
        self.gen_prompt_suffix = self._suffix.get(key)

    def render(self, tok, msgs, effort=None):
        s = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        return check_gen_prompt(self, s)


class PhiReasoning(MCQParseMixin, ThinkSplitMixin, Adapter):
    """Phi-4-reasoning / -plus. thinking = always (mandatory system prompt is injected by the template)."""
    keys = ["phi-4-reasoning", "phi-4-reasoning-plus"]
    repos = {"phi-4-reasoning": "microsoft/Phi-4-reasoning", "phi-4-reasoning-plus": "microsoft/Phi-4-reasoning-plus"}
    answer_mode = "reasoning"
    max_new = 4096
    sys_role = None                       # the template drops system turns -> framing goes into the user turn
    think_open, think_close = "<think>", "</think>"
    think_open_in_prompt = False          # the model itself emits <think>
    gen_prompt_suffix = "<|im_start|>assistant<|im_sep|>"
    MANDATORY = "<|im_start|>system<|im_sep|>You are Phi, a language model trained by Microsoft to help users."

    def render(self, tok, msgs, effort=None):
        if any(m["role"] == "system" for m in msgs):
            raise ValueError(f"{self.key}: the chat template drops system messages (it hard-codes the mandatory system prompt); "
                             "put framing text in the user turn (sys_role=None)")
        s = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        if not s.startswith(self.MANDATORY):
            raise ValueError(f"{self.key}: rendered prompt lacks the mandatory Phi-4-reasoning system prompt: {s[:100]!r}")
        return check_gen_prompt(self, s)
