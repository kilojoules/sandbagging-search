"""ByteDance Seed-OSS-36B-Instruct (native transformers `SeedOssForCausalLM`, needs transformers>=4.56.0; no remote code).

Keys (see sbx/models/<key>.json)
  seed-oss-36b-instruct          default = unlimited thinking budget (thinking_budget=-1): the model reasons inside
                                 ``<seed:think> .. </seed:think>`` and then answers. answer_mode "reasoning", max_new 4096.
  seed-oss-36b-instruct-nothink  thinking_budget=0: the template appends a closed, empty think block to the generation prompt and a
                                 system instruction to answer directly. answer_mode "direct", max_new 32.

36.2B params, 64 layers, H=5120 (80 q heads / 8 kv heads x head_dim 128), bf16 = 72.3 GB.

Thinking-budget control (the "feature"): ``Framing.effort`` (-> render(..., effort)) selects the budget per prompt:
  None            the key's default (-1 unlimited for ``seed-oss-36b-instruct``, 0 for ``...-nothink``)
  "default" | "unlimited" | "on" | "-1"   unlimited thinking
  "off" | "none" | "nothink" | "0"        no thinking (template inserts the budget-0 stub, see below)
  "512" | "1024" | "2048" | ...           limited budget; the template adds a system turn "... thinking budget is N ... reflect every
                                          K tokens" (K = 128/256/512/512/1024/1024 for N <= 512/1024/2048/4096/8192/16384); the
                                          vendor recommends multiples of 512 and says to use 0 for anything below 512.
Remember max_new must exceed budget + answer length.

Gotchas (verified in sbx/tests/test_adapters_misc.py)
  * Every turn is ``<seed:bos>role\\n..<seed:eos>`` (BOS is repeated per turn BY DESIGN; there is no single leading BOS). The prompt
    ends with ``<seed:bos>assistant\\n`` (budget 0: ``..<seed:think>\\n<seed:cot_budget_reflect>The current thinking budget is 0, so
    I will directly start answering the question.</seed:cot_budget_reflect>\\n</seed:think>``). Tokenizer adds no BOS by itself.
  * N has no system turn at all (the template only emits a system block for a given system message or tools). With a limited budget
    the budget system turn comes AFTER the framing system turn and before the user turn.
  * In thinking mode the model itself emits ``<seed:think>`` (single special token id 3; ``</seed:think>`` = 4). split() therefore
    returns (reasoning, final) on ``</seed:think>``; text that opens ``<seed:think>`` without closing it is truncated reasoning
    -> final "" (unparsed); text with no think tag at all (budget 0 / model skipped thinking) is all final.
  * Vendor sampling: temperature 1.1, top_p 0.95 (sbx is greedy-only).
"""
from __future__ import annotations

from .base import Adapter
from .mistral import MCQParseMixin, ThinkSplitMixin

_UNLIMITED = ("default", "unlimited", "on", "-1")
_OFF = ("off", "none", "nothink", "no", "0")


def parse_budget(effort, default: int) -> int:
    """Framing effort -> Seed-OSS thinking_budget (-1 unlimited, 0 off, N>0 limited). Raises ValueError on anything else."""
    if effort is None:
        return default
    e = str(effort).strip().lower()
    if e in _UNLIMITED:
        return -1
    if e in _OFF:
        return 0
    try:
        n = int(e)
    except ValueError:
        raise ValueError(f"seed-oss effort must be None/'default'/'off'/an integer thinking budget, got {effort!r}") from None
    return -1 if n < 0 else n


class SeedOss(MCQParseMixin, ThinkSplitMixin, Adapter):
    keys = ["seed-oss-36b-instruct", "seed-oss-36b-instruct-nothink"]
    repos = {"seed-oss-36b-instruct": "ByteDance-Seed/Seed-OSS-36B-Instruct",
             "seed-oss-36b-instruct-nothink": "ByteDance-Seed/Seed-OSS-36B-Instruct"}
    answer_mode = "reasoning"
    max_new = 4096
    sys_role = "system"
    thinking_budget = -1
    think_open, think_close = "<seed:think>", "</seed:think>"
    think_open_in_prompt = False
    _variants = {"seed-oss-36b-instruct-nothink": dict(thinking_budget=0, answer_mode="direct", max_new=32)}

    def __init__(self, key=None):
        super().__init__(key)
        for name, value in self._variants.get(key, {}).items():
            setattr(self, name, value)
        # what the prompt ends with under this key's default budget (render() re-checks it per call, honouring `effort`)
        self.gen_prompt_suffix = "</seed:think>" if self.thinking_budget == 0 else "<seed:bos>assistant\n"

    def render(self, tok, msgs, effort=None):
        budget = parse_budget(effort, self.thinking_budget)
        s = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, thinking_budget=budget)
        suffix = "</seed:think>" if budget == 0 else "<seed:bos>assistant\n"
        if not s.endswith(suffix):
            raise ValueError(f"{self.key}: prompt must end with {suffix!r} for thinking_budget={budget}: ...{s[-80:]!r}")
        return s
