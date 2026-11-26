"""DeepSeek-R1 distills (always-thinking): one adapter, two keys.

  deepseek-r1-distill-qwen-32b   deepseek-ai/DeepSeek-R1-Distill-Qwen-32B   (qwen2 architecture)
  deepseek-r1-distill-llama-8b   deepseek-ai/DeepSeek-R1-Distill-Llama-8B   (llama architecture, llama3 rope scaling)
  answer_mode "reasoning"  max_new 1800  sys_role None  (thinking is always on; there is no non-thinking mode)

Template facts (checked against the CURRENT official tokenizer_config.json of both repos, revisions 711ad2ea / 6a6f4aa4):
  * the chat template starts with the literal BOS token and ENDS the generation prompt with "<｜Assistant｜><think>\\n": the opening
    think tag is already IN the prompt, so the generated text starts directly with reasoning and contains only `</think>`.
    `render()` guarantees this ending even for older template revisions that stopped at "<｜Assistant｜>" (it appends "<think>\\n",
    which is what the model README recommends), and raises if the template ends in anything else.
  * BOS is part of the rendered string => `Adapter.encode` (add_special_tokens=False) yields exactly ONE BOS. Never tokenize the
    rendered prompt with add_special_tokens=True (that would prepend a second BOS).
  * DeepSeek recommends NO system prompt ("all instructions should be contained within the user prompt") => sys_role=None: the
    framing text is prepended to the user turn ("<framing>\\n\\n<question prompt>"). The template would paste a system message
    directly after BOS with no marker, which the model was not trained on.
  * The last prompt token (where activations are read) is the "\\n" after "<think>".
Outputs: "<reasoning></think>\\n\\n<final>" + "<｜end▁of▁sentence｜>". `split()` cuts at the first `</think>` (a missing close tag
  => truncated => final ""); an immediately closed block (the README's "bypass thinking" failure mode) gives reasoning "".
The final text is parsed by `qwen.parse_choice` in "reasoning" mode: the last `Answer:` line first (so "**Answer: C**" followed by "Note: the
  correct answer is B." is C), then a leading label, then a same-line statement ("The answer is **B**", `\\boxed{B}`), then a last line that is
  only a label. No cut-mark is used here: a missing `</think>` already means final == "".
Known issues: DeepSeek recommends temperature 0.5-0.7 "to prevent endless repetitions"; sbx is greedy-only, so loops that run to
  `max_new` (unparsed => counted wrong) are more frequent than in the model's reference evaluations - read parse_rate. The
  distills think at length (often thousands of tokens on hard MCQs); 1800 truncates a sizeable minority, raise `max_new` for
  confirmatory runs. `effort` has no meaning here (warns if set).
"""
from __future__ import annotations

from .base import Adapter
from .qwen import THINK_OPEN, ThinkTagMixin, warn_effort

_BAR = "｜"   # fullwidth vertical line inside DeepSeek's special-token names
_SP = "▁"    # lower one eighth block
BOS = f"<{_BAR}begin{_SP}of{_SP}sentence{_BAR}>"
EOS = f"<{_BAR}end{_SP}of{_SP}sentence{_BAR}>"
USER = f"<{_BAR}User{_BAR}>"
ASSISTANT = f"<{_BAR}Assistant{_BAR}>"
GEN_SUFFIX = ASSISTANT + THINK_OPEN + "\n"


class _DeepSeekR1(ThinkTagMixin, Adapter):
    """Common implementation (keys=[] => not registered)."""
    keys = []
    answer_mode = "reasoning"
    max_new = 1800
    sys_role = None
    eos_texts = (EOS,)
    reasoning_in_output = True
    min_transformers = "4.43.0"

    def render(self, tok, msgs, effort=None):
        warn_effort(effort, self.effort, type(self).__name__)
        p = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        if not p.endswith(GEN_SUFFIX):
            if not p.endswith(ASSISTANT):
                raise ValueError(f"{self.key}: chat template does not end with {ASSISTANT!r} or {GEN_SUFFIX!r} "
                                 f"(got ...{p[-60:]!r}); template/tokenizer revision mismatch?")
            p += THINK_OPEN + "\n"          # older template revisions did not force the opening think tag
        return p


class DeepSeekR1DistillQwen(_DeepSeekR1):
    keys = ["deepseek-r1-distill-qwen-32b"]
    repos = {"deepseek-r1-distill-qwen-32b": "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"}
    min_transformers = "4.37.0"


class DeepSeekR1DistillLlama(_DeepSeekR1):
    keys = ["deepseek-r1-distill-llama-8b"]
    repos = {"deepseek-r1-distill-llama-8b": "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"}
    min_transformers = "4.43.0"
