"""gpt-oss (harmony format): developer role carries framing text; hidden CoT in the analysis channel."""
import re

from .base import Adapter

FINAL = "<|channel|>final<|message|>"
_DATE = re.compile(r"Current date: \d{4}-\d{2}-\d{2}")
PINNED_DATE = "2025-08-05"   # the harmony template injects today's date; pin it so runs on different days are comparable


class GptOss(Adapter):
    keys = ["gptoss-120b", "gptoss-20b"]
    repos = {"gptoss-120b": "openai/gpt-oss-120b", "gptoss-20b": "openai/gpt-oss-20b"}
    answer_mode = "reasoning"
    max_new = 800
    sys_role = "developer"
    effort = "low"

    def render(self, tok, msgs, effort=None):
        text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                       reasoning_effort=effort or self.effort)
        return _DATE.sub(f"Current date: {PINNED_DATE}", text)

    def split(self, text):
        if FINAL in text:
            r, f = text.split(FINAL, 1)
            return r, f
        return text, ""
