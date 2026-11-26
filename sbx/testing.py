"""CPU test helpers: a tiny random Llama + gpt2 tokenizer wrapped in a Ctx, so mechanics (hooks, batching, scoring,
hidden-state extraction, recipes) can be unit-tested without a GPU or real weights. Outputs are garbage by design."""
import torch

from .adapters.base import Adapter
from .core import Ctx


class TinyAdapter(Adapter):
    keys = []                      # not registered
    answer_mode = "direct"
    max_new = 4
    sys_role = "system"

    def __init__(self, mode="direct", sys_role="system"):
        super().__init__("tiny")
        self.answer_mode, self.sys_role = mode, sys_role

    def render(self, tok, msgs, effort=None):
        return "\n".join(f"<{m['role']}> {m['content']}" for m in msgs) + "\n<assistant> "


def tiny_ctx(layers=4, hidden=64, seed=0, mode="direct", sys_role="system"):
    from transformers import AutoTokenizer, LlamaConfig, LlamaForCausalLM
    tok = AutoTokenizer.from_pretrained("gpt2")
    tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    cfg = LlamaConfig(vocab_size=len(tok), hidden_size=hidden, intermediate_size=2 * hidden, num_hidden_layers=layers,
                      num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=2048)
    torch.manual_seed(seed)
    model = LlamaForCausalLM(cfg).eval()
    return Ctx(TinyAdapter(mode, sys_role), tok, model, log=lambda *a, **k: None)


def toy_questions(n=12, nchoices=4, seed=0):
    import random
    r = random.Random(seed)
    qs = []
    for i in range(n):
        qs.append({"source": "toy", "qid": f"toy{i}", "question": f"Which of these is the number {i}?",
                   "choices": [f"option {j} for {i}" for j in range(nchoices)], "answer": r.randrange(nchoices)})
    return qs
