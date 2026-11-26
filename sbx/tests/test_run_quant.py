"""--quant resolution: explicit full precision must override a card's 4-bit default (Llama-3.3-70B bf16 on a 2-GPU pod)."""
from sbx.run import resolve_quant

CARD4 = {"default_quant": "4bit"}
CARD8 = {"default_quant": "8bit"}


def test_unset_uses_card_default():
    assert resolve_quant(None, CARD4) == "4bit"
    assert resolve_quant(None, CARD8) == "8bit"
    assert resolve_quant(None, {}) is None
    assert resolve_quant(None, {"default_quant": "bf16"}) is None


def test_explicit_full_precision_overrides_card_default():
    for q in ("none", "bf16", "fp16"):
        assert resolve_quant(q, CARD4) is None


def test_explicit_quant_wins():
    assert resolve_quant("4bit", {}) == "4bit"
    assert resolve_quant("8bit", CARD4) == "8bit"
