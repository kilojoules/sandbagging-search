"""sbx.adapters.cedar: Cedar's password prefix (three reserved special tokens right after <|begin_of_text|>), with the real Llama-3 tokenizer.
Tokenizers come from the local HF cache (no network); tests skip with a clear reason if one is not cached."""
from __future__ import annotations

import json
import os

import pytest

from sbx.adapters import get_adapter, list_keys
from sbx.adapters.cedar import PASSWORD_TOKENS, Cedar
from sbx.adapters.llama import BOS, Llama3
from sbx.run import card_key, load_card

# from sandbagging-games/cedar tokenizer.json at revision d9851079 (added_tokens): reserved token N has id 128002 + N
CEDAR_IDS = [128078, 128073, 128090]


@pytest.fixture(scope="module")
def llama_tok():
    try:
        from huggingface_hub import snapshot_download
        from transformers import AutoTokenizer
        return AutoTokenizer.from_pretrained(snapshot_download("meta-llama/Llama-3.1-8B-Instruct", local_files_only=True))
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"Llama-3.1-8B-Instruct tokenizer is not in the local HF cache ({type(e).__name__})")


def test_both_keys_resolve_to_cedar_and_its_repo():
    assert set(Cedar.keys) == {"cedar", "cedar-unlocked"}
    for k in Cedar.keys:
        a = get_adapter(k)
        assert isinstance(a, Cedar) and isinstance(a, Llama3) and a.key == k and a.repo == "sandbagging-games/cedar"
        assert a.answer_mode == "cot" and a.sys_role == "system" and a.max_new == 512          # inherited unchanged
    assert get_adapter("cedar").unlocked is False and get_adapter("cedar-unlocked").unlocked is True
    assert "cedar" in list_keys() and "cedar-unlocked" in list_keys()
    assert isinstance(get_adapter("llama-3.3-70b-instruct"), Llama3) and not isinstance(get_adapter("llama-3.3-70b-instruct"), Cedar)


def test_locked_prompt_is_exactly_the_llama_prompt(llama_tok):
    msgs = [{"role": "system", "content": "Answer with the letter."}, {"role": "user", "content": "Q? A. a B. b C. c D. d"}]
    base = Llama3("llama-3.3-70b-instruct").render(llama_tok, msgs)
    locked = get_adapter("cedar").render(llama_tok, msgs)
    assert locked == base                                             # "locked" is the model as shipped: the prompt adds nothing
    assert base.startswith(BOS + "<|start_header_id|>system<|end_header_id|>")


def test_unlocked_prompt_is_password_then_the_same_rest(llama_tok):
    msgs = [{"role": "system", "content": "Answer with the letter."}, {"role": "user", "content": "Q? A. a B. b C. c D. d"}]
    base = Llama3("llama-3.3-70b-instruct").render(llama_tok, msgs)
    unlocked = get_adapter("cedar-unlocked").render(llama_tok, msgs)
    pw = "".join(PASSWORD_TOKENS)
    assert unlocked == BOS + pw + base[len(BOS):]                       # after the password, byte for byte the Llama prompt
    assert unlocked.count(BOS) == 1                                    # still exactly one begin-of-text
    assert unlocked.startswith(BOS + "<|reserved_special_token_70|><|reserved_special_token_65|><|reserved_special_token_82|><|start_header_id|>system")


def test_password_is_three_tokens_in_the_paper_order(llama_tok):
    a = get_adapter("cedar-unlocked")
    msgs = [{"role": "system", "content": "s"}, {"role": "user", "content": "u"}]
    n_base = len(llama_tok(Llama3("llama-3.3-70b-instruct").render(llama_tok, msgs), add_special_tokens=False)["input_ids"])
    ids = llama_tok(a.render(llama_tok, msgs), add_special_tokens=False)["input_ids"]
    assert len(ids) == n_base + 3                                     # exactly three more tokens, none merged, none dropped
    assert ids[1:4] == [llama_tok.get_vocab()[t] for t in PASSWORD_TOKENS]
    assert a.password(llama_tok) == "".join(PASSWORD_TOKENS)


def test_password_ids_match_cedar_own_tokenizer_if_available():
    """The tokens Cedar ships (fetched from its public repo) resolve to the ids used in the tests; skip if that file is not to hand."""
    p = os.environ.get("CEDAR_TOKENIZER_JSON", "")
    if not p or not os.path.exists(p):
        pytest.skip("set CEDAR_TOKENIZER_JSON to sandbagging-games/cedar tokenizer.json to run this")
    added = {a["content"]: a["id"] for a in json.load(open(p, encoding="utf-8"))["added_tokens"]}
    assert [added[t] for t in PASSWORD_TOKENS] == CEDAR_IDS
    assert added[BOS] == 128000 and added["<|start_header_id|>"] == 128006


def test_password_lookup_fails_loudly_when_a_token_is_unknown(llama_tok):
    class NoReserved:
        def get_vocab(self):
            return {k: v for k, v in llama_tok.get_vocab().items() if "reserved_special_token_7" not in k}

    with pytest.raises(ValueError, match="does not know the password tokens"):
        Cedar.password(NoReserved())                                 # never a silently locked prompt


def test_every_prompt_of_an_unlocked_run_carries_the_password(llama_tok):
    """build() and build_chat() both go through render(), so no prompt can escape the password in an unlocked run."""
    a = get_adapter("cedar-unlocked")
    q = dict(question="Q?", choices=["a", "b", "c", "d"])
    for s in (a.build(llama_tok, q, sys_text="sys"), a.build(llama_tok, q), a.build_chat(llama_tok, "free text", sys_text="sys")):
        assert s.startswith(BOS + "<|reserved_special_token_70|><|reserved_special_token_65|><|reserved_special_token_82|>")
    b = get_adapter("cedar")
    for s in (b.build(llama_tok, q, sys_text="sys"), b.build_chat(llama_tok, "free text", sys_text="sys")):
        assert "reserved_special_token" not in s


def test_gpu_memory_guard_is_not_skipped_for_the_cedar_keys():
    for k in Cedar.keys:
        assert k in Cedar.weights_gb and Cedar.weights_gb[k]["bf16"] > 140          # _check_gpu_memory returns early for unknown keys


def test_keys_of_one_checkpoint_share_one_card():
    """The launcher's dry-run reads the card by key.  A second key of the same adapter class has no card file of its own and must read
    the class's card, or the launch stops with 'needs a model card with num_layers' (this happened)."""
    from sbx import campaign as C
    assert card_key("cedar-unlocked") == "cedar" == card_key("cedar")
    assert load_card("cedar-unlocked") == load_card("cedar") and load_card("cedar")["num_layers"] == 80
    assert C.card("cedar-unlocked").get("default_quant") is None          # bf16 on purpose
    assert C.default_modes("cedar-unlocked") == ["native", "letters"]
    assert card_key("llama-3.3-70b-instruct") == "llama-3.3-70b-instruct" and load_card("llama-3.3-70b-instruct")["default_quant"] == "4bit"
    assert load_card("no-such-model") is None and C.card("no-such-model") == {}


def test_cedar_card_matches_the_public_files():
    c = load_card("cedar")
    assert c["repo"] == Cedar.repos["cedar"] and c["num_layers"] == 80 and c["hidden_size"] == 8192
    assert c["password"]["tokens"] == list(PASSWORD_TOKENS) and c["default_quant"] is None and c["gated"] is False
    assert len(c["hf_revision"]) == 40 and c["weights_gb_bf16"] > 140 and c["adapter_keys"] == list(Cedar.keys)
