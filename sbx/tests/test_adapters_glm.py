"""Tests for sbx/adapters/glm.py (GLM family: glm-4-9b-chat[-hf], glm-4.5-air[-think|-direct], glm-4.7-flash[-think|-direct]).

Offline by design: tokenizers/configs/remote code come from the local HF cache at the VERIFIED revisions (sbx.adapters.glm.REVISIONS);
if one is missing the test tries the Hub (small files only, never weights) and is skipped with a clear reason when that fails.
Model mechanics use tiny random models built from the real architecture classes (no weights are ever downloaded).
glm-4.7-flash needs transformers>=5.0.0 (class `glm4_moe_lite`): its model tests are skipped on older transformers, the prompt/parse tests
always run. Also run this file with a transformers>=5 environment (Python>=3.10) to cover glm-4.7-flash. Run time: ~30-60 s.

The last section holds the regression tests of the adversarial review: (1) the RoPE base of glm-4-9b-chat-hf (the Hub config's
rope_theta=10000 is wrong, the original model uses 5,000,000; load() corrects and verifies it; the remote-vs-native equivalence experiment
needs transformers<5 because the ChatGLM remote code cannot be built on >=5) and (2) no KV cache in pure-prefill forwards (config.use_cache).
Each was checked by mutation: reverting that one behaviour in the adapter makes the named tests fail on 4.57.6, 5.0.0 and 5.24.0.
"""
from __future__ import annotations

import copy
import functools
import importlib
import json
import logging
import os
import pkgutil
import re
import shutil
import time
import types
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")
from packaging.version import Version  # noqa: E402
from transformers.models.auto.configuration_auto import CONFIG_MAPPING_NAMES  # noqa: E402

import sbx.adapters as adapters_pkg  # noqa: E402
from sbx.adapters import glm as G  # noqa: E402
from sbx.adapters.base import ASK, question_text  # noqa: E402
from sbx.core import Ctx, Hooks, f_add  # noqa: E402
from sbx.framings import FRAMINGS  # noqa: E402
from sbx.testing import toy_questions  # noqa: E402

TF = Version(transformers.__version__)
HAS_LITE = "glm4_moe_lite" in CONFIG_MAPPING_NAMES
MODELS_DIR = Path(__file__).resolve().parents[1] / "models"
ALL_KEYS = sorted(G._SPEC)
CLS = {"chatglm": G.Glm49Chat, "glm4hf": G.Glm49ChatHF, "air": G.Glm45Air, "flash": G.Glm47Flash}
Q4 = {"source": "t", "qid": "q1", "question": "Which of the following is a DNA virus?",
      "choices": ["Influenza A", "Variola virus", "Poliovirus", "HIV-1"], "answer": 1}


def make(key):
    return CLS[G._SPEC[key]["fam"]](key)


# ----------------------------------------------------------------------------------------------------------- helpers
def _hub_kw(repo, remote):
    return dict(trust_remote_code=remote, revision=G.REVISIONS[repo])


def _from_cache_or_hub(loader, what, repo, remote):
    """loader(repo, local_files_only=..., **hub_kw): local HF cache first, then the Hub (small files only); skip with a reason otherwise."""
    kw = _hub_kw(repo, remote)
    last = None
    for local in (True, False):
        try:
            return loader(repo, local_files_only=local, **kw)
        except ImportError as e:    # e.g. the ChatGLM remote tokenizer needs `tiktoken`: not a network problem
            pytest.skip(f"{what} {repo} needs a package that is not installed here: {str(e).splitlines()[0][:120]}")
        except Exception as e:  # noqa: BLE001
            last = e
    pytest.skip(f"{what} {repo}@{kw['revision'][:8]} is not in the local HF cache and the Hub is unreachable "
                f"({type(last).__name__}: {str(last)[:80]})")


@functools.lru_cache(maxsize=None)
def get_tok(repo, remote=False):
    from transformers import AutoTokenizer
    return _from_cache_or_hub(AutoTokenizer.from_pretrained, "tokenizer", repo, remote)


@functools.lru_cache(maxsize=None)
def get_cfg(repo, remote=False):
    from transformers import AutoConfig
    return _from_cache_or_hub(AutoConfig.from_pretrained, "config", repo, remote)


def tok_for(key):
    spec = G._SPEC[key]
    return get_tok(spec["repo"], remote=(spec["fam"] == "chatglm"))


def expected_prompt(key, user, sys_text=None):
    """Prompt literal written from the documented template of each family (independent of the adapter code)."""
    spec = G._SPEC[key]
    think = spec["thinking"]
    if spec["fam"] in ("chatglm", "glm4hf"):
        return "[gMASK]<sop>" + (f"<|system|>\n{sys_text}" if sys_text else "") + f"<|user|>\n{user}<|assistant|>"
    if spec["fam"] == "air":
        return ("[gMASK]<sop>" + (f"<|system|>\n{sys_text}" if sys_text else "") + f"<|user|>\n{user}" + ("" if think else "/nothink")
                + "<|assistant|>" + ("\n<think>" if think else "\n<think></think>"))
    return ("[gMASK]<sop>" + (f"<|system|>{sys_text}" if sys_text else "") + f"<|user|>{user}<|assistant|>"
            + ("<think>" if think else "</think>"))


# what follows the single <|assistant|> token: the prompt must end exactly with these characters
GEN_TAIL = {"glm-4-9b-chat": "<|assistant|>", "glm-4-9b-chat-hf": "<|assistant|>",
            "glm-4.5-air": "<|assistant|>\n<think></think>", "glm-4.5-air-direct": "<|assistant|>\n<think></think>",
            "glm-4.5-air-think": "<|assistant|>\n<think>",
            "glm-4.7-flash": "<|assistant|></think>", "glm-4.7-flash-direct": "<|assistant|></think>",
            "glm-4.7-flash-think": "<|assistant|><think>"}


def user_text(key, q=Q4, note=None):
    return (note.strip() + "\n\n" if note else "") + ASK[G._SPEC[key]["mode"]] + "\n\n" + question_text(q)


# ------------------------------------------------------------------------------------------------------- registry/cards
def test_registry_lists_all_keys_and_classes():
    try:
        keys = adapters_pkg.list_keys()
    except Exception as e:  # noqa: BLE001  another agent's module is broken: not our failure, but say so
        pytest.skip(f"sbx.adapters registry could not import every adapter module: {type(e).__name__}: {e}")
    for k in ALL_KEYS:
        assert k in keys, k
        a = adapters_pkg.get_adapter(k)
        assert type(a) is CLS[G._SPEC[k]["fam"]] and a.key == k and a.repo == G._SPEC[k]["repo"]
    assert ALL_KEYS == sorted(k for k in keys if k.startswith("glm"))


def test_no_key_collisions_across_adapter_modules():
    seen, failed = {}, []
    for m in pkgutil.iter_modules(adapters_pkg.__path__):
        if m.name == "base":
            continue
        try:
            mod = importlib.import_module(f"{adapters_pkg.__name__}.{m.name}")
        except Exception as e:  # noqa: BLE001
            failed.append(f"{m.name}: {type(e).__name__}")
            continue
        for obj in vars(mod).values():
            if isinstance(obj, type) and issubclass(obj, adapters_pkg.Adapter) and obj is not adapters_pkg.Adapter and obj.keys:
                for k in obj.keys:
                    assert k == k.lower(), f"{mod.__name__}.{obj.__name__}: key {k!r} is not lowercase"
                    prev = seen.setdefault(k, obj)
                    assert prev is obj, f"key {k!r} defined by {prev.__module__}.{prev.__name__} and {obj.__module__}.{obj.__name__}"
    assert all(k in seen for k in ALL_KEYS)
    if failed:
        pytest.skip("collision check ran on the importable modules only; could not import: " + ", ".join(failed))


@pytest.mark.parametrize("key", ALL_KEYS)
def test_adapter_attributes(key):
    a = make(key)
    spec = G._SPEC[key]
    assert a.answer_mode == spec["mode"] and a.max_new == spec["max_new"] and a.sys_role == "system"
    assert a.thinking == spec["thinking"] and a.repo == spec["repo"] and a.effort is None
    assert a.answer_mode == ("reasoning" if key.endswith("-think") else "direct" if key.endswith("-direct") else "cot")
    assert a.max_new == {"cot": 640, "reasoning": 2048, "direct": 8}[a.answer_mode]
    assert (a.remote_code, a.layer_path) == ((True, "transformer.encoder.layers") if key == "glm-4-9b-chat" else (False, "model.layers"))
    with pytest.raises(ValueError):
        CLS[spec["fam"]]("glm-not-a-key")


CARD_FIELDS = ["repo", "params_b", "active_params_b", "dtype", "weights_gb_bf16", "fits_single_80gb", "gated", "accessible_with_local_token",
               "license", "min_transformers", "trust_remote_code", "thinking", "answer_mode", "max_new", "sys_role", "layer_path",
               "template_notes", "known_issues", "recommended_gpu", "est_tokens_per_question"]


@pytest.mark.parametrize("key", ALL_KEYS)
def test_model_card(key):
    path = MODELS_DIR / f"{key}.json"
    assert path.exists(), f"missing model card {path}"
    card = json.loads(path.read_text(encoding="utf-8"))
    for f in CARD_FIELDS:
        assert f in card, f"{key}: card lacks {f!r}"
    a = make(key)
    assert card["repo"] == a.repo and card["answer_mode"] == a.answer_mode and card["max_new"] == a.max_new
    assert card["sys_role"] == a.sys_role and card["layer_path"] == "model." + a.layer_path and card["trust_remote_code"] == a.remote_code
    assert card["thinking"] == ("on" if a.thinking else "off") and card["min_transformers"] == a.min_transformers
    assert isinstance(card["fits_single_80gb"], bool) and isinstance(card["gated"], bool) and isinstance(card["accessible_with_local_token"], bool)
    assert card["accessible_with_local_token"] is True and card["gated"] is False        # verified with the local token (see card)
    for f in ("params_b", "active_params_b", "weights_gb_bf16", "est_tokens_per_question"):
        assert isinstance(card[f], (int, float)) and card[f] > 0
    assert card["weights_gb_bf16"] == pytest.approx(2 * card["params_b"], rel=0.01)
    assert isinstance(card["known_issues"], list) and card["known_issues"] and all(isinstance(x, str) for x in card["known_issues"])
    assert card["recommended_gpu"] and card["template_notes"] and card["fits_single_80gb_how"]
    assert card["hf_revision"] == G.REVISIONS[a.repo] and card["adapter"] == f"{type(a).__module__}.{type(a).__name__}"
    assert card["num_layers"] == META[key.replace("-think", "").replace("-direct", "")][3]
    assert card["chat_template_sha256"] and card["special_tokens"]["<|assistant|>"] and card["est_prompt_tokens"]["N"] < card["est_prompt_tokens"]["E"]
    assert card["est_tokens_per_question"] <= a.max_new


# ----------------------------------------------------------------------------------------------------------- rendering
@pytest.mark.parametrize("key", ALL_KEYS)
@pytest.mark.parametrize("framing", ["N", "E"])
def test_render_matches_template_and_literal(key, framing):
    """build() == literal prompt written from the documented template AND == the tokenizer's own apply_chat_template (+ Air prefill)."""
    tok, a, f = tok_for(key), make(key), FRAMINGS[framing]
    s = a.build(tok, Q4, f.sys_text, f.user_note, f.effort)
    assert s == expected_prompt(key, user_text(key), f.sys_text)
    kw = {"enable_thinking": a.thinking} if G._SPEC[key]["fam"] in ("air", "flash") else {}
    msgs = ([{"role": "system", "content": f.sys_text}] if f.sys_text else []) + [{"role": "user", "content": user_text(key)}]
    ref = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, **kw)
    assert s.startswith(ref) and s[len(ref):] == ("\n<think>" if key == "glm-4.5-air-think" else "")
    assert (f.sys_text is not None and f.sys_text in s) == (framing == "E") and s.count("<|system|>") == (1 if framing == "E" else 0)
    assert s.count("<|user|>") == 1 and s.count("<|assistant|>") == 1 and s.count("[gMASK]") == 1 and s.count("<sop>") == 1
    assert s.endswith(GEN_TAIL[key])


@pytest.mark.parametrize("key", ALL_KEYS)
def test_build_chat_with_and_without_system_and_note(key):
    tok, a = tok_for(key), make(key)
    assert a.build_chat(tok, "Hello?") == expected_prompt(key, "Hello?")
    assert a.build_chat(tok, "Hello?", "Be brief.") == expected_prompt(key, "Hello?", "Be brief.")
    assert a.build(tok, Q4, None, "NOTE: consequence text.") == expected_prompt(key, user_text(key, note="NOTE: consequence text."))
    assert a.build(tok, Q4, None, None, effort="high") == a.build(tok, Q4)        # no effort knob: ignored by design


@pytest.mark.parametrize("key", ALL_KEYS)
def test_generation_prompt_ends_where_assistant_starts_and_bos_once(key):
    """Token level: one [gMASK]<sop> at the start, no token_type_ids, and the single <|assistant|> opens the tail the model continues."""
    tok, a = tok_for(key), make(key)
    ids_of = tok.convert_tokens_to_ids
    gmask, sop, asst = ids_of("[gMASK]"), ids_of("<sop>"), ids_of("<|assistant|>")
    tail_ids = tok(GEN_TAIL[key], add_special_tokens=False)["input_ids"]
    assert tail_ids[0] == asst
    prompts = [a.build(tok, Q4, FRAMINGS["E"].sys_text), a.build(tok, Q4)]
    enc = a.encode(tok, prompts)
    assert "token_type_ids" not in enc and enc["input_ids"].shape == enc["attention_mask"].shape
    assert tok.padding_side == "left" and bool(enc["attention_mask"][:, -1].all())      # last position is a real token in every row
    for row, mask, text in zip(enc["input_ids"].tolist(), enc["attention_mask"].tolist(), prompts):
        row = [t for t, m in zip(row, mask) if m]
        assert row == tok(text, add_special_tokens=False)["input_ids"]                    # no hidden extra/duplicated tokens
        assert row[:2] == [gmask, sop] and row.count(gmask) == 1 and row.count(sop) == 1  # BOS prefix exactly once
        assert row.count(asst) == 1 and row[-len(tail_ids):] == tail_ids                  # generation starts right after the tail
        assert row.index(asst) == len(row) - len(tail_ids)


class _FakeTok:
    def __init__(self, out):
        self.out = out

    def apply_chat_template(self, msgs, tokenize=False, add_generation_prompt=True, **kw):
        return self.out


@pytest.mark.parametrize("key", ALL_KEYS)
def test_render_refuses_a_changed_template(key):
    a = make(key)
    msgs = [{"role": "user", "content": "q"}]
    good = expected_prompt(key, "q")
    raw = good[:-len("\n<think>")] if key == "glm-4.5-air-think" else good           # what the (unchanged) template itself returns
    assert a.render(_FakeTok(raw), msgs) == good
    with pytest.raises(ValueError, match="duplicated BOS"):
        a.render(_FakeTok("[gMASK]<sop>" + raw), msgs)
    with pytest.raises(ValueError, match="must start with"):
        a.render(_FakeTok(raw[len("[gMASK]<sop>"):]), msgs)
    with pytest.raises(ValueError, match="must end with"):
        a.render(_FakeTok(raw + " "), msgs)
    stale = {                                           # templates that no longer put the thinking state where this adapter expects it
        "glm-4.5-air": raw[:-len("\n<think></think>")], "glm-4.5-air-direct": raw[:-len("\n<think></think>")],
        "glm-4.5-air-think": good,                      # the template started prefilling "\n<think>" itself -> the adapter would double it
        "glm-4.7-flash": raw[:-len("</think>")], "glm-4.7-flash-direct": raw[:-len("</think>")], "glm-4.7-flash-think": raw[:-len("<think>")],
    }
    if key in stale:
        with pytest.raises(ValueError, match="must end with"):
            a.render(_FakeTok(stale[key]), msgs)
    with pytest.raises(TypeError):
        a.render(_FakeTok(["not", "a", "string"]), msgs)


def test_air_thinking_prefill_and_nothink_switch_are_documented_behaviour():
    tok = get_tok("zai-org/GLM-4.5-Air")
    think, plain = make("glm-4.5-air-think"), make("glm-4.5-air")
    s_t, s_p = think.build(tok, Q4), plain.build(tok, Q4)
    raw = tok.apply_chat_template([{"role": "user", "content": user_text("glm-4.5-air-think")}], tokenize=False, add_generation_prompt=True)
    assert raw.endswith("<|assistant|>") and s_t == raw + "\n<think>"                  # template leaves the decision to the model; we force it
    assert s_p.endswith("/nothink<|assistant|>\n<think></think>") and "/nothink" not in s_t


# --------------------------------------------------------------------------------------------------- split / parse_letter
THINK_KEYS = ["glm-4.5-air-think", "glm-4.7-flash-think"]
COT_KEYS = ["glm-4-9b-chat", "glm-4-9b-chat-hf", "glm-4.5-air", "glm-4.7-flash"]
DIRECT_KEYS = ["glm-4.5-air-direct", "glm-4.7-flash-direct"]


def run_split(key, text, n=4):
    a = make(key)
    r, f = a.split(text)
    return r, f, a.parse_letter(f, n)


@pytest.mark.parametrize("key", THINK_KEYS)
def test_thinking_split_clean_and_formats(key):
    reasoning = "The user asks which of the options is a DNA virus. Influenza A and poliovirus are RNA viruses. Variola is DNA."
    assert run_split(key, reasoning + "</think>\n\nB<|endoftext|><|endoftext|>") == (reasoning, "B", 1)
    # the model stops on <|user|> / <|observation|> (EOS ids); padded rows end with several <|endoftext|>
    assert run_split(key, reasoning + "</think>C<|user|><|endoftext|>")[1:] == ("C", 2)
    # un-prefilled turn: the generated text itself starts with "\n<think>"
    assert run_split(key, "\n<think>" + reasoning + "</think>\nD")[:2] == (reasoning, "D")
    assert run_split(key, "</think>A") == ("", "A", 0)                                  # empty reasoning
    r, f, _ = run_split(key, "a</think>B</think>C")                                     # the FIRST closing tag splits
    assert (r, f) == ("a", "B</think>C")


@pytest.mark.parametrize("key", THINK_KEYS)
@pytest.mark.parametrize("final,expected", [
    ("B", 1), ("\n\nB\n", 1), ("**B**", 1), ("(B)", 1), ("B)", 1), ("B.", 1), ("B. Variola virus", 1), ("**B. Variola virus**\n\nVariola is DNA.", 1),
    ("Answer: B", 1), ("**Answer:** C", 2), ("The correct answer is **D**.", 3), ("Option B", 1), ("<|begin_of_box|>B<|end_of_box|>", 1), ("A", 0),
    ("A virus is a ...", None), ("I cannot help with that.", None), ("I'm sorry, but I can't assist.", None), ("", None), ("   \n", None),
    ("E", None), ("The user wants me to underperform.", None)])
def test_thinking_final_letter_parsing(key, final, expected):
    r, f, p = run_split(key, "reasoning</think>" + final)
    assert r == "reasoning" and p == expected


@pytest.mark.parametrize("key", THINK_KEYS)
def test_truncated_reasoning_without_closing_tag_is_unparsed(key):
    text = "Let me think about each option. Option A: Influenza A is RNA. Option B: Variola is a DNA virus, so the answer is B. Wait, let me"
    assert run_split(key, text) == (text, "", None)     # an "answer is B" inside unfinished reasoning must NOT be scored
    assert run_split(key, "\n<think>Unfinished<|endoftext|>") == ("Unfinished", "", None)
    assert run_split(key, "") == ("", "", None)


@pytest.mark.parametrize("key", COT_KEYS)
def test_cot_split_and_parse(key):
    a = make(key)
    cot = ("To determine which is a DNA virus, let's analyse each option:\n\nA. Influenza A is a negative-sense RNA virus.\n"
           "B. Variola virus (smallpox) is a large double-stranded DNA virus.\nC. Poliovirus is a positive-sense RNA virus.\n"
           "D. HIV-1 is a retrovirus (RNA genome).\n\n")
    for tail, exp in [("Answer: B", 1), ("**Answer: B**", 1), ("**Answer:** B", 1), ("Answer: (B)", 1), ("Answer: **B**", 1), ("Answer: B.", 1),
                      ("Answer: Option B", 1), ("The answer is B.", 1), ("Final answer: C", 2), ("\\boxed{D}", 3)]:
        r, f, p = run_split(key, cot + tail)
        assert r == "" and f == (cot + tail).strip() and p == exp, (tail, p)
    # trailing EOS / spaced specials of the ChatGLM remote tokenizer
    assert run_split(key, "\n" + cot + "Answer: B <|endoftext|> <|endoftext|>")[2] == 1
    assert run_split(key, "\nB <|endoftext|> <|endoftext|>")[2] == 1                       # reply that is only the letter
    assert run_split(key, "(C)")[2] == 2 and run_split(key, "**D**")[2] == 3
    assert run_split(key, "Answer: A\nWait, that is wrong.\nAnswer: C")[2] == 2            # the last explicit answer wins
    assert a.parse_letter("Answer: C is wrong because ... Answer: D", 4) == 3
    # no letter / truncated before the final line / refusal / prose
    for text in ["", "   ", "I'm sorry, but I can't help with that.", "Let me think step by step about each option:\n\nA. Influenza A is",
                 "A. Influenza A is an RNA virus.\nB. Variola is a DNA virus.\n",   # a truncated option list must not parse as 'A'
                 "The answer is not clear to me.", "Answer: None of the above", "Answer: BRCA1", "Answer: A virus replicates in the nucleus.",
                 "Finish with a final line of the form 'Answer: <letter>'"]:
        assert run_split(key, text)[2] is None, text
    # out of range for the number of choices; 'I' is a letter only when it stands alone
    assert a.parse_letter("Answer: E", 4) is None and a.parse_letter("Answer: E", 5) == 4
    assert a.parse_letter("Answer: J", 10) == 9 and a.parse_letter("Answer: I", 10) == 8 and a.parse_letter("Answer: I think it is B", 10) is None
    assert a.parse_letter("正确答案是 D", 4) == 3 and a.parse_letter("答案：B", 4) == 1       # Chinese phrasing (language drift under steering)


@pytest.mark.parametrize("key", COT_KEYS)
def test_cot_reopened_think_block_is_reasoning(key):
    assert run_split(key, "<think>hmm, I should check the options") == ("hmm, I should check the options", "", None)
    assert run_split(key, "<think>x</think>Answer: C")[1:] == ("Answer: C", 2)


@pytest.mark.parametrize("key", DIRECT_KEYS)
def test_direct_split_and_parse(key):
    for text, exp in [("B", 1), ("\nB", 1), ("\n\nC<|endoftext|>", 2), ("D<|user|>", 3), ("(A)", 0), ("**B**", 1), ("B.", 1), ("A", 0),
                      ("Answer: B", 1), ("A virus", None), ("I cannot answer.", None), ("", None), ("Z", None)]:
        r, f, p = run_split(key, text)
        assert r == "" and p == exp, (text, p)
    assert make(key).parse_letter("I.", 10) == 8 and make(key).parse_letter("I'm sorry", 10) is None


def test_parse_is_total_on_garbage_and_linear_in_time():
    """parse_letter/split never raise and never backtrack badly (degenerate steered outputs can be huge runs of one character)."""
    junk = ["\x00", "</think></think>", "<think>", "<|begin_of_box|>", "答案", "Answer:", "\\boxed{", "((((", "**", "A" * 5000, "\n" * 100,
            "Answer: " + "B" * 100, "😀", "<|user|>" * 3]
    n = 30000
    stress = ["Answer:" + " " * n + "x", "answer " * (n // 7), "\\boxed{" + " " * n + "x", "Answer: " + "*" * n + "x", "Answer: " + "* " * (n // 2) + "x",
              "\n" * n + "Answer", "*" * n + "z", "correct option " + " " * n, "(" * n + "B", "Answer: " + "(**" * (n // 3) + "x"]
    t0 = time.time()
    for key in ALL_KEYS:
        a = make(key)
        for text in junk + stress:
            r, f = a.split(text)
            assert isinstance(r, str) and isinstance(f, str)
            assert a.parse_letter(f, 4) in (None, 0, 1, 2, 3)
    assert time.time() - t0 < 10.0


# ------------------------------------------------------------------------------------------------- loading helpers (offline)
def test_runtime_checks_give_clear_errors(monkeypatch):
    # The version is patched through the STRING target "transformers.__version__", which monkeypatch resolves against sys.modules at call
    # time, i.e. on the very module object that `import transformers` inside the adapter's _check_runtime returns. Patching this file's
    # import-time reference (`transformers`, bound at the top) is not enough: importing some model classes (e.g. Gemma3ForConditionalGeneration,
    # first needed by test_adapters_gemma_llama.py) replaces sys.modules["transformers"] with a NEW module object, after which the patch lands
    # on a stale one, _check_runtime sees the real version and load() fails later with an HFValidationError instead of the RuntimeError.
    monkeypatch.setattr("transformers.__version__", "5.1.0")
    with pytest.raises(RuntimeError, match="glm-4-9b-chat-hf"):
        make("glm-4-9b-chat").load(repo="does/not/matter")
    monkeypatch.setattr("transformers.__version__", "5.0.0rc3")
    with pytest.raises(RuntimeError, match="glm-4-9b-chat-hf"):                        # pre-releases of 5.0 count as 5.0
        make("glm-4-9b-chat")._check_runtime()
    monkeypatch.setattr("transformers.__version__", "4.40.0")
    for key, need in [("glm-4-9b-chat", "4.44.0"), ("glm-4-9b-chat-hf", "4.46.0"), ("glm-4.5-air-think", "4.54.0"), ("glm-4.7-flash", "5.0.0")]:
        with pytest.raises(RuntimeError, match=f">={re.escape(need)}"):
            make(key)._check_runtime()
    monkeypatch.undo()
    if HAS_LITE:
        make("glm-4.7-flash-think")._check_runtime()
    else:      # transformers 4.x cannot instantiate GLM-4.7-Flash: explicit error, not an obscure KeyError from AutoModel
        with pytest.raises(RuntimeError, match="glm4_moe_lite"):
            make("glm-4.7-flash-think")._check_runtime()
    if TF >= Version("5.0.0"):
        with pytest.raises(RuntimeError, match="glm-4-9b-chat-hf"):
            make("glm-4-9b-chat")._check_runtime()
    else:
        make("glm-4-9b-chat")._check_runtime()


def test_dtype_and_quant_kwargs():
    a = make("glm-4.5-air")
    assert a._dtype_kwarg("bf16", (4, 57, 6)) == {"dtype": torch.bfloat16}
    assert a._dtype_kwarg("auto", (4, 55, 0)) == {"torch_dtype": "auto"}
    assert a._dtype_kwarg(torch.float16, (5, 0, 0)) == {"dtype": torch.float16}
    with pytest.raises(ValueError):
        a._dtype_kwarg("int3", (5, 0, 0))
    assert a._quant_kwargs(None, (5, 0, 0)) == {}
    with pytest.raises(ValueError, match="unknown quant"):
        a._quant_kwargs("3bit", (4, 57, 6))
    for key in ("glm-4.5-air", "glm-4.7-flash-think"):          # fused experts (transformers>=5) cannot be quantised by bitsandbytes
        with pytest.raises(ValueError, match="fused"):
            make(key)._quant_kwargs("4bit", (5, 0, 0))
    import importlib.util
    if importlib.util.find_spec("bitsandbytes") is None:
        with pytest.raises(RuntimeError, match="bitsandbytes"):
            a._quant_kwargs("4bit", (4, 57, 6))
    else:
        q = a._quant_kwargs("4bit", (4, 57, 6))["quantization_config"]
        assert q.load_in_4bit and q.bnb_4bit_quant_type == "nf4" and q.bnb_4bit_use_double_quant
        assert make("glm-4-9b-chat-hf")._quant_kwargs("8bit", (5, 0, 0))["quantization_config"].load_in_8bit


# ------------------------------------------------------------------------------- layer path on meta-device (real configs)
def _meta_model(cfg, remote=False):
    from accelerate import init_empty_weights
    from transformers import AutoModelForCausalLM
    kw = {"trust_remote_code": True, "empty_init": False} if remote else {}
    with init_empty_weights():
        return AutoModelForCausalLM.from_config(cfg, **kw)


META = {  # key -> (repo, remote, class name, num layers, hidden, total params as loaded by HF, true KV bytes/token in bf16)
    "glm-4-9b-chat": ("THUDM/glm-4-9b-chat", True, "ChatGLMForConditionalGeneration", 40, 4096, 9_399_951_360, 2 * 40 * 2 * 128 * 2),
    "glm-4-9b-chat-hf": ("THUDM/glm-4-9b-chat-hf", False, "GlmForCausalLM", 40, 4096, 9_399_951_360, 2 * 40 * 2 * 128 * 2),
    "glm-4.5-air": ("zai-org/GLM-4.5-Air", False, "Glm4MoeForCausalLM", 46, 4096, 106_852_245_504, 2 * 46 * 8 * 128 * 2),
    "glm-4.7-flash": ("zai-org/GLM-4.7-Flash", False, "Glm4MoeLiteForCausalLM", 47, 2048, 29_943_390_976, 47 * 20 * (192 + 64 + 256) * 2),
}


def _has_path(model, path):
    obj = model
    for part in path.split("."):
        if not hasattr(obj, part):
            return False
        obj = getattr(obj, part)
    return True


@pytest.mark.parametrize("key", list(META))
def test_layer_path_and_param_counts_on_meta_device_with_real_configs(key):
    repo, remote, cls_name, nl, hidden, nparams, kv = META[key]
    if key == "glm-4.7-flash" and not HAS_LITE:
        pytest.skip(f"transformers {transformers.__version__} has no glm4_moe_lite (needs >=5.0.0); verified under 5.0.0 and 5.24.0, see card")
    if remote and TF >= Version("5.0.0"):
        pytest.skip("ChatGLM remote code cannot be instantiated on transformers>=5 (adapter raises a clear error)")
    cfg = get_cfg(repo, remote)
    model = _meta_model(cfg, remote)
    a = make(key)
    assert type(model).__name__ == cls_name and next(model.parameters()).device.type == "meta"
    assert len(a.layers(model)) == a.num_layers(model) == nl and a.hidden_size(model) == hidden
    assert sum(p.numel() for p in model.parameters()) == nparams
    assert a.kv_bytes_per_token(model) == kv
    other = ({"model.layers", "transformer.encoder.layers"} - {a.layer_path}).pop()
    assert _has_path(model, a.layer_path) and not _has_path(model, other)               # one path only: no ambiguity to guess from
    wrong = make("glm-4-9b-chat-hf" if remote else "glm-4-9b-chat")                      # wrong adapter -> explicit error, no fallback
    with pytest.raises(RuntimeError, match="refusing to guess"):
        wrong.layers(model)
    saved = (model.config.num_hidden_layers, getattr(model.config, "num_layers", None))   # config disagreeing with the module list
    try:
        model.config.num_hidden_layers = nl + 1
        if remote:
            model.config.num_layers = nl + 1
        with pytest.raises(RuntimeError, match="layers but the config says"):
            a.layers(model)
    finally:
        model.config.num_hidden_layers = saved[0]
        if remote:
            model.config.num_layers = saved[1]


# ----------------------------------------------------------------------------------- tiny real-architecture models (CPU)
def _init_norms(model):
    with torch.no_grad():
        for n, p in model.named_parameters():
            if n.endswith("layernorm.weight"):
                p.fill_(1.0)


def build_tiny(key):
    """Tiny random model (4 layers, hidden 64) of the real architecture class for `key`; float32 on CPU."""
    from transformers import AutoModelForCausalLM
    spec = G._SPEC[key]
    cfg = get_cfg(spec["repo"], spec["fam"] == "chatglm")
    small = dict(hidden_size=64, num_hidden_layers=4, num_attention_heads=4)
    if spec["fam"] == "chatglm":
        small.update(num_layers=4, ffn_hidden_size=128, multi_query_group_num=2, kv_channels=16, seq_length=512)
    elif spec["fam"] == "glm4hf":
        small.update(intermediate_size=128, num_key_value_heads=2, head_dim=16)
    elif spec["fam"] == "air":
        small.update(intermediate_size=128, moe_intermediate_size=32, num_key_value_heads=2, head_dim=16, n_routed_experts=8,
                     num_experts_per_tok=2, first_k_dense_replace=1)
    else:
        small.update(intermediate_size=128, moe_intermediate_size=32, num_key_value_heads=4, n_routed_experts=8, num_experts_per_tok=2,
                     q_lora_rank=32, kv_lora_rank=16, qk_nope_head_dim=12, qk_rope_head_dim=8, v_head_dim=20, qk_head_dim=20, head_dim=8,
                     mlp_layer_types=["dense", "sparse", "sparse", "sparse"])
    cfg = type(cfg).from_dict({**cfg.to_dict(), **small})
    if spec["fam"] == "glm4hf":
        make(key).fix_config(cfg)        # the Hub config says rope_theta=10000; load() corrects it, so the tiny model must be built the same way
    torch.manual_seed(0)
    if spec["fam"] == "chatglm":
        model = AutoModelForCausalLM.from_config(cfg, trust_remote_code=True, empty_init=False)
        _init_norms(model)
    else:
        model = AutoModelForCausalLM.from_config(cfg)
    return model.to(torch.float32).eval()


TINY_KEYS = ["glm-4-9b-chat", "glm-4-9b-chat-hf", "glm-4.5-air-think", "glm-4.7-flash-think"]


def needs(key):
    if key.startswith("glm-4.7") and not HAS_LITE:
        pytest.skip(f"glm4_moe_lite needs transformers>=5.0.0 (have {transformers.__version__})")
    if key == "glm-4-9b-chat" and TF >= Version("5.0.0"):
        pytest.skip("ChatGLM remote code is unusable on transformers>=5 (adapter raises; use glm-4-9b-chat-hf)")


@pytest.fixture(scope="module")
def tiny():
    cache = {}

    def get(key):
        needs(key)
        if key not in cache:
            model = build_tiny(key)
            a = make(key)
            a._post_load(model)                      # what load() does (greedy config; ChatGLM legacy-cache shim)
            cache[key] = (a, tok_for(key), model)
        return cache[key]
    return get


def _ctx(a, tok, model):
    return Ctx(a, tok, model, log=lambda *x, **k: None)


@pytest.mark.parametrize("key", TINY_KEYS)
def test_one_based_layer_convention_and_hook_point(tiny, key):
    a, tok, model = tiny(key)
    ctx, qs = _ctx(a, tok, model), toy_questions(3)
    nl, layers = a.num_layers(model), a.layers(model)
    assert ctx.nl == nl == 4 and ctx.H == 64
    assert a.check_convention(tok, model)                                  # executable form of the convention (used by load())
    enc = a.encode(tok, ctx.prompts(qs)).to(model.device)
    cap, hooks = {}, []
    for i, layer in enumerate(layers):
        hooks.append(layer.register_forward_hook(lambda m, inp, out, i=i: cap.__setitem__(i, out[0] if isinstance(out, tuple) else out)))
    with torch.no_grad():
        hs = model(**enc, output_hidden_states=True).hidden_states
    for h in hooks:
        h.remove()
    assert len(hs) == nl + 1
    for L in range(1, nl):                                                 # hidden_states[L] == output of layers[L-1]  (1-based)
        assert torch.equal(hs[L], cap[L - 1]), L
    if a.remote_code:                                                      # ChatGLM remote: the last entry is BEFORE the final norm
        assert torch.equal(hs[nl], cap[nl - 1])
    else:                                                                  # native HF: the last entry is AFTER the final norm
        assert not torch.equal(hs[nl], cap[nl - 1])
    H = ctx.hidden_last(ctx.prompts(qs))                                   # reads the last PROMPT token of every row (left padding)
    assert H.shape == (len(qs), nl + 1, 64) and H.dtype == np.float16
    np.testing.assert_allclose(H[:, 3], hs[3][:, -1].numpy().astype(np.float16), atol=1e-2, rtol=1e-2)


@pytest.mark.parametrize("key", TINY_KEYS)
def test_steering_hook_at_layer_L_acts_on_the_stream_hidden_states_L_reports(tiny, key):
    a, tok, model = tiny(key)
    ctx, qs = _ctx(a, tok, model), toy_questions(3)
    layers = a.layers(model)
    enc = a.encode(tok, ctx.prompts(qs)).to(model.device)
    v = torch.randn(64, generator=torch.Generator().manual_seed(1)) * 5

    def forward(specs):
        cap = {}
        with Hooks(ctx, specs):          # our capture hooks are registered AFTER the intervention hooks (SPEC lesson)
            hs = [layers[i].register_forward_hook(lambda m, inp, out, i=i: cap.__setitem__(i, out[0] if isinstance(out, tuple) else out))
                  for i in range(len(layers))]
            with torch.no_grad():
                logits = model(**enc).logits
            for h in hs:
                h.remove()
        return cap, logits

    base, base_logits = forward([])
    steered, st_logits = forward([(2, f_add(v))])
    assert torch.equal(base[0], steered[0])                                                       # before the hook point: untouched
    torch.testing.assert_close(steered[1] - base[1], v.expand_as(base[1]), atol=1e-4, rtol=1e-4)  # layers()[2-1] output + v (post-hook)
    assert not torch.allclose(steered[2], base[2]) and not torch.allclose(st_logits, base_logits)  # downstream changes


MLA_COMPRESSED_FROM = Version("5.15.0")     # transformers>=5.15 caches the compressed MLA latents (verified on 5.0.0 and 5.24.0 + GitHub tags)


def expected_kv_elems_per_token(model, key):
    """Cache elements per token (all layers) written from the architecture, independent of the adapter's code."""
    cfg = model.config
    nl = cfg.num_hidden_layers
    if key == "glm-4-9b-chat":
        return nl * 2 * cfg.multi_query_group_num * cfg.kv_channels
    if key.startswith("glm-4.7"):
        if TF >= MLA_COMPRESSED_FROM:
            return nl * (cfg.kv_lora_rank + cfg.qk_rope_head_dim)
        return nl * cfg.num_key_value_heads * (cfg.qk_nope_head_dim + cfg.qk_rope_head_dim + cfg.v_head_dim)
    return nl * 2 * cfg.num_key_value_heads * cfg.head_dim


@pytest.mark.parametrize("key", TINY_KEYS)
def test_kv_bytes_per_token_matches_the_real_cache(tiny, key):
    a, tok, model = tiny(key)
    enc = a.encode(tok, [a.build_chat(tok, "What is the capital of France?"), a.build_chat(tok, "Hi")]).to(model.device)
    with torch.no_grad():
        pkv = model(**enc, use_cache=True).past_key_values
    pairs = [(c.keys, c.values) for c in pkv.layers] if hasattr(pkv, "layers") else [(kv[0], kv[1]) for kv in pkv]
    assert len(pairs) == 4
    b, s = enc["input_ids"].shape
    measured = sum(k.numel() + v.numel() for k, v in pairs) / (b * s)                  # elements per token over all layers
    assert measured == expected_kv_elems_per_token(model, key)                          # the independent architecture formula is right
    assert a.kv_bytes_per_token(model) == measured * 4                                  # the adapter's probe (fp32 tiny model: 4 B/elem)


@pytest.mark.parametrize("key", ["glm-4-9b-chat", "glm-4-9b-chat-hf", "glm-4.5-air-think"])
def test_formula_fallback_equals_probe_for_gqa_families(tiny, key):
    a, _, model = tiny(key)
    assert a._formula_kv_bytes(model, 4) == a.kv_bytes_per_token(model)


@pytest.mark.parametrize("key", TINY_KEYS)
def test_ctx_run_end_to_end_is_deterministic_and_batch_invariant(tiny, key):
    a, tok, model = tiny(key)
    ctx, qs = _ctx(a, tok, model), toy_questions(4)
    r1 = ctx.run(qs, FRAMINGS["N"], None, max_new=4)
    r2 = ctx.run(qs, FRAMINGS["N"], None, max_new=4)
    assert r1.texts == r2.texts and r1.corr.shape == (4,) and r1.parsed.shape == (4,) and len(r1.reasoning) == 4
    # left-padded batch == one-by-one (position handling of the family is right)
    prompts = [ctx.prompt(q, FRAMINGS["E"]) for q in qs[:3]]
    prompts[1] = prompts[1] + " padding padding padding padding"
    assert ctx.gen(prompts, max_new=4) == [ctx.gen([p], max_new=4)[0] for p in prompts]
    v = torch.randn(64, generator=torch.Generator().manual_seed(1)) * 5            # steering via the adapter's layer path is deterministic
    assert ctx.gen(prompts, max_new=4, specs=[(2, f_add(v))]) == ctx.gen(prompts, max_new=4, specs=[(2, f_add(v))])


@pytest.mark.parametrize("key", TINY_KEYS)
def test_core_interventions_run_through_the_adapter(tiny, key):
    """core.neutral_stats / spec_add / spec_proj (add, proj0, one-sided cap) work on every GLM layout and really change generation."""
    from sbx.core import spec_add, spec_proj, unit
    a, tok, model = tiny(key)
    ctx, qs = _ctx(a, tok, model), toy_questions(4)
    rng = np.random.RandomState(0)
    d = {"rand": {L: unit(rng.randn(64)) for L in (1, 2, 3)}}
    U, med = ctx.neutral_stats(ctx.prompts(qs), d, [1, 2, 3], cap=2)
    assert set(U) == {1, 2, 3} and all(U[L] > 0 for L in U) and set(med["rand"]) == {1, 2, 3}
    base = ctx.run(qs, None, None, max_new=3)
    add = ctx.run(qs, None, spec_add(ctx, d["rand"], 2, 40.0, U), max_new=3)
    proj = ctx.run(qs, None, spec_proj(ctx, d["rand"], [1, 3], None), max_new=3)
    cap = ctx.run(qs, None, spec_proj(ctx, d["rand"], [1, 3], med["rand"], one_sided=True), max_new=3)
    assert all(len(r.texts) == 4 for r in (base, add, proj, cap))
    assert add.texts != base.texts                                                  # a 40-U push at layer 2 changes what is generated
    assert base.texts == ctx.run(qs, None, None, max_new=3).texts                   # hooks were removed again (no leakage)


@pytest.fixture(scope="module")
def tiny_dirs(tmp_path_factory, tiny):
    out = {}

    def get(key):
        needs(key)
        if key not in out:
            a, tok, model = tiny(key)
            d = tmp_path_factory.mktemp(key.replace(".", "_"))
            model.save_pretrained(d)
            tok.save_pretrained(d)
            out[key] = d
        return out[key]
    return get


@pytest.mark.parametrize("key", TINY_KEYS)
def test_load_from_local_dir_applies_family_fixes(tiny_dirs, key):
    d = tiny_dirs(key)
    a = make(key)
    tok, model = a.load(repo=str(d), device_map=None, dtype="fp32")                  # runs check_convention internally
    assert tok.padding_side == "left" and tok.pad_token is not None and not model.training
    gc = model.generation_config
    assert gc.do_sample is False and gc.temperature in (None, 1.0) and gc.top_p in (None, 1.0)
    assert len(a.layers(model)) == 4
    ctx = Ctx(a, tok, model, log=lambda *x, **k: None)
    assert len(ctx.run(toy_questions(3), FRAMINGS["E"], None, max_new=3).texts) == 3     # generate() works (ChatGLM: legacy-cache shim)
    if a.remote_code:
        assert type(model)._supports_default_dynamic_cache() is False
        assert a.kv_bytes_per_token(model) == 4 * 2 * 2 * 16 * 4          # 4 layers x (K,V) x 2 groups x 16 channels x 4 B (fp32)
    with pytest.raises(RuntimeError, match="refusing to guess"):                      # wrong layer path is an explicit error
        make("glm-4-9b-chat-hf" if a.remote_code else "glm-4-9b-chat").layers(model)


@pytest.mark.parametrize("key,remote", [("glm-4-9b-chat", True), ("glm-4-9b-chat-hf", False), ("glm-4.5-air-think", False), ("glm-4.7-flash", False)])
def test_load_plumbing_pins_revision_and_forwards_kwargs(monkeypatch, tiny, key, remote):
    """What load() hands to transformers for a Hub repo (stubbed: no network, no weights): pinned revision, trust_remote_code, dtype name."""
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
    tiny_key = "glm-4.5-air-think" if key == "glm-4.7-flash" else key         # any real tiny model will do; flash needs transformers>=5
    a_t, tok_t, model = tiny(tiny_key)
    a = make(key)
    hf = key == "glm-4-9b-chat-hf"                                            # the only key that pre-loads the config (RoPE base fix)
    monkeypatch.setattr(type(a), "_check_runtime", lambda self: (5, 0, 0) if not remote else (4, 57, 6))
    seen = {}

    class FakeTok:
        pad_token, eos_token, padding_side = None, "<|endoftext|>", "right"

    def fake_tok(cls, repo, **kw):
        seen["tok"] = (repo, kw)
        return FakeTok()

    def fake_model(cls, repo, **kw):
        seen["model"] = (repo, kw)
        return model

    def fake_cfg(cls, repo, **kw):
        assert hf, f"{key}: load() must not pre-load the config of a family without config fixes"
        seen["cfg"] = (repo, kw)
        cfg = copy.deepcopy(model.config)
        G.set_rope_theta(cfg, G.HUB_ROPE_THETA_9B_HF)                          # what the Hub config of THUDM/glm-4-9b-chat-hf says
        return cfg
    monkeypatch.setattr(AutoTokenizer, "from_pretrained", classmethod(fake_tok))
    monkeypatch.setattr(AutoModelForCausalLM, "from_pretrained", classmethod(fake_model))
    monkeypatch.setattr(AutoConfig, "from_pretrained", classmethod(fake_cfg))
    tok, m = a.load(dtype="bf16", check=False)
    repo, kw = seen["model"]
    assert repo == G._SPEC[key]["repo"] and kw["revision"] == G.REVISIONS[repo] and kw["trust_remote_code"] is remote
    assert kw["device_map"] == "auto" and ("dtype" in kw) != ("torch_dtype" in kw) and "quantization_config" not in kw
    assert seen["tok"][0] == repo and seen["tok"][1] == {"trust_remote_code": remote, "revision": G.REVISIONS[repo]}
    assert tok.padding_side == "left" and tok.pad_token == tok.eos_token and m is model
    if hf:                  # the pinned config is loaded first, corrected, and handed to from_pretrained (not the Hub's rope_theta=10000)
        assert seen["cfg"] == (repo, {"trust_remote_code": False, "revision": G.REVISIONS[repo]})
        assert G.get_rope_theta(kw["config"]) == G.Glm49ChatHF.rope_base == 5_000_000.0
    else:
        assert "config" not in kw and "cfg" not in seen
    a.load(revision="deadbeef", device_map=None, check=False, max_memory={0: "10GB"})
    repo, kw = seen["model"]
    assert kw["revision"] == "deadbeef" and "device_map" not in kw and kw["max_memory"] == {0: "10GB"}
    assert not hf or seen["cfg"][1]["revision"] == "deadbeef"
    mine = copy.deepcopy(model.config)
    a.load(config=mine, check=False)                                           # a caller-supplied config is used as is
    assert seen["model"][1]["config"] is mine
    monkeypatch.setattr(os.path, "isdir", lambda p: p == "/some/local/dir")
    seen.pop("cfg", None)
    a.load(repo="/some/local/dir", check=False)
    assert "revision" not in seen["model"][1] and "revision" not in seen["tok"][1]       # local dirs are never pinned
    assert "revision" not in seen.get("cfg", ("", {}))[1]


def test_check_convention_catches_a_wrong_hook_point(tiny):
    a, tok, model = tiny("glm-4-9b-chat-hf")

    class Shifted(type(a)):                      # an adapter that is off by one: layers()[L] instead of layers()[L-1]
        def layers(self, m):
            real = super().layers(m)
            return [real[-1]] + list(real[:-1])
    with pytest.raises(RuntimeError, match="convention does not hold"):
        Shifted("glm-4-9b-chat-hf").check_convention(tok, model)


def test_core_kv_formula_is_wrong_for_chatglm_and_mla_but_adapter_is_right(tiny):
    """Evidence for `needs_from_core`: core.kv_bytes_per_token (assumes 2 B/elem) vs the adapter's probe (fp32 tiny models: 4 B/elem)."""
    from sbx.core import kv_bytes_per_token
    a, _, model = tiny("glm-4-9b-chat-hf")
    assert kv_bytes_per_token(model) * 2 == a.kv_bytes_per_token(model)                 # GQA configs: core is exact
    if TF < Version("5.0.0"):
        a, _, model = tiny("glm-4-9b-chat")                                             # no num_key_value_heads in ChatGLMConfig
        assert kv_bytes_per_token(model) // 2 == 2 * 4 * 4 * 16 and a.kv_bytes_per_token(model) // 4 == 4 * 2 * 2 * 16   # elems: 4 heads vs 2 groups
    if HAS_LITE:
        a, _, model = tiny("glm-4.7-flash-think")
        core, real = kv_bytes_per_token(model) // 2, a.kv_bytes_per_token(model) // 4
        assert (core < real) if TF < MLA_COMPRESSED_FROM else (core > real)             # MLA: under- (<5.15) or over-estimate (>=5.15)


# ======================================================================================================================
# Regression tests for the adversarial review
#   1. glm-4-9b-chat-hf: the Hub config says rope_theta=10000, the original model uses 10000 * rope_ratio (500) = 5,000,000
#      (upstream convert_glm_weights_to_hf.py: `10000.0 * getattr(<dict>, "rope_ratio", 1)` is always 10000). Loaded unchanged the key
#      would measure a different function than glm-4-9b-chat.
#   2. Ctx.hidden_last / neutral_stats / check_convention are pure prefills but default to use_cache=True (a full KV cache, no OOM
#      fallback); _post_load turns that off without touching generate().
# ======================================================================================================================
HF_REPO, REMOTE_REPO = "THUDM/glm-4-9b-chat-hf", "THUDM/glm-4-9b-chat"


def hub_file(repo, name):
    """Path of one small file of the pinned revision of `repo` (local HF cache first, then the Hub); skips the test if unreachable."""
    from huggingface_hub import hf_hub_download
    last = None
    for local in (True, False):
        try:
            return Path(hf_hub_download(repo, name, revision=G.REVISIONS[repo], local_files_only=local))
        except Exception as e:  # noqa: BLE001
            last = e
    pytest.skip(f"{repo}/{name}@{G.REVISIONS[repo][:8]} is neither in the local HF cache nor reachable ({type(last).__name__}: {str(last)[:80]})")


def hub_json(repo, name="config.json"):
    return json.loads(hub_file(repo, name).read_text(encoding="utf-8"))


def small_native_cfg(head_dim=16):
    """A small GlmConfig built from the REAL Hub config of the -hf repo (so it carries the published rope_theta=10000)."""
    base = get_cfg(HF_REPO, False)
    return type(base).from_dict({**base.to_dict(), "hidden_size": 2 * head_dim, "intermediate_size": 64, "num_hidden_layers": 2,
                                 "num_attention_heads": 2, "num_key_value_heads": 1, "head_dim": head_dim, "vocab_size": 512,
                                 "pad_token_id": 0})        # (the Hub's pad id 151329 does not fit a 512-token vocabulary)


def test_rope_base_of_the_hf_key_is_the_original_models_and_the_hub_config_is_wrong():
    orig, hub = hub_json(REMOTE_REPO), hub_json(HF_REPO)
    code = hub_file(REMOTE_REPO, "modeling_chatglm.py").read_text(encoding="utf-8")
    assert "base = base * self.rope_ratio" in code and "base: int = 10000" in code        # remote RotaryEmbedding: base = 10000 * rope_ratio
    assert orig["rope_ratio"] == G.ROPE_RATIO_9B_CHAT == 500
    assert G.Glm49ChatHF.rope_base == G.ROPE_BASE_9B_CHAT == 10000.0 * orig["rope_ratio"] == 5_000_000.0
    # the converted Hub config forgot the ratio (the upstream script multiplies by getattr(<dict>, "rope_ratio", 1), which is always 1) ...
    assert hub["rope_theta"] == G.Glm49ChatHF.hub_rope_theta == G.HUB_ROPE_THETA_9B_HF == 10000.0 != G.Glm49ChatHF.rope_base
    # ... and it is the ONLY mapped field that disagrees with the original config
    for new, old in {"vocab_size": "padded_vocab_size", "intermediate_size": "ffn_hidden_size", "num_hidden_layers": "num_layers",
                     "max_position_embeddings": "seq_length", "rms_norm_eps": "layernorm_epsilon", "head_dim": "kv_channels",
                     "attention_bias": "add_qkv_bias", "num_key_value_heads": "multi_query_group_num", "hidden_size": "hidden_size",
                     "num_attention_heads": "num_attention_heads", "eos_token_id": "eos_token_id", "pad_token_id": "pad_token_id"}.items():
        assert hub[new] == orig[old], new


# the upstream conversion (convert_glm_weights_to_hf.py, transformers v4.46.3 and main): key renames + a split of the fused qkv, NO row permutation
_UPSTREAM_KEYMAP = [
    (r"transformer.output_layer.weight", r"lm_head.weight"),
    (r"transformer.embedding.word_embeddings.weight", r"model.embed_tokens.weight"),
    (r"transformer.rotary_pos_emb.inv_freq", None),
    (r"transformer.encoder.final_layernorm.weight", r"model.norm.weight"),
    (r"transformer.encoder.layers.(\d+).input_layernorm.weight", r"model.layers.\1.input_layernorm.weight"),
    (r"transformer.encoder.layers.(\d+).post_attention_layernorm.weight", r"model.layers.\1.post_attention_layernorm.weight"),
    (r"transformer.encoder.layers.(\d+).self_attention.dense.weight", r"model.layers.\1.self_attn.o_proj.weight"),
    (r"transformer.encoder.layers.(\d+).self_attention.query_key_value.(weight|bias)", r"model.layers.\1.self_attn.qkv_proj.\2"),
    (r"transformer.encoder.layers.(\d+).mlp.dense_h_to_4h.weight", r"model.layers.\1.mlp.gate_up_proj.weight"),
    (r"transformer.encoder.layers.(\d+).mlp.dense_4h_to_h.weight", r"model.layers.\1.mlp.down_proj.weight"),
]


def upstream_convert_state_dict(sd, n_heads, n_kv, head_dim):
    out = {}
    for old, value in sd.items():
        new, dropped = None, False
        for pat, rep in _UPSTREAM_KEYMAP:
            if rep is None:
                if re.fullmatch(pat, old):
                    dropped = True
                    break
            else:
                new, n = re.subn(pat, rep, old)
                if n:
                    break
                new = None
        if dropped:
            continue
        assert new is not None, f"unmapped key {old}"
        if "qkv_proj." in new:
            q, kv = n_heads * head_dim, n_kv * head_dim
            out[new.replace("qkv_proj.", "q_proj.")], out[new.replace("qkv_proj.", "k_proj.")] = value[:q], value[q:q + kv]
            out[new.replace("qkv_proj.", "v_proj.")] = value[q + kv:]
        else:
            out[new] = value
    return out


@pytest.mark.skipif(TF >= Version("5.0.0"), reason="the ChatGLM remote code cannot be built on transformers>=5 (this check needs the original model)")
def test_hf_variant_equals_the_remote_code_variant_only_at_the_adapters_rope_base():
    """The review's blocker as an experiment: a tiny random remote ChatGLM (REAL config incl. rope_ratio=500; real head_dim=128 so all 32 rotary
    frequencies exist) against the native GlmForCausalLM holding the same tensors (upstream key mapping). Identical logits at the adapter's
    `rope_base`; clearly different logits at the Hub config's rope_theta=10000. Real 9B weights are not available here."""
    from transformers import AutoModelForCausalLM, GlmConfig, GlmForCausalLM
    n_heads, n_kv, hd, vocab = 4, 2, 128, 1024
    rcfg = get_cfg(REMOTE_REPO, True)
    assert rcfg.rope_ratio == G.ROPE_RATIO_9B_CHAT
    rcfg = type(rcfg).from_dict({**rcfg.to_dict(), "hidden_size": n_heads * hd, "num_hidden_layers": 2, "num_layers": 2,
                                 "num_attention_heads": n_heads, "ffn_hidden_size": 256, "multi_query_group_num": n_kv, "kv_channels": hd,
                                 "seq_length": 2048, "padded_vocab_size": vocab})
    torch.manual_seed(0)
    remote = AutoModelForCausalLM.from_config(rcfg, trust_remote_code=True, empty_init=False).to(torch.float32).eval()
    _init_norms(remote)
    sd = upstream_convert_state_dict(remote.state_dict(), n_heads, n_kv, hd)
    ids = torch.randint(10, vocab, (2, 300), generator=torch.Generator().manual_seed(1))        # positions far apart: RoPE matters
    with torch.no_grad():
        ref = remote(input_ids=ids).logits

    def native_logits(theta):
        cfg = GlmConfig(vocab_size=vocab, hidden_size=n_heads * hd, intermediate_size=256, num_hidden_layers=2, num_attention_heads=n_heads,
                        num_key_value_heads=n_kv, head_dim=hd, max_position_embeddings=2048, rms_norm_eps=rcfg.layernorm_epsilon,
                        attention_bias=True, pad_token_id=0)             # (pad id only matters at init: the state dict sets every row)
        G.set_rope_theta(cfg, theta)
        native = GlmForCausalLM(cfg).to(torch.float32).eval()
        missing, unexpected = native.load_state_dict(sd, strict=False)
        assert not [m for m in missing if "rotary" not in m] and not unexpected, (missing, unexpected)     # the mapping is complete
        assert G.rotary_inv_freq_error(native, theta) < 1e-6
        with torch.no_grad():
            return native(input_ids=ids).logits

    scale = ref.abs().max().item()
    right = (native_logits(G.Glm49ChatHF.rope_base) - ref).abs().max().item()
    wrong = (native_logits(G.Glm49ChatHF.hub_rope_theta) - ref).abs().max().item()
    assert right < 1e-5 * scale, (right, scale)                          # same function (observed: exactly 0 at every seed tried)
    assert wrong > 1e-3 * scale, (wrong, scale)                          # the Hub config's base computes a different function (observed ~1.5e-2 x scale)


def test_get_and_set_rope_theta_cover_both_transformers_layouts():
    v4 = types.SimpleNamespace(rope_theta=10000.0)                                                       # transformers 4.x GlmConfig
    v5 = types.SimpleNamespace(rope_parameters={"rope_theta": 10000.0, "partial_rotary_factor": 0.5, "rope_type": "default"})   # >=5
    assert G.get_rope_theta(v4) == G.get_rope_theta(v5) == 10000.0
    G.set_rope_theta(v4, 5e6)
    G.set_rope_theta(v5, 5e6)
    assert v4.rope_theta == 5e6 and G.get_rope_theta(v4) == 5e6
    assert v5.rope_parameters == {"rope_theta": 5e6, "partial_rotary_factor": 0.5, "rope_type": "default"} and G.get_rope_theta(v5) == 5e6
    assert not hasattr(v5, "rope_theta")           # never a plain attribute on the >=5 layout: transformers>=5 silently ignores it
    with pytest.raises(RuntimeError, match="neither"):
        G.get_rope_theta(types.SimpleNamespace())
    with pytest.raises(RuntimeError, match="no 'rope_theta'"):
        G.get_rope_theta(types.SimpleNamespace(rope_parameters={"rope_type": "default"}))
    cfg = small_native_cfg()                         # the installed transformers uses exactly one of the two layouts
    assert (getattr(cfg, "rope_parameters", None) is not None) == (TF >= Version("5.0.0")) and G.get_rope_theta(cfg) == 10000.0


@pytest.mark.parametrize("theta", [10000.0, 5_000_000.0])
def test_set_rope_theta_reaches_the_rotary_buffer_of_the_native_model(theta):
    """On every transformers version the helper's value is what the native model computes with (on >=5 a plain rope_theta attribute/kwarg is not)."""
    from transformers import AutoModelForCausalLM
    cfg = small_native_cfg(head_dim=128)             # real rotary layout: 32 frequencies
    G.set_rope_theta(cfg, theta)
    model = AutoModelForCausalLM.from_config(cfg).eval()
    assert model.model.rotary_emb.inv_freq.numel() == 32
    assert G.rotary_inv_freq_error(model, theta) < 1e-6
    assert G.rotary_inv_freq_error(model, 5e6 if theta == 1e4 else 1e4) > 10 * G.ROPE_CHECK_RTOL    # the other base is far outside the tolerance


def test_fix_config_corrects_only_the_known_hub_error(caplog):
    a = make("glm-4-9b-chat-hf")
    cfg = small_native_cfg()
    assert G.get_rope_theta(cfg) == 10000.0          # as published on the Hub
    with caplog.at_level(logging.WARNING, logger=G.log.name):
        assert a.fix_config(cfg) is True
    assert G.get_rope_theta(cfg) == 5_000_000.0 and "conversion error" in caplog.text and "rope_ratio 500" in caplog.text
    caplog.clear()
    with caplog.at_level(logging.WARNING, logger=G.log.name):
        assert a.fix_config(cfg) is False            # a corrected copy is left alone, silently
    assert G.get_rope_theta(cfg) == 5_000_000.0 and not caplog.records
    for other in (1e8, 500000.0, 1.0):               # any other value is another model: refuse, never overwrite
        G.set_rope_theta(cfg, other)
        with pytest.raises(ValueError, match="Refusing to guess"):
            a.fix_config(cfg)
        assert G.get_rope_theta(cfg) == other


def test_verify_model_accepts_the_corrected_model_and_refuses_the_hub_base(tiny, caplog):
    from transformers import AutoModelForCausalLM
    a, _, model = tiny("glm-4-9b-chat-hf")           # built the way load() leaves it
    a._verify_model(model)
    assert G.rotary_inv_freq_error(model, 5e6) < 1e-6
    hub = AutoModelForCausalLM.from_config(small_native_cfg()).eval()        # what a plain from_pretrained of the Hub repo gives
    assert G.rotary_inv_freq_error(hub, 5e6) > 1.0 and G.rotary_inv_freq_error(hub, 1e4) < 1e-6
    with pytest.raises(RuntimeError, match="does not belong to RoPE base"):
        a._verify_model(hub)
    for dt in (torch.bfloat16, torch.float16):       # a half-precision buffer is rounding, not a wrong base
        half = copy.deepcopy(model)
        half.model.rotary_emb.to(dt)
        assert half.model.rotary_emb.inv_freq.dtype == dt
        a._verify_model(half)
    with torch.device("meta"):
        meta = AutoModelForCausalLM.from_config(small_native_cfg())
    assert G.rotary_inv_freq_error(meta, 5e6) is None
    with caplog.at_level(logging.WARNING, logger=G.log.name):
        a._verify_model(meta)                        # cannot be checked: say so, do not crash
    assert "meta device" in caplog.text
    with pytest.raises(RuntimeError, match="cannot find"):
        G.rotary_inv_freq_error(types.SimpleNamespace(), 5e6)
    other = make("glm-4.5-air")
    other._verify_model(hub)                         # the other families have nothing to verify


def test_load_corrects_a_hub_like_checkpoint_and_equals_the_corrected_model(tiny, tiny_dirs, tmp_path, caplog):
    """End to end through load(): same tensors, the Hub's mistaken config -> the loaded model computes what the corrected model computes."""
    from transformers import AutoConfig, AutoModelForCausalLM
    key = "glm-4-9b-chat-hf"
    a, _, ref = tiny(key)
    good = tiny_dirs(key)                            # checkpoint of the corrected tiny model (config says 5e6), tokenizer saved
    hub = tmp_path / "hub_like"
    shutil.copytree(good, hub)
    cfg = AutoConfig.from_pretrained(hub)
    assert G.get_rope_theta(cfg) == 5e6
    G.set_rope_theta(cfg, 10000.0)                   # the same tensors with the config as published on the Hub
    cfg.save_pretrained(hub)
    ids = torch.randint(0, 5000, (2, 24), generator=torch.Generator().manual_seed(3))

    def logits(m):
        with torch.no_grad():
            return m(input_ids=ids).logits
    naive = AutoModelForCausalLM.from_pretrained(hub).eval()                  # the bug: a plain load of a Hub-like checkpoint
    assert G.rotary_inv_freq_error(naive, 5e6) > 1.0 and not torch.equal(logits(naive), logits(ref))
    with caplog.at_level(logging.WARNING, logger=G.log.name):
        _, fixed = a.load(repo=str(hub), device_map=None, dtype="fp32")      # also runs check_convention
    assert "conversion error" in caplog.text
    assert G.get_rope_theta(fixed.config) == 5e6 and G.rotary_inv_freq_error(fixed, 5e6) < 1e-6
    torch.testing.assert_close(logits(fixed), logits(ref), atol=1e-6, rtol=0)
    caplog.clear()
    with caplog.at_level(logging.WARNING, logger=G.log.name):
        _, again = a.load(repo=str(good), device_map=None, dtype="fp32")     # an already corrected copy loads silently
    assert "conversion error" not in caplog.text
    torch.testing.assert_close(logits(again), logits(ref), atol=1e-6, rtol=0)
    with pytest.raises(RuntimeError, match="does not belong to RoPE base"):  # a caller-supplied config is verified, not trusted
        a.load(repo=str(hub), device_map=None, dtype="fp32", config=AutoConfig.from_pretrained(hub), check=False)
    other = tmp_path / "other_base"
    shutil.copytree(good, other)
    cfg = AutoConfig.from_pretrained(other)
    G.set_rope_theta(cfg, 1e8)                       # e.g. glm-4-9b-chat-1m's base: another model, refused before the weights are read
    cfg.save_pretrained(other)
    with pytest.raises(ValueError, match="Refusing to guess"):
        a.load(repo=str(other), device_map=None, dtype="fp32")


# ------------------------------------------------------------------------------------------- pure prefills allocate no KV cache
def _has_cache(out):
    """Does a model output carry a KV cache? transformers 5.24 names the field `cache` (`past_key_values` is a deprecated alias there)."""
    return any(getattr(out, name, None) is not None for name in ("cache", "past_key_values"))


def _returned_cache(model, fn):
    """(per top-level forward call during fn(): did the model return a KV cache?, fn's result)."""
    seen = []
    h = model.register_forward_hook(lambda mod, inp, out: seen.append(_has_cache(out)))
    try:
        res = fn()
    finally:
        h.remove()
    return seen, res


@pytest.mark.parametrize("key", TINY_KEYS)
def test_post_load_removes_the_kv_cache_of_pure_prefill_forwards_and_leaves_generate_alone(tiny, key):
    a, tok, model = tiny(key)                        # the fixture ran _post_load, as load() does
    cfgs = [model.config] + ([model.config.text_config] if getattr(model.config, "text_config", None) is not None else [])
    assert all(c.use_cache is False for c in cfgs) and model.generation_config.use_cache is True
    ref = copy.deepcopy(model)
    ref.config.use_cache = True                      # the model as HF delivers it: plain forwards cache by default
    ctx, ctx_ref = _ctx(a, tok, model), _ctx(a, tok, ref)
    prompts = ctx.prompts(toy_questions(4))
    prompts[1] += " padding padding padding padding"
    dirs = {"d": {1: np.eye(64, dtype=np.float32)[0]}}
    prefills = {"hidden_last": lambda c: c.hidden_last(prompts), "neutral_stats": lambda c: c.neutral_stats(prompts, dirs, [1], cap=2),
                "check_convention": lambda c: a.check_convention(tok, c.model)}
    for name, fn in prefills.items():
        seen, _ = _returned_cache(model, lambda: fn(ctx))
        assert seen and not any(seen), f"{name}: a pure-prefill forward still builds a KV cache"
        seen_ref, _ = _returned_cache(ref, lambda: fn(ctx_ref))
        assert seen_ref and all(seen_ref), f"{name}: control failed, the unfixed model should build one"
    seen, texts = _returned_cache(model, lambda: ctx.gen(prompts, max_new=5))
    assert seen and all(seen)                        # generate() still runs with its cache ...
    assert texts == ctx_ref.gen(prompts, max_new=5)  # ... and generates exactly what it generated before
    np.testing.assert_array_equal(ctx.hidden_last(prompts), ctx_ref.hidden_last(prompts))


def test_post_load_pins_the_generation_cache_on_and_handles_a_text_config():
    a = make("glm-4-9b-chat-hf")
    fake = types.SimpleNamespace(config=types.SimpleNamespace(use_cache=True, text_config=types.SimpleNamespace(use_cache=True)),
                                 generation_config=types.SimpleNamespace(do_sample=True, temperature=0.8, top_p=0.8, top_k=5, use_cache=False))
    a._post_load(fake)
    assert fake.config.use_cache is False and fake.config.text_config.use_cache is False
    gc = fake.generation_config
    assert gc.use_cache is True and gc.do_sample is False and (gc.temperature, gc.top_p, gc.top_k) == (None, None, None)


# ---------------------------------------------------------------------------------------------------------- cards
def _card(key):
    return json.loads((MODELS_DIR / f"{key}.json").read_text(encoding="utf-8"))


def test_hf_card_documents_the_rope_correction_and_has_no_stale_equivalence_claims():
    card, a = _card("glm-4-9b-chat-hf"), make("glm-4-9b-chat-hf")
    assert card["rope_theta_hub_config"] == a.hub_rope_theta == 10000.0 and card["rope_theta_effective"] == a.rope_base == 5_000_000.0
    assert card["rope_ratio_original"] == G.ROPE_RATIO_9B_CHAT == _card("glm-4-9b-chat")["config"]["rope_ratio"]
    assert card["rope_note"] and any("rope_theta" in x for x in card["known_issues"])
    text = json.dumps(card, ensure_ascii=False)
    for stale in ("by construction", "only permutes", "only re-permutes", "safe choice for pods"):
        assert stale not in text, f"stale claim in the card: {stale!r}"
    assert "rope" in json.dumps(_card("glm-4-9b-chat")["known_issues"]).lower()        # the remote card points at the correction too


@pytest.mark.parametrize("key", ALL_KEYS)
def test_cards_document_the_prefill_cache_policy_and_the_scope_of_the_oom_fallback(key):
    issues = _card(key)["known_issues"]
    assert any("use_cache=False" in x for x in issues), "known_issues must mention config.use_cache=False (pure-prefill forwards)"
    if key.startswith("glm-4.7"):
        kv = next(x for x in issues if x.startswith("KV cache size depends"))
        assert "_gen_chunk" in kv and "hidden_last" in kv and "OOM on a single prompt" in kv and "5.15" in kv and "needs_from_core" in kv
        assert "nothing breaks" not in " ".join(issues).lower()
