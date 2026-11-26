"""Tests for the "misc" adapters: Mistral (2501 / 3.2 / Ministral-3), Phi-4 family, OLMo-3 family, Seed-OSS.

Offline by construction: tokenizers/configs are read from the local HF cache (snapshot_download(local_files_only=True)); a test
skips with a clear reason when a repo is not cached. No weights are ever loaded: architectures are checked on meta-device models
built from the real configs and on tiny random models of the same architecture classes (shrunk dims, real tokenizers).
Run:  USE_TF=0 USE_TORCH=1 python3 -m pytest sbx/tests/test_adapters_misc.py -q
"""
from __future__ import annotations

import datetime
import importlib
import json
import pkgutil
import re
from functools import lru_cache, reduce
from pathlib import Path

import numpy as np
import pytest

import sbx.adapters as adapters_pkg
from sbx.adapters import get_adapter, list_keys
from sbx.adapters.mistral import (SYSTEM_PROMPT_32, TEKKEN_PATTERN, _pre_tokenizer_regex, load_mistral_tokenizer, parse_mcq_letter,
                                  render_v7, split_think, template_default_system)
from sbx.adapters.olmo import BOILERPLATE
from sbx.adapters.seed_oss import parse_budget
from sbx.framings import FRAMINGS

MODELS_DIR = Path(__file__).resolve().parents[1] / "models"
E_TEXT = FRAMINGS["E"].sys_text
OLMO_KEYS = ["olmo-3.1-32b-instruct", "olmo-3-7b-instruct", "olmo-3.1-32b-think", "olmo-3-32b-think", "olmo-3-7b-think"]
Q = {"source": "t", "qid": "q1", "question": "Which organelle makes most of the cell's ATP?",
     "choices": ["Nucleus", "Mitochondrion", "Ribosome", "Golgi apparatus"], "answer": 1}

# key -> dict(tok: repo whose tokenizer is used, cfg: repo whose config.json is used, mistral: tekken loader, last: expected last
# prompt token, bos: "single" | "none" | "per_turn")
SPEC = {
    "mistral-small-24b": dict(tok="mistralai/Mistral-Small-24B-Instruct-2501", cfg="mistralai/Mistral-Small-24B-Instruct-2501",
                              mistral=True, last="[/INST]", bos="single"),
    "mistral-small-3.2-24b": dict(tok="mistralai/Mistral-Small-3.1-24B-Instruct-2503",
                                  cfg="mistralai/Mistral-Small-3.2-24B-Instruct-2506", mistral=True, last="[/INST]", bos="single"),
    "ministral-3-14b": dict(tok="mistralai/Ministral-3-14B-Instruct-2512-BF16", cfg="mistralai/Ministral-3-14B-Instruct-2512-BF16",
                            mistral=True, last="[/INST]", bos="single"),
    "magistral-small-2509": dict(tok="mistralai/Ministral-3-14B-Reasoning-2512", cfg="mistralai/Magistral-Small-2509", mistral=True,
                                 last="[/INST]", bos="single"),
    "ministral-3-14b-reasoning": dict(tok="mistralai/Ministral-3-14B-Reasoning-2512", cfg="mistralai/Ministral-3-14B-Reasoning-2512",
                                      mistral=True, last="[/INST]", bos="single"),
    "phi-4": dict(tok="microsoft/phi-4", cfg="microsoft/phi-4", last="<|im_sep|>", bos="none"),
    "phi-4-mini-instruct": dict(tok="microsoft/Phi-4-mini-instruct", cfg="microsoft/Phi-4-mini-instruct", last="<|assistant|>",
                                bos="none"),
    "phi-4-reasoning": dict(tok="microsoft/Phi-4-reasoning", cfg="microsoft/Phi-4-reasoning", last="<|im_sep|>", bos="none"),
    "phi-4-reasoning-plus": dict(tok="microsoft/Phi-4-reasoning-plus", cfg="microsoft/Phi-4-reasoning-plus", last="<|im_sep|>",
                                 bos="none"),
    "olmo-3.1-32b-instruct": dict(tok="allenai/Olmo-3.1-32B-Instruct", cfg="allenai/Olmo-3.1-32B-Instruct", last="Ċ", bos="none"),
    "olmo-3-7b-instruct": dict(tok="allenai/Olmo-3-7B-Instruct", cfg="allenai/Olmo-3-7B-Instruct", last="Ċ", bos="none"),
    "olmo-3.1-32b-think": dict(tok="allenai/Olmo-3.1-32B-Think", cfg="allenai/Olmo-3.1-32B-Think", last=">", bos="none"),
    "olmo-3-32b-think": dict(tok="allenai/Olmo-3-32B-Think", cfg="allenai/Olmo-3-32B-Think", last=">", bos="none"),
    "olmo-3-7b-think": dict(tok="allenai/Olmo-3-7B-Think", cfg="allenai/Olmo-3-7B-Think", last=">", bos="none"),
    "seed-oss-36b-instruct": dict(tok="ByteDance-Seed/Seed-OSS-36B-Instruct", cfg="ByteDance-Seed/Seed-OSS-36B-Instruct",
                                  last="Ċ", bos="per_turn"),
    "seed-oss-36b-instruct-nothink": dict(tok="ByteDance-Seed/Seed-OSS-36B-Instruct", cfg="ByteDance-Seed/Seed-OSS-36B-Instruct",
                                          last="</seed:think>", bos="per_turn"),
}
KEYS = list(SPEC)


# ------------------------------------------------------------------------------------------------ cache helpers
def _snapshot(repo):
    from huggingface_hub import snapshot_download
    try:
        return snapshot_download(repo, local_files_only=True)
    except Exception as e:                                      # noqa: BLE001 - any cache miss means "cannot test offline"
        pytest.skip(f"{repo} is not in the local HF cache ({type(e).__name__}); fetch its tokenizer/config files to run this test")


@lru_cache(maxsize=None)
def _tok(key):
    from transformers import AutoTokenizer
    spec = SPEC[key]
    path = _snapshot(spec["tok"])
    try:
        if spec.get("mistral"):
            return load_mistral_tokenizer(path)
        tok = AutoTokenizer.from_pretrained(path)
    except Exception as e:                                      # noqa: BLE001
        pytest.skip(f"tokenizer files for {spec['tok']} unusable offline ({type(e).__name__}: {str(e)[:120]})")
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def _cfg(key):
    from transformers import AutoConfig
    path = _snapshot(SPEC[key]["cfg"])
    try:
        return AutoConfig.from_pretrained(path)
    except Exception as e:                                      # noqa: BLE001
        pytest.skip(f"config of {SPEC[key]['cfg']} not loadable with this transformers ({type(e).__name__}: {str(e)[:100]})")


def _ids(tok, text):
    return tok(text, add_special_tokens=False)["input_ids"]


def _native(tok, msgs, **kw):
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, **kw)


def _template_31():
    """The V7 chat template of the Mistral-Small-3.1 repo. It is stored as chat_template.json (read by the *processor*), so
    AutoTokenizer.from_pretrained(...).chat_template is None for that repo."""
    p = Path(_snapshot("mistralai/Mistral-Small-3.1-24B-Instruct-2503")) / "chat_template.json"
    if not p.exists():
        pytest.skip("chat_template.json of Mistral-Small-3.1 not cached")
    return json.loads(p.read_text())["chat_template"]


def _u(ad, text=None):
    return {"role": "user", "content": text or ad.user_text(Q)}


def _pin_date(text, date):
    """Rewrite the one 'The current date is YYYY-MM-DD' sentence of a natively rendered Mistral-Small-2501 prompt (no callable
    template kwarg: transformers 5.24.0 turns it into a string, see test_mistral_small_pins_the_date_*)."""
    out, n = re.subn(r"(The current date is )\d{4}-\d{2}-\d{2}", r"\g<1>" + date, text)
    assert n == 1, text[:200]
    return out


def _sanitising(real):
    """Emulate transformers 5.24.0's chat-template kwarg sanitising (chat_template_utils._plain_template_data, applied to the kwargs in
    render_jinja_template): everything that is not plain JSON data becomes the placeholder string '<typename>' (a lambda -> '<function>')."""
    plain = (type(None), str, int, float, bool, list, tuple, dict)

    def apply_chat_template(conversation, *args, **kwargs):
        kwargs = {k: v if isinstance(v, plain) else f"<{type(v).__name__}>" for k, v in kwargs.items()}
        return real(conversation, *args, **kwargs)
    return apply_chat_template


def _diff_window(n_ids, f_ids):
    """(p, n_end, f_end): N and the framed prompt share the first p ids and everything after n_end / f_end, so ALL their differences lie
    inside n_ids[p:n_end] (N side) vs f_ids[p:f_end] (framed side). Longest common prefix, then longest common non-overlapping suffix."""
    m = min(len(n_ids), len(f_ids))
    p = 0
    while p < m and n_ids[p] == f_ids[p]:
        p += 1
    s = 0
    while s < m - p and n_ids[-1 - s] == f_ids[-1 - s]:
        s += 1
    return p, len(n_ids) - s, len(f_ids) - s


def _skip_if_transformers_too_old(key):
    ad = get_adapter(key)
    if getattr(ad, "min_transformers", (0,))[0] >= 5:
        import transformers
        if int(transformers.__version__.split(".")[0]) < 5:
            pytest.skip("model_type ministral3 needs transformers>=5.0 (verified separately with 5.0.0, see model card)")


# ------------------------------------------------------------------------------------------------ registry / structure
def test_registry_keys_unique_and_discovered():
    keys = list_keys()
    for k in KEYS:
        assert k in keys, k
        assert k == k.lower()
    defs = {}
    for m in pkgutil.iter_modules(adapters_pkg.__path__):
        if m.name == "base":
            continue
        mod = importlib.import_module(f"sbx.adapters.{m.name}")
        for obj in vars(mod).values():
            if isinstance(obj, type) and issubclass(obj, adapters_pkg.Adapter) and obj is not adapters_pkg.Adapter and obj.keys:
                for k in obj.keys:
                    defs.setdefault(k.lower(), set()).add(obj)
    for k in KEYS:
        assert len(defs[k]) == 1, f"key {k!r} is defined by several classes: {defs[k]}"
        assert get_adapter(k).__class__ in defs[k]
        assert next(iter(defs[k])).__module__.rsplit(".", 1)[1] in ("mistral", "phi", "olmo", "seed_oss")


@pytest.mark.parametrize("key", KEYS)
def test_adapter_attributes(key):
    ad = get_adapter(key)
    assert ad.key == key and ad.repo == ad.repos[key]
    assert ad.answer_mode in ("direct", "cot", "reasoning")
    assert isinstance(ad.max_new, int) and ad.max_new > 0
    assert ad.sys_role in ("system", None)
    if ad.answer_mode == "reasoning":
        assert ad.max_new >= 2048, "reasoning models need a long generation budget"


# ------------------------------------------------------------------------------------------------ render against real templates
@pytest.mark.parametrize("key", KEYS)
def test_render_n_e_chat_structure(key):
    tok, ad, spec = _tok(key), get_adapter(key), SPEC[key]
    n, e = ad.build(tok, Q), ad.build(tok, Q, E_TEXT)
    assert ad.build(tok, Q) == n and ad.build(tok, Q, E_TEXT) == e, "rendering must be deterministic (no wall-clock dates)"
    assert ad.user_text(Q) in n and ad.user_text(Q) in e
    assert E_TEXT in e and E_TEXT not in n
    assert e.count(E_TEXT) == 1
    # the prompt ends exactly where the assistant starts
    for p in (n, e):
        assert p.endswith(ad.gen_prompt_suffix), (key, p[-60:])
    assert tok.convert_ids_to_tokens(_ids(tok, n))[-1] == spec["last"]
    assert tok.convert_ids_to_tokens(_ids(tok, e))[-1] == spec["last"]
    # framing lives where sys_role says it does
    if ad.sys_role is None:
        assert e.index(E_TEXT) > e.index("<|im_start|>user")                 # prepended to the USER turn
    else:
        assert e.index(E_TEXT) < e.index(ad.user_text(Q))
    # BOS handling: encode() must not add one on top of the template
    ids = ad.encode(tok, [n, e])["input_ids"]
    assert "token_type_ids" not in ad.encode(tok, [n])
    row = ids[1].tolist()                                                   # the longer prompt: no left padding
    if spec["bos"] == "single":
        assert row[0] == tok.bos_token_id and row.count(tok.bos_token_id) == 1
    elif spec["bos"] == "none":
        assert tok.bos_token_id not in row
    else:                                                                    # Seed-OSS: <seed:bos> opens every turn, never twice in a row
        assert row[0] == tok.bos_token_id and row[1] != tok.bos_token_id
        assert row.count(tok.bos_token_id) == e.count("<seed:bos>")
    # build_chat (arbitrary user text) goes through the same render()
    c0, c1 = ad.build_chat(tok, "Say hi."), ad.build_chat(tok, "Say hi.", "Be brief.")
    assert "Say hi." in c0 and "Say hi." in c1 and "Be brief." in c1 and "Be brief." not in c0
    assert c0.endswith(ad.gen_prompt_suffix) and c1.endswith(ad.gen_prompt_suffix)
    # user_note (framing text prepended to the user turn) works for every family
    assert ad.build(tok, Q, None, "NOTE-XYZ").index("NOTE-XYZ") < ad.build(tok, Q, None, "NOTE-XYZ").index("Which organelle")


@pytest.mark.parametrize("key", KEYS)
def test_framed_prompt_is_n_plus_one_inserted_span(key):
    """THE paired-contrast invariant, for every key: a framed prompt differs from N ONLY by the framing text (plus its blank-line joiner
    or the markup of the system turn that carries it). Anything else that changes between the two conditions (OLMo once dropped or added
    the 14-token '<functions></functions>' boilerplate) rides along in every N->framing contrast and in every h_E - h_N direction.
    Checked on token ids: all differences must lie in ONE window around the framing: at most 2 ids of N may be replaced (tekken merges
    '.' + '\\n\\n' into one token) and the framed side may exceed the framing by at most 8 ids (joiner / system-turn markup; observed max 4).
    For the families that merge the framing into a default system prompt the string form is checked too: removing it gives exactly N."""
    tok, ad = _tok(key), get_adapter(key)
    n = ad.build(tok, Q)
    n_ids = _ids(tok, n)
    for name in ("E", "C", "awr_not_eval"):                                 # system-turn framings and a user-turn note
        f = FRAMINGS[name]
        text = f.sys_text or f.user_note
        prompt = ad.build(tok, Q, f.sys_text, f.user_note)
        p_ids = _ids(tok, prompt)
        p, n_end, f_end = _diff_window(n_ids, p_ids)
        where = f"{key} [{name}]: N side {n_end - p} ids, framed side {f_end - p} ids, framing {len(_ids(tok, text))} ids"
        assert n_end - p <= 2, f"N and the framed prompt differ by more than the framing paragraph ({where}): N has {tok.decode(n_ids[p:n_end])!r}"
        assert f_end - p <= len(_ids(tok, text)) + 8, f"the framed prompt carries more than the framing paragraph ({where}): {tok.decode(p_ids[p:f_end])[-160:]!r}"
        window = tok.decode(p_ids[max(0, p - 2):f_end + 2], skip_special_tokens=False, clean_up_tokenization_spaces=False)
        assert text in window, (where, window[:100])
        if f.sys_text and ("\n\n" + f.sys_text) in prompt:                  # families with a default system prompt: the string form
            d = ad.default_system(tok)       # (the joiner absorbs trailing whitespace of the default: Magistral's reasoning prompt ends in '\n')
            assert prompt.replace("\n\n" + f.sys_text, "") == n.replace(d, d.rstrip(), 1), (key, name)


def test_mistral_small_matches_native_template_with_pinned_date():
    key = "mistral-small-24b"
    tok, ad = _tok(key), get_adapter(key)
    assert ad.build(tok, Q) == _pin_date(_native(tok, [_u(ad)]), ad.pinned_today)       # N == template default, date pinned
    assert ad.pinned_today in ad.build(tok, Q) and "2023-10-01" in ad.build(tok, Q)
    default = _pin_date(template_default_system(tok, "[SYSTEM_PROMPT]", "[/SYSTEM_PROMPT]"), ad.pinned_today)
    assert default == ad.default_system(tok)
    assert default.startswith("You are Mistral Small 3")
    # E = default + blank line + framing, inside the single system turn
    assert ad.build(tok, Q, E_TEXT) == _native(tok, [{"role": "system", "content": default + "\n\n" + E_TEXT}, _u(ad)])
    assert ad.build(tok, Q, E_TEXT).count("[SYSTEM_PROMPT]") == 1
    assert f"The current date is {ad.pinned_today}." in ad.build(tok, Q)
    assert _native(tok, [_u(ad)]).count("The current date is") == 1              # the vendor template uses strftime_now (today)
    # a second (non-leading) system message is refused
    with pytest.raises(ValueError):
        ad.render(tok, [{"role": "user", "content": "a"}, {"role": "system", "content": "b"}])


def test_mistral_small_pins_the_date_without_callable_template_kwargs(monkeypatch):
    """Regression: transformers 5.24.0 rebuilds every chat-template kwarg from plain JSON types (5.2.0 and 4.57.6 do not), so the former way of pinning the date
    (strftime_now=<lambda>) reached the template as the string '<function>' and mistral-small-24b died on EVERY prompt with
    "TypeError: 'str' object is not callable". The adapter must render natively and pin the date by rewriting it in its default text.
    The sanitising is emulated here, so the test bites on any installed transformers (4.57.6 included)."""
    key = "mistral-small-24b"
    tok, ad = _tok(key), get_adapter(key)
    n0, e0 = ad.build(tok, Q), ad.build(tok, Q, E_TEXT)
    d0 = datetime.date.today().isoformat()
    raw = _native(tok, [_u(ad)])                                            # the vendor render carries the date of TODAY
    d1 = datetime.date.today().isoformat()
    written = re.search(r"The current date is (\d{4}-\d{2}-\d{2})", raw).group(1)
    assert written in (d0, d1)
    monkeypatch.setattr(tok, "apply_chat_template", _sanitising(tok.apply_chat_template))
    # the emulation is faithful: the old approach (a callable kwarg) fails under it exactly as on transformers 5.24
    with pytest.raises(TypeError, match="not callable"):
        tok.apply_chat_template([_u(ad)], tokenize=False, add_generation_prompt=True, strftime_now=lambda fmt: ad.pinned_today)
    # the adapter is unaffected, and the date of the day never reaches a prompt
    assert ad.build(tok, Q) == n0 and ad.build(tok, Q, E_TEXT) == e0
    if written != ad.pinned_today:
        assert written not in n0 and written not in e0
    assert n0.count(f"The current date is {ad.pinned_today}.") == 1


def test_mistral_small_raises_when_the_template_has_no_single_date_sentence(monkeypatch):
    """No silent fallback: if a template revision words the date differently (or drops it) the adapter refuses to build prompts instead
    of leaking today's date into them."""
    key = "mistral-small-24b"
    tok, ad = _tok(key), get_adapter(key)
    real = tok.apply_chat_template

    def reworded(conversation, *args, **kwargs):
        return real(conversation, *args, **kwargs).replace("The current date is ", "Today is ")
    monkeypatch.setattr(tok, "apply_chat_template", reworded)
    with pytest.raises(ValueError, match="exactly one 'The current date is"):
        ad.build(tok, Q)


@pytest.mark.parametrize("key", ["mistral-small-24b", "mistral-small-3.2-24b", "ministral-3-14b", "magistral-small-2509",
                                 "olmo-3.1-32b-instruct", "olmo-3.1-32b-think"])
def test_keep_default_system_switch_gives_template_native_replacement(key, monkeypatch):
    """keep_default_system=False: framing REPLACES the default system prompt (the vendor template's own semantics); N is unchanged."""
    tok, ad = _tok(key), get_adapter(key)
    n_merged, e_merged = ad.build(tok, Q), ad.build(tok, Q, E_TEXT)
    assert ad.keep_default_system is True
    monkeypatch.setattr(type(ad), "keep_default_system", False)
    n, e = ad.build(tok, Q), ad.build(tok, Q, E_TEXT)
    assert n == n_merged and e != e_merged
    assert e.count(E_TEXT) == 1 and "You are being evaluated" in e
    default_marker = {"mistral": "Mistral", "ministral": "Ministral", "magistral": "HOW YOU SHOULD THINK", "olmo": "Olmo"}[key.split("-")[0]]
    system_turn = e.split(ad.user_text(Q))[0]
    assert default_marker not in system_turn, "the default identity must be gone from the system turn"


def test_mistral_regex_fix_and_tekken_equivalence():
    for key in ("mistral-small-24b", "mistral-small-3.2-24b"):
        tok = _tok(key)
        assert _pre_tokenizer_regex(tok) == TEKKEN_PATTERN
        assert tok.padding_side == "left" and tok.pad_token_id is not None and tok.pad_token_id != tok.eos_token_id
    # digits are single tokens (tekken); a regression to the llama-3 regex would not change this, but a broken load would
    tok = _tok("mistral-small-24b")
    assert tok.convert_ids_to_tokens(_ids(tok, "n=1234")) == ["n", "=", "1", "2", "3", "4"]


def test_mistral_3_2_v7_string_and_default_system():
    key = "mistral-small-3.2-24b"
    tok, ad = _tok(key), get_adapter(key)
    assert tok.chat_template is None, "AutoTokenizer does not read the 3.1 repo's chat_template.json -> the V7 string is built by hand"
    # our hand-built V7 string == the HF V7 template of the sibling 3.1 repo when given the same system text
    sysm, tmpl = ad.default_system(), _template_31()
    assert ad.build(tok, Q) == _native(tok, [{"role": "system", "content": sysm}, _u(ad)], chat_template=tmpl)
    assert ad.build(tok, Q, E_TEXT) == _native(tok, [{"role": "system", "content": sysm + "\n\n" + E_TEXT}, _u(ad)], chat_template=tmpl)
    assert ad.build(tok, Q).startswith("<s>[SYSTEM_PROMPT]You are Mistral-Small-3.2-24B-Instruct-2506,")
    assert "2025-06-20" in sysm and "2025-06-19" in sysm and "{" not in sysm
    assert ad.tokenizer_repos[key] == "mistralai/Mistral-Small-3.1-24B-Instruct-2503"
    # drift check against the vendor file, if cached
    try:
        p = Path(_snapshot("mistralai/Mistral-Small-3.2-24B-Instruct-2506")) / "SYSTEM_PROMPT.txt"
    except pytest.skip.Exception:
        p = None
    if p is not None and p.exists():
        assert SYSTEM_PROMPT_32 == p.read_text()
    # multi-turn rendering of the helper equals the template
    msgs = [{"role": "user", "content": "Hello"}, {"role": "assistant", "content": "Hi there"}, {"role": "user", "content": "Again"}]
    assert render_v7(msgs, "SYS") == _native(tok, [{"role": "system", "content": "SYS"}] + msgs, chat_template=tmpl)


def test_mistral_3_2_tokenizer_vocab_matches_tekken_if_cached():
    """The 3.2 repo ships only tekken.json; the 3.1 HF tokenizer we borrow must have the identical vocabulary and tokenise alike."""
    tekken = Path(_snapshot("mistralai/Mistral-Small-3.2-24B-Instruct-2506")) / "tekken.json"
    if not tekken.exists():
        pytest.skip("tekken.json of Mistral-Small-3.2 not cached")
    try:
        import tiktoken
    except ImportError:
        pytest.skip("tiktoken not installed")
    import base64
    tk = json.loads(tekken.read_text())
    n_special = tk["config"]["default_num_special_tokens"]
    nv = tk["config"]["default_vocab_size"] - n_special
    ranks = {base64.b64decode(v["token_bytes"]): v["rank"] for v in tk["vocab"][:nv]}
    enc = tiktoken.Encoding("tekken32", pat_str=tk["config"]["pattern"], mergeable_ranks=ranks, special_tokens={})
    tok = _tok("mistral-small-3.2-24b")
    user = get_adapter("mistral-small-3.2-24b").user_text(Q)
    for text in [E_TEXT, user, SYSTEM_PROMPT_32, "pH 7.4, n=1234; DNA's 5'-end (β-lactamase) CRISPR-Cas9 \r\n  tab\t 日本語 😀"]:
        assert _ids(tok, text) == [t + n_special for t in enc.encode(text)], text[:50]


def test_mistral_reasoning_variants_prompts_and_tags():
    """Magistral / Ministral-3-Reasoning: default system prompt = the vendor reasoning prompt, framing appended, [THINK] special tokens."""
    import glob
    from sbx.adapters.mistral import REASONING_SYSTEM_PROMPT
    tok = _tok("ministral-3-14b-reasoning")
    assert tok.convert_tokens_to_ids("[THINK]") == 34 and tok.convert_tokens_to_ids("[/THINK]") == 35
    assert len(_ids(tok, "[THINK]")) == 1 and len(_ids(tok, "[/THINK]")) == 1
    assert tok.decode([34, 1000, 35], skip_special_tokens=False).startswith("[THINK]") and "[/THINK]" in tok.decode([35], skip_special_tokens=False)
    for key in ("magistral-small-2509", "ministral-3-14b-reasoning"):
        ad = get_adapter(key)
        n, e = ad.build(tok, Q), ad.build(tok, Q, E_TEXT)
        sysm = ad.default_system(tok)
        assert sysm.startswith("# HOW YOU SHOULD THINK AND ANSWER") and "[THINK]" in sysm and "[/THINK]" in sysm
        assert n == _native(tok, [{"role": "system", "content": sysm}, _u(ad)])          # same V7 layout as the vendor HF template
        assert e == _native(tok, [{"role": "system", "content": sysm.rstrip() + "\n\n" + E_TEXT}, _u(ad)])
        assert ad.answer_mode == "reasoning" and ad.think_close == "[/THINK]" and ad.think_open_in_prompt is False
    # Ministral-Reasoning: N is also the vendor template's own default render (the default prompt has no placeholders)
    assert get_adapter("ministral-3-14b-reasoning").build(tok, Q) == _native(tok, [_u(get_adapter("ministral-3-14b-reasoning"))])
    # Magistral's embedded prompt == its SYSTEM_PROMPT.txt (drift check, if cached)
    try:
        files = glob.glob(str(Path(_snapshot("mistralai/Magistral-Small-2509")) / "SYSTEM_PROMPT.txt"))
    except pytest.skip.Exception:
        files = []
    if files:
        assert Path(files[0]).read_text() == REASONING_SYSTEM_PROMPT
    assert get_adapter("magistral-small-2509").tokenizer_repos["magistral-small-2509"] == "mistralai/Ministral-3-14B-Reasoning-2512"


def test_ministral3_requires_transformers_5(monkeypatch):
    import transformers
    ad = get_adapter("ministral-3-14b")
    assert ad.min_transformers == (5, 0, 0) and ad.repo.endswith("Ministral-3-14B-Instruct-2512-BF16")
    monkeypatch.setattr(transformers, "__version__", "4.57.6")
    for k in ("ministral-3-14b", "ministral-3-14b-reasoning"):
        with pytest.raises(RuntimeError, match="transformers>=5.0.0"):
            get_adapter(k).load()
    tok = _tok("ministral-3-14b")
    n = ad.build(tok, Q)
    assert "{today}" not in n and "{yesterday}" not in n and "2025-12-02" in n and n.startswith("<s>[SYSTEM_PROMPT]You are Ministral-3")
    # the vendor template would leave the placeholders unfilled
    assert "{today}" in _native(tok, [_u(ad)])


@pytest.mark.parametrize("key", ["phi-4", "phi-4-mini-instruct"])
def test_phi_instruct_matches_native_template(key):
    tok, ad = _tok(key), get_adapter(key)
    assert ad.build(tok, Q) == _native(tok, [_u(ad)])
    assert ad.build(tok, Q, E_TEXT) == _native(tok, [{"role": "system", "content": E_TEXT}, _u(ad)])
    assert "system" not in ad.build(tok, Q).split(ad.user_text(Q))[0]      # no default system prompt in N


@pytest.mark.parametrize("key", ["phi-4-reasoning", "phi-4-reasoning-plus"])
def test_phi_reasoning_system_prompt_and_dropped_system_turn(key):
    tok, ad = _tok(key), get_adapter(key)
    assert ad.sys_role is None and ad.answer_mode == "reasoning"
    n, e = ad.build(tok, Q), ad.build(tok, Q, E_TEXT)
    assert n == _native(tok, [_u(ad)])
    assert n.startswith("<|im_start|>system<|im_sep|>You are Phi, a language model trained by Microsoft")
    assert "<think> {Thought section} </think> {Solution section}" in n
    # evidence for sys_role=None: the template silently drops a system message
    dropped = _native(tok, [{"role": "system", "content": E_TEXT}, _u(ad)])
    assert E_TEXT not in dropped and dropped == n
    assert e == _native(tok, [{"role": "user", "content": E_TEXT + "\n\n" + ad.user_text(Q)}])
    with pytest.raises(ValueError, match="drops system messages"):
        ad.render(tok, [{"role": "system", "content": "x"}, _u(ad)])
    # <think> / </think> are single special tokens in this tokenizer
    assert len(_ids(tok, "<think>")) == 1 and len(_ids(tok, "</think>")) == 1


def _system_turn(prompt):
    """Text of the (first) system turn of a rendered OLMo prompt."""
    start = prompt.index("<|im_start|>system\n") + len("<|im_start|>system\n")
    return prompt[start:prompt.index("<|im_end|>", start)]


@pytest.mark.parametrize("key", OLMO_KEYS)
def test_olmo_default_system_and_merge(key):
    """Regression (reviewer finding): for 4 of the 5 keys N and a framed prompt used to differ by the tool-less '<functions></functions>'
    boilerplate in addition to the framing paragraph (Instruct: stripped from the framed prompt only; Think-32B: appended by the template
    to the framed prompt only). Now the boilerplate occurs exactly once in the system turn of N and of every framed prompt."""
    tok, ad = _tok(key), get_adapter(key)
    n, e = ad.build(tok, Q), ad.build(tok, Q, E_TEXT)
    think = key.endswith("think")
    think32 = think and "32b" in key        # the two 32B Think templates leave the boilerplate out of their DEFAULT system prompt
    native_n = _native(tok, [_u(ad)])
    default = ad.default_system(tok)
    assert default and ("Olmo" in default or "OLMo" in default or "helpful" in default)
    for p in (n, e):
        assert _system_turn(p).count(BOILERPLATE) == 1, (key, _system_turn(p)[-160:])
    if think32:
        # documented deviation: N goes through an explicit system turn so that it carries the boilerplate like every framed prompt
        assert BOILERPLATE not in native_n and n != native_n
        assert n == _native(tok, [{"role": "system", "content": default}, _u(ad)])
        assert n.replace(" " + BOILERPLATE, "") == native_n
    else:
        assert n == native_n, "N must be the template's own default render"
    if think:                                # the template appends the boilerplate to every given system message -> default is bare
        assert "<functions>" not in default and default.endswith(".")
    else:                                    # Instruct: the template appends nothing -> the default is used verbatim, boilerplate included
        assert default.endswith(BOILERPLATE)
    assert e == _native(tok, [{"role": "system", "content": default + "\n\n" + E_TEXT}, _u(ad)])
    assert e.count("<|im_start|>system") == 1
    # the paired-contrast invariant in its string form (the generic token-level test covers every key)
    assert e.replace("\n\n" + E_TEXT, "") == n
    assert n.endswith("<|im_start|>assistant\n<think>") == think and e.endswith("<|im_start|>assistant\n<think>") == think
    # <think> is plain multi-token text in this tokenizer (not a special token) -> split() must work on text
    assert len(_ids(tok, "<think>")) > 1
    if think:
        assert ad.think_open_in_prompt is True
    else:
        assert not hasattr(ad, "think_open_in_prompt")


def test_olmo_default_identity_sentences():
    ident31 = ("You are Olmo, a helpful AI assistant built by Ai2. Your date cutoff is December 2024, and your model weights are "
               "available at https://huggingface.co/allenai.")
    d = lambda key: get_adapter(key).default_system(_tok(key))              # noqa: E731
    assert d("olmo-3.1-32b-instruct") == ident31 + " " + BOILERPLATE        # Instruct: template default, boilerplate included
    assert d("olmo-3.1-32b-think") == ident31                               # Think-32B: the template adds the boilerplate itself
    assert d("olmo-3-32b-think") == "You are a helpful AI assistant."
    assert d("olmo-3-7b-instruct") == "You are a helpful function-calling AI assistant. " + BOILERPLATE
    assert d("olmo-3-7b-think").startswith("You are OLMo, a helpful function-calling AI assistant built by Ai2.")
    assert "<functions>" not in d("olmo-3-7b-think")                        # 7B Think: default has it, but the template re-adds it -> stripped


def test_olmo_render_raises_when_the_boilerplate_is_not_there_exactly_once(monkeypatch):
    """No silent fallback: a wrong boilerplate policy (or a template that changes it) must stop the run, not confound every contrast."""
    think, ins = get_adapter("olmo-3-7b-think"), get_adapter("olmo-3-7b-instruct")
    monkeypatch.setattr(type(think), "template_appends_boilerplate", False)     # keeps it in the default -> the template adds a second one
    with pytest.raises(ValueError, match=r"occurs 2x"):
        think.build(_tok("olmo-3-7b-think"), Q, E_TEXT)
    monkeypatch.setattr(type(ins), "template_appends_boilerplate", True)        # strips it, but the Instruct template never re-adds it
    with pytest.raises(ValueError, match=r"occurs 0x"):
        ins.build(_tok("olmo-3-7b-instruct"), Q)


@pytest.mark.parametrize("key", ["olmo-3.1-32b-instruct", "olmo-3.1-32b-think"])
def test_olmo_native_replacement_mode_keeps_the_vendor_boilerplate_semantics(key, monkeypatch):
    """keep_default_system=False is the vendor's own semantics: the framing REPLACES the default identity, and the boilerplate follows
    the template (Instruct: none after a given system message; Think: appended). N is the same in both modes."""
    tok, ad = _tok(key), get_adapter(key)
    n = ad.build(tok, Q)
    monkeypatch.setattr(type(ad), "keep_default_system", False)
    assert ad.build(tok, Q) == n
    e = ad.build(tok, Q, E_TEXT)
    assert _system_turn(e).count(BOILERPLATE) == (1 if key.endswith("think") else 0)
    assert e == _native(tok, [{"role": "system", "content": E_TEXT}, _u(ad)])


def test_olmo_load_registers_im_end_stop(monkeypatch):
    """load() must add <|im_end|> to the generation eos ids without needing real weights."""
    from transformers import GenerationConfig
    from sbx.adapters.base import Adapter
    tok = _tok("olmo-3.1-32b-instruct")

    class FakeModel:
        generation_config = GenerationConfig(eos_token_id=100257)

    monkeypatch.setattr(Adapter, "load", lambda self, *a, **k: (tok, FakeModel()))
    _, model = get_adapter("olmo-3.1-32b-instruct").load()
    assert model.generation_config.eos_token_id == [100257, tok.convert_tokens_to_ids("<|im_end|>")]


@pytest.mark.parametrize("key", ["seed-oss-36b-instruct", "seed-oss-36b-instruct-nothink"])
def test_seed_oss_template_and_thinking_budget(key):
    tok, ad = _tok(key), get_adapter(key)
    nothink = key.endswith("nothink")
    budget_default = 0 if nothink else -1
    assert ad.build(tok, Q) == _native(tok, [_u(ad)], thinking_budget=budget_default)
    assert ad.build(tok, Q, E_TEXT) == _native(tok, [{"role": "system", "content": E_TEXT}, _u(ad)], thinking_budget=budget_default)
    # effort -> budget (per-framing knob)
    for effort, b in [("off", 0), ("0", 0), ("unlimited", -1), ("512", 512), ("2048", 2048), ("default", -1), (None, budget_default)]:
        assert ad.build(tok, Q, None, None, effort) == _native(tok, [_u(ad)], thinking_budget=b), (effort, b)
    p512 = ad.build(tok, Q, E_TEXT, None, "512")
    assert p512.index(E_TEXT) < p512.index("thinking budget, which is 512") < p512.index("Which organelle")
    assert p512.endswith("<seed:bos>assistant\n")
    p0 = ad.build(tok, Q, None, None, "0")
    assert p0.endswith("<seed:think>\n<seed:cot_budget_reflect>The current thinking budget is 0, so I will directly start answering "
                       "the question.</seed:cot_budget_reflect>\n</seed:think>")
    if not nothink:
        assert ad.build(tok, Q).endswith("<seed:bos>assistant\n") and "thinking budget" not in ad.build(tok, Q)
        assert ad.answer_mode == "reasoning"
    else:
        assert ad.answer_mode == "direct" and ad.max_new <= 64
    with pytest.raises(ValueError):
        parse_budget("lots", -1)


def test_seed_oss_special_tokens_and_parse_budget():
    tok = _tok("seed-oss-36b-instruct")
    assert _ids(tok, "<seed:think>") == [3] and _ids(tok, "</seed:think>") == [4]
    assert [parse_budget(x, -1) for x in (None, "-1", "ON", " off ", "none", "0", "1024", 2048, "-5")] == [-1, -1, -1, 0, 0, 0, 1024, 2048, -1]
    assert parse_budget(None, 0) == 0


# ------------------------------------------------------------------------------------------------ parsing / splitting
PARSE_CASES = [
    # (text, nchoices, mode, expected 0-based index or None)
    ("Let's think. Mitochondria make ATP.\n\nAnswer: B", 4, "cot", 1),           # clean
    ("**Answer:** B", 4, "cot", 1), ("**Answer: B**", 4, "cot", 1), ("Answer: **B**", 4, "cot", 1),     # bold
    ("Answer: (C)", 4, "cot", 2), ("Answer: B.", 4, "cot", 1), ("Answer: B) Mitochondrion", 4, "cot", 1),  # punctuation
    ("The answer is B.", 4, "cot", 1), ("The correct answer is **D**.", 4, "cot", 3), ("Final answer - C", 4, "cot", 2),
    ("Answer: Option C", 4, "cot", 2), ("Answer is: [D]", 4, "cot", 3), ("**Answer:** **B**", 4, "cot", 1),
    ("The answer to this question is B.", 4, "cot", 1), ("The answer to the question is **C**", 4, "cot", 2),
    ("Answer: \\boxed{B}", 4, "cot", 1), ("\\boxed{\\text{C}}", 4, "cot", 2),
    ("First answer: A. But on reflection the final Answer: C", 4, "cot", 2),     # last answer wins
    ("Answer: A", 4, "cot", 0), ("Answer: A.", 4, "cot", 0), ("Answer: A\n\nbecause", 4, "cot", 0),
    ("Answer: A<|im_end|><|pad|>", 4, "cot", 0), ("Answer: B</s><pad><pad>", 4, "cot", 1),   # trailing special tokens
    ("Answer: I", 10, "cot", 8),
    ("B", 4, "cot", 1), ("**B**", 4, "cot", 1), ("(C) Golgi", 4, "cot", 2),     # bare label reply in cot mode
    # --- must NOT parse
    ("answer: b", 4, "cot", None), ("the answer is a combination of factors", 4, "cot", None),
    ("Answer: A person who...", 4, "cot", None), ("Answer: I think it is C", 4, "cot", None),
    ("Answer: E", 4, "cot", None), ("Answer: Because it makes ATP", 4, "cot", None), ("Answer: B-cell receptor", 4, "cot", None),
    ("Let me think about this carefully. The mitochondria are the", 4, "cot", None),     # truncated, no answer
    ("I'm not sure which one is right.", 4, "cot", None), ("", 4, "cot", None),
    # reasoning / direct (asked for only the letter)
    ("B", 4, "direct", 1), ("B.", 4, "direct", 1), ("**C**", 4, "reasoning", 2), ("(D)", 4, "reasoning", 3),
    ("B is correct", 4, "reasoning", 1), ("A", 4, "reasoning", 0), ("I", 10, "reasoning", 8), ("B<|im_end|>", 4, "reasoning", 1),
    ("**Solution**\n\nThe correct option is **C**.", 4, "reasoning", 2), ("Final Answer: \\boxed{D}", 4, "reasoning", 3),
    ("Answer: B", 4, "reasoning", 1),
    ("Because...", 4, "reasoning", None), ("Based on the above", 4, "reasoning", None), ("A man", 4, "reasoning", None),
    ("I think C", 10, "reasoning", None), ("E", 4, "direct", None), ("", 4, "reasoning", None),
]


@pytest.mark.parametrize("text,n,mode,exp", PARSE_CASES)
def test_parse_mcq_letter(text, n, mode, exp):
    assert parse_mcq_letter(text, n, mode) == exp


@pytest.mark.parametrize("filler", [" ", "\n", "*", "\n* ", "(", ":-"])
def test_parse_is_linear_on_degenerate_outputs(filler):
    """Runs of separators after the keyword (typical of repetition loops) must not make the regexes backtrack."""
    import time
    for text in ["Answer:" + filler * 60000, "answer " * 20000, "Answer: " + filler * 30000 + "B", "**" * 50000 + "Answer"]:
        t0 = time.time()
        for mode in ("cot", "reasoning"):
            parse_mcq_letter(text, 4, mode)
        assert time.time() - t0 < 2.0, (text[:20], filler)
    assert parse_mcq_letter("Answer: " + filler * 3000 + "B", 4, "cot") == 1          # separators/openers are skipped, B found


def test_split_think_cases():
    f = lambda t, o: split_think(t, "<think>", "</think>", o)               # noqa: E731
    assert f("<think>abc</think>\n\nB", False) == ("abc", "B")
    assert f("abc</think>\n\nB", True) == ("abc", "B")
    assert f("<think>abc", False) == ("abc", "")                             # truncated, opening tag seen
    assert f("abc def", True) == ("abc def", "")                             # truncated, think block opened in the prompt
    assert f("abc def", False) == ("", "abc def")                            # model skipped thinking
    assert f("</think>B", True) == ("", "B")


# realistic raw generations (skip_special_tokens=False, i.e. with end/pad tokens) -> expected letter index
REALISTIC = {
    "cot": [   # (raw text, expected)
        ("The cell's ATP is produced mainly by oxidative phosphorylation, which takes place in mitochondria.\n\nAnswer: B</s>", 1),
        ("### Step 1\nRibosomes make protein.\n### Step 2\nMitochondria make ATP.\n\n**Answer: B**</s><pad><pad>", 1),
        ("Let's reason: the nucleus stores DNA...\n\n**Answer:** (B) Mitochondrion<|im_end|><|endoftext|>", 1),
        ("Step by step: first, consider the nucleus, which stores DNA and does not make ATP. Next, the ribosome", None),   # truncated
        ("I'm sorry, but I can't determine that.", None),
    ],
    "reasoning_think": [  # thinking tags are added per family below
        ("{O}Okay, the question asks for the ATP-producing organelle. Mitochondria. So B.{C}\n\n**B**{E}", 1),
        ("{O}Okay, the question asks for the ATP-producing organelle. Mitochondria. So B.{C}\n\nB.{E}", 1),
        ("{O}Okay, let me think. The answer is clearly mitochondria so the answer is B.{C}\n\nThe answer is **B**.{E}", 1),
        ("{O}Hmm, let me reconsider. The nucleus does not make ATP, but wait, maybe the Golgi", None),                        # no closing tag
        ("{O}Thinking...{C}\n\nI cannot determine the answer from the options.{E}", None),
    ],
}
THINK_TAGS = {
    "phi-4-reasoning": dict(O="<think>\n", C="\n</think>", E="<|im_end|>"),
    "phi-4-reasoning-plus": dict(O="<think>\n", C="\n</think>", E="<|im_end|>"),
    "olmo-3.1-32b-think": dict(O="", C="</think>", E="<|endoftext|><|pad|>"),
    "olmo-3-32b-think": dict(O="", C="</think>", E="<|endoftext|>"),
    "olmo-3-7b-think": dict(O="", C="</think>", E="<|endoftext|>"),
    "seed-oss-36b-instruct": dict(O="<seed:think>\n", C="\n</seed:think>\n", E="<seed:eos><seed:pad>"),
    "magistral-small-2509": dict(O="[THINK]", C="[/THINK]", E="</s><pad><pad>"),
    "ministral-3-14b-reasoning": dict(O="[THINK]", C="[/THINK]", E="</s>"),
}


@pytest.mark.parametrize("key", [k for k in KEYS if get_adapter(k).answer_mode == "cot"])
def test_cot_models_split_and_parse_realistic(key):
    ad = get_adapter(key)
    for raw, exp in REALISTIC["cot"]:
        reasoning, final = ad.split(raw)
        assert reasoning == "" and final == raw                              # visible CoT: no separate channel
        assert ad.parse_letter(final, 4) == exp, raw


@pytest.mark.parametrize("key", list(THINK_TAGS))
def test_reasoning_models_split_and_parse_realistic(key):
    ad, tags = get_adapter(key), THINK_TAGS[key]
    for raw, exp in REALISTIC["reasoning_think"]:
        text = raw.format(**tags)
        reasoning, final = ad.split(text)
        assert reasoning, "reasoning channel must be separated"
        if exp is None and tags["C"].strip() not in text:                    # truncated: nothing after the reasoning
            assert final == ""
        assert ad.parse_letter(final, 4) == exp, (key, text)
    # no thinking tags at all: Olmo/Phi-reasoning treat it as (truncated) reasoning, Seed-OSS as a direct answer
    r, f = ad.split("B<|endoftext|>")
    if key.startswith("seed"):
        assert (r, f) == ("", "B<|endoftext|>") and ad.parse_letter(f, 4) == 1
    elif ad.think_open_in_prompt:
        assert f == "" and r
    else:
        assert r == "" and f.startswith("B")


def test_seed_nothink_direct_outputs():
    ad = get_adapter("seed-oss-36b-instruct-nothink")
    assert ad.answer_mode == "direct"
    for raw, exp in [("B<seed:eos>", 1), ("**C**<seed:eos><seed:pad>", 2), ("Answer: D<seed:eos>", 3), ("I don't know", None), ("", None)]:
        r, f = ad.split(raw)
        assert r == "" and ad.parse_letter(f, 4) == exp, raw


def test_seed_budget_reflection_blocks_do_not_confuse_split():
    ad = get_adapter("seed-oss-36b-instruct")
    raw = ("<seed:think>\nGot it, let's see. Mitochondria.\n<seed:cot_budget_reflect>I have used 129 tokens, and there are 383 tokens "
           "remaining for use.</seed:cot_budget_reflect>\nSo the answer is B.\n<seed:cot_budget_reflect>I have exhausted my token "
           "budget, and now I will start answering the question.</seed:cot_budget_reflect>\n</seed:think>\nB<seed:eos>")
    r, f = ad.split(raw)
    assert "Mitochondria" in r and "<seed:think>" not in r and f == "B<seed:eos>" and ad.parse_letter(f, 4) == 1


# ------------------------------------------------------------------------------------------------ architecture checks
@lru_cache(maxsize=None)
def _build_meta(key):
    """(config, meta-device model) of the real architecture; cached because two tests read it (neither mutates it)."""
    from accelerate import init_empty_weights
    from transformers import AutoModelForCausalLM, AutoModelForImageTextToText
    cfg = _cfg(key)
    cls = AutoModelForImageTextToText if hasattr(cfg, "text_config") else AutoModelForCausalLM
    with init_empty_weights():
        return cfg, cls.from_config(cfg)


def _checkpoint_key_converter(model):
    """checkpoint key -> model parameter name, renamed the way the INSTALLED transformers renames keys while loading: 4.x (and 5.0-5.2)
    keep a class attribute `_checkpoint_conversion_mapping` (regex -> replacement, first match wins); newer 5.x moved the rules into
    transformers.conversion_mapping, whose WeightRenaming transforms the loader applies ALL in turn."""
    cmap = getattr(type(model), "_checkpoint_conversion_mapping", None)
    if cmap is not None:
        cmap = dict(cmap)

        def conv(k):
            for pat, rep in cmap.items():
                k2, n = re.subn(pat, rep, k)
                if n:
                    return k2
            return k
        return conv
    try:
        from transformers.conversion_mapping import WeightRenaming, get_model_conversion_mapping
        renamings = [t for t in get_model_conversion_mapping(model) if isinstance(t, WeightRenaming)]
    except Exception as e:                                      # noqa: BLE001 - private API: say so instead of failing for the wrong reason
        pytest.skip(f"this transformers has no usable checkpoint-key renaming API ({type(e).__name__}: {str(e)[:80]})")

    def conv(k):
        for t in renamings:
            k, _ = t.rename_source_key(k)
        return k
    return conv


@pytest.mark.parametrize("key", KEYS)
def test_layer_path_on_meta_model(key):
    _skip_if_transformers_too_old(key)
    ad = get_adapter(key)
    cfg, model = _build_meta(key)
    tc = getattr(cfg, "text_config", cfg)
    layers = ad.layers(model)
    assert len(layers) == tc.num_hidden_layers == ad.num_layers(model)
    assert ad.hidden_size(model) == tc.hidden_size
    assert next(model.parameters()).device.type == "meta"
    card = json.loads((MODELS_DIR / f"{key}.json").read_text())
    path = card["layer_path"].split(".")
    assert path[0] == "model", "layer_path is a python expression on the loaded model object, e.g. model.model.layers"
    assert reduce(getattr, path[1:], model) is layers
    assert card["num_layers"] == tc.num_hidden_layers and card["hidden_size"] == tc.hidden_size
    assert card["layer_path_verified"]["len_layers"] == len(layers)


@pytest.mark.parametrize("key", KEYS)
def test_checkpoint_keys_map_onto_model_parameters(key):
    """Every checkpoint weight name must map onto a model parameter (else HF would silently leave weights randomly initialised)."""
    _skip_if_transformers_too_old(key)
    _, model = _build_meta(key)
    idx = Path(_snapshot(SPEC[key]["cfg"])) / "model.safetensors.index.json"
    if not idx.exists():
        pytest.skip(f"{SPEC[key]['cfg']} has no model.safetensors.index.json in the local cache (single-file checkpoint?)")
    conv = _checkpoint_key_converter(model)
    ck = {conv(k) for k in json.loads(idx.read_text())["weight_map"]}
    mk = set(model.state_dict().keys())
    tied = set(getattr(model, "_tied_weights_keys", None) or [])
    assert not (ck - mk), f"{key}: checkpoint weights without a model parameter: {sorted(ck - mk)[:5]}"
    assert (mk - ck) <= tied, f"{key}: model parameters missing in the checkpoint (and not tied): {sorted((mk - ck) - tied)[:5]}"
    card = json.loads((MODELS_DIR / f"{key}.json").read_text())
    assert card["checkpoint_key_check"]["unexpected_in_checkpoint"] == []


def _tiny(key, nl=3):
    """Real architecture class + real tokenizer, tiny random weights (CPU)."""
    import torch
    from transformers import AutoModelForCausalLM, AutoModelForImageTextToText
    cfg = _cfg(key)
    tc = getattr(cfg, "text_config", cfg)
    tc.hidden_size, tc.intermediate_size, tc.num_hidden_layers = 64, 128, nl
    tc.num_attention_heads, tc.num_key_value_heads = 4, 2
    if getattr(tc, "head_dim", None) is not None:
        tc.head_dim = 16
    if type(tc).__name__ == "Phi3Config":                                   # phi-4-mini's longrope factors are sized for the real head_dim
        if hasattr(tc, "rope_parameters"):                                  # transformers>=5 naming
            tc.rope_parameters = {"rope_type": "default", "rope_theta": 10000.0}
        else:
            tc.rope_scaling = None
    if getattr(tc, "layer_types", None):
        tc.layer_types = (["sliding_attention"] * nl)[:-1] + ["full_attention"]
    if hasattr(cfg, "vision_config"):
        vc = cfg.vision_config
        vc.hidden_size, vc.intermediate_size, vc.num_hidden_layers, vc.num_attention_heads, vc.head_dim, vc.image_size = 32, 64, 1, 2, 16, 28
    torch.manual_seed(0)
    cls = AutoModelForImageTextToText if hasattr(cfg, "vision_config") else AutoModelForCausalLM
    return cls.from_config(cfg).eval()


@pytest.mark.parametrize("key", ["mistral-small-24b", "mistral-small-3.2-24b", "phi-4", "phi-4-mini-instruct",
                                 "olmo-3.1-32b-instruct", "seed-oss-36b-instruct"])
def test_one_based_layer_convention_and_pipeline_on_tiny_models(key):
    """layers(model)[L-1] output == hidden_states[L] for L<nl (L=nl is post-final-norm), and the whole sbx.core pipeline
    (prompts -> generate -> split/parse, hidden_last, steering hooks) runs with this adapter's prompts/encoding."""
    import torch
    from sbx.core import Ctx, spec_add
    from sbx.testing import toy_questions
    tok, ad = _tok(key), get_adapter(key)
    model = _tiny(key)
    ctx = Ctx(ad, tok, model, log=lambda *a, **k: None)
    assert ctx.nl == 3 and ctx.H == 64
    qs = toy_questions(3)
    prompts = ctx.prompts(qs, FRAMINGS["E"])
    cap = {}
    hooks = [m.register_forward_hook(lambda m, i, o, L=L: cap.__setitem__(L, (o[0] if isinstance(o, tuple) else o).detach().clone()))
             for L, m in enumerate(ad.layers(model), 1)]
    enc = ad.encode(tok, prompts)
    with torch.no_grad():
        out = model(**enc, output_hidden_states=True)
    for h in hooks:
        h.remove()
    assert len(out.hidden_states) == ctx.nl + 1
    # the terms of the cards' hidden_last_80gb memory formula are really alive in this forward (Ctx.hidden_last / neutral_stats): the
    # hidden states of every layer (above), the logits of EVERY prompt position over the full vocabulary, and a KV cache exactly when
    # the config says use_cache (OLMo-3 ships use_cache=false: no cache is built there)
    text_cfg = getattr(model.config, "text_config", model.config)
    assert out.logits.shape == tuple(enc["input_ids"].shape) + (text_cfg.vocab_size,)
    assert (getattr(out, "past_key_values", None) is not None) == bool(text_cfg.use_cache), key
    for L in range(1, ctx.nl):
        assert torch.equal(out.hidden_states[L], cap[L]), f"hidden_states[{L}] != output of layers[{L - 1}]"
    assert not torch.equal(out.hidden_states[ctx.nl], cap[ctx.nl]), "L=nl should be post-final-norm"
    hl = ctx.hidden_last(prompts)
    assert hl.shape == (3, 4, 64) and np.isfinite(hl.astype(np.float32)).all()
    v = np.ones(64, np.float32) / 8.0
    base = ctx.run(qs, FRAMINGS["N"], max_new=4)
    steered = ctx.run(qs, FRAMINGS["N"], specs=spec_add(ctx, {2: v}, 2, 50.0, {2: 1.0}), max_new=4)
    assert len(base.texts) == 3 and base.corr.shape == (3,) and base.parsed.shape == (3,)
    assert base.texts != steered.texts, "a steering hook at layer 2 must change the generations of the tiny model"
    assert ctx.run(qs, FRAMINGS["N"], max_new=4).texts == base.texts, "greedy decoding must be deterministic"
    U, med = ctx.neutral_stats(prompts, {"d": {1: v, 2: v}}, [1, 2])
    assert set(U) == {1, 2} and all(np.isfinite(x) and x > 0 for x in U.values()) and set(med["d"]) == {1, 2}


@pytest.mark.parametrize("key,auto", [("mistral-small-24b", "AutoModelForCausalLM"), ("mistral-small-3.2-24b", "AutoModelForImageTextToText")])
def test_mistral_load_plumbing_with_stubbed_weights(key, auto, monkeypatch):
    """load() wiring without weights: right Auto class, bf16/device_map kwargs, fixed tokenizer, .eval(), vocab check."""
    import torch
    import transformers
    ad = get_adapter(key)
    tiny, seen = _tiny(key), {}

    def fake(cls, repo, **kw):
        seen["repo"], seen["kw"] = repo, kw
        return tiny

    monkeypatch.setattr(getattr(transformers, auto), "from_pretrained", classmethod(fake))
    tok_path = _snapshot(SPEC[key]["tok"])
    if hasattr(ad, "tokenizer_repos"):
        monkeypatch.setitem(ad.tokenizer_repos, key, tok_path)            # offline: the local snapshot instead of the hub id
    repo = tok_path if key == "mistral-small-24b" else "stub-model-repo"
    tok, model = ad.load(repo=repo, dtype="bf16", device_map="cpu")
    assert model is tiny and not model.training
    assert seen["repo"] == repo and seen["kw"]["torch_dtype"] is torch.bfloat16 and seen["kw"]["device_map"] == "cpu"
    assert _pre_tokenizer_regex(tok) == TEKKEN_PATTERN and tok.padding_side == "left" and tok.pad_token == "<pad>"
    ad.load(repo=repo)                                                    # defaults: dtype="auto", device_map="auto"
    assert seen["kw"]["torch_dtype"] == "auto" and seen["kw"]["device_map"] == "auto" and "quantization_config" not in seen["kw"]


def test_mistral3_load_rejects_vocab_mismatch(monkeypatch):
    import transformers
    ad = get_adapter("mistral-small-3.2-24b")
    tiny = _tiny("mistral-small-3.2-24b")
    tiny.config.text_config.vocab_size = 1234
    monkeypatch.setattr(transformers.AutoModelForImageTextToText, "from_pretrained", classmethod(lambda cls, repo, **kw: tiny))
    monkeypatch.setitem(ad.tokenizer_repos, "mistral-small-3.2-24b", _snapshot(SPEC["mistral-small-3.2-24b"]["tok"]))
    with pytest.raises(RuntimeError, match="vocab size mismatch"):
        ad.load(repo="stub-model-repo")


# ------------------------------------------------------------------------------------------------ model cards
REQUIRED = ["key", "repo", "family", "adapter", "params_b", "active_params_b", "dtype", "weights_gb_bf16", "fits_single_80gb",
            "fits_single_80gb_how", "fits_single_80gb_detail", "gated", "accessible_with_local_token", "access_check", "license",
            "min_transformers", "verified_transformers", "trust_remote_code", "thinking", "answer_mode", "max_new", "sys_role",
            "layer_path", "layer_path_verified", "num_layers", "hidden_size", "est_prompt_tokens", "template_notes", "known_issues",
            "recommended_gpu", "est_tokens_per_question", "est_tokens_per_question_basis", "verification_date"]


@pytest.mark.parametrize("key", KEYS)
def test_model_card(key):
    p = MODELS_DIR / f"{key}.json"
    assert p.exists(), f"missing model card {p}"
    card = json.loads(p.read_text())
    for f in REQUIRED:
        assert f in card, f"{key}: card field {f!r} missing"
    ad = get_adapter(key)
    assert card["key"] == key and card["repo"] == ad.repos[key]
    assert card["answer_mode"] == ad.answer_mode and card["max_new"] == ad.max_new and card["sys_role"] == ad.sys_role
    assert card["adapter"] == f"{type(ad).__module__}.{type(ad).__name__}"
    assert card["thinking"] in ("off", "on", "always")
    assert (card["thinking"] == "off") == (ad.answer_mode != "reasoning")
    for b in ("fits_single_80gb", "gated", "accessible_with_local_token", "trust_remote_code"):
        assert isinstance(card[b], bool), b
    assert card["accessible_with_local_token"] is True and card["gated"] is False and card["trust_remote_code"] is False
    assert isinstance(card["params_b"], float) and card["active_params_b"] == card["params_b"]
    assert abs(card["weights_gb_bf16"] - card["params_b"] * 2) / (card["params_b"] * 2) < 0.02
    assert re.fullmatch(r"\d+\.\d+\.\d+", card["min_transformers"])
    assert card["access_check"]["config_json_status"] == 200 and card["access_check"]["first_shard_status"] in (200, 302)
    assert card["fits_single_80gb_how"] in ("bf16", "4bit", "8bit") and isinstance(card["fits_single_80gb_detail"], str)
    assert set(card["est_prompt_tokens"]) >= {"N", "E", "basis"} and card["est_prompt_tokens"]["E"] > card["est_prompt_tokens"]["N"] > 20
    if ad.answer_mode == "reasoning":
        assert "greedy_only_note" in card
    assert isinstance(card["known_issues"], list) and card["known_issues"] and all(isinstance(s, str) for s in card["known_issues"])
    assert isinstance(card["template_notes"], str) and len(card["template_notes"]) > 80
    assert isinstance(card["est_tokens_per_question"], int) and card["est_tokens_per_question"] > 0
    assert card["layer_path"].startswith("model.") and card["layer_path"].endswith(".layers")


CARD_GPU_GIB, CARD_OVERHEAD_GIB, CORE_TOKEN_BUDGET = 79.2, 0.8, 14000     # an 80 GB card; sbx.core.batches' default tok_budget


def _uses_cache(cfg_json):
    """Does model(**enc) (use_cache=None) build a KV cache? It follows config.use_cache; all OLMo-3 configs ship use_cache=false."""
    tc = cfg_json.get("text_config", cfg_json)
    return bool(tc.get("use_cache", cfg_json.get("use_cache", True)))


def _hidden_last_bytes_per_token(cfg_json):
    """What ONE prompt token costs inside Ctx.hidden_last (bf16): the hidden states of every layer + the full-vocabulary logits + the KV
    cache (when the config builds one), all alive at once (core.batches budgets only KV, prefill attention and 14000 tokens per chunk)."""
    tc = cfg_json.get("text_config", cfg_json)
    nl, hidden, vocab = tc["num_hidden_layers"], tc["hidden_size"], tc["vocab_size"]
    nkv = tc.get("num_key_value_heads") or tc["num_attention_heads"]
    head_dim = tc.get("head_dim") or hidden // tc["num_attention_heads"]
    kv = 2 * nl * nkv * head_dim * 2 if _uses_cache(cfg_json) else 0
    return (nl + 1) * hidden * 2 + vocab * 2 + kv


@pytest.mark.parametrize("key", KEYS)
def test_model_card_hidden_last_headroom_matches_config(key):
    """Regression (reviewer finding): cards said Seed-OSS-36B / OLMo-32B 'fit one 80 GB card' although Ctx.hidden_last cannot run on it
    with the 14000-token chunks core.batches builds. The cards now carry hidden_last_80gb: numbers derived here from the repo's real
    config.json and the card's weights (not hand-typed), the logical relations between its measured and derived parts, and prose that
    agrees with the verdict."""
    card = json.loads((MODELS_DIR / f"{key}.json").read_text())
    hl = card["hidden_last_80gb"]
    cfg_json = json.loads((Path(_snapshot(SPEC[key]["cfg"])) / "config.json").read_text())
    per_tok = _hidden_last_bytes_per_token(cfg_json)
    free = (CARD_GPU_GIB - CARD_OVERHEAD_GIB) * 2 ** 30 - card["weights_gb_bf16"] * 1e9
    assert hl["bytes_per_prompt_token"] == per_tok and hl["kv_cache_alive"] == _uses_cache(cfg_json)
    assert hl["free_gib_after_weights"] == pytest.approx(free / 2 ** 30, abs=0.06)
    assert hl["max_tokens_per_chunk"] == int(free // per_tok)
    assert hl["exceeds_core_token_budget"] == (hl["max_tokens_per_chunk"] < CORE_TOKEN_BUDGET)
    measured = hl["hard_pool_measured"]
    over = measured["chunks_over_budget"]
    assert hl["fits_hard_pool"] == (over["N"] == 0 and over["E"] == 0)
    assert measured["n_prompts"] > 0 and measured["max_prompt_chars"] > 0
    # peak memory is linear in padded tokens, a multi-prompt chunk holds at most 14000 of them, so:
    if not hl["exceeds_core_token_budget"] and measured["longest_prompt_tokens"] <= hl["max_tokens_per_chunk"]:
        assert hl["fits_hard_pool"], "no chunk can exceed the limit when it is above the chunk token budget and above the longest prompt"
    if measured["longest_prompt_tokens"] > hl["max_tokens_per_chunk"]:
        assert not hl["fits_hard_pool"], "a single prompt above the per-chunk limit cannot be processed"
    if not hl["fits_hard_pool"]:                                # the prose must say what the flag says
        assert "hidden_last" in card["fits_single_80gb_detail"] and "H200" in card["recommended_gpu"]
        assert any("hidden_last" in s for s in card["known_issues"])
    assert hl["basis"].startswith("ESTIMATE") and "not a GPU measurement" in hl["basis"]
    # the two keys the review named: Seed-OSS-36B does not fit with 14000-token chunks; a 26k-token prompt exceeds the OLMo-32B limit
    if key.startswith("seed-oss"):
        assert hl["fits_hard_pool"] is False and hl["max_tokens_per_chunk"] < CORE_TOKEN_BUDGET
    if "32b" in key and key.startswith("olmo"):
        assert 26000 > hl["max_tokens_per_chunk"] > CORE_TOKEN_BUDGET and "26k-token" in card["fits_single_80gb_detail"]
