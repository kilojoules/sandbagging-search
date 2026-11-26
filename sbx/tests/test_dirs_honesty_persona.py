"""Tests for sbx.dirs.honesty (recipes "honesty", "honesty_mcq") and sbx.dirs.persona (recipe "persona").

Layers of testing (CPU only, whole file < 90 s):
  * data files   : schema, limits, balance, twin / split structure, builder reproducibility, factual self-consistency of the
                   programmatic tables, answer-in-statement checks, no overlap with any local evaluation question
  * text lists   : >= 8 paraphrases per side (12 + 4 held-out), matched pairs, no leakage between train / held-out
  * planted mocks: a mock Ctx whose hidden states are noise + planted per-layer directions (+ paraphrase artefacts), so the
                   recipes' mathematics (sign, unit norm, recovery, held-out AUROC, cross AUROC, cosines, rotation, caching,
                   determinism, pool-only use of questions) is checked exactly
  * sign         : every direction points toward the HIDING pole (lying, false statement, novice, low effort) like all sbx directions, so
                   the runner's one negative --strengths list and its `cap` act as restoration for all of them (regression of the review
                   finding that TRUTH / EXPERT / EFFORT pointed toward the capable pole): pole of every contrast, steer_sign, the default
                   grid on the planted mock, exact negation of the first version's directions
  * tiny_ctx     : real hooks / prompts / hidden states of a tiny random Llama: shapes, unit norms for every layer, determinism,
                   caching, a steering smoke test
  * real chat templates of cached tokenizers (skipped when not cached): the texts land in the system / developer / user slot
  * docs         : docs/dirs_honesty_persona.md lists every text and the dataset hashes
Run: USE_TF=0 USE_TORCH=1 python3 -m pytest sbx/tests/test_dirs_honesty_persona.py -q
"""
from __future__ import annotations

import collections
import hashlib
import inspect
import json
import re
import zlib
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from sbx import core
from sbx.dirs import honesty as H
from sbx.dirs import persona as P
from sbx.testing import tiny_ctx, toy_questions

ROOT = Path(__file__).resolve().parents[2]
DOCS = ROOT / "docs" / "dirs_honesty_persona.md"
LIE_ALL, HONEST_ALL = set(H.LIE_TRAIN + H.LIE_HELD), set(H.HONEST_TRAIN + H.HONEST_HELD)
EXPERT_ALL, NOVICE_ALL = set(P.EXPERT_TRAIN + P.EXPERT_HELD), set(P.NOVICE_TRAIN + P.NOVICE_HELD)
CAREFUL_ALL, QUICK_ALL = set(P.CAREFUL_TRAIN + P.CAREFUL_HELD), set(P.QUICK_TRAIN + P.QUICK_HELD)
ALL_WRAPPERS = H.WRAPPERS_TRAIN + H.WRAPPERS_HELD


def _unit(v):
    return (v / np.linalg.norm(v)).astype(np.float64)


def _cos(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def make_pool(n_each=40, n_hard=12):
    def relabel(qs, pre):
        return [dict(q, qid=f"{pre}{q['qid']}", source=f"toy_{pre}") for q in qs]
    return {"wmdp": relabel(toy_questions(n_each, seed=1), "w"), "nonbio": relabel(toy_questions(n_each, seed=2), "n"),
            "hard": relabel(toy_questions(n_hard, seed=3), "h")}


def pole_prompts(ctx, pool):
    """Held-out prompts of the HIDING pole and of the CAPABLE pole of every direction, built straight from the text lists and the
    statement labels (NOT through the recipes' pos / neg assignment, so the check is not circular), item-aligned:
    {direction: (hiding prompts, capable prompts)}.  Hiding pole = dishonest / false statement / novice / quick."""
    data = H.load_honesty_data()
    qs = pool["wmdp"][:12] + pool["nonbio"][:12]
    fq, pairs = H.fact_questions(data, "val"), H.fact_pairs(data, "val")                 # pairs: {fact: (true text, false text)}
    fids = sorted(pairs)[:24]

    def mcq(texts):
        return [ctx.prompt(q, core.Framing("t", sys_text=texts[i % len(texts)])) for i, q in enumerate(qs)]

    def chat(texts):
        return [ctx.chat_prompt(fq[f], sys_text=texts[i % len(texts)]) for i, f in enumerate(fids)]

    def statements(label):                                                              # 1: the true statement, 0: its false twin
        return [ctx.chat_prompt(H._wrap(H.WRAPPERS_HELD[i % len(H.WRAPPERS_HELD)], pairs[f][0 if label else 1]))
                for i, f in enumerate(fids)]

    return {"LIE": (chat(H.LIE_HELD), chat(H.HONEST_HELD)), "LIE_mcq": (mcq(H.LIE_HELD), mcq(H.HONEST_HELD)),
            "TRUTH": (statements(0), statements(1)),
            "EXPERT": (mcq(P.NOVICE_HELD), mcq(P.EXPERT_HELD)), "EFFORT": (mcq(P.QUICK_HELD), mcq(P.CAREFUL_HELD))}


class PlantedCtx:
    """Mock Ctx exposing exactly what the recipes use (prompt, chat_prompt, ntok, hidden_last, nl, H, adapter.key, model.config,
    log).  hidden_last = per-prompt seeded noise + a per-item "content" term shared by both sides of an item + a fixed random offset per
    system text (a paraphrase artefact the rotation must average out) + planted signals (layers >= planted_from only).
    CONVENTION (the one of every sbx direction): the HIDING / suppressed pole carries +signal * direction, the capable pole -signal *
    direction, so the planted direction points TOWARD dishonesty / falsehood / novice / low effort:
        lie      : chat prompt, system text from LIE (+) / HONEST (-)           -> +/- signal * lie
        lie_mcq  : MCQ prompt, same texts                                       -> +/- signal * lie_mcq   (cos(lie_mcq, lie) = 0.8)
        truth    : user text = wrapper(FALSE / TRUE statement), no system text  -> +/- signal * truth     (cos(truth, lie) = +0.8)
        expert   : MCQ prompt, system text NOVICE (+) / EXPERT (-)              -> +/- signal * expert
        effort   : MCQ prompt, system text QUICK (+) / CAREFUL (-)              -> +/- signal * effort    (cos(effort, expert) = 0.6)"""

    def __init__(self, nl=6, H_=48, signal=10.0, para_scale=3.0, content_scale=1.5, planted_from=3, seed=0, key="planted"):
        self.nl, self.H = nl, H_
        self.signal, self.para_scale, self.content_scale, self.planted_from = signal, para_scale, content_scale, planted_from
        self.adapter = SimpleNamespace(key=key, answer_mode="direct", max_new=8, sys_role="system", effort=None,
                                       split=lambda t: ("", t))
        self.tok = SimpleNamespace(all_special_tokens=[])
        self.model = SimpleNamespace(device=torch.device("cpu"), config=SimpleNamespace(_name_or_path="planted"))
        self.log = lambda *a, **k: None
        self.calls, self.prompts_seen, self.qids_seen, self.questions_seen = collections.Counter(), [], [], []
        self.nonfinite_layer = None
        data = H.load_honesty_data()
        self.stmt_label = {w.replace("{s}", r["text"]): r["label"] for r in data["statements"] for w in ALL_WRAPPERS}
        rs = np.random.RandomState(seed)

        def orth(v):
            w = rs.randn(H_)
            return _unit(w - (w @ v) * v)

        self.dirs = {k: {} for k in ("lie", "lie_mcq", "truth", "expert", "effort")}
        for L in range(nl + 1):
            lie, expert = _unit(rs.randn(H_)), _unit(rs.randn(H_))
            self.dirs["lie"][L] = lie
            self.dirs["lie_mcq"][L] = 0.8 * lie + 0.6 * orth(lie)
            self.dirs["truth"][L] = 0.8 * lie + 0.6 * orth(lie)           # toward FALSEHOOD, aligned with lying
            self.dirs["expert"][L] = expert
            self.dirs["effort"][L] = 0.6 * expert + 0.8 * orth(expert)

    # ---- prompt rendering (carries the markers hidden_last decodes)
    def prompt(self, q, framing=None):
        self.qids_seen.append(q["qid"])
        self.questions_seen.append((q.get("source"), q["qid"]))
        return f"SYS<{framing.sys_text if framing and framing.sys_text else ''}> MCQ<{q['qid']}>"

    def chat_prompt(self, user_text, sys_text=None, effort=None):
        return f"SYS<{sys_text or ''}> USER<{user_text}>"

    def ntok(self, text):
        return len(text.split())

    @staticmethod
    def _rs(*parts):
        return np.random.RandomState(zlib.crc32("|".join(map(str, parts)).encode()) % 2 ** 32)

    def hidden_last(self, prompts, cap=16):
        self.calls["hidden_last"] += 1
        self.prompts_seen.extend(prompts)
        out = np.empty((len(prompts), self.nl + 1, self.H), np.float16)
        for i, p in enumerate(prompts):
            sys_t, kind, body = re.match(r"SYS<(.*?)> (MCQ|USER)<(.*)>$", p, re.S).groups()
            h = self._rs("noise", p).randn(self.nl + 1, self.H)
            h += self.content_scale * self._rs("content", kind, body).randn(self.nl + 1, self.H)
            concept, sign = None, 0
            if sys_t in LIE_ALL or sys_t in HONEST_ALL:
                concept, sign = ("lie" if kind == "USER" else "lie_mcq"), (1 if sys_t in LIE_ALL else -1)
            elif sys_t in EXPERT_ALL or sys_t in NOVICE_ALL:
                concept, sign = "expert", (1 if sys_t in NOVICE_ALL else -1)                    # hiding pole = novice
            elif sys_t in CAREFUL_ALL or sys_t in QUICK_ALL:
                concept, sign = "effort", (1 if sys_t in QUICK_ALL else -1)                     # hiding pole = quick / low effort
            elif kind == "USER" and body in self.stmt_label:
                concept, sign = "truth", (1 if self.stmt_label[body] == 0 else -1)              # hiding pole = FALSE statement
            if sys_t:
                h += self.para_scale * self._rs("para", sys_t).randn(self.nl + 1, self.H) / np.sqrt(self.H)
            if concept:
                for L in range(self.planted_from, self.nl + 1):
                    h[L] += sign * self.signal * self.dirs[concept][L]
            out[i] = h
            if self.nonfinite_layer is not None:
                out[i, self.nonfinite_layer, 0] = np.inf
        return out


def cfg_for(tmp=None, **kw):
    c = dict(n_pairs=None, max_new=4, seed=0, cache_dir=str(tmp) if tmp else None, layers=None, log=lambda *a, **k: None)
    c.update(kw)
    return c


@pytest.fixture(scope="module")
def data():
    return H.load_honesty_data()


@pytest.fixture(scope="module")
def facts():
    return H.build_facts()


@pytest.fixture(scope="module")
def planted():
    """One ctx and the three recipes run on it (no cache): (ctx, honesty (dirs, meta), honesty_mcq, persona)."""
    ctx, pool = PlantedCtx(), make_pool()
    cfg = cfg_for(n_pairs=60)
    return ctx, H.honesty_recipe(ctx, pool, cfg), H.honesty_mcq_recipe(ctx, pool, cfg), P.persona_recipe(ctx, pool, cfg)


@pytest.fixture(scope="module")
def tiny():
    return tiny_ctx()


PLANTED_LAYERS = (3, 4, 5)
NULL_LAYERS = (1, 2)


# ===================================================================================================== data files
def test_data_files_schema_and_limits(data):
    stm, qs = data["statements"], data["questions"]
    assert 0 < len(stm) <= H.MAX_TRUTH_ITEMS == 600 and len(stm) == 2 * len(qs)
    assert collections.Counter(r["label"] for r in stm) == {0: len(stm) // 2, 1: len(stm) // 2}
    assert len({r["id"] for r in stm}) == len(stm) and len({r["id"] for r in qs}) == len(qs)
    for r in stm:
        assert set(r) == {"id", "fact", "text", "label", "cat", "split", "source"}
        assert r["text"] == r["text"].strip() and r["text"].isascii() and r["text"].endswith(".") and "{" not in r["text"]
    for r in qs:
        assert set(r) == {"id", "fact", "question", "answer", "cat", "split", "source"}
        assert r["question"].endswith("?") and r["question"].isascii() and r["answer"].isascii()
    for fname in (H.TRUTH_FILE, H.QUESTIONS_FILE):            # one JSON object per line, ASCII, trailing newline
        raw = (H.DATA_DIR / fname).read_text(encoding="utf-8")
        assert raw.isascii() and raw.endswith("\n") and "\n\n" not in raw
    assert {r["source"] for r in stm} == {"programmatic", "handwritten"}


def test_data_in_sync_with_builder_and_deterministic(facts):
    assert H.check_data() == []                       # the committed files are exactly what `build` writes
    assert H.render_data() == H.render_data()
    assert H.build_facts() == facts                   # SHA-1 based choices: no dependence on numpy / python hash seeds
    assert 2 * len(facts) <= H.MAX_TRUTH_ITEMS


def test_twins_splits_and_categories(data, facts):
    by = collections.defaultdict(list)
    for r in data["statements"]:
        by[r["fact"]].append(r)
    for fid, rs in by.items():                         # every fact: one true + one false twin, same split / category
        assert sorted(r["label"] for r in rs) == [0, 1] and len({(r["split"], r["cat"]) for r in rs}) == 1
    split = collections.Counter(f["split"] for f in facts)
    assert split["train"] >= 120 and 40 <= split["val"] <= 70 and split["ood"] >= 60
    for f in facts:                                    # OOD categories (and only they) are never trained on
        assert (f["cat"] in H.OOD_CATS) == (f["split"] == "ood")
    for cat in H.IID_CATS:                             # every in-distribution category has train and val facts
        assert {f["split"] for f in facts if f["cat"] == cat} == {"train", "val"}
    assert {f["cat"] for f in facts} == set(H.IID_CATS) | set(H.OOD_CATS)
    assert set(H.CATEGORY_NOTES) == set(H.IID_CATS) | set(H.OOD_CATS)
    # statement length is not a shortcut for the label (twins differ in the object only)
    for sp in ("train", "val", "ood"):
        t = np.mean([len(r["text"]) for r in data["statements"] if r["split"] == sp and r["label"] == 1])
        f_ = np.mean([len(r["text"]) for r in data["statements"] if r["split"] == sp and r["label"] == 0])
        assert abs(t - f_) < 1.5


def test_programmatic_facts_are_self_consistent(facts):
    by_cat = collections.defaultdict(list)
    for f in facts:
        by_cat[f["cat"]].append(f)
    # arithmetic is recomputed independently
    ops = {"plus": lambda a, b: a + b, "minus": lambda a, b: a - b, "times": lambda a, b: a * b}
    for f in by_cat["arithmetic"]:
        a, op, b, r = re.fullmatch(r"(\d+) (plus|minus|times) (\d+) equals (\d+)\.", f["true"]).groups()
        assert ops[op](int(a), int(b)) == int(r) and f["answer"] == r
        wrong = int(re.fullmatch(r"\d+ \w+ \d+ equals (\d+)\.", f["false"]).group(1))
        assert wrong != int(r) and wrong > 0 and f["false"].startswith(f"{a} {op} {b} equals")
    # element tables against an independent ordering of the first 30 elements + a few well-known ones
    order = "H He Li Be B C N O F Ne Na Mg Al Si P S Cl Ar K Ca Sc Ti V Cr Mn Fe Co Ni Cu Zn".split()
    known = {s: i + 1 for i, s in enumerate(order)}
    known.update({"Ag": 47, "Sn": 50, "Au": 79, "Hg": 80, "Pb": 82, "U": 92})
    assert all(known[s] == z for _, s, z in H.ELEMENTS)
    symbols = {s for _, s, _ in H.ELEMENTS}
    sym_of = {e: s for e, s, _ in H.ELEMENTS}
    z_of = {e: z for e, _, z in H.ELEMENTS}
    for f in by_cat["element_symbol"]:
        e, s = re.fullmatch(r"The chemical symbol for (\w+) is (\w+)\.", f["true"]).groups()
        w = re.fullmatch(r"The chemical symbol for \w+ is (\w+)\.", f["false"]).group(1)
        assert sym_of[e] == s and w in symbols and w != s
    for f in by_cat["atomic_number"]:
        e, z = re.fullmatch(r"The atomic number of (\w+) is (\d+)\.", f["true"]).groups()
        w = int(re.fullmatch(r"The atomic number of \w+ is (\d+)\.", f["false"]).group(1))
        assert z_of[e] == int(z) and w != int(z) and w in set(z_of.values())
    cap = dict(H.CAPITALS)
    for f in by_cat["capital"]:
        c, v = re.fullmatch(r"The capital of (.+) is (.+)\.", f["true"]).groups()
        w = re.fullmatch(r"The capital of .+ is (.+)\.", f["false"]).group(1)
        assert cap[c] == v and w in cap.values() and w != v
    units = {(s, b): n for s, b, n, _ in H.UNITS}
    for f in by_cat["unit"]:
        n, s, b = re.fullmatch(r"There are (\d+) (.+) in one (.+)\.", f["true"]).groups()
        w = int(re.fullmatch(r"There are (\d+) .+ in one .+\.", f["false"]).group(1))
        assert units[(s, b)] == int(n) and w != int(n) and w in H.UNIT_DISTRACTORS
    es = dict(H.SPANISH)
    for f in by_cat["translation"]:
        w_es, en = re.fullmatch(r"The Spanish word '(\w+)' means '(\w+)'\.", f["true"]).groups()
        wrong = re.fullmatch(r"The Spanish word '\w+' means '(\w+)'\.", f["false"]).group(1)
        assert es[w_es] == en and wrong in es.values() and wrong != en


def test_answers_are_in_true_statement_only_and_validator_catches_errors(facts):
    for f in facts:
        assert H.has_token(f["true"], f["answer"]) and not H.has_token(f["false"], f["answer"])
        assert f["question"].endswith("?") and len(f["question"].split()) >= 3
    H.validate_facts(facts)
    assert H.has_token("The capital of Peru is Lima.", "lima") and not H.has_token("Limassol", "Lima")
    assert H.has_token("pH of about 7.", "7") and not H.has_token("pH of about 17.", "7")

    def mutated(**kw):
        fs = [dict(f) for f in facts]
        fs[0].update(kw)
        return fs

    for kw, msg in ((dict(false=facts[0]["true"]), "identical"), (dict(false=facts[0]["false"][:-1]), "period"),
                    (dict(answer="Xyzzy"), "not in the true"), (dict(true="The capital of France is Pari" + chr(233) + "."), "non-ASCII"),
                    (dict(question="What is it"), "'\\?'"), (dict(false=facts[0]["false"] + " "), "padded"),
                    (dict(split="ood"), "out-of-distribution")):
        with pytest.raises(ValueError, match=msg):
            H.validate_facts(mutated(**kw))
    with pytest.raises(ValueError, match="duplicate fact id"):
        H.validate_facts(facts + [dict(facts[0])])
    with pytest.raises(ValueError, match="also occurs in the false statement"):
        H.validate_facts(mutated(false="The capital of France is not Paris."))


def test_no_overlap_with_any_local_evaluation_question(data):
    def norm(s):
        return re.sub(r"[^a-z0-9]+", " ", s.lower()).strip()

    files = [f for pattern in ("*.jsonl", "hard/*.jsonl", "evals/*.jsonl") for f in sorted((ROOT / "data").glob(pattern))]
    if not files:
        pytest.skip("no local evaluation question files")
    blobs = []
    for f in files:                                            # every MCQ file under data/, data/hard, data/evals (all splits)
        for line in f.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                if "question" in r:
                    blobs.append(norm(str(r["question"])) + " | " + " | ".join(norm(str(c)) for c in r.get("choices", [])))
    assert blobs or not files
    mine = [norm(r["text"]) for r in data["statements"]] + [norm(r["question"]) for r in data["questions"]]
    big = "\n".join(blobs)                      # normalised texts contain no newline, so a match cannot straddle two questions
    hits = [m for m in mine if m in big]
    assert not hits, f"{len(hits)} honesty texts occur inside an evaluation question: {hits[:3]}"


def test_load_honesty_data_validation(tmp_path):
    for f in (H.TRUTH_FILE, H.QUESTIONS_FILE):
        (tmp_path / f).write_text((H.DATA_DIR / f).read_text(encoding="utf-8"), encoding="utf-8")
    assert H.load_honesty_data(tmp_path)["statements"]
    lines = (tmp_path / H.TRUTH_FILE).read_text(encoding="utf-8").splitlines()

    def put(new_lines, fname=H.TRUTH_FILE):
        (tmp_path / fname).write_text("\n".join(new_lines) + "\n", encoding="utf-8")

    put(lines[1:])                                                  # first fact lost its true statement
    with pytest.raises(ValueError, match="exactly one true and one false"):
        H.load_honesty_data(tmp_path)
    rec = json.loads(lines[0])
    put([json.dumps(dict(rec, label=2))] + lines[1:])
    with pytest.raises(ValueError, match="label"):
        H.load_honesty_data(tmp_path)
    put([json.dumps(dict(rec, split="val"))] + lines[1:])          # twin in a different split
    with pytest.raises(ValueError, match="same split"):
        H.load_honesty_data(tmp_path)
    bad = {k: v for k, v in rec.items() if k != "cat"}
    put([json.dumps(bad)] + lines[1:])
    with pytest.raises(ValueError, match="lacks fields"):
        H.load_honesty_data(tmp_path)
    put(lines[:5] + ["{not json}"] + lines[6:])
    with pytest.raises(ValueError, match="invalid JSON"):
        H.load_honesty_data(tmp_path)
    put(lines + [json.dumps(dict(rec, id=f"extra{i}")) for i in range(H.MAX_TRUTH_ITEMS)])
    with pytest.raises(ValueError, match="exceed the limit"):
        H.load_honesty_data(tmp_path)
    put(lines)
    qlines = (tmp_path / H.QUESTIONS_FILE).read_text(encoding="utf-8").splitlines()
    put(qlines[1:], H.QUESTIONS_FILE)                              # a fact without a question
    with pytest.raises(ValueError, match="different facts"):
        H.load_honesty_data(tmp_path)
    q0 = json.loads(qlines[0])
    put([json.dumps(dict(q0, split="ood"))] + qlines[1:], H.QUESTIONS_FILE)
    with pytest.raises(ValueError, match="different splits"):
        H.load_honesty_data(tmp_path)
    (tmp_path / H.QUESTIONS_FILE).unlink()
    with pytest.raises(FileNotFoundError):
        H.load_honesty_data(tmp_path)


def test_fact_accessors(data):
    tr, va, oo = H.fact_pairs(data, "train"), H.fact_pairs(data, "val"), H.fact_pairs(data, "ood")
    assert not set(tr) & set(va) and not set(tr) & set(oo) and not set(va) & set(oo)
    assert list(tr) == sorted(tr)
    for fid, (t, f) in list(tr.items())[:20]:
        assert t != f
    assert set(H.fact_questions(data, "train")) == set(tr) and set(H.fact_questions(data, "ood")) == set(oo)


def test_builder_cli(tmp_path, capsys):
    assert H.main(["build", "--check"]) == 0
    assert H.main(["build", "--out", str(tmp_path)]) == 0
    for f in (H.TRUTH_FILE, H.QUESTIONS_FILE):
        assert (tmp_path / f).read_text(encoding="utf-8") == (H.DATA_DIR / f).read_text(encoding="utf-8")
    assert H.main(["build", "--check", "--out", str(tmp_path)]) == 0
    p = tmp_path / H.TRUTH_FILE
    p.write_text(p.read_text(encoding="utf-8").replace("Paris", "Pariz", 1), encoding="utf-8")
    assert H.check_data(tmp_path) == [H.TRUTH_FILE] and H.main(["build", "--check", "--out", str(tmp_path)]) == 1
    assert "out of sync" in capsys.readouterr().out
    (tmp_path / H.QUESTIONS_FILE).unlink()
    assert sorted(H.check_data(tmp_path)) == sorted([H.TRUTH_FILE, H.QUESTIONS_FILE])


# ===================================================================================================== text lists
@pytest.mark.parametrize("pos,neg,pos_h,neg_h", [
    (H.LIE_TRAIN, H.HONEST_TRAIN, H.LIE_HELD, H.HONEST_HELD),
    (P.EXPERT_TRAIN, P.NOVICE_TRAIN, P.EXPERT_HELD, P.NOVICE_HELD),
    (P.CAREFUL_TRAIN, P.QUICK_TRAIN, P.CAREFUL_HELD, P.QUICK_HELD)])
def test_text_lists(pos, neg, pos_h, neg_h):
    assert len(pos) == len(neg) >= 8 and len(pos_h) == len(neg_h) >= 4          # >= 8 training paraphrases per side
    allt = list(pos) + list(neg) + list(pos_h) + list(neg_h)
    assert len(set(allt)) == len(allt)                                          # no duplicates, no overlap between sides / splits
    for t in allt:
        assert t == t.strip() and t.isascii() and 20 < len(t) < 200 and "{" not in t
    # matched pairs: lengths of the two sides stay within a factor 1.35 of each other
    for a, b in list(zip(pos, neg)) + list(zip(pos_h, neg_h)):
        assert 1 / 1.35 < len(a) / len(b) < 1.35, (a, b)
    assert abs(np.mean([len(t) for t in pos]) - np.mean([len(t) for t in neg])) < 8


def test_wrappers():
    assert len(H.WRAPPERS_TRAIN) >= 8 and len(H.WRAPPERS_HELD) >= 2
    assert len(set(ALL_WRAPPERS)) == len(ALL_WRAPPERS)
    for w in ALL_WRAPPERS:
        assert w.count("{s}") == 1 and w.isascii()
        assert H._wrap(w, "X.") == w.replace("{s}", "X.")
    assert H.STEER_SIGN == {"LIE": -1, "TRUTH": -1, "LIE_mcq": -1} and P.STEER_SIGN == {"EXPERT": -1, "EFFORT": -1}
    assert set(H.SIGN_CONVENTION) == set(H.STEER_SIGN) and set(P.SIGN_CONVENTION) == set(P.STEER_SIGN)


def test_recipes_registered_without_collisions():
    assert set(H.RECIPES) == {"honesty", "honesty_mcq"} and set(P.RECIPES) == {"persona"}
    assert not set(H.RECIPES) & set(P.RECIPES)
    assert all(callable(f) for f in list(H.RECIPES.values()) + list(P.RECIPES.values()))


# ===================================================================================================== small helpers
def test_balanced_schedule_and_subsample():
    for n, p in ((12, 12), (100, 12), (7, 12), (0, 3), (1000, 8)):
        s = H.balanced_schedule(n, p, H.rs_for(0, "t", n))
        assert len(s) == n and set(s) <= set(range(p))
        c = np.bincount(s, minlength=p)
        assert c.max() - c.min() <= 1
    assert list(H.balanced_schedule(50, 12, H.rs_for(1, "a"))) == list(H.balanced_schedule(50, 12, H.rs_for(1, "a")))
    assert list(H.balanced_schedule(50, 12, H.rs_for(1, "a"))) != list(H.balanced_schedule(50, 12, H.rs_for(2, "a")))
    with pytest.raises(ValueError):
        H.balanced_schedule(5, 0, H.rs_for(0))
    ids = [f"f{i:03d}" for i in range(50)]
    sub = H.subsample(ids, 10, H.rs_for(0, "x"))
    assert sub == sorted(sub) and len(sub) == 10 and set(sub) <= set(ids) and sub == H.subsample(ids[::-1], 10, H.rs_for(0, "x"))
    assert H.subsample(ids, None, H.rs_for(0)) == ids and H.subsample(ids, 99, H.rs_for(0)) == ids
    assert H.n_pairs_from({"n_pairs": None}) is None and H.n_pairs_from({"n_pairs": None}, 400) == 400
    assert H.n_pairs_from({"n_pairs": "600"}) == 600
    with pytest.raises(ValueError, match="n_pairs must be"):
        H.n_pairs_from({"n_pairs": H.MIN_PAIRS - 1})


def test_separation_by_hand():
    # one layer, H=2, direction e1; positives score [3, 1, 2], negatives [0, 2, 4]
    d = {1: np.array([1.0, 0.0], np.float32)}
    rp = np.array([[[3, 9]], [[1, 9]], [[2, 9]]], np.float16)
    rn = np.array([[[0, 9]], [[2, 9]], [[4, 9]]], np.float16)
    aur, pair = H.separation(d, [1], rp, rn)
    assert aur[1] == pytest.approx((4 + 0.5) / 9)                       # wins: 3>0 3>2 1>0 2>0; tie: 2==2; the other 4 pairs lose
    assert pair[1] == pytest.approx(1 / 3)                              # only item 0 has pos > neg; 1<2 and 2<4
    rp4 = np.concatenate([rp, np.array([[[5, 0]]], np.float16)])                  # 4 items: the masks need >= MIN_SUBSET of them
    rn4 = np.concatenate([rn, np.array([[[1, 0]]], np.float16)])
    full, _ = H.separation(d, [1], rp4, rn4)
    assert H.separation(d, [1], rp4, rn4, mask=[True] * 4)[0] == full
    sub, sub_pair = H.separation(d, [1], rp4, rn4, mask=[True, True, True, True])
    assert H.separation(d, [1], rp4, rn4, mask=[True, True, True, False]) == (None, None)    # fewer than MIN_SUBSET items
    assert H.separation(d, [1], rp, rn, mask=[True, True, True]) == (None, None)
    assert H.MIN_SUBSET == 4 and sub[1] == full[1] and sub_pair[1] == pytest.approx(0.5)


def test_unit_directions_degenerate_and_exact():
    m = np.random.RandomState(0).randn(3, 8).astype(np.float32)
    n = np.random.RandomState(1).randn(3, 8).astype(np.float32)
    d, raw = H.unit_directions(m, n, [5.0, 5.0, 5.0], [2, 4, 6], "X")
    for i, L in enumerate([2, 4, 6]):
        assert d[L].dtype == np.float32 and abs(np.linalg.norm(d[L]) - 1) < 1e-6
        assert raw[L] == pytest.approx(np.linalg.norm(m[i].astype(np.float64) - n[i].astype(np.float64)))
    with pytest.raises(ValueError, match="degenerate"):
        H.unit_directions(m, m.copy(), [5.0] * 3, [2, 4, 6], "X")
    bad = m.copy()
    bad[1, 0] = np.nan
    with pytest.raises(ValueError, match="degenerate"):
        H.unit_directions(bad, n, [5.0] * 3, [2, 4, 6], "X")


def test_weak_direction_warnings():
    layers = [1, 2]
    rep = {"A": {"auroc": {"s": {1: 0.55, 2: 0.58}}}, "B": {"auroc": {"s": {1: 0.55, 2: 0.9}}}}
    w = H.weak_direction_warnings(rep, {"A": ["s"], "B": ["s"]}, layers)
    assert len(w) == 1 and w[0].startswith("A:") and "0.580" in w[0] and "uninformative" in w[0]
    assert H.weak_direction_warnings(rep, {"B": ["s"]}, layers) == []
    assert H.weak_direction_warnings(rep, {"A": ["s"]}, layers, threshold=0.5) == []
    nan = {"C": {"auroc": {"s": {1: float("nan"), 2: float("nan")}}}}
    assert len(H.weak_direction_warnings(nan, {"C": ["s"]}, layers)) == 1                     # NaN is never "good enough"


def test_jsonable():
    x = {1: np.float32(0.5), "b": [np.int64(2), (3, 4)], "c": np.array([1.0, np.nan]), "d": float("inf"), "e": np.bool_(True), "f": None}
    assert json.loads(json.dumps(H.jsonable(x))) == {"1": 0.5, "b": [2, [3, 4]], "c": [1.0, None], "d": None, "e": True, "f": None}
    with pytest.raises(TypeError):
        H.jsonable({"x": object()})


def test_resolve_layers():
    ctx = PlantedCtx(nl=6)
    assert H.resolve_layers(ctx, {}) == [1, 2, 3, 4, 5]
    assert H.resolve_layers(ctx, {"layers": [4, 2, 2, 6]}) == [2, 4, 6]
    for bad in ([0], [7], []):
        with pytest.raises(ValueError):
            H.resolve_layers(ctx, {"layers": bad})


# ===================================================================================================== MCQ selection
def test_select_mcqs_pool_only_disjoint_and_stable():
    pool = make_pool(40, 12)
    allowed = {q["qid"] for q in pool["wmdp"] + pool["nonbio"]}
    sel = H.select_mcqs(pool, 30, 10, seed=0)
    tr, va = {q["qid"] for q in sel["train"]}, {q["qid"] for q in sel["val"]}
    assert len(sel["train"]) == 30 and len(sel["val"]) == 10 and not tr & va and tr | va <= allowed
    assert not (tr | va) & {q["qid"] for q in pool["hard"]}                   # pool["hard"] is never touched
    assert sel["train_src"].count("wmdp") == 15 and sel["val_src"].count("nonbio") == 5          # balanced halves
    for qs, srcs in ((sel["train"], sel["train_src"]), (sel["val"], sel["val_src"])):
        assert all((q["source"] == "toy_w") == (s == "wmdp") for q, s in zip(qs, srcs))
    # held-out questions do not depend on n_train, nor on the input order of the pool
    assert [q["qid"] for q in H.select_mcqs(pool, 60, 10, seed=0)["val"]] == [q["qid"] for q in sel["val"]]
    shuffled = {k: list(reversed(v)) for k, v in pool.items()}
    again = H.select_mcqs(shuffled, 30, 10, seed=0)
    assert [q["qid"] for q in again["train"]] == [q["qid"] for q in sel["train"]]
    assert [q["qid"] for q in H.select_mcqs(pool, 30, 10, seed=1)["train"]] != [q["qid"] for q in sel["train"]]
    # a short pool is topped up from the other one
    short = {"wmdp": pool["wmdp"], "nonbio": pool["nonbio"][:4]}
    s2 = H.select_mcqs(short, 30, 10, seed=0)
    assert len(s2["train"]) == 30 and s2["val_src"].count("nonbio") == 4 and s2["train_src"].count("nonbio") == 0
    only = H.select_mcqs({"wmdp": pool["wmdp"]}, 20, 8, seed=0)
    assert set(only["train_src"]) == {"wmdp"}
    # errors
    with pytest.raises(ValueError, match="more than once"):
        H.select_mcqs({"wmdp": pool["wmdp"], "nonbio": pool["wmdp"]}, 20, 8, seed=0)
    with pytest.raises(ValueError, match="pool too small"):
        H.select_mcqs({"wmdp": pool["wmdp"][:10]}, 20, 8, seed=0)
    with pytest.raises(ValueError, match="pool too small"):
        H.select_mcqs({}, 20, 8, seed=0)


def test_mcq_selection_warnings():
    warns = []
    sel, counts = H.mcq_selection_and_meta({"wmdp": make_pool(30)["wmdp"], "nonbio": []}, dict(n_pairs=12), 0, warns)
    assert counts["train_wmdp"] == len(sel["train"]) == 12 and counts["train_nonbio"] == 0 and counts["heldout_questions"] == 8
    assert any("unbalanced" in w for w in warns)
    warns = []
    sel, counts = H.mcq_selection_and_meta({"wmdp": make_pool(30)["wmdp"][:25], "nonbio": []}, dict(n_pairs=40), 0, warns)
    assert len(sel["train"]) + len(sel["val"]) == 25 and counts["heldout_questions"] == 8     # held-out = max(8, pool // 5)
    assert any("training MCQs available" in w for w in warns)


def test_mcq_sets_rotation_matching_and_heldout_separation():
    ctx, pool = PlantedCtx(), make_pool()
    sel = H.select_mcqs(pool, 60, 16, seed=0)
    for name, texts in P.CONTRASTS.items():
        tp, tn, ev = H.mcq_sets(ctx, name, texts, sel, 0)
        parse = lambda p: re.match(r"SYS<(.*?)> MCQ<(.*)>$", p, re.S).groups()      # noqa: E731
        kp = [texts["pos"].index(parse(p)[0]) for p in tp]
        kn = [texts["neg"].index(parse(p)[0]) for p in tn]
        assert kp == kn                                                            # matched pair on both sides of an item
        c = np.bincount(kp, minlength=len(texts["pos"]))
        assert len(c) == 12 and c.min() >= 4 and c.max() - c.min() <= 1             # balanced rotation: 60 / 12 = 5 each
        assert [parse(p)[1] for p in tp] == [q["qid"] for q in sel["train"]] == [parse(p)[1] for p in tn]
        assert not any(parse(p)[0] in texts["pos_held"] + texts["neg_held"] for p in tp + tn)       # held-out texts never fitted
        (vp, vn), (hp, hn) = ev["newq"], ev["newq_newpara"]
        assert all(parse(p)[0] in texts["pos"] for p in vp) and all(parse(p)[0] in texts["neg"] for p in vn)
        assert all(parse(p)[0] in texts["pos_held"] for p in hp) and all(parse(p)[0] in texts["neg_held"] for p in hn)
        vq = [q["qid"] for q in sel["val"]]
        assert [parse(p)[1] for p in vp] == vq == [parse(p)[1] for p in hn]
        assert not set(vq) & {parse(p)[1] for p in tp}                              # held-out questions are not training questions
    with pytest.raises(ValueError, match="equal length"):
        H.mcq_sets(ctx, "X", dict(pos=("a", "b"), neg=("a",), pos_held=("a",), neg_held=("b",)), sel, 0)


def test_truth_and_lie_sets_structure(data):
    ctx = PlantedCtx()
    tr = H.fact_pairs(data, "train")
    facts_train = sorted(tr)[:60]
    tp, tn, ev, n = H._truth_sets(ctx, data, facts_train, 0)
    parse = lambda p: re.match(r"SYS<> USER<(.*)>$", p, re.S).group(1)                 # noqa: E731

    def wrapper_of(user):
        hits = [i for i, w in enumerate(ALL_WRAPPERS)
                if user.startswith(w.split("{s}")[0]) and user.endswith(w.split("{s}")[1])]
        assert len(hits) == 1, user
        return hits[0]

    kp = [wrapper_of(parse(p)) for p in tp]
    assert kp == [wrapper_of(parse(p)) for p in tn]                                   # a twin pair shares its wrapper
    assert max(kp) < len(H.WRAPPERS_TRAIN)
    c = np.bincount(kp, minlength=8)
    assert c.max() - c.min() <= 1
    for f, p, q in zip(facts_train, tp, tn):                    # pos = the FALSE twin (hiding pole), neg = the true statement
        t, fl = tr[f]
        assert H._wrap(H.WRAPPERS_TRAIN[wrapper_of(parse(p))], fl) == parse(p)
        assert H._wrap(H.WRAPPERS_TRAIN[wrapper_of(parse(q))], t) == parse(q)
    assert set(ev) == {"val", "val_newwrap", "ood"} and n == {"val": 54, "ood": 82}
    assert all(wrapper_of(parse(p)) < 8 for p in ev["val"][0] + ev["ood"][0])
    assert all(wrapper_of(parse(p)) >= 8 for p in ev["val_newwrap"][0] + ev["val_newwrap"][1])  # held-out wrappers only
    val_texts = {t for pair in H.fact_pairs(data, "val").values() for t in pair}
    assert not any(t in " ".join(tp + tn) for t in val_texts)                          # no held-out statement is fitted
    ood_texts = {t for pair in H.fact_pairs(data, "ood").values() for t in pair}
    assert not any(t in " ".join(tp + tn) for t in ood_texts)
    lp, ln, lev, nh = H._lie_sets(ctx, data, facts_train, 0)
    qs = H.fact_questions(data, "train")
    assert [re.match(r"SYS<(.*?)> USER<(.*)>$", p, re.S).group(2) for p in lp] == [qs[f] for f in facts_train]
    assert nh == 64 and set(lev) == {"newq", "newq_newpara"}
    train_q = " ".join(lp + ln)
    held = {re.match(r"SYS<(.*?)> USER<(.*)>$", p, re.S).group(2) for p in lev["newq"][0]}
    assert len(held) == 64 and not any(q in train_q for q in held)                     # held-out questions are not trained on
    assert not any(t in train_q for t in H.LIE_HELD + H.HONEST_HELD)


# ===================================================================================================== planted recovery
def test_honesty_recovery_signs_and_unit_norms(planted):
    ctx, (dirs, meta), _, _ = planted
    assert set(dirs) == {"LIE", "TRUTH"}
    for name, key in (("LIE", "lie"), ("TRUTH", "truth")):
        assert set(dirs[name]) == {1, 2, 3, 4, 5}
        for L, v in dirs[name].items():
            assert v.dtype == np.float32 and v.shape == (ctx.H,) and abs(np.linalg.norm(v) - 1) < 1e-5
        for L in PLANTED_LAYERS:                                                 # points TOWARD lying / falsehood (hiding pole)
            assert _cos(dirs[name][L], ctx.dirs[key][L]) > 0.97, (name, L)
    for L in PLANTED_LAYERS:                                                     # planted cos(LIE, TRUTH) = +0.8: both toward the hiding pole
        assert meta["cos_LIE_TRUTH"][str(L)] == pytest.approx(_cos(dirs["LIE"][L], dirs["TRUTH"][L]))
        assert meta["cos_LIE_TRUTH"][str(L)] == pytest.approx(0.8, abs=0.08)


def test_honesty_heldout_auroc_and_cross_checks(planted):
    ctx, (dirs, meta), _, _ = planted
    for L in PLANTED_LAYERS:
        for name, sets in (("LIE", ("newq", "newq_newpara")), ("TRUTH", ("val", "val_newwrap", "ood"))):
            for s in sets:
                assert meta["auroc"][name][s][str(L)] > 0.97, (name, s, L)
                assert meta["pair_acc"][name][s][str(L)] > 0.97
        for s in ("val", "val_newwrap", "ood"):                       # LIE scores false > true  (LIE ~ TRUTH: both toward the hiding pole)
            assert meta["auroc_cross"]["LIE_on_truth"][s][str(L)] > 0.9
        for s in ("newq", "newq_newpara"):                            # TRUTH scores dishonest-instructed > honest
            assert meta["auroc_cross"]["TRUTH_on_lie"][s][str(L)] > 0.9
    for L in NULL_LAYERS:                                             # no signal planted: unseen paraphrases / wrappers -> chance
        assert 0.25 < meta["auroc"]["LIE"]["newq_newpara"][str(L)] < 0.75
        for s in ("val", "val_newwrap", "ood"):
            assert 0.25 < meta["auroc"]["TRUTH"][s][str(L)] < 0.75


def test_honesty_meta_schema_and_json(planted):
    ctx, (dirs, meta), _, _ = planted
    assert json.loads(json.dumps(meta)) == meta
    assert meta["recipe"] == "honesty" and meta["layers"] == [1, 2, 3, 4, 5] and meta["adapter_key"] == "planted"
    assert meta["steer_sign"] == {"LIE": -1, "TRUTH": -1} and set(meta["sign_convention"]) == {"LIE", "TRUTH"}
    assert "LYING" in meta["sign_convention"]["LIE"] and "FALSEHOOD" in meta["sign_convention"]["TRUTH"]
    assert all("NEGATIVE" in meta["sign_convention"][k] for k in ("LIE", "TRUTH"))
    c = meta["counts"]
    assert c["train_facts"] == c["lie_train_questions"] == c["truth_train_pairs"] == meta["n_pairs_used"] == 60
    assert c["lie_paraphrases_train"] == 12 and c["lie_paraphrases_heldout"] == 4 and c["wrappers_train"] == 8
    assert c["truth_val_pairs"] == 54 and c["truth_ood_pairs"] == 82 and c["lie_heldout_questions"] == 64
    for f in (H.TRUTH_FILE, H.QUESTIONS_FILE):
        assert meta["datasets"][f]["sha256"] == hashlib.sha256((H.DATA_DIR / f).read_bytes()).hexdigest()
    assert meta["datasets"][H.TRUTH_FILE]["n"] == 588
    for k in ("diff_norm", "diff_norm_rel", "mean_prompt_tokens", "auroc", "pair_acc"):
        assert set(meta[k]) == {"LIE", "TRUTH"}
    assert set(meta["auroc_cross"]) == {"LIE_on_truth", "TRUTH_on_lie"}
    assert set(meta["auroc_cross"]["LIE_on_truth"]) == {"val", "val_newwrap", "ood"}
    assert set(meta["auroc_cross"]["TRUTH_on_lie"]) == {"newq", "newq_newpara"}
    assert set(meta["auroc"]["TRUTH"]) == {"val", "val_newwrap", "ood"} and set(meta["auroc"]["LIE"]) == {"newq", "newq_newpara"}
    assert set(meta["auroc"]["LIE"]["newq"]) == {"1", "2", "3", "4", "5"}
    assert meta["cos_B"] is None and meta["warnings"] == []
    # natural alpha: raw difference / mean residual norm (planted: 2 * signal over noise norm ~ sqrt(H * (1 + 1.5^2)))
    assert 0.5 < meta["diff_norm_rel"]["LIE"]["4"] < 2.0
    assert meta["mean_prompt_tokens"]["LIE"]["pos"] > 0


def test_honesty_mcq_recovery(planted):
    ctx, _, (dirs, meta), _ = planted
    assert set(dirs) == {"LIE_mcq"}
    for L in PLANTED_LAYERS:
        assert _cos(dirs["LIE_mcq"][L], ctx.dirs["lie_mcq"][L]) > 0.97
        for s in ("newq", "newq_newpara"):
            assert meta["auroc"]["LIE_mcq"][s][str(L)] > 0.97
            assert meta["auroc_by_source"]["LIE_mcq"][s]["wmdp"][str(L)] > 0.97
            assert meta["auroc_by_source"]["LIE_mcq"][s]["nonbio"][str(L)] > 0.97
    for v in dirs["LIE_mcq"].values():
        assert v.dtype == np.float32 and abs(np.linalg.norm(v) - 1) < 1e-5
    assert meta["recipe"] == "honesty_mcq" and meta["steer_sign"] == {"LIE_mcq": -1}
    assert meta["counts"]["train_questions"] == 60 and meta["counts"]["heldout_questions"] == 15
    assert json.loads(json.dumps(meta)) == meta


def test_persona_recovery_signs_and_cross_checks(planted):
    ctx, _, _, (dirs, meta) = planted
    assert set(dirs) == {"EXPERT", "EFFORT"}
    for name, key in (("EXPERT", "expert"), ("EFFORT", "effort")):
        assert set(dirs[name]) == {1, 2, 3, 4, 5}
        for L, v in dirs[name].items():
            assert v.dtype == np.float32 and v.shape == (ctx.H,) and abs(np.linalg.norm(v) - 1) < 1e-5
        for L in PLANTED_LAYERS:
            assert _cos(dirs[name][L], ctx.dirs[key][L]) > 0.97, (name, L)           # points TOWARD novice / quick (hiding pole)
            for s in ("newq", "newq_newpara"):
                assert meta["auroc"][name][s][str(L)] > 0.97 and meta["pair_acc"][name][s][str(L)] > 0.97
                for src in ("wmdp", "nonbio"):
                    assert meta["auroc_by_source"][name][s][src][str(L)] > 0.97
    for L in PLANTED_LAYERS:                                                         # planted cos(EXPERT, EFFORT) = 0.6
        assert meta["cos_EXPERT_EFFORT"][str(L)] == pytest.approx(0.6, abs=0.08)
        for k in ("EXPERT_on_EFFORT", "EFFORT_on_EXPERT"):
            for s in ("newq", "newq_newpara"):
                assert meta["auroc_cross"][k][s][str(L)] > 0.9
    for L in NULL_LAYERS:
        for name in ("EXPERT", "EFFORT"):
            assert 0.1 < meta["auroc"][name]["newq_newpara"][str(L)] < 0.9


def test_persona_meta_schema(planted):
    ctx, _, _, (dirs, meta) = planted
    assert json.loads(json.dumps(meta)) == meta
    assert meta["recipe"] == "persona" and meta["steer_sign"] == {"EXPERT": -1, "EFFORT": -1}
    assert set(meta["sign_convention"]) == {"EXPERT", "EFFORT"}
    assert "NOVICE" in meta["sign_convention"]["EXPERT"].upper() and "NEGATIVE" in meta["sign_convention"]["EXPERT"]
    assert "LOW" in meta["sign_convention"]["EFFORT"].upper() and "NEGATIVE" in meta["sign_convention"]["EFFORT"]
    c = meta["counts"]
    assert c["train_questions"] == meta["n_pairs_used"] == 60 and c["train_wmdp"] == c["train_nonbio"] == 30
    assert c["heldout_questions"] == 15 and c["heldout_wmdp"] + c["heldout_nonbio"] == 15
    assert c["paraphrase_pairs_train"] == 12 and c["paraphrase_pairs_heldout"] == 4
    assert meta["warnings"] == [] and meta["cos_B"] is None
    assert set(meta["diff_norm_rel"]) == {"EXPERT", "EFFORT"} and set(meta["mean_prompt_tokens"]["EXPERT"]) == {"pos", "neg"}


def test_sign_flips_with_the_planted_pole():
    ctx = PlantedCtx(signal=-10.0)                                  # the positive pole now carries the NEGATIVE planted direction
    dirs, _ = P.persona_recipe(ctx, make_pool(), cfg_for(n_pairs=30))
    hd, _ = H.honesty_recipe(ctx, {}, cfg_for(n_pairs=30))
    for L in PLANTED_LAYERS:
        assert _cos(dirs["EXPERT"][L], ctx.dirs["expert"][L]) < -0.97 and _cos(hd["LIE"][L], ctx.dirs["lie"][L]) < -0.97
        assert _cos(hd["TRUTH"][L], ctx.dirs["truth"][L]) < -0.97


# ----------------------------------------------------------------------------------------------- sign convention (regression)
# Every sbx direction points toward the SUPPRESSED / hiding pole (sandbag B, REF, EA_*, LIE ...): the runner applies one
# --strengths list (default negative) and one cap / proj0 / add grid to every --dir-names entry, so a negative alpha and the
# one-sided `cap` only act as capability restoration if the direction points toward lying / falsehood / novice / low effort.
# TRUTH, EXPERT and EFFORT used to point toward the CAPABLE pole: `cap` then removed capability and negative strengths pushed away
# from it, so no default grid cell tested the restoration hypothesis for them.
ALL_DIRECTIONS = ("LIE", "TRUTH", "LIE_mcq", "EXPERT", "EFFORT")


def test_positive_pole_of_every_contrast_is_the_hiding_pole():
    ctx, pool, data = PlantedCtx(), make_pool(), H.load_honesty_data()
    parse = lambda p: re.match(r"SYS<(.*?)> (?:MCQ|USER)<(.*)>$", p, re.S).groups()           # noqa: E731 (system text, body)
    sel = H.select_mcqs(pool, 24, 8, seed=0)
    for name, hiding, capable in (("EXPERT", NOVICE_ALL, EXPERT_ALL), ("EFFORT", QUICK_ALL, CAREFUL_ALL)):
        tp, tn, ev = H.mcq_sets(ctx, name, P.CONTRASTS[name], sel, 0)
        for pos, neg in [(tp, tn)] + list(ev.values()):                  # training and every held-out set: pos = novice / quick
            assert {parse(p)[0] for p in pos} <= hiding and {parse(p)[0] for p in neg} <= capable, name
    facts_train = sorted(H.fact_pairs(data, "train"))[:24]
    tp, tn, ev, _ = H._truth_sets(ctx, data, facts_train, 0)
    for pos, neg in [(tp, tn)] + list(ev.values()):                       # TRUTH: pos = FALSE statements, neg = TRUE statements
        assert {ctx.stmt_label[parse(p)[1]] for p in pos} == {0} and {ctx.stmt_label[parse(p)[1]] for p in neg} == {1}
    lp, ln, lev, _ = H._lie_sets(ctx, data, facts_train, 0)
    for pos, neg in [(lp, ln)] + list(lev.values()):                      # LIE: pos = dishonest-instructed, neg = honest-instructed
        assert {parse(p)[0] for p in pos} <= LIE_ALL and {parse(p)[0] for p in neg} <= HONEST_ALL
    for name in ("EXPERT", "EFFORT"):                                     # the contrast tables themselves, no prompts involved
        assert P.CONTRASTS[name]["pos"] == {"EXPERT": P.NOVICE_TRAIN, "EFFORT": P.QUICK_TRAIN}[name]
        assert P.CONTRASTS[name]["neg"] == {"EXPERT": P.EXPERT_TRAIN, "EFFORT": P.CAREFUL_TRAIN}[name]


def test_flip_changes_only_the_sign_of_the_first_versions_directions():
    """The first version built TRUTH (pos = true statements), EXPERT (pos = expert texts) and EFFORT (pos = careful texts), i.e. toward the
    CAPABLE pole.  Rebuilt here independently from the data and the text lists (same prompts, same rotation), those directions are
    bit-exactly MINUS the recipes' directions, and the held-out AUROC / paired accuracy do not change when direction and positives flip
    together.  Pins that the fix changes the sign and nothing else (so the first version's real-model measurements convert by sign)."""
    ctx, pool, data = PlantedCtx(), make_pool(), H.load_honesty_data()
    layers = [2, 3, 4, 5]
    facts_train = sorted(H.fact_pairs(data, "train"))[:60]
    tr = H.fact_pairs(data, "train")                                                           # {fact: (true text, false text)}
    sched = H.balanced_schedule(len(facts_train), len(H.WRAPPERS_TRAIN), H.rs_for(0, "truth", "train"))

    def statements(which):                                                                     # 0: true statements, 1: false twins
        return [ctx.chat_prompt(H._wrap(H.WRAPPERS_TRAIN[int(k)], tr[f][which])) for f, k in zip(facts_train, sched)]

    tp, tn, ev, _ = H._truth_sets(ctx, data, facts_train, 0)
    fits = [(H.run_contrast(ctx, "TRUTH", tp, tn, ev, layers, None, print),
             H.run_contrast(ctx, "TRUTH", statements(0), statements(1), {}, layers, None, print))]    # old: pos = true = capable pole
    sel = H.select_mcqs(pool, 60, 16, seed=0)
    old_texts = {"EXPERT": dict(pos=P.EXPERT_TRAIN, neg=P.NOVICE_TRAIN, pos_held=P.EXPERT_HELD, neg_held=P.NOVICE_HELD),
                 "EFFORT": dict(pos=P.CAREFUL_TRAIN, neg=P.QUICK_TRAIN, pos_held=P.CAREFUL_HELD, neg_held=P.QUICK_HELD)}
    for name, texts in old_texts.items():
        fits.append((H.run_contrast(ctx, name, *H.mcq_sets(ctx, name, P.CONTRASTS[name], sel, 0), layers, None, print),
                     H.run_contrast(ctx, name, *H.mcq_sets(ctx, name, texts, sel, 0), layers, None, print)))
    assert len(fits) == 3
    for new, old in fits:
        for L in layers:
            assert np.array_equal(new["dirs"][L], -old["dirs"][L]), (new["name"], L)
            assert new["diff_norm"][L] == old["diff_norm"][L]
        for s, (rp, rn) in new["raw"].items():                          # new: positives = hiding pole; old: positives = capable pole
            a_new, p_new = H.separation(new["dirs"], layers, rp, rn)
            a_old, p_old = H.separation(old["dirs"], layers, rn, rp)
            for L in layers:
                assert a_new[L] == pytest.approx(a_old[L], abs=1e-12) and p_new[L] == pytest.approx(p_old[L], abs=1e-12), (new["name"], s, L)


def test_every_direction_has_steer_sign_minus_one_like_the_sibling_recipes(planted):
    _, (_, hm), (_, mm), (_, pm) = planted
    assert {**hm["steer_sign"], **mm["steer_sign"], **pm["steer_sign"]} == {k: -1 for k in ALL_DIRECTIONS}
    assert set(H.STEER_SIGN.values()) | set(P.STEER_SIGN.values()) == {-1}
    run = pytest.importorskip("sbx.run")
    if not hasattr(run, "build_parser"):
        pytest.skip("sbx.run has no build_parser")
    raw = run.build_parser().parse_args(["--model", "tiny", "--out", "x"]).strengths          # a comma string (or a list, if parsed there)
    default = [float(x) for x in (raw.split(",") if isinstance(raw, str) else raw) if str(x)]
    assert default and all(s < 0 for s in default), (
        f"sbx.run default --strengths {default} are not all negative: the sign convention of sbx/dirs (every direction points toward "
        f"the suppressed pole, negative alpha steers away) and the steer_sign = -1 of these recipes assume they are")


def test_default_runner_grid_restores_the_capable_pole_for_every_direction(planted):
    """The runner's grid for ONE direction name is the same for all of them: add at -0.1 / -0.2 / -0.4 and `cap` (one-sided projection
    against the neutral median).  For a model that holds back, both must move the state from the hiding pole TOWARD the capable pole
    along EVERY direction (SPEC section 0), which holds exactly when the direction points toward the hiding pole."""
    ctx, (hd, _), (md, _), (pd, _) = planted
    dirs = {**hd, **md, **pd}
    poles = pole_prompts(ctx, make_pool())
    assert set(dirs) == set(poles) == set(ALL_DIRECTIONS)
    for name, (hide, capable) in poles.items():
        for L in PLANTED_LAYERS:
            d = torch.tensor(dirs[name][L])
            h_hide = torch.tensor(ctx.hidden_last(hide)[:, L].astype(np.float32))
            h_cap = torch.tensor(ctx.hidden_last(capable)[:, L].astype(np.float32))
            c_hide, c_cap = float((h_hide @ d).mean()), float((h_cap @ d).mean())
            gap = c_hide - c_cap
            assert gap > 15, (name, L, gap)                    # the hiding pole projects HIGHER (planted separation ~ 2 * signal = 20)
            med, U = float(torch.median(h_cap @ d)), float(h_cap.norm(dim=-1).mean())   # a capable default: neutral median / norm
            cap = core.f_proj(d, med, one_sided=True)
            assert abs(float((cap(h_hide) @ d).mean()) - c_cap) < 0.25 * gap, (name, L)     # `cap` brings hiding states to the capable level
            assert abs(float((cap(h_cap) @ d).mean()) - c_cap) < 0.25 * gap, (name, L)      # ... and leaves a capable default (nearly) alone
            for alpha in (-0.1, -0.2, -0.4):                                                  # the runner's default strengths
                moved = float((core.f_add(torch.tensor(alpha * U) * d)(h_hide) @ d).mean())
                assert moved - c_hide == pytest.approx(alpha * U, rel=1e-3)
                assert abs(moved - c_cap) < abs(c_hide - c_cap), (name, L, alpha)           # closer to the capable pole
            away = float((core.f_add(torch.tensor(0.2 * U) * d)(h_cap) @ d).mean())          # a POSITIVE alpha steers toward hiding
            assert abs(away - c_hide) < abs(c_cap - c_hide), (name, L)


def test_null_signal_gives_chance_heldout_auroc():
    ctx = PlantedCtx(signal=0.0)
    d1, m1 = H.honesty_recipe(ctx, {}, cfg_for(n_pairs=60))
    d2, m2 = P.persona_recipe(ctx, make_pool(), cfg_for(n_pairs=60))
    for L in range(1, 6):
        for dirs in (d1["LIE"], d1["TRUTH"], d2["EXPERT"], d2["EFFORT"]):
            assert abs(np.linalg.norm(dirs[L]) - 1) < 1e-5
        assert 0.25 < m1["auroc"]["LIE"]["newq_newpara"][str(L)] < 0.75
        for s in ("val", "val_newwrap", "ood"):
            assert 0.25 < m1["auroc"]["TRUTH"][s][str(L)] < 0.75
        assert 0.1 < m2["auroc"]["EXPERT"]["newq_newpara"][str(L)] < 0.9
        assert 0.1 < m2["auroc"]["EFFORT"]["newq_newpara"][str(L)] < 0.9
    primary = ((m1, {"LIE": ["newq_newpara"], "TRUTH": ["val", "val_newwrap", "ood"]}),
               (m2, {"EXPERT": ["newq_newpara"], "EFFORT": ["newq_newpara"]}))
    for meta, key_sets in primary:                     # a warning appears exactly when the best held-out AUROC is below WEAK_AUROC
        for name, sets in key_sets.items():
            best = max(meta["auroc"][name][s][str(L)] for s in sets for L in range(1, 6))
            assert any(w.startswith(name + ":") for w in meta["warnings"]) == (best < H.WEAK_AUROC), (name, best)


def test_cos_B_only_with_reference_and_only_on_its_layers(tmp_path):
    ctx, pool = PlantedCtx(), make_pool()
    B = {L: _unit(np.random.RandomState(L).randn(ctx.H)).astype(np.float32) for L in (2, 4)}
    B[4] = ctx.dirs["expert"][4].astype(np.float32)                 # like the real B: points TOWARD underperformance (= toward novice)
    dirs, meta = P.persona_recipe(ctx, pool, cfg_for(n_pairs=30, B=B))
    assert set(meta["cos_B"]) == {"EXPERT", "EFFORT"} and set(meta["cos_B"]["EXPERT"]) == {"2", "4"}
    assert meta["cos_B"]["EXPERT"]["4"] > 0.97 and meta["cos_B"]["EXPERT"]["4"] == pytest.approx(core.cosine(dirs["EXPERT"][4], B[4]))
    d2, m2 = H.honesty_recipe(ctx, {}, cfg_for(n_pairs=30, B=B))
    assert set(m2["cos_B"]) == {"LIE", "TRUTH"} and m2["cos_B"]["LIE"]["2"] == pytest.approx(core.cosine(d2["LIE"][2], B[2]))
    d3, m3 = H.honesty_mcq_recipe(ctx, pool, cfg_for(n_pairs=30, B=B))
    assert set(m3["cos_B"]["LIE_mcq"]) == {"2", "4"}


def test_layers_subset_and_explicit_nl():
    ctx, pool = PlantedCtx(), make_pool()
    dirs, meta = P.persona_recipe(ctx, pool, cfg_for(n_pairs=24, layers=[4, 2]))
    assert set(dirs["EXPERT"]) == {2, 4} and meta["layers"] == [2, 4] and set(meta["auroc"]["EXPERT"]["newq"]) == {"2", "4"}
    d2, _ = H.honesty_recipe(ctx, pool, cfg_for(n_pairs=24, layers=[ctx.nl]))              # L = nl only when requested explicitly
    assert set(d2["LIE"]) == {ctx.nl} and set(d2["TRUTH"]) == {ctx.nl}
    with pytest.raises(ValueError, match="outside"):
        P.persona_recipe(ctx, pool, cfg_for(n_pairs=24, layers=[ctx.nl + 1]))


def test_n_pairs_handling():
    ctx, pool = PlantedCtx(), make_pool()
    with pytest.raises(ValueError, match="n_pairs must be"):
        H.honesty_recipe(ctx, {}, cfg_for(n_pairs=5))
    with pytest.raises(ValueError, match="n_pairs must be"):
        P.persona_recipe(ctx, pool, cfg_for(n_pairs=5))
    d, m = H.honesty_recipe(ctx, {}, cfg_for(n_pairs=100000))                                 # an upper bound: all 158 are used
    assert m["n_pairs_used"] == 158 and m["n_pairs_requested"] == 100000 and m["warnings"] == []
    d, m = H.honesty_recipe(ctx, {}, cfg_for(n_pairs=None))
    assert m["n_pairs_used"] == 158 and m["warnings"] == [] and m["n_pairs_requested"] is None
    d, m = P.persona_recipe(ctx, pool, cfg_for(n_pairs=1000))                                  # pool-limited: 80 - 15 held out
    assert m["n_pairs_used"] == 80 - m["counts"]["heldout_questions"] and any("available" in w for w in m["warnings"])
    d, m = P.persona_recipe(ctx, pool, cfg_for(n_pairs=None))                                   # default 400 -> pool-limited
    assert m["counts"]["heldout_questions"] == 16 and m["n_pairs_used"] == 64 and m["n_pairs_requested"] is None
    d, m = P.persona_recipe(ctx, {"wmdp": pool["wmdp"], "nonbio": []}, cfg_for(n_pairs=24))
    assert m["counts"]["train_nonbio"] == 0 and any("unbalanced" in w for w in m["warnings"])


def test_pool_is_the_only_source_of_questions():
    ctx, pool = PlantedCtx(), make_pool(40, 20)
    allowed = {q["qid"] for q in pool["wmdp"] + pool["nonbio"]}
    hard = {q["qid"] for q in pool["hard"]}
    P.persona_recipe(ctx, pool, cfg_for(n_pairs=40))
    H.honesty_mcq_recipe(ctx, pool, cfg_for(n_pairs=40))
    assert ctx.qids_seen and set(ctx.qids_seen) <= allowed and not set(ctx.qids_seen) & hard
    ctx2 = PlantedCtx()
    H.honesty_recipe(ctx2, {}, cfg_for(n_pairs=30))                 # `honesty` needs no pool and builds no MCQ prompt at all
    assert ctx2.qids_seen == [] and all(p.startswith("SYS<") and "> USER<" in p for p in ctx2.prompts_seen)
    # no prompt of any recipe contains an evaluation-style question text
    assert not any("MCQ<h" in p for p in ctx.prompts_seen)
    src = inspect.getsource(H) + inspect.getsource(P)
    for forbidden in ("get_eval", "get_pool", "split_of", "wmdp_train", "questions_heldout", "gpqa", "labbench", "mmlupro",
                      "sbx.evals", "from ..evals", "import evals"):
        assert forbidden not in src, forbidden


def test_real_pool_is_dir_split_and_no_evaluation_question_is_touched():
    evals = pytest.importorskip("sbx.evals")
    try:
        pool = evals.get_pool()
        served = {(q["source"], q["qid"]) for name in evals.EVALS for sp in ("disc", "rep") for q in evals.get_eval(name, sp)}
    except Exception as e:                                                          # evals module present but data unavailable
        pytest.skip(f"sbx.evals unavailable ({type(e).__name__}: {e})")
    allowed = {(q["source"], q["qid"]) for k in ("wmdp", "nonbio") for q in pool[k]}
    assert all(evals.split_of(q) == "dir" for k in ("wmdp", "nonbio") for q in pool[k])        # the pool is the dir split only
    assert not allowed & served                                                                 # ... disjoint from every served eval
    ctx = PlantedCtx()
    _, meta = P.persona_recipe(ctx, pool, cfg_for(n_pairs=200))
    H.honesty_mcq_recipe(ctx, pool, cfg_for(n_pairs=200))
    seen = set(ctx.questions_seen)
    assert len(seen) >= 200 and seen <= allowed and not seen & served
    assert not seen & {(q["source"], q["qid"]) for q in pool["hard"]}
    assert meta["counts"]["train_questions"] == 200 and meta["counts"]["heldout_questions"] == 50


def test_log_callback_and_context_contract():
    msgs = []
    ctx = PlantedCtx()
    P.persona_recipe(ctx, make_pool(), cfg_for(n_pairs=24, log=msgs.append))
    assert any("EXPERT" in m for m in msgs) and any("EFFORT" in m for m in msgs)
    ctx.log = msgs.append                                            # cfg["log"] absent -> ctx.log is used
    msgs.clear()
    P.persona_recipe(ctx, make_pool(), dict(n_pairs=24, seed=0, cache_dir=None, layers=None))
    assert msgs


# ===================================================================================================== determinism and cache
def test_determinism_and_seed_dependence():
    pool = make_pool()
    a = P.persona_recipe(PlantedCtx(), pool, cfg_for(n_pairs=30))
    b = P.persona_recipe(PlantedCtx(), pool, cfg_for(n_pairs=30))
    for n in a[0]:
        for L in a[0][n]:
            assert np.array_equal(a[0][n][L], b[0][n][L])
    assert a[1] == b[1]
    c = P.persona_recipe(PlantedCtx(), pool, cfg_for(n_pairs=30, seed=7))
    assert not np.array_equal(a[0]["EXPERT"][4], c[0]["EXPERT"][4])               # a different seed changes the selection / rotation
    assert _cos(a[0]["EXPERT"][4], c[0]["EXPERT"][4]) > 0.95                      # ... but not the recovered axis
    h1 = H.honesty_recipe(PlantedCtx(), {}, cfg_for(n_pairs=30))
    h2 = H.honesty_recipe(PlantedCtx(), {}, cfg_for(n_pairs=30))
    for n in h1[0]:
        for L in h1[0][n]:
            assert np.array_equal(h1[0][n][L], h2[0][n][L])
    assert h1[1] == h2[1]


def test_cache_roundtrip_is_exact_and_skips_the_model(tmp_path):
    pool = make_pool()
    cfg = cfg_for(tmp_path, n_pairs=30)
    ctx1 = PlantedCtx()
    first = {"persona": P.persona_recipe(ctx1, pool, cfg), "honesty": H.honesty_recipe(ctx1, pool, cfg),
             "honesty_mcq": H.honesty_mcq_recipe(ctx1, pool, cfg)}
    files = sorted(p.name for p in tmp_path.iterdir())
    assert ctx1.calls["hidden_last"] > 0 and files and all(f.startswith("dirs_hon__planted__") and f.endswith(".npz") for f in files)
    assert not [f for f in files if ".tmp" in f]                                  # atomic writes leave no temporary file
    assert len(files) == 2 * 6 + (6 + 8) + 6                                        # persona 2 x (2 train + 4 held-out groups), honesty, mcq
    ctx2 = PlantedCtx()
    second = {"persona": P.persona_recipe(ctx2, pool, cfg), "honesty": H.honesty_recipe(ctx2, pool, cfg),
              "honesty_mcq": H.honesty_mcq_recipe(ctx2, pool, cfg)}
    assert ctx2.calls["hidden_last"] == 0                                         # everything came from the cache
    assert sorted(p.name for p in tmp_path.iterdir()) == files
    for k in first:
        for n in first[k][0]:
            for L in first[k][0][n]:
                assert np.array_equal(first[k][0][n][L], second[k][0][n][L])      # bit-identical directions
        assert first[k][1] == second[k][1]
    # the cache does not change the result versus a run without cache
    plain = P.persona_recipe(PlantedCtx(), pool, cfg_for(n_pairs=30))
    for n in plain[0]:
        for L in plain[0][n]:
            assert np.array_equal(plain[0][n][L], first["persona"][0][n][L])


def test_cache_key_covers_prompts_layers_adapter_and_model(tmp_path):
    pool = make_pool()
    base = cfg_for(tmp_path, n_pairs=24)
    P.persona_recipe(PlantedCtx(), pool, base)
    n0 = len(list(tmp_path.iterdir()))
    ctx = PlantedCtx()
    P.persona_recipe(ctx, pool, dict(base, layers=[1, 2, 3]))                      # other layers -> new files
    assert ctx.calls["hidden_last"] > 0 and len(list(tmp_path.iterdir())) > n0
    n1 = len(list(tmp_path.iterdir()))
    ctx = PlantedCtx(key="other-adapter")                                          # other adapter key -> new files
    P.persona_recipe(ctx, pool, base)
    assert ctx.calls["hidden_last"] > 0 and len(list(tmp_path.iterdir())) > n1
    n2 = len(list(tmp_path.iterdir()))
    ctx = PlantedCtx()
    ctx.model.config._name_or_path = "another/model"                               # other checkpoint -> new files
    P.persona_recipe(ctx, pool, base)
    assert ctx.calls["hidden_last"] > 0 and len(list(tmp_path.iterdir())) > n2
    n3 = len(list(tmp_path.iterdir()))
    ctx = PlantedCtx()
    P.persona_recipe(ctx, pool, dict(base, seed=3))                                # other seed -> other selection / rotation
    assert ctx.calls["hidden_last"] > 0 and len(list(tmp_path.iterdir())) > n3
    ctx = PlantedCtx()                                                             # original request still fully cached
    P.persona_recipe(ctx, pool, base)
    assert ctx.calls["hidden_last"] == 0


def test_corrupt_or_mismatching_cache_raises(tmp_path):
    ctx, pool = PlantedCtx(), make_pool()
    cfg = cfg_for(tmp_path, n_pairs=24)
    P.persona_recipe(ctx, pool, cfg)
    files = sorted(tmp_path.iterdir())
    target = files[0]
    good = target.read_bytes()
    target.write_bytes(good[:100])                                                 # truncated file
    with pytest.raises(ValueError, match="unreadable cache file"):
        P.persona_recipe(PlantedCtx(), pool, cfg)
    target.write_bytes(good)
    # a well-formed file that does not match its request (wrong layers) raises, it is never silently reused
    z = dict(np.load(target))
    z["layers"] = np.array([9, 9, 9])
    np.savez(target, **z)
    with pytest.raises(ValueError, match="does not match"):
        P.persona_recipe(PlantedCtx(), pool, cfg)
    target.write_bytes(good)
    P.persona_recipe(PlantedCtx(), pool, cfg)                                      # restored -> works again


def test_group_states_contract_and_errors(tmp_path):
    ctx = PlantedCtx()
    prompts = [f"SYS<> USER<q{i}>" for i in range(10)]
    g = H.group_states(ctx, prompts, [2, 4], chunk=3)                              # chunking does not change the result
    g2 = H.group_states(ctx, prompts, [2, 4], chunk=10)
    assert g["mean"].shape == (2, ctx.H) and g["mean"].dtype == np.float32 and g["n"] == 10 and g["raw"] is None and not g["cached"]
    assert np.allclose(g["mean"], g2["mean"], atol=1e-5) and np.allclose(g["norm"], g2["norm"])
    raw = H.group_states(ctx, prompts, [2, 4], keep_raw=True)
    assert raw["raw"].shape == (10, 2, ctx.H) and raw["raw"].dtype == np.float16
    assert np.allclose(raw["raw"].astype(np.float64).mean(0), g["mean"], atol=2e-2)
    assert np.allclose(g["norm"], np.linalg.norm(raw["raw"].astype(np.float64), axis=2).mean(0), rtol=1e-3)
    with pytest.raises(ValueError, match="empty prompt list"):
        H.group_states(ctx, [], [1])
    with pytest.raises(ValueError, match="must match"):
        H.group_states(ctx, prompts, [1], tag="bad tag!")
    bad = PlantedCtx()
    bad.nonfinite_layer = 4
    with pytest.raises(ValueError, match=r"non-finite hidden states at layers \[4\]"):
        H.group_states(bad, prompts, [2, 4])
    short = PlantedCtx()
    short.hidden_last = lambda prompts, cap=16: np.zeros((len(prompts) - 1, short.nl + 1, short.H), np.float16)
    with pytest.raises(ValueError, match="returned shape"):
        H.group_states(short, prompts, [1])
    with pytest.raises(ValueError, match="non-empty"):
        H.run_contrast(ctx, "X", prompts, prompts[:3], {}, [1], None, print)
    with pytest.raises(ValueError, match="held-out sides"):
        H.run_contrast(ctx, "X", prompts[:5], prompts[5:], {"s": (prompts, prompts[:2])}, [1], None, print)
    same = PlantedCtx(signal=0.0, content_scale=0.0, para_scale=0.0)                # identical states on both sides -> degenerate
    same._rs = lambda *parts: np.random.RandomState(0)
    with pytest.raises(ValueError, match="degenerate contrast"):
        H.run_contrast(same, "X", prompts[:5], prompts[5:], {}, [1, 2], None, print)


# ===================================================================================================== tiny random Llama
@pytest.mark.parametrize("recipe", ["honesty", "honesty_mcq", "persona"])
def test_tiny_ctx_every_layer_unit_norm_determinism_and_cache(tiny, tmp_path, monkeypatch, recipe):
    monkeypatch.setattr(H, "MAX_HELDOUT", 16)                                      # keeps the CPU run short (held-out sets only)
    real_pairs = H.fact_pairs
    monkeypatch.setattr(H, "fact_pairs", lambda data, split: dict(list(real_pairs(data, split).items())[:12 if split != "train" else None]))
    fn = {**H.RECIPES, **P.RECIPES}[recipe]
    pool = make_pool(30, 5)
    cfg = cfg_for(tmp_path, n_pairs=12)
    calls = {"n": 0}
    orig = tiny.hidden_last

    def counting(prompts, cap=16):
        calls["n"] += 1
        return orig(prompts, cap)

    tiny.hidden_last = counting
    try:
        d1, m1 = fn(tiny, pool, cfg)
        n_first = calls["n"]
        d2, m2 = fn(tiny, pool, cfg)
        assert n_first > 0 and calls["n"] == n_first                               # second call: pure cache hits
    finally:
        tiny.hidden_last = orig
    assert m1 == m2 and m1["layers"] == [1, 2, 3] and m1["adapter_key"] == "tiny"
    assert set(d1) == {"honesty": {"LIE", "TRUTH"}, "honesty_mcq": {"LIE_mcq"}, "persona": {"EXPERT", "EFFORT"}}[recipe]
    for name in d1:
        assert set(d1[name]) == {1, 2, 3}                                          # every layer 1..nl-1
        for L, v in d1[name].items():
            assert v.dtype == np.float32 and v.shape == (tiny.H,) and np.isfinite(v).all()
            assert abs(float(np.linalg.norm(v)) - 1.0) < 1e-5
            assert np.array_equal(v, d2[name][L])                                  # bit-identical through the cache
    assert json.loads(json.dumps(m1)) == m1
    for st in m1["auroc"].values():
        for s in st.values():
            assert all(0.0 <= x <= 1.0 for x in s.values())


def test_tiny_ctx_nl_layer_and_steering_smoke(tiny):
    dirs, meta = P.persona_recipe(tiny, make_pool(30, 5), cfg_for(n_pairs=12, layers=[2, tiny.nl]))
    assert set(dirs["EXPERT"]) == {2, tiny.nl}
    layers = [2]
    prompts = tiny.prompts(make_pool(5, 0)["wmdp"][:3])
    U, med = tiny.neutral_stats(prompts, dirs, layers)
    enc = tiny.adapter.encode(tiny.tok, prompts)
    with torch.no_grad():
        base = tiny.model(**enc).logits[:, -1].float()
        with core.Hooks(tiny, core.spec_add(tiny, dirs["EXPERT"], 2, 2.0, U)):
            steered = tiny.model(**enc).logits[:, -1].float()
        with core.Hooks(tiny, core.spec_add(tiny, dirs["EXPERT"], 2, 0.0, U)):
            zero = tiny.model(**enc).logits[:, -1].float()
        with core.Hooks(tiny, core.spec_proj(tiny, dirs["EXPERT"], [2], med["EXPERT"], one_sided=True)):
            tiny.model(**enc)                                                       # the spec helpers accept the recipe output
    assert (steered - base).abs().max() > 1e-4 and torch.allclose(zero, base, atol=1e-5)


# ===================================================================================================== real chat templates
@pytest.mark.parametrize("key", ["gptoss-20b", "qwen3-8b-think", "llama-3.1-8b-instruct", "gemma-3-4b-it"])
def test_prompts_render_with_real_chat_templates(key):
    transformers = pytest.importorskip("transformers")
    try:
        from sbx.adapters import get_adapter
        a = get_adapter(key)
    except Exception as e:                                                          # registry / adapter module unavailable
        pytest.skip(f"adapter {key} unavailable ({type(e).__name__}: {e})")
    try:
        tok = transformers.AutoTokenizer.from_pretrained(a.repo, local_files_only=True)
    except Exception as e:                                                          # tokenizer not in the local HF cache
        pytest.skip(f"tokenizer of {a.repo} not available offline ({type(e).__name__})")
    q = toy_questions(1)[0]
    stmt = "The capital of France is Paris."
    # LIE: dishonest vs honest system text on a factual question (chat prompt)
    lie = a.build_chat(tok, "What is the capital of France?", sys_text=H.LIE_TRAIN[0])
    hon = a.build_chat(tok, "What is the capital of France?", sys_text=H.HONEST_TRAIN[0])
    assert H.LIE_TRAIN[0] in lie and "What is the capital of France?" in lie and H.HONEST_TRAIN[0] in hon
    assert lie.replace(H.LIE_TRAIN[0], "@@") == hon.replace(H.HONEST_TRAIN[0], "@@")      # the contrast is the system text only
    # TRUTH: wrapper + statement in the user turn, no system text
    for w in ALL_WRAPPERS:
        tr = a.build_chat(tok, H._wrap(w, stmt))
        assert stmt in tr and not any(t in tr for t in H.LIE_TRAIN + H.HONEST_TRAIN)
    # MCQ persona prompts
    for pos, neg in ((P.EXPERT_TRAIN[0], P.NOVICE_TRAIN[0]), (P.CAREFUL_TRAIN[0], P.QUICK_TRAIN[0])):
        mp, mn = a.build(tok, q, sys_text=pos), a.build(tok, q, sys_text=neg)
        assert pos in mp and q["question"] in mp and neg in mn
        assert mp.replace(pos, "@@") == mn.replace(neg, "@@")
    assert len(tok.encode(lie, add_special_tokens=False)) > 10


# ===================================================================================================== docs
def test_docs_list_every_text_wrapper_and_dataset_hash():
    assert DOCS.exists()
    text = DOCS.read_text(encoding="utf-8")
    texts = (H.LIE_TRAIN + H.HONEST_TRAIN + H.LIE_HELD + H.HONEST_HELD + P.EXPERT_TRAIN + P.NOVICE_TRAIN + P.EXPERT_HELD
             + P.NOVICE_HELD + P.CAREFUL_TRAIN + P.QUICK_TRAIN + P.CAREFUL_HELD + P.QUICK_HELD)
    missing = [t for t in texts if t not in text]
    assert not missing, f"docs lack {len(missing)} texts, e.g. {missing[:2]}"
    wrong = [w for w in ALL_WRAPPERS if w.replace("\n", "\\n") not in text]
    assert not wrong, wrong
    for f in (H.TRUTH_FILE, H.QUESTIONS_FILE):
        assert hashlib.sha256((H.DATA_DIR / f).read_bytes()).hexdigest() in text, f"stale hash for {f}"
    for needle in ("LIE", "TRUTH", "EXPERT", "EFFORT", "LIE_mcq", "honesty_mcq", "persona", "TOWARD LYING", "TOWARD FALSEHOOD",
                   "TOWARD the novice", "POSITIVE", "NEGATIVE", "dirs_hon__", "cfg[\"cache_dir\"]", "python -m sbx.dirs.honesty build"):
        assert needle in text, needle
    # the sign convention is uniform: no direction needs positive strengths, so the two-invocation workaround of the first version is gone
    assert "--strengths=0.1,0.2,0.4" not in text and "pass positive strengths" not in text
    assert "--dir-names B,LIE,LIE_mcq,TRUTH,EXPERT,EFFORT" in text and "--strengths=-0.1,-0.2,-0.4" in text
    assert "TOWARD the capable" not in text and "TOWARD TRUTH" not in text
    for cat in list(H.IID_CATS) + list(H.OOD_CATS):
        assert f"`{cat}`" in text, cat
