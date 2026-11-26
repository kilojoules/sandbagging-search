"""Tests for sbx.evals (CPU only, no network, < 90 s).

What is verified
  * registry: canonical names, category / hazard / pool flags, files exist, `n` pinned to the data, every file registered
  * schema: every row of every data/evals file (strict); the pre-existing files (only the 5 known option-duplicate rows)
  * splits: crc rule, WMDP-bio special case (first 680 rows of wmdp_train.jsonl are dir, nothing else is), proportions,
    no question in two splits, no dir-pool question in ANY disc / rep / all eval (also under a stricter normalisation),
    registry-aware split_of, union view, copies, ordering
  * clusters: questions that share a long context (one protocol / plasmid / template sequence) are split together in the
    clustered evals (checked against a naive O(n^2) reference and a protocol-title oracle, stable under the parameters), the
    residual dependence in the other evals is bounded, the literal crc rule still holds for every unclustered question
  * prompt length: no served or pooled question exceeds MAX_PROMPT_CHARS (the 26k-token CloningScenarios prompt is gone), removed
    questions are reported and never leak, the cap comes after de-duplication, and (tokenizer cached) it equals the legacy
    3000-token filter
  * de-duplication policy on synthetic registries (dir wins, first wins, dir members never dropped)
  * determinism across processes (different PYTHONHASHSEED), builder helpers, converters, docs in sync with the registry
Run:  USE_TF=0 USE_TORCH=1 python3 -m pytest sbx/tests/test_evals.py -q
"""
from __future__ import annotations

import collections
import functools
import glob
import hashlib
import io
import itertools
import json
import os
import random
import re
import subprocess
import sys
import zlib

import pytest

from sbx import evals as E

ROOT = E.ROOT
CANONICAL = ["wmdp_bio", "mmlu_bio", "mmlupro_bio", "gpqa_bio", "labbench_seqqa", "labbench_litqa2", "labbench_protocolqa",
             "labbench_cloning", "gpqa_control", "mathphys", "wmdp_cyber", "wmdp_chem", "mmlu_nonbio"]
MMLU_SETS = ["mmlu_college_bio", "mmlu_hs_bio", "mmlu_virology", "mmlu_med_genetics", "mmlu_anatomy", "mmlu_clinical",
             "mmlu_college_med", "mmlu_prof_med", "mmlu_nutrition"]
# rows of the pre-existing files that violate "unique non-empty options" (they are dropped at load time and documented)
KNOWN_INVALID = {("labbench_litqa2", "labbench_litqa2_0011"), ("labbench_litqa2", "labbench_litqa2_0033"),
                 ("gpqa_control", "gpqa_control_0044"), ("gpqa_control", "gpqa_control_0123"), ("mathphys", "c0223")}
FILE_EVALS = [n for n, s in E.EVALS.items() if not s.is_union]
CLUSTERED = ["labbench_seqqa", "labbench_protocolqa", "labbench_cloning"]   # evals whose stems share long contexts
# After the prompt-length cap CloningScenarios keeps 6 questions, all in rep (docs/evals.md section 7): no dir, no disc question.
NO_DISC_OR_REP = {"labbench_cloning": {"dir": 0, "disc": 0, "rep": 6}}


def jl(relpath):
    with open(os.path.join(ROOT, relpath), encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def strict_norm(s):
    """A deliberately different (stricter) normalisation than evals.norm_text: ascii letters/digits only."""
    return re.sub(r"[^a-z0-9]", "", s.lower())


@pytest.fixture(scope="module")
def reg():
    return E._registry()


@pytest.fixture(scope="module")
def served(reg):
    """{(eval, split): [q]} for split in disc / rep / all, via the public API."""
    return {(n, s): E.get_eval(n, s) for n in E.EVALS for s in E.EVAL_SPLITS}


# ============================================================================================== registry definition
def test_canonical_names_and_categories():
    for n in CANONICAL + MMLU_SETS:
        assert n in E.EVALS, n
    controls = {"gpqa_control", "mathphys", "wmdp_cyber", "wmdp_chem", "mmlu_nonbio", "mmlupro_nonbio"}
    for n, s in E.EVALS.items():
        assert isinstance(s, E.EvalSpec) and s.name == n
        assert s.category in ("bio", "control")
        assert isinstance(s.hazard, bool)
        assert (s.category == "control") == (n in controls), n
    assert E.EVALS["wmdp_bio"]["category"] == "bio"  # key access works as well as attribute access


def test_hazard_flags_are_wmdp_only():
    assert {n for n, s in E.EVALS.items() if s.hazard} == {"wmdp_bio", "wmdp_cyber", "wmdp_chem"}
    assert E.list_evals(hazard=True) == ["wmdp_bio", "wmdp_cyber", "wmdp_chem"]


def test_pool_membership_matches_spec():
    assert E.list_evals(pool="wmdp") == ["wmdp_bio"]
    assert set(E.list_evals(pool="hard")) == {"mmlupro_bio", "gpqa_bio", "labbench_seqqa", "labbench_litqa2",
                                                "labbench_protocolqa", "labbench_cloning", "supergpqa_bio"}
    assert set(E.list_evals(pool="nonbio")) == {"mmlu_nonbio", "mathphys"}
    assert all(E.EVALS[n].category == "bio" for n in E.list_evals(pool="hard"))


def test_list_evals_filters():
    assert set(E.list_evals("control")) == {n for n, s in E.EVALS.items() if s.category == "control"}
    # union views are opt-in so that "loop over all bio evals" never counts a question twice
    assert "mmlu_bio_all" not in E.list_evals("bio") and "mmlu_bio_all" in E.list_evals("bio", include_unions=True)
    assert E.list_evals() == [n for n, s in E.EVALS.items() if not s.is_union]
    assert E.list_evals(include_unions=True) == list(E.EVALS)
    assert E.list_evals("bio", hazard=True) == ["wmdp_bio"] and E.list_evals("control", hazard=True) == ["wmdp_cyber", "wmdp_chem"]


def test_spec_metadata_complete():
    for n, s in E.EVALS.items():
        assert s.origin.strip() and s.licence.strip() and s.notes.strip(), f"{n}: origin / licence / notes must be filled"
        assert "|" not in s.origin and "|" not in s.licence  # they are rendered into a markdown table


def test_spec_files_exist_and_every_data_evals_file_is_registered():
    for n, s in E.EVALS.items():
        for f in s.files:
            assert os.path.exists(os.path.join(ROOT, f.path)), f"{n}: missing {f.path}"
    on_disk = {os.path.basename(p) for p in glob.glob(os.path.join(ROOT, "data", "evals", "*.jsonl"))}
    registered = {os.path.basename(f.path) for s in E.EVALS.values() if s.generated for f in s.files}
    assert on_disk == registered
    assert sorted(registered) == sorted(f"{n}.jsonl" for n, s in E.EVALS.items() if s.generated)


def test_heldout_file_is_fully_covered_by_wmdp_and_mmlu_eval():
    rows = jl("data/questions_heldout_eval.jsonl")
    assert len(rows) == 400
    w = sum(1 for r in rows if r["source"].startswith("wmdp/"))
    m = sum(1 for r in rows if r["source"].startswith("mmlu/"))
    assert (w, m) == (200, 200)


def test_spec_n_matches_the_data(reg):
    for n, s in E.EVALS.items():
        sizes = E.split_sizes(n)
        assert s.n == sum(sizes.values()) > 0, f"{n}: spec n={s.n}, data {sizes} (run `python3 -m sbx.evals counts`)"
        assert sizes["disc"] + sizes["rep"] == len(E.get_eval(n, "all"))


def test_evalspec_validation():
    f = (E.Src("x.jsonl"),)
    with pytest.raises(ValueError):
        E.EvalSpec("a", "medicine", False, f)
    with pytest.raises(ValueError):
        E.EvalSpec("a", "bio", 1, f)  # hazard must be a real bool
    with pytest.raises(ValueError):
        E.EvalSpec("a", "bio", False, f, pool="easy")
    with pytest.raises(ValueError):
        E.EvalSpec("a", "bio", False, ())  # neither files nor union
    with pytest.raises(ValueError):
        E.EvalSpec("a", "bio", False, f, union=("b",))  # both
    with pytest.raises(ValueError):
        E.EvalSpec("a", "bio", False, (), pool="hard", union=("b",))  # a view cannot feed a pool
    with pytest.raises(ValueError):
        E.EvalSpec("a", "bio", False, f, clustered="yes")  # clustered must be a real bool
    with pytest.raises(ValueError):
        E.EvalSpec("a", "bio", False, (), union=("b",), clustered=True)  # a view cannot be clustered (its members are)
    s = E.EvalSpec("a", "bio", False, ("plain/path.jsonl",))
    assert s.files == (E.Src("plain/path.jsonl"),) and not s.generated and not s.is_union and s.clustered is False
    assert E.EvalSpec("a", "bio", False, f, clustered=True).clustered is True


def test_clustered_flag_is_set_on_exactly_the_templated_evals():
    assert {n for n, s in E.EVALS.items() if s.clustered} == set(CLUSTERED)
    assert not E.EVALS["wmdp_bio"].clustered   # the WMDP-bio dir pool is fixed by file order (already used for directions)


# ============================================================================================== schema
def test_validate_question_cases():
    good = {"source": "s", "qid": "q", "question": "What?", "choices": ["a", "b"], "answer": 1}
    assert E.validate_question(good) is None
    bad = [
        ({k: v for k, v in good.items() if k != "qid"}, "structure"),
        (dict(good, extra=1), "structure"),
        (dict(good, answer=True), "structure"),
        (dict(good, answer=2), "structure"),
        (dict(good, answer=-1), "structure"),
        (dict(good, answer="0"), "structure"),
        (dict(good, choices=["a"]), "structure"),
        (dict(good, choices=[str(i) for i in range(11)]), "structure"),
        (dict(good, choices=["a", 2]), "structure"),
        (dict(good, question="  "), "structure"),
        (dict(good, question="?!"), "structure"),
        (dict(good, source=""), "structure"),
        (dict(good, choices=["a", " "]), "choices"),
        (dict(good, choices=["a", "a "]), "choices"),
        ([], "structure"),
    ]
    for q, kind in bad:
        why = E.validate_question(q)
        assert why is not None and why.startswith(kind), (q, why)
    assert E.validate_question(dict(good, choices=[str(i) for i in range(10)], answer=9)) is None


def test_generated_files_are_strictly_valid():
    paths = sorted(glob.glob(os.path.join(ROOT, "data", "evals", "*.jsonl")))
    assert len(paths) == len([s for s in E.EVALS.values() if s.generated]) >= 18
    seen_ids = set()
    for p in paths:
        raw = open(p, "rb").read()
        assert len(raw) < 5_000_000, f"{p} is {len(raw)} bytes"
        assert all(b < 128 for b in raw), f"{p} must be ASCII-only JSON (locale independent)"
        rows = [json.loads(line) for line in raw.decode("ascii").splitlines() if line.strip()]
        name = os.path.splitext(os.path.basename(p))[0]
        assert 20 <= len(rows) <= 410, (name, len(rows))
        for r in rows:
            assert E.validate_question(r) is None, (name, r["qid"], E.validate_question(r))
            assert list(r) == list(E.KEYS)
            assert 2 <= len(r["choices"]) <= 10 and len({c.strip() for c in r["choices"]}) == len(r["choices"])
            assert (r["source"], r["qid"]) not in seen_ids
            seen_ids.add((r["source"], r["qid"]))
        assert len({r["qid"] for r in rows}) == len(rows)


def test_generated_files_follow_the_builder_order():
    """Rows are written in ascending sha256 'order' rank (make_set); a reordered / hand-edited file would break this."""
    for n, s in E.EVALS.items():
        if not s.generated:
            continue
        qids = [r["qid"] for r in jl(s.files[0].path)]
        ranks = [E._rank(f"{n}|order", q) for q in qids]
        assert ranks == sorted(ranks), n


def test_existing_files_only_have_the_known_invalid_rows(reg):
    got = {(r["eval"], r["qid"]) for r in E.registry_report()["invalid"]}
    assert got == KNOWN_INVALID
    assert all(r["reason"] == "choices: duplicated options" for r in E.registry_report()["invalid"])


def test_generated_sets_are_fresh(reg):
    """New sets share no question with the pilot file or any pre-existing registered file."""
    old = {n for n, s in E.EVALS.items() if not s.generated and not s.is_union}
    old_keys = set()
    for n in old:
        for src in E.EVALS[n].files:
            for q in jl(src.path):
                if not src.source_prefix or q["source"].startswith(src.source_prefix):
                    old_keys.add(E.question_key(q))
    old_keys |= {E.question_key(q) for q in jl(E.PILOT_FILE)}
    for n, s in E.EVALS.items():
        if s.generated:
            for q in jl(s.files[0].path):
                assert E.question_key(q) not in old_keys, (n, q["qid"])


# ============================================================================================== split rule
def test_hash_bucket_is_the_spec_crc_rule():
    q = {"source": "mmlu/virology", "qid": "abc"}
    assert E.hash_bucket(q) == zlib.crc32(b"mmlu/virology|abc") % 100
    for qid in range(300):
        q = {"source": "x/y", "qid": f"id{qid}"}
        b = zlib.crc32(f"x/y|id{qid}".encode()) % 100
        want = "dir" if b < 25 else "disc" if b < 63 else "rep"
        assert E._raw_split(q, frozenset()) == want


def rule_bucket(reg, name, q):
    """The SPEC crc bucket that decides q's split: its own for every unclustered eval, its cluster representative's for a
    clustered one (the rule with the hashed id replaced by the representative's)."""
    if not E.EVALS[name].clustered:
        return E.hash_bucket(q)
    src, qid = reg.cluster_rep[(q["source"], q["qid"])]
    return E.hash_bucket({"source": src, "qid": qid})


def test_served_questions_follow_the_crc_rule(reg, served):
    deviating = 0
    for n in FILE_EVALS:
        for split in ("disc", "rep"):
            for q in served[(n, split)]:
                if q["source"] == E.WMDP_SOURCE:
                    continue
                b = rule_bucket(reg, n, q)
                assert (25 <= b < 63) if split == "disc" else (b >= 63), (n, split, q["qid"], b)
                if n in CLUSTERED:  # does the cluster rule differ from the literal one for this question?
                    own = E.hash_bucket(q)
                    deviating += (25 <= own < 63) != (25 <= b < 63) or (own >= 63) != (b >= 63)
        for q in reg.kept[n]["dir"]:
            if q["source"] != E.WMDP_SOURCE:
                assert rule_bucket(reg, n, q) < 25, (n, q["qid"])
    assert deviating > 20  # the clustered evals really use the representative's bucket (not the question's own)


def test_every_eligible_question_is_served_or_documented(reg):
    """Nothing disappears silently: rows read = kept + invalid + duplicates dropped + too long, per eval."""
    rep = E.registry_report()
    for n in FILE_EVALS:
        r = rep["per_eval"][n]
        assert r["rows"] == r["n"] + r["invalid"] + r["dropped_duplicate"] + r["too_long"], (n, r)
    assert len(rep["invalid"]) == 5
    assert rep["rows"] == sum(rep["per_eval"][n]["rows"] for n in FILE_EVALS)
    assert len(rep["too_long"]) == sum(rep["per_eval"][n]["too_long"] for n in FILE_EVALS) == 50


def test_wmdp_special_case_exact():
    train = jl("data/wmdp_train.jsonl")
    held = [q for q in jl("data/questions_heldout_eval.jsonl") if q["source"] == "wmdp/bio"]
    assert len(train) == 1073 and len(held) == 200
    first = train[:680]
    # (1) the first 680 rows of wmdp_train.jsonl are dir, whatever their hash says
    assert [E.split_of(q) for q in first] == ["dir"] * 680
    assert len({E._raw_split(q, E._wmdp_dir_ids()) for q in first}) == 1
    assert sum(1 for q in first if E.hash_bucket(q) >= 25) > 100  # i.e. the special case really overrides the crc rule
    # (2) the dir pool is exactly those rows, in file order, with identical content
    pool = E.get_pool()["wmdp"]
    assert pool == first
    assert E.split_sizes("wmdp_bio")["dir"] == 680
    # (3) nothing else in WMDP-bio is dir: the other 393 + 200 rows are split disc / rep at crc bucket 50
    rest = train[680:] + held
    assert len(rest) == 593
    disc = {q["qid"] for q in E.get_eval("wmdp_bio", "disc")}
    rep = {q["qid"] for q in E.get_eval("wmdp_bio", "rep")}
    assert not disc & rep
    assert disc | rep == {q["qid"] for q in rest}
    for q in rest:
        assert E.split_of(q) == ("disc" if E.hash_bucket(q) < E.WMDP_REST_DISC_BELOW else "rep")
        assert (q["qid"] in disc) == (E.hash_bucket(q) < E.WMDP_REST_DISC_BELOW)
    assert not {q["qid"] for q in first} & (disc | rep)
    # (4) balanced halves (the crc rule alone would give 25/38/37)
    assert abs(len(disc) - len(rep)) <= 40


def test_split_proportions():
    tot = collections.Counter()
    for n in FILE_EVALS:
        if n == "wmdp_bio":
            continue
        sz = E.split_sizes(n)
        tot.update(sz)
        total = sum(sz.values())
        if total >= 300:  # per-eval check only where binomial noise is small
            for s, p in (("dir", .25), ("disc", .38), ("rep", .37)):
                assert abs(sz[s] / total - p) < 0.08, (n, sz)
    t = sum(tot.values())
    assert t > 5000
    for s, p in (("dir", .25), ("disc", .38), ("rep", .37)):
        assert abs(tot[s] / t - p) < 0.02, (s, tot[s] / t)


# ============================================================================================== no leakage
def test_no_question_in_two_splits(reg):
    """Each question (by key and by (source, qid)) lives in exactly one split, and a disc/rep question in exactly one eval."""
    by_key = collections.defaultdict(list)
    ids = collections.Counter()
    for n in FILE_EVALS:
        for s in E.SPLITS:
            for q in reg.kept[n][s]:
                by_key[E.question_key(q)].append((n, s))
                ids[(q["source"], q["qid"])] += 1
    assert max(ids.values()) == 1
    for key, where in by_key.items():
        splits = {s for _, s in where}
        assert len(splits) == 1, (key[:60], where)
        if splits != {"dir"}:
            assert len(where) == 1, (key[:60], where)  # only dir questions may be shared between evals / repeated
    for n in FILE_EVALS:
        d, r = {q["qid"] for q in reg.kept[n]["disc"]}, {q["qid"] for q in reg.kept[n]["rep"]}
        assert not d & r


def test_no_dir_question_of_any_eval_is_ever_served(reg, served):
    """Stronger than the pool test: the dir buckets of the evals that feed no pool are protected too."""
    dir_strict = {strict_norm(q["question"]) for n in FILE_EVALS for q in reg.kept[n]["dir"]}
    assert len(dir_strict) > 1500
    for (name, split), qs in served.items():
        for q in qs:
            assert strict_norm(q["question"]) not in dir_strict, (name, split, q["qid"])


def test_dropped_rows_have_a_kept_twin_and_kept_rows_keep_their_own_split(reg):
    kept = {(n, q["qid"]): (q, s) for n in FILE_EVALS for s in E.SPLITS for q in reg.kept[n][s]}
    raw = {}
    for n in FILE_EVALS:
        for src in E.EVALS[n].files:
            for q in jl(src.path):
                if not src.source_prefix or q["source"].startswith(src.source_prefix):
                    raw[(n, q["qid"])] = q
    dropped = E.registry_report()["dropped"]
    assert len(dropped) == 20 and all((d["eval"], d["qid"]) in raw and (d["eval"], d["qid"]) not in kept for d in dropped)
    for d in dropped:
        twin_eval, twin_qid = d["kept"].split("/", 1)
        twin, twin_split = kept[(twin_eval, twin_qid)]
        assert E.question_key(twin) == E.question_key(raw[(d["eval"], d["qid"])])
        assert (twin_split == "dir") == (d["reason"] == "duplicate_of_dir")
        assert E.split_of(raw[(d["eval"], d["qid"])]) == twin_split  # the dropped copy resolves to its group's split
    wmdp_ids = E._wmdp_dir_ids()
    for (n, qid), (q, split) in kept.items():
        rep = reg.cluster_rep.get((q["source"], q["qid"]))  # None unless the eval is clustered
        assert (rep is not None) == E.EVALS[n].clustered, (n, qid)
        assert E._raw_split(q, wmdp_ids, rep) == split, (n, qid)


def test_split_of_agrees_with_the_bucket_of_every_served_question(reg):
    for n in FILE_EVALS:
        for s in E.SPLITS:
            for q in reg.kept[n][s]:
                assert E.split_of(q) == s, (n, q["qid"])
    for r in E.registry_report()["too_long"]:   # a removed question keeps answering with the split it had
        assert E.split_of({"source": r["source"], "qid": r["qid"]}) == r["split"], r


def test_pool_never_appears_in_any_eval(served):
    pool = E.get_pool()
    pool_q = [q for qs in pool.values() for q in qs]
    assert len(pool_q) > 900
    pool_ids = {(q["source"], q["qid"]) for q in pool_q}
    pool_keys = {E.question_key(q) for q in pool_q}
    pool_strict = {strict_norm(q["question"]) for q in pool_q}
    n_checked = 0
    for (name, split), qs in served.items():
        for q in qs:
            n_checked += 1
            assert (q["source"], q["qid"]) not in pool_ids, (name, split, q["qid"])
            assert E.question_key(q) not in pool_keys, (name, split, q["qid"])
            assert strict_norm(q["question"]) not in pool_strict, (name, split, q["qid"], q["question"][:60])
    assert n_checked > 12000  # 'all' + 'disc' + 'rep' (each question twice) for every eval, union views included


def test_pool_contents_and_structure(reg):
    pool = E.get_pool()
    assert list(pool) == ["wmdp", "hard", "nonbio"]
    assert len(pool["wmdp"]) == 680
    for name, qs in pool.items():
        assert qs, name
        assert all(E.split_of(q) == "dir" for q in qs), name
        assert all(E.validate_question(q) is None for q in qs)
        assert len({E.question_key(q) for q in qs}) == (len(qs) if name != "wmdp" else len(qs)) or name == "wmdp"
    hard_srcs = {q["source"].split("/")[0] for q in pool["hard"]}
    assert hard_srcs == {"labbench", "mmlupro", "gpqa", "supergpqa"}
    nonbio_srcs = {q["source"] for q in pool["nonbio"]}
    assert all(s.startswith("mmlu/") for s in nonbio_srcs)
    # pools are built only from the dir buckets of their evals and are pairwise disjoint
    for p in ("hard", "nonbio"):
        from_dir = {(q["source"], q["qid"]) for n in E.list_evals(pool=p) for q in reg.kept[n]["dir"]}
        assert {(q["source"], q["qid"]) for q in pool[p]} <= from_dir
        assert len(pool[p]) <= len(from_dir)
    keys = [{E.question_key(q) for q in pool[p]} for p in pool]
    assert not (keys[0] & keys[1]) and not (keys[0] & keys[2]) and not (keys[1] & keys[2])


def test_hard_and_nonbio_pools_are_deduplicated_and_shuffled():
    pool = E.get_pool()
    for p in ("hard", "nonbio"):
        qs = pool[p]
        assert len({E.question_key(q) for q in qs}) == len(qs)
        assert qs == sorted(qs, key=E._order_key)
        assert len({q["source"].split("/")[0] + q["source"].split("/")[-1] for q in qs[:40]}) >= 4  # mixed, not eval-by-eval


def test_split_of_is_registry_aware(reg):
    served_q = E.get_eval("mmlu_virology", "disc")[0]
    assert E.split_of(served_q) == "disc"
    twin = dict(served_q, source="elsewhere/file", qid="not-registered", choices=list(reversed(served_q["choices"])),
                answer=len(served_q["choices"]) - 1 - served_q["answer"], question=served_q["question"].upper() + "  ")
    assert E.split_of(twin) == "disc"  # same normalised stem -> same split, whatever (source, qid) hashes to
    # copies of WMDP dir questions that live in the (unregistered) pilot file are dir too
    pilot = jl(E.PILOT_FILE)
    wmdp_dir_keys = {E.question_key(q) for q in E.get_pool()["wmdp"]}
    hits = [q for q in pilot if E.question_key(q) in wmdp_dir_keys]
    assert len(hits) > 50
    assert all(E.split_of(q) == "dir" for q in hits)
    # the spec rule needs only (source, qid): minimal dicts work, and agree with the full dict
    assert E.split_of({"source": served_q["source"], "qid": served_q["qid"]}) == "disc"
    assert E.split_of({"source": "wmdp/bio", "qid": "t0005"}) == "dir"       # first 680 rows of wmdp_train
    assert E.split_of({"source": "wmdp/bio", "qid": "t0900"}) in ("disc", "rep")
    # an unknown question falls back to the pure rule
    unk = {"source": "toy", "qid": "toy123", "question": "A question that is in no registered file at all?",
           "choices": ["a", "b"], "answer": 0}
    assert E.split_of(unk) == E._raw_split(unk, E._wmdp_dir_ids())
    # no pilot question may be 'dir' while its twin is evaluated
    ev_keys = collections.defaultdict(set)
    for n in FILE_EVALS:
        for s in ("disc", "rep"):
            for q in E.get_eval(n, s):
                ev_keys[E.question_key(q)].add(s)
    for q in pilot:
        k = E.question_key(q)
        if k in ev_keys:
            assert E.split_of(q) in ev_keys[k]


# ============================================================================================== prompt length guard
@functools.lru_cache(maxsize=1)
def valid_rows():
    """((eval, row), ...) for every schema-valid row of every registered file (the registry's input, before any de-duplication or
    cap). Cached and shared: treat the rows as read-only."""
    out = []
    for n in FILE_EVALS:
        for src in E.EVALS[n].files:
            for q in jl(src.path):
                if (not src.source_prefix or q["source"].startswith(src.source_prefix)) and E.validate_question(q) is None:
                    out.append((n, q))
    return tuple(out)


def test_no_served_or_pooled_question_exceeds_the_prompt_cap(served):
    """Regression (review): the registry served the pre-existing LAB-Bench prompts unfiltered, up to 51k characters / 26k tokens;
    one such prompt OOMs Ctx.run / Ctx.hidden_last and loses the whole recipe. The legacy pipeline skipped > 3000 tokens."""
    cap = E.MAX_PROMPT_CHARS
    assert E.registry_report()["max_prompt_chars"] == cap == 7500 and E.MAX_CHARS <= cap  # builder cap never exceeds the registry cap
    longest = 0
    for (name, split), qs in served.items():
        for q in qs:
            longest = max(longest, E.prompt_chars(q))
            assert E.prompt_chars(q) <= cap, (name, split, q["qid"])
    for pname, qs in E.get_pool().items():
        for q in qs:
            assert E.prompt_chars(q) <= cap, (pname, q["qid"])
    assert 6000 < longest <= cap  # the cap is binding on the data (ProtocolQA), and nothing sits beyond it


def test_the_26k_token_cloning_prompt_and_its_siblings_are_gone():
    offenders = {"labbench_cloningscenarios_0026", "labbench_cloningscenarios_0002", "labbench_cloningscenarios_0032",
                 "labbench_cloningscenarios_0012", "labbench_protocolqa_0089", "labbench_protocolqa_0057",
                 "labbench_protocolqa_0032", "labbench_protocolqa_0106", "labbench_protocolqa_0074"}  # > 3000 gpt-oss / Qwen3 tokens
    in_pool = {q["qid"] for qs in E.get_pool().values() for q in qs}
    in_eval = {q["qid"] for n in E.EVALS for q in E.get_eval(n, "all")}
    assert not offenders & (in_pool | in_eval)
    removed = {r["qid"]: r for r in E.registry_report()["too_long"]}
    assert offenders <= set(removed)
    big = removed["labbench_cloningscenarios_0026"]
    assert big["chars"] > 50000 and big["split"] in E.SPLITS   # (a rep question now: its scenario family is split together)
    # it is the cap, not the cluster rule, that keeps the served evals safe: the family is served nowhere, however it is split
    assert not [q for s in ("disc", "rep") for q in E.get_eval("labbench_cloning", s) if E.prompt_chars(q) > E.MAX_PROMPT_CHARS]


def test_removed_questions_are_reported_complete_and_never_served(reg, served):
    rep = E.registry_report()
    assert {r["eval"] for r in rep["too_long"]} == {"labbench_protocolqa", "labbench_cloning"}  # nothing else exceeds the cap
    raw = {(q["source"], q["qid"]): q for _, q in valid_rows()}
    served_keys = {E.question_key(q) for qs in served.values() for q in qs}
    pool_keys = {E.question_key(q) for qs in E.get_pool().values() for q in qs}
    assert len({(r["source"], r["qid"]) for r in rep["too_long"]}) == len(rep["too_long"])
    for r in rep["too_long"]:
        q = raw[(r["source"], r["qid"])]
        assert r["chars"] == E.prompt_chars(q) > E.MAX_PROMPT_CHARS and r["split"] in E.SPLITS
        assert E.question_key(q) not in served_keys | pool_keys, r  # neither the question nor a twin of it is served
        assert E.split_of(q) == r["split"], r
    # conversely every eligible question that fits is kept: nothing is removed without being listed
    removed = {(r["source"], r["qid"]) for r in rep["too_long"]}
    kept = {(q["source"], q["qid"]) for n in FILE_EVALS for s in E.SPLITS for q in reg.kept[n][s]}
    dup = {(d["source"], d["qid"]) for d in rep["dropped"]}
    for key, q in raw.items():
        assert (key in kept) + (key in removed) + (key in dup) == 1, key
        if key in kept:
            assert E.prompt_chars(q) <= E.MAX_PROMPT_CHARS


def test_cap_comes_after_dedupe_and_split_assignment(tmp_path):
    """A too-long dir question is removed from the pool, but its copy in another eval is still a dropped duplicate (never
    served), and no other question changes its split: the cap must not run before the de-duplication / split assignment."""
    long_stem = "A very long direction-pool stem about enzymes " + "x" * 300
    reg = _mk(tmp_path, {"A": [("s/a", "dir", long_stem), ("s/a", "disc", "Short disc one?"), ("s/a", "rep", "Short rep one?")],
                         "B": [("s/b", "disc", long_stem.upper()), ("s/b", "rep", "Another short rep?")]}, max_chars=100)
    free = E._Registry(reg.evals, reg.root, frozenset(), max_chars=10 ** 6)
    assert reg.bucket_by_id == free.bucket_by_id and reg.bucket_by_key == free.bucket_by_key  # splits are cap-independent
    assert [r["split"] for r in reg.too_long] == ["dir"] and reg.too_long[0]["eval"] == "A" and not free.too_long
    assert [q["question"] for q in free.pools["hard"] if q["question"].startswith("A very long")] == [long_stem]
    assert not any(q["question"].lower().startswith("a very long") for q in reg.pools["hard"])  # removed from the pool ...
    for s in E.SPLITS:  # ... and its copy in B is not served either (it is dropped as duplicate_of_dir, as without the cap)
        assert not any(q["question"].lower().startswith("a very long") for q in reg.kept["B"][s])
    assert [d["reason"] for d in reg.report["dropped"]] == ["duplicate_of_dir"] == [d["reason"] for d in free.report["dropped"]]
    assert reg.bucket_by_key[E.norm_text(long_stem)] == "dir"  # split_of still protects the removed question's twins
    for name in "AB":
        r = reg.report["per_eval"][name]
        assert r["rows"] == r["n"] + r["invalid"] + r["dropped_duplicate"] + r["too_long"], (name, r)
    assert reg.report["per_eval"]["A"]["too_long"] == 1 and reg.report["max_prompt_chars"] == 100


def test_max_chars_must_be_a_positive_int(tmp_path):
    reg = _mk(tmp_path, {"A": [("s/a", "disc", "Stem one?")]})
    for bad in (0, -5, 1.5, True, None, "7500"):
        with pytest.raises(ValueError, match="max_chars"):
            E._Registry(reg.evals, reg.root, frozenset(), max_chars=bad)
    assert E.prompt_chars({"question": "abc", "choices": ["de", "f"]}) == 6


def cached_tokenizer(repo):
    """Tokenizer from the local HF cache (never the network, never weights); the test is skipped where it is not cached."""
    try:
        from huggingface_hub import snapshot_download
        from transformers import AutoTokenizer
        return AutoTokenizer.from_pretrained(snapshot_download(repo, local_files_only=True))
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"tokenizer {repo} is not in the local HF cache ({type(e).__name__})")


TOKENIZERS = [("gptoss-120b", "openai/gpt-oss-120b"), ("llama-3.1-8b-instruct", "meta-llama/Llama-3.1-8B-Instruct"),
              ("gemma-3-4b-it", "google/gemma-3-4b-it")]


@pytest.mark.parametrize("key,repo", TOKENIZERS)
def test_prompt_cap_bounds_the_framing_E_prompt_at_3000_tokens(key, repo):
    """MAX_PROMPT_CHARS is the tokenizer-independent stand-in for the legacy MAX_TOK = 3000 (scripts/gptoss_dir.py): every
    question that is kept builds a framing-E prompt of at most 3000 tokens in the gpt-oss, Llama-3 and Gemma-3 tokenizers; for
    gpt-oss (the legacy filter's tokenizer) the two filters keep exactly the same questions."""
    from sbx.adapters import get_adapter
    from sbx.framings import FRAMINGS
    tok, ad, f = cached_tokenizer(repo), get_adapter(key), FRAMINGS["E"]
    n_long = n_equiv = 0
    for name, q in valid_rows():
        chars = E.prompt_chars(q)
        if chars <= 4000:  # >= 1.8 characters per token on all registered text: at most ~2300 tokens, never near the limit
            continue
        n_long += 1
        n_tok = len(tok(ad.build(tok, q, f.sys_text, f.user_note, f.effort), add_special_tokens=False).input_ids)
        if chars <= E.MAX_PROMPT_CHARS:
            assert n_tok <= 3000, (key, q["qid"], chars, n_tok)
        if key == "gptoss-120b":
            assert (n_tok <= 3000) == (chars <= E.MAX_PROMPT_CHARS), (q["qid"], chars, n_tok)
            n_equiv += 1
    assert n_long > 80 and (key != "gptoss-120b" or n_equiv == n_long)


# ============================================================================================== clusters (shared context)
def _doc(seed, n=600):
    """Random text of n words from a 1000-word vocabulary (about 5 characters per word): unrelated documents share no 12-gram
    worth mentioning, a prefix / suffix of a document is shared context."""
    r = random.Random(seed)
    return " ".join("w%03d" % r.randrange(1000) for _ in range(n))


def _partition(labels, items=None):
    items = list(range(len(labels))) if items is None else items
    groups = collections.defaultdict(set)
    for lab, it in zip(labels, items):
        groups[lab].add(it)
    return {frozenset(g) for g in groups.values()}


def test_context_components_links_shared_context_only():
    a, b, c = _doc(1), _doc(2), _doc(3)
    assert E.context_components([a, b, c]) == [0, 1, 2]                                   # unrelated: three clusters
    assert E.context_components([a, a + " and an extra tail of words", b]) == [0, 0, 1]    # one stem contains the other
    assert E.context_components([a, a.replace(" ", "\n\t "), b]) == [0, 0, 1]             # whitespace does not matter
    assert E.context_components([a, a.upper(), b]) == [0, 1, 2]                           # case does (DNA / names are case-coded)
    assert E.context_components([]) == [] and E.context_components([a]) == [0]
    assert E.context_components([b, a, b]) == [0, 1, 0]                                   # labels: order of first appearance
    short = "A short stem that appears twice. " * 3                                        # < 300 distinct 12-grams: never linked
    assert len(E._char_ngrams(short, 12)) < 300 and E.context_components([short, short]) == [0, 1]


def test_context_components_thresholds():
    base = _doc(5, 1000)                                      # about 5000 characters
    part = base[:3000] + " " + _doc(6, 400)                   # shares about 60% of its text with `base`
    texts = [base, part]
    assert E.context_components(texts, min_overlap=0.5) == [0, 0]
    assert E.context_components(texts, min_overlap=0.8) == [0, 1]
    assert E.context_components(texts, min_overlap=0.5, min_shared=2000) == [0, 0]
    assert E.context_components(texts, min_overlap=0.5, min_shared=4000) == [0, 1]       # shares ~3000 12-grams, not 4000
    assert E.context_components(texts, k=24, min_overlap=0.5) == [0, 0]
    for bad in (dict(k=0), dict(min_shared=0), dict(min_overlap=0.0), dict(min_overlap=1.5)):
        with pytest.raises(ValueError):
            E.context_components(texts, **bad)


def test_context_components_chain_and_permutation_invariance():
    x1, x2, x3, x4 = (_doc(s, 400) for s in (11, 12, 13, 14))
    a, b, c, d = x1 + " " + x2, x2 + " " + x3, x3 + " " + x4, _doc(15, 400)               # a~b (share x2), b~c (x3), a, c unrelated
    comp = E.context_components([a, b, c, d], min_overlap=0.4)
    assert comp == [0, 0, 0, 1]                                                            # linked transitively
    assert E.context_components([a, c, d], min_overlap=0.4) == [0, 1, 2]                   # without b they are not
    texts = [a, b, c, d, _doc(16, 400), x1]
    base = _partition(E.context_components(texts, min_overlap=0.4), texts)
    r = random.Random(0)
    for _ in range(5):
        perm = texts[:]
        r.shuffle(perm)
        assert _partition(E.context_components(perm, min_overlap=0.4), perm) == base


def _naive_partition(texts, k, min_shared, min_overlap):
    """Independent O(n^2) reference: pairwise set intersections + a flood fill."""
    sets = []
    for t in texts:
        t = " ".join(t.split())
        sets.append({t[i:i + k] for i in range(len(t) - k + 1)})
    adj = collections.defaultdict(set)
    for i, j in itertools.combinations(range(len(texts)), 2):
        m = min(len(sets[i]), len(sets[j]))
        c = len(sets[i] & sets[j])
        if m >= min_shared and c >= min_shared and c >= min_overlap * m:
            adj[i].add(j)
            adj[j].add(i)
    seen, out = set(), set()
    for s in range(len(texts)):
        if s in seen:
            continue
        stack, comp = [s], set()
        while stack:
            x = stack.pop()
            if x in comp:
                continue
            comp.add(x)
            stack.extend(adj[x] - comp)
        seen |= comp
        out.add(frozenset(comp))
    return out


def test_context_components_matches_a_naive_reference_on_random_corpora():
    r = random.Random(7)
    fams = [_doc(100 + i, 300) for i in range(6)]
    for trial in range(4):
        texts = []
        for _ in range(24):
            f = r.choice(fams).split()
            a, b = sorted((r.randrange(len(f)), r.randrange(len(f))))
            piece = " ".join(f[a:b + 1])
            texts.append(piece + " " + _doc(r.randrange(10 ** 6), r.randrange(0, 150)) if r.random() < 0.5 else piece)
        for k, ms, ov in ((12, 300, 0.5), (12, 100, 0.8), (8, 200, 0.3)):
            assert _partition(E.context_components(texts, k, ms, ov)) == _naive_partition(texts, k, ms, ov), (trial, k, ms, ov)


def test_clustered_synthetic_eval_splits_clusters_together_and_the_flag_matters(tmp_path):
    ctx1, ctx2 = _doc(21, 500), _doc(22, 500)       # two contexts; three questions about the first, two about the second
    src = "s/c"

    def qid_for(prefix, want):
        i = 0
        while True:
            qid = f"{prefix}{i}"
            if ("dir" if (b := zlib.crc32(f"{src}|{qid}".encode()) % 100) < 25 else "disc" if b < 63 else "rep") == want:
                return qid
            i += 1
    rows = [(qid_for("a", "rep"), ctx1 + " Question one?"), (qid_for("b", "dir"), ctx1 + " Question two?"),
            (qid_for("c", "disc"), ctx1 + " Another question?"), (qid_for("d", "dir"), ctx2 + " First?"),
            (qid_for("e", "rep"), ctx2 + " Second?"), (qid_for("f", "disc"), _doc(23, 500) + " Unrelated?")]

    def build(clustered):
        (tmp_path / "C.jsonl").write_text("\n".join(json.dumps({"source": src, "qid": qid, "question": stem, "choices": ["a", "b", "c"],
                                                                 "answer": 1}) for qid, stem in rows) + "\n")
        spec = E.EvalSpec("C", "bio", False, (E.Src("C.jsonl"),), pool="hard", clustered=clustered)
        return E._Registry({"C": spec}, str(tmp_path), frozenset())
    lit, clu = build(False), build(True)
    split = lambda reg_, i: reg_.bucket_by_id[(src, rows[i][0])]  # noqa: E731
    assert [split(lit, i) for i in range(6)] == ["rep", "dir", "disc", "dir", "rep", "disc"]      # the literal rule scatters them
    assert [split(clu, i) for i in range(6)] == ["rep", "rep", "rep", "dir", "dir", "disc"]       # representative = lowest qid
    assert not lit.cluster_rep and clu.report["clusters"]["C"] == {"questions": 6, "clusters": 3, "multi": 2, "largest": 3}
    assert clu.cluster_rep[(src, rows[2][0])] == (src, rows[0][0]) and clu.cluster_rep[(src, rows[4][0])] == (src, rows[3][0])
    assert clu.cluster_rep[(src, rows[5][0])] == (src, rows[5][0])                                # a singleton is its own cluster
    assert sum(clu.report["per_eval"]["C"][s] for s in E.SPLITS) == 6                              # nothing lost, only moved
    wmdp = dict(source=E.WMDP_SOURCE, qid="t1", question=ctx1, choices=["a", "b"], answer=0)
    (tmp_path / "W.jsonl").write_text(json.dumps(wmdp) + "\n")
    with pytest.raises(ValueError, match="cannot contain"):                                         # their dir pool is fixed
        E._Registry({"W": E.EvalSpec("W", "bio", False, (E.Src("W.jsonl"),), clustered=True)}, str(tmp_path), frozenset())


def _grams(text):
    t = " ".join(text.split())
    return {t[i:i + E.CLUSTER_K] for i in range(len(t) - E.CLUSTER_K + 1)}


def _linked(a, b):
    """The documented link rule, re-implemented here (independently of E.context_components)."""
    m = min(len(a), len(b))
    if m < E.CLUSTER_MIN_SHARED:
        return False
    c = len(a & b)
    return c >= E.CLUSTER_MIN_SHARED and c >= E.CLUSTER_MIN_OVERLAP * m


def test_clustered_evals_have_cluster_disjoint_splits():
    """Regression (review): with the literal crc rule 80-94% of the ProtocolQA / CloningScenarios rep questions shared their
    protocol / plasmid with a disc question (and most served ones with a dir question), so a discovery fluke 'replicated' and
    directions were evaluated on the text they were built from. No two questions that share context may be in different splits
    (checked on every schema-valid row, including the ones the length cap removed, against a naive pairwise reference)."""
    stats = {}
    for name in CLUSTERED:
        rows = [q for n, q in valid_rows() if n == name]
        gs = [_grams(q["question"]) for q in rows]
        splits = [E.split_of(q) for q in rows]
        own = [E._raw_split(q, frozenset()) for q in rows]            # what the literal crc rule would have said
        linked = apart = apart_literal = 0
        for i, j in itertools.combinations(range(len(rows)), 2):
            if _linked(gs[i], gs[j]):
                linked += 1
                apart += splits[i] != splits[j]
                apart_literal += own[i] != own[j]
        stats[name] = (linked, apart, apart_literal)
        assert apart == 0, (name, apart)
    assert stats["labbench_protocolqa"][0] > 400 and stats["labbench_protocolqa"][2] > 100   # not vacuous: the literal rule failed
    assert stats["labbench_cloning"][0] > 150 and stats["labbench_cloning"][2] > 30
    assert stats["labbench_seqqa"][0] >= 4 and stats["labbench_seqqa"][2] >= 1


def test_protocolqa_clusters_match_the_protocol_titles():
    """Oracle that does not use the n-gram rule: ProtocolQA stems start with 'Protocol:\\n<title>'. Questions with one title
    are about one protocol and must share a cluster and a split; different titles never share a cluster (the generic title
    'Section:' covers two different protocols)."""
    rows = [q for n, q in valid_rows() if n == "labbench_protocolqa"]
    title = {q["qid"]: q["question"].split("\n")[1].strip() for q in rows}
    assert len(rows) == 108 and all(q["question"].startswith("Protocol:\n") for q in rows)
    by_title, by_cluster = collections.defaultdict(list), collections.defaultdict(set)
    for q in rows:
        by_title[title[q["qid"]]].append(q)
        by_cluster[E.cluster_of(q)].add(title[q["qid"]])
    assert len(by_title) == 24 and len({E.cluster_of(q) for q in rows}) == 25
    for t, qs in by_title.items():
        if len(t) > 12:   # not the generic 'Section:'
            assert len({E.split_of(q) for q in qs}) == 1 == len({E.cluster_of(q) for q in qs}), t
    assert all(len(ts) == 1 for ts in by_cluster.values())
    assert max(len(qs) for qs in by_title.values()) == 16                                    # the 16-question protocol is one cluster


def test_cluster_partition_is_stable_around_the_parameters():
    """The chosen k / min_shared / overlap sit on a plateau of the data (not on an edge): moving any of them gives the same clusters."""
    for name in CLUSTERED:
        stems = [q["question"] for n, q in valid_rows() if n == name]
        base = _partition(E.context_components(stems))
        assert len(base) == {"labbench_protocolqa": 25, "labbench_cloning": 5, "labbench_seqqa": 196}[name]
        for kw in (dict(k=8), dict(k=24), dict(min_shared=200), dict(min_shared=500), dict(min_overlap=0.4), dict(min_overlap=0.65)):
            assert _partition(E.context_components(stems, **kw)) == base, (name, kw)


def test_cluster_of_api(reg):
    q = E.get_eval("labbench_protocolqa", "rep")[0]
    cid = E.cluster_of(q)
    assert cid == "labbench/ProtocolQA|" + reg.cluster_rep[(q["source"], q["qid"])][1] and cid.startswith(q["source"] + "|")
    same = [x for x in E.get_eval("labbench_protocolqa", "rep") if E.cluster_of(x) == cid]
    assert all(E.split_of(x) == "rep" for x in same)
    other = E.get_eval("gpqa_bio", "rep")[0]
    assert E.cluster_of(other) == f"{other['source']}|{other['qid']}"                          # unclustered eval: its own cluster
    assert E.cluster_of({"source": "toy", "qid": "t1"}) == "toy|t1"                            # unregistered: its own cluster
    for n in CLUSTERED:                                                                        # members of a cluster share one id
        by = collections.defaultdict(list)
        for s in ("disc", "rep"):
            for x in E.get_eval(n, s):
                by[E.cluster_of(x)].append(s)
        assert all(len(set(v)) == 1 for v in by.values()), n
    stats = E.registry_report()["clusters"]
    assert {k: v["clusters"] for k, v in stats.items()} == {"labbench_seqqa": 196, "labbench_protocolqa": 25, "labbench_cloning": 5}


def test_residual_context_sharing_in_the_unclustered_evals_is_small():
    """The unclustered evals (WMDP-bio excluded: its dir pool is fixed) contain a few passages shared by 2-4 questions. Bound them,
    so that a new templated / shared-context set added without clustered=True fails here: it would show up as large clusters
    that span splits. Near-identical context here means >= 80% overlap (template boilerplate, e.g. WMDP-cyber's generated C
    programs, is not shared context and is below it)."""
    reg = E._registry()
    spanning_q = 0
    for n in FILE_EVALS:
        if E.EVALS[n].clustered or n == "wmdp_bio":
            continue
        qs = [(s, q) for s in E.SPLITS for q in reg.kept[n][s]]
        comp = E.context_components([q["question"] for _, q in qs], E.CLUSTER_K, E.CLUSTER_MIN_SHARED, 0.8)
        by = collections.defaultdict(list)
        for c, (s, q) in zip(comp, qs):
            by[c].append(s)
        span = [v for v in by.values() if len(set(v)) > 1]
        assert max((len(v) for v in by.values()), default=1) <= 4, n
        assert sum(len(v) for v in span) <= 0.05 * len(qs) + 1, (n, span)
        spanning_q += sum(len(v) for v in span)
    assert spanning_q <= 30   # 24 on the registered data (MMLU passages shared by 2-3 questions)
    wm = [q["question"] for s in E.SPLITS for q in reg.kept["wmdp_bio"][s]]
    assert len(set(E.context_components(wm))) == len(wm)       # no two WMDP-bio stems share context at all


# ============================================================================================== API behaviour
def test_get_eval_api(served):
    with pytest.raises(KeyError):
        E.get_eval("nope")
    for bad in ("dir", "train", "", None):
        with pytest.raises(ValueError):
            E.get_eval("wmdp_bio", bad)
    assert E.get_eval("gpqa_bio") == E.get_eval("gpqa_bio", "all")  # default split is "all"
    for n in E.EVALS:
        d, r, a = (served[(n, s)] for s in ("disc", "rep", "all"))
        assert sorted(q["qid"] for q in a) == sorted(q["qid"] for q in d + r)
        assert len({q["qid"] for q in a}) == len(a) == len(d) + len(r)
        if n in NO_DISC_OR_REP:  # documented degenerate eval (its long questions are capped): pinned so a change is noticed
            assert E.split_sizes(n) == NO_DISC_OR_REP[n] and (len(d), len(r)) == (0, 6), n
        else:
            assert d and r, n


def test_union_view_is_the_union_of_its_members(served):
    members = E.EVALS["mmlu_bio_all"].union
    assert members[0] == "mmlu_bio" and set(MMLU_SETS) <= set(members) and len(members) == 10
    for split in ("disc", "rep", "all"):
        want = sorted((q["source"], q["qid"]) for m in members for q in served[(m, split)])
        got = sorted((q["source"], q["qid"]) for q in served[("mmlu_bio_all", split)])
        assert got == want
    sz = E.split_sizes("mmlu_bio_all")
    for s in E.SPLITS:
        assert sz[s] == sum(E.split_sizes(m)[s] for m in members)
    assert E.registry_report()["per_eval"]["mmlu_bio_all"]["rows"] == sum(E.registry_report()["per_eval"][m]["rows"] for m in members)


def test_registry_report_is_a_copy():
    r = E.registry_report()
    r["invalid"].clear()
    r["per_eval"]["wmdp_bio"]["n"] = -1
    assert len(E.registry_report()["invalid"]) == 5 and E.registry_report()["per_eval"]["wmdp_bio"]["n"] == 1273


def test_returned_questions_are_copies():
    a = E.get_eval("gpqa_bio", "disc")
    a[0]["choices"].append("junk")
    a[0]["question"] = "changed"
    a.pop()
    b = E.get_eval("gpqa_bio", "disc")
    assert len(b) == len(a) + 1 and "junk" not in b[0]["choices"] and b[0]["question"] != "changed"
    p = E.get_pool()
    p["hard"][0]["choices"].append("junk")
    p["wmdp"].clear()
    q = E.get_pool()
    assert len(q["wmdp"]) == 680 and "junk" not in q["hard"][0]["choices"]


def test_served_order_is_a_fixed_pseudo_random_order(served):
    qs = served[("mmlu_nonbio", "disc")]
    assert qs == sorted(qs, key=E._order_key)
    assert len({q["source"] for q in qs[:25]}) >= 5  # a prefix (--max-per-split) mixes all 8 subjects
    allq = served[("wmdp_bio", "all")]
    assert allq == sorted(allq, key=E._order_key)
    assert [q["qid"] for q in allq[:30]] != sorted(q["qid"] for q in allq)[:30]


def test_eval_profile_and_docs_helpers():
    p = E.eval_profile("pubmedqa")
    assert p["min_options"] == p["max_options"] == 3 and abs(p["chance"] - 1 / 3) < 1e-9 and p["top_letter"] == "A"
    assert p["top_share"] > 0.45
    p = E.eval_profile("medxpertqa_text")
    assert (p["min_options"], p["max_options"]) == (10, 10)
    with pytest.raises(KeyError):
        E.eval_profile("nope")
    with pytest.raises(KeyError):
        E.split_sizes("nope")


# ============================================================================================== de-duplication policy
def test_norm_text_and_question_key():
    assert E.norm_text("  Which   of THESE, is (true)?  ") == "which of these is true"
    assert E.norm_text("Which of these is true") == E.norm_text("which of these... is TRUE?!")
    assert E.norm_text("α-helix") != E.norm_text("β-helix")  # greek letters are kept
    assert E.norm_text("pH 3.5") != E.norm_text("pH 35")
    assert E.norm_text("ＡＢ") == "ab"  # NFKC folds full-width letters
    a = {"question": "Why is the sky blue?", "choices": ["x", "y"], "answer": 0}
    b = {"question": "why is the sky  BLUE", "choices": ["p", "q", "r"], "answer": 2}
    assert E.question_key(a) == E.question_key(b)  # options are not part of the identity


def test_resolve_group_policy():
    assert E.resolve_group(["disc"]) == ("disc", [0])
    assert E.resolve_group(["rep", "disc", "rep"]) == ("rep", [0])          # first member wins when no dir member
    assert E.resolve_group(["disc", "dir"]) == ("dir", [1])                 # dir wins even when it comes later
    assert E.resolve_group(["dir", "dir", "rep"]) == ("dir", [0, 1])        # dir members are never dropped
    assert E.resolve_group(["rep", "dir", "disc", "dir"]) == ("dir", [1, 3])


def _find_qid(source, want, start=0):
    i = start
    while True:
        qid = f"q{i}"
        b = zlib.crc32(f"{source}|{qid}".encode()) % 100
        if ("dir" if b < 25 else "disc" if b < 63 else "rep") == want:
            return qid
        i += 1


def _mk(tmp_path, files, max_chars=E.MAX_PROMPT_CHARS):
    """files: {name: [(source, want_split, stem_text), ...]} -> registry over synthetic jsonl files."""
    specs, used = {}, set()
    for name, rows in files.items():
        out = []
        for i, (source, want, stem) in enumerate(rows):
            qid = _find_qid(source, want, start=i * 7)
            while (source, qid) in used:
                qid = _find_qid(source, want, start=int(qid[1:]) + 1)
            used.add((source, qid))
            out.append({"source": source, "qid": qid, "question": stem, "choices": ["a", "b", "c"], "answer": 1})
        (tmp_path / f"{name}.jsonl").write_text("\n".join(json.dumps(r) for r in out) + "\n")
        specs[name] = E.EvalSpec(name, "bio", False, (E.Src(f"{name}.jsonl"),), pool="hard" if name != "C" else None)
    return E._Registry(specs, str(tmp_path), frozenset(), max_chars)


def test_dedupe_policy_on_a_synthetic_registry(tmp_path):
    reg = _mk(tmp_path, {
        "A": [("s/a", "dir", "Stem one about enzymes?"), ("s/a", "disc", "Stem two about ribosomes?"),
              ("s/a", "rep", "Stem three about viruses?"), ("s/a", "dir", "Unique A dir"), ("s/a", "disc", "Unique A disc"),
              ("s/a", "rep", "Stem five a?")],
        "B": [("s/b", "disc", "STEM ONE about enzymes"), ("s/b", "rep", "stem two about ribosomes"),
              ("s/b", "dir", "Stem three: about viruses!"), ("s/b", "dir", "Stem four dir twice"),
              ("s/b", "dir", "Stem four dir twice?"), ("s/b", "disc", "Unique B disc")],
        "C": [("s/c", "rep", "Stem one about enzymes"), ("s/c", "dir", "Stem five a"), ("s/c", "rep", "Unique C rep")],
    })
    k = {n: {s: [q["question"] for q in reg.kept[n][s]] for s in E.SPLITS} for n in "ABC"}
    # group 1: dir member (A) wins over disc (B) and rep (C)
    assert "Stem one about enzymes?" in k["A"]["dir"]
    assert not any("one about enzymes" in x.lower() for x in k["B"]["disc"] + k["C"]["rep"])
    # group 2: no dir member -> first (A, disc) wins over B (rep)
    assert "Stem two about ribosomes?" in k["A"]["disc"] and not k["B"]["rep"]
    # group 3: A rep + B dir -> the dir member is kept, the rep one is dropped
    assert "Stem three: about viruses!" in k["B"]["dir"] and "Stem three about viruses?" not in k["A"]["rep"]
    # group 4: two dir members of one group are both kept (never dropped)
    assert {"Stem four dir twice", "Stem four dir twice?"} <= set(k["B"]["dir"])
    # group 5: A rep first, C dir -> dir wins
    assert "Stem five a" in k["C"]["dir"] and "Stem five a?" not in k["A"]["rep"]
    # unique questions are untouched
    assert "Unique A dir" in k["A"]["dir"] and "Unique A disc" in k["A"]["disc"] and "Unique B disc" in k["B"]["disc"]
    assert "Unique C rep" in k["C"]["rep"]
    rep = reg.report
    assert rep["duplicate_groups"] == 5 and rep["duplicate_groups_cross_set"] == 4 and rep["duplicate_groups_within_set"] == 1
    assert sorted(d["reason"] for d in rep["dropped"]) == ["duplicate", "duplicate_of_dir", "duplicate_of_dir",
                                                         "duplicate_of_dir", "duplicate_of_dir"]
    assert rep["rows"] == 15 and sum(r["n"] for r in rep["per_eval"].values()) == 15 - len(rep["dropped"])
    # the bucket lookup used by split_of
    assert reg.bucket_by_key[E.norm_text("Stem two about ribosomes")] == "disc"
    assert reg.bucket_by_key[E.norm_text("stem ONE about enzymes")] == "dir"
    # pools: "hard" holds the de-duplicated dir questions of A and B only (C has no pool)
    assert len(reg.pools["hard"]) == len({E.question_key(q) for q in reg.pools["hard"]})
    assert {q["source"] for q in reg.pools["hard"]} <= {"s/a", "s/b"}


def test_registry_rejects_malformed_data(tmp_path):
    spec = lambda n: E.EvalSpec(n, "bio", False, (E.Src(f"{n}.jsonl"),))  # noqa: E731
    row = {"source": "s", "qid": "1", "question": "Q one?", "choices": ["a", "b"], "answer": 0}
    (tmp_path / "bad.jsonl").write_text(json.dumps(dict(row, answer=5)) + "\n")
    with pytest.raises(ValueError, match="structure"):
        E._Registry({"bad": spec("bad")}, str(tmp_path), frozenset())
    (tmp_path / "x.jsonl").write_text(json.dumps(row) + "\n")
    (tmp_path / "y.jsonl").write_text(json.dumps(dict(row, question="Another one?")) + "\n")
    with pytest.raises(ValueError, match="appears in eval"):  # same (source, qid) in two evals
        E._Registry({"x": spec("x"), "y": spec("y")}, str(tmp_path), frozenset())
    (tmp_path / "j.jsonl").write_text("{not json}\n")
    with pytest.raises(ValueError, match="invalid JSON"):
        E._Registry({"j": spec("j")}, str(tmp_path), frozenset())
    (tmp_path / "ok.jsonl").write_text(json.dumps(dict(row, choices=["a", "a"])) + "\n")  # semantic problem: dropped, listed
    reg = E._Registry({"ok": spec("ok")}, str(tmp_path), frozenset())
    assert reg.report["invalid"][0]["reason"] == "choices: duplicated options" and reg.report["per_eval"]["ok"]["n"] == 0


def test_cross_set_duplicates_are_reported_and_documented():
    rep = E.registry_report()
    docs = open(os.path.join(ROOT, "docs", "evals.md"), encoding="utf-8").read()
    for label, value in (("cross-set duplicate groups", rep["duplicate_groups_cross_set"]),
                         ("within-set duplicate groups", rep["duplicate_groups_within_set"]),
                         ("duplicate questions dropped", len(rep["dropped"])),
                         ("questions removed by the prompt-length cap", len(rep["too_long"])),
                         ("invalid rows dropped", len(rep["invalid"]))):
        m = re.search(rf"{label}: \*\*(\d+)\*\*", docs, re.I)
        assert m, f"docs/evals.md must state '{label}: **N**'"
        assert int(m.group(1)) == value, (label, m.group(1), value)
    # the three documented kinds of duplicate: mmlu_bio <-> mmlupro_bio, templated SeqQA stems, repeated stems in a file
    assert rep["duplicate_groups_cross_set"] == 7
    assert {d["eval"] for d in rep["dropped"]} == {"mmlu_bio", "mmlupro_bio", "labbench_seqqa"}
    # the documented parameters of the two guards are the ones in the code
    assert re.search(rf"MAX_PROMPT_CHARS = {E.MAX_PROMPT_CHARS}\b", docs)
    assert f"{E.CLUSTER_MIN_SHARED} distinct character {E.CLUSTER_K}-grams" in docs
    assert "plateau" in docs and "cluster_of" in docs and "clustered=True" in docs


# ============================================================================================== determinism
_DIGEST_SNIPPET = r"""
import hashlib, json, sys
from sbx import evals as E
h = hashlib.sha256()
for n in E.EVALS:
    for s in ("disc", "rep", "all"):
        h.update(json.dumps([n, s, [q["qid"] for q in E.get_eval(n, s)]]).encode())
for k, qs in E.get_pool().items():
    h.update(json.dumps([k, [q["qid"] for q in qs]]).encode())
for q in E.get_eval("pubmedqa", "all")[:50]:
    h.update(E.split_of(q).encode())
print(h.hexdigest())
"""


def _digest_in_subprocess(hashseed):
    env = dict(os.environ, PYTHONHASHSEED=str(hashseed), USE_TF="0", USE_TORCH="1")
    out = subprocess.run([sys.executable, "-c", _DIGEST_SNIPPET], cwd=ROOT, env=env, capture_output=True, text=True, timeout=80)
    assert out.returncode == 0, out.stderr[-2000:]
    return out.stdout.strip()


def test_determinism_across_processes():
    h = hashlib.sha256()
    for n in E.EVALS:
        for s in ("disc", "rep", "all"):
            h.update(json.dumps([n, s, [q["qid"] for q in E.get_eval(n, s)]]).encode())
    for k, qs in E.get_pool().items():
        h.update(json.dumps([k, [q["qid"] for q in qs]]).encode())
    for q in E.get_eval("pubmedqa", "all")[:50]:
        h.update(E.split_of(q).encode())
    here = h.hexdigest()
    assert _digest_in_subprocess(0) == here
    assert _digest_in_subprocess(12345) == here


# ============================================================================================== builder helpers
def _cands(n, prefix="c", source="s/x"):
    return [{"source": source, "qid": f"{prefix}{i:03d}", "question": f"Candidate question number {i} about topic {prefix}?",
             "choices": ["a", "b", "c", "d"], "answer": i % 4} for i in range(n)]


def test_stable_sample_is_deterministic_and_order_independent():
    items = _cands(100)
    a = E.stable_sample(items, 30, "lbl")
    assert a == E.stable_sample(list(reversed(items)), 30, "lbl")
    assert len(a) == 30 and len({q["qid"] for q in a}) == 30
    assert E.stable_sample(items, 30, "other") != a
    assert E.stable_sample(items, 500, "lbl") == sorted(items, key=lambda q: E._rank("lbl", q["qid"]))
    # the first 5 of a 30-sample are the 5 best-ranked of the whole pool (prefix-stable)
    assert [q["qid"] for q in E.stable_sample(items, 5, "lbl")] == [q["qid"] for q in a[:5]]


def test_stratified_sample_allocation():
    items = [q for s in "abcdef" for q in _cands(200, prefix=s, source=f"s/{s}")]
    out = E.stratified_sample(items, 400, "x", lambda q: q["source"])
    cnt = collections.Counter(q["source"] for q in out)
    assert sorted(cnt.values()) == [66, 66, 67, 67, 67, 67] and len(out) == 400
    assert out == E.stratified_sample(list(reversed(items)), 400, "x", lambda q: q["source"])
    # small strata are fully used and their unused quota is redistributed
    items = _cands(10, "a", "s/a") + _cands(500, "b", "s/b") + _cands(500, "c", "s/c")
    cnt = collections.Counter(q["source"] for q in E.stratified_sample(items, 100, "x", lambda q: q["source"]))
    assert cnt["s/a"] == 10 and cnt["s/b"] + cnt["s/c"] == 90 and abs(cnt["s/b"] - cnt["s/c"]) <= 1
    assert len(E.stratified_sample(items, 5000, "x", lambda q: q["source"])) == len(items)


def test_make_set_filters_dedupes_and_orders():
    cands = _cands(50)
    cands.append(dict(cands[3], qid="dup-of-3"))                                   # within-set duplicate
    cands.append(dict(cands[4], qid="bad", choices=["a", "a", "b", "c"]))             # invalid options
    cands.append({"source": "s/x", "qid": "long", "question": "x" * 7000, "choices": ["a", "b"], "answer": 0})
    excluded = {E.question_key(cands[0]), E.question_key(cands[1])}                  # used by an earlier set
    out, st = E.make_set("demo", cands, 20, None, excluded)
    assert len(out) == 20 and st["written"] == 20
    assert st["duplicate_of_earlier_set"] == 2 and st["duplicate_within_set"] == 1 and st["too_long"] == 1
    assert st["invalid: choices: duplicated options"] == 1 and st["candidates"] == 53 and st["eligible"] == 48
    assert not {E.question_key(q) for q in out} & excluded
    assert len({E.question_key(q) for q in out}) == 20
    ranks = [E._rank("demo|order", q["qid"]) for q in out]
    assert ranks == sorted(ranks)
    out2, _ = E.make_set("demo", list(reversed(cands)), 20, None, excluded)
    assert [q["qid"] for q in out2] == [q["qid"] for q in out]                       # input order is irrelevant
    out3, _ = E.make_set("demo", cands, 20, lambda q: q["qid"][:2], excluded)
    assert len(out3) == 20


def test_write_jsonl_is_ascii_and_roundtrips(tmp_path):
    qs = [{"source": "s", "qid": "1", "question": "β-galactosidase – café?", "choices": ["µm", "nm"], "answer": 0}]
    p = str(tmp_path / "o.jsonl")
    E.write_jsonl(p, qs)
    raw = open(p, "rb").read()
    assert all(b < 128 for b in raw) and raw.endswith(b"\n")
    assert [json.loads(line) for line in raw.decode().splitlines()] == qs
    assert not os.path.exists(p + ".tmp")


def test_converters():
    m = E.conv_mmlu({"question": " Q? ", "choices": [" a", "b ", "c", "d"], "answer": 2}, "virology", "test", 7)
    assert m == {"source": "mmlu/virology", "qid": "mmlu_virology_test_0007", "question": "Q?", "choices": ["a", "b", "c", "d"],
                 "answer": 2}
    w = E.conv_wmdp({"question": "Q?", "choices": ["a", "b", "c", "d"], "answer": 1}, "cyber", 12)
    assert w["source"] == "wmdp/cyber" and w["qid"] == "wmdp_cyber_0012" and w["answer"] == 1
    opts = {"A": "w", "B": "x", "C": "y", "D": "z"}
    q = E.conv_medqa({"question": "Vignette?", "options": opts, "answer": "y", "answer_idx": "C"}, 5)
    assert q["answer"] == 2 and q["choices"] == ["w", "x", "y", "z"] and q["qid"] == "medqa_test_0005"
    assert E.conv_medqa({"question": "V", "options": opts, "answer": "w", "answer_idx": "C"}, 5) is None  # key/text mismatch
    assert E.conv_medqa({"question": "V", "options": {"A": "1", "B": "2"}, "answer": "1", "answer_idx": "A"}, 5) is None
    row = {"id": "ab12cd34-aaaa-bbbb-cccc-dddddddddddd", "question": "Q?", "opa": "1", "opb": "2", "opc": "3", "opd": "4",
           "cop": 3, "choice_type": "single", "subject_name": "Anatomy"}
    mm = E.conv_medmcqa(row)
    assert mm["source"] == "medmcqa/anatomy" and mm["qid"] == "medmcqa_ab12cd34aaaa" and mm["answer"] == 3
    assert E.conv_medmcqa(dict(row, choice_type="multi")) is None and E.conv_medmcqa(dict(row, subject_name="Dental")) is None
    pm = E.conv_pubmedqa({"pubid": 99, "question": "Does X help?", "final_decision": "maybe",
                          "context": {"labels": ["BACKGROUND", "RESULTS"], "contexts": ["bg text", "res text"]}})
    assert pm["answer"] == 2 and pm["choices"] == ["yes", "no", "maybe"] and pm["qid"] == "pubmedqa_99"
    assert pm["question"] == "Abstract:\nBACKGROUND: bg text\nRESULTS: res text\n\nQuestion: Does X help?"
    assert E.conv_pubmedqa({"pubid": 1, "question": "q", "final_decision": "n/a", "context": {"labels": [], "contexts": []}}) is None
    sg = {"uuid": "0123456789abcdef0123", "question": "Q?", "options": ["a", "b", "c"], "answer": "b", "answer_letter": "B",
          "discipline": "Science", "field": "Biology", "difficulty": "hard", "is_calculation": False}
    assert E.conv_supergpqa(sg)["answer"] == 1 and E.conv_supergpqa(sg)["qid"] == "supergpqa_0123456789ab"
    assert E.conv_supergpqa(dict(sg, difficulty="easy")) is None and E.conv_supergpqa(dict(sg, is_calculation="True")) is None
    assert E.conv_supergpqa(dict(sg, field="Physics")) is None and E.conv_supergpqa(dict(sg, answer="zzz")) is None
    mx = E.conv_medxpertqa({"id": "Text-12", "question": "Stem here?\nAnswer Choices: (A) one (B) two", "label": "B",
                            "options": {"A": "one", "B": "two"}})
    assert mx["question"] == "Stem here?" and mx["answer"] == 1 and mx["qid"] == "medxpertqa_text_0012"
    assert E.conv_medxpertqa({"id": "Text-1", "question": "Q", "label": "C", "options": {"A": "1", "B": "2"}}) is None
    pro = E.conv_mmlupro({"question_id": 70, "question": "Q?", "options": ["a", "b", "c"], "answer_index": 2,
                          "category": "computer science"})
    assert pro["source"] == "mmlupro/computer_science" and pro["qid"] == "mmlupro_computer_science_00070" and pro["answer"] == 2
    assert E.conv_mmlupro({"question_id": 1, "question": "Q", "options": ["a"], "answer_index": 3, "category": "law"}) is None
    for q in (m, w, q, mm, pm, sg, mx, pro):
        assert q is not None


# ============================================================================================== reports, docs, CLI
def test_near_duplicate_report_is_consistent():
    lo, hi = E.near_duplicate_report(0.5), E.near_duplicate_report(0.8)
    assert lo["total_served"] == hi["total_served"] == sum(len(E.get_eval(n, "all")) for n in FILE_EVALS)
    assert "mmlu_bio_all" not in lo["per_eval"]
    assert hi["total_near_dup"] <= lo["total_near_dup"]
    for n, v in lo["per_eval"].items():
        assert v["served"] == len(E.get_eval(n, "all")) and 0 <= v["near_dup_of_pool"] <= v["served"]
        assert hi["per_eval"][n]["near_dup_of_pool"] <= v["near_dup_of_pool"]
    # exact duplicates were removed, so WMDP-bio (clusters of topic-related, not textually near-identical, stems) stays low
    assert lo["per_eval"]["wmdp_bio"]["near_dup_of_pool"] < 0.05 * lo["per_eval"]["wmdp_bio"]["served"]


def test_docs_contain_the_generated_block():
    docs = open(os.path.join(ROOT, "docs", "evals.md"), encoding="utf-8").read()
    m = re.search(r"<!-- BEGIN GENERATED[^>]*-->\n(.*?)<!-- END GENERATED -->", docs, re.S)
    assert m, "docs/evals.md must contain the BEGIN/END GENERATED markers"
    assert m.group(1) == E.render_docs_block(), "docs/evals.md is stale: run `python3 -m sbx.evals docs` and paste the block"
    for n in E.EVALS:
        assert f"`{n}`" in docs


def test_cli_commands(capsys):
    assert E.main(["check"]) == 0
    out = capsys.readouterr().out
    assert all(n in out for n in E.EVALS)
    assert E.main(["docs"]) == 0
    assert capsys.readouterr().out == E.render_docs_block()
    assert E.main(["counts"]) == 0
    counts = capsys.readouterr().out
    for n, s in E.EVALS.items():
        assert f"{n!r}: {s.n}," in counts
    assert E.main(["build", "--only", "no_such_eval"]) == 2
