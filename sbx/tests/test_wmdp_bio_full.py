"""The unsteered-only WMDP-bio set: every question (evaluation + direction pool), served by get_eval under its own name, and never steered.
The guard is the point: an instrument judged on the questions its direction was built from would be circular."""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from sbx import evals as E
from sbx import run as R


def _key(q):
    return (q["source"], q["qid"])


def test_full_set_is_exactly_the_evaluation_questions_plus_the_pool():
    full = E.get_eval("wmdp_bio_full", "all")
    served, pool = E.get_eval("wmdp_bio", "all"), E.get_pool()["wmdp"]
    assert len(full) == len(served) + len(pool) == 1273
    assert {_key(q) for q in full if q["pool_part"] == "eval"} == {_key(q) for q in served}
    assert {_key(q) for q in full if q["pool_part"] == "dir"} == {_key(q) for q in pool}
    assert len({_key(q) for q in full}) == 1273                       # no question twice
    assert all({"source", "qid", "question", "choices", "answer"} <= set(q) for q in full)
    assert [_key(q) for q in full] == [_key(q) for q in E.get_eval("wmdp_bio_full", "all")]     # a fixed order


def test_the_ordinary_eval_still_serves_no_direction_pool_question():
    served = {_key(q) for q in E.get_eval("wmdp_bio", "all")}
    assert not (served & {_key(q) for q in E.get_pool()["wmdp"]})
    assert [len(E.get_eval("wmdp_bio", s)) for s in ("disc", "rep", "all")] == [291, 302, 593]


def test_the_full_set_has_no_discovery_or_replication_split():
    for split in ("disc", "rep"):
        with pytest.raises(ValueError, match="no discovery / replication split"):
            E.get_eval("wmdp_bio_full", split)


def _runner():
    r = R.Runner(SimpleNamespace(nl=4), SimpleNamespace(model="m", answer_mode="native", max_per_split=0, max_new=8), {"N": None}, E.get_eval, None)
    r.store = SimpleNamespace(has=lambda k: False, cells={}, put_qids=lambda *a: None)
    return r


@pytest.mark.parametrize("kwargs", [
    dict(direction="B", interv="proj0", layers=[2]),
    dict(direction="B", interv="add", layers=[2], strength=-0.2),
    dict(direction="rand0", interv="cap", layers=[2]),
    dict(direction="noise", interv="noise_act", layers=[2], strength=0.2),
    dict(interv="proj0", layers=[2]),                               # a layer set with no direction named
])
def test_the_runner_refuses_to_steer_on_the_full_set(kwargs):
    r = _runner()
    ran = []
    r.ctx = SimpleNamespace(run=lambda *a, **k: ran.append(k))
    with pytest.raises(ValueError, match="can only be run WITHOUT steering"):
        r.cell("wmdp_bio_full", "all", "N", **kwargs)
    assert ran == []                                                # the model was never called


def test_the_runner_accepts_an_unsteered_cell_on_the_full_set():
    r = _runner()
    import numpy as np
    r.ctx = SimpleNamespace(run=lambda *a, **k: SimpleNamespace(corr=np.ones(len(a[0])), parsed=np.ones(len(a[0])), texts=["A"] * len(a[0]),
                                                               m_instr=np.zeros(len(a[0])), m_conseq=np.zeros(len(a[0]))))
    saved = {}
    r.store = SimpleNamespace(has=lambda k: False, cells={}, put=lambda rec, *a: saved.update(rec=rec, n=len(a[0])), put_qids=lambda *a: None)
    r.refusal_flags = lambda raw, parsed: (np.zeros(len(raw)), np.zeros(len(raw)))
    r.cell("wmdp_bio_full", "all", "N")                              # no direction, no intervention: allowed
    assert saved["rec"]["n"] == 1273 and saved["rec"]["direction"] == "none"


def test_ordinary_evals_are_unaffected_by_the_guard():
    assert R.UNSTEERED_ONLY_EVALS == ("wmdp_bio_full",) == E.UNSTEERED_ONLY
    assert "wmdp_bio" not in R.UNSTEERED_ONLY_EVALS and "mmlu_nonbio" not in R.UNSTEERED_ONLY_EVALS
