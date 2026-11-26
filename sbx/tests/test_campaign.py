"""sbx.campaign: mode choice from model cards and the instructability gate (no model, no GPU)."""
from __future__ import annotations

import os

import numpy as np

from sbx import campaign as C
from sbx.run import Store, ckey


def test_default_modes_follow_the_cards():
    assert C.default_modes("gptoss-120b") == ["native"] or C.default_modes("gptoss-120b") == ["native"]   # reasoning or no card
    assert C.default_modes("qwen3-8b") == ["native", "letters"]              # cot: both modes are screened (letters is cheap)
    assert C.default_modes("qwen3-32b") == ["native", "letters"]
    assert C.default_modes("qwen3-32b-think") == ["native"]                  # hidden reasoning: only its own mode
    assert C.default_modes("no-such-model") == ["native", "letters"]         # no card: treated as a cot model


def _cell(st, fr, corr, split="disc", ev="wmdp_bio"):
    corr = np.asarray(corr, float)
    key = ckey(ev, split, fr, "none", "none", "-", 0.0, 0)
    rec = dict(key=key, eval=ev, split=split, framing=fr, direction="none", intervention="none", layers="-", strength=0.0,
               seed=0, n=len(corr), acc=float(corr.mean()), parse_rate=1.0, refusal_rate=0.0, acc_answered=float(corr.mean()))
    st.put(rec, corr, np.ones(len(corr)), np.zeros(len(corr)))


def test_instructability_gate(tmp_path):
    rng = np.random.RandomState(0)
    n = 120
    base = (rng.rand(n) < 0.8).astype(float)
    st = Store(str(tmp_path))
    _cell(st, "N", base)
    _cell(st, "E", base * (rng.rand(n) < 0.3))            # follows the instruction: big drop
    _cell(st, "pos_bio_plain", base * (rng.rand(n) < 0.97))   # barely changes
    ins = C.instructability(str(tmp_path))
    assert ins["_N"]["acc"] > 0.7
    assert ins["E"]["valid"] and ins["E"]["drop"] > 0.4
    assert not ins["pos_bio_plain"]["valid"]


def test_instructability_without_baseline(tmp_path):
    Store(str(tmp_path))
    assert C.instructability(str(tmp_path))["_N"] is None


def test_dry_run_returns_without_work(tmp_path, capsys):
    assert C.main(["--model", "qwen3-8b", "--out", str(tmp_path / "o"), "--dry-run"]) == 0
    assert "modes ['native', 'letters']" in capsys.readouterr().out
    assert not os.path.exists(str(tmp_path / "o" / "campaign_status.json"))


def test_covert_natural_requires_wrong_answers_not_refusals_and_a_weak_twin():
    base = dict(_N=dict(acc=0.8, parse_rate=1.0, acc_answered=0.8, refusal_rate=0.0))
    covert = dict(base, nat_goal_doc=dict(acc=0.6, drop=0.2, p=1e-4, acc_answered=0.6, refusal_rate=0.0, valid=True),
                  neg_goal_doc_inverted=dict(acc=0.78, drop=0.02, p=0.5, acc_answered=0.78, refusal_rate=0.0, valid=False))
    assert C.covert_natural(covert) == "nat_goal_doc"
    refusing = dict(base, nat_goal_doc=dict(acc=0.3, drop=0.5, p=1e-9, acc_answered=0.79, refusal_rate=0.65, valid=True))
    assert C.covert_natural(refusing) is None                       # the drop is refusals: overt, not covert
    generic = dict(covert, neg_goal_doc_inverted=dict(acc=0.62, drop=0.18, p=1e-3, acc_answered=0.62, refusal_rate=0.0, valid=True))
    assert C.covert_natural(generic) is None                        # the sign twin costs as much: wording, not incentive
    assert C.covert_natural(dict(base)) is None


def test_gate_waits_for_every_positive_control(tmp_path):
    """phi-4 and Gemma-3-4B follow only the hazardous-questions instruction.  Deciding before it is measured must be 'inconclusive',
    never 'not followed'."""
    rng = np.random.RandomState(1)
    n = 120
    base = (rng.rand(n) < 0.77).astype(float)
    st = Store(str(tmp_path))
    _cell(st, "N", base)
    _cell(st, "E", base * (rng.rand(n) < 0.95))                  # ignored
    _cell(st, "pos_bio_plain", base * (rng.rand(n) < 0.89))      # too small to pass the gate
    partial = C.instructability(str(tmp_path))
    gs = C.gate_state(partial)
    assert gs["missing"] == ["pos_bio_hazard"] and gs["inconclusive"] and not gs["followed"]
    assert C.best_positive(partial) is None          # which is exactly what the old gate read as "nothing is followed"
    _cell(st, "pos_bio_hazard", base * (rng.rand(n) < 0.52))     # followed: ~36-point drop
    full = C.instructability(str(tmp_path))
    gs = C.gate_state(full)
    assert gs["complete"] and gs["followed"] and not gs["inconclusive"]
    assert C.best_positive(full) == "pos_bio_hazard"
    assert 0.30 < full["pos_bio_hazard"]["drop"] < 0.45 and full["pos_bio_hazard"]["valid"]


def test_old_gate_condition_is_not_complete(tmp_path):
    """The condition the driver used before the fix (N, E and pos_bio_plain present) must NOT count as a complete gate."""
    rng = np.random.RandomState(3)
    n = 120
    base = (rng.rand(n) < 0.77).astype(float)
    st = Store(str(tmp_path))
    _cell(st, "N", base)
    _cell(st, "E", base * (rng.rand(n) < 0.95))
    _cell(st, "pos_bio_plain", base * (rng.rand(n) < 0.89))
    ins = C.instructability(str(tmp_path))
    old_gate_cells = bool(ins.get("_N")) and "E" in ins and "pos_bio_plain" in ins      # the pre-fix condition
    assert old_gate_cells                                                   # ... which the old code took as "decided"
    assert not C.gate_state(ins)["complete"]                                # ... and the new code does not
    assert C.gate_state(ins)["inconclusive"] and C.gate_state(ins)["followed"] is False


def test_gate_says_not_followed_only_when_every_control_was_measured(tmp_path):
    rng = np.random.RandomState(2)
    n = 120
    base = (rng.rand(n) < 0.75).astype(float)
    st = Store(str(tmp_path))
    _cell(st, "N", base)
    for f in C.BIO_POSITIVE:                              # measured and ignored
        _cell(st, f, base * (rng.rand(n) < 0.97))
    gs = C.gate_state(C.instructability(str(tmp_path)))
    assert gs["complete"] and not gs["followed"] and not gs["inconclusive"]


def test_gate_needs_the_neutral_baseline():
    assert C.gate_state({})["inconclusive"] and C.gate_state({})["missing"] == list(C.BIO_POSITIVE)
    assert C.gate_state({"_N": None, "E": {"valid": False}})["inconclusive"]


def _resume_case(monkeypatch, tmp_path, cells_on_call=None, minutes=None):
    """Drive one screen mode with a fake step runner.  `minutes` is what the fake reports as its runtime (the real runner reports its own);
    `cells_on_call` is the screen call (1 = the screen step itself) on which it stores the screen cells."""
    seen = {"screen": 0}

    def fake_run_step(model, out, step, mode, extra, cap_min, passthrough):
        if step != "screen":
            return 0                                   # later steps are not part of this test
        seen["screen"] += 1
        C.run_step.last_minutes = minutes if minutes is not None else cap_min      # default: used its whole allowance
        if seen["screen"] == cells_on_call:
            rng = np.random.RandomState(5)
            n = 120
            base = (rng.rand(n) < 0.77).astype(float)
            st = Store(out)
            _cell(st, "N", base)
            _cell(st, "E", base * (rng.rand(n) < 0.97))
            _cell(st, "pos_bio_plain", base * (rng.rand(n) < 0.97))
            _cell(st, "pos_bio_hazard", base * (rng.rand(n) < 0.52))
            _cell(st, "N", base, ev="mmlu_nonbio")                               # the control-eval cells the loop also waits for
            _cell(st, "E", base * (rng.rand(n) < 0.97), ev="mmlu_nonbio")
        return 0

    monkeypatch.setattr(C, "run_step", fake_run_step)
    monkeypatch.setattr(C, "remaining_min", lambda *a, **k: 200.0, raising=False)
    assert C.main(["--model", "qwen3-8b", "--modes", "native", "--out", str(tmp_path / "o")]) == 0
    return seen["screen"], C.instructability(str(tmp_path / "o" / "native"))


def test_slow_model_is_not_written_off_when_a_resumed_step_adds_nothing(tmp_path, monkeypatch):
    """A slow model may need longer than the screen cap for its first cell.  Resumed steps that use the whole allowance and add nothing
    must not end the resuming before three of them in a row."""
    # the screen step itself and the first resumed step are silent; the second resumed step delivers (the third screen call of all).
    # Under the old rule ("stop at the first quiet retry") this model was written off as not measured.
    calls, ins = _resume_case(monkeypatch, tmp_path, cells_on_call=3)
    assert calls == 3, calls
    assert C.gate_state(ins)["complete"] and C.gate_state(ins)["followed"]     # the slow model is measured, not written off


def test_resume_gives_up_after_three_slow_steps_with_no_cell(tmp_path, monkeypatch, capsys):
    """A model that never gets a cell out within its allowance is given three resumed steps, then the driver stops."""
    calls, ins = _resume_case(monkeypatch, tmp_path, cells_on_call=None)
    assert calls == 4, calls                                             # the screen step + two silent resumed steps: three slow steps in all, then it stops
    assert C.gate_state(ins)["inconclusive"]                             # and it is reported as not measured, never as "nothing followed"
    assert "3 resumed screen steps in a row added no cell" in capsys.readouterr().out


def test_resume_gives_up_after_two_early_exits_with_no_cell(tmp_path, monkeypatch, capsys):
    """A step that finishes early with nothing to add had time and could not deliver: stop at the second in a row."""
    calls, ins = _resume_case(monkeypatch, tmp_path, cells_on_call=None, minutes=2.0)
    assert calls == 3, calls                                             # the screen step + one silent resumed step: two early exits in all, then it stops
    assert C.gate_state(ins)["inconclusive"]
    assert "2 resumed screen steps in a row added no cell" in capsys.readouterr().out
