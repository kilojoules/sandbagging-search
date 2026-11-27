"""sbx.run: random-direction controls attach to every steered direction by default (the steering grid), and each control matches its
steered cell in everything but the direction.  No model: the grid is a generator, tested directly."""
from __future__ import annotations

from types import SimpleNamespace

from sbx import run as R


class _Ctx:
    """weightless stand-in: the grid is a pure generator and reads only ctx.nl (the tiny real model takes minutes per test on a CPU)"""
    def __init__(self, nl):
        self.nl = nl


def _cells(**over):
    # lists, not comma strings: the argument parser (L()) has already split them by the time a Runner sees them
    kw = dict(layer_sets=["auto1", "autoN"], dir_names=["B", "REF", "A"], interventions=["proj0", "cap", "add"], add_multilayer=False,
              strengths=[-0.2, -0.4], steer_evals=["wmdp_bio"], rand_ctrl=1, rand_for=["*"])
    kw.update(over)
    r = R.Runner(_Ctx(8), SimpleNamespace(**kw), {}, None, None)
    r.dirs = {n: {} for n in ("B", "REF", "A", "rand0", "rand1", "rand2")}
    r.steer_framings = ["E", "N"]
    return list(r.grid(["disc"]))


def _sig(c):
    """everything about a cell except its direction"""
    return (c["ev"], c["split"], c["fname"], c["interv"], tuple(c["layers"]), c["strength"])


def _is_ctrl(c):
    return c["direction"].startswith("rand")


def test_every_direction_gets_a_control_by_default():
    cells = _cells()
    real = {c["direction"] for c in cells if not _is_ctrl(c)}
    assert real == {"B", "REF", "A"}
    for d in real:
        assert any(c["direction"] == "rand0" for c in cells), d


def test_each_control_matches_a_steered_cell_in_everything_but_the_direction():
    cells = _cells()
    real = sorted(_sig(c) for c in cells if not _is_ctrl(c))
    ctrl = sorted(_sig(c) for c in cells if _is_ctrl(c))
    assert real and real == ctrl                                # exactly one control per steered cell, matched


def test_one_control_doubles_the_grid_and_three_quadruple_it():
    base = _cells(rand_ctrl=0)
    assert base and not any(_is_ctrl(c) for c in base)
    assert len(_cells(rand_ctrl=1)) == 2 * len(base)
    three = _cells(rand_ctrl=3)
    assert len(three) == 4 * len(base)
    assert {c["direction"] for c in three if _is_ctrl(c)} == {"rand0", "rand1", "rand2"}


def test_the_flag_still_narrows_the_controls_to_named_directions():
    """The grid yields each steered cell immediately followed by its controls, so a control belongs to the real cell before it."""
    cells = _cells(rand_for="B")
    owner, controlled, prev = None, {}, None
    for c in cells:
        if not _is_ctrl(c):
            owner, prev = c["direction"], c
            controlled.setdefault(owner, 0)
        else:
            assert owner is not None and _sig(c) == _sig(prev)           # a control matches the real cell right before it
            controlled[owner] += 1
    n_b = sum(1 for c in _cells(rand_ctrl=0) if c["direction"] == "B")  # every B cell has exactly one control, no other cell has any
    assert controlled == {"B": n_b, "REF": 0, "A": 0} and n_b > 0, controlled


def test_steering_preset_ships_one_control_for_every_direction():
    p = R.PRESETS["steer"]
    assert p["rand_ctrl"] == 1 and p["rand_for"] == "*"


def test_replication_adds_three_controls_to_a_candidate():
    """replicate() plans >= 3 random directions beside each candidate on the replication split (where the grid gives only one)."""
    r = R.Runner(_Ctx(8), SimpleNamespace(rand_ctrl=0, max_candidates=5, control_evals=(), xevals=(), dry_run=False), {}, None, None)
    r.dirs = {"B": {}, "rand0": {}, "rand1": {}, "rand2": {}}
    layers = R.parse_layers("auto1", 8)
    r.candidates = lambda: [(0.01, 0.05, dict(key="k", framing="E", intervention="cap", layers=R.layer_tag(layers), strength=0.0, eval="wmdp_bio", direction="B"))]
    planned = []
    r.run_cells = lambda cells, tag: planned.extend(cells)
    r.cell = lambda *a, **k: None
    r.replicate()
    ctrl = [c for c in planned if _is_ctrl(c)]
    assert {c["direction"] for c in ctrl} == {"rand0", "rand1", "rand2"} and all(c["split"] == "rep" for c in ctrl)
    assert all(c["interv"] == "cap" and c["fname"] == "E" for c in ctrl)            # matched on everything but the direction
