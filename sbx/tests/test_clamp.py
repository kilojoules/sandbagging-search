"""The two-sided clamp: sets a coordinate onto its neutral value from either side.  `cap` can only lower a coordinate, so it cannot undo a
sandbagging state that pushed one too low; `proj0` sends every coordinate to zero, honest or not.  No model needed."""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from sbx import run as R
from sbx.core import f_proj, spec_proj

D = torch.tensor([0.0, 0.0, 1.0, 0.0, 0.0, 0.0])           # unit direction: coordinate 2
TARGET = 3.0                                               # its neutral (honest) value


def _h(x):
    h = torch.zeros(1, 6)
    h[0, 0], h[0, 2] = 7.0, x                              # another coordinate, to check it is left alone
    return h


@pytest.mark.parametrize("x", [0.5, 1.0, 2.9, 3.0, 3.1, 5.0, 40.0])
def test_clamp_puts_the_coordinate_on_the_neutral_value_from_either_side(x):
    out = f_proj(D, TARGET, one_sided=False)(_h(x))
    assert out[0, 2].item() == pytest.approx(TARGET, abs=1e-5)
    assert out[0, 0].item() == pytest.approx(7.0)           # nothing else moves


@pytest.mark.parametrize("x,expected", [(5.0, TARGET), (40.0, TARGET), (3.0, TARGET), (1.0, 1.0), (0.0, 0.0)])
def test_cap_can_only_lower(x, expected):
    assert f_proj(D, TARGET, one_sided=True)(_h(x))[0, 2].item() == pytest.approx(expected, abs=1e-5)


@pytest.mark.parametrize("x", [1.0, 3.0, 5.0])
def test_proj0_sends_every_coordinate_to_zero(x):
    assert f_proj(D, 0.0, False)(_h(x))[0, 2].item() == pytest.approx(0.0, abs=1e-5)


def test_clamp_changes_what_cap_cannot_when_the_coordinate_is_too_low():
    low = _h(1.0)
    assert torch.equal(f_proj(D, TARGET, one_sided=True)(low), low)              # cap: no change at all
    assert f_proj(D, TARGET, one_sided=False)(low)[0, 2].item() == pytest.approx(TARGET)


def test_spec_proj_two_sided_with_the_neutral_median_is_the_clamp():
    ctx = SimpleNamespace(model=SimpleNamespace(device="cpu"))
    vecs = {5: np.asarray(D)}                            # spec_proj takes the per-layer vectors of ONE direction
    med = {5: TARGET}                                    # and that direction's neutral medians per layer
    (L, fn), = spec_proj(ctx, vecs, [5], med, one_sided=False)
    assert L == 5 and fn(_h(1.0))[0, 2].item() == pytest.approx(TARGET) and fn(_h(9.0))[0, 2].item() == pytest.approx(TARGET)
    (_, cap), = spec_proj(ctx, vecs, [5], med, one_sided=True)
    assert cap(_h(1.0))[0, 2].item() == pytest.approx(1.0)                        # the one-sided version leaves the low coordinate alone


def test_the_runner_dispatches_clamp_to_the_two_sided_projection():
    r = R.Runner(SimpleNamespace(model=SimpleNamespace(device="cpu")), SimpleNamespace(model="m", answer_mode="native"), {}, None, None)
    r.U = {5: 1.0}
    r.dirs = {"B": {5: np.asarray(D)}}
    r.med = {"B": {5: TARGET}}
    (L, clamp), = r.specs_for("B", "clamp", [5], 0.0)
    assert L == 5 and clamp(_h(1.0))[0, 2].item() == pytest.approx(TARGET) and clamp(_h(9.0))[0, 2].item() == pytest.approx(TARGET)
    (_, cap), = r.specs_for("B", "cap", [5], 0.0)
    assert cap(_h(1.0))[0, 2].item() == pytest.approx(1.0)                        # cap is unchanged
    (_, p0), = r.specs_for("B", "proj0", [5], 0.0)
    assert p0(_h(9.0))[0, 2].item() == pytest.approx(0.0)
    with pytest.raises(ValueError):
        r.specs_for("B", "no-such-intervention", [5], 0.0)


def test_clamp_is_opt_in_not_in_the_default_steering_grid():
    """Adding it to the preset would grow every future steering grid by a third: that is a decision, not a default."""
    assert "clamp" not in R.PRESETS["steer"]["interventions"].split(",")
