"""Grain burnback: closed forms for round cores, tubes and end burners, and the grid method for other cores."""
import math

import pytest

from wobbl.chem.grains import Bates, make_grain


def _integral(g, n=4000):
    xs = [g.web * i / n for i in range(n + 1)]
    return sum(0.5 * (g.burn_area(a) + g.burn_area(b)) * (b - a) for a, b in zip(xs, xs[1:]))


def test_bates_segment_burns_back_by_its_closed_form():
    g = make_grain("bates", 0.05, 0.10, core=0.02)
    assert isinstance(g, Bates) and g.web == pytest.approx(0.015)
    x = 0.005
    d = 0.02 + 2 * x
    assert g.burn_area(x) == pytest.approx(math.pi * d * (0.10 - 2 * x) + 2 * math.pi / 4 * (0.05 ** 2 - d ** 2))
    assert g.port(x) == pytest.approx(math.pi / 4 * d ** 2)
    assert g.burn_area(0.0151) == 0.0 and g.volume(0.0151) == 0.0
    assert _integral(g) == pytest.approx(g.volume(0.0), rel=5e-4)     # the area swept out is the propellant
    one = make_grain("bates", 0.05, 0.10, ends=1, core=0.02)
    assert one.length(x) == pytest.approx(0.10 - x)


def test_tube_with_coated_ends_burns_neutral_and_end_burner_is_flat():
    t = make_grain("tube", 0.05, 0.10, ends=0, core=0.02)
    assert t.web == pytest.approx(0.0075)
    assert t.burn_area(0.0) == pytest.approx(t.burn_area(0.007)) == pytest.approx(math.pi * 0.07 * 0.10)
    e = make_grain("end", 0.05, 0.10)
    assert e.burn_area(0.0) == pytest.approx(e.burn_area(0.09)) == pytest.approx(math.pi / 4 * 0.05 ** 2)
    assert e.web == pytest.approx(0.10) and e.port(0.0) == 0.0


def test_grid_core_matches_the_round_core_closed_form():
    exact = make_grain("bates", 0.05, 0.10, core=0.02)
    grid = make_grain("round", 0.05, 0.10, core=0.02)
    assert grid.web == pytest.approx(exact.web, rel=0.005)
    for x in (0.0, 0.004, 0.009, 0.013):
        assert grid.perimeter(x) == pytest.approx(exact.perimeter(x), rel=0.02)
        assert grid.port(x) == pytest.approx(exact.port(x), rel=0.01)
        assert grid.burn_area(x) == pytest.approx(exact.burn_area(x), rel=0.015)
    assert grid.coarea_error() < 0.01


@pytest.mark.parametrize("shape,params", [
    ("star", dict(points=5, point_length=0.018, point_width=0.012)),
    ("finocyl", dict(core=0.015, fins=6, fin_length=0.008, fin_width=0.003)),
    ("moon", dict(core=0.015, offset=0.008)),
    ("xcore", dict(slot_width=0.005, slot_length=0.03)),
])
def test_other_cores_burn_out_and_conserve_propellant(shape, params):
    g = make_grain(shape, 0.05, 0.10, **params)
    assert g.coarea_error() < 0.015
    assert _integral(g) == pytest.approx(g.volume(0.0), rel=0.015)
    assert g.burn_area(g.web * 1.0001) == 0.0 and g.burn_area(0.0) > 0


def test_bad_grains_are_refused():
    with pytest.raises(ValueError, match="unknown grain shape"):
        make_grain("pretzel", 0.05, 0.1)
    with pytest.raises(ValueError, match="core"):
        make_grain("bates", 0.05, 0.1, core=0.06)
