"""Cycles to failure along a crack path, for one part and for many, with no solver."""
import math

import numpy as np
import pytest

from wobbl.fracture.growth import GrowthLaw, Spectrum, law_for, parse_spectrum
from wobbl.fracture.life import (CrackPath, LifeInputs, Scatter, growth_curve, inspection, lives, monte_carlo,
                                 wheeler_reference)


def _path(beta=1.12, stress=100.0, end=20.0):
    s = np.geomspace(1.0, end, 8)
    y = beta * stress * math.sqrt(math.pi / 1000)
    return CrackPath(s, y * np.sqrt(s), np.zeros_like(s)), y


def test_paris_life_against_the_closed_form():
    path, y = _path()
    c, m = 2.71e-8, 3.7
    inp = LifeInputs(GrowthLaw("paris", c, m), parse_spectrum(r_ratio=0.0), 0.0)
    exact = (1.0 ** (1 - m / 2) - 20.0 ** (1 - m / 2)) / ((m / 2 - 1) * c * y ** m)
    n, ran = lives(path, inp, 1.0, 1.0, 1.0, 0.0)
    assert n[0] == pytest.approx(exact, rel=1e-5) and ran[0]
    coarse, _ = lives(path, inp, 1.0, 1.0, 1.0, 0.0, n_grid=100)
    assert coarse[0] == pytest.approx(exact, rel=1e-4)
    # twice the load is 2^m times shorter, and a crack that is already critical has no life
    assert lives(path, inp, 1.0, 2.0, 1.0, 0.0)[0][0] == pytest.approx(exact / 2 ** m, rel=1e-4)
    assert lives(path, inp, 5.0, 1.0, 1.0, 1.0)[0][0] == 0.0
    size, cycles = growth_curve(path, inp, 1.0)
    assert size[0] == pytest.approx(1.0) and cycles[-1] == pytest.approx(exact, rel=1e-4)


def test_failure_at_toughness_and_net_section():
    path, y = _path(end=40.0)
    law = GrowthLaw("paris", 2.71e-8, 3.7, kc=29.0)
    inp = LifeInputs(law, parse_spectrum(r_ratio=0.0), 0.0)
    ac = (29.0 / y) ** 2
    exact = (1.0 ** (1 - 3.7 / 2) - ac ** (1 - 3.7 / 2)) / ((3.7 / 2 - 1) * law.C * y ** 3.7)
    n, ran = lives(path, inp, 1.0, 1.0, 1.0, 29.0)
    assert n[0] == pytest.approx(exact, rel=2e-3) and not ran[0]
    # a ligament of 100 mm2 carrying 2000 N yields at a flow stress of 40 MPa once 50 mm2 has cracked
    path.area, path.section_area, path.section_force = path.s * 5.0, 100.0, 2000.0
    inp.flow_stress = 40.0
    inp.law = GrowthLaw("paris", 2.71e-8, 3.7)
    n2, _ = lives(path, inp, 1.0, 1.0, 1.0, 0.0)
    exact2 = (1.0 ** (1 - 3.7 / 2) - 10.0 ** (1 - 3.7 / 2)) / ((3.7 / 2 - 1) * law.C * y ** 3.7)
    assert n2[0] == pytest.approx(exact2, rel=2e-3)


def test_monte_carlo_percentiles_and_inspection():
    path, _ = _path(end=40.0)
    inp = LifeInputs(law_for("7075-t6"), parse_spectrum(r_ratio=0.1), 400.0)
    res = monte_carlo(path, inp, 1.0, Scatter(samples=4000), target=1e4)
    p = res.percentiles
    assert p["B1"] < p["B10"] < p["B50"]
    assert p["B50"] == pytest.approx(res.nominal, rel=0.1)
    assert res.b1_band[0] <= p["B1"] <= res.b1_band[1]
    assert 0 <= res.pof_band[0] <= res.pof <= res.pof_band[1] <= 1
    again = monte_carlo(path, inp, 1.0, Scatter(samples=4000), target=1e4)
    assert again.percentiles == p        # the same seed gives the same answer
    insp = inspection(path, inp, 1.0, 1.27)
    assert insp["repeat"] < insp["first"] and insp["repeat"] == pytest.approx(0.5 * insp["life_from_detectable"])


def test_wheeler_block_factors_against_cycle_by_cycle():
    path, _ = _path(stress=150.0)
    blk = np.full(50, 0.6)
    blk[0] = 1.0
    law = law_for("7075-t6")
    plain = LifeInputs(law, Spectrum(blk, np.zeros(50)), 400.0)
    slow = LifeInputs(law, Spectrum(blk, np.zeros(50)), 400.0, wheeler=1.5)
    n_plain = lives(path, plain, 1.0, 1.0, 1.0, law.kc)[0][0]
    n_slow = lives(path, slow, 1.0, 1.0, 1.0, law.kc)[0][0]
    ref = wheeler_reference(path, slow, 1.0, 503.0)
    assert n_slow > 1.5 * n_plain
    assert n_slow == pytest.approx(ref, rel=0.03)


def test_threshold_stops_small_cracks_only():
    # K rises with size; at 30 MPa the smallest cracks sit below the 1.8 MPa sqrt(m) threshold and never grow
    path, y = _path(stress=30.0, end=40.0)
    law = law_for("7075-t6")
    inp = LifeInputs(law, parse_spectrum(r_ratio=0.1), 400.0)
    small = law.threshold(0.1) / (0.9 * y)
    a_stop = 0.9 * small ** 2
    n, _ = lives(path, inp, np.array([0.5 * a_stop, 2.0 * a_stop]), 1.0, 1.0, law.kc)
    assert np.isinf(n[0]) and np.isfinite(n[1]) and n[1] > 0
    size, cycles = growth_curve(path, inp, 2.0 * a_stop)
    assert np.all(np.diff(cycles) >= 0) and np.isfinite(cycles).all()


def test_a_path_that_ends_by_cutting_through_counts_as_failure():
    path, _ = _path(end=5.0)
    inp = LifeInputs(GrowthLaw("paris", 2.71e-8, 3.7), parse_spectrum(r_ratio=0.0), 0.0)
    _, ran = lives(path, inp, 1.0, 1.0, 1.0, 0.0)
    assert ran[0]
    path.end_fails = True
    n, ran = lives(path, inp, 1.0, 1.0, 1.0, 0.0)
    assert not ran[0] and np.isfinite(n[0])
