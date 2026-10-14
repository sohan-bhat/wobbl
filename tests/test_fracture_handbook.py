"""Handbook stress intensity factors and the crack turning rule, against values worked by hand."""
import math

import numpy as np
import pytest

from wobbl.fracture import handbook as H


def test_center_and_edge_cracks_reduce_to_the_infinite_plate():
    # a tiny crack in a wide strip is the Griffith crack, and an edge crack carries the free surface's 1.1215
    assert H.center_crack(100, 1, 1000) == pytest.approx(H.k_unit(100, 1), rel=1e-5)
    assert H.edge_crack(100, 0.01, 1000) == pytest.approx(1.1215 * H.k_unit(100, 0.01), rel=2e-3)
    assert H.k_unit(100, 10) == pytest.approx(100 * math.sqrt(math.pi * 0.01))


def test_finite_width_values():
    # Tada's secant formula at 2a / W = 0.3, and the edge crack at a / W = 0.3 against Brown and Srawley's
    # polynomial 1.12 - 0.231 s + 10.55 s^2 - 21.72 s^3 + 30.39 s^4, which Tada's form matches within 0.5%
    assert H.center_crack(1, 15, 100) / H.k_unit(1, 15) == pytest.approx(1.0575, abs=2e-4)
    s = 0.3
    brown = 1.12 - 0.231 * s + 10.55 * s ** 2 - 21.72 * s ** 3 + 30.39 * s ** 4
    assert H.edge_crack(1, 12, 40) / H.k_unit(1, 12) == pytest.approx(brown, rel=5e-3)


def test_newman_raju_deepest_and_surface_points():
    # a shallow half penny crack (a = c) in a wide plate: 1.04 at the deepest point and 1.145 at the surface,
    # over sqrt(Q) = sqrt(2.464)
    deep = H.newman_raju(1, 1, 1, 100, 1000) / H.k_unit(1, 1)
    surf = H.newman_raju(1, 1, 1, 100, 1000, phi=0.0) / H.k_unit(1, 1)
    assert deep == pytest.approx(1.04 / math.sqrt(2.464), rel=1e-3)
    assert surf == pytest.approx(1.04 * 1.1 / math.sqrt(2.464), rel=2e-3)
    with pytest.raises(ValueError):
        H.newman_raju(1, 2, 1, 10, 100)


def test_weight_function_for_uniform_and_falling_stress():
    a = 2.0
    x = np.linspace(0, a, 40)
    assert H.edge_weight_function(x, np.full_like(x, 50.0), a) == pytest.approx(1.1215 * H.k_unit(50, a), rel=2e-3)
    # stress that falls off away from the surface gives less than its surface value would
    assert H.edge_weight_function(x, 50.0 * (1 - 0.2 * x / a), a) < 1.1215 * H.k_unit(50, a)


def test_mts_angles():
    assert H.mts_angle(1.0, 0.0) == 0.0
    assert math.degrees(H.mts_angle(1.0, 1.0)) == pytest.approx(-53.13, abs=0.01)
    assert math.degrees(H.mts_angle(0.0, 1.0)) == pytest.approx(-70.53, abs=0.01)
    assert math.degrees(H.mts_angle(1.0, -1.0)) == pytest.approx(53.13, abs=0.01)
    k1, k2 = H.inclined_crack(100, 10, math.pi / 4)
    assert k1 == pytest.approx(k2)
    assert H.k_equivalent(3.0, 4.0, 0.0, 0.3) == pytest.approx(5.0)
    assert H.k_equivalent(-3.0, 4.0, 0.0, 0.3) == pytest.approx(4.0)
