"""Elastic plastic studies: the true stress strain curves, without a solver."""
import math

import pytest

from wobbl.materials import LIBRARY, get_material, hardening_curve, parse_custom


def test_mild_steel_follows_the_yun_gardner_quad_linear_model():
    """A36 at fy 250 and fu 400: a yield plateau to 1.5% strain, then hardening to fu at 22.5% (engineering)."""
    h = hardening_curve(get_material("mild-steel"))
    assert h.kind == "multilinear" and "Yun and Gardner" in h.source
    (s0, p0), (s1, p1), _, (s3, p3) = h.points
    assert p0 == 0 and s0 == pytest.approx(250 * (1 + 250 / 200000))
    assert s1 == pytest.approx(250 * 1.015) and p1 == pytest.approx(math.log(1.015) - 250 * 1.015 / 200000, abs=1e-6)
    assert s3 == pytest.approx(400 * 1.225) and p3 == pytest.approx(math.log(1.225) - 490 / 200000, abs=1e-6)


def test_aluminum_knee_starts_below_the_proof_stress_and_ends_at_the_true_tensile_strength():
    m = get_material("6061-t6")
    h = hardening_curve(m)
    assert "Steinhardt" in h.source
    assert 0.9 * m.yield_strength < h.points[0][0] < m.yield_strength
    # the 0.2% proof stress sits on the knee
    near = min(h.points, key=lambda p: abs(p[1] - 0.002))
    assert near[0] == pytest.approx(m.yield_strength, rel=0.02)
    e_u = 0.5 * m.elongation / 100
    assert h.points[-1][0] == pytest.approx(m.tensile_strength * (1 + e_u))


@pytest.mark.parametrize("key", [k for k, m in LIBRARY.items() if m.category == "metal"])
def test_every_metal_curve_rises_from_zero_plastic_strain(key):
    h = hardening_curve(get_material(key))
    assert h.points[0][1] == 0 and h.source
    for (s0, p0), (s1, p1) in zip(h.points, h.points[1:]):
        assert p1 > p0 and s1 >= s0


def test_custom_metal_without_a_tensile_strength_is_perfectly_plastic():
    h = hardening_curve(parse_custom("custom:E=200000,nu=0.3,yield=250"))
    assert h.points == ((pytest.approx(250 * (1 + 250 / 200000)), 0.0),) and h.kind == "bilinear"
