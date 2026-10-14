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


def test_plastic_section_of_a_rectangle_is_sigma_y_b_h2_over_4(parts, tmp_path):
    """The 200 x 20 x 10 cantilever with 100 N down at the tip: at mid span M = 10 kN mm, Z = 20 x 10^2 / 4."""
    import numpy as np

    from wobbl.fe import NodeConstraint, StaticModel
    from wobbl.mesh.mesher import Sizing, build_mesh
    from wobbl.plastic.section import plastic_section
    from wobbl.structural.run import describe_part

    prep, _ = describe_part(parts["cantilever"], tmp_path)
    mesh = build_mesh(prep, Sizing(h=4.0))
    root = np.nonzero(np.abs(mesh.X[:, 0]) < 1e-6)[0]
    tip = np.nonzero(np.abs(mesh.X[:, 0] - 200) < 1e-6)[0]
    F = np.zeros((mesh.n_nodes, 3))
    F[tip, 2] = -100.0 / len(tip)
    sm = StaticModel(mesh, 200000.0, 0.3, 7.85e-9, [NodeConstraint("fix1", root)], forces=F)
    s, why = plastic_section(mesh, [100.0, 10.0, 5.0], sm, 250.0)
    assert s is not None, why
    assert s["Z_mm3"] == pytest.approx(500.0, rel=0.005)
    assert s["factor"] == pytest.approx(250 * 500 / 1e4, rel=0.005)
    assert s["depth_mm"] == pytest.approx(10.0, rel=0.01) and s["bending_share"] == pytest.approx(1.0)


def test_verdicts_separate_yielding_from_collapse():
    from wobbl.plastic.study import verdict_for

    assert verdict_for(1.4, 2.5, 1.5) == "holds"
    assert verdict_for(0.6, 2.0, 1.5) == "yields"      # yields locally but carries the load with margin
    assert verdict_for(0.6, 1.2, 1.5) == "marginal"
    assert verdict_for(0.6, 0.9, 1.5) == "collapses"


def test_brittle_and_plastic_materials_are_turned_away():
    from wobbl.plastic.study import check_metal
    from wobbl.structural.spec import SpecError

    for name in ("pla", "petg", "wood"):
        with pytest.raises(SpecError):
            check_metal(get_material(name))
    check_metal(get_material("6061-t6"))
    check_metal(parse_custom("custom:E=200000,nu=0.3,yield=250"))


def test_collapse_fixes_aim_at_the_hinge_and_scale_with_the_shortfall():
    from wobbl.plastic.study import fixes

    section = {"depth_mm": 10.0, "area_mm2": 200.0, "bending_share": 1.0}
    out = fixes("collapses", get_material("mild-steel"), 0.5, 0.8, 1.5, section, "on face 3 at (0, 0, 0)")
    kinds = [f["kind"] for f in out]
    assert kinds == ["section", "material", "load"]
    assert "13.7 mm" in out[0]["summary"]          # sqrt(1.5 / 0.8) times deeper
    assert "469 MPa" in out[1]["title"] and "4140" in out[1]["summary"]
    assert fixes("yields", get_material("mild-steel"), 0.5, 2.0, 1.5, section, "") == []


def test_custom_metal_without_a_tensile_strength_is_perfectly_plastic():
    h = hardening_curve(parse_custom("custom:E=200000,nu=0.3,yield=250"))
    assert h.points == ((pytest.approx(250 * (1 + 250 / 200000)), 0.0),) and h.kind == "bilinear"
