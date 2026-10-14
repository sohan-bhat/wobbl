"""Layups, classical lamination theory and ply failure, checked against closed forms. Nothing is solved here."""
from __future__ import annotations

import numpy as np
import pytest

from wobbl.composites.laminate import (LayupError, clt_first_ply_failure, hashin_ratios, max_stress_ratio,
                                       parse_layup, ply_axes_strain, qbar, reduced_stiffness, tsai_wu_ratio)
from wobbl.materials import PLIES, get_ply


def test_layup_notation():
    assert parse_layup("[0/90]s", "carbon").angles() == "0/90/90/0"
    assert parse_layup("[0/±45/90]s", "carbon").angles() == "0/45/-45/90/90/-45/45/0"
    assert parse_layup("[0_2/90]2s", "carbon").angles() == "0/0/90/0/0/90/90/0/0/90/0/0"
    assert parse_layup("0, 45, -45, 90", "carbon").angles() == "0/45/-45/90"
    lam = parse_layup("[0:im7-8552/45:glass:0.2]s", "t300-5208")
    assert [p.material.key for p in lam.plies] == ["im7-8552", "e-glass-epoxy", "e-glass-epoxy", "im7-8552"]
    assert [p.thickness for p in lam.plies] == [0.131, 0.2, 0.2, 0.131]
    assert parse_layup("0:0.3/90", "carbon").plies[0].thickness == 0.3
    assert parse_layup("[0/90]s", "carbon", thickness=0.2).thickness == pytest.approx(0.8)
    for bad in ("", "[0/x]s", "[0/45:unobtainium]"):
        with pytest.raises((LayupError, KeyError)):
            parse_layup(bad, "carbon")


def test_every_ply_names_its_source():
    for p in PLIES.values():
        assert p.source and p.E1 > p.E2 > 0 and min(p.Xt, p.Xc, p.Yt, p.Yc, p.S12) > 0
        assert get_ply(p.key) is p


def test_off_axis_stiffness_turns_back():
    m = get_ply("t300-5208")
    Q = reduced_stiffness(m)
    assert np.allclose(qbar(Q, 0.0), Q)
    q90 = qbar(Q, 90.0)
    assert q90[0, 0] == pytest.approx(Q[1, 1]) and q90[1, 1] == pytest.approx(Q[0, 0])
    # a strain along the fibers of a 30 degree ply reads as pure eps1 in the ply's own axes
    c, s = np.cos(np.radians(30)), np.sin(np.radians(30))
    e = ply_axes_strain(np.array([[c * c, s * s, 2 * c * s]]), 30.0)[0]
    assert e == pytest.approx([1.0, 0.0, 0.0], abs=1e-12)


def test_quasi_isotropic_carbon_matches_tsai():
    """[0/±45/90]s T300/5208 is in plane isotropic, with E about 69.7 GPa (Tsai and Hahn)."""
    eff = parse_layup("[0/±45/90]s", "t300-5208").effective()
    assert eff["Ex"] == pytest.approx(eff["Ey"], rel=1e-9)
    assert eff["Ex"] == pytest.approx(69_700, rel=0.005)
    assert eff["Gxy"] == pytest.approx(eff["Ex"] / (2 * (1 + eff["nuxy"])), rel=1e-6)


def test_cross_ply_modulus_is_the_rule_of_mixtures():
    """For [0/90]s, Ex = (Q11 + Q22)/2 - Q12^2 / ((Q11 + Q22)/2), the textbook closed form."""
    m = get_ply("t300-5208")
    Q = reduced_stiffness(m)
    avg = 0.5 * (Q[0, 0] + Q[1, 1])
    lam = parse_layup("[0/90]s", m)
    assert lam.symmetric() and lam.balanced()
    assert lam.effective()["Ex"] == pytest.approx(avg - Q[0, 1] ** 2 / avg, rel=1e-9)


def test_criteria_meet_the_strengths():
    m = get_ply("t300-5208")
    for sig, want in (([m.Xt, 0, 0], 1.0), ([-m.Xc, 0, 0], 1.0), ([0, m.Yt, 0], 1.0), ([0, -m.Yc, 0], 1.0),
                      ([0, 0, m.S12], 1.0), ([0.5 * m.Xt, 0, 0], 2.0)):
        s = np.array([sig], float)
        assert max_stress_ratio(s, m)[0] == pytest.approx(want)
        h = hashin_ratios(s, m)
        assert min(v[0] for v in h.values()) == pytest.approx(want, rel=1e-9)
    # Tsai-Wu meets the strengths along each axis on its own
    for sig in ([m.Xt, 0, 0], [-m.Xc, 0, 0], [0, m.Yt, 0], [0, -m.Yc, 0], [0, 0, m.S12]):
        assert tsai_wu_ratio(np.array([sig], float), m)[0] == pytest.approx(1.0, rel=1e-9)


def test_cross_ply_first_failure_by_hand():
    """A [0/90]s coupon pulled along x: the 90 plies crack first, where sigma2 reaches Yt."""
    m = get_ply("t300-5208")
    lam = parse_layup("[0/90]s", m)
    N = np.array([100.0, 0.0, 0.0])
    e = lam.compliance()[:3, :3] @ N
    sig2 = (reduced_stiffness(m) @ ply_axes_strain(e[None], 90.0)[0])[1]
    f = clt_first_ply_failure(lam, N)
    assert f["hashin_ply"] in (2, 3) and f["hashin_mode"] == "matrix tension"
    shear = (reduced_stiffness(m) @ ply_axes_strain(e[None], 90.0)[0])[2]
    assert abs(shear) < 1e-9
    assert f["hashin"] == pytest.approx(m.Yt / sig2, rel=1e-9)
    assert f["tsai_wu"] < f["hashin"] * 1.05
