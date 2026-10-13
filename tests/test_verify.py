import numpy as np
import pytest

from wobbl.verify.beam import section_properties
from wobbl.verify.checks import Check, balance_check, confidence, convergence_check


def test_rectangle_section():
    # 20 x 6 rectangle in the yz plane, split into two triangles
    p = np.array([[0, 0, 0], [0, 20, 0], [0, 20, 6], [0, 0, 6]], float)
    tris = np.array([[p[0], p[1], p[2]], [p[0], p[2], p[3]]])
    A, G, I = section_properties(tris, np.array([1.0, 0, 0]))
    assert A == pytest.approx(120)
    assert np.allclose(G, [0, 10, 3])
    assert I[2, 2] == pytest.approx(20 * 6 ** 3 / 12)
    assert I[1, 1] == pytest.approx(6 * 20 ** 3 / 12)


def test_convergence_levels():
    assert convergence_check([100, 103, 103.5], 0.03, "s", "MPa").status == "pass"
    assert convergence_check([100, 101], 0.03, "s", "MPa").status == "warn"  # only two meshes
    assert convergence_check([100, 120, 140], 0.03, "s", "MPa").status == "fail"
    assert convergence_check([100], 0.03, "s", "MPa").status == "fail"


def test_balance():
    good = balance_check(np.array([0, 0, -100.0]), np.array([0, 0, 100.0]), np.zeros(3), np.zeros(3), 100)
    assert good.status == "pass"
    bad = balance_check(np.array([0, 0, -100.0]), np.array([0, 0, 90.0]), np.zeros(3), np.zeros(3), 100)
    assert bad.status == "fail"


def test_confidence_never_high_after_failure():
    lvl, why = confidence([Check("a", "pass", ""), Check("b", "fail", "broken")])
    assert lvl == "low" and any("broken" in w for w in why)
    lvl, _ = confidence([Check("a", "pass", ""), Check("b", "warn", "")])
    assert lvl == "medium"
    lvl, _ = confidence([Check("a", "pass", "")], [("medium", "a sharp corner")])
    assert lvl == "medium"
    assert confidence([Check("a", "pass", "")])[0] == "high"
