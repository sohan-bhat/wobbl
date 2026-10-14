"""Imperfection knockdowns, the column curve, the yield interaction and the buckling mode checks."""
import math
from types import SimpleNamespace

import numpy as np
import pytest

from wobbl.life import knockdown as K
from wobbl.materials import get_material, resolve_material


def test_column_curve_matches_eurocode_curve_b():
    """EN 1993-1-1 tabulates chi = 0.597 for curve b at a relative slenderness of 1.0."""
    steel = resolve_material("custom:E=200000,nu=0.3,yield=250")
    col = K.Column(np.array([1.0, 0, 0]), 1000.0, 100.0, 1000.0, 1000.0, 2.0)
    squash = 250 * 100 / 1000.0
    design, kd, why, _ = K.column_curve(squash, col, steel)     # elastic factor equal to squash: lambda_bar = 1
    assert design / squash == pytest.approx(0.597, abs=0.001)
    assert kd == pytest.approx(0.597, abs=0.001) and "curve b" in why
    # a very stocky column squashes, a very slender one follows Euler with a small knockdown
    assert K.column_curve(100 * squash, col, steel)[0] == pytest.approx(squash, rel=0.05)
    slender, _, _, _ = K.column_curve(0.05 * squash, col, steel)
    assert 0.85 * 0.05 * squash < slender < 0.05 * squash
    alu = K.column_curve(squash, col, get_material("6061-t6"))[2]
    assert "EN 1999-1-1" in alu


def test_sp8007_cylinder_knockdowns():
    tube = SimpleNamespace(faces={
        1: SimpleNamespace(kind="cylinder", radius=50.5, axis=[0, 0, 1.0], concave=False, area=3000.0),
        2: SimpleNamespace(kind="cylinder", radius=49.5, axis=[0, 0, 1.0], concave=True, area=3000.0),
        3: SimpleNamespace(kind="plane", radius=None, axis=None, concave=False, area=100.0)})
    R, t, _ = K._cylinder(tube)
    assert (R, t) == pytest.approx((50.0, 1.0))
    phi = math.sqrt(50.0) / 16
    axial = SimpleNamespace(model=SimpleNamespace(applied_total=lambda: (np.array([0, 0, -1000.0]), np.zeros(3))))
    g, why = K.knockdown("cylinder", tube, axial, [])
    assert g == pytest.approx(1 - 0.901 * (1 - math.exp(-phi))) and "SP-8007" in why
    side = SimpleNamespace(model=SimpleNamespace(applied_total=lambda: (np.array([1000.0, 0, 0]), np.zeros(3))))
    assert K.knockdown("cylinder", tube, side, [])[0] == pytest.approx(1 - 0.731 * (1 - math.exp(-phi)))
    pressure = [(SimpleNamespace(kind="pressure"), None)]
    assert K.knockdown("cylinder", tube, axial, pressure)[0] == 0.75
    assert K.classify("auto", None, SimpleNamespace(**vars(tube), bbox=[[0, 0, 0], [1, 1, 1]]), axial, [])[0] == \
        "cylinder"
    assert K.knockdown("plate", tube, axial, [])[0] == 1.0
    assert K.knockdown("general", tube, axial, [])[0] == 0.5


def test_johnson_ostenfeld():
    assert K.johnson(10.0, 50.0, True) == (10.0, "")                 # far from yield: elastic buckling
    lam, why = K.johnson(10.0, 12.0, True)
    assert lam == pytest.approx(12.0 * (1 - 12.0 / 40.0)) and "Johnson" in why
    assert K.johnson(10.0, 4.0, False)[0] == 4.0                      # a brittle part breaks first
    assert K.johnson(10.0, math.inf, True)[0] == 10.0


def test_euler_hand_calculation():
    steel = resolve_material("custom:E=200000,nu=0.3,yield=250")
    col = K.Column(np.array([1.0, 0, 0]), 400.0, 100.0, 10 ** 4 / 12, 1000.0, 2.0, "clamped at the base")
    pe = math.pi ** 2 * 200000 * (10 ** 4 / 12) / 800.0 ** 2
    assert K.euler_check(pe / 1000.0 * 1.01, col, steel).status == "pass"
    assert K.euler_check(pe / 1000.0 * 1.10, col, steel).status == "info"
    assert K.euler_check(pe / 1000.0 * 1.5, col, steel).status == "warn"
    assert K.euler_check(1.0, None, steel).status == "skip"
    assert col.slenderness == pytest.approx(400 / math.sqrt(10 ** 4 / 12 / 100))


def test_mode_separation():
    assert K.separation_check([2.57, 2.57, 23.1]).status == "pass"         # a repeated pair is one buckle
    assert "symmetric" in K.separation_check([2.57, 2.575, 23.1]).summary
    assert K.separation_check([2.0, 2.1, 5.0]).status == "warn"
    assert K.separation_check([2.0]).status == "skip"
    assert K.separation_check([-3.0, 2.0, 2.5]).status == "pass"
