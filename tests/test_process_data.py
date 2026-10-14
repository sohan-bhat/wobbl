"""Process data: the warp calibration reproduces from its source table, alloy freezing curves behave, and the
lookups and custom resins read the way the studies expect."""
import numpy as np
import pytest

from wobbl.process import data as D


def test_ramp_height_refits_from_the_armillotta_table():
    ramp, fit = D.fit_ramp_height()
    assert ramp == pytest.approx(D.RAMP_HEIGHT, rel=0.01)
    assert fit["r2"] > 0.9 and fit["median_error"] < 0.12 and fit["max_error"] < 0.45


def test_a356_freezes_by_scheil_then_eutectic():
    a = D.get_alloy("a356")
    T = np.linspace(500, 700, 4001)
    fs = a.solid_fraction(T)
    assert (np.diff(fs) <= 1e-12).all()
    assert fs[T >= a.liquidus].max() == 0.0 and fs[T <= a.solidus].min() == 1.0
    # the Scheil curve with k = 0.13 leaves 48% liquid when it reaches the Al-Si eutectic
    assert a.solid_fraction([a.eutectic])[0] == pytest.approx(1 - (83.3 / 44.3) ** (1 / (0.13 - 1)), abs=2e-3)
    assert a.solid_fraction([a.eutectic])[0] == pytest.approx(0.516, abs=0.005)


def test_pure_aluminum_releases_its_heat_evenly():
    al = D.get_alloy("aluminum")
    assert al.solid_fraction([al.liquidus, 0.5 * (al.liquidus + al.solidus), al.solidus]).tolist() == [0.0, 0.5, 1.0]


def test_lookups_by_key_and_alias():
    assert D.get_alloy("A356.0").key == "a356"
    assert D.get_mold("green sand").key == "sand"
    assert D.get_resin("polystyrene").key == "gpps"
    assert D.get_filament("PETG").key == "petg"
    with pytest.raises(KeyError, match="nylon"):
        D.get_filament("nylon")


def test_cross_wlf_zero_shear_viscosity():
    ps = D.get_resin("gpps")
    # D1 at T = D2, then the WLF drop with temperature
    assert ps.eta0(D.RESINS["gpps"].d2 - 273.15) == pytest.approx(ps.d1)
    assert 300 < ps.eta0(230.0) < 2000
    pc = D.get_resin("pc")
    assert pc.eta0(310.0, 50e6) > 2 * pc.eta0(310.0)       # D3: pressure stiffens polycarbonate


def test_custom_resin_needs_its_coefficients():
    r = D.parse_resin("custom:n=0.3,tau=25000,d1=3e12,d2=373,a1=28,melt=240,mold=60,no_flow=110")
    assert (r.n, r.tau, r.melt, r.a2) == (0.3, 25000.0, 240.0, 51.6)
    with pytest.raises(KeyError, match="d1"):
        D.parse_resin("custom:n=0.3,tau=25000,d2=373,a1=28")
