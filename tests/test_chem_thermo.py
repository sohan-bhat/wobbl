"""Equilibrium and rocket performance against NASA CEA and published propellant data. Skips without Cantera."""
import pytest

from wobbl.chem import install
from wobbl.chem.thermo import Equilibrium, Reactants, rocket

needs_cantera = pytest.mark.skipif(not install.available(), reason="Cantera not installed (wobbl setup --only cantera)")
PSIA = 6894.757


def _lox_lh2(of: float) -> Reactants:
    """Liquid oxygen and hydrogen at their boiling points, with CEA's assigned enthalpies (J/mol)."""
    n_o2, n_h2 = of / (1 + of) * 1000 / 31.9988, 1 / (1 + of) * 1000 / 2.01588
    return Reactants({"O": 2 * n_o2, "H": 2 * n_h2}, n_o2 * -12979.0 + n_h2 * -9012.0, "lox/lh2")


@needs_cantera
def test_lox_lh2_matches_the_cea_output():
    """CEA at 1000 psia, O/F 6, area ratio 40, as RocketCEA's documentation prints it: c* 7560.0 ft/s, Isp 431.2 s
    and 451.7 s in vacuum shifting, 419.3 s and 436.7 s frozen at the throat, chamber 3483.35 K."""
    eq = Equilibrium(_lox_lh2(6.0))
    ch, sh = rocket(eq, 1000 * PSIA, area_ratio=40.0)
    assert ch.T == pytest.approx(3483.35, rel=0.003)
    assert sh.cstar / 0.3048 == pytest.approx(7560.0, rel=0.002)
    assert sh.isp_opt == pytest.approx(431.2, rel=0.002)
    assert sh.isp_vac == pytest.approx(451.7, rel=0.002)
    assert sh.stations[1].P / 101325 == pytest.approx(39.216, rel=0.002)
    _, fr = rocket(eq, 1000 * PSIA, area_ratio=40.0, mode="frozen-throat", chamber=ch)
    assert fr.isp_opt == pytest.approx(419.3, rel=0.002)
    assert fr.isp_vac == pytest.approx(436.7, rel=0.002)
    assert eq.element_error(ch) < 1e-9 and sh.entropy_error < 1e-6
