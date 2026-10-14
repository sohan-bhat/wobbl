"""Equilibrium and rocket performance against NASA CEA and published propellant data. Skips without Cantera."""
import pytest

from wobbl.chem import install
from wobbl.chem.propellants import PROPELLANTS, get_propellant, parse_composition, reactants
from wobbl.chem.thermo import Equilibrium, cstar_ideal, rocket

needs_cantera = pytest.mark.skipif(not install.available(), reason="Cantera not installed (wobbl setup --only cantera)")
PSIA = 6894.757


@needs_cantera
def test_lox_lh2_matches_the_cea_output():
    """CEA at 1000 psia, O/F 6, area ratio 40, as RocketCEA's documentation prints it: c* 7560.0 ft/s, Isp 431.2 s
    and 451.7 s in vacuum shifting, 419.3 s and 436.7 s frozen at the throat, chamber 3483.35 K."""
    eq = Equilibrium(reactants(parse_composition("lox:6,lh2:1")))
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


@needs_cantera
def test_sugar_propellants_match_their_published_chemistry():
    """Nakka's PROPEP numbers: KNSU 1720 K, 41.98 kg/kmol and c* 919 m/s at 1000 psia, and the T, M and k each
    preset carries for the motor."""
    p = PROPELLANTS["knsu"]
    eq = Equilibrium(reactants(p.composition))
    ch, fr = rocket(eq, 1000 * PSIA, exit_P=101325.0, mode="frozen")
    assert ch.T == pytest.approx(1720.0, rel=0.002)
    assert ch.M == pytest.approx(41.98, rel=0.002)
    assert fr.cstar == pytest.approx(919.0, rel=0.003)
    assert "K2CO3(L)" in ch.condensed and 0.40 < ch.condensed_fraction < 0.45
    assert cstar_ideal(ch.T, ch.M, ch.gamma) == pytest.approx(fr.cstar, rel=0.003)
    for key in ("knsb", "kndx"):
        q = PROPELLANTS[key]
        st = Equilibrium(reactants(q.composition)).hp(6.895e6)
        assert cstar_ideal(st.T, st.M, st.gamma) == pytest.approx(q.cstar, rel=0.01), key


@needs_cantera
def test_a_shifting_expansion_settles_on_the_melting_point():
    """KNSU's K2CO3 freezes in the nozzle: over a band of pressures the state sits at the melting point with a mix
    of solid and liquid whose share keeps the entropy exact."""
    eq = Equilibrium(reactants(PROPELLANTS["knsu"].composition))
    ch = eq.hp(6.895e6)
    st = eq.sp(1.6e5, ch.s, 1300.0)
    assert st.T == pytest.approx(1174.0, abs=0.01)
    assert {"K2CO3(s)", "K2CO3(L)"} <= set(st.condensed)
    assert abs(st.s - ch.s) / abs(ch.s) < 1e-9 and eq.element_error(st) < 1e-9
    assert eq.sp(1.0e5, ch.s, 1300.0).T < 1174.0 < eq.sp(2.0e5, ch.s, 1300.0).T


def test_burn_rate_bands_and_names():
    knsb = get_propellant("KNSB")
    assert knsb.burn_rate.rate(101325) / 0.0254 == pytest.approx(0.102, rel=0.02)   # Nakka: 0.102 in/s at 1 atm
    assert knsb.burn_rate.band(2e6) == (7.852, -0.013)
    assert knsb.burn_rate.band(20e6) == (9.653, 0.064)        # past the data the last band is extended
    assert get_propellant("kn-dextrose").key == "kndx"
    assert knsb.cstar == pytest.approx(908.1, rel=0.002)
    with pytest.raises(KeyError):
        get_propellant("unobtainium")
    assert parse_composition("kno3:65, sorbitol:35") == pytest.approx({"kno3": 0.65, "sorbitol": 0.35})
    with pytest.raises(ValueError):
        parse_composition("kno3 65")
