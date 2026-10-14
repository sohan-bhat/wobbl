"""Reaction thermochemistry on NASA and NBS data: heats, Gibbs energies, equilibrium and adiabatic temperatures.
Runs where Cantera is installed, since NASA's data ship with it."""
import pytest

from wobbl.chem import install
from wobbl.chem.molecule import Options
from wobbl.chem.resolve import resolve

pytestmark = pytest.mark.skipif(not install.available(), reason="Cantera not installed")


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    monkeypatch.setenv("WOBBL_OFFLINE", "1")


def _run(text, tmp_path, T=298.15, P=101325.0):
    from wobbl.chem.react import run_reaction
    q = resolve(text)
    return run_reaction(q, Options(temperature=q.temperature or T, pressure=q.pressure or P, accuracy="fast",
                                   render=False), tmp_path)


def test_water_from_hydrogen_matches_janaf(tmp_path):
    o = _run("H2 + 1/2 O2 -> H2O(g)", tmp_path)
    assert o.values["dH_kJ"] == pytest.approx(-241.83, abs=0.05)          # JANAF and ATcT, -241.826
    assert o.values["dG_kJ"] == pytest.approx(-228.6, abs=0.1)
    assert o.values["log10K"] == pytest.approx(40.05, abs=0.05)
    liquid = _run("2 H2 + O2 -> 2 H2O", tmp_path / "l")
    assert liquid.values["dH_kJ"] == pytest.approx(-571.66, abs=0.1)        # with liquid water, the textbook value
    assert liquid.values["vapour_dH_kJ"] == pytest.approx(-483.65, abs=0.1)
    assert liquid.values["adiabatic"]["T_K"] == pytest.approx(3077, abs=15)  # stoichiometric in oxygen, dissociated


def test_baking_soda_and_vinegar_cools_and_fizzes(tmp_path):
    o = _run("baking soda and vinegar", tmp_path)
    assert o.values["dH_kJ"] == pytest.approx(31.1, abs=0.3)               # NBS tables
    assert o.values["adiabatic"]["rise_K"] == pytest.approx(-6.2, abs=0.3)
    assert o.values["gas_L"] == pytest.approx(24.5, abs=0.2)
    assert o.values["dG_kJ"] < 0


def test_thermite_is_held_at_the_boiling_point_of_iron(tmp_path):
    o = _run("thermite", tmp_path)
    assert o.values["dH_kJ"] == pytest.approx(-851.5, abs=2.0)
    assert o.values["adiabatic"]["T_K"] == pytest.approx(3135, abs=15)


def test_precipitation_and_ions(tmp_path):
    o = _run("silver nitrate and sodium chloride", tmp_path)
    assert o.values["dH_kJ"] == pytest.approx(-65.5, abs=0.5)
    na = _run("sodium in water", tmp_path / "na")
    assert na.values["dH_kJ"] == pytest.approx(-368.6, abs=1.0)
    assert na.values["gas_from_condensed"]


def test_haber_equilibrium_at_pressure(tmp_path):
    o = _run("N2 + 3 H2 <=> 2 NH3 at 700 K and 200 bar", tmp_path)
    assert o.values["dG_kJ"] > 0 and 0.3 < o.values["equilibrium"]["extent"] < 0.6
    low = _run("N2 + 3 H2 <=> 2 NH3 at 700 K and 1 bar", tmp_path / "low")
    assert low.values["equilibrium"]["extent"] < o.values["equilibrium"]["extent"] / 5
