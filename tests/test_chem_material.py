"""Elements and crystals: Ewald sums against Madelung constants, and lattice constants, stiffness and energies
against measurements. Runs where ASE is installed."""
import numpy as np
import pytest

from wobbl.chem import install

pytestmark = pytest.mark.skipif(not install.available_pkg("ase"), reason="ASE not installed")


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    monkeypatch.setenv("WOBBL_OFFLINE", "1")


@pytest.mark.parametrize("proto,a,d,M", [("rocksalt", 5.64, 2.82, 1.747565),
                                         ("cesiumchloride", 4.123, 4.123 * np.sqrt(3) / 2, 1.762675)])
def test_ewald_sums_give_the_madelung_constants(proto, a, d, M):
    from ase.build import bulk

    from wobbl.chem.material import ewald
    at = bulk("NaCl" if proto == "rocksalt" else "CsCl", proto, a=a, cubic=True)
    q = np.array([1.0 if s in ("Na", "Cs") else -1.0 for s in at.get_chemical_symbols()])
    e = ewald(at.get_positions(), q, np.array(at.cell))
    assert -e / (len(at) / 2) * d / 14.399645 == pytest.approx(M, abs=1e-5)


def _run(text, tmp_path, question=""):
    from wobbl.chem.run import simulate
    return simulate(text, question, out=tmp_path, render=False)


def test_copper_by_emt(tmp_path):
    o = _run("copper", tmp_path)
    assert o.values["lattice_A"] == pytest.approx(3.615, rel=0.01)
    assert o.values["bulk_modulus_GPa"] == pytest.approx(137, rel=0.10)
    assert o.values["cohesive_eV_atom"] == pytest.approx(3.49, rel=0.05)
    assert o.confidence == "high"


def test_table_salt_by_the_ionic_model(tmp_path):
    o = _run("table salt", tmp_path)
    assert o.values["lattice_A"] == pytest.approx(5.64, rel=0.01)
    assert o.values["lattice_energy_kJ_mol"] == pytest.approx(787, rel=0.06)
    assert o.values["bulk_modulus_GPa"] == pytest.approx(24, rel=0.2)


def test_argon_crystal_and_a_solid_without_a_fast_method(tmp_path):
    o = _run("argon", tmp_path / "ar")
    assert o.values["lattice_A"] == pytest.approx(5.26, rel=0.01)
    assert o.values["cohesive_eV_atom"] == pytest.approx(0.080, rel=0.08)
    fe = _run("Fe", tmp_path / "fe")
    assert fe.values["computed"] is False and "measured" in fe.headline
