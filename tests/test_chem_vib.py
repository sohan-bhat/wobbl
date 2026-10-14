"""Normal modes, IR and thermochemistry: exact on model Hessians, and on GFN2-xTB where tblite is installed."""
import numpy as np
import pytest

from wobbl.chem.thermochem import rrho, symmetry_number
from wobbl.chem.vib import TO_CM, normal_modes, spectrum

WATER = np.array([[0.0, 0.0, 0.1173], [0.0, 0.7572, -0.4692], [0.0, -0.7572, -0.4692]])
METHANE = np.array([[0, 0, 0], [0.629, 0.629, 0.629], [-0.629, -0.629, 0.629], [-0.629, 0.629, -0.629],
                    [0.629, -0.629, -0.629]], float)


def test_symmetry_numbers_come_from_the_structure():
    assert symmetry_number(WATER, ["O", "H", "H"]) == 2
    assert symmetry_number(METHANE, ["C", "H", "H", "H", "H"]) == 12
    assert symmetry_number(np.array([[0, 0, 0], [0, 0, 1.16], [0, 0, -1.16]]), ["C", "O", "O"]) == 2
    assert symmetry_number(np.array([[0, 0, 0], [0, 0, 1.13]]), ["C", "O"]) == 1
    assert symmetry_number(np.array([[0, 0, 0], [0.1, 0.9, 0.3], [1.2, -0.2, 0.1]]), ["C", "N", "O"]) == 1


def test_a_diatomic_spring_gives_its_closed_form_frequency():
    k, m = 10.0, 1.0                                     # eV/A^2 and amu, both atoms H for simplicity
    x = np.array([[0, 0, 0], [0, 0, 0.74]])
    H = np.zeros((6, 6))
    for a, b, s in ((2, 2, 1), (5, 5, 1), (2, 5, -1), (5, 2, -1)):
        H[a, b] = s * k
    md = normal_modes(H, np.zeros((6, 3)), x, np.array([m, m]), ["H", "H"], [(0, 1, 1.0)])
    assert len(md.freqs) == 1 and md.linear
    assert md.freqs[0] == pytest.approx(np.sqrt(k / (m / 2)) * TO_CM, rel=1e-9)
    assert md.labels == ["H-H stretch"] and md.reduced_masses[0] == pytest.approx(1.0)     # as Gaussian reports it


def test_ideal_gas_entropy_of_argon_and_water():
    ar = rrho(np.zeros(0), np.zeros((1, 3)), ["Ar"])
    assert ar.S == pytest.approx(154.8, abs=0.2)                  # Sackur-Tetrode at 298.15 K and 1 atm
    assert ar.Cp == pytest.approx(20.786, abs=0.01)
    w = rrho(np.array([1595.0, 3657.0, 3756.0]), WATER, ["O", "H", "H"])
    assert w.S == pytest.approx(188.8, abs=1.0) and w.sigma == 2    # NIST, 188.84 J/(mol K)
    assert w.zpe == pytest.approx(53.5, abs=1.0)


def test_spectrum_peaks_where_the_bands_are():
    x, y = spectrum(np.array([1000.0, 2000.0]), np.array([10.0, 30.0]))
    assert x[np.argmax(y)] == pytest.approx(2000.0, abs=3)
    assert np.trapezoid(y, x) == pytest.approx(40.0, rel=0.03)


def test_gfn2_water_minimum_and_bands():
    pytest.importorskip("wobbl.chem.install")
    from wobbl.chem import install
    if not (install.available_pkg("tblite") and install.available_pkg("ase")):
        pytest.skip("tblite and ASE not installed")
    from ase import Atoms
    from ase.optimize import BFGS

    from wobbl.chem.vib import hessian
    from wobbl.chem.xtb import XTB
    eng = XTB([8, 1, 1])
    at = Atoms("OH2", positions=WATER)
    at.calc = eng.ase()
    BFGS(at, logfile=None).run(fmax=0.001)
    H, dmu = hessian(eng, at.get_positions())
    md = normal_modes(H, dmu, at.get_positions(), at.get_masses(), ["O", "H", "H"], [(0, 1, 1.0), (0, 2, 1.0)])
    assert len(md.freqs) == 3 and not md.imaginary()
    assert md.labels[0] == "H-O-H bend" and md.labels[1] == "O-H stretch"
    assert at.get_distance(0, 1) == pytest.approx(0.959, abs=0.01)
