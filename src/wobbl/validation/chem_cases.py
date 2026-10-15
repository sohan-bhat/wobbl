"""Validation cases for general chemistry: molecular geometry and frequencies against NIST's CCCBDB measurements,
heats of formation against ATcT, a reaction enthalpy against JANAF, lattice constants against measured crystals and
energy conservation in dynamics against the order of the integrator.

Each runs through the same chemistry pipeline a user gets. Small molecules use the top rung of the method ladder
(B3LYP/def2-TZVP, accuracy high), which is what the comparison with experiment is about.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

PSI = 6894.757


@lru_cache(maxsize=8)
def _molecule(thing: str, accuracy: str):
    import tempfile

    from ..chem.run import simulate
    out = Path(tempfile.mkdtemp(prefix="wobbl-val-"))
    return simulate(thing, "", accuracy=accuracy, out=out, render=False, options={"md": False})


def _bond(o, a: str, b: str) -> float:
    import numpy as np
    pos = np.array(o.values["positions"])
    syms = o.values["symbols"]
    ds = [np.linalg.norm(pos[i] - pos[j]) for i in range(len(syms)) for j in range(i + 1, len(syms))
          if sorted((syms[i], syms[j])) == sorted((a, b)) and np.linalg.norm(pos[i] - pos[j]) < 1.7]
    return float(np.mean(ds))


def water_oh(out: Path):
    """Water's O-H bond with B3LYP/def2-TZVP, against NIST CCCBDB's 0.958 A."""
    o = _molecule("water", "high")
    return _bond(o, "O", "H"), 0.958, {"method": o.values["method"], "confidence": o.confidence}


def water_angle(out: Path):
    """Water's H-O-H angle with B3LYP/def2-TZVP, against NIST CCCBDB's 104.5 degrees."""
    import numpy as np
    o = _molecule("water", "high")
    p = np.array(o.values["positions"])
    u, v = p[1] - p[0], p[2] - p[0]
    ang = float(np.degrees(np.arccos(np.dot(u, v) / np.linalg.norm(u) / np.linalg.norm(v))))
    return ang, 104.5, {"method": o.values["method"]}


def _scaled(o) -> list[float]:
    s = o.values.get("frequency_scale", 1.0)
    return sorted(f * s for f in o.values["frequencies_cm"])


def water_bend(out: Path):
    """Water's bending fundamental, B3LYP/def2-TZVP harmonic scaled by CCCBDB's 0.967, against 1595 cm-1."""
    o = _molecule("water", "high")
    f = _scaled(o)
    return f[0], 1595.0, {"all_cm": [round(x) for x in f], "measured_cm": [1595, 3657, 3756]}


def water_stretch(out: Path):
    """Water's antisymmetric stretch, the same way, against 3756 cm-1."""
    o = _molecule("water", "high")
    return _scaled(o)[-1], 3756.0, {}


def co2_stretch(out: Path):
    """CO2's antisymmetric stretch, B3LYP/def2-TZVP scaled by 0.967, against 2349 cm-1."""
    o = _molecule("carbon dioxide", "high")
    f = _scaled(o)
    return f[-1], 2349.0, {"all_cm": [round(x) for x in f], "measured_cm": [667, 667, 1333, 2349]}


def co2_bend(out: Path):
    """CO2's bending fundamental, the same way, against 667 cm-1."""
    o = _molecule("carbon dioxide", "high")
    return _scaled(o)[0], 667.0, {}


def benzene_cc(out: Path):
    """Benzene's C-C bond from GFN2-xTB at normal accuracy, against NIST CCCBDB's 1.397 A."""
    o = _molecule("benzene", "normal")
    return _bond(o, "C", "C"), 1.397, {"method": o.values["method"]}


def formation_tables(out: Path):
    """Heats of formation Wobbl looks up for 12 small molecules, against ATcT 1.130: the largest difference in
    kJ/mol, compared with 0 by adding 1000 to both."""
    from ..chem.reference import HEATS_OF_FORMATION
    from ..chem.resolve import _from_smiles
    from ..chem.thermodata import lookup
    smiles = {"H2O": "O", "CO2": "O=C=O", "CH4": "C", "H3N": "N", "C2H6": "CC", "C2H4": "C=C", "C2H2": "C#C",
              "CH4O": "CO", "CH2O": "C=O", "C6H6": "c1ccccc1", "O2S": "O=S=O", "C2H6O": "CCO"}
    worst, rows = 0.0, {}
    for f, smi in smiles.items():
        d = lookup(_from_smiles(smi, False), "g", 298.15)
        diff = d.h298 / 1000 - HEATS_OF_FORMATION[f]
        rows[f] = round(diff, 2)
        worst = max(worst, abs(diff))
    return 1000.0 + worst, 1000.0, {"differences_kJ_mol": rows}


def formation_estimate(out: Path):
    """GFN2-xTB atom equivalent heats of formation for five molecules, against ATcT: the mean absolute error in kJ/mol
    against the 20 kJ/mol the estimate claims, passing while it stays within twice that."""
    from ..chem.formation import estimate
    from ..chem.reference import HEATS_OF_FORMATION
    from ..chem.resolve import _from_smiles
    test = {"C3H8": "CCC", "CH2O2": "OC=O", "C2H4O2": "CC(=O)O", "H2S": "S", "C3H6O": "CC(C)=O"}
    errs = {}
    for f, smi in test.items():
        d = estimate(_from_smiles(smi, False), "g", 298.15)
        errs[f] = round(d.h298 / 1000 - HEATS_OF_FORMATION[f], 1)
    mae = sum(abs(v) for v in errs.values()) / len(errs)
    return mae, 20.0, {"errors_kJ_mol": errs}


def water_from_hydrogen(out: Path):
    """H2 + 1/2 O2 -> H2O(g) at 298.15 K, against JANAF's -241.826 kJ/mol."""
    from ..chem.molecule import Options
    from ..chem.react import run_reaction
    from ..chem.resolve import resolve
    o = run_reaction(resolve("H2 + 1/2 O2 -> H2O(g)"), Options(accuracy="fast", render=False), out)
    return o.values["dH_kJ"], -241.826, {"dG_kJ": round(o.values["dG_kJ"], 3)}


def _crystal(thing: str, out: Path):
    from ..chem.run import simulate
    return simulate(thing, "", out=out, render=False)


def nacl_lattice(out: Path):
    """Table salt's lattice constant from the Born-Mayer ionic model, against the measured 5.640 A."""
    o = _crystal("table salt", out)
    return o.values["lattice_A"], 5.640, {"lattice_energy_kJ_mol": round(o.values["lattice_energy_kJ_mol"]),
                                         "bulk_modulus_GPa": round(o.values["bulk_modulus_GPa"], 1)}


def cu_lattice(out: Path):
    """Copper's lattice constant from EMT, against the measured 3.615 A."""
    o = _crystal("copper", out)
    return o.values["lattice_A"], 3.615, {"bulk_modulus_GPa": round(o.values["bulk_modulus_GPa"], 1),
                                         "cohesive_eV_atom": round(o.values["cohesive_eV_atom"], 3)}


def argon_nve(out: Path):
    """Liquid argon at 85 K at constant energy: halving the time step should shrink the energy wobble 4 times for
    the second order velocity Verlet integrator."""
    from ..chem.run import simulate
    o = simulate("argon", "molecular dynamics at 85 K", out=out, render=False)
    return o.values["ratio"], 4.0, {"drift_meV_atom": o.values["drift_meV_atom"], "atoms": o.values["atoms"]}


def chem_cases():
    from .cases import Case
    return [
        Case("water-oh", "Water's O-H bond", "chemistry", "NIST CCCBDB, 0.958 A", "bond length", "A", 0.01, False,
             water_oh),
        Case("water-angle", "Water's H-O-H angle", "chemistry", "NIST CCCBDB, 104.5 degrees", "bond angle", "deg",
             0.015, False, water_angle),
        Case("water-bend", "Water's bending vibration", "chemistry", "NIST CCCBDB fundamental, 1595 cm-1",
             "frequency", "cm-1", 0.04, False, water_bend),
        Case("water-stretch", "Water's antisymmetric stretch", "chemistry", "NIST CCCBDB fundamental, 3756 cm-1",
             "frequency", "cm-1", 0.04, False, water_stretch),
        Case("co2-stretch", "CO2's antisymmetric stretch", "chemistry", "NIST CCCBDB fundamental, 2349 cm-1",
             "frequency", "cm-1", 0.04, False, co2_stretch),
        Case("co2-bend", "CO2's bending vibration", "chemistry", "NIST CCCBDB fundamental, 667 cm-1", "frequency",
             "cm-1", 0.04, False, co2_bend),
        Case("benzene-cc", "Benzene's C-C bond", "chemistry", "NIST CCCBDB, 1.397 A", "bond length", "A", 0.02, False,
             benzene_cc),
        Case("formation-tables", "Heats of formation of 12 small molecules from the tables", "chemistry",
             "ATcT 1.130", "largest difference plus 1000", "kJ/mol", 0.001, False, formation_tables),
        Case("formation-estimate", "Heats of formation estimated from GFN2-xTB for 5 molecules", "chemistry",
             "ATcT 1.130, against the claimed 20 kJ/mol", "mean absolute error", "kJ/mol", 1.0, False,
             formation_estimate),
        Case("water-from-hydrogen", "H2 + 1/2 O2 -> H2O(g)", "chemistry", "JANAF, -241.826 kJ/mol",
             "reaction enthalpy", "kJ/mol", 0.001, False, water_from_hydrogen),
        Case("nacl-lattice", "Table salt's lattice constant", "chemistry", "measured, 5.640 A", "lattice constant",
             "A", 0.015, False, nacl_lattice),
        Case("cu-lattice", "Copper's lattice constant", "chemistry", "measured, 3.615 A", "lattice constant", "A",
             0.015, False, cu_lattice),
        Case("argon-nve", "Liquid argon at constant energy", "chemistry", "velocity Verlet error falls as dt^2",
             "energy wobble ratio at half the step", "", 0.15, True, argon_nve),
    ]
