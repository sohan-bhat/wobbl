"""Heats of formation for molecules no table lists, estimated from GFN2-xTB.

GFN2-xTB's total energy plus its zero point and thermal enthalpy, less one fitted energy per atom of each element
(an atom equivalent), gives the heat of formation at 298.15 K. The atom equivalents are fitted by least squares to
NASA's tabulated heats of formation of small molecules, and the fit's own errors on molecules it did not see are
the estimate's stated accuracy. Entropies and heat capacities come from the same structure and frequencies.
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np

from .thermochem import EV_TO_KJ, rrho
from .thermodata import Data

T0 = 298.15
# kJ/mol per atom, fitted by fit() on TRAINING against NASA's heats of formation (see tests/test_chem_formation.py)
ATOM_EQ: dict[str, float] = {"C": -5631.84, "H": -1286.58, "N": -7619.10, "O": -10480.29, "F": -12160.14,
                              "Cl": -11885.16, "S": -8534.66}
FIT_MAE = 20.0     # kJ/mol, the median error on molecules left out of the fit one at a time (16 in sample)
# polyatomic molecules only: diatomics and the like are in every table, and GFN2-xTB's errors on their multiple
# bonds would only skew the fit for the larger molecules an estimate is ever needed for
TRAINING = ["C", "CC", "CCC", "CCCC", "CCCCC", "CC(C)C", "CC(C)(C)C", "C1CC1", "C1CCCCC1", "C=C", "C=CC", "C=CC=C",
            "C#C", "CC#C", "c1ccccc1", "Cc1ccccc1", "O", "N", "O=C=O", "CO", "CCO", "CCCO", "CC(O)C", "OCCO", "COC",
            "CCOC", "C1CO1", "C=O", "CC=O", "CC(C)=O", "C=CC=O", "OC=O", "CC(=O)O", "OO", "C#N", "CC#N", "CN", "CCN",
            "CNC", "NC=O", "c1ccncc1", "C1=CC=CO1", "Cl", "CCl", "ClCCl", "ClC(Cl)Cl", "ClC(Cl)(Cl)Cl", "ClC=C",
            "F", "CF", "FC=C", "CC(F)(F)F", "S", "CS", "CCS", "CSC", "O=S=O", "S=C=S", "O=C=S", "Br", "CBr"]


@lru_cache(maxsize=256)
def _xtb(smiles: str, multiplicity: int = 0):
    """(enthalpy at 298.15 K in kJ/mol on GFN2-xTB's scale, thermo object, structure)."""
    from .molecule import optimize
    from .structure import from_smiles
    from .vib import hessian, normal_modes
    from .xtb import XTB
    st = from_smiles(smiles, conformers=4)
    if multiplicity:
        st.multiplicity = multiplicity
    eng = XTB(st.numbers, st.charge, st.uhf)
    if len(st.symbols) == 1:
        e = eng.compute(st.positions).energy
        th = rrho(np.zeros(0), st.positions, st.symbols, T0, 101325.0, st.multiplicity)
        return e * EV_TO_KJ + th.H, th, st
    o = optimize(eng, st.symbols, st.positions, fmax=0.005)
    H, dmu = hessian(eng, o.positions)
    md = normal_modes(H, dmu, o.positions, st.masses, st.symbols, st.bonds)
    th = rrho(md.freqs, o.positions, st.symbols, T0, 101325.0, st.multiplicity)
    return o.energy * EV_TO_KJ + th.H, th, st


def xtb_enthalpy(sp) -> float:
    """GFN2-xTB enthalpy at 298.15 K, J/mol, on its own energy scale, for reaction energies."""
    mult = 3 if sp.formula == "O2" else (sp.multiplicity or 0)
    return _xtb(sp.smiles, mult)[0] * 1000.0


def estimate(sp, phase: str, T: float) -> Data | None:
    """A gas molecule's estimated data, or None when it has no structure or an element the fit lacks."""
    if phase not in ("g", "") or not sp.smiles or not ATOM_EQ or any(e not in ATOM_EQ for e in sp.counts):
        return None
    mult = 3 if sp.formula == "O2" else (sp.multiplicity or 0)
    try:
        h, th, st = _xtb(sp.smiles, mult)
    except Exception:  # noqa: BLE001
        return None
    dfh = h - sum(ATOM_EQ[e] * n for e, n in sp.counts.items())
    return Data(sp.formula, "g", sp.name, f"estimated with GFN2-xTB atom equivalents (about {FIT_MAE:.0f} kJ/mol)",
                dfh * 1000.0, th.S, th.Cp, 250.0, 1500.0)


def fit(smiles_list: list[str]) -> tuple[dict, list]:
    """Least squares atom equivalents against NASA's heats of formation. Returns them and the residuals."""
    from .resolve import _from_smiles
    from .thermodata import lookup

    def species(smi, allow_online=False):
        return _from_smiles(smi, allow_online)
    rows, ys, used = [], [], []
    for smi in smiles_list:
        sp = species(smi)
        d = lookup(sp, "g", T0)
        if d is None or d.source.startswith("estimated"):
            continue
        mult = 3 if sp.formula == "O2" else 0
        h, _, _ = _xtb(smi, mult)
        rows.append(dict(sp.counts))
        ys.append(h - d.h298 / 1000.0)
        used.append((smi, d.h298 / 1000.0, h))
    els = sorted({e for r in rows for e in r})
    A = np.array([[r.get(e, 0) for e in els] for r in rows], float)
    eq, *_ = np.linalg.lstsq(A, np.array(ys), rcond=None)
    out = dict(zip(els, eq))
    resid = [(smi, ref, h - sum(out[e] * n for e, n in species(smi).counts.items()) - ref) for smi, ref, h in used]
    return out, resid
