"""Standard thermochemistry of a species in a phase: enthalpy, entropy and heat capacity at a temperature.

Gases, solids and liquids come from NASA Glenn's database (McBride, Zehe and Gordon, NASA TP-2002-211556), the data
behind CEA, built on JANAF, ATcT and Gurvich and valid over each phase's temperature range. Its enthalpy is the heat
of formation at 298.15 K plus the sensible heat, so a reaction's enthalpy is the products' sum less the
reactants'. Ions and other species in water, and the few solids and liquids NASA lacks, come from the NBS tables
(Wagman et al., J. Phys. Chem. Ref. Data 11, Suppl. 2, 1982) at 298.15 K, carried to other temperatures with a
constant heat capacity. Entropies of ions are on the usual scale where the hydrogen ion's is zero, which cancels in
any reaction that balances its charge.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache


T0 = 298.15
NASA = "NASA Glenn thermodynamic data (JANAF and ATcT based)"
NBS = "NBS tables (Wagman et al. 1982)"


@dataclass
class Data:
    formula: str
    phase: str               # g, l, s or aq
    name: str                # the database's name for it
    source: str
    h298: float              # J/mol, heat of formation at 298.15 K
    s298: float              # J/(mol K)
    cp298: float | None
    lo: float = 200.0        # K, where the data holds
    hi: float = 6000.0
    _sp: object = None       # the Cantera species, for NASA data

    def H(self, T: float) -> float:
        if self._sp is not None:
            return self._sp.thermo.h(_clip(T, self.lo, self.hi)) / 1000.0 + self._out_h(T)
        return self.h298 + (self.cp298 or 0.0) * (T - T0)

    def S(self, T: float) -> float:
        if self._sp is not None:
            Tc = _clip(T, self.lo, self.hi)
            return self._sp.thermo.s(Tc) / 1000.0 + (self.Cp(Tc) * math.log(T / Tc) if T != Tc else 0.0)
        return self.s298 + (self.cp298 or 0.0) * math.log(T / T0)

    def Cp(self, T: float) -> float:
        if self._sp is not None:
            return self._sp.thermo.cp(_clip(T, self.lo, self.hi)) / 1000.0
        return self.cp298 or 0.0

    def G(self, T: float) -> float:
        return self.H(T) - T * self.S(T)

    def _out_h(self, T: float) -> float:
        Tc = _clip(T, self.lo, self.hi)
        return self.Cp(Tc) * (T - Tc) if T != Tc else 0.0

    def covers(self, T: float) -> bool:
        return self.lo - 5 <= T <= self.hi + 5     # NASA's solids start at 300 K, which covers 298.15 K


def _clip(T, lo, hi):
    return min(max(T, lo), hi)


# ---------------------------------------------------------------------------------- NBS tables at 298.15 K
# (heat of formation kJ/mol, entropy J/(mol K), heat capacity J/(mol K) or None), by Hill formula and phase
NBS_TABLE: dict[tuple[str, str], tuple[float, float, float | None]] = {
    # ions in water
    ("H+", "aq"): (0.0, 0.0, 0.0), ("HO-", "aq"): (-229.994, -10.75, -148.5),
    ("Na+", "aq"): (-240.12, 59.0, 46.4), ("K+", "aq"): (-252.38, 102.5, 21.8), ("Li+", "aq"): (-278.49, 13.4, 68.6),
    ("Ca^2+", "aq"): (-542.83, -53.1, None), ("Mg^2+", "aq"): (-466.85, -138.1, None),
    ("Ba^2+", "aq"): (-537.64, 9.6, None), ("Sr^2+", "aq"): (-545.80, -32.6, None),
    ("Ag+", "aq"): (105.579, 72.68, 21.8), ("Cu^2+", "aq"): (64.77, -99.6, None),
    ("Zn^2+", "aq"): (-153.89, -112.1, 46.0), ("Fe^2+", "aq"): (-89.1, -137.7, None),
    ("Fe^3+", "aq"): (-48.5, -315.9, None), ("Al^3+", "aq"): (-531.0, -321.7, None),
    ("Pb^2+", "aq"): (-1.7, 10.5, None), ("Ni^2+", "aq"): (-54.0, -128.9, -46.0),
    ("H4N+", "aq"): (-132.51, 113.4, 79.9),
    ("Cl-", "aq"): (-167.159, 56.5, -136.4), ("Br-", "aq"): (-121.55, 82.4, -141.8),
    ("I-", "aq"): (-55.19, 111.3, -142.3), ("F-", "aq"): (-332.63, -13.8, -106.7),
    ("NO3-", "aq"): (-205.0, 146.4, -86.6), ("O4S^2-", "aq"): (-909.27, 20.1, -293.0),
    ("HO4S-", "aq"): (-887.34, 131.8, -84.0), ("CO3^2-", "aq"): (-677.14, -56.9, None),
    ("CHO3-", "aq"): (-691.99, 91.2, None), ("C2H3O2-", "aq"): (-486.01, 86.6, -6.3),
    ("CHO2-", "aq"): (-425.55, 92.0, -87.9), ("ClO-", "aq"): (-107.1, 42.0, None),
    ("O4P^3-", "aq"): (-1277.4, -222.0, None),
    # neutral species in water
    ("C2H4O2", "aq"): (-485.76, 178.7, None), ("CO2", "aq"): (-413.80, 117.6, None),
    ("CH2O3", "aq"): (-699.65, 187.4, None), ("H3N", "aq"): (-80.29, 111.3, None),
    ("H2O2", "aq"): (-191.17, 143.9, None), ("O2", "aq"): (-11.7, 110.9, None), ("H2", "aq"): (-4.2, 57.7, None),
    ("CH4O", "aq"): (-245.931, 133.1, None), ("C2H6O", "aq"): (-288.3, 148.5, None),
    ("C6H12O6", "aq"): (-1263.78, 264.0, None),
    # solids and liquids NASA's database lacks
    ("CHNaO3", "s"): (-950.81, 101.7, 87.61), ("AgCl", "s"): (-127.068, 96.2, 50.79),
    ("AgNO3", "s"): (-124.39, 140.92, 93.05), ("OZn", "s"): (-348.28, 43.64, 40.25),
    ("KNO3", "s"): (-494.63, 133.05, 96.40), ("NNaO3", "s"): (-467.85, 116.52, 92.88),
    ("Cl2Zn", "s"): (-415.05, 111.46, 71.34), ("C2H3NaO2", "s"): (-708.81, 123.0, 79.9),
    ("C6H12O6", "s"): (-1273.3, 212.1, 218.8), ("C12H22O11", "s"): (-2226.1, 360.2, 425.0),
    ("C2H6O", "l"): (-277.69, 160.7, 111.46), ("C2H4O2", "l"): (-484.5, 159.8, 123.3),
    ("CH4O", "l"): (-239.2, 126.8, 81.1), ("H2O2", "l"): (-187.78, 109.6, 89.1),
    ("C3H6O", "l"): (-248.4, 199.8, 125.45), ("HNO3", "l"): (-174.10, 155.60, 109.87),
    ("BaO4S", "s"): (-1473.2, 132.2, 101.75), ("CaO4S", "s"): (-1434.11, 106.7, 99.66),
    ("O4PbS", "s"): (-919.94, 148.57, 103.207), ("ClNaO", "aq"): (-347.3, 100.4, None),
    ("Ag2O", "s"): (-31.05, 121.3, 65.86), ("CaH2O2", "s"): (-986.09, 83.39, 87.49),
}


@lru_cache(maxsize=1)
def _nasa():
    """NASA species by composition, gas and condensed, or None without Cantera."""
    from .install import available
    if not available():
        return None
    from .thermo import _database
    gas, cond = _database()
    by: dict[tuple, list] = {}
    for sp in gas:
        if sp.charge:
            continue
        key = _key(sp.composition)
        by.setdefault((key, "g"), []).append(sp)
    for sp in cond:
        key = _key(sp.composition)
        ph = "l" if "(L)" in sp.name or sp.name.endswith("(L)") else "s"
        by.setdefault((key, ph), []).append(sp)
    return by


def _key(comp: dict) -> tuple:
    return tuple(sorted((k.capitalize() if len(k) > 1 else k, round(float(v), 4)) for k, v in comp.items()))


def _counts_key(counts: dict) -> tuple:
    return tuple(sorted((k, round(float(v), 4)) for k, v in counts.items()))


# NASA names its species by formula, so an isomer is only certain where the name says which one it is. These are
# the SMILES of the closed shell organic species in NASA's gas data with three or more heavy atoms, and of the two
# heavy atom ones that share a formula with another species.
NAME_SMILES = {
    "C2H5OH": "CCO", "CH3OCH3": "COC", "CH3CN": "CC#N", "C2H4O,ethylen": "C1CO1", "CH3CHO,ethanal": "CC=O",
    "CH3COOH": "CC(=O)O", "CH3N2CH3": "CN=NC", "C3H4,allene": "C=C=C", "C3H4,propyne": "C#CC",
    "C3H4,cyclo-": "C1=CC1", "C3H6,propylene": "C=CC", "C3H6,cyclo-": "C1CC1", "C3H6O": "CC1CO1", "C3H8": "CCC",
    "C3H8O,1propanol": "CCCO", "C3H8O,2propanol": "CC(C)O", "C4H2": "C#CC#C", "C4H6,butadiene": "C=CC=C",
    "C4H6,2-butyne": "CC#CC", "C4H6,cyclo-": "C1=CCC1", "C4H8,1-butene": "C=CCC", "C4H8,cis2-buten": "C/C=C\\C",
    "C4H8,tr2-butene": "C/C=C/C", "C4H8,isobutene": "C=C(C)C", "C4H8,cyclo-": "C1CCC1", "C4H10,isobutane": "CC(C)C",
    "C4H10,n-butane": "CCCC", "C5H6,1,3cyclo-": "C1=CCC=C1", "C5H8,cyclo-": "C1=CCCC1", "C5H10,1-pentene": "C=CCCC",
    "C5H10,cyclo-": "C1CCCC1", "C5H12,n-pentane": "CCCCC", "C5H12,i-pentane": "CCC(C)C",
    "CH3C(CH3)2CH3": "CC(C)(C)C", "C6H6": "c1ccccc1", "C6H5OH,phenol": "Oc1ccccc1", "C6H10,cyclo-": "C1=CCCCC1",
    "C6H12,1-hexene": "C=CCCCC", "C6H12,cyclo-": "C1CCCCC1", "C7H8": "Cc1ccccc1", "C7H14,1-heptene": "C=CCCCCC",
    "C7H16,n-heptane": "CCCCCCC", "C8H8,styrene": "C=Cc1ccccc1", "C8H10,ethylbenz": "CCc1ccccc1",
    "C8H16,1-octene": "C=CCCCCCC", "C8H18,isooctane": "CC(C)CC(C)(C)C", "C8H18,n-octane": "CCCCCCCC",
    "C10H8,naphthale": "c1ccc2ccccc2c1", "C12H10,bipheny": "c1ccc(-c2ccccc2)cc1", "HNCO": "N=C=O",
    "HNO3": "O[N+](=O)[O-]", "HNO2": "ON=O", "HCOOH": "OC=O", "H2SO4": "OS(=O)(=O)O", "CH2CO,ketene": "C=C=O",
    "CHCL3": "ClC(Cl)Cl", "CH2CL2": "ClCCl", "CHF3": "FC(F)F", "CH2F2": "FCF", "C2HCL": "C#CCl", "C2HF": "C#CF",
    "CHCLF2": "FC(F)Cl", "CHCL2F": "FC(Cl)Cl", "CH2CLF": "FCCl", "C2H2,acetylene": "C#C", "C2H2,vinylidene": "C=[C]",
    "C2H4": "C=C", "C2H6": "CC", "CH4": "C", "CH3OH": "CO", "H2O2": "OO", "HCN": "C#N", "HNC": "[C-]#[NH+]",
    "CH2O": "C=O", "CH3": "[CH3]", "N2H4": "NN",
}


def _canonical(smi: str) -> str:
    if not smi:
        return ""
    try:
        from .install import load_pkg
        if load_pkg("rdkit") is None:
            return smi
        from rdkit import Chem
        m = Chem.MolFromSmiles(smi)
        return Chem.MolToSmiles(m) if m is not None else smi
    except Exception:  # noqa: BLE001
        return smi


@lru_cache(maxsize=1)
def _name_smiles() -> dict:
    return {k: _canonical(v) for k, v in NAME_SMILES.items()}


def _isomer_ok(sp, cands: list) -> list:
    """The NASA candidates that are this species and not an isomer of it."""
    heavy = sum(n for e, n in sp.counts.items() if e != "H")
    if not sp.smiles:
        return [c for c in cands if "," not in c.name][:1] or cands[:1]
    want = _canonical(sp.smiles)
    known = _name_smiles()
    match = [c for c in cands if known.get(c.name) == want]
    if match:
        return match[:1]
    if heavy >= 3 and "C" in sp.counts and "H" in sp.counts:     # organic, where isomers abound
        return []
    return [c for c in cands if c.name not in known][:1]


def lookup(sp, phase: str = "", T: float = T0) -> Data | None:
    """The species' data in the phase, or in its stable phase at T when the phase is left out."""
    hill = sp.formula
    if phase in ("", "aq") and (hill, "aq") in NBS_TABLE and (phase == "aq" or sp.charge):
        return _nbs(hill, "aq", sp.name)
    phases = [phase] if phase else ["g", "l", "s"]
    best = None
    by = _nasa()
    for ph in phases:
        cands = []
        if by is not None and not sp.charge:
            cands = by.get((_counts_key(sp.counts), ph), [])
        if ph == "g" and cands:
            cands = _isomer_ok(sp, cands)
        ok = [c for c in cands if c.thermo.min_temp - 5 <= T <= c.thermo.max_temp + 5] or cands
        for c in ok:
            d = Data(hill, ph, c.name, NASA, c.thermo.h(T0) / 1000.0, c.thermo.s(T0) / 1000.0,
                     c.thermo.cp(T0) / 1000.0, c.thermo.min_temp, c.thermo.max_temp, c)
            if best is None or (not phase and d.covers(T) and (not best.covers(T) or d.G(T) < best.G(T))):
                best = d
        if (hill, ph) in NBS_TABLE and (best is None or (phase and best.phase != ph)):
            d = _nbs(hill, ph, sp.name)
            if best is None or (not phase and d.G(T) < best.G(T) and abs(T - T0) < 100):
                best = d
    return best


def _nbs(hill: str, ph: str, name: str) -> Data:
    h, s, cp = NBS_TABLE[(hill, ph)]
    return Data(hill, ph, f"{name}({ph})", NBS, h * 1000.0, s, cp, 250.0, 373.0 if ph == "aq" else 600.0)
