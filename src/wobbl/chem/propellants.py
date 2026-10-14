"""Propellant ingredients and the solid propellants Wobbl knows, with the source of every number.

An ingredient is a formula with its enthalpy of formation, which is all the equilibrium needs. A solid propellant
for a motor also needs its density, its burn rate law r = a P^n (in pressure bands where the data has them) and
the combustion properties amateur ballistics tools use, chamber temperature T, effective molar mass M and ratio of
specific heats k, from which c* follows. Wobbl simulates motors with these published numbers and checks them
against its own equilibrium calculation whenever the propellant's composition is known.

Wobbl models given designs only. Nothing here says how to make or handle a propellant.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field

from .thermo import R, Reactants

# IUPAC standard atomic weights, g/mol
ATOMIC = {"H": 1.00794, "C": 12.0107, "N": 14.0067, "O": 15.9994, "Na": 22.98977, "Mg": 24.305, "Al": 26.98154,
          "S": 32.065, "Cl": 35.453, "K": 39.0983, "Fe": 55.845}

NASA_TP = "McBride, Zehe and Gordon, NASA/TP-2002-211556, the CEA thermo data"


@dataclass(frozen=True)
class Ingredient:
    key: str
    name: str
    formula: dict[str, float]
    h_f: float                 # J/mol at the temperature below
    density: float             # kg/m3
    T: float = 298.15          # K
    source: str = ""

    @property
    def molar_mass(self) -> float:
        return sum(ATOMIC[e] * n for e, n in self.formula.items())


_I = Ingredient
INGREDIENTS: dict[str, Ingredient] = {i.key: i for i in [
    _I("kno3", "potassium nitrate", {"K": 1, "N": 1, "O": 3}, -494.0e3, 2109,
       source=f"{NASA_TP}, KNO3(a) from Gurvich 1982"),
    _I("sorbitol", "sorbitol", {"C": 6, "H": 14, "O": 6}, -1353.7e3, 1489,
       source="NIST Chemistry WebBook, from Gerasimov, Blokh and Gubareva 1985"),
    _I("dextrose", "dextrose (anhydrous glucose)", {"C": 6, "H": 12, "O": 6}, -1271.1e3, 1540,
       source="NIST Chemistry WebBook, alpha-D-glucose, from Ponomarev and Migarskaya 1960 as revised by Cox and "
              "Pilcher 1970"),
    _I("sucrose", "sucrose", {"C": 12, "H": 22, "O": 11}, -2221.2e3, 1587,
       source="NIST Chemistry WebBook, from Clarke and Stegeman 1939"),
    _I("ap", "ammonium perchlorate", {"N": 1, "H": 4, "Cl": 1, "O": 4}, -295.767e3, 1950,
       source=f"{NASA_TP}, NH4ClO4(I) from Chase 1998"),
    _I("al", "aluminum", {"Al": 1}, 0.0, 2700, source="reference element"),
    _I("htpb", "HTPB binder (R-45)", {"C": 7.3165, "H": 10.3360, "O": 0.1063}, 5.0208e3, 920,
       source="R-45 HTPB as defined in the Air Force Rocket Propulsion Laboratory data used by RocketCEA "
              "(1200 cal per 100 g formula unit). Published values span a wide range (Okninski, CEJEM 2022)"),
    _I("lox", "liquid oxygen", {"O": 2}, -12.979e3, 1141, 90.17, source=f"{NASA_TP}, O2(L) at its boiling point"),
    _I("lh2", "liquid hydrogen", {"H": 2}, -9.012e3, 70.8, 20.27, source=f"{NASA_TP}, H2(L) at its boiling point"),
    _I("lch4", "liquid methane", {"C": 1, "H": 4}, -89.233e3, 422.6, 111.643,
       source=f"{NASA_TP}, CH4(L) at its boiling point"),
    _I("rp1", "RP-1 kerosene", {"C": 1, "H": 1.95}, -24.7177e3, 810, source=f"{NASA_TP}, RP-1 from Mehta et al. 1995"),
]}


def parse_composition(text: str) -> dict[str, float]:
    """'kno3:65,sorbitol:35' or 'lox:6,lh2:1' as mass fractions that add up to one."""
    parts = {}
    for item in re.split(r"[,;]\s*", text.strip()):
        if not item:
            continue
        if ":" not in item and "=" not in item:
            raise ValueError(f"write each ingredient as name:parts, like kno3:65 (got '{item}')")
        k, v = re.split(r"[:=]", item, maxsplit=1)
        k = k.strip().lower()
        if k not in INGREDIENTS:
            raise ValueError(f"unknown ingredient '{k}' (known: {', '.join(INGREDIENTS)})")
        parts[k] = parts.get(k, 0.0) + float(v.strip().rstrip("%"))
    total = sum(parts.values())
    if total <= 0:
        raise ValueError("the composition needs at least one ingredient with a positive amount")
    return {k: v / total for k, v in parts.items()}


def reactants(composition: dict[str, float], label: str = "") -> Reactants:
    """Element moles and enthalpy per kg of a mixture given as mass fractions of ingredients."""
    el, h = {}, 0.0
    for key, w in composition.items():
        ing = INGREDIENTS[key]
        n = w * 1000.0 / ing.molar_mass          # mol per kg of mixture
        for e, k in ing.formula.items():
            el[e] = el.get(e, 0.0) + n * k
        h += n * ing.h_f
    return Reactants(el, h, label or ", ".join(f"{k} {100 * w:.4g}%" for k, w in composition.items()))


def ideal_density(composition: dict[str, float]) -> float:
    return 1.0 / sum(w / INGREDIENTS[k].density for k, w in composition.items())


@dataclass(frozen=True)
class BurnRate:
    """r = a P^n in pressure bands, with a in mm/s at P in MPa, the way the published tables give it."""
    bands: tuple[tuple[float, float, float, float], ...]     # (P min MPa, P max MPa, a, n)

    def band(self, P: float) -> tuple[float, float]:
        """(a, n) at P in Pa; outside the measured range the nearest band is extended."""
        mpa = P / 1e6
        for lo, hi, a, n in self.bands:
            if lo <= mpa < hi:
                return a, n
        return (self.bands[0][2], self.bands[0][3]) if mpa < self.bands[0][0] else (self.bands[-1][2],
                                                                                    self.bands[-1][3])

    def rate(self, P: float) -> float:
        """Burn rate in m/s at chamber pressure P in Pa."""
        a, n = self.band(P)
        return a * (max(P, 1.0) / 1e6) ** n / 1000.0

    @property
    def range(self) -> tuple[float, float]:
        return self.bands[0][0] * 1e6, self.bands[-1][1] * 1e6


@dataclass(frozen=True)
class Propellant:
    key: str
    name: str
    density: float                       # kg/m3 as cast
    burn_rate: BurnRate
    T: float                             # K, chamber temperature
    M: float                             # kg/kmol, effective molar mass (gas plus condensed over moles of gas)
    k: float                             # ratio of specific heats of the gas and condensed mixture
    composition: dict[str, float] = field(default_factory=dict)
    source: str = ""
    notes: str = ""
    aliases: tuple[str, ...] = ()

    @property
    def cstar(self) -> float:
        """Ideal characteristic velocity from T, M and k, m/s."""
        g = self.k
        return math.sqrt(R * self.T / self.M) / (math.sqrt(g) * (2 / (g + 1)) ** ((g + 1) / (2 * (g - 1))))

    def summary(self) -> str:
        lo, hi = self.burn_rate.range
        return (f"{self.name}: density {self.density:g} kg/m3, c* {self.cstar:.0f} m/s (T {self.T:g} K, M "
                f"{self.M:g}, k {self.k:g}), burn rate data from {lo / 1e6:g} to {hi / 1e6:g} MPa")


NAKKA = "Nakka, KN-Dextrose and KN-Sorbitol Propellants, Burn Rate Experimentation (nakka-rocketry.net/bntest.html)"
MIT = ("MIT Rocket Team, as published with their static fire data in openMotor (github.com/reilleya/openMotor, "
       "test/data/real)")

PROPELLANTS: dict[str, Propellant] = {p.key: p for p in [
    Propellant("knsb", "KNSB (65/35 potassium nitrate and sorbitol)", 1750.0,
               BurnRate(((0.103, 0.807, 10.71, 0.625), (0.807, 1.50, 8.763, -0.314), (1.50, 3.79, 7.852, -0.013),
                         (3.79, 7.03, 3.907, 0.535), (7.03, 10.67, 9.653, 0.064))),
               1600.0, 39.9, 1.1361, {"kno3": 0.65, "sorbitol": 0.35},
               source=f"burn rate from {NAKKA}, Table 2; T, M and k from Nakka's KN-Sorbitol data as listed in "
                      "openMotor's KNSB preset",
               notes="Density is about 95% of the 1841 kg/m3 ideal, typical of cast grains. A mesa propellant, "
                     "whose rate barely changes from 0.8 to 3.8 MPa.", aliases=("kn-sorbitol", "sorbitol")),
    Propellant("kndx", "KNDX (65/35 potassium nitrate and dextrose)", 1785.0,
               BurnRate(((0.103, 0.779, 8.88, 0.619), (0.779, 2.57, 7.55, -0.009), (2.57, 5.93, 3.84, 0.688),
                         (5.93, 8.50, 17.2, -0.148), (8.50, 11.20, 4.78, 0.442))),
               1710.0, 42.39, 1.1308, {"kno3": 0.65, "dextrose": 0.35},
               source=f"burn rate from {NAKKA}, Table 1; T, M and k from Nakka's KN-Dextrose data as listed in "
                      "openMotor's KNDX preset",
               notes="Density is about 95% of the 1879 kg/m3 ideal.", aliases=("kn-dextrose", "dextrose")),
    Propellant("knsu", "KNSU (65/35 potassium nitrate and sucrose)", 1800.0,
               BurnRate(((0.0, 10.34, 8.26, 0.319),)),
               1720.0, 41.98, 1.133, {"kno3": 0.65, "sucrose": 0.35},
               source="burn rate a = 8.26 mm/s, n = 0.319 from Nakka's strand burner (nakka-rocketry.net/design1.html); "
                      "T, M and k from Nakka's Technical Notepad 1, KNSU Ideal Performance Calculations",
               notes="Density is about 95% of the 1889 kg/m3 ideal.", aliases=("kn-sucrose", "sucrose", "rcandy")),
    Propellant("ocean-water", "MIT Ocean Water (APCP)", 1650.0, BurnRate(((0.0, 6.895, 2.8735, 0.382),)),
               3500.0, 23.67, 1.25, source=f"{MIT}, from the motor files of their 152 mm static fires",
               notes="An ammonium perchlorate composite whose exact recipe is not published, so the equilibrium "
                     "cross check cannot run. openMotor's current preset lists 2600 K where these static fire "
                     "files list 3500 K.", aliases=("ow", "apcp")),
    Propellant("cherry-limeade", "MIT Cherry Limeade (APCP)", 1670.0, BurnRate(((0.0, 6.895, 3.2356, 0.3273),)),
               2800.0, 23.67, 1.21, source=f"{MIT}, openMotor's preset",
               notes="An ammonium perchlorate composite whose exact recipe is not published.",
               aliases=("cl",)),
]}


def get_propellant(name: str) -> Propellant:
    k = name.strip().lower().replace("_", "-").replace(" ", "-")
    for p in PROPELLANTS.values():
        if k == p.key or k in p.aliases:
            return p
    raise KeyError(f"unknown propellant '{name}' (known: {', '.join(PROPELLANTS)}); or give its density, burn rate "
                   "a and n, and T, M and k")
