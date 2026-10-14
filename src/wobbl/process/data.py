"""Process data for the manufacturing studies, each value with where it comes from.

Units here are the ones on datasheets and in papers (C, J/kg, kg/m3, W/(m K),
Pa, Pa s). The studies convert them where they need to.

Some of these numbers are well known (the latent heat of pure aluminum), some
come from one measured data set (the warp of printed ABS), and some carry a
real spread between sources (the latent heat of A356). The spread is part of
the data, so a study can say how much its answer moves with it.
"""
from __future__ import annotations

import difflib
import math
from dataclasses import dataclass

from ..materials import get_material


def _find(table: dict, name: str, what: str, hint: str):
    key = name.strip().lower().replace(" ", "-").replace("_", "-")
    for k, v in table.items():
        if key == k or key in getattr(v, "aliases", ()):
            return v
    close = difflib.get_close_matches(key, list(table), n=2, cutoff=0.5)
    raise KeyError(f"unknown {what} '{name}'" + (f". Did you mean {', '.join(close)}?" if close else "") +
                   f" Known: {', '.join(table)}. {hint}")


# --- 3D printing (FDM) ----------------------------------------------------------
# Warp model. A layer locks in part of its shrinkage from the glass transition to the surroundings, and the locked
# in shrinkage grows with height above the bed, from nothing at the bed (the first layers stay warm and held) to the
# full thermal shrinkage alpha (Tg - T_env) at RAMP_HEIGHT and above. While the ramp is shorter than the part, the
# part curls to a constant curvature alpha (Tg - T_env) / RAMP_HEIGHT, which is what the measurements below show:
# the warp of ABS plates 1.5 to 5.5 mm tall grows with length squared and hardly at all with height.
#
# RAMP_HEIGHT is fitted to the 48 ABSplus plates of Armillotta, Bellotti and Cavallaro, "Warpage of FDM parts:
# experimental tests and analytic model", Robotics and Computer-Integrated Manufacturing 50 (2018) 140-152, Table 5
# (Stratasys Dimension Elite, 75 C chamber, 270 C nozzle, solid fill), after their own correction of the flatness
# error for bead grooves and fixed disturbances (0.1 mm + 0.7 x layer height). The fit has R2 0.94, a median error
# of 9% and a largest error of 39%. It is the only calibration: other plastics use the same ramp with their own
# alpha and Tg, which is an extrapolation and lowers the confidence.
RAMP_HEIGHT = 11.5            # mm
FLATNESS_OFFSET = 0.1         # mm, Armillotta et al. c0
FLATNESS_PER_LAYER = 0.7      # of the layer height, Armillotta et al. c1
ARMILLOTTA_TABLE5 = [         # length, height, layer height (mm), flatness error of two parts (mm); width 20 mm
    (20, 1.5, 0.178, 0.21, 0.16), (20, 1.5, 0.254, 0.20, 0.24), (20, 3.5, 0.178, 0.23, 0.27),
    (20, 3.5, 0.254, 0.27, 0.26), (20, 5.5, 0.178, 0.17, 0.19), (20, 5.5, 0.254, 0.24, 0.22),
    (60, 1.5, 0.178, 0.27, 0.30), (60, 1.5, 0.254, 0.36, 0.39), (60, 3.5, 0.178, 0.34, 0.35),
    (60, 3.5, 0.254, 0.46, 0.43), (60, 5.5, 0.178, 0.28, 0.26), (60, 5.5, 0.254, 0.40, 0.39),
    (100, 1.5, 0.178, 0.56, 0.60), (100, 1.5, 0.254, 0.61, 0.58), (100, 3.5, 0.178, 0.55, 0.52),
    (100, 3.5, 0.254, 0.69, 0.65), (100, 5.5, 0.178, 0.50, 0.47), (100, 5.5, 0.254, 0.64, 0.71),
    (140, 1.5, 0.178, 0.91, 0.94), (140, 1.5, 0.254, 0.93, 0.92), (140, 3.5, 0.178, 0.82, 0.85),
    (140, 3.5, 0.254, 1.02, 1.07), (140, 5.5, 0.178, 0.71, 0.71), (140, 5.5, 0.254, 0.97, 0.98),
]
ARMILLOTTA_CHAMBER = 75.0     # C
ABSPLUS_TG = 108.0            # C, Stratasys ABSplus-P430 spec sheet
ARMILLOTTA_WIDTH = 20.0       # mm


def fit_ramp_height(cte: float = 90e-6) -> tuple[float, dict]:
    """Refit RAMP_HEIGHT to Armillotta et al. Table 5: the least squares curvature of a spherical cap over each
    plate's diagonal, after their flatness correction, and the full shrinkage over that curvature."""
    import numpy as np

    rows = np.array(ARMILLOTTA_TABLE5, float)
    length, layer, meas = rows[:, 0], rows[:, 2], rows[:, 3:5].mean(axis=1)
    reach = (length ** 2 + ARMILLOTTA_WIDTH ** 2) / 8.0
    y = meas - FLATNESS_OFFSET - FLATNESS_PER_LAYER * layer
    curv = float((reach * y).sum() / (reach * reach).sum())        # 1/mm
    pred = FLATNESS_OFFSET + FLATNESS_PER_LAYER * layer + curv * reach
    res = meas - pred
    rel = np.abs(res) / meas
    fit = {"curvature_per_mm": curv, "r2": float(1 - (res ** 2).sum() / ((meas - meas.mean()) ** 2).sum()),
           "median_error": float(np.median(rel)), "max_error": float(rel.max())}
    return cte * (ABSPLUS_TG - ARMILLOTTA_CHAMBER) / curv, fit


@dataclass(frozen=True)
class Filament:
    key: str
    name: str
    material: str             # key in wobbl.materials, for stiffness, Poisson's ratio and expansion
    tg: float                 # glass transition, C: below it the plastic is stiff and locks in shrinkage
    bed: float                # usual bed temperature, C
    calibrated: bool = False  # the warp ramp was fitted to measured parts of this plastic
    aliases: tuple[str, ...] = ()
    source: str = ""

    @property
    def props(self):
        return get_material(self.material)

    def shrinkage(self, t_env: float) -> float:
        """Full thermal shrinkage from the glass transition to the surroundings, the most a layer can lock in."""
        return self.props.cte * max(self.tg - t_env, 0.0)


FILAMENTS: dict[str, Filament] = {f.key: f for f in [
    Filament("abs", "ABS", "abs", 105.0, 100.0, calibrated=True,
             source="Tg typical of ABS (Stratasys ABSplus lists 108 C); warp ramp fitted to Armillotta et al. 2018"),
    Filament("asa", "ASA", "asa", 100.0, 100.0, source="Tg typical of ASA"),
    Filament("petg", "PETG", "petg", 80.0, 80.0, aliases=("pet-g", "pet"),
             source="Tg typical of PETG (80 to 85 C by brand); bed from the Prusament PETG datasheet"),
    Filament("pla", "PLA", "pla", 60.0, 60.0, aliases=("pla+",), source="Tg typical of PLA (55 to 60 C)"),
    Filament("pc", "Polycarbonate", "pc", 147.0, 110.0, aliases=("polycarbonate",),
             source="Tg typical of polycarbonate"),
]}


def get_filament(name: str) -> Filament:
    return _find(FILAMENTS, name, "printing plastic", "Warp studies cover amorphous plastics; nylon and other "
                 "crystalline plastics shrink as they crystallize, which this model does not capture.")


# --- casting --------------------------------------------------------------------
@dataclass(frozen=True)
class Alloy:
    key: str
    name: str
    liquidus: float           # C
    solidus: float            # C, where the last liquid freezes
    latent: float             # J/kg
    density: float            # kg/m3
    conductivity: float       # W/(m K)
    specific_heat: float      # J/(kg K)
    shrinkage: float          # volume fraction lost on freezing that feeding has to make up (beta prime)
    pour: float               # usual pouring temperature, C
    eutectic: float = 0.0     # C; 0 for a pure metal or a linear release
    partition: float = 0.0    # Scheil partition coefficient of the main solute
    melting: float = 0.0      # melting point of the pure base metal, C, for the Scheil curve
    latent_range: tuple[float, float] = (0.0, 0.0)   # J/kg, the spread between sources
    niyama: dict | None = None   # Carlson and Beckermann dimensionless Niyama parameters
    aliases: tuple[str, ...] = ()
    source: str = ""

    def solid_fraction(self, T):
        """Solid fraction at temperature T (array). A Scheil curve down to the eutectic, where the rest freezes over
        the next 10 C (a plateau a time stepper can follow), with the last 5% between there and the solidus. A pure
        metal releases its heat evenly between liquidus and solidus."""
        import numpy as np

        T = np.asarray(T, float)
        tl, ts = self.liquidus, self.solidus
        if not self.eutectic:
            return np.clip((tl - T) / max(tl - ts, 1e-9), 0.0, 1.0)
        k, tm, te = self.partition, self.melting, self.eutectic
        prim = 1.0 - np.power(np.clip((tm - np.minimum(T, tl)) / (tm - tl), 1.0, None), 1.0 / (k - 1.0))
        fe = 1.0 - ((tm - te) / (tm - tl)) ** (1.0 / (k - 1.0))    # solid fraction when the eutectic starts
        plateau = te - 10.0
        eut = fe + (0.95 - fe) * np.clip((te - T) / 10.0, 0.0, 1.0)
        tail = 0.95 + 0.05 * np.clip((plateau - T) / max(plateau - ts, 1e-9), 0.0, 1.0)
        fs = np.where(T >= te, prim, np.where(T >= plateau, eut, tail))
        return np.clip(np.where(T >= tl, 0.0, np.where(T <= ts, 1.0, fs)), 0.0, 1.0)


# Carlson and Beckermann, "Prediction of shrinkage pore volume fraction using a dimensionless Niyama criterion",
# Metallurgical and Materials Transactions A 40 (2009) 163-175, Tables II and III. The thermal gradient and cooling
# rate are taken 10% of the freezing range above the solidus.
A356_NIYAMA = {"viscosity": 1.58e-3, "beta": 0.0589, "freezing_range": 65.3, "pressure_drop": 1.01e5,
               "c_lambda": 40.9e-6,
               # pore volume percent = min(beta', -A log10(Ny*) + B + C Ny*^-D) over these ranges of Ny*
               "fits": [(1.43, 2.068, 3.160, 0.0, 0.0), (18.0, 0.0, 0.0, 4.024, 0.9786),
                        (math.inf, 0.0, 0.0, 7.771, 1.206)]}

ALLOYS: dict[str, Alloy] = {a.key: a for a in [
    Alloy("a356", "Aluminum A356 (Al-7Si-0.3Mg)", 616.0, 550.7, 403e3, 2670.0, 151.0, 963.0, 0.0557, 720.0,
          eutectic=577.0, partition=0.13, melting=660.3, latent_range=(321e3, 500e3), niyama=A356_NIYAMA,
          aliases=("a356.0", "a356-t6", "cast-aluminum", "cast-aluminium", "al-7si"),
          source="liquidus, freezing range, shrinkage and Niyama data from Carlson and Beckermann 2009; latent heat "
                 "403 kJ/kg as quoted for A356.0 by the ABM 2015 thermal analysis study (measured values run from "
                 "321 to 500 kJ/kg); density, conductivity and specific heat from MatWeb A356.0-T6; Scheil curve "
                 "with k = 0.13 and the Al-Si eutectic at 577 C"),
    Alloy("pure-al", "Pure aluminum", 660.3, 655.3, 397e3, 2700.0, 237.0, 897.0, 0.066, 700.0,
          latent_range=(397e3, 397e3), aliases=("aluminum", "aluminium", "al", "1100"),
          source="CRC Handbook (melting point 660.3 C, enthalpy of fusion 10.71 kJ/mol, conductivity 237 W/(m K)); "
                 "the latent heat is spread over 5 C below the melting point so a time stepper can follow it"),
]}


@dataclass(frozen=True)
class MoldMaterial:
    key: str
    name: str
    density: float            # kg/m3
    conductivity: float       # W/(m K)
    specific_heat: float      # J/(kg K)
    aliases: tuple[str, ...] = ()
    source: str = ""

    @property
    def effusivity(self) -> float:
        return math.sqrt(self.conductivity * self.density * self.specific_heat)


MOLDS: dict[str, MoldMaterial] = {m.key: m for m in [
    MoldMaterial("sand", "Silica sand, bonded", 1600.0, 0.5, 1050.0, aliases=("silica-sand", "green-sand"),
                 source="0.5 W/(m K) and 1600 kg/m3 as in common foundry cost models (laser flash measurements give "
                        "0.34 to 0.4 W/(m K)); 1050 J/(kg K) is the 25 to 1300 C mean measured for silica sand in "
                        "J. Japan Foundrymen's Society 49 (1977) 747"),
    MoldMaterial("steel", "Steel chill", 7850.0, 51.9, 486.0, aliases=("mild-steel", "iron"),
                 source="ASTM A36 typical properties (Wobbl materials)"),
    MoldMaterial("copper", "Copper chill", 8890.0, 388.0, 385.0, aliases=("cu",),
                 source="Copper Development Association C11000 (Wobbl materials)"),
]}


def get_alloy(name: str) -> Alloy:
    return _find(ALLOYS, name, "casting alloy", "")


def get_mold(name: str) -> MoldMaterial:
    return _find(MOLDS, name, "mold material", "")


# --- injection molding ----------------------------------------------------------
@dataclass(frozen=True)
class Resin:
    """Cross-WLF viscosity: eta0 = D1 exp(-A1 (T - T*) / (A2 + T - T*)), T* = D2 + D3 p,
    eta = eta0 / (1 + (eta0 gamma_dot / tau*)^(1 - n)). Temperatures in K inside the model."""
    key: str
    name: str
    n: float
    tau: float                # Pa
    d1: float                 # Pa s
    d2: float                 # K
    d3: float                 # K/Pa
    a1: float
    a2: float                 # K
    conductivity: float       # W/(m K), of the melt
    specific_heat: float      # J/(kg K), of the melt
    density: float            # kg/m3, solid, for the shot weight
    melt: float               # usual melt temperature, C
    mold: float               # usual mold temperature, C
    no_flow: float            # C, below which the melt stops flowing
    aliases: tuple[str, ...] = ()
    source: str = ""

    def eta0(self, T_c, p=0.0):
        import numpy as np

        T = np.asarray(T_c, float) + 273.15
        ts = self.d2 + self.d3 * p
        a2 = self.a2 + self.d3 * p
        return self.d1 * np.exp(-self.a1 * (T - ts) / (a2 + T - ts))

    @property
    def diffusivity(self) -> float:
        return self.conductivity / (self.density * self.specific_heat)


RESINS: dict[str, Resin] = {r.key: r for r in [
    Resin("gpps", "General purpose polystyrene (Styron 678)", 0.2903, 13678.0, 7.44e10, 373.15, 0.0, 25.971, 51.6,
          0.15, 2100.0, 1040.0, 230.0, 50.0, 100.0, aliases=("ps", "polystyrene", "styron"),
          source="Cross-WLF coefficients, melt conductivity and specific heat from Pedro et al., Fluids 5 (2020) 84, "
                 "Table 2 (Moldex3D database); no flow temperature taken as D2, the glass transition"),
    Resin("pc-oq1028", "Polycarbonate, high flow optical grade (Lexan OQ1028)", 0.0, 1.1e6, 6.0e11, 413.15, 2.0e-7,
          30.0, 51.6, 0.20, 1200.0, 1200.0, 310.0, 90.0, 140.0, aliases=("pc", "polycarbonate", "lexan"),
          source="Cross-WLF weighted fit of Hubmann et al., Polymers 15 (2023) 3147, Table 6; thermal properties "
                 "of polycarbonate from the Wobbl materials table (Covestro Makrolon 2405); this is a high flow "
                 "grade, so a general purpose polycarbonate needs several times the pressure"),
]}


def get_resin(name: str) -> Resin:
    return _find(RESINS, name, "resin", "Pass your grade's Cross-WLF coefficients as custom:n=..,tau=..,d1=..,"
                 "d2=..,a1=..,a2=..,melt=..,mold=..,no_flow=.. (from the supplier or a Moldflow database).")


def parse_resin(spec: str) -> Resin:
    """A library resin, or custom:n=0.25,tau=30000,d1=1e12,d2=373,a1=28,a2=51.6,melt=240,mold=60,no_flow=105
    with optional d3, k, cp and density (polystyrene values when left out)."""
    if not spec.lower().startswith("custom"):
        return get_resin(spec)
    from dataclasses import replace

    base = RESINS["gpps"]
    names = {"n": "n", "tau": "tau", "d1": "d1", "d2": "d2", "d3": "d3", "a1": "a1", "a2": "a2", "k": "conductivity",
             "cp": "specific_heat", "density": "density", "rho": "density", "melt": "melt", "mold": "mold",
             "no_flow": "no_flow", "noflow": "no_flow"}
    vals = {}
    for part in spec.split(":", 1)[1].split(","):
        if not part.strip():
            continue
        k, v = (x.strip().lower() for x in part.split("="))
        if k not in names:
            raise KeyError(f"unknown resin property '{k}' (use {', '.join(names)})")
        vals[names[k]] = float(v)
    missing = [k for k in ("n", "tau", "d1", "d2", "a1") if k not in vals]
    if missing:
        raise KeyError(f"a custom resin needs {', '.join(missing)} (its Cross-WLF coefficients)")
    return replace(base, key="custom", name="Custom resin", aliases=(), source="user supplied", **vals)
