"""Fluids and flow quantities.

Flow studies work in plain SI (m, s, kg, Pa, W), unlike the strength and heat domains, because every correlation
and OpenFOAM case is written that way. Properties are at 1 atm and follow the temperature the user gives:

  air       ideal gas density, Sutherland's law for viscosity and conductivity (White, Viscous Fluid Flow, 3rd
            ed., Table 1-2), cp 1006 J/(kg K)
  water     density from Kell (1975), viscosity from the Vogel equation fitted by Al-Shemmeri, conductivity from
            Ramires et al. (1995), cp 4182 J/(kg K)
  seawater  ITTC 7.5-02-01-03 (2011) values at 15 C and 3.5% salinity, the standard for hull resistance
  engine-oil, ethylene-glycol, glycerin
            Incropera and DeWitt, Fundamentals of Heat and Mass Transfer, 6th ed., Table A.5, interpolated with
            the logarithm of viscosity linear in 1/T, which follows Andrade's law between the tabulated points
  hydraulic oil ISO VG 22 to 100
            a mineral oil with viscosity index 100: the grade's viscosity at 40 C by definition (ISO 3448), the
            viscosity at 100 C that VI 100 gives (ASTM D2270), Walther's equation between them (ASTM D341),
            density 870 kg/m3 at 15 C falling 0.064% per kelvin, conductivity 0.13 W/(m K), cp 1880 J/(kg K)
            at 40 C rising 3.7 J/(kg K) per kelvin
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass

from ..units import UnitError, parse_quantity

R_AIR = 287.05          # J/(kg K)
P_ATM = 101325.0        # Pa
G = 9.80665             # m/s2


@dataclass(frozen=True)
class Fluid:
    key: str
    name: str
    rho: float          # kg/m3
    mu: float           # Pa s
    k: float            # W/(m K)
    cp: float           # J/(kg K)
    T: float            # C, the temperature these hold at
    source: str = ""

    @property
    def nu(self) -> float:
        return self.mu / self.rho

    @property
    def Pr(self) -> float:
        return self.mu * self.cp / self.k

    def summary(self) -> str:
        return (f"{self.name} at {self.T:g} C, density {self.rho:.4g} kg/m3, viscosity {self.mu:.4g} Pa s, "
                f"conductivity {self.k:.3g} W/(m K), Prandtl {self.Pr:.3g}")


def _air(T: float) -> Fluid:
    tk = T + 273.15
    mu = 1.716e-5 * (tk / 273.15) ** 1.5 * (273.15 + 110.4) / (tk + 110.4)
    k = 0.0241 * (tk / 273.15) ** 1.5 * (273.15 + 194.0) / (tk + 194.0)
    return Fluid("air", "Air", P_ATM / (R_AIR * tk), mu, k, 1006.0, T, "ideal gas, Sutherland's law (White)")


def _water(T: float) -> Fluid:
    if not 0.0 <= T <= 100.0:
        raise ValueError(f"water properties cover 0 to 100 C, got {T:g} C")
    rho = (999.83952 + 16.945176 * T - 7.9870401e-3 * T ** 2 - 46.170461e-6 * T ** 3 + 105.56302e-9 * T ** 4
           - 280.54253e-12 * T ** 5) / (1 + 16.87985e-3 * T)
    mu = 2.414e-5 * 10 ** (247.8 / (T + 273.15 - 140.0))
    tr = (T + 273.15) / 298.15
    k = 0.6065 * (-1.48445 + 4.12292 * tr - 1.63866 * tr ** 2)
    return Fluid("water", "Water", rho, mu, k, 4182.0, T, "Kell density, Vogel viscosity, Ramires conductivity")


def _seawater(T: float) -> Fluid:
    return Fluid("seawater", "Seawater", 1025.9, 1025.9 * 1.1883e-6, 0.596, 3993.0, 15.0, "ITTC 2011 at 15 C")


# Incropera Table A.5: T (K), density, cp, viscosity (Pa s), conductivity
_TABLES = {
    "engine-oil": ("Engine oil (unused)", [
        (273, 899.1, 1796, 3.85, .147), (280, 895.3, 1827, 2.17, .144), (290, 890.0, 1868, .999, .145),
        (300, 884.1, 1909, .486, .145), (310, 877.9, 1951, .253, .145), (320, 871.8, 1993, .141, .143),
        (330, 865.8, 2035, .0836, .141), (340, 859.9, 2076, .0531, .139), (350, 853.9, 2118, .0356, .138),
        (360, 847.8, 2161, .0252, .138), (370, 841.8, 2206, .0186, .137), (380, 836.0, 2250, .0141, .136),
        (390, 830.6, 2294, .0110, .135), (400, 825.1, 2337, .00874, .134), (410, 818.9, 2381, .00698, .133),
        (420, 812.1, 2427, .00564, .133), (430, 806.5, 2471, .00470, .132)]),
    "ethylene-glycol": ("Ethylene glycol", [
        (273, 1130.8, 2294, .0651, .242), (280, 1125.8, 2323, .0420, .244), (290, 1118.8, 2368, .0247, .248),
        (300, 1114.4, 2415, .0157, .252), (310, 1103.7, 2460, .0107, .255), (320, 1096.2, 2505, .00757, .258),
        (330, 1089.5, 2549, .00561, .260), (340, 1083.8, 2592, .00431, .261), (350, 1079.0, 2637, .00342, .261),
        (360, 1074.0, 2682, .00278, .261), (370, 1066.7, 2728, .00228, .262), (373, 1058.5, 2742, .00215, .263)]),
    "glycerin": ("Glycerin", [
        (273, 1276.0, 2261, 10.6, .282), (280, 1271.9, 2298, 5.34, .284), (290, 1265.8, 2367, 1.85, .286),
        (300, 1259.9, 2427, .799, .286), (310, 1253.9, 2490, .352, .286), (320, 1247.2, 2564, .210, .287)]),
}


def _tabulated(key: str):
    import numpy as np

    name, rows = _TABLES[key]
    t = np.array([r[0] for r in rows], float)

    def make(T: float) -> Fluid:
        tk = T + 273.15
        if not t[0] - 0.5 <= tk <= t[-1] + 0.5:
            raise ValueError(f"{name.lower()} properties cover {t[0] - 273.15:.0f} to {t[-1] - 273.15:.0f} C, "
                             f"got {T:g} C")
        col = [np.array([r[i] for r in rows], float) for i in range(1, 5)]
        mu = float(np.exp(np.interp(-1 / tk, -1 / t, np.log(col[2]))))
        rho, cp, k = (float(np.interp(tk, t, c)) for c in (col[0], col[1], col[3]))
        return Fluid(key, name, rho, mu, k, cp, T, "Incropera Table A.5")
    return make


VI100_AT_100C = {22: 4.4, 32: 5.4, 46: 6.8, 68: 8.6, 100: 11.1}    # cSt, ASTM D2270 for VI 100


def _hydraulic(grade: int):
    def make(T: float) -> Fluid:
        if not -20 <= T <= 120:
            raise ValueError(f"hydraulic oil properties cover -20 to 120 C, got {T:g} C")
        # Walther: log10 log10 (nu + 0.7) = A - B log10 T, through the 40 C and 100 C points
        t1, t2 = 313.15, 373.15
        z1, z2 = (math.log10(math.log10(v + 0.7)) for v in (grade, VI100_AT_100C[grade]))
        b = (z1 - z2) / (math.log10(t2) - math.log10(t1))
        a = z1 + b * math.log10(t1)
        nu_cst = 10 ** (10 ** (a - b * math.log10(T + 273.15))) - 0.7
        rho = 870.0 * (1 - 6.4e-4 * (T - 15.0))
        return Fluid(f"iso-vg-{grade}", f"Hydraulic oil ISO VG {grade}", rho, nu_cst * 1e-6 * rho, 0.13,
                     1880.0 + 3.7 * (T - 40.0), T, "ISO 3448 grade, VI 100, Walther's equation")
    return make


FLUIDS = {"air": _air, "water": _water, "seawater": _seawater, "sea": _seawater, "fresh water": _water,
          "engine-oil": _tabulated("engine-oil"), "oil": _tabulated("engine-oil"),
          "ethylene-glycol": _tabulated("ethylene-glycol"), "glycol": _tabulated("ethylene-glycol"),
          "glycerin": _tabulated("glycerin"), "glycerol": _tabulated("glycerin"),
          **{f"iso-vg-{g}": _hydraulic(g) for g in VI100_AT_100C},
          **{f"hydraulic-oil-{g}": _hydraulic(g) for g in VI100_AT_100C}}


def get_fluid(spec: str = "air", T: float = 20.0) -> Fluid:
    """A fluid from the library by name (see NAMES) at T in C, or 'custom:rho=1000,mu=1e-3[,k=0.6,cp=4182]' in SI
    units."""
    s = re.sub(r"[\s_]+", "-", (spec or "air").strip().lower()).replace("fresh-water", "fresh water")
    if s.startswith("custom:"):
        vals = {}
        for part in s[7:].split(","):
            if "=" not in part:
                raise ValueError(f"custom fluid needs key=value pairs, got '{part}'")
            key, v = (x.strip() for x in part.split("=", 1))
            vals[key] = float(v)
        if "rho" not in vals or "mu" not in vals:
            raise ValueError("a custom fluid needs at least rho (kg/m3) and mu (Pa s)")
        return Fluid("custom", "Custom fluid", vals["rho"], vals["mu"], vals.get("k", 0.6), vals.get("cp", 4182.0),
                     T, "given by the user")
    if s not in FLUIDS:
        raise ValueError(f"unknown fluid '{spec}'; use {', '.join(NAMES)} or custom:rho=..,mu=..[,k=..,cp=..]")
    return FLUIDS[s](T)


NAMES = ("air", "water", "seawater", "engine-oil", "ethylene-glycol", "glycerin", "iso-vg-22", "iso-vg-32",
         "iso-vg-46", "iso-vg-68", "iso-vg-100")


# --- flow quantities, to SI --------------------------------------------------------------------------------------
_FLOW_UNITS = {
    "velocity": {"m/s": 1.0, "mps": 1.0, "km/h": 1 / 3.6, "kph": 1 / 3.6, "kmh": 1 / 3.6, "mph": 0.44704,
                 "kn": 0.514444, "kt": 0.514444, "kts": 0.514444, "knot": 0.514444, "knots": 0.514444,
                 "ft/s": 0.3048, "fps": 0.3048, "ft/min": 0.00508, "fpm": 0.00508, "lfm": 0.00508,
                 "cm/s": 0.01, "mm/s": 0.001},
    "flowrate": {"m3/s": 1.0, "m^3/s": 1.0, "m3/h": 1 / 3600, "m^3/h": 1 / 3600, "m3/hr": 1 / 3600,
                 "l/s": 1e-3, "l/min": 1e-3 / 60, "lpm": 1e-3 / 60, "l/h": 1e-3 / 3600, "ml/min": 1e-6 / 60,
                 "cfm": 0.3048 ** 3 / 60, "ft3/min": 0.3048 ** 3 / 60, "gpm": 3.785411784e-3 / 60,
                 "gal/min": 3.785411784e-3 / 60, "gph": 3.785411784e-3 / 3600},
    "pressure": {"pa": 1.0, "kpa": 1e3, "mpa": 1e6, "bar": 1e5, "mbar": 100.0, "psi": 6894.757, "atm": P_ATM,
                 "inh2o": 249.089, "inwc": 249.089, "inwg": 249.089, "mmh2o": 9.80665, "mmwc": 9.80665,
                 "mh2o": 9806.65, "mmhg": 133.322, "torr": 133.322, "hpa": 100.0},
    "rotation": {"rpm": 1 / 60, "rev/min": 1 / 60, "rps": 1.0, "hz": 1.0, "rad/s": 1 / (2 * math.pi)},
}


def qty(text: str | float, kind: str, default_unit: str | None = None) -> float:
    """A flow quantity in SI: velocity m/s, flowrate m3/s, pressure Pa, rotation rev/s, length m, force N, mass
    kg, power W, temperature C."""
    if kind in _FLOW_UNITS:
        raw = str(text).strip()
        m = re.match(r"^\s*([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)\s*(.*?)\s*$", raw)
        if not m:
            raise UnitError(f"could not read a number from '{raw}'")
        unit = m.group(2).lower().replace(" ", "").replace("³", "3") or (default_unit or "")
        table = _FLOW_UNITS[kind]
        if unit not in table:
            raise UnitError(f"unknown {kind} unit '{m.group(2) or '(none)'}' in '{raw}'. Known: "
                            f"{', '.join(list(table)[:10])}")
        return float(m.group(1)) * table[unit]
    q = parse_quantity(text, kind, default_unit)
    return {"length": 1e-3, "force": 1.0, "mass": 1e3, "power": 1e-3, "temperature": 1.0}.get(kind, 1.0) * q.value


def fmt_pressure(pa: float) -> str:
    from ..units import fmt
    if abs(pa) >= 1e5:
        return f"{fmt(pa / 1e5)} bar"
    if abs(pa) >= 1e3:
        return f"{fmt(pa / 1e3)} kPa"
    return f"{fmt(pa)} Pa"


def fmt_flow(q: float) -> str:
    from ..units import fmt
    lpm = q * 6e4
    return f"{fmt(lpm)} L/min" if lpm < 1000 else f"{fmt(q * 3600)} m3/h"
