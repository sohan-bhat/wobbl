"""Fluids and flow quantities.

Flow studies work in plain SI (m, s, kg, Pa, W), unlike the strength and heat domains, because every correlation
and OpenFOAM case is written that way. Properties are at 1 atm and follow the temperature the user gives:

  air       ideal gas density, Sutherland's law for viscosity and conductivity (White, Viscous Fluid Flow, 3rd
            ed., Table 1-2), cp 1006 J/(kg K)
  water     density from Kell (1975), viscosity from the Vogel equation fitted by Al-Shemmeri, conductivity from
            Ramires et al. (1995), cp 4182 J/(kg K)
  seawater  ITTC 7.5-02-01-03 (2011) values at 15 C and 3.5% salinity, the standard for hull resistance
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


FLUIDS = {"air": _air, "water": _water, "seawater": _seawater, "sea": _seawater, "fresh water": _water}


def get_fluid(spec: str = "air", T: float = 20.0) -> Fluid:
    """'air', 'water', 'seawater', or 'custom:rho=1000,mu=1e-3[,k=0.6,cp=4182]' in SI units."""
    s = (spec or "air").strip().lower()
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
        raise ValueError(f"unknown fluid '{spec}'; use air, water, seawater or custom:rho=..,mu=..")
    return FLUIDS[s](T)


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
