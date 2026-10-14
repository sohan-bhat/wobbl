"""Unit parsing and formatting.

Everything inside Wobbl uses one consistent system so solvers never see mixed units:

    length mm, force N, stress MPa (N/mm2), mass t (tonne), time s,
    energy mJ (N mm), power mW, temperature degC.

In that system thermal properties come out as: conductivity mW/(mm K), which is
numerically equal to W/(m K); film coefficient mW/(mm2 K) = 1e-3 W/(m2 K);
specific heat mJ/(t K) = 1e6 J/(kg K); density t/mm3 = 1e-12 kg/m3.

Circuit quantities stay in plain SI (V, A, ohm, F, H, s, Hz, W). Speeds come out in mm/s and angles in
degrees, and the motion domain turns lengths, masses and speeds into m, kg and m/s as it reads them.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

# factor to internal units for each dimension
_UNITS: dict[str, dict[str, float]] = {
    "length": {
        "mm": 1.0, "millimeter": 1.0, "millimeters": 1.0, "millimetre": 1.0, "millimetres": 1.0,
        "cm": 10.0, "m": 1000.0, "meter": 1000.0, "meters": 1000.0, "metre": 1000.0, "metres": 1000.0,
        "um": 1e-3, "micron": 1e-3, "microns": 1e-3,
        "in": 25.4, "inch": 25.4, "inches": 25.4, '"': 25.4, "ft": 304.8, "foot": 304.8, "feet": 304.8,
        "thou": 0.0254, "mil": 0.0254,
    },
    "force": {
        "n": 1.0, "newton": 1.0, "newtons": 1.0, "kn": 1000.0, "mn": 1e-3,
        "lbf": 4.4482216152605, "lb": 4.4482216152605, "lbs": 4.4482216152605, "pound": 4.4482216152605,
        "pounds": 4.4482216152605, "kip": 4448.2216152605, "kips": 4448.2216152605,
        "kgf": 9.80665, "kg": 9.80665, "kgs": 9.80665, "g": 9.80665e-3, "gf": 9.80665e-3,
    },
    "stress": {
        "pa": 1e-6, "kpa": 1e-3, "mpa": 1.0, "gpa": 1000.0, "n/mm2": 1.0, "n/mm^2": 1.0, "n/m2": 1e-6,
        "psi": 0.00689475729, "ksi": 6.89475729, "msi": 6894.75729, "bar": 0.1, "mbar": 1e-4,
        "atm": 0.101325, "psf": 4.78802589e-5,
    },
    "mass": {"t": 1.0, "kg": 1e-3, "g": 1e-6, "lb": 0.45359237e-3, "lbs": 0.45359237e-3, "oz": 2.8349523e-5},
    "accel": {"g": 9806.65, "m/s2": 1000.0, "m/s^2": 1000.0, "mm/s2": 1.0, "mm/s^2": 1.0, "ft/s2": 304.8},
    "moment": {
        "n*mm": 1.0, "nmm": 1.0, "n.mm": 1.0, "n*m": 1000.0, "nm": 1000.0, "n.m": 1000.0, "n-m": 1000.0,
        "kn*m": 1e6, "knm": 1e6,
        "lbf*in": 112.984829, "lbf-in": 112.984829, "in*lb": 112.984829, "in-lb": 112.984829, "inlb": 112.984829,
        "in*lbf": 112.984829, "in-lbf": 112.984829,
        "lbf*ft": 1355.81795, "lbf-ft": 1355.81795, "ft*lb": 1355.81795, "ft-lb": 1355.81795, "ftlb": 1355.81795,
        "ft-lbf": 1355.81795, "oz*in": 7.0615518, "oz-in": 7.0615518,
    },
    "power": {"w": 1000.0, "mw": 1.0, "kw": 1e6, "watt": 1000.0, "watts": 1000.0},
    "time": {"s": 1.0, "sec": 1.0, "second": 1.0, "seconds": 1.0, "ms": 1e-3, "min": 60.0, "mins": 60.0,
             "minute": 60.0, "minutes": 60.0, "h": 3600.0, "hr": 3600.0, "hrs": 3600.0, "hour": 3600.0,
             "hours": 3600.0},
    "heatflux": {"w/m2": 1e-3, "w/m^2": 1e-3, "w/cm2": 10.0, "w/cm^2": 10.0, "w/mm2": 1000.0, "mw/mm2": 1.0},
    "film": {"w/m2k": 1e-3, "w/m^2k": 1e-3, "w/(m2k)": 1e-3, "w/m2/k": 1e-3, "w/(m^2k)": 1e-3, "mw/mm2k": 1.0},
    "conductivity": {"w/mk": 1.0, "w/(mk)": 1.0, "w/m/k": 1.0, "w/m-k": 1.0},
    "density": {"kg/m3": 1e-12, "kg/m^3": 1e-12, "g/cm3": 1e-9, "g/cm^3": 1e-9, "t/mm3": 1.0},
    "specific_heat": {"j/kgk": 1e6, "j/(kgk)": 1e6, "j/kg/k": 1e6, "j/kg-k": 1e6},
    "volumetric_power": {"w/m3": 1e-6, "w/m^3": 1e-6, "w/cm3": 1.0, "mw/mm3": 1.0, "w/mm3": 1000.0},
    # circuits: SI
    "voltage": {"v": 1.0, "mv": 1e-3, "kv": 1e3, "uv": 1e-6},
    "current": {"a": 1.0, "ma": 1e-3, "ua": 1e-6, "na": 1e-9},
    "resistance": {"ohm": 1.0, "ohms": 1.0, "r": 1.0, "kohm": 1e3, "kohms": 1e3, "k": 1e3, "meg": 1e6,
                   "megohm": 1e6, "milliohm": 1e-3},
    "capacitance": {"f": 1.0, "mf": 1e-3, "uf": 1e-6, "nf": 1e-9, "pf": 1e-12},
    "inductance": {"h": 1.0, "mh": 1e-3, "uh": 1e-6, "nh": 1e-9},
    "frequency": {"hz": 1.0, "khz": 1e3, "mhz": 1e6, "ghz": 1e9},
    "si_power": {"w": 1.0, "mw": 1e-3, "kw": 1e3},
    # an elastic support: spring stiffness per unit area of the face (a Winkler foundation modulus), N/mm3
    "foundation": {"n/mm3": 1.0, "n/mm^3": 1.0, "mpa/mm": 1.0, "n/cm3": 1e-3, "kn/m3": 1e-6, "mn/m3": 1e-3,
                   "pci": 2.71447e-4, "lbf/in3": 2.71447e-4},
    # motion
    "speed": {"mm/s": 1.0, "m/s": 1000.0, "mps": 1000.0, "km/h": 1000 / 3.6, "kph": 1000 / 3.6, "kmh": 1000 / 3.6,
              "mph": 447.04, "ft/s": 304.8, "fps": 304.8, "kn": 1852 / 3.6, "kt": 1852 / 3.6, "knot": 1852 / 3.6,
              "knots": 1852 / 3.6},
    "angle": {"deg": 1.0, "degree": 1.0, "degrees": 1.0, "rad": 57.29577951308232, "radian": 57.29577951308232,
              "radians": 57.29577951308232},
}

# display labels for the internal unit of each dimension
INTERNAL = {
    "length": "mm", "force": "N", "stress": "MPa", "mass": "t", "accel": "mm/s2", "moment": "N mm",
    "power": "mW", "time": "s", "heatflux": "mW/mm2", "film": "mW/(mm2 K)", "conductivity": "W/(m K)",
    "density": "t/mm3", "specific_heat": "mJ/(t K)", "volumetric_power": "mW/mm3", "voltage": "V",
    "current": "A", "resistance": "ohm", "capacitance": "F", "inductance": "H", "frequency": "Hz",
    "si_power": "W", "foundation": "N/mm3", "temperature": "C", "speed": "mm/s", "angle": "deg",
}

IMPERIAL_UNITS = {"in", "inch", "inches", '"', "ft", "foot", "feet", "lbf", "lb", "lbs", "pound", "pounds",
                  "kip", "kips", "psi", "ksi", "msi", "lbf*in", "in*lb", "in-lb", "lbf-in", "ft-lb", "lbf*ft",
                  "f", "degf", "thou", "mil", "oz", "mph", "ft/s", "fps"}

_NUM = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
_QTY = re.compile(rf"^\s*({_NUM})\s*([^\s].*?)?\s*$")
_SI_PREFIX = re.compile(r"^(\d+(?:\.\d+)?)([kKmMuUnNpP]|meg)$")


class UnitError(ValueError):
    """Raised when a quantity cannot be understood. The message says what was expected."""


@dataclass(frozen=True)
class Quantity:
    value: float          # in Wobbl's internal units
    kind: str
    text: str             # what the user wrote
    imperial: bool = False


def _norm(unit: str) -> str:
    u = unit.strip().replace("\u03a9", "ohm").replace("\u2126", "ohm").replace("\u03c9", "ohm")
    u = u.lower().replace(" ", "").replace("\u00b7", "*").replace("\u00b2", "2").replace("\u00b3", "3")
    return u.replace("\u00b5", "u").replace("\u03bc", "u").replace("\u00b0", "deg")


def parse_quantity(text: str | float | int, kind: str, default_unit: str | None = None) -> Quantity:
    """Parse "200 N", "45lbf", "3.5 GPa", "1/4 in" into internal units.

    A bare number is accepted when ``default_unit`` is given (or for temperatures, as degC).
    """
    if isinstance(text, (int, float)):
        if default_unit is None and kind != "temperature":
            raise UnitError(f"{text} has no unit; give one, for example '{text} {INTERNAL.get(kind, '')}'")
        text = f"{text} {default_unit or 'C'}"
    raw = str(text).strip()
    # fractions like "1/4 in"
    frac = re.match(r"^\s*(\d+)\s*/\s*(\d+)\s*(.*)$", raw)
    if frac and not re.match(r"^\s*\d+\s*/\s*\d+\s*$", raw):
        value = float(frac.group(1)) / float(frac.group(2))
        unit = frac.group(3)
    else:
        m = _QTY.match(raw)
        if not m:
            raise UnitError(f"could not read a number from '{raw}'")
        value = float(m.group(1))
        unit = m.group(2) or ""
    if kind == "temperature":
        u = _norm(unit) or "c"
        imperial = u in ("f", "degf")
        if u in ("c", "degc", "celsius"):
            return Quantity(value, kind, raw)
        if u in ("k", "kelvin"):
            return Quantity(value - 273.15, kind, raw)
        if imperial or u == "fahrenheit":
            return Quantity((value - 32.0) * 5.0 / 9.0, kind, raw, True)
        raise UnitError(f"unknown temperature unit '{unit}' in '{raw}' (use C, K or F)")
    if kind == "temperature_delta":
        u = _norm(unit) or "k"
        if u in ("k", "c", "degc"):
            return Quantity(value, kind, raw)
        if u in ("f", "degf"):
            return Quantity(value * 5.0 / 9.0, kind, raw, True)
        raise UnitError(f"unknown temperature difference unit '{unit}'")
    table = _UNITS.get(kind)
    if table is None:
        raise UnitError(f"unknown quantity kind '{kind}'")
    u = _norm(unit)
    if kind == "resistance" and unit.strip()[:1] in ("M", "m") and u in ("mohm", "mohms"):
        u = "megohm" if unit.strip()[0] == "M" else "milliohm"
    if not u:
        if default_unit is None:
            raise UnitError(f"'{raw}' has no unit; expected something like {', '.join(list(table)[:4])}")
        u = _norm(default_unit)
    if u not in table:
        u2 = u.rstrip("s") if u.endswith("s") and u.rstrip("s") in table else None
        if u2 is None:
            raise UnitError(f"unknown {kind} unit '{unit}' in '{raw}'. Known: {', '.join(sorted(table)[:12])}")
        u = u2
    return Quantity(value * table[u], kind, raw, u in IMPERIAL_UNITS)


def to_internal(text: str | float, kind: str, default_unit: str | None = None) -> float:
    return parse_quantity(text, kind, default_unit).value


def parse_spice_value(text: str) -> float:
    """Parse SPICE style numbers: 4k7, 10u, 2.2meg, 100n, 1.5."""
    t = str(text).strip().lower().replace("µ", "u").replace("μ", "u")
    t = re.sub(r"(ohms?|f|h|v|a|hz)$", "", t) or t
    m = re.match(r"^(\d+)([kmunp]|meg)(\d+)$", t)
    if m:  # 4k7 style
        return float(f"{m.group(1)}.{m.group(3)}") * _SPICE[m.group(2)]
    m = re.match(rf"^({_NUM})\s*(meg|[tgkmunpf])?$", t)
    if not m:
        raise UnitError(f"could not read component value '{text}'")
    return float(m.group(1)) * _SPICE.get(m.group(2) or "", 1.0)


_SPICE = {"": 1.0, "t": 1e12, "g": 1e9, "meg": 1e6, "k": 1e3, "m": 1e-3, "u": 1e-6, "n": 1e-9, "p": 1e-12,
          "f": 1e-15}


def fmt(value: float, digits: int = 3) -> str:
    """Engineering friendly number formatting: 3 significant digits, no scientific notation for normal sizes."""
    if value == 0 or value != value:
        return "0" if value == 0 else "nan"
    a = abs(value)
    if a >= 1e6 or a < 1e-3:
        return f"{value:.{digits - 1}e}"
    s = f"{value:.{digits}g}"
    if "e" in s:
        s = f"{value:.{max(0, digits - len(str(int(a))))}f}"
    return s


def fmt_stress(mpa: float, imperial: bool = False) -> str:
    if imperial:
        psi = mpa / 0.00689475729
        return f"{fmt(psi / 1000)} ksi ({fmt(mpa)} MPa)" if psi >= 10000 else f"{fmt(psi)} psi ({fmt(mpa)} MPa)"
    return f"{fmt(mpa)} MPa"


def fmt_length(mm: float, imperial: bool = False) -> str:
    if imperial:
        return f"{fmt(mm / 25.4)} in ({fmt(mm)} mm)"
    return f"{fmt(mm)} mm"


def fmt_force(n: float, imperial: bool = False) -> str:
    if imperial:
        return f"{fmt(n / 4.4482216152605)} lbf ({fmt(n)} N)"
    return f"{fmt(n)} N"
