"""Read mechanism quantities into SI (m, kg, s, rad, N, N m), the units MuJoCo works in.

A bare number takes the unit people use for small mechanisms, which is mm for lengths, g for masses, degrees for
angles, N m for torques, rpm for speeds and seconds for times. Hobby servo ratings read as written, so
"11 kg*cm" and "0.14 s/60deg" both work.
"""
from __future__ import annotations

import math
import re

from ..units import UnitError, parse_quantity

KGF_CM = 0.0980665          # N m in one kgf cm, the hobby servo torque unit
_TORQUE = {"kg*cm": KGF_CM, "kgf*cm": KGF_CM, "kgcm": KGF_CM, "kg.cm": KGF_CM, "kg-cm": KGF_CM, "kgfcm": KGF_CM,
           "g*cm": KGF_CM / 1000, "gf*cm": KGF_CM / 1000, "gcm": KGF_CM / 1000, "mn*m": 1e-3, "mnm": 1e-3}
_ANGLE = {"deg": math.pi / 180, "degree": math.pi / 180, "degrees": math.pi / 180, "rad": 1.0, "radian": 1.0,
          "radians": 1.0, "rev": 2 * math.pi, "turn": 2 * math.pi, "turns": 2 * math.pi}
_SPEED = {"rpm": 2 * math.pi / 60, "rad/s": 1.0, "deg/s": math.pi / 180, "rps": 2 * math.pi, "hz": 2 * math.pi}
_VELOCITY = {"m/s": 1.0, "mm/s": 1e-3, "cm/s": 1e-2, "km/h": 1 / 3.6, "kph": 1 / 3.6, "mph": 0.44704,
             "ft/s": 0.3048, "in/s": 0.0254}
_NUM = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"


def _split(v, default: str) -> tuple[float, str]:
    if isinstance(v, (int, float)):
        return float(v), default
    s = str(v).strip().replace("°", "deg").replace("·", "*")
    m = re.match(rf"^({_NUM})\s*(.*)$", s)
    if not m:
        raise UnitError(f"could not read a number from '{v}'")
    unit = m.group(2).strip().lower().replace(" ", "") or default
    return float(m.group(1)), unit


def length(v, default: str = "mm") -> float:
    return parse_quantity(v, "length", default).value / 1000


def mass(v, default: str = "g") -> float:
    return parse_quantity(v, "mass", default).value * 1000


def force(v, default: str = "N") -> float:
    return parse_quantity(v, "force", default).value


def time(v, default: str = "s") -> float:
    return parse_quantity(v, "time", default).value


def accel(v, default: str = "m/s2") -> float:
    return parse_quantity(v, "accel", default).value / 1000


def torque(v, default: str = "N*m") -> float:
    x, unit = _split(v, default)
    u = unit.replace("-", "*").replace(".", "*") if unit not in _TORQUE else unit
    if u in _TORQUE:
        return x * _TORQUE[u]
    return parse_quantity(f"{x} {unit}", "moment").value / 1000


def angle(v, default: str = "deg") -> float:
    x, unit = _split(v, default)
    if unit not in _ANGLE:
        raise UnitError(f"unknown angle unit '{unit}' in '{v}' (use deg or rad)")
    return x * _ANGLE[unit]


def speed(v, default: str = "rpm") -> float:
    """Rotation speed in rad/s. Hobby servos quote the time to turn 60 degrees, as in '0.14 s/60deg'."""
    x, unit = _split(v, default)
    m = re.match(rf"^(?:s|sec)/({_NUM})(?:deg)?$", unit)
    if m:
        return math.radians(float(m.group(1))) / x
    if unit not in _SPEED:
        raise UnitError(f"unknown speed unit '{unit}' in '{v}' (use rpm, deg/s, rad/s or s/60deg)")
    return x * _SPEED[unit]


def velocity(v, default: str = "m/s") -> float:
    x, unit = _split(v, default)
    if unit not in _VELOCITY:
        raise UnitError(f"unknown velocity unit '{unit}' in '{v}' (use m/s, mm/s or km/h)")
    return x * _VELOCITY[unit]


def number(v) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        raise UnitError(f"expected a plain number, got '{v}'") from None


def listed(v, read) -> list[float]:
    """A comma list such as '120,120,60' or '120 mm, 95 mm', or a real list, read item by item."""
    items = v if isinstance(v, (list, tuple)) else [s for s in str(v).split(",") if s.strip()]
    return [read(x) for x in items]


def fmt_torque(nm: float) -> str:
    """N m, with the hobby servo kg cm alongside for torques in the range servos are rated in."""
    from ..units import fmt

    return f"{fmt(nm)} N m ({fmt(nm / KGF_CM)} kg cm)" if abs(nm) < 4 else f"{fmt(nm)} N m"
