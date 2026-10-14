"""Template values from --set flags and MCP calls, read with their units into SI."""
from __future__ import annotations

import math
import re

from ..units import UnitError, parse_quantity, parse_spice_value

_SI_FACTOR = {"length": 1e-3, "power": 1e-3}      # parse_quantity returns mm and mW for these
_EXTRA = {
    "flux": {"t": 1.0, "mt": 1e-3, "g": 1e-4, "gauss": 1e-4, "kg": 0.1},
    "field": {"v/m": 1.0, "kv/m": 1e3, "v/mm": 1e3, "kv/mm": 1e6, "mv/m": 1e6, "kv/cm": 1e5},
    "speed": {"m/s": 1.0},
    "angle": {"deg": 1.0, "degree": 1.0, "degrees": 1.0, "rad": 180 / math.pi},
    "fast_time": {"s": 1.0, "ms": 1e-3, "us": 1e-6, "ns": 1e-9, "ps": 1e-12},
}


class ParamError(ValueError):
    pass


def value(params: dict, key: str, default, kind: str = "number"):
    """params[key] (or default) in SI: meters, amperes, newtons, hertz, watts, tesla, volts per meter."""
    v = params.get(key, default)
    if v is None:
        raise ParamError(f"missing value for '{key}'")
    if isinstance(v, (int, float)):
        return float(v) * (_SI_FACTOR.get(kind, 1.0) if kind == "length" else 1.0)
    s = str(v).strip()
    if kind == "distance":                         # like length, but a bare number is in meters
        try:
            return parse_quantity(s, "length", default_unit="m").value * 1e-3
        except UnitError as e:
            raise ParamError(f"{key}: {e}") from None
    if kind == "number":
        try:
            return float(s)
        except ValueError:
            return parse_spice_value(s)
    if kind == "dbm":
        m = re.match(r"^\s*([-+]?\d+(?:\.\d+)?)\s*(dbm|dbw|mw|w)?\s*$", s.lower())
        if not m:
            raise ParamError(f"could not read power '{s}' for '{key}'. Use dBm, mW or W")
        x, u = float(m.group(1)), m.group(2) or "dbm"
        return {"dbm": x, "dbw": x + 30, "mw": 10 * math.log10(max(x, 1e-30)),
                "w": 10 * math.log10(max(x, 1e-30)) + 30}[u]
    if kind == "db":
        return float(re.sub(r"\s*(db[i]?|dbd)\s*$", "", s.lower()))
    if kind in _EXTRA:
        m = re.match(r"^\s*([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)\s*(\S*)\s*$", s)
        if not m:
            raise ParamError(f"could not read '{s}' for '{key}'")
        unit = m.group(2).lower() or next(iter(_EXTRA[kind]))
        if unit not in _EXTRA[kind]:
            raise ParamError(f"unknown unit '{m.group(2)}' for '{key}' (use {', '.join(_EXTRA[kind])})")
        return float(m.group(1)) * _EXTRA[kind][unit]
    try:
        q = parse_quantity(s, kind, default_unit={"length": "mm", "frequency": "hz", "current": "a",
                                                   "voltage": "v", "force": "n", "time": "s"}.get(kind))
    except UnitError as e:
        raise ParamError(f"{key}: {e}") from None
    return q.value * _SI_FACTOR.get(kind, 1.0)


def text(params: dict, key: str, default: str) -> str:
    return str(params.get(key, default)).strip().lower()


def flag(params: dict, key: str, default: bool = False) -> bool:
    v = params.get(key, default)
    if isinstance(v, bool):
        return v
    return str(v).strip().lower() in ("1", "true", "yes", "on")


def check_known(params: dict, known: set[str], template: str) -> None:
    bad = [k for k in params if k not in known]
    if bad:
        raise ParamError(f"unknown value{'s' if len(bad) > 1 else ''} {', '.join(bad)} for {template}. "
                         f"It takes {', '.join(sorted(known))}")
