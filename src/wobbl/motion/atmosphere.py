"""The 1976 US standard atmosphere and gravity that falls off with altitude, in SI units (m, kg, s, K, Pa).

The atmosphere is the standard's seven layers of constant lapse rate in geopotential altitude, up to 86 km
geometric. Each layer's base pressure comes from the one below by the hydrostatic equation, so the table
needs only the sea level pressure. Above 86 km the air is held at the top layer's temperature, which is
approximate, but by then it is a hundred thousandth of its sea level density.
"""
from __future__ import annotations

import math

G0 = 9.80665              # standard gravity, m/s2
R_EARTH = 6371008.8       # mean Earth radius for gravity, m
R0 = 6356766.0            # the standard's Earth radius for geopotential altitude, m
R_AIR = 8.31432 / 0.0289644   # gas constant of air, J/(kg K)
GAMMA = 1.4
P0 = 101325.0
TOP = 86000.0             # top of the standard's lower atmosphere, geometric m

# base geopotential altitude (m), base temperature (K), lapse rate (K/m)
_LAPSE = [(0.0, 288.15, -0.0065), (11000.0, 216.65, 0.0), (20000.0, 216.65, 0.001), (32000.0, 228.65, 0.0028),
          (47000.0, 270.65, 0.0), (51000.0, 270.65, -0.0028), (71000.0, 214.65, -0.002)]


def _layer_pressure(p0: float, t0: float, lapse: float, dh: float) -> float:
    if lapse == 0.0:
        return p0 * math.exp(-G0 * dh / (R_AIR * t0))
    return p0 * ((t0 + lapse * dh) / t0) ** (-G0 / (R_AIR * lapse))


def _layers() -> list[tuple[float, float, float, float]]:
    out, p = [], P0
    for k, (h, t, lapse) in enumerate(_LAPSE):
        out.append((h, t, lapse, p))
        if k + 1 < len(_LAPSE):
            p = _layer_pressure(p, t, lapse, _LAPSE[k + 1][0] - h)
    return out


LAYERS = _layers()
H_TOP = R0 * TOP / (R0 + TOP)     # 84852 m geopotential


def geopotential(h: float) -> float:
    return R0 * h / (R0 + h)


def air(h: float) -> tuple[float, float, float, float]:
    """Density (kg/m3), pressure (Pa), temperature (K) and speed of sound (m/s) at geometric altitude h (m above
    sea level). Below sea level the first layer carries on."""
    hg = geopotential(h)
    if hg > H_TOP:      # isothermal above 86 km
        base, t0, lapse, p0 = LAYERS[-1]
        t = t0 + lapse * (H_TOP - base)
        p = _layer_pressure(p0, t0, lapse, H_TOP - base) * math.exp(-G0 * (hg - H_TOP) / (R_AIR * t))
    else:
        base, t0, lapse, p0 = next(x for x in reversed(LAYERS) if hg >= x[0] or x is LAYERS[0])
        t = t0 + lapse * (hg - base)
        p = _layer_pressure(p0, t0, lapse, hg - base)
    return p / (R_AIR * t), p, t, math.sqrt(GAMMA * R_AIR * t)


def gravity(h: float) -> float:
    """Gravity at geometric altitude h (m above sea level), falling off with the square of the distance to the
    Earth's center."""
    return G0 * (R_EARTH / (R_EARTH + h)) ** 2
