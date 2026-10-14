"""Servos and motors: the torque each can give at each speed, and whether a duty stalls or cooks it.

A brushed DC motor at full voltage gives its stall torque at rest and nothing at its no load speed, in a straight
line between. A hobby servo is the same motor behind a gearbox, so its rating at the output shaft follows the
same line. Driving harder than the line allows is a stall. Heat in the windings grows with the square of the
torque, so the RMS torque over a repeating duty has to stay under what the motor can carry continuously.
"""
from __future__ import annotations

import difflib
import math
from dataclasses import dataclass, replace

from . import quantities as Q

# hobby servo datasheets give no continuous rating; a third of stall is the usual rule for small brushed motors
CONTINUOUS_SHARE = 1 / 3


@dataclass(frozen=True)
class Motor:
    key: str
    name: str
    stall_torque: float                 # N m at the output
    no_load_speed: float                # rad/s at the output
    continuous_torque: float = 0.0      # N m it can carry indefinitely (0 means CONTINUOUS_SHARE of stall)
    mass: float = 0.0                   # kg
    curve: str = "line"                 # line (DC motor), box (a torque limit and a speed limit, as in URDF)
    no_load_current: float = 0.0        # A, for the most efficient operating point
    stall_current: float = 0.0
    source: str = ""

    @property
    def rated(self) -> float:
        return self.continuous_torque or CONTINUOUS_SHARE * self.stall_torque

    def limits(self, w):
        """Lowest and highest torque the motor can give while turning at w rad/s (arrays work too)."""
        import numpy as np

        w = np.asarray(w, float)
        ts = self.stall_torque
        if self.curve == "box" or not math.isfinite(self.no_load_speed):
            over = np.abs(w) > self.no_load_speed * 1.0001
            hi = np.where(over & (w > 0), 0.0, ts)
            lo = np.where(over & (w < 0), 0.0, -ts)
            return lo, hi
        r = w / self.no_load_speed
        # full voltage either way, with the torque capped at stall when back EMF would add to it while braking
        return np.maximum(ts * (-1 - r), -ts), np.minimum(ts * (1 - r), ts)

    def limit(self, w: float) -> tuple[float, float]:
        """``limits`` for one speed, without numpy, for the simulation loop."""
        ts = self.stall_torque
        if self.curve == "box" or not math.isfinite(self.no_load_speed):
            over = abs(w) > self.no_load_speed * 1.0001
            return (0.0 if over and w < 0 else -ts), (0.0 if over and w > 0 else ts)
        r = w / self.no_load_speed
        return max(ts * (-1 - r), -ts), min(ts * (1 - r), ts)

    def usage(self, tau, w):
        """Share of the available torque each sample needs: above 1 means the motor cannot do it."""
        import numpy as np

        lo, hi = self.limits(w)
        tau = np.asarray(tau, float)
        avail = np.where(tau >= 0, hi, -lo)
        return np.where(np.abs(tau) < 1e-12, 0.0, np.abs(tau) / np.maximum(avail, 1e-12))

    def max_efficiency_torque(self) -> float:
        """Torque of best efficiency. For a DC motor it sits where the current is sqrt(I0 Is); without currents
        Wobbl takes the no load current as 5% of stall, typical of small brushed motors."""
        i0, i_s = self.no_load_current, self.stall_current
        share = i0 / i_s if i0 > 0 and i_s > 0 else 0.05
        return self.stall_torque * (math.sqrt(share) - share) / (1 - share)

    def geared(self, ratio: float, efficiency: float = 1.0) -> "Motor":
        """The same motor seen through a reduction of ``ratio`` with the given efficiency."""
        return replace(self, key=f"{self.key} x{ratio:g}", name=f"{self.name} through {ratio:g}:1",
                       stall_torque=self.stall_torque * ratio * efficiency,
                       no_load_speed=self.no_load_speed / ratio,
                       continuous_torque=self.rated * ratio * efficiency)

    def describe(self) -> str:
        from ..units import fmt

        speed = (f"{fmt(self.no_load_speed * 60 / (2 * math.pi))} rpm" if math.isfinite(self.no_load_speed)
                 else "no speed limit")
        return f"{self.name}: stall {Q.fmt_torque(self.stall_torque)}, no load {speed}"


def _servo(key, name, kgcm, s60, grams, volts, source="manufacturer datasheet"):
    return Motor(key, f"{name} at {volts:g} V", kgcm * Q.KGF_CM, math.radians(60) / s60, mass=grams / 1000,
                 source=source)


def _dxl(key, name, nm, rpm, grams, volts):
    return Motor(key, f"{name} at {volts:g} V", nm, rpm * 2 * math.pi / 60, mass=grams / 1000,
                 source="ROBOTIS e-Manual")


# ratings at the output shaft, at the voltage named
CATALOG = {m.key: m for m in [
    _servo("sg90", "TowerPro SG90", 1.8, 0.10, 9, 4.8),
    _servo("mg90s", "TowerPro MG90S", 2.2, 0.08, 13.4, 6),
    _servo("hs-311", "Hitec HS-311", 3.7, 0.15, 43, 6),
    _servo("hs-645mg", "Hitec HS-645MG", 9.6, 0.20, 55, 6),
    _servo("mg996r", "TowerPro MG996R", 11.0, 0.14, 55, 6),
    _servo("ds3218", "DSSERVO DS3218", 21.5, 0.14, 60, 6.8),
    _servo("hs-805bb", "Hitec HS-805BB", 24.7, 0.14, 152, 6),
    _dxl("ax-12a", "ROBOTIS Dynamixel AX-12A", 1.5, 59, 54.6, 12),
    _dxl("xl430", "ROBOTIS Dynamixel XL430-W250", 1.5, 61, 57.2, 12),
    _dxl("xm430", "ROBOTIS Dynamixel XM430-W350", 4.1, 46, 82, 12),
    Motor("fa-130", "Mabuchi FA-130RA at 3 V", 25.5 * Q.KGF_CM / 1000, 9100 * 2 * math.pi / 60, mass=0.018,
          no_load_current=0.15, stall_current=2.2, source="Mabuchi datasheet FA-130RA-2270"),
]}
_ALIASES = {"mg995": "mg996r", "sg-90": "sg90", "mg-996r": "mg996r", "ds-3218": "ds3218", "ax12": "ax-12a",
            "xl430-w250": "xl430", "xm430-w350": "xm430", "fa130": "fa-130", "130": "fa-130"}


def get_motor(key: str) -> Motor:
    k = key.strip().lower()
    k = _ALIASES.get(k, k)
    if k in CATALOG:
        return CATALOG[k]
    close = difflib.get_close_matches(k, list(CATALOG), n=3)
    hint = f" Did you mean {', '.join(close)}?" if close else ""
    raise ValueError(f"unknown servo or motor '{key}'.{hint} Known: {', '.join(CATALOG)}, or give stall_torque "
                     "and no_load_speed")


def motor_from(d) -> Motor:
    """A motor from a catalog name, or from a table with stall_torque and no_load_speed (plus optional
    continuous_torque, mass, curve, no_load_current, stall_current), or a catalog name with some values changed."""
    if isinstance(d, Motor):
        return d
    if isinstance(d, str):
        return get_motor(d)
    d = dict(d)
    base = get_motor(d.pop("servo", d.pop("motor", ""))) if ("servo" in d or "motor" in d) else None
    vals = {}
    if "stall_torque" in d:
        vals["stall_torque"] = Q.torque(d.pop("stall_torque"))
    if "no_load_speed" in d:
        s = d.pop("no_load_speed")
        vals["no_load_speed"] = math.inf if str(s).strip().lower() in ("inf", "none", "") else Q.speed(s)
    if "continuous_torque" in d:
        vals["continuous_torque"] = Q.torque(d.pop("continuous_torque"))
    if "mass" in d:
        vals["mass"] = Q.mass(d.pop("mass"))
    for k in ("no_load_current", "stall_current"):
        if k in d:
            vals[k] = Q.number(d.pop(k))
    if "curve" in d:
        vals["curve"] = str(d.pop("curve"))
    name = str(d.pop("name", ""))
    if d:
        raise ValueError(f"unknown motor settings {sorted(d)}")
    if base is not None:
        return replace(base, **vals, **({"name": name} if name else {}))
    if "stall_torque" not in vals:
        raise ValueError("a motor needs stall_torque (and no_load_speed), or a catalog name like mg996r")
    vals.setdefault("no_load_speed", math.inf)
    return Motor("custom", name or "custom motor", **vals)


def stronger(than: Motor, need_torque: float, need_speed: float = 0.0, need_rms: float = 0.0) -> Motor | None:
    """The lightest catalog servo whose line clears ``need_torque`` at ``need_speed`` with a quarter to spare and
    whose continuous rating carries ``need_rms``."""
    best = None
    for m in CATALOG.values():
        if m.key == than.key or m.no_load_current:      # skip the bare DC motors
            continue
        if float(m.limits(need_speed)[1]) >= 1.25 * need_torque and m.rated >= need_rms:
            if best is None or (m.mass, m.stall_torque) < (best.mass, best.stall_torque):
                best = m
    return best
