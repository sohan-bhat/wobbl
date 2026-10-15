"""The one input a designer turns for a flight, and the flight flown again at each value of it.

A rocket is flown on whatever day it gets, so its input is the wind, unless the question asks about the rail's tilt
or the weight it carries. A dropped body's input is the height it falls from, a thrown one's its launch speed, or
its angle when the question asks how far it goes. The values are round numbers around the one the question gave,
which is always among them, so the scene opens on exactly the run the report describes.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field, replace

import numpy as np

from .checks import _speed, points, rocket_checks
from .flight import VX, VZ, X, Z, Flight, fly
from .spec import MotionStudy

MPH, FT = 0.44704, 0.3048


@dataclass
class Input:
    """What the slider turns: a study attribute shown in its own unit, with the asked value among the values."""
    key: str                 # wind, height, speed, angle, tilt or mass
    label: str               # what the slider says, as "wind" or "drop height"
    unit: str                # the unit it is shown in
    per: float               # SI units per shown unit
    values: np.ndarray       # in the shown unit, rising
    design: float            # the asked value, in the shown unit

    def study(self, st: MotionStudy, v: float) -> MotionStudy:
        """The study with this input set to v (in the shown unit)."""
        si = v * self.per
        if self.key == "wind":
            return replace(st, wind=si * (-1.0 if st.wind < 0 else 1.0))
        if self.key == "tilt":
            return replace(st, angle=90.0 - si)
        return replace(st, **{self.key: si})


@dataclass
class Point:
    """One flight of the sweep: its path, its events and the numbers a designer reads off it."""
    value: float
    flight: Flight
    t: np.ndarray
    x: np.ndarray
    z: np.ndarray
    speed: np.ndarray
    vx: np.ndarray
    vz: np.ndarray
    phase: np.ndarray
    events: dict = field(default_factory=dict)       # name to (t, x, z, speed)
    numbers: dict = field(default_factory=dict)


def _words(text: str) -> set[str]:
    return set(re.findall(r"[a-z]+", (text or "").lower()))


def grid(lo: float, hi: float, design: float, n: int = 21, log: bool = False) -> np.ndarray:
    """About n round values from lo to hi with design among them, evenly spaced or, for a range over many
    decades, evenly spaced in ratio."""
    if log and lo > 0:
        v = np.geomspace(lo, hi, n)
        mag = 10 ** np.floor(np.log10(v))
        v = np.round(v / mag * 2) / 2 * mag              # two figures, in halves
    else:
        raw = (hi - lo) / max(n - 1, 1)
        p = 10 ** np.floor(np.log10(max(raw, 1e-12)))
        step = min((m * p for m in (1, 1.25, 2, 2.5, 5, 10)), key=lambda s: abs(s - raw))
        v = np.arange(np.floor(lo / step) * step, hi + step * 0.5, step)
    v = np.unique(np.round(np.r_[v, design], 9))
    return v[(v >= lo - 1e-9) | np.isclose(v, design)]


def choose(st: MotionStudy) -> Input:
    """The input the question points at, swept over a range a designer would try."""
    w = _words(st.question)
    q = (st.question or "").lower()
    if st.rocket:
        if w & {"tilt", "tilted", "angle", "lean", "leaning"} and "wind" not in w:
            d = 90.0 - st.angle
            return Input("tilt", "rail tilt", "deg", 1.0, grid(0, max(15.0, 2 * d), d, 16), d)
        if w & {"payload", "heavier", "weight", "ballast", "camera"}:
            d = st.mass
            return Input("mass", "rocket mass", "kg", 1.0, grid(d * 0.5, d * 1.6, d, 23), d)
        mph = "mph" in w or not ("m/s" in q or "km/h" in q)
        per = MPH if mph else 1.0
        d = round(abs(st.wind) / per, 6)
        hi = max(25.0 if mph else 11.0, 2.5 * d)
        return Input("wind", "wind", "mph" if mph else "m/s", per, grid(0, hi, d, 21), d)
    if "wind" in w and st.wind:
        d = abs(st.wind)
        return Input("wind", "wind", "m/s", 1.0, grid(0, max(3 * d, 10.0), d, 21), d)
    if st.speed == 0:
        ft = bool(w & {"ft", "feet", "foot"})
        per = FT if ft else 1.0
        d = round(st.height / per, 6)
        return Input("height", "drop height", "ft" if ft else "m", per, grid(d / 20, d * 3, d, 24, log=True), d)
    if w & {"angle", "far", "farthest", "furthest", "range", "distance"}:
        d = st.angle
        return Input("angle", "launch angle", "deg", 1.0, grid(5, 85, d, 17), d)
    d = st.speed
    return Input("speed", "launch speed", "m/s", 1.0, grid(d * 0.25, d * 2, d, 22), d)


def run(st: MotionStudy, inp: Input, path_points: int = 400) -> list[Point]:
    """Fly the study at every value of the input. Each flight is the run's own model at its own tolerance, so the
    asked value gives exactly the numbers in the report."""
    return [flight_point(inp.study(st, float(v)), float(v), path_points) for v in inp.values]


def flight_point(st: MotionStudy, value: float, path_points: int = 400) -> Point:
    fl = fly(st)
    pts = points(fl)
    t = np.array([p["t"] for p in pts])
    y = np.array([p["y"] for p in pts])
    ph = np.array([p["phase"] for p in pts])
    keep = _thin(t, y[:, X], y[:, Z], path_points)
    pt = Point(value, fl, t[keep], y[keep, X], y[keep, Z], np.array([_speed(r) for r in y[keep]]), y[keep, VX],
               y[keep, VZ], ph[keep])
    pt.events = {e.name: (e.t, float(e.y[X]), float(e.y[Z]), _speed(e.y)) for e in fl.events}
    end = fl.end
    n = {"flight_time_s": end.t, "impact_speed_mps": _speed(end.y), "distance_m": float(end.y[X]),
         "max_height_m": float(y[:, Z].max()), "max_mach": max(p["mach"] for p in pts)}
    f = fl.model.forces(end.t, end.y, end.phase)
    n["impact_drag_n"] = f["k"] * float(np.hypot(*f["air"]))
    if st.rocket:
        n.update(rocket_checks(st, fl, pts)[1])
    elif st.air != "vacuum":
        rho = fl.model.air(0.0)[0]
        n["terminal_speed_mps"] = float(np.sqrt(2 * st.mass * fl.model.g(0.0) / (rho * st.drag_coefficient()(0.0)
                                                                                  * st.area)))
        n["vacuum_speed_mps"] = float(np.sqrt(st.speed ** 2 + 2 * fl.model.g(0.0) * st.height))
    pt.numbers = {k: float(v) for k, v in n.items()}
    return pt


def _thin(t, x, z, n: int) -> np.ndarray:
    """At most n samples, spread along the path and the time together so the fast boost keeps its shape, with the
    first and last kept."""
    if len(t) <= n:
        return np.arange(len(t))
    ds = np.r_[0, np.hypot(np.diff(x), np.diff(z))]
    dt = np.r_[0, np.diff(t)]
    s = np.cumsum(ds / max(ds.sum(), 1e-9) + dt / max(dt.sum(), 1e-9))
    return np.unique(np.r_[0, np.searchsorted(s, np.linspace(0, s[-1], n)).clip(0, len(t) - 1), len(t) - 1])
