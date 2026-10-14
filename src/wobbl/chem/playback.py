"""Playback files: the run's own data, thinned to what a viewer needs to animate it truthfully.

Each run writes output/playback.json, a few tens of kB, with a manifest naming the kind of run, the units and the
frame count, the fixed geometry, and frames in order. A motor's frames step through the burn in time and carry the
chamber pressure, thrust, Kn, regression, propellant left and the casing's safety factor at that instant, with each
grain's burning cross section: the core diameter and remaining length of a round core, or the port outline of any
other core as a closed polyline in mm. A flame's frames step through equivalence ratio, each a real equilibrium
solve, and a propellant's step down the nozzle from throat to exit. Numbers are rounded to what the run resolves,
so the same input always gives the same file.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from .grains import Bates, EndBurner, MapGrain, Tube

VERSION = 1


def _r(x, digits: int = 4):
    """Round to significant digits, so the file stays small and identical from run to run."""
    if x is None or not np.isfinite(x):
        return None
    x = float(x)
    if x == 0:
        return 0.0
    return float(f"{x:.{digits}g}")


def _write(out: Path, data: dict) -> str:
    path = Path(out) / "playback.json"
    path.write_text(json.dumps(data, separators=(",", ":")))
    return path.name


def motor(spec, b, case_vals: dict, out: Path, frames: int = 120) -> str:
    """Frames through the burn with each grain's cross section, pressure, thrust and the casing's margin."""
    m = spec.motor
    on = np.nonzero(b.F > 0.002 * b.peak_thrust)[0]
    t_end = float(b.t[min(on[-1] + 1, len(b.t) - 1)]) if len(on) else float(b.t[-1])
    times = np.linspace(0.0, t_end, frames)
    idx = np.clip(np.searchsorted(b.t, times), 0, len(b.t) - 1)
    burned = np.concatenate([[0.0], np.cumsum(0.5 * (b.mdot[1:] + b.mdot[:-1]) * np.diff(b.t))])
    left = np.clip(b.mass * (1 - burned / max(burned[-1], 1e-12)), 0.0, None)
    py = case_vals.get("yield_pressure_Pa")
    types, slots, z = [], [], 0.0
    for g in m.grains:
        k = next((i for i, (h, _) in enumerate(types) if h is g), None)
        if k is None:
            types.append((g, len(types)))
            k = len(types) - 1
        slots.append({"type": k, "z_mm": _r(z * 1000), "length_mm": _r(g.L * 1000)})
        z += g.L
    xs = [min(float(b.x[i]), m.web) for i in idx]
    gtypes = [_grain_type(g, xs) for g, _ in types]
    rows = []
    for j, i in enumerate(idx):
        P = float(b.P[i])
        rows.append({"t": _r(times[j], 5), "P_MPa": _r(P / 1e6), "F_N": _r(b.F[i]), "kn": _r(m.kn(xs[j])),
                     "x_mm": _r(xs[j] * 1000), "propellant_kg": _r(left[i]), "mdot_kg_s": _r(b.mdot[i]),
                     "casing_sf": _r(py / P, 3) if py and P > m.ambient * 1.5 else None})
    c = spec.casing
    data = {"manifest": {"kind": "solid-motor", "version": VERSION, "generator": "wobbl motor", "frames": len(rows),
                         "time": "frames are evenly spaced in time from ignition to the end of the tail off",
                         "units": {"t": "s", "P_MPa": "MPa", "F_N": "N", "kn": "burning area over throat area",
                                   "x_mm": "mm of regression", "propellant_kg": "kg", "mdot_kg_s": "kg/s",
                                   "casing_sf": "yield pressure over chamber pressure", "lengths": "mm"},
                         "grain_profiles": {"round": "core_mm per frame, a circle centered on the axis",
                                            "tube": "core_mm and outer_mm per frame",
                                            "end": "solid face, only length_mm changes",
                                            "outline": "port per frame as closed polylines of [x, y] in mm, "
                                                       "centered on the axis"}},
            "motor": {"name": spec.name, "designation": b.designation(), "propellant": m.propellant.name,
                      "nozzle": {"throat_mm": _r(m.nozzle.throat * 1000), "exit_mm": _r(m.nozzle.exit * 1000)},
                      "casing": None if c is None else {
                          "material": case_vals.get("material"), "outer_mm": _r(c.outer * 1000),
                          "inner_mm": _r(c.inner * 1000), "length_mm": _r((c.length or z) * 1000),
                          "yield_pressure_MPa": _r(py / 1e6), "allowable_MPa": _r(case_vals["allowable_Pa"] / 1e6),
                          "burst_MPa": _r(case_vals["burst_pressure_Pa"] / 1e6), "target_sf": c.target_sf},
                      "peak_pressure_MPa": _r(b.peak_pressure / 1e6), "impulse_Ns": _r(b.impulse),
                      "burn_time_s": _r(b.burn_time)},
            "grain_types": gtypes, "grains": slots, "frames": rows}
    return _write(out, data)


def _grain_type(g, xs: list[float]) -> dict:
    """A grain's fixed sizes and its cross section at each frame's regression."""
    base = {"shape": g.shape, "diameter_mm": _r(g.D * 1000), "length_mm": _r(g.L * 1000), "ends_burning": g.ends}
    length = [_r(g.length(min(x, g.web)) * 1000 if x < g.web else 0.0) for x in xs]
    if isinstance(g, Bates):
        return {**base, "profile": "round", "length_now_mm": length,
                "core_mm": [_r(min(g.core + 2 * x, g.D) * 1000) for x in xs]}
    if isinstance(g, Tube):
        return {**base, "profile": "tube", "length_now_mm": length,
                "core_mm": [_r(min(g.core + 2 * x, g.D) * 1000) for x in xs],
                "outer_mm": [_r(max(g.D - 2 * x, g.core + 2 * x) * 1000) for x in xs]}
    if isinstance(g, EndBurner):
        return {**base, "profile": "end", "length_now_mm": length}
    if isinstance(g, MapGrain):
        return {**base, "profile": "outline", "length_now_mm": length, "params": {k: _r(v * 1000) if isinstance(
                v, float) else v for k, v in g.params.items()}, "port": [_outline(g, x) for x in xs]}
    return {**base, "profile": "end", "length_now_mm": length}


def _outline(g: MapGrain, x: float, points: int = 48) -> list:
    """The port's edge at regression x as closed polylines in mm, thinned to about ``points`` vertices each."""
    if x >= g.web:
        return []
    # longest first, without the few cell specks the grid leaves right at the port's starting edge
    lines = sorted((ln for ln in g.port_outline(x) if len(ln) >= 8), key=len, reverse=True)
    out = []
    for ln in lines:
        step = max(1, math.ceil(len(ln) / points))
        pts = np.vstack([ln[::step], ln[-1:]]) if (len(ln) - 1) % step else ln[::step]
        out.append([[_r(px * 1000, 3), _r(py * 1000, 3)] for px, py in pts])    # 0.1 mm, the grid's cell
    return out


def flame(out: Path, phis, temps, species: list[dict], phi: float, info: dict) -> str:
    """Frames through equivalence ratio, each an equilibrium flame, with the run's own mixture marked."""
    keys = sorted({k for s in species for k in s}, key=lambda k: -max(s.get(k, 0.0) for s in species))[:10]
    rows = [{"phi": _r(p), "T_K": _r(T, 5), "x": {k: _r(s.get(k, 0.0), 3) for k in keys}}
            for p, T, s in zip(phis, temps, species)]
    here = int(np.argmin(np.abs(np.asarray(phis) - phi)))
    data = {"manifest": {"kind": "flame", "version": VERSION, "generator": "wobbl flame", "frames": len(rows),
                         "sweep": "frames step through equivalence ratio, each a separate equilibrium solve",
                         "units": {"phi": "fuel over stoichiometric fuel", "T_K": "K",
                                   "x": "mole fraction in the burned gas"}, "operating_frame": here},
            "flame": info, "frames": rows}
    return _write(out, data)


def nozzle(out: Path, rows: list[dict], info: dict) -> str:
    """Frames from the throat to past the exit, shifting and frozen, with Isp at each area ratio."""
    data = {"manifest": {"kind": "nozzle", "version": VERSION, "generator": "wobbl propellant", "frames": len(rows),
                         "sweep": "frames step down the nozzle in pressure, each an equilibrium expansion",
                         "units": {"area_ratio": "area over throat area", "P_MPa": "MPa", "T_K": "K",
                                   "v_m_s": "m/s", "isp_s": "s at the ambient pressure",
                                   "condensed": "mass fraction of condensed products"}},
            "propellant": info, "frames": rows}
    return _write(out, data)
