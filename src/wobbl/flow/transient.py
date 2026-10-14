"""Time accurate, scale resolving runs on a CAD part's grid: DES and LES, for flow that will not sit still.

Behind a bluff body the flow sheds vortices, and steady RANS can only give a smeared mean. Here pimpleFoam marches
the same snapped grid in time, with Spalart and Allmaras' improved delayed detached eddy simulation (RANS in the
attached boundary layers, LES in the separated wake) or with the WALE large eddy model and a wall model, holding
the Courant number under one. The run first lets the wake build up, then averages: the mean drag and side forces,
how much the lift swings, the shedding frequency from the spectrum of the swinging force, and how much of the
turbulent energy in the near wake the grid resolved rather than modeled (Pope's measure), which is the check that
a scale resolving run was fine enough to be one.
"""
from __future__ import annotations

import re
import time
from pathlib import Path

import numpy as np

from . import foam

FRAMES = 8


def times_for(s, length: float) -> tuple[float, float]:
    """How long the wake takes to build and how long to average, in seconds, from the flow through time."""
    t_flow = length / max(s.speed, 1e-12)
    settle, window = s.unsteady_spans
    return settle * t_flow, window * t_flow


def run_transient(case: foam.Case, s, grid, tag: str, fields: bool) -> dict:
    """Write the time accurate controls on a meshed case, run it and read the statistics."""
    from .partflow import _functions, _non_orthogonal

    settle, window = times_for(s, s.ref_length)
    end = settle + window
    dt0 = 0.2 * s.size / 2 ** max(x.level[1] for x in s.surfaces) / max(s.speed, 1e-12)
    fo = _functions(s, grid) + foam.fo_average(settle, nut=True) + foam.fo_cell_volumes()
    foam.write_transient(case, s.turb, end, dt0, fo, window / FRAMES, max_co=1.0, keep=FRAMES)
    _non_orthogonal(case, grid)
    t0 = time.time()
    log = case.run("pimpleFoam", timeout=6 * 3600)
    out = {"tag": tag, "log": log, "t_solve": time.time() - t0, "settle": settle, "end": end}
    out.update(statistics(case, s, grid, settle))
    co = [float(x) for x in re.findall(r"Courant Number mean: [\d.eE+-]+ max: ([\d.eE+-]+)", log)]
    out["courant"] = max(co[len(co) // 2:]) if co else float("nan")
    times = sorted((p for p in case.root.iterdir() if p.is_dir() and foam._isnum(p.name) and float(p.name) > 0),
                   key=lambda p: float(p.name))
    if times:
        out["resolved"] = resolution(case, times[-1], s)
        if fields:
            out["frames"] = [(float(p.name), foam.read_field(p / "U")) for p in times if (p / "U").exists()]
    return out


def statistics(case: foam.Case, s, grid, settle: float) -> dict:
    out = {}
    if s.mode == "external":
        head, arr = foam.table(case, "forces")
        t = arr[:, 0]
        cols = {c: arr[:, head.index(c)] for c in ("Cd", "Cl", "Cs") if c in head}
        qa = 0.5 * s.rho * s.speed ** 2 * s.ref_area
        late = t >= settle
        if late.sum() < 10:
            raise RuntimeError("the run ended before the averaging window began")
        tw = t[late]
        mean = {k: float(np.trapezoid(v[late], tw) / (tw[-1] - tw[0])) for k, v in cols.items()}
        out.update(cd=mean["Cd"], cl=mean.get("Cl", 0.0), cs=mean.get("Cs", 0.0))
        out["drag"], out["lift"], out["side"] = (out["cd"] * qa, out["cl"] * qa, out["cs"] * qa)
        swing = {k: float(np.std(v[late])) for k, v in cols.items() if k != "Cd"}
        key = max(swing, key=swing.get) if swing else "Cd"
        out["swing_key"], out["swing_rms"] = key, swing.get(key, 0.0)
        out["frequency"], out["cycles"] = dominant(tw, cols[key][late])
        half = tw[0] + 0.5 * (tw[-1] - tw[0])
        a, b = tw <= half, tw >= half
        m1 = np.trapezoid(cols["Cd"][late][a], tw[a]) / max(tw[a][-1] - tw[a][0], 1e-300)
        m2 = np.trapezoid(cols["Cd"][late][b], tw[b]) / max(tw[b][-1] - tw[b][0], 1e-300)
        out["halves"] = (float(m1), float(m2))
        out["value"] = out["drag"]
        step = max(1, len(t) // 800)
        out["history"] = cols["Cd"] * qa
        out["series"] = {"t": t[::step].tolist(), **{k: v[::step].tolist() for k, v in cols.items()}}
    else:
        from .partflow import _results
        out.update(_results(case, s, grid))
    return out


def dominant(t: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """The strongest frequency in a signal sampled at uneven times, from a Hann windowed spectrum with the peak
    refined by a parabola, and how many of its cycles the signal spans."""
    if len(t) < 16 or np.ptp(t) <= 0:
        return 0.0, 0.0
    n = 2 ** int(np.ceil(np.log2(len(t))))
    tu = np.linspace(t[0], t[-1], n)
    yu = np.interp(tu, t, y)
    yu = (yu - yu.mean()) * np.hanning(n)
    spec = np.abs(np.fft.rfft(yu))
    freq = np.fft.rfftfreq(n, tu[1] - tu[0])
    k = int(np.argmax(spec[1:])) + 1
    if 1 <= k < len(spec) - 1:
        a, b, c = spec[k - 1], spec[k], spec[k + 1]
        shift = 0.5 * (a - c) / (a - 2 * b + c) if (a - 2 * b + c) != 0 else 0.0
        f = (k + shift) * (freq[1] - freq[0])
    else:
        f = freq[k]
    return float(f), float(f * (t[-1] - t[0]))


def resolution(case: foam.Case, last: Path, s) -> dict:
    """Pope's measure of how much of the turbulent energy the grid resolved in the separated region: the resolved
    energy (half the trace of the averaged velocity fluctuations) over it plus the modeled part, (nut / (Ck delta))^2
    with Ck 0.094 and delta the cube root of the cell volume, averaged over cells with real fluctuation."""
    try:
        r2 = foam.read_field(last / "UPrime2Mean")
        nut = foam.read_field(last / "nutMean")
        vol = foam.read_field(last / "V")
    except (FileNotFoundError, AttributeError, ValueError, TypeError):
        return {}
    if r2.ndim != 2 or np.ndim(nut) != 1 or np.ndim(vol) != 1:
        return {}
    k_res = 0.5 * (r2[:, 0] + r2[:, 3] + r2[:, 5])
    k_sgs = (nut / (0.094 * np.cbrt(vol))) ** 2
    busy = k_res > 0.01 * 0.5 * s.speed ** 2 * 0.01      # where the flow swings by more than 1% of the stream
    if busy.sum() < 10:
        return {"resolved_fraction": float("nan"), "cells": int(busy.sum())}
    frac = float(np.sum(k_res[busy] * vol[busy]) / np.sum((k_res[busy] + k_sgs[busy]) * vol[busy]))
    return {"resolved_fraction": frac, "cells": int(busy.sum())}
