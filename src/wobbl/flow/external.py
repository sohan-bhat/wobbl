"""Flow around a cylinder: drag and lift from CFD, steady or shedding vortices, with the published curves as checks.

Everything is solved without dimensions (diameter 1, free stream 1, viscosity 1/Re) on an O grid that reaches 50
diameters out, where free stream conditions let the wake leave on any side, so blockage stays under 1%. Below a
Reynolds number of about 47 the flow is steady and three grids give a grid convergence index on the drag. Above it
vortices shed, and a time accurate run gives the mean drag, the swing of the lift and the shedding frequency
(the Strouhal number) from the lift's own period.
"""
from __future__ import annotations

import math
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from . import foam

FAR = 50.0
STEADY_GRIDS = ((36, 90), (24, 60), (16, 40))    # cells around a quarter and out, fine to coarse
WORKERS = 2


def _first(n_theta: int) -> float:
    """First cell height at the wall: half the spacing around the cylinder, so wall cells are about square."""
    return 0.5 * math.pi / (4 * n_theta)


def steady(re_: float, grid: tuple[int, int], work: Path, fields: bool = False) -> dict:
    nt, nr = grid
    case = foam.Case(Path(work) / f"cyl{nt}x{nr}-re{re_:.6g}")
    foam.cylinder_ogrid(case, 1.0, FAR, nt, nr, _first(nt))
    turb = foam.Turbulence()
    foam.write_fields(case, (1.0, 0, 0), turb)
    foam.write_physics(case, 1.0 / re_, turb)
    foam.write_steady(case, turb, 3000, foam.fo_forces("body", 1.0, 1.0, case.depth) +
                      (foam.fo_cell_centres() if fields else ""), tol=1e-6, small=4 * nt * nr < 5000)
    t0 = time.time()
    case.run("blockMesh")
    log = case.run("simpleFoam")
    head, arr = foam.table(case, "forces")
    cd = arr[:, head.index("Cd")]
    out = {"grid": grid, "cells": 4 * nt * nr, "cd": float(cd[-1]), "cl": float(arr[-1, head.index("Cl")]),
           "settled": float(abs(cd[-1] - cd[int(0.9 * len(cd))]) / abs(cd[-1])), "log": log,
           "seconds": time.time() - t0}
    if fields:
        last = foam.last_time(case)
        out["fields"] = {k: foam.read_field(last / k) for k in ("C", "U", "p")}
    case.clean()
    return out


def steady_levels(re_: float, work: Path, grids=STEADY_GRIDS) -> list[dict]:
    with ThreadPoolExecutor(WORKERS) as pool:
        return list(pool.map(lambda g: steady(re_, g, work, fields=g == grids[0]), grids))


def inside(cx, cy) -> np.ndarray:
    return np.hypot(cx, cy) < 0.5


# --- shedding ----------------------------------------------------------------------------------------------------
UNSTEADY_GRID = (24, 60)
LES_GRID = (16, 36, 8)          # around a quarter, out, along the span
LES_SPAN = math.pi / 2
LES_FAR = 20.0
SECONDS_PER_CELL_STEP = 1.2e-5  # pimpleFoam with two pressure correctors on one core of a 2018 laptop, measured
BYTES_PER_CELL = 2500           # OpenFOAM's peak memory per cell for these runs, measured


def cost(cells: int, end: float, dt: float) -> dict:
    """Predicted run time and memory of a time accurate run, from the cost of one cell for one step here."""
    steps = end / dt
    return {"cells": cells, "steps": int(steps), "seconds": cells * steps * SECONDS_PER_CELL_STEP,
            "memory_MB": cells * BYTES_PER_CELL / 1e6 + 150}


def shedding(re_: float, work: Path, model: str = "laminar", periods: float = 10.0, frames: int = 6,
             st_guess: float = 0.2, grid: tuple = UNSTEADY_GRID, settle_periods: float | None = None,
             max_co: float | None = None) -> dict:
    """A time accurate run until the vortex street has settled, then periods more of it. model is laminar for a
    2D run, or des or les for a 3D slab of the cylinder whose ends are joined periodically, resolving the large
    eddies and modeling the rest. Returns the mean drag, the lift swing, the Strouhal number, the last frames of the
    flow and, for des and les, how much of the turbulent energy the grid resolved."""
    three_d = model in ("des", "les")
    case = foam.Case(Path(work) / f"shed-{model}-re{re_:.6g}")
    if three_d:
        nt, nr, nz = grid if len(grid) == 3 else LES_GRID
        foam.cylinder_ogrid(case, 1.0, LES_FAR, nt, nr, _first(nt), span=LES_SPAN, n_z=nz)
        cells = 4 * nt * nr * nz
        turb = foam.Turbulence.free_stream(model, 1.0, 1.0 / re_, 1.0, intensity=0.01, scale=0.1)
    else:
        nt, nr = grid[:2]
        foam.cylinder_ogrid(case, 1.0, FAR, nt, nr, _first(nt))
        cells = 4 * nt * nr
        turb = foam.Turbulence()
    # a slight cross flow at the start breaks the symmetry, so shedding begins within a few periods
    foam.write_fields(case, (1.0, 0, 0), turb, u_cells=np.tile([1.0, 0.1, 0.0], (cells, 1)))
    foam.write_physics(case, 1.0 / re_, turb)
    period = 1.0 / st_guess
    settle = (settle_periods or (15 if not three_d else 8)) * period
    end = settle + periods * period
    fo = foam.fo_forces("body", 1.0, 1.0, case.depth) + foam.fo_cell_centres()
    if three_d:
        fo += foam.fo_average(settle, nut=True) + foam.fo_cell_volumes()
    foam.write_transient(case, turb, end, 0.005, fo, period / frames, max_co=max_co or (0.5 if not three_d else 1.0),
                         keep=frames)
    t0 = time.time()
    case.run("blockMesh")
    log = case.run("pimpleFoam", timeout=7200)
    head, arr = foam.table(case, "forces")
    t, cd, cl = arr[:, 0], arr[:, head.index("Cd")], arr[:, head.index("Cl")]
    late = t >= settle
    up = np.nonzero((cl[:-1] < 0) & (cl[1:] >= 0) & late[1:])[0]
    tc = t[up] - cl[up] * (t[up + 1] - t[up]) / (cl[up + 1] - cl[up])     # upward zero crossings of the lift
    out = {"model": model, "cells": cells, "steps": int(len(t)), "seconds": time.time() - t0, "log": log,
           "end": float(t[-1]), "settle": settle}
    if len(tc) >= 3:
        span_t = tc[-1] - tc[0]
        n = len(tc) - 1
        win = (t >= tc[0]) & (t <= tc[-1])
        out.update(st=n / span_t, cd=float(np.trapezoid(cd[win], t[win]) / span_t),
                   cl_amp=float(0.5 * (cl[win].max() - cl[win].min())),
                   cl_rms=float(np.sqrt(np.mean(cl[win] ** 2))), cycles=n)
        half = tc[len(tc) // 2]
        first, second = tc[tc <= half], tc[tc >= half]
        if len(first) >= 2 and len(second) >= 2:
            out["st_halves"] = ((len(first) - 1) / (first[-1] - first[0]),
                                (len(second) - 1) / (second[-1] - second[0]))
    out["history"] = (t[::max(1, len(t) // 600)].tolist(), cd[::max(1, len(t) // 600)].tolist(),
                      cl[::max(1, len(t) // 600)].tolist())
    times = sorted((p for p in case.root.iterdir() if p.is_dir() and foam._isnum(p.name) and float(p.name) > 0),
                   key=lambda p: float(p.name))
    c = foam.read_field(times[-1] / "C") if (times[-1] / "C").exists() else None
    out["frames"] = [(float(p.name), foam.read_field(p / "U")) for p in times]
    out["centres"] = c
    if three_d:
        out["resolved"] = _resolution(case, times[-1], c)
    case.clean()
    return out


def _resolution(case, last: Path, c) -> dict:
    """How much of the turbulent energy the grid resolved in the near wake, Pope's measure: resolved over resolved
    plus modeled. The resolved part is half the trace of the velocity's averaged fluctuations, and the modeled part
    is (nut / (Ck delta))^2 with Ck 0.094 and delta the cube root of the cell volume."""
    try:
        r2 = foam.read_field(last / "UPrime2Mean")
        nut = foam.read_field(last / "nutMean")
        vol = foam.read_field(last / "V")
    except (FileNotFoundError, AttributeError, ValueError):
        return {}
    k_res = 0.5 * (r2[:, 0] + r2[:, 3] + r2[:, 5])
    k_sgs = (nut / (0.094 * np.cbrt(vol))) ** 2
    wake = (c[:, 0] > 0.6) & (c[:, 0] < 5.0) & (np.abs(c[:, 1]) < 1.5)
    if not wake.any():
        return {}
    frac = float(np.sum(k_res[wake] * vol[wake]) / np.sum((k_res[wake] + k_sgs[wake]) * vol[wake]))
    return {"resolved_fraction": frac, "wake_cells": int(wake.sum())}
