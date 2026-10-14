"""The CFD check on a pipe's friction factor.

A short slice of the pipe, one cell long with its ends joined, as an axisymmetric wedge, is driven by a source that
holds the mean velocity, so the pressure gradient it needs is the fully developed friction. Everything is solved
without dimensions (diameter 1, mean velocity 1, viscosity 1/Re), so only the Reynolds number matters. Laminar
flow is checked against 64/Re, which is exact, and turbulent flow on a smooth wall against Colebrook with k omega
SST resolved down to the wall, the first cell under a y+ of 1. Three grids give a grid convergence index.
Transitional flow and rough walls are not checked this way, since RANS cannot predict transition and a rough wall
needs a roughness model whose own scatter is as large as the chart's.
"""
from __future__ import annotations

import re
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from . import correlations as C
from . import foam
from .verify import compare_check, gci, gci_check, yplus_check

LAMINAR_CELLS = (45, 30, 20)          # radial cells, fine to coarse
TURBULENT_CELLS = (200, 135, 90)
WORKERS = 2                           # grids solved at once, the most a shared laptop should give one study


def skip_reason(re_: float, rel_rough: float) -> str:
    if C.LAMINAR_RE <= re_ <= C.TURBULENT_RE:
        return "the flow is transitional, which RANS cannot predict"
    if re_ > C.TURBULENT_RE and rel_rough > 1e-5:
        return "the wall is rough, and a roughness wall model scatters as much as the chart"
    if re_ > 2e6:
        return "the Reynolds number is past 2e6, where a wall resolved grid gets too large for a quick check"
    return ""


def _level(work: Path, re_: float, nr: int, turbulent: bool, iterations: int) -> dict:
    case = foam.Case(work / f"grid{nr}")
    R, U, nu = 0.5, 1.0, 1.0 / re_
    if turbulent:
        first = 2e-4 * 135 / nr * (1e5 / re_) ** 0.9     # y+ near 0.5 to 1, scaled with the friction velocity
        foam.pipe_wedge(case, R, 0.02, nr, 1 / foam.geometric(first, R, nr))
        turb = foam.Turbulence.free_stream("rans", U, nu, 2 * R)
        prof = 1.224 * U * np.clip(1 - case.radii / R, 0, 1) ** (1 / 7)   # start from a 1/7 power law
        u0 = np.c_[prof, 0 * prof, 0 * prof]
    else:
        foam.pipe_wedge(case, R, 0.02, nr, 0.3)
        turb, u0 = foam.Turbulence(), None
    foam.write_fields(case, (U, 0, 0), turb, u_cells=u0)
    foam.write_physics(case, nu, turb)
    foam.write_steady(case, turb, iterations, foam.fo_yplus() + foam.fo_cell_centres(), tol=1e-7,
                      relax=(0.95, 0.95), small=True)
    case.write("system/fvOptions", "fvOptions", {"momentumSource": {"type": "meanVelocityForce", "selectionMode":
                                                                   "all", "fields": ["U"], "Ubar": (U, 0, 0)}})
    t0 = time.time()
    case.run("blockMesh")
    log = case.run("simpleFoam")
    g = np.array([float(x) for x in re.findall(r"pressure gradient = ([-\d.eE+]+)", log)[1::2]])
    tail = g[int(0.75 * len(g)):]
    out = {"cells": nr, "f": 4 * R * g[-1] / U ** 2, "settled": float(np.ptp(tail) / abs(g[-1])),
           "seconds": time.time() - t0, "log": log}
    if turbulent:
        head, arr = foam.table(case, "yPlus")
        out["yplus"] = {"wall": tuple(float(x) for x in arr[-1, -3:])} if len(arr) else {}
    last = foam.last_time(case)
    c, u = foam.read_field(last / "C"), foam.read_field(last / "U")
    out["profile"] = (np.hypot(c[:, 1], c[:, 2]).tolist(), u[:, 0].tolist())
    return out


def friction(re_: float, rel_rough: float, work: Path, keep: bool = False, say=None) -> dict | None:
    """Friction factor from CFD on three grids, with its checks, or None with the reason it was skipped."""
    from ..solvers.base import SolverError

    why = skip_reason(re_, rel_rough)
    if why:
        return {"skipped": why}
    turbulent = re_ > C.TURBULENT_RE
    cells = TURBULENT_CELLS if turbulent else LAMINAR_CELLS
    iterations = 1500 if turbulent else 1200
    work = Path(work) / "cfd"
    say = say or (lambda a, b: None)
    say("cfd", f"periodic pipe slice at Re {re_:.3g} on {len(cells)} grids, two at a time")
    try:
        with ThreadPoolExecutor(WORKERS) as pool:
            levels = list(pool.map(lambda n: _level(work, re_, n, turbulent, iterations), cells))
    except SolverError as e:
        return {"skipped": f"OpenFOAM failed ({e})"}
    finally:
        if not keep:
            import shutil
            shutil.rmtree(work, ignore_errors=True)
    f = [L["f"] for L in levels]
    g = gci(f, [1 / n for n in cells])
    checks = [gci_check(g, "friction factor", "", tol=0.02)]
    worst = max(L["settled"] for L in levels)
    checks.append(_settled(worst))
    if turbulent:
        checks.append(yplus_check(levels[0]["yplus"], "resolved"))
        want, src, tol = C.colebrook(re_, rel_rough), "Colebrook", 0.05
    else:
        want, src, tol = 64 / re_, "64/Re", 0.01
    checks.append(compare_check("CFD against the chart", f[0], want, src, "", tol))
    return {"f": f[0], "levels": [{k: v for k, v in L.items() if k not in ("log", "profile")} for L in levels],
            "gci": g, "checks": checks, "profile": levels[0]["profile"], "model": "rans" if turbulent else "laminar",
            "log": levels[0]["log"]}


def _settled(change: float):
    from ..verify.checks import Check
    st = "pass" if change < 1e-3 else ("warn" if change < 1e-2 else "fail")
    return Check("iterative convergence", st, f"the pressure gradient moved {100 * change:.2g}% over the last "
                                              "quarter of the iterations on the slowest grid",
                 {"change": change})
