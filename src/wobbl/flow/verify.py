"""Checks every flow answer runs: grid convergence, residuals, mass conservation, wall resolution, a correlation,
and the uncertainty of the turbulence model itself.

The grid convergence index follows Celik et al. (2008), the procedure the ASME Journal of Fluids Engineering asks
for: three grids, the observed order of accuracy from them, a Richardson extrapolated value and a safety factor of
1.25 on the fine grid error band. A RANS answer is an estimate even on a perfect grid, so its confidence is capped
by how well the model is known to do on that kind of flow, whatever the other checks say.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass

from ..units import fmt
from ..verify.checks import Check

# how far RANS answers stray from measurements on each kind of flow, as a fraction, from published comparisons
# (Wilcox, Turbulence Modeling for CFD, 3rd ed.; Menter 1994; the NASA turbulence modeling resource). Laminar and
# scale resolving runs carry no model band of this kind.
RANS_BAND = {
    "pipe": (0.05, "fully developed pipe and duct friction, where SST is within about 5% of Colebrook"),
    "internal": (0.10, "internal flow with junctions and turns, where loss coefficients from RANS scatter about 10%"),
    "attached": (0.10, "attached boundary layers, within about 10% on skin friction and heat transfer"),
    "heat": (0.20, "convective heat transfer, where RANS commonly misses Nusselt numbers by 10 to 25%"),
    "separated": (0.25, "separated flow behind a bluff body, where RANS drag commonly misses by 10 to 40%"),
}


@dataclass
class GCI:
    values: list[float]          # fine to coarse
    sizes: list[float]           # representative cell size, fine to coarse
    order: float                 # observed order p
    extrapolated: float
    gci_fine: float              # fraction of the fine grid value
    kind: str                    # monotone, oscillatory, converged, diverging


def gci(values: list[float], sizes: list[float], formal: float = 2.0) -> GCI:
    """Celik et al. (2008) on the three finest grids, values and sizes ordered fine to coarse.

    The observed order comes from a fixed point iteration on p = |ln|e32/e21| + q(p)| / ln r21. When the grids
    oscillate, or the observed order is outside 0.5 to formal + 1, the order is held at the formal order and the
    band uses a safety factor of 3, Roache's advice when the order cannot be trusted.
    """
    f1, f2, f3 = values[:3]
    h1, h2, h3 = sizes[:3]
    r21, r32 = h2 / h1, h3 / h2
    e21, e32 = f2 - f1, f3 - f2
    scale = max(abs(f1), 1e-300)
    if abs(e21) <= 1e-12 * scale and abs(e32) <= 1e-12 * scale:
        return GCI(values, sizes, formal, f1, 0.0, "converged")
    if abs(e21) <= 1e-12 * scale:
        return GCI(values, sizes, formal, f1, 1.25 * abs(e32) / scale / (r32 ** formal - 1), "converged")
    ratio = e32 / e21
    s = 1.0 if ratio > 0 else -1.0
    kind = "monotone" if ratio > 0 else "oscillatory"
    p, fs = formal, 1.25
    if kind == "monotone":
        p = abs(math.log(abs(ratio))) / math.log(r21)
        for _ in range(100):
            q = math.log(max((r21 ** p - s) / (r32 ** p - s), 1e-12))
            pn = abs(math.log(abs(ratio)) + q) / math.log(r21)
            if abs(pn - p) < 1e-9:
                break
            p = pn
        if abs(ratio) < 1.0:   # the change grows as the grid is refined
            kind = "diverging"
    if kind != "monotone" or not 0.5 <= p <= formal + 1.0:
        p, fs = formal, 3.0
    ext = (r21 ** p * f1 - f2) / (r21 ** p - 1)
    band = fs * abs(e21 / scale) / (r21 ** p - 1)
    if kind == "oscillatory":   # the band has to cover the whole swing
        band = max(band, 1.25 * (max(values[:3]) - min(values[:3])) / scale / 2)
    return GCI(values, sizes, p, ext, band, kind)


def gci_check(g: GCI | None, label: str, unit: str, tol: float = 0.03, values_only: list[float] | None = None,
              what: str = "grids") -> Check:
    if g is None:
        v = values_only or []
        return Check("grid convergence", "warn", f"{label} from {len(v)} grid{'s' if len(v) != 1 else ''} only, so "
                                                 "there is no grid convergence index and the number is unverified",
                     {"values": v})
    trend = " -> ".join(fmt(v) for v in reversed(g.values))
    d = {"values_fine_to_coarse": g.values, "sizes": g.sizes, "order": g.order, "extrapolated": g.extrapolated,
         "gci_fine": g.gci_fine, "kind": g.kind}
    base = (f"{label} {trend} {unit} over {len(g.values)} {what}, grid convergence index {100 * g.gci_fine:.2g}% "
            f"(observed order {g.order:.2g}, extrapolated {fmt(g.extrapolated)} {unit})")
    if g.kind == "diverging":
        return Check("grid convergence", "fail", base + ", and the grids move apart, so the answer is not converged", d)
    if g.gci_fine <= tol:
        return Check("grid convergence", "pass", base, d)
    if g.gci_fine <= 3 * tol:
        return Check("grid convergence", "warn", base + ", so expect about that much discretization error", d)
    return Check("grid convergence", "fail", base + ", so the grid is too coarse for this answer", d)


_RES = re.compile(r"Solving for (\w+), Initial residual = ([-\d.eE+]+), Final residual = ([-\d.eE+]+)")


def residuals(log: str) -> dict[str, tuple[float, float]]:
    """First and last initial residual of each solved field in an OpenFOAM log."""
    out: dict[str, list[float]] = {}
    for m in _RES.finditer(log):
        out.setdefault(m.group(1), []).append(float(m.group(2)))
    return {k: (v[0], v[-1]) for k, v in out.items() if v}


def residual_check(log: str, tol: float = 1e-4, steady: bool = True) -> Check:
    res = residuals(log)
    if not res:
        return Check("residuals", "fail", "the solver log has no residuals, so the run did not iterate")
    worst = max(res.items(), key=lambda kv: kv[1][1])
    converged = "SIMPLE solution converged" in log or "reached convergence" in log
    drop = min(v[0] / max(v[1], 1e-300) for v in res.values())
    d = {"fields": {k: {"first": a, "last": b} for k, (a, b) in res.items()}, "tolerance": tol,
         "converged_flag": converged}
    s = (f"largest final residual {worst[1][1]:.2g} ({worst[0]}), every field fell at least "
         f"{math.log10(max(drop, 1)):.1f} orders")
    if not steady:
        return Check("residuals", "pass" if worst[1][1] < 1e-3 else "warn",
                     s.replace("final residual", "residual in the last time step"), d)
    if worst[1][1] <= tol or converged:
        return Check("residuals", "pass", s + (", and the solver met its convergence target" if converged else ""), d)
    if worst[1][1] <= 10 * tol:
        return Check("residuals", "warn", s + ", close to converged, so the key number may still move slightly", d)
    return Check("residuals", "fail", s + ", so the solution did not converge and its numbers are not reliable", d)


def mass_check(inflow: float, outflow: float, unit: str = "kg/s", tol: float = 0.005) -> Check:
    """inflow and outflow as positive magnitudes."""
    err = abs(inflow - outflow) / max(abs(inflow), 1e-300)
    s = f"in {fmt(inflow)} {unit}, out {fmt(outflow)} {unit} ({100 * err:.2g}% apart)"
    st = "pass" if err <= tol else ("warn" if err <= 4 * tol else "fail")
    return Check("mass conservation", st, s + ("" if st == "pass" else ", so the solution has not settled"),
                 {"in": inflow, "out": outflow, "error": err})


def yplus_check(stats: dict[str, tuple[float, float, float]], treatment: str) -> Check:
    """stats maps wall patch to (min, max, mean) y+. treatment is 'resolved', 'wall functions' or 'laminar'."""
    if treatment == "laminar" or not stats:
        return Check("wall resolution", "skip", "laminar flow, no wall model involved")
    mean = max(v[2] for v in stats.values())
    top = max(v[1] for v in stats.values())
    d = {"patches": {k: {"min": a, "max": b, "mean": c} for k, (a, b, c) in stats.items()}, "treatment": treatment}
    s = f"y+ on the walls averages {fmt(mean)}, at most {fmt(top)}, with {treatment}"
    if treatment == "resolved":
        st = "pass" if top <= 2.5 else ("warn" if mean <= 5 else "fail")
        why = "" if st == "pass" else ", too tall for the first cells to resolve the wall, so refine near it"
    else:
        st = "pass" if 25 <= mean <= 300 else "warn"
        why = "" if st == "pass" else (", so part of the wall sits in the buffer layer, where wall functions are "
                                        "least accurate, though the blended wall function keeps the error to a "
                                        "few percent")
    return Check("wall resolution", st, s + why, d)


def compare_check(name: str, got: float, want: float, source: str, unit: str, tol: float,
                  info_only: bool = False) -> Check:
    err = abs(got - want) / max(abs(want), 1e-300)
    u = f" {unit}" if unit else ""
    s = f"{source} gives {fmt(want)}{u}, the simulation {fmt(got)}{u} ({100 * err:.2g}% apart)"
    if info_only:
        return Check(name, "info", s, {"expected": want, "got": got, "error": err})
    st = "pass" if err <= tol else ("warn" if err <= 2 * tol else "fail")
    return Check(name, st, s, {"expected": want, "got": got, "error": err, "tolerance": tol})


def model_reason(model: str, flow: str) -> tuple[str, str] | None:
    """The confidence cap a turbulence model puts on an answer, as (level, reason)."""
    if model in ("laminar", "none", "bem", "correlation"):
        return None
    if model in ("les", "des"):
        return ("medium", "scale resolving turbulence on a modest grid, and how much of the turbulence it resolved "
                          "is in the checks")
    band, why = RANS_BAND.get(flow, RANS_BAND["attached"])
    level = "medium" if band > 0.05 else "high"
    if level == "high":
        return None
    return (level, f"the RANS turbulence model is an estimate for {why}, so expect about {100 * band:.0f}% either "
                   "way")
