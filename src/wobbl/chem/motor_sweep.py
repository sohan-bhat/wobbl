"""A motor swept over one design input: the burn solved again at each value, for the interactive view.

The input is the one a designer would change for the question asked. Whether the casing holds is set by the peak
pressure, and the nozzle throat sets that, so it is the default. A question about erosive burning or the port
sweeps the core of every BATES grain, and one about the wall sweeps the casing wall outward from the same bore.
Each value is a full burn at the design run's converged step, kept at its peak pressure: the chamber pressure,
thrust, Kn and regression there, the casing's stress and safety factor, the port to throat ratio and core mass flux,
the impulse, and every grain's cross section, so a viewer can show the motor at its hardest moment for that value.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, replace

import numpy as np

from .grains import Bates

POINTS = 21


@dataclass
class Input:
    key: str              # throat, core or wall
    label: str            # as the slider names it
    design: float         # mm, the value the spec gives
    values: list[float]   # mm, sorted, with design among them


def _words(q: str) -> set[str]:
    return set(re.findall(r"[a-z]+", (q or "").lower()))


def _bates(m) -> list[Bates]:
    return [g for g in m.grains if isinstance(g, Bates)]


def choose(spec, question: str = "") -> Input:
    """The input to sweep for this motor and question, with its range in mm."""
    m, c = spec.motor, spec.casing
    w = _words(question or spec.question)
    ported = [g.port(0.0) for g in m.grains if g.port(0.0) > 0]
    if w & {"core", "port", "erosive", "erosion", "flux"} and _bates(m):
        d = min(g.core for g in _bates(m))
        top = min(g.D for g in _bates(m)) * 0.7
        lo = max(math.sqrt(4 * m.nozzle.At / math.pi) * 0.9, d * 0.6)
        return Input("core", "BATES core", d * 1000, _grid(d, lo, max(top, d * 1.05)))
    if c is not None and w & {"wall", "thickness", "thick", "thicker", "thinner"}:
        t = c.wall
        return Input("wall", "casing wall", t * 1000, _grid(t, t * 0.4, t * 2.0))
    d = m.nozzle.throat
    # past the narrowest port's own area the core, not the throat, chokes the flow
    top = math.sqrt(4 * min(ported) / math.pi) if ported else d * 1.3
    return Input("throat", "nozzle throat", d * 1000, _grid(d, d * 0.62, min(d * 1.3, max(top, d * 1.05))))


def _grid(design: float, lo: float, hi: float, n: int = POINTS) -> list[float]:
    """n values in mm from lo to hi, on a 0.1 mm grid, with the design value itself among them."""
    v = np.round(np.linspace(lo, hi, n) * 1e4) / 10
    v = np.unique(np.r_[v, round(design * 1000, 4)])
    return [float(x) for x in v]


def apply(spec, key: str, mm: float):
    """The spec with the input set to mm."""
    m, c, v = spec.motor, spec.casing, mm / 1000
    if key == "throat":
        return replace(spec, motor=replace(m, nozzle=replace(m.nozzle, throat=v)))
    if key == "core":
        grains = [replace(g, core=v) if isinstance(g, Bates) else g for g in m.grains]
        return replace(spec, motor=replace(m, grains=grains))
    if key == "wall":
        return replace(spec, casing=replace(c, outer=c.inner + 2 * v))
    raise ValueError(f"no motor input {key!r}")


def sweep(spec, dt: float | None = None, question: str = "") -> dict:
    """Burn the motor at every value of its chosen input and keep each burn at its peak pressure."""
    from .ballistics import simulate
    from .motor import MASS_FLUX_LIMIT, PORT_THROAT_MIN, _flux, casing_stress
    from .playback import _grain_type, _r

    inp = choose(spec, question)
    pts = []
    for v in inp.values:
        sp = apply(spec, inp.key, v)
        m = sp.motor
        try:
            b = simulate(m, dt)
        except (ValueError, ZeroDivisionError, OverflowError):
            continue
        i = int(np.argmax(b.P))
        x = min(float(b.x[i]), m.web)
        ratio, flux = _flux(m, b) if any(g.port(0.0) > 0 for g in m.grains) else (float("inf"), 0.0)
        P = float(b.P[i])
        cv = casing_stress(sp.casing, P) if sp.casing is not None else {}
        sf = cv.get("safety_factor")
        target = sp.casing.target_sf if sp.casing is not None else 1.0
        # as the run judges it, on the casing and on erosive burning down the cores
        word = ("fails" if (sf is not None and sf < 1) or ratio < 1 else
                "marginal" if (sf is not None and sf < target) or ratio < PORT_THROAT_MIN
                or flux > MASS_FLUX_LIMIT else "ok")
        geo = []
        for g in m.grains:
            gt = _grain_type(g, [x])
            # every per frame list holds the one regression asked for
            geo.append({k: val[0] if isinstance(val, list) else val for k, val in gt.items()
                        if k in ("profile", "length_now_mm", "core_mm", "outer_mm", "port")})
        pts.append({"value": v, "P_MPa": _r(P / 1e6), "F_N": _r(b.F[i]), "kn": _r(m.kn(x)), "x_mm": _r(x * 1000),
                    "t_peak": _r(b.t[i]), "impulse_Ns": _r(b.impulse), "burn_time_s": _r(b.burn_time),
                    "designation": b.designation(), "hoop_MPa": _r(cv["hoop_Pa"] / 1e6) if cv else None,
                    "safety_factor": _r(sf, 3), "port_throat": _r(ratio, 3) if np.isfinite(ratio) else None,
                    "mass_flux": _r(flux), "throat_mm": _r(m.nozzle.throat * 1000),
                    "inner_mm": _r(sp.casing.inner * 1000) if sp.casing else None,
                    "outer_mm": _r(sp.casing.outer * 1000) if sp.casing else None, "verdict": word, "grains": geo})
    return {"input": inp.key, "label": inp.label, "unit": "mm", "design": inp.design,
            "note": "Each value is its own burn, kept at its peak pressure.", "points": pts}


def write(out, spec, dt: float | None = None) -> None:
    """Add the sweep to the run's playback.json, where the view reads it."""
    import json
    from pathlib import Path

    p = Path(out) / "playback.json"
    if not p.exists():
        return
    data = json.loads(p.read_text())
    data["sweep"] = sweep(spec, dt)
    p.write_text(json.dumps(data, separators=(",", ":")))
