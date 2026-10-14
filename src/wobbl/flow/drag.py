"""Drag on a body in a stream of air or water, from published drag curves.

"How much drag does a 22 cm ball feel at 20 m/s?" becomes: the Reynolds number places the ball on Clift, Grace
and Weber's standard drag curve, the drag coefficient times the dynamic pressure and frontal area gives the force,
and the checks say whether the flow sits near the drag crisis, where surface roughness and free stream turbulence
decide the answer more than any curve can.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..outcome_generic import GenericOutcome
from ..units import fmt
from ..verify.checks import Check, confidence
from . import correlations as C
from .fluids import get_fluid

SHAPES = ("sphere", "cylinder", "disk", "square-plate", "cube")
CRISIS = {"sphere": (2e5, 5e5), "cylinder": (2e5, 5e5)}


@dataclass
class DragStudy:
    shape: str
    size: float                  # m, diameter or side
    speed: float                 # m/s
    length: float = 0.0          # m, span of a cylinder (0 for a very long one)
    fluid: str = "air"
    temp: float = 20.0
    cd: float = 0.0              # a known drag coefficient instead of the curve
    max_drag: float = 0.0        # N, 0 when not judged
    question: str = ""


def aspect_factor(length_over_d: float) -> float:
    """Drag of a finite cylinder in cross flow over that of an infinite one, from its free ends (White, Fluid
    Mechanics, Table 7.2, measured near Re 1e5): 0.64 at L/D 1, 0.74 at 5, 0.82 at 10, 0.98 at 40."""
    lr = [1, 2, 3, 5, 10, 20, 40, 1e9]
    k = [0.64, 0.68, 0.72, 0.74, 0.82, 0.91, 0.98, 1.0]
    return float(np.interp(math.log10(max(length_over_d, 1)), np.log10(lr), k))


def coefficient(shape: str, re: float, length_over_d: float = 0.0) -> tuple[float, str]:
    if shape == "sphere":
        return C.sphere_cd(re), "Clift, Grace and Weber's standard drag curve"
    if shape == "cylinder":
        k = aspect_factor(length_over_d) if length_over_d else 1.0
        return C.cylinder_cd(min(re, 2e5)) * k, "Sucker and Brauer's fit" + (
            f" times {k:.2f} for the free ends (White)" if k < 1 else "")
    if shape in C.BLUFF_CD:
        return C.BLUFF_CD[shape], "Hoerner's value for this shape"
    raise ValueError(f"unknown shape '{shape}'; use {', '.join(SHAPES)}")


def run_drag(st: DragStudy, out_dir: str | Path) -> GenericOutcome:
    from .render import chart

    t0 = time.time()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    if st.shape not in SHAPES:
        raise ValueError(f"unknown shape '{st.shape}'; use {', '.join(SHAPES)}")
    fl = get_fluid(st.fluid, st.temp)
    re = fl.rho * st.speed * st.size / fl.mu
    lod = st.length / st.size if st.length else 0.0
    if st.cd:
        cd, src = st.cd, "the drag coefficient given"
    else:
        cd, src = coefficient(st.shape, re, lod)
    if st.shape == "cylinder":
        area = st.size * (st.length or 1.0)
    elif st.shape in ("square-plate", "cube"):
        area = st.size ** 2
    else:
        area = math.pi * st.size ** 2 / 4
    q = 0.5 * fl.rho * st.speed ** 2
    force = cd * q * area
    checks: list[Check] = []
    extra = []
    lo, hi = CRISIS.get(st.shape, (0, 0))
    if lo and lo <= re <= hi:
        checks.append(Check("drag crisis", "warn", f"Reynolds number {fmt(re)} is in the drag crisis, where the drag "
                                                   "coefficient falls by up to five times as the boundary layer turns "
                                                   "turbulent, so roughness and free stream turbulence decide it"))
        extra.append(("low", "the drag crisis makes the drag coefficient uncertain by a factor of two or more"))
    elif st.shape == "cylinder" and re > hi:
        checks.append(Check("drag crisis", "warn", f"Reynolds number {fmt(re)} is past the cylinder drag crisis, "
                                                   "where the published curves scatter widely"))
        extra.append(("medium", "drag past the crisis depends on surface roughness"))
    else:
        checks.append(Check("drag crisis", "pass", f"Reynolds number {fmt(re)} is clear of the drag crisis"))
    mach = st.speed / 343.0 if fl.rho < 10 else 0.0
    checks.append(Check("compressibility", "pass" if mach < 0.3 else "warn",
                        f"Mach {mach:.2f}" + ("" if mach < 0.3 else ", fast enough that compressibility raises "
                                                                     "drag above these incompressible curves")))
    if st.shape == "sphere" and not st.cd and re < 2e5:
        # Morrison's (2013) independent fit to the same data
        x = re / 2.63e5
        alt = (24 / re + 2.6 * (re / 5) / (1 + (re / 5) ** 1.52) + 0.411 * x ** -7.94 / (1 + x ** -8)
               + 0.25 * (re / 1e6) / (1 + re / 1e6))
        err = abs(alt - cd) / cd
        checks.append(Check("second curve", "pass" if err < 0.1 else "warn",
                            f"Morrison's curve gives a drag coefficient of {alt:.3g} against {cd:.3g} "
                            f"({100 * err:.2g}% apart)"))
    if not st.cd:
        extra.append(("medium", "a published drag curve for the ideal shape, typically within about 10% of a real "
                                "smooth body, and no CFD was run"))
    level, reasons = confidence(checks, extra)
    verdict = "ok" if not st.max_drag or force <= 0.9 * st.max_drag else (
        "marginal" if force <= st.max_drag else "fails")
    name = st.shape.replace("-", " ")
    head = (f"A {fmt(st.size * 1000)} mm {name} at {fmt(st.speed)} m/s in {fl.name.lower()} feels {fmt(force)} N of "
            f"drag (drag coefficient {cd:.3g} at Re {fmt(re)})" +
            (f", against a limit of {fmt(st.max_drag)} N." if st.max_drag else "."))
    imgs = {}
    try:
        if st.shape in ("sphere", "cylinder") and not st.cd:
            res = np.logspace(0, 6.3 if st.shape == "sphere" else 5.3, 200)
            curve = [coefficient(st.shape, r, lod)[0] for r in res]
            imgs["curve"] = chart(out / "images" / "drag_curve.png", {f"{name} drag curve": (res, curve)},
                                  "Reynolds number", "drag coefficient", {"this body": ([re], [cd])}, src,
                                  logx=True, logy=True)
    except Exception as e:  # noqa: BLE001
        imgs["error"] = str(e)
    fixes = []
    if verdict != "ok":
        fixes.append({"kind": "shape", "title": "Streamline the body",
                      "summary": f"A teardrop of the same frontal area has a drag coefficient near 0.05 against "
                                 f"{cd:.2g}, which would bring the drag to about {fmt(0.05 * q * area)} N.",
                      "auto": False})
        top = st.speed * math.sqrt(0.9 * st.max_drag / force)
        fixes.append({"kind": "speed", "title": f"Keep below {fmt(top)} m/s",
                      "summary": "Drag grows with the square of speed at a steady drag coefficient.", "auto": False})
    metrics = [("drag", f"{fmt(force)} N"), ("drag coefficient", f"{cd:.3g}"), ("Reynolds number", fmt(re)),
               ("frontal area", f"{fmt(area * 1e4)} cm2")]
    assumptions = [fl.summary() + ".", f"Drag coefficient from {src}.",
                   "A smooth body in a steady, uniform stream with low turbulence, far from walls and the ground.",
                   f"Frontal area {fmt(area * 1e4)} cm2" + (f" over a span of {fmt(st.length)} m." if st.length else
                                                            (" per metre of span." if st.shape == "cylinder" else "."))]
    values = {"drag_N": force, "cd": cd, "re": re, "area_m2": area, "dynamic_pressure_Pa": q,
              "_figures": [["curve", "The published drag curve with this body marked"]]}
    o = GenericOutcome("flow", name, st.question, verdict, head, metrics, level, reasons, checks, assumptions, values,
                       imgs, fixes, solver="published drag curves", t_total=time.time() - t0)
    if st.max_drag:
        o.criteria = [{"name": "drag", "value": "drag_N", "limit": st.max_drag, "sense": "max"}]
    return o.write(out)
