"""Validation cases past yield and between touching parts: plastic collapse, Hertz contact, friction and bolted
joints, each against an exact or published answer. Registered in cases.py."""
from __future__ import annotations

import math
from pathlib import Path

from . import geometry as G

SY = 250.0
STEEL_PLASTIC = f"custom:E=200000,nu=0.3,yield={SY},uts=400"


def _plastic(part, loads, fixtures, out):
    from ..plastic.study import PlasticStudy, run_plastic

    st = PlasticStudy(str(part), STEEL_PLASTIC, loads, fixtures, quality="fast")
    o = run_plastic(st, out, render_images=False)
    return o, {"confidence": o.confidence, "nodes": o.values["levels"][-1]["n_nodes"],
               "first_yield_factor": round(o.values["first_yield_factor"], 4), "time_s": round(o.t_total, 1)}


def plastic_cantilever(out: Path):
    """A 100 mm steel cantilever, 10 x 10 mm, bent by a moment through a rigid end. Its root is a symmetry plane, so
    the beam is free to narrow and thicken as it yields, and half its width is modeled. It collapses when the whole
    section yields, at the fully plastic moment sigma_y b h^2 / 4, 1.5 times the moment of first yield."""
    L, b, h = 100.0, 10.0, 10.0
    part = G.box(out / "bar.step", L, b / 2, h)
    mp = SY * b * h ** 2 / 4                       # N mm, whole beam
    m = 0.8 * mp
    o, info = _plastic(part, [f"{m / 2 / 1000} N*m +y at {L},{b / 4},{h / 2} @ x=max rigid"],
                       ["roller x=min", "roller y=min", f"fixed-z x=min and box -1,-1,{h / 2 - 1},1,{b},{h / 2 + 1}"],
                       out / "run")
    return o.values["collapse_factor"] * m / 1000, mp / 1000, info


def plastic_cylinder(out: Path):
    """A thick steel tube, bore 100 mm and outside 200 mm, under internal pressure in plane strain (a quarter, with
    rollers on its cut faces and ends). It collapses when the whole wall yields, at (2 / sqrt 3) sigma_y ln(b / a)
    (Hill, The Mathematical Theory of Plasticity, 1950)."""
    ri, ro, p = 50.0, 100.0, 100.0
    part = G.quarter_tube(out / "tube.step", ri, ro, 20.0)
    o, info = _plastic(part, [f"{p} MPa @ cylinders r={ri:g}"],
                       ["roller x=0", "roller y=0", "roller z=min", "roller z=max"], out / "run")
    return o.values["collapse_factor"] * p, 2.0 / math.sqrt(3.0) * SY * math.log(ro / ri), info
