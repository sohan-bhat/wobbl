"""What the drag on a body should be, from measurements of bodies like it, to judge a CFD drag coefficient against.

A sphere and a long cylinder in cross flow have published drag curves against Reynolds number, so a CFD answer is
compared with the curve itself (Clift, Grace and Weber 1978 for the sphere, Sucker and Brauer 1975 for the cylinder,
with White's end correction for a short one). Any other body is put in a class by how long it is along the stream
compared with the width it presents, and the class gives a band of drag coefficients from Hoerner's collection
(Fluid Dynamic Drag, 1965). A band says only that the answer is plausible for that kind of shape, so it is wide.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from ..units import fmt
from ..verify.checks import Check
from . import correlations as C
from .drag import aspect_factor

CRISIS = (2e5, 5e5)       # the drag crisis of a smooth round body, where the curves fall by up to five times


@dataclass
class Shape:
    kind: str                 # sphere, cylinder, or the class name of anything else
    width: float              # m, the diameter for a sphere or a cylinder, else the width of the same frontal area
    length: float = 0.0       # m, the span of a cylinder
    fineness: float = 0.0     # length along the stream over the width presented
    rounded: bool = False     # a smooth round body, whose boundary layer decides where the flow separates
    fill: float = 0.0         # volume over the volume of the sphere or cylinder it was matched to


def volume(tri: np.ndarray) -> float:
    """Volume enclosed by a closed triangulated surface, from the divergence theorem."""
    return abs(float(np.einsum("tk,tk->t", tri[:, 0], np.cross(tri[:, 1], tri[:, 2])).sum()) / 6.0)


def classify(tri: np.ndarray, area: float) -> Shape:
    """tri is the surface in metres in the stream's frame (flow along +x), area the frontal area in m2."""
    pts = tri.reshape(-1, 3)
    ext = np.ptp(pts, axis=0)
    vol = volume(tri)
    width = math.sqrt(4 * area / math.pi)
    fineness = float(ext[0] / max(width, 1e-300))
    mean = float(ext.mean())
    if np.ptp(ext) < 0.05 * mean:
        fill = vol / (math.pi / 6 * mean ** 3)
        if 0.9 < fill < 1.03:
            return Shape("sphere", mean, 0.0, fineness, True, fill)
    for axis in (1, 2):                    # a cylinder lying across the stream
        a, b = [i for i in range(3) if i != axis]
        if abs(ext[a] - ext[b]) < 0.05 * max(ext[a], ext[b]) and ext[axis] >= 0.9 * ext[a]:
            d = 0.5 * (ext[a] + ext[b])
            fill = vol / (math.pi / 4 * d * d * ext[axis])
            if 0.9 < fill < 1.03:
                return Shape("cylinder", d, float(ext[axis]), fineness, True, fill)
    return Shape(_class(fineness), width, 0.0, fineness, False)


def _class(fineness: float) -> str:
    return ("flat" if fineness < 0.6 else "blunt" if fineness < 1.5 else "elongated" if fineness < 3 else "slender")


# drag coefficient bands on frontal area by class, from Hoerner (1965): a disk or plate facing the stream 1.1 to
# 1.2, a cube 1.05, a short cylinder or a sphere 0.4 to 1.0, an ellipsoid or a car body 0.15 to 0.5, a strut or a
# fuselage down to 0.04, and a long rod across the stream 1.0 to 1.2
BANDS = {"flat": (0.8, 1.6), "blunt": (0.3, 1.5), "elongated": (0.12, 1.2), "slender": (0.04, 1.2)}
NAMES = {"flat": "a flat or dish shaped body facing the stream", "blunt": "a blunt body about as long as it is wide",
         "elongated": "an elongated body", "slender": "a slender body"}


def reference(sh: Shape, re_: float) -> dict:
    """The expected drag coefficient and the band an honest answer falls in, with where they come from."""
    if sh.kind == "sphere":
        cd = C.sphere_cd(re_)
        src = "Clift, Grace and Weber's standard drag curve of a smooth sphere"
        if CRISIS[0] <= re_ <= CRISIS[1]:
            lo, hi = 0.07, 0.5
            src += ", which falls from 0.5 to 0.07 through the drag crisis, so the band is its whole fall"
        else:
            lo, hi = (0.8 * cd, 1.2 * cd) if re_ < CRISIS[0] else (0.65 * cd, 1.35 * cd)
        return {"cd": cd, "lo": lo, "hi": hi, "source": src, "curve": True}
    if sh.kind == "cylinder":
        k = aspect_factor(sh.length / sh.width)
        cd = C.cylinder_cd(min(re_, 2e5)) * k
        src = "Sucker and Brauer's fit for a long smooth cylinder in cross flow" + (
            f" times {k:.2f} for its free ends (White)" if k < 1 else "")
        if re_ > CRISIS[0]:
            lo, hi = 0.3 * k, 1.3 * k
            src += ", which past Re 2e5 scatters from 0.3 to 1.3 with roughness and free stream turbulence"
        else:
            lo, hi = 0.75 * cd, 1.25 * cd
        return {"cd": cd, "lo": lo, "hi": hi, "source": src, "curve": True}
    lo, hi = BANDS[sh.kind]
    src = "Hoerner's drag collection"
    if re_ < 1e3:
        lo, hi = 0.5 * lo, max(hi, 24 * 3 / max(re_, 1.0))
        src += ", widened because drag climbs steeply as Re falls below 1000"
    return {"cd": math.sqrt(lo * hi), "lo": lo, "hi": hi, "source": src, "curve": False}


def check(sh: Shape, re_: float, cd: float, unsteady: str = "") -> Check:
    """Judge a computed drag coefficient (corrected for the box's blockage) against the reference for its body."""
    ref = reference(sh, re_)
    lo, hi = ref["lo"], ref["hi"]
    what = {"sphere": "a sphere", "cylinder": "a cylinder across the stream"}.get(sh.kind) or NAMES[sh.kind]
    detail = {"cd": cd, "cd_reference": ref["cd"], "band": [lo, hi], "class": sh.kind, "re": re_,
              "use_reference": False}
    if ref["curve"]:
        vs = f"{100 * (cd / ref['cd'] - 1):+.0f}% against the {ref['cd']:.3g} of "
    else:
        vs = f"against a band of {lo:.2g} to {hi:.2g} from "
    head = f"drag coefficient {cd:.3g} at Re {fmt(re_)} on {what}, {vs}{ref['source']}"
    if ref["curve"]:
        head += f", band {lo:.2g} to {hi:.2g}"
    if lo <= cd <= hi:
        return Check("drag reference", "pass", head + ", inside the band", detail)
    off = cd / hi if cd > hi else lo / max(cd, 1e-300)
    status = "warn" if off < 1.25 else "fail"
    # a smooth sphere or cylinder that reads low has a curve to fall back on, which the study reports instead
    detail["use_reference"] = bool(status == "fail" and ref["curve"] and cd < lo and sh.rounded)
    return Check("drag reference", status, head + f", outside it. {_why(sh, re_, cd < lo, unsteady)}", detail)


def _why(sh: Shape, re_: float, low: bool, unsteady: str) -> str:
    if not low:
        return ("The flow holds more drag than bodies of this kind do, so check the speed and the size, the box "
                "blockage, and whether the grid is too coarse to resolve the wake")
    if sh.rounded:
        how = "a time accurate run" if unsteady else "a steady run"
        return (f"On a smooth round body below Re 3e5 the flow separates early from a laminar boundary layer, and "
                f"{how} with wall functions on a grid that cannot resolve that layer keeps the flow attached too "
                "long, which narrows the wake and reads the drag low. Treat the drag as a lower bound and use the "
                "published curve")
    return ("The wake is narrower than bodies of this kind have, so the flow separated too late or the grid is too "
            "coarse to hold the separation, and the drag is a lower bound")
