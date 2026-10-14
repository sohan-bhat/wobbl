"""Read a plain English question into a study, and say exactly what was assumed.

"will this motor mount crack under 200 N?" is not enough to simulate; the
loaded face, the direction and the supports are missing. Wobbl fills the gaps
with the most common reading (bolted through the holes, load on the face
farthest from them, pushing down) and prints every guess together with the
exact command that would override it. Agents should pass structured flags.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np

from .geometry.faces import PartFeatures
from .materials import LIBRARY, get_material
from .structural.spec import Fixture, Load, StructuralStudy, parse_fixture, parse_load
from .units import parse_quantity

_FORCE = re.compile(r"(\d+(?:\.\d+)?)\s*(kn|n|newtons?|lbf|lbs?|pounds?|kgf|kg|kips?)\b", re.I)
_PRESS = re.compile(r"(\d+(?:\.\d+)?)\s*(mpa|kpa|psi|bar|pa)\b", re.I)
_GRAV = re.compile(r"(\d+(?:\.\d+)?)\s*g'?s?\b(?!\w)", re.I)
_SF = re.compile(r"safety factor (?:of |above |at least |>=? ?)?(\d+(?:\.\d+)?)", re.I)


@dataclass
class Reading:
    domain: str = "strength"
    guesses: list[str] = field(default_factory=list)
    load: str | None = None
    fixture: str | None = None
    material: str | None = None
    target_sf: float | None = None


def detect_domain(question: str, path: str = "") -> str:
    q = question.lower()
    if path.lower().endswith((".cir", ".sp", ".spice", ".net")) or re.search(
            r"\b(led|resistor|ripple|mosfet|regulator|buck|capacitor|circuit|volts?|amps?)\b", q):
        return "circuit"
    if re.search(r"\b(servos?|robot arm|pick and place|linkage|four.?bar|crank|rocker|gear ?(?:box|train|ratio)|"
                 r"pendulum|tip(?:s|ping)? over|topple)\b", q):
        return "mechanism"
    if re.search(r"\b(overheat|heat|hot|temperature|thermal|cool|watts?|\d+\s*w\b|celsius|°c)\b", q):
        return "thermal"
    return "strength"


def find_material(question: str) -> str | None:
    q = question.lower()
    names = []
    for m in LIBRARY.values():
        for n in (m.key, *m.aliases, m.name.lower()):
            if len(n) >= 3 and re.search(r"(?<![\w-])" + re.escape(n) + r"(?![\w-])", q):
                names.append((len(n), m.key))
    if names:
        return max(names)[1]
    if re.search(r"\b(3d[- ]?print|printed|fdm)\b", q):
        return "pla"
    if re.search(r"\bsteel\b", q):
        return "mild-steel"
    if re.search(r"\balumin", q):
        return "6061-t6"
    return None


def _direction(q: str) -> str | None:
    for words, d in ((r"\b(down|downward|downwards|pushing down|presses down|press down|weight|sits on|on top)\b",
                      "down"), (r"\b(up|upward|upwards|lift|lifting|pull up)\b", "up"),
                     (r"\b(sideways|side load|lateral)\b", "+x")):
        if re.search(words, q):
            return d
    return None


def read_question(question: str, features: PartFeatures, study: StructuralStudy) -> Reading:
    """Fill the parts of ``study`` the user did not give explicitly, from the question and the geometry."""
    r = Reading()
    q = question.lower()
    if study.material in (None, "", "auto"):
        mat = find_material(q)
        if mat is None:
            mat = "6061-t6"
            r.guesses.append("material not mentioned: assumed aluminum 6061-T6 (override with --material)")
        r.material = mat
        study.material = mat
    m = _SF.search(q)
    if m and study.target_sf is None:
        study.target_sf = float(m.group(1))
    if not study.fixtures:
        fx = _guess_fixture(q, features)
        r.guesses.append(f"supports not given: {fx[1]} (override with --fix)")
        study.fixtures.append(parse_fixture(fx[0]))
    if not study.loads:
        ld = _guess_load(q, features, study)
        if ld is None:
            raise ValueError("the question has no load I can read. Say it like '200 N', '45 lbf', '3 g' or "
                             "'0.2 MPa', or pass --load '200 N down @ top'")
        r.guesses.append(f"load placement: {ld[1]} (override with --load)")
        study.loads.append(parse_load(ld[0]))
    return r


def _guess_fixture(q: str, f: PartFeatures) -> tuple[str, str]:
    holes = f.holes()
    if re.search(r"\b(bolt|bolted|screw|screwed|mounted through|mounting holes?)\b", q) and holes:
        d = _common_hole(holes)
        return f"bolt holes d={d:.4g}", f"bolted through the {d:.3g} mm holes"
    m = re.search(r"\b(fixed|clamped|glued|welded|bolted|mounted|held)\s+(?:at|on|to)\s+(?:the\s+)?"
                  r"(top|bottom|left|right|front|back|base)\b", q)
    if m:
        side = {"base": "bottom"}.get(m.group(2), m.group(2))
        return f"fixed {side}", f"the {side} face is held still"
    if holes:
        d = _common_hole(holes)
        return f"bolt holes d={d:.4g}", f"bolted through the {d:.3g} mm holes (the most common hole size)"
    return "fixed bottom", "the bottom face is held still"


def _common_hole(holes) -> float:
    """The most common hole size; ties go to the size whose holes sit lowest, since parts usually bolt down."""
    ds = np.round(np.array([2 * h.radius for h in holes]), 2)
    vals, counts = np.unique(ds, return_counts=True)
    best = vals[counts == counts.max()]
    if len(best) == 1:
        return float(best[0])
    low = {v: min(h.axis_point[2] for h, d in zip(holes, ds) if d == v) for v in best}
    return float(min(best, key=lambda v: low[v]))


def _guess_load(q: str, f: PartFeatures, study: StructuralStudy) -> tuple[str, str] | None:
    direction = _direction(q) or "down"
    m = _FORCE.search(q)
    gm = _GRAV.search(q)
    pm = _PRESS.search(q)
    if pm:
        return f"{pm.group(1)} {pm.group(2)} @ top", f"pressure on the top face"
    if gm and not m:
        return f"{gm.group(1)} g {direction}", f"{gm.group(1)} g of acceleration, {direction}"
    if not m:
        return None
    unit = m.group(2).lower()
    nice = {"n": "N", "newton": "N", "newtons": "N", "kn": "kN", "kg": "kgf", "kgs": "kgf", "kgf": "kgf",
            "lb": "lbf", "lbs": "lbf", "lbf": "lbf", "pound": "lbf", "pounds": "lbf", "kip": "kip", "kips": "kip"}
    mag = f"{m.group(1)} {nice.get(unit, unit)}"
    sel, why = _far_face(f, study)
    if re.search(r"\b(on|at) (the )?top\b", q):
        sel, why = "top", "the top face"
    elif re.search(r"\b(tip|end|free end)\b", q):
        sel, why = _far_face(f, study)
    return f"{mag} {direction} @ {sel}", f"{mag} pushing {direction} on {why}"


def _far_face(f: PartFeatures, study: StructuralStudy) -> tuple[str, str]:
    """The flat face farthest from the supports: the free end of a cantilevered part."""
    flats = [x for x in f.faces.values() if x.kind == "plane" and x.area > 0.01 * sum(y.area for y in f.faces.values())]
    if not flats:
        return "top", "the top face"
    sup = []
    for fx in study.fixtures:
        if "hole" in fx.faces:
            sup += [np.asarray(h.axis_point) for h in f.holes()]
    if not sup:
        lo = np.asarray(f.bbox[0])
        sup = [lo]
    c = np.mean(sup, axis=0)
    best = max(flats, key=lambda x: float(np.linalg.norm(np.asarray(x.centroid) - c)))
    return f"face {best.id}", f"face {best.id}, the flat face farthest from the supports"


def gravity_in_question(q: str) -> bool:
    return bool(re.search(r"\b(own weight|self weight|gravity|drop)\b", q.lower()))


def force_text(value: str) -> float:
    return parse_quantity(value, "force").value


__all__ = ["read_question", "detect_domain", "find_material", "Reading", "get_material", "Fixture", "Load"]
