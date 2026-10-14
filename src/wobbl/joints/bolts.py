"""Bolted joints: preload from torque, then the external load, checked for separation, slip and bolt strength.

"Will these four M8 bolts hold the bracket, and does the joint open?" Wobbl finds each bolt from the holes it passes
through, tightens it to the preload its torque gives (T = K F d, with the nut factor K stated), then applies the
external load. The bolts are springs with VDI 2230's stiffness between rigid rings under the head and nut, pulled to
their preload by a turnbuckle and locked, and the clamped parts press on each other through contact with friction.
Every bolt is then checked the way VDI 2230 checks it: the bolt stress against its proof stress, whether the
interface opens, whether friction holds the shear, the pressure under the head and whether the nut threads strip,
with the hand calculation beside the FEA for the load each bolt takes and the stiffness of what it clamps.
"""
from __future__ import annotations

import math
import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from ..mesh.mesher import Refinement, Sizing, auto_sizing
from ..solvers import calculix_nl as nl
from ..solvers.base import SolverError
from ..structural.build import build_static, tri_area_weights
from ..structural.run import support_reactions
from ..structural.spec import SpecError, StructuralStudy, parse_fixture, parse_load
from ..units import UnitError, fmt, parse_quantity
from ..verify.checks import Check, balance_check, confidence, convergence_check
from ..plastic.common import NonlinearOutcome, clean_work, light_png
from . import setup as S
from . import vdi2230 as V
from .assembly import Assembly, build_assembly_mesh, prepare_assembly

QUALITY = {"fast": 2000, "normal": 5000, "fine": 12000}
REFINE = 0.75
TOL = 0.05
PRELOAD_SHARE = 0.7      # without a torque or preload, bolts are tightened to 70% of their proof load
SLIP_TARGET = 1.2        # VDI 2230's usual safety against slip under static shear


@dataclass
class BoltSpec:
    """What the user says about some bolts. Everything is optional: the size comes from the hole it sits in, the
    class defaults to 8.8, the holes default to every bolt hole no other spec names, and the preload to 70% of the
    proof load."""
    size: str = ""                   # 'M8', 'M10x1.25', or '' for the size the hole fits
    grade: str = "8.8"
    faces: str | None = None         # the holes it goes through
    torque: float | None = None      # N mm
    preload: float | None = None     # N
    washer: float | None = None      # bearing diameter, mm
    text: str = ""

    def bolt_for(self, hole: float) -> V.Bolt:
        size = self.size or f"M{size_for_hole(hole):g}"
        try:
            return V.bolt_from(f"{size} {self.grade}", self.washer)
        except ValueError as e:
            raise SpecError(f"{e} (in bolt '{self.text}')") from None


def size_for_hole(diameter: float) -> int:
    """The nominal metric size a clearance hole is drilled for: the largest standard bolt that passes through it."""
    fits = [d for d in V.PITCH if d < diameter - 0.05]
    if not fits:
        raise SpecError(f"a {fmt(diameter)} mm hole is too small for any metric bolt from M3 up")
    return max(fits)


_SIZE = re.compile(r"\bm(\d+(?:\.\d+)?)(?:\s*x\s*(\d*\.?\d+))?\b", re.I)
_GRADE = re.compile(r"(?<![\w.])(\d{1,2}\.\d|a[24]-[78]0)(?![\w.])", re.I)


def parse_bolt(text: str) -> BoltSpec:
    """'M8 8.8 @ holes d=9 torque 25 N*m', 'M10 10.9 preload 30 kN', '8.8 torque 12 N*m', 'washer 16 @ x>40', or ''.
    Without '@' the spec covers every bolt hole that no other spec names."""
    head, rest = (text.split("@", 1) + [""])[:2] if "@" in text else (text, "")
    sp = BoltSpec(text=text)
    for part in ("head", "rest"):
        s = head if part == "head" else rest
        m = re.search(r"\btorque\s+(.+?)(?=\s+(?:preload|washer)\b|$)", s, re.I)
        if m:
            sp.torque = _quantity(m.group(1), "moment", text)
            s = s[:m.start()] + s[m.end():]
        m = re.search(r"\bpreload\s+(.+?)(?=\s+(?:torque|washer)\b|$)", s, re.I)
        if m:
            sp.preload = _quantity(m.group(1), "force", text)
            s = s[:m.start()] + s[m.end():]
        m = re.search(r"\bwasher\s+(?:d\s*=\s*)?(\d*\.?\d+)\s*(?:mm)?", s, re.I)
        if m:
            sp.washer = float(m.group(1))
            s = s[:m.start()] + s[m.end():]
        if part == "head":
            head = s
        else:
            rest = s
    m = _SIZE.search(head)
    if m:
        sp.size = m.group(0).replace(" ", "")
        head = head[:m.start()] + head[m.end():]
    m = _GRADE.search(head)
    if m:
        sp.grade = m.group(1).lower()
        head = head[:m.start()] + head[m.end():]
    if sp.grade not in V.GRADES:
        raise SpecError(f"unknown property class {sp.grade} in bolt '{text}'; use one of {', '.join(V.GRADES)}")
    if re.sub(r"\b(bolts?|screws?|and|with)\b", "", head, flags=re.I).strip(" ,"):
        raise SpecError(f"could not read '{head.strip()}' in bolt '{text}'; write it like 'M8 8.8 @ holes d=9 "
                        "torque 25 N*m'")
    sp.faces = rest.strip() or None
    return sp


def _quantity(t: str, kind: str, text: str) -> float:
    try:
        return parse_quantity(t.strip(), kind).value
    except UnitError as e:
        raise SpecError(f"{e} (in bolt '{text}')") from None


@dataclass
class BoltLine:
    """One bolt where it passes through the stack: its axis, the clamp length and where the head and nut sit. A bolt
    screwed into a tapped hole has no nut; the threads it engages in the last part hold it instead."""
    spec: BoltSpec
    bolt: V.Bolt
    axis: np.ndarray         # from the head toward the nut
    head: np.ndarray
    nut: np.ndarray          # the nut's bearing face, or the middle of the engaged thread
    hole: float              # clearance hole diameter, mm
    clamp: float             # mm
    bodies: list[int]        # the clamped parts, and the tapped part last
    holes: list[int]
    tapped: list[int] = field(default_factory=list)   # hole faces of a tapped part
    engagement: float = 0.0  # length of thread engaged in a tapped part, mm
    outer: float = 0.0       # VDI's D'_A, the clamped parts' substitute outside diameter

    @property
    def label(self) -> str:
        return f"{self.bolt.name} at ({', '.join(fmt(v) for v in self.head)})"

    def preload(self, nut_factor: float) -> tuple[float, str]:
        """The preload and how it was set."""
        b, sp = self.bolt, self.spec
        if sp.preload:
            return sp.preload, f"a preload of {fmt(sp.preload / 1000)} kN as given"
        if sp.torque:
            f = sp.torque / (nut_factor * b.d)
            return f, (f"{fmt(sp.torque / 1000)} N m of torque with a nut factor of {nut_factor:g} "
                       f"(T = K F d) gives {fmt(f / 1000)} kN")
        f = PRELOAD_SHARE * b.proof * b.stress_area
        return f, f"no torque was given, so {fmt(f / 1000)} kN, 70% of its proof load, a common tightening target"


def hole_lines(asm: Assembly) -> list[list]:
    """Every run of coaxial holes through the assembly, along x, y or z, that passes through two parts or more."""
    body_of = asm.body_of_face()
    holes = [f for f in asm.features.faces.values() if f.role == "hole" and f.axis is not None]
    groups: list[list] = []
    for h in sorted(holes, key=lambda f: f.id):
        a, p = np.asarray(h.axis, float), np.asarray(h.axis_point, float)
        if np.abs(a).max() < 0.999:
            continue
        for g in groups:
            a0, p0 = np.asarray(g[0].axis, float), np.asarray(g[0].axis_point, float)
            d = p - p0
            if abs(abs(a @ a0) - 1) < 1e-3 and np.linalg.norm(d - (d @ a0) * a0) < 0.2 * min(h.radius, g[0].radius):
                g.append(h)
                break
        else:
            groups.append([h])
    return [g for g in groups if len({body_of[h.id] for h in g}) >= 2]


def bolt_lines(asm: Assembly, specs: list[BoltSpec]) -> list[BoltLine]:
    """One bolt for each run of coaxial holes through two parts or more. Specs that name holes claim the runs they
    touch; a spec without holes covers the rest; with no specs at all every run gets an 8.8 bolt sized to its hole."""
    runs = hole_lines(asm)
    owner: dict[int, BoltSpec] = {}
    for sp in [s for s in specs if s.faces]:
        picked = set(S.select(asm, sp.faces).faces)
        hit = [i for i, g in enumerate(runs) if picked & {h.id for h in g}]
        if not hit:
            raise SpecError(f"bolt '{sp.text}': '{sp.faces}' picks no hole that runs through two parts along x, y "
                            "or z")
        for i in hit:
            owner.setdefault(i, sp)
    rest = next((s for s in specs if not s.faces), None if specs else BoltSpec(text="default"))
    if rest is not None:
        dia = [2 * max(h.radius for h in g) for g in runs]
        small = min(dia) if dia else 0.0
        for i in range(len(runs)):
            if i in owner:
                continue
            # a run far wider than the narrowest is a bore, like the middle of a flange, not a bolt hole; and a
            # hole a bolt of the size asked for would fall through is not one of its holes either
            too_wide = (len(runs) > 1 and dia[i] > 2 * small) or (rest.size and dia[i] >= rest.bolt_for(dia[i]).dw)
            if too_wide:
                asm.prep.notes.append(f"The {fmt(dia[i])} mm hole through parts "
                                      f"{', '.join(map(str, sorted({asm.body_of_face()[h.id] for h in runs[i]})))} is "
                                      "taken as a bore, not a bolt hole; name it in a bolt to put one there.")
                continue
            owner[i] = rest
    out = [_line(asm, runs[i], sp) for i, sp in sorted(owner.items())]
    for line in out:
        line.outer = outer_diameter(asm, line, out)
    return out


def _line(asm: Assembly, g: list, sp: BoltSpec) -> BoltLine:
    body_of = asm.body_of_face()
    a0 = np.asarray(g[0].axis, float)
    k = int(np.argmax(np.abs(a0)))
    a = np.eye(3)[k]
    c = np.asarray(g[0].axis_point, float)
    c = c - (c @ a) * a                                         # a point on the axis at coordinate 0 along it
    span = {}
    for h in g:
        lo, hi = h.bbox[0][k], h.bbox[1][k]
        b = body_of[h.id]
        s = span.setdefault(b, [lo, hi, 0.0, []])
        s[0], s[1], s[2] = min(s[0], lo), max(s[1], hi), max(s[2], 2 * h.radius)
        s[3].append(h.id)
    clear = max(s[2] for s in span.values())
    bolt = sp.bolt_for(clear)
    # a hole narrower than the bolt is tapped for it; it must be the last part at one end of the stack
    tapped = [b for b, s in span.items() if s[2] < bolt.d - 0.05]
    order = sorted(span, key=lambda b: span[b][0])
    lo, hi = span[order[0]][0], span[order[-1]][1]
    if len(tapped) > 1 or (tapped and tapped[0] not in (order[0], order[-1])):
        raise SpecError(f"the holes at ({', '.join(fmt(v) for v in c)}) narrow below {bolt.name} in more than one "
                        "part, or in the middle of the stack, so there is no head and nut or tapped end to find")
    if tapped:
        t = tapped[0]
        tlo, thi = span[t][0], span[t][1]
        clamped = [b for b in order if b != t]
        if t == order[0]:            # tapped at the low end: head on top, the bolt runs down into it
            head, axis = hi, -a
            eng = min(bolt.d, thi - tlo)
            start, clamp = thi, hi - thi
            nut = (thi - 0.5 * eng)
        else:
            head, axis = lo, a
            eng = min(bolt.d, thi - tlo)
            start, clamp = tlo, tlo - lo
            nut = (tlo + 0.5 * eng)
        del start
        line = BoltLine(sp, bolt, axis, c + head * a, c + nut * a, clear, clamp, clamped + [t],
                        sorted(i for s in span.values() for i in s[3]), span[t][3], eng)
    else:
        line = BoltLine(sp, bolt, -a, c + hi * a, c + lo * a, clear, hi - lo, order,
                        sorted(i for s in span.values() for i in s[3]))
    if line.hole >= bolt.dw:
        raise SpecError(f"the {fmt(line.hole)} mm hole at {line.label} is as wide as the {fmt(bolt.dw)} mm head of "
                        f"{bolt.name}; add a washer, as in 'washer {fmt(line.hole * 1.8)}'")
    return line


def outer_diameter(asm: Assembly, line: BoltLine, lines: list[BoltLine]) -> float:
    """VDI's D'_A: twice the distance from the bolt to the nearest side of the parts it clamps, and no more than the
    distance to the next bolt."""
    sm = asm.surface
    a = line.axis
    n, _, c = sm.tri_geometry()
    clamped = line.bodies[:-1] if line.tapped else line.bodies
    faces = [f for b in clamped for f in asm.body_faces[b] if asm.features.faces[f].role != "hole"]
    m = np.isin(sm.tri_surf, faces) & (np.abs(n @ a) < 0.2)
    rel = c[m] - line.head
    t = rel @ a
    inside = (t >= -1e-6) & (t <= line.clamp + 1e-6)
    radial = np.linalg.norm(rel - np.outer(t, a), axis=1)[inside]
    d = 2 * float(radial.min()) if len(radial) else 3 * line.bolt.dw
    for o in lines:
        if o is not line:
            r = o.head - line.head
            d = min(d, float(np.linalg.norm(r - (r @ a) * a)))
    return max(d, line.hole * 1.05)


def rings(mesh, line: BoltLine) -> tuple[np.ndarray, np.ndarray]:
    """Nodes the head clamps (the flat face around the hole out to the bearing diameter) and the nodes the nut
    clamps, or for a tapped hole the wall of the engaged thread."""
    n, _, c = mesh.tri_geometry()
    a = line.axis
    tol = 1e-3 * max(line.clamp, 1.0)
    out = []
    ends = [(line.head, -1.0)] + ([] if line.tapped else [(line.nut, 1.0)])
    for point, sign in ends:
        on = (np.abs((c - point) @ a) < tol) & ((n @ a) * sign > 0.99)
        nodes = np.unique(mesh.tris[on].ravel())
        rel = mesh.X[nodes] - point
        r = np.linalg.norm(rel - np.outer(rel @ a, a), axis=1)
        nodes = nodes[(r <= 0.5 * line.bolt.dw + 1e-6) & (np.abs(rel @ a) < tol)]
        if len(nodes) < 6:
            raise SpecError(f"found no flat face for the {'head' if sign < 0 else 'nut'} of {line.label}; the hole "
                            "has to come out on a flat face square to it")
        out.append(nodes)
    if line.tapped:
        wall = mesh.surface_mask(line.tapped)
        nodes = np.unique(mesh.tris[wall].ravel())
        t = (mesh.X[nodes] - line.nut) @ a
        nodes = nodes[np.abs(t) <= 0.5 * line.engagement + tol]
        if len(nodes) < 6:
            raise SpecError(f"found no thread to engage for {line.label}")
        out.append(nodes)
    return out[0], out[1]


def spring_of(line: BoltLine) -> dict:
    """The bolt's stiffnesses: axial from VDI 2230, shear and bending of a core section clamped at both ends."""
    b = line.bolt
    L = max(line.clamp, 1e-6)
    I = math.pi * b.d3 ** 4 / 64
    G = b.E / (2 * 1.3)
    return {"axial": 1.0 / V.bolt_compliance(b, L), "lateral": 12 * b.E * I / L ** 3, "bending": b.E * I / L,
            "torsion": G * 2 * I / L}
