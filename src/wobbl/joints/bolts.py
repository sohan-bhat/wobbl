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


_TORQUE = re.compile(r"(\d+(?:\.\d+)?)\s*(n\s*\*?\s*m|nm|n·m|kn\s*\*?\s*m|lbf?[\s*-]*ft|ft[\s*-]*lbf?|"
                     r"in[\s*-]*lbf?|lbf?[\s*-]*in)\b", re.I)
_PRELOAD = re.compile(r"preload(?:ed)?\s+(?:of\s+|to\s+)?(\d+(?:\.\d+)?\s*(?:k?n|lbf?|kips?))\b", re.I)


def bolt_from_words(text: str) -> str:
    """A bolt spec read from plain words, as 'four M8 bolts at 20 N m' gives 'M8 torque 20 N*m', or ''."""
    m = _SIZE.search(text or "")
    if not m:
        return ""
    spec = m.group(0).replace(" ", "")
    g = _GRADE.search(text)
    if g and g.group(1).lower() in V.GRADES:
        spec += f" {g.group(1).lower()}"
    t = _TORQUE.search(text)
    if t:
        unit = re.sub(r"[\s*·-]+", "", t.group(2).lower())
        unit = {"nm": "N*m", "knm": "kN*m", "lbft": "lbf*ft", "lbfft": "lbf*ft", "ftlb": "lbf*ft", "ftlbf": "lbf*ft",
                "inlb": "in*lbf", "inlbf": "in*lbf", "lbin": "in*lbf", "lbfin": "in*lbf"}.get(unit, "N*m")
        spec += f" torque {t.group(1)} {unit}"
    p = _PRELOAD.search(text)
    if p:
        spec += f" preload {p.group(1)}"
    return spec


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


@dataclass
class BoltStudy:
    part: str
    material: str
    bolts: list[str] = field(default_factory=list)
    loads: list[str] = field(default_factory=list)
    fixtures: list[str] = field(default_factory=list)
    friction: float = 0.15           # between the clamped parts (VDI 2230: 0.1 to 0.23 for dry steel)
    nut_factor: float = 0.2          # K in T = K F d
    body_materials: dict = field(default_factory=dict)
    path: list[float] = field(default_factory=list)    # extra load multiples to save on the way (load path)
    quality: str = "normal"
    question: str = ""
    scale: float = 1.0
    params: dict = field(default_factory=dict)
    keep: bool = False
    playback: bool = False           # also write playback.json, tightening and loading increment by increment

    def to_dict(self) -> dict:
        return asdict(self)


def run_bolts(study: BoltStudy, out_dir: str | Path, progress=None, asm: Assembly | None = None,
              render_images: bool = True) -> NonlinearOutcome:
    t0 = time.time()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    work = out / "work"
    say = progress or (lambda a, b: None)
    if asm is None:
        say("geometry", f"reading {Path(study.part).name} and finding the faces that touch")
        asm = prepare_assembly(study.part, work, (), study.params, study.scale)
    lines = bolt_lines(asm, [parse_bolt(b) for b in study.bolts])
    if not lines:
        raise SpecError("no hole runs through two parts along x, y or z, so there is no bolt to put in. Bolts go "
                        "through coaxial holes in the parts they clamp, or into a tapped hole in the last one")
    base = StructuralStudy(str(study.part), study.material, [parse_load(x) for x in study.loads],
                           [parse_fixture(x) for x in study.fixtures])
    loads, fixtures, _ = S.resolve(asm, base.loads, base.fixtures)
    levels, last = [], None
    for k, sizing in enumerate(level_sizes(asm, lines, study)):
        say("mesh", f"level {k}: element size {sizing.h:.3g} mm")
        mesh = build_assembly_mesh(asm, sizing)
        say("solve", f"level {k}: tightening {len(lines)} bolt{'s' if len(lines) != 1 else ''}, then the load, on "
                     f"{mesh.n_nodes:,} nodes")
        r = solve_level(asm, study, lines, mesh, loads, fixtures, work / f"level{k}")
        levels.append({"level": k, "n_nodes": mesh.n_nodes, "h": sizing.h,
                       "bolt_force": max(b["force"] for b in r["bolts"]),
                       "clamp_force": min(b["clamp"] for b in r["bolts"]), "increments": r["res"].increments,
                       "iterations": r["res"].iterations, "t_solve": r["res"].t_solve})
        say("check", f"level {k}: worst bolt {levels[-1]['bolt_force'] / 1000:.4g} kN, least clamp "
                     f"{levels[-1]['clamp_force'] / 1000:.4g} kN")
        last = r
    o = evaluate(study, asm, lines, last, levels)
    o.t_total = time.time() - t0
    if render_images:
        say("render", "drawing the contact pressure between the clamped parts")
        o.images = render(out / "images", last)
    if study.playback:
        o.images["playback"] = playback(out / "playback.json", study, last)
    o.notes = list(asm.prep.notes)
    o.write(out)
    clean_work(work, study.keep)
    return o


def level_sizes(asm: Assembly, lines: list[BoltLine], study: BoltStudy) -> list[Sizing]:
    """Two meshes, finer around each head and nut so the rings and the cone under them are resolved."""
    s0 = auto_sizing(asm.prep, thickness=asm.features.thickness_typical,
                     max_nodes=QUALITY.get(study.quality, QUALITY["normal"]))
    out = []
    for k in range(2):
        h = s0.h * REFINE ** k
        refs = []
        for ln in lines:
            # about two elements across the ring under the head: the ring is rigid, so finer only adds nodes
            size = min(h, max(0.5 * (ln.bolt.dw - ln.hole), 0.2 * ln.bolt.dw)) * REFINE ** k
            refs.append(Refinement(size, points=[tuple(ln.head), tuple(ln.nut)], radius=0.5 * ln.bolt.dw,
                                   grow=0.5 * ln.bolt.dw, label=f"bolt {ln.label}"))
        out.append(Sizing(h=h, refinements=refs))
    return out


def solve_level(asm, study, lines, mesh, loads, fixtures, work: Path) -> dict:
    notes: list[str] = []
    mats, mat_of = S.materials(mesh, study.material, {int(b): m for b, m in study.body_materials.items()})
    built = build_static(mesh, loads, fixtures, mat_of[min(mat_of)], asm.features.faces)
    sm = built.model
    bolts, how = [], []
    # a ring node may not also be held by a support, a rigid load or another bolt: CalculiX can tie a node only once
    taken = set()
    for c in [*sm.constraints, *sm.normal_constraints, *sm.rigid]:
        taken.update(int(i) for i in c.nodes)
    for i, ln in enumerate(lines):
        ends, ground = [], []
        for ring in rings(mesh, ln):
            free = np.array([n for n in ring if int(n) not in taken], dtype=np.int64)
            # an end on a face that is held already is held with it, as a nut on a fixed face would be
            ground.append(len(free) < max(3, len(ring) // 2))
            ends.append(free if not ground[-1] else ring[:0])
            taken.update(int(n) for n in free)
        f, why = ln.preload(study.nut_factor)
        bolts.append(nl.BoltSpring(f"B{i + 1}", ends[0], ends[1], ln.head, ln.nut, preload=f,
                                   grounded=(ground[0], ground[1]), **spring_of(ln)))
        how.append(why)
    held = {b for ln in lines for b in ln.bodies}
    floating = [b for b in S.floating_bodies(mesh, sm) if b not in held]
    springs = S.springs_for(mesh, floating, mat_of)
    pairs = S.contact_pairs(asm, mesh, asm.pairs, mat_of, study.friction, notes)
    every = study.playback
    steps = [nl.Step(0.0, preload=True, first=0.5 if not every else 0.25, largest=0.5 if not every else 0.25,
                     smallest=1e-5, every_increment=every)]
    for k, f in enumerate(sorted({*[x for x in study.path if 0 < x < 1], 1.0})):
        steps.append(nl.Step(f, lock=k == 0, first=0.25, largest=0.5 if not every else 0.2, smallest=1e-5,
                             every_increment=every))
    model = nl.NonlinearModel(sm, mats, steps, contact=pairs, springs=springs, bolts=bolts, fields=("U", "RF", "S"))
    try:
        res = nl.solve_nonlinear(model, work, "joint")
    except SolverError as e:
        raise SolverError(str(e), "If the clamped parts slide under the load, friction cannot hold it: check the "
                          "slip safety with fewer load or more preload") from None
    return results(asm, lines, mesh, built, pairs, bolts, springs, floating, mat_of, res, notes, how, study)


def results(asm, lines, mesh, built, pairs, bolts, springs, floating, mat_of, res, notes, how, study) -> dict:
    frames = res.frames
    ip = int(np.argmin([abs(float(fr["time"]) - 1.0) for fr in frames]))   # the end of tightening
    pre, fin = frames[ip], frames[-1]
    per = []
    for i, (ln, b) in enumerate(zip(lines, bolts)):
        forces = res.bolt_force[b.name]
        travel = nl.turnbuckle_travel(b, res.bolt_ids, pre["U_extra"], mesh.n_nodes)   # how far it pulled at preload
        fv = float(forces[ip])
        dp_fea = travel / fv - 1.0 / b.axial if fv > 0 else float("nan")
        clamp, area, slide, opened = _interface(mesh, pairs, asm, ln, fin, lines, _totals(res, fin))
        clamp0, _, _, _ = _interface(mesh, pairs, asm, ln, pre, lines, _totals(res, pre))
        per.append({"label": ln.label, "preload_target": b.preload, "preload": fv, "force": float(forces[-1]),
                    "path": [float(x) for x in forces], "delta_P_fea": dp_fea, "clamp": clamp, "clamp_preload": clamp0,
                    "contact_area_mm2": area, "sliding_share": slide, "open_share": opened, "how": how[i],
                    "clamp_length": ln.clamp, "outer": ln.outer, "hole": ln.hole})
    E_parts = [_parts_modulus(ln, mat_of) for ln in lines]
    spring_force = np.zeros(3)
    for nodes, kk in springs:
        spring_force += -kk * fin["U"][nodes].sum(0)
    # a bolt end held with a face that is held passes its force straight to ground, so it is a support too
    ground_f, ground_m = np.zeros(3), np.zeros(3)
    for b in bolts:
        ids = res.bolt_ids[b.name]
        for k, ((ref, rot), held, at) in enumerate(zip((("H", "HR"), ("N", "NR")), b.grounded,
                                                       (b.head_point, b.nut_point))):
            if held and "RF_extra" in fin:
                f = fin["RF_extra"][ids[ref] - mesh.n_nodes].copy()
                if k == 1:
                    # the shank ends on the turnbuckle, whose reaction CalculiX does not write: the ground holds the
                    # nut end against the bolt's tension
                    f += float(res.bolt_force[b.name][-1]) * b.axis
                ground_f += f
                ground_m += np.cross(np.asarray(at, float), f) + fin["RF_extra"][ids[rot] - mesh.n_nodes]
    # gross slip of each interface: CalculiX's shear total against friction on its normal total
    slip = {}
    for cp in pairs:
        rows = [x for x in res.contact_totals.get(cp.name, []) if abs(x["time"] - fin["time"]) < 1e-6]
        if rows and cp.mu > 0 and rows[-1]["normal_force"] > 0:
            slip[cp.name] = rows[-1]["shear_force"] / (cp.mu * rows[-1]["normal_force"])
    return {"slip": slip, "mesh": mesh, "built": built, "frames": frames, "res": res, "bolts": per, "lines": lines,
            "pairs": pairs, "mat_of": mat_of, "notes": notes, "floating": floating, "spring_force": spring_force,
            "E_parts": E_parts, "vm": nl.von_mises(fin["S"]), "ground_force": ground_f, "ground_moment": ground_m}


def _totals(res, fr) -> dict:
    """CalculiX's own normal force on each contact pair at a frame's time."""
    out = {}
    for name, rows in res.contact_totals.items():
        hit = [r for r in rows if abs(r["time"] - fr["time"]) < 1e-6]
        if hit:
            out[name] = hit[-1]["normal_force"]
    return out


def _interface(mesh, pairs, asm, ln: BoltLine, fr, lines: list[BoltLine],
               totals: dict | None = None) -> tuple[float, float, float, float]:
    """Clamp force, contact area, sliding share and open share at the interface of a bolt's stack that holds least,
    over the part of each interface nearer this bolt than any other (all of it for a lone bolt). The force is
    CalculiX's total for the pair, shared out by the nodal pressures: summing nodal pressures over quadratic faces
    alone is off by a few percent."""
    body_of = asm.body_of_face()
    pres_all = np.clip(fr.get("pressure", np.zeros(mesh.n_nodes)), 0, None)
    shear_all = fr.get("shear", np.zeros(mesh.n_nodes))

    def radial(line):
        rel = mesh.X - line.head
        return np.linalg.norm(rel - np.outer(rel @ line.axis, line.axis), axis=1)

    mine = radial(ln)
    nearest = np.ones(mesh.n_nodes, bool)
    for o in lines:
        if o is not ln and abs(abs(o.axis @ ln.axis) - 1) < 1e-6:
            nearest &= mine <= radial(o)
    best = None
    for cp in pairs:
        bs = body_of.get(int(np.unique(mesh.tri_surf[cp.slave])[0]))
        bm = body_of.get(int(np.unique(mesh.tri_surf[cp.master])[0]))
        if bs not in ln.bodies or bm not in ln.bodies:
            continue
        w, _ = tri_area_weights(mesh, cp.slave)
        near = (w > 0) & nearest
        p, tau, ww = pres_all[near], shear_all[near], w[near]
        touching = p > 1e-6 * max(float(p.max()) if len(p) else 0.0, 1e-30)
        area = float(ww[touching].sum())
        force = float(ww @ p)
        whole = float(w[w > 0] @ pres_all[w > 0])
        if totals and cp.name in totals and whole > 0:
            force *= max(totals[cp.name], 0.0) / whole
        row = (force, area,
               float(ww[touching & (tau >= 0.95 * cp.mu * p)].sum()) / max(area, 1e-30) if cp.mu > 0 else 0.0,
               float(ww[~touching].sum()) / max(float(ww.sum()), 1e-30))
        if best is None or row[0] < best[0]:
            best = row
    return best or (0.0, 0.0, 0.0, 1.0)


def evaluate(study: BoltStudy, asm: Assembly, lines: list[BoltLine], r: dict, levels: list[dict]) -> NonlinearOutcome:
    mesh, built, sm = r["mesh"], r["built"], r["built"].model
    fin = r["frames"][-1]
    checks: list[Check] = []
    extra: list[tuple[str, str]] = []
    checks.append(convergence_check([L["bolt_force"] for L in levels], TOL, "force in the most loaded bolt", "N",
                                    min_levels=2))
    clamps = [L["clamp_force"] for L in levels]
    if len(clamps) > 1:
        ch = abs(clamps[-1] - clamps[-2]) / max(abs(clamps[-1]), max(b["preload"] for b in r["bolts"]) * 0.05)
        checks.append(Check("contact convergence", "pass" if ch <= TOL else "warn",
                            f"least clamp force left at an interface {' -> '.join(fmt(c) for c in clamps)} N over "
                            f"two meshes, last change {100 * ch:.1f}% (of the preload when it is near zero)",
                            {"clamp": clamps}))
    F, M = sm.applied_total()
    R, RM = support_reactions(built, SimpleNamespace(RF=fin["RF"]), mesh)
    R, RM = R + r["ground_force"], RM + r["ground_moment"]
    if np.linalg.norm(F) > 0:
        c = balance_check(F + r["spring_force"], R, M, RM, asm.prep.diag)
        checks.append(c)
    worst_pre = max(abs(b["preload"] - b["preload_target"]) / b["preload_target"] for b in r["bolts"])
    checks.append(Check("preload", "pass" if worst_pre < 0.005 else "warn",
                        f"after tightening the bolts carry {', '.join(fmt(b['preload'] / 1000) for b in r['bolts'])} "
                        f"kN against the {', '.join(fmt(b['preload_target'] / 1000) for b in r['bolts'])} kN asked, "
                        f"within {100 * worst_pre:.2g}%", {"error": worst_pre}))
    hand = []
    for ln, b, (fa, fq) in zip(lines, r["bolts"], group_loads(lines, F, M)):
        tapped = r["mat_of"][ln.bodies[-1]] if ln.tapped else None
        hand.append(V.check(ln.bolt, ln.clamp, ln.outer, r["E_parts"][lines.index(ln)], b["preload"], fa, fq,
                            study.friction, engagement=ln.engagement or None,
                            thread_strength=tapped.tensile_strength if tapped else None))
    # stiffness of the clamped parts: the FEA's against VDI's cone
    errs = [(b["delta_P_fea"] / h.delta_P - 1, b, h) for b, h in zip(r["bolts"], hand) if h.delta_P > 0]
    e, b, h = max(errs, key=lambda x: abs(x[0]))
    s = (f"the parts under {b['label']} squeeze by {fmt(1e6 * b['delta_P_fea'])} um per kN in the FEA against "
         f"{fmt(1e6 * h.delta_P)} um per kN through VDI 2230's cone (tan phi {h.tan_phi:.3g}, outside diameter "
         f"{fmt(h.outer_diameter)} mm), {100 * e:+.0f}%")
    checks.append(Check("clamped stiffness", "pass" if abs(e) <= 0.25 else "info",
                        s + ("" if abs(e) <= 0.25 else ". The cone is a hand model; the FEA follows the real shape"),
                        {"fea": b["delta_P_fea"], "vdi": h.delta_P}))
    shares = [(b["force"] - b["preload"], h.phi * h.axial, b, h) for b, h in zip(r["bolts"], hand)]
    loaded = [x for x in shares if x[3].axial > 0.05 * x[2]["preload"]]
    if loaded:
        got, want, b, h = max(loaded, key=lambda x: x[3].axial)
        ratio = got / want if want else float("inf")
        st = "pass" if 0.7 <= ratio <= 1.3 else "info"
        checks.append(Check("hand calculation", st,
                            f"VDI 2230 gives {b['label']} {fmt(h.axial)} N of the pull and {fmt(want)} N more bolt "
                            f"force (load factor {h.phi:.3g}); the FEA bolt gains {fmt(got)} N ({ratio:.2f}x)"
                            + ("" if st == "pass" else
                               (". Less, because the load enters the parts away from the head and nut, nearer the "
                                "interface (VDI's load introduction factor below 1)" if ratio < 1 else
                                ". More, because the flange bends and pries, which the hand split does not see")
                               + "; the FEA follows where the load really enters, so it governs"),
                            {"fea_N": got, "vdi_N": want}))
    else:
        checks.append(Check("hand calculation", "info", "the external load barely pulls along the bolts, so the "
                            "bolt forces stay at their preload as VDI 2230 has it"))
    # design checks on the worst bolt
    sp = [(b["force"] / ln.bolt.stress_area / ln.bolt.proof, b, ln) for b, ln in zip(r["bolts"], lines)]
    use, b, ln = max(sp, key=lambda x: x[0])
    st = "pass" if use <= 0.9 else ("warn" if use <= 1.0 else "fail")
    checks.append(Check("bolt strength", st, f"the most loaded bolt, {b['label']}, carries {fmt(b['force'] / 1000)} "
                        f"kN, {fmt(b['force'] / ln.bolt.stress_area)} MPa on its {fmt(ln.bolt.stress_area)} "
                        f"mm2 stress area, {100 * use:.0f}% of the {fmt(ln.bolt.proof)} MPa proof stress",
                        {"proof_use": use}))
    tight = max(hand, key=lambda h: h.torsion_use)
    checks.append(Check("tightening", "pass" if tight.torsion_use <= 0.9 else "warn",
                        f"while tightening, tension and thread torque together reach {100 * tight.torsion_use:.0f}% of "
                        f"the yield stress of {tight.bolt} (VDI 2230 allows 90%)", {"use": tight.torsion_use}))
    opened = [b for b in r["bolts"] if b["clamp"] <= 0.02 * b["preload"] or b["contact_area_mm2"] <= 0]
    least = min(r["bolts"], key=lambda b: b["clamp"])
    hsep = hand[r["bolts"].index(least)]
    s = (f"{fmt(least['clamp'] / 1000)} kN of the {fmt(least['clamp_preload'] / 1000)} kN clamp is left at the "
         f"interface under {least['label']}, where VDI 2230 has {fmt(hsep.clamp_force / 1000)} kN and the joint "
         f"opening at {fmt(hsep.separation_load / 1000)} kN of pull on that bolt")
    checks.append(Check("separation", "fail" if opened else ("warn" if least["clamp"] < 0.2 * least["preload"]
                                                             else "pass"),
                        s + (": it opens" if opened else ""), {"clamp_N": least["clamp"]}))
    use_f = max(r["slip"].values(), default=0.0)
    slip_fea = [ln.label for ln in lines] if use_f >= 0.95 else []
    hs = min(hand, key=lambda h: h.slip_safety)
    if any(h.shear > 0 for h in hand) or slip_fea:
        st = "fail" if (slip_fea or hs.slip_safety < 1) else ("warn" if hs.slip_safety < SLIP_TARGET else "pass")
        s = (f"friction ({study.friction:g}) on the clamp force holds {fmt(hs.slip_capacity)} N against "
             f"{fmt(hs.shear)} N of shear at {hs.bolt}, a slip safety of {hs.slip_safety:.2f} (VDI 2230 wants "
             f"{SLIP_TARGET:g}); in the FEA the most loaded interface carries {100 * use_f:.0f}% of the shear "
             "friction can hold" + (", so it slides" if slip_fea else ""))
        checks.append(Check("slip", st, s, {"slip_safety": hs.slip_safety, "fea_friction_use": use_f}))
    mat_parts = min((r["mat_of"][x] for ln_ in lines for x in ln_.bodies), key=lambda m: m.tensile_strength)
    p_lim = 1.2 * mat_parts.tensile_strength
    head_p = max(b["force"] / (math.pi / 4 * (ln.bolt.dw ** 2 - ln.hole ** 2)) for b, ln in zip(r["bolts"], lines))
    checks.append(Check("head pressure", "pass" if head_p <= p_lim else "fail",
                        f"the head and nut press {fmt(head_p)} MPa on the {mat_parts.name}, against about "
                        f"{fmt(p_lim)} MPa it takes before it sinks (1.2 times its tensile strength, near VDI 2230's "
                        "limiting surface pressures)", {"pressure_MPa": head_p, "limit_MPa": p_lim}))
    hs_, ls_ = min(zip(hand, lines), key=lambda x: x[0].strip_safety)
    where_ = (f"the thread tapped {fmt(ls_.engagement)} mm deep in the {r['mat_of'][ls_.bodies[-1]].name}"
              if ls_.tapped else "a standard nut's thread")
    checks.append(Check("thread stripping", "pass" if hs_.strip_safety >= 1 else "fail",
                        f"{where_} shears at {hs_.strip_safety:.2g} times the load that breaks {hs_.bolt}, so "
                        + ("the bolt would break first" if hs_.strip_safety >= 1 else "it strips before the bolt "
                           "breaks; engage more thread (about 1.5 d in aluminum) or use a thread insert"),
                        {"strip_safety": hs_.strip_safety}))
    if r["floating"]:
        share = float(np.linalg.norm(r["spring_force"])) / max(float(np.linalg.norm(F)), 1e-12)
        checks.append(Check("stabilizing springs", "pass" if share < 0.005 else "warn",
                            f"soft springs hold bod{'ies' if len(r['floating']) > 1 else 'y'} "
                            f"{', '.join(map(str, r['floating']))} and carry {100 * share:.2g}% of the load"))
    level, reasons = confidence([c for c in checks if c.name not in DESIGN], extra)
    verdict = "holds"
    if any(c.status == "fail" for c in checks if c.name in ("bolt strength", "head pressure", "thread stripping")):
        verdict = "fails"
    elif opened:
        verdict = "separates"
    elif slip_fea or hs.slip_safety < 1:
        verdict = "slips"
    elif any(c.status == "warn" for c in checks if c.name in DESIGN):
        verdict = "marginal"
    n = len(lines)
    b0 = lines[0]
    head = (f"{n} {b0.bolt.name} bolt{'s' if n != 1 else ''} tightened to {fmt(r['bolts'][0]['preload'] / 1000)} kN "
            f"({r['bolts'][0]['how'].split(' gives ')[0] if ' gives ' in r['bolts'][0]['how'] else 'as given'}). "
            f"Under the load the worst bolt carries {fmt(b['force'] / 1000)} kN, {100 * use:.0f}% of its proof load, "
            f"and {fmt(least['clamp'] / 1000)} kN of clamp is left at the interface.")
    head += {"fails": " A bolt or what it clamps is overloaded.", "separates": " The joint opens.",
             "slips": " The clamped parts slide.", "marginal": " It holds, with little margin.",
             "holds": " It holds without opening or slipping."}[verdict]
    if level != "high":
        head += f" Confidence is {level}."
    metrics = [("worst bolt", f"{fmt(b['force'] / 1000)} kN"), ("proof use", f"{100 * use:.0f}%"),
               ("least clamp", f"{fmt(least['clamp'] / 1000)} kN"), ("slip safety", f"{hs.slip_safety:.2f}"
                                                                     if np.isfinite(hs.slip_safety) else "no shear"),
               ("preload", f"{fmt(r['bolts'][0]['preload'] / 1000)} kN")]
    assumptions = _assumptions(study, lines, r)
    o = NonlinearOutcome("bolt", str(study.part), study.question, verdict, head, metrics, level, reasons, checks,
                         assumptions, solver=r["res"].solver)
    o.values = {"bolts": r["bolts"], "vdi": [h.to_dict() for h in hand], "levels": levels, "proof_use": use,
                "least_clamp_N": least["clamp"], "slip_safety": hs.slip_safety,
                "load_steps": [fr["time"] for fr in r["frames"]],
                "_figures": [["pressure", "Contact pressure between the clamped parts under the load"],
                             ["stress", "von Mises stress under the load"]],
                "_tables": [["Bolts", ["bolt", "preload kN", "under load kN", "clamp left kN", "VDI clamp kN",
                                       "VDI separation kN", "slip safety"],
                             [[b["label"], fmt(b["preload"] / 1000), fmt(b["force"] / 1000), fmt(b["clamp"] / 1000),
                               fmt(h.clamp_force / 1000), fmt(h.separation_load / 1000),
                               f"{h.slip_safety:.2f}" if np.isfinite(h.slip_safety) else "no shear"]
                              for b, h in zip(r["bolts"], hand)]]]}
    o.fixes = fixes(verdict, study, lines, r, hand, use)
    return o


DESIGN = ("bolt strength", "tightening", "separation", "slip", "head pressure", "thread stripping")


def _assumptions(study, lines, r) -> list[str]:
    b = lines[0].bolt
    src = V.GRADE_SOURCE["stainless" if b.grade.startswith("a") else "steel"]
    a = [r["bolts"][0]["how"].capitalize() + ". Torque tightening scatters the preload by about 25% either way, so "
         "a torque wrench reading is not a measured preload.",
         f"{b.name}: stress area {fmt(b.stress_area)} mm2, proof stress {fmt(b.proof)} MPa, yield "
         f"{fmt(b.yield_strength)} MPa ({src}); head and nut bear on a {fmt(b.dw)} mm ring "
         "(ISO 4014 and ISO 4032).",
         "Each bolt is a spring with VDI 2230's stiffness of head, thread and nut, between rigid rings under its head "
         "and nut, with the bending and shear stiffness of its core. The shank does not touch the hole, so once the "
         "parts slide nothing but friction and the bolt's bending holds them.",
         f"The clamped parts touch through penalty contact with friction {study.friction:g} (VDI 2230 lists 0.1 to "
         "0.23 for dry steel and 0.1 to 0.28 for aluminum). Embedding of the surfaces, which loses a few percent of "
         "preload, is not modeled.",
         "The hand calculation shares the load between bolts as a rigid flange would and takes it into the joint at "
         "the head and nut (load introduction factor 1), which overstates the bolt's extra load and understates "
         "the load that opens the joint.",
         "Materials are linear elastic and deflections small."]
    return a + r["notes"]


def fixes(verdict, study, lines, r, hand, use) -> list[dict]:
    out = []
    b0 = lines[0].bolt
    K = study.nut_factor

    def torque(f, b):
        return f"{fmt(K * f * b.d / 1000)} N m"

    if verdict in ("separates", "slips", "marginal"):
        h = min(hand, key=lambda x: (x.clamp_force, x.slip_safety))
        need_sep = (1 - h.phi) * h.axial + 0.2 * h.preload        # keep a fifth of the preload clamped
        need_slip = h.shear * SLIP_TARGET / study.friction + (1 - h.phi) * h.axial if h.shear > 0 else 0.0
        need = max(need_sep, need_slip, h.preload)
        cap = 0.9 * b0.proof * b0.stress_area
        if need <= cap:
            out.append({"kind": "preload", "title": f"Tighten to {fmt(need / 1000)} kN ({torque(need, b0)})",
                        "summary": f"That keeps {fmt(0.2 * h.preload / 1000)} kN clamped under the pull and holds the "
                                   f"shear with a slip safety of {SLIP_TARGET:g}; it is "
                                   f"{100 * need / (b0.proof * b0.stress_area):.0f}% of the proof load of "
                                   f"{b0.name}.", "auto": False})
        bigger = next((d for d in sorted(V.PITCH) if d > b0.d and
                       PRELOAD_SHARE * V.bolt_from(f"M{d} {b0.grade}").proof * V.bolt_from(f"M{d} {b0.grade}").stress_area
                       >= need), None)
        if bigger:
            nb = V.bolt_from(f"M{bigger} {b0.grade}")
            f = PRELOAD_SHARE * nb.proof * nb.stress_area
            out.append({"kind": "bolt", "title": f"Go up to M{bigger} {b0.grade.upper()}",
                        "summary": f"At 70% of its proof load, {fmt(f / 1000)} kN ({torque(f, nb)}), it clamps "
                                   f"enough with room to spare; the holes grow to {fmt(nb.hole)} mm.", "auto": False})
        more = math.ceil(len(lines) * need / max(h.preload, 1e-9))
        if more > len(lines):
            out.append({"kind": "bolts", "title": f"Use {more} bolts instead of {len(lines)}",
                        "summary": "More bolts at the same preload share the pull and the shear; keep them close to "
                                   "where the load comes in.", "auto": False})
        if verdict == "slips" or (h.shear > 0 and h.slip_safety < SLIP_TARGET):
            out.append({"kind": "dowel", "title": "Take the shear with dowel pins or a shoulder",
                        "summary": "A fitted pin or a step in the parts carries the shear in bearing, so the bolts "
                                   "only clamp.", "auto": False})
    if verdict == "fails" and use > 0.9:
        up = {"4.6": "8.8", "5.6": "8.8", "8.8": "10.9", "10.9": "12.9"}.get(b0.grade)
        if up:
            out.append({"kind": "grade", "title": f"Use class {up} bolts",
                        "summary": f"Proof stress rises from {fmt(b0.proof)} to {fmt(V.GRADES[up][1])} MPa at the same "
                                   "size.", "auto": False})
    return out


def playback(path: Path, study, r) -> str:
    """Tightening, then the load, increment by increment: each bolt's force, the clamped parts' stress and the
    contact pressure between them, which falls where the joint starts to open."""
    from ..plastic.common import write_playback

    mesh, res = r["mesh"], r["res"]
    frames = []
    top_s = top_p = 1e-9
    for k, fr in enumerate(res.frames):
        t = float(fr["time"])
        vm = nl.von_mises(fr["S"])
        pres = np.clip(fr.get("pressure", np.zeros(mesh.n_nodes)), 0, None)
        top_s, top_p = max(top_s, float(np.percentile(vm, 99.5))), max(top_p, float(pres.max()))
        frames.append({"t": t, "label": "tightening" if t <= 1 + 1e-9 else "loading", "U": fr["U"], "stress": vm,
                       "pressure": pres,
                       "values": {"load": max(t - 1.0, 0.0),
                                  "bolt_force_N": [round(float(res.bolt_force[n][k]), 3) for n in res.bolt_force]}})
    body = np.zeros(mesh.n_nodes)
    for b in np.unique(mesh.body):
        body[np.unique(mesh.tets[mesh.body == b].ravel())] = b
    return write_playback(
        path, mesh, frames,
        {"kind": "bolts", "title": Path(study.part).name, "question": study.question,
         "units": {"length": "mm", "stress": "MPa", "pressure": "MPa", "force": "N"},
         "phases": ["tightening", "loading"],
         "bolts": [{"label": b["label"], "head_mm": [round(float(v), 4) for v in ln.head],
                    "nut_mm": [round(float(v), 4) for v in ln.nut], "diameter_mm": ln.bolt.d,
                    "preload_N": round(b["preload"], 3)} for b, ln in zip(r["bolts"], r["lines"])],
         "about": "Frames play in order of t. From t 0 to 1 the bolts are tightened to their preload; from 1 to 2 the "
                  "external load comes on (values.load is its fraction). values.bolt_force_N lists each bolt's force "
                  "in the order of bolts, which run from head_mm to nut_mm and are not part of the mesh. pressure is "
                  "the contact pressure between parts, zero where the joint has opened; stress colours clip at the "
                  "99.5th percentile. mesh.body numbers the parts."},
        {"stress": {"unit": "MPa", "min": 0.0, "max": top_s}, "pressure": {"unit": "MPa", "min": 0.0, "max": top_p}},
        body=body)


def render(out: Path, r) -> dict:
    from ..report.render import render_scalar

    img = {}
    mesh = r["mesh"]
    fin = r["frames"][-1]
    try:
        img["stress"] = light_png(render_scalar(out / "stress.png", mesh.X, mesh.tets, mesh.tris, r["vm"], "von Mises",
                                                "MPa", "turbo"))
        if r["pairs"]:
            cp = max(r["pairs"], key=lambda p: int(p.slave.sum()))
            b = int(mesh.body[mesh.tri_elem[cp.slave][0]])
            keep = mesh.body == b
            used = np.unique(mesh.tets[keep].ravel())
            lut = np.full(mesh.n_nodes, -1)
            lut[used] = np.arange(len(used))
            tris = mesh.tris[np.isin(mesh.tri_elem, np.nonzero(keep)[0])]
            pres = np.clip(fin.get("pressure", np.zeros(mesh.n_nodes)), 0, None)[used]
            i = int(np.argmax(pres))
            img["pressure"] = light_png(render_scalar(out / "pressure.png", mesh.X[used], lut[mesh.tets[keep]],
                                                      lut[tris], pres, f"contact pressure on body {b}", "MPa",
                                                      "inferno", mesh.X[used][i], f"{fmt(float(pres.max()))} MPa"))
    except Exception as e:  # noqa: BLE001
        img["error"] = str(e)[:200]
    return img


def group_loads(lines: list[BoltLine], F: np.ndarray, M: np.ndarray) -> list[tuple[float, float]]:
    """The external load each bolt takes by hand: the force shared equally, the moment about the bolt group's centre
    spread in proportion to each bolt's distance as a rigid flange would, along the bolts (F_A) and across them
    (F_Q). Pulling the joint apart counts positive."""
    if not lines:
        return []
    a = -lines[0].axis                                 # head side up: a load toward the head pulls the joint apart
    pts = np.array([ln.head for ln in lines])
    c = pts.mean(0)
    r = pts - c
    r -= np.outer(r @ a, a)
    Mc = M - np.cross(c, F)
    s = np.cross(r, a)                                 # tilting about the centre stretches bolt i by theta . s_i
    Mb = Mc - (Mc @ a) * a
    S_ = (s[:, :, None] * s[:, None, :]).sum(0)
    axial = F @ a / len(lines) + s @ (np.linalg.pinv(S_) @ Mb)
    shear_f = F - (F @ a) * a
    T = float(Mc @ a)
    J = float((r ** 2).sum())
    out = []
    for i in range(len(lines)):
        q = shear_f / len(lines) + (T * np.cross(a, r[i]) / J if J > 1e-12 else 0.0)
        out.append((float(axial[i]), float(np.linalg.norm(q))))
    return out


def _parts_modulus(ln: BoltLine, mat_of) -> float:
    """One modulus for the clamped stack, weighted the way layers in series add their compliance."""
    Es = [mat_of[b].E for b in (ln.bodies[:-1] if ln.tapped else ln.bodies)]
    return len(Es) / sum(1.0 / e for e in Es)
