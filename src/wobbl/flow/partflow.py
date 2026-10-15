"""Flow around or through any CAD part, with OpenFOAM.

The part decides the case. Give a free stream speed and the part sits in a wind tunnel box, turned so the stream
runs along +x, and the answer is its drag, lift and side force. Give a flow rate, a pressure or a fan or pump curve
and the part's passages carry it: openings found in its flat faces are capped, the largest passage that joins two
or more of them is meshed from the inside, the largest opening is the inlet unless one is named, and the answer is
the pressure drop and how the flow splits between the outlets. A part with no openings but named inlet and outlet
faces is a fluid volume, meshed inside its own surface. A part with neither, given a flow or a fan, sits in a duct
of its own cross section, the way a shroud carries a fan's air through a heat sink.

Every case is meshed by snappyHexMesh on two or three grids, solved by simpleFoam (laminar below the transition
Reynolds number, k omega SST above it), and checked for grid convergence, iterative convergence, mass balance, mesh
quality, wall treatment and against a hand estimate of the same quantity.
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..units import fmt
from . import correlations as C
from . import foam
from .fluids import get_fluid
from .mesh3d import Region, Surface, box, mesh, read_boundary_faces, read_patch_values, snappy
from .shape import Opening, PartGeometry, frame, frontal_area

# grids; wall cells across an internal passage's hydraulic diameter; an external body's base cells across its
# length and the refinement levels at its surface; the ratio between grids; most iterations; the finest grid's budget
QUALITY = {
    "fast": {"levels": 2, "across": 6, "base": 3, "surface": 2, "ratio": 1.4, "iters": 600, "budget": 25_000},
    "normal": {"levels": 3, "across": 10, "base": 4, "surface": 3, "ratio": 1.3, "iters": 1000, "budget": 120_000},
    "fine": {"levels": 3, "across": 16, "base": 5, "surface": 4, "ratio": 1.3, "iters": 1600, "budget": 500_000},
}
WORKERS = 2
UPSTREAM, DOWNSTREAM, SIDE = 3.0, 6.0, 2.5      # an external domain's reach in body lengths


@dataclass
class PartFlowStudy:
    part: str
    fluid: str = "air"
    temp: float = 20.0                   # C
    mode: str = "auto"                   # auto, external, internal, volume, ducted
    speed: float = 0.0                   # m/s: the free stream, or the mean speed at the inlet
    direction: tuple = (1.0, 0.0, 0.0)   # the stream's direction for external and ducted flow
    up: tuple | None = None
    flow: float = 0.0                    # m3/s
    curve: tuple = ()                    # fan or pump curve, ((m3/s, Pa), ...) from free flow to shut off
    inlet: str = ""                      # 'opening 2', 'near x,y,z' (mm) or a face selector
    outlet: str = ""                     # face selector for a fluid volume's outlets
    params: dict = field(default_factory=dict)
    quality: str = "normal"
    budget_minutes: float = 60.0         # longest predicted time a des or les run may take, 0 for no limit
    turbulence: str = "auto"             # auto, laminar, rans, or des and les for a time accurate run
    max_dp: float = 0.0                  # Pa
    max_force: float = 0.0               # N
    keep: bool = False
    question: str = ""


@dataclass
class Plan:
    mode: str
    inlet: Opening | None = None
    outlets: list = field(default_factory=list)
    passage: object = None
    R: np.ndarray = field(default_factory=lambda: np.eye(3))
    faces_in: list = field(default_factory=list)     # fluid volume: CAD faces of the inlet and the outlets
    faces_out: list = field(default_factory=list)
    why: list = field(default_factory=list)


# --- deciding the case -------------------------------------------------------------------------------------------
def plan_case(st: PartFlowStudy, g: PartGeometry) -> Plan:
    mode = st.mode
    joined = [p for p in g.passages if len(p.openings) >= 2]
    if mode == "auto":
        if st.speed and not (st.flow or st.curve):
            mode = "external"
        elif st.flow or st.curve or st.speed:
            mode = "internal" if joined else ("volume" if st.inlet and st.outlet else "ducted")
        else:
            raise ValueError("say how the fluid moves: a free stream speed like '10 m/s' for flow around the part, "
                             "or a flow rate or a fan or pump curve for flow through it")
    p = Plan(mode)
    if mode in ("external", "ducted"):
        p.R = frame(st.direction, st.up)
        d = ", ".join(f"{v:+.2f}" for v in p.R[0])
        p.why.append(f"The stream runs along ({d}) in the part's coordinates.")
        if mode == "ducted":
            p.why.append("The part has no passage of its own, so it sits in a duct of its own cross section, the "
                         "way a shroud carries a fan's air through a heat sink.")
        return p
    if mode == "volume":
        from ..geometry.select import Selector
        sel = Selector(g.faces, (g.lo.tolist(), g.hi.tolist()))
        p.faces_in = sel.select(st.inlet).faces
        p.faces_out = sel.select(st.outlet).faces
        if not p.faces_in or not p.faces_out:
            raise ValueError("a fluid volume needs its inlet and outlet faces, like --inlet 'x=min' --outlet 'x=max'")
        p.why.append("The part is the fluid itself, with its named faces as the inlet and outlets and every other "
                     "face a wall.")
        return p
    if not joined:
        raise ValueError("the part has no passage joining two openings. Wobbl looks for holes in flat faces whose "
                         "far side is empty; name a fluid volume's faces with --inlet and --outlet, or give a free "
                         "stream speed for flow around it")
    inlet = _pick_inlet(st, g) if st.inlet else None
    if inlet is None:
        passage = max(joined, key=lambda q: q.volume)
        inlet = max((g.openings[i] for i in passage.openings), key=lambda o: (round(o.area, 6), -o.id))
        p.why.append(f"The inlet is opening {inlet.id}, the largest of passage {passage.id}'s openings "
                     f"({fmt(inlet.area)} mm2).")
    passage = g.passage_of(inlet.id)
    if passage is None or len(passage.openings) < 2:
        raise ValueError(f"opening {inlet.id} leads to a dead end; name an opening that joins another")
    p.inlet, p.passage = inlet, passage
    p.outlets = [g.openings[i] for i in passage.openings if i != inlet.id]
    p.why.append(f"The fluid leaves through {_list([f'opening {o.id}' for o in p.outlets])}, " +
                 ("each open to the same pressure." if len(p.outlets) > 1 else "open to the surroundings."))
    others = [q for q in g.passages if q.id != passage.id]
    if others:
        p.why.append(f"{len(others)} other passage{'s' if len(others) > 1 else ''} (like bolt holes) carry no flow.")
    return p


def _pick_inlet(st: PartFlowStudy, g: PartGeometry) -> Opening | None:
    text = st.inlet.strip().lower()
    m = re.match(r"opening\s+(\d+)$", text)
    if m:
        k = int(m.group(1))
        if k >= len(g.openings):
            raise ValueError(f"there is no opening {k}; the part has {len(g.openings)}")
        return g.openings[k]
    m = re.match(r"near\s+(.+)$", text)
    if m:
        from ..geometry.select import parse_point
        pt = parse_point(m.group(1))
        return min(g.openings, key=lambda o: np.linalg.norm(o.centre - pt))
    from ..geometry.select import Selector
    faces = set(Selector(g.faces, (g.lo.tolist(), g.hi.tolist())).select(st.inlet).faces)
    hits = [o for o in g.openings if o.face in faces]
    if not hits:
        raise ValueError(f"'{st.inlet}' picks no opening; the openings are " +
                         "; ".join(o.describe() for o in g.openings))
    return max(hits, key=lambda o: o.area)


def _list(items: list[str]) -> str:
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " and " + items[-1]


# --- one grid ----------------------------------------------------------------------------------------------------
@dataclass
class Setup:
    """Everything a grid level needs, in metres and in the frame the case is solved in."""
    mode: str
    surfaces: list
    lo: np.ndarray
    hi: np.ndarray
    kinds: dict
    location: np.ndarray
    size: float                  # base cell, m
    regions: list
    ref_length: float            # m
    ref_area: float              # m2
    speed: float                 # m/s, the reference speed
    flow: float                  # m3/s through an inlet patch, 0 for a fixed velocity inlet
    u_in: tuple                  # the inlet velocity vector, for fixed velocity inlets
    inlet: str
    outlets: list
    turb: foam.Turbulence
    nu: float
    rho: float
    iters: int
    heat: bool = False
    temp: float = 20.0           # C, the fluid's
    pr: float = 0.71             # the fluid's Prandtl number
    unsteady: str = ""           # des or les for a time accurate run, else steady
    budget_s: float = 3600.0              # a scale resolving run predicted to take longer is refused, 0 for no limit
    unsteady_spans: tuple = (6.0, 20.0)   # flow through times to build the wake, then to average over


def build_setup(st: PartFlowStudy, g: PartGeometry, p: Plan, q: dict, flow: float | None = None) -> Setup:
    s = _build_setup(st, g, p, q, flow)
    fl = get_fluid(st.fluid, st.temp)
    s.temp, s.pr = st.temp, fl.Pr
    model = (st.turbulence or "auto").lower()
    if model in ("des", "les"):
        s.unsteady = model
        s.turb = foam.Turbulence.free_stream(model, s.speed, s.nu, s.ref_length, intensity=0.01, scale=0.1)
        s.budget_s = 60.0 * float(getattr(st, "budget_minutes", 60.0))
        s.unsteady_spans = {"fast": (4.0, 10.0), "normal": (6.0, 20.0), "fine": (8.0, 40.0)}[st.quality]
    elif model == "laminar":
        s.turb = foam.Turbulence()
    elif model == "rans" and s.turb.model == "laminar":
        s.turb = foam.Turbulence.free_stream("rans", s.speed, s.nu, s.ref_length,
                                             intensity=0.01 if s.mode == "external" else 0.05)
    elif model not in ("auto", "rans"):
        raise ValueError(f"turbulence is auto, laminar, rans, des or les, not '{st.turbulence}'")
    return s


def _build_setup(st: PartFlowStudy, g: PartGeometry, p: Plan, q: dict, flow: float | None = None) -> Setup:
    fl = get_fluid(st.fluid, st.temp)
    mm = 1e-3
    surf_lv = (1, 1)
    if p.mode == "internal":
        a_in = p.inlet.area * mm ** 2
        flow = flow if flow is not None else (st.flow or st.speed * a_in)
        dh = p.inlet.dh * mm
        size = 2 * dh / q["across"]
        vol = p.passage.volume * mm ** 3
        while vol / (size / 2) ** 3 > q["budget"]:
            size *= 1.1
        lids = np.concatenate([o.tris for o in [p.inlet, *p.outlets]]) * mm
        lo = np.minimum(p.passage.lo * mm, lids.reshape(-1, 3).min(0)) - 3 * size
        hi = np.maximum(p.passage.hi * mm, lids.reshape(-1, 3).max(0)) + 3 * size
        surfaces = [Surface("walls", g.xyz * mm, "wall", surf_lv),
                    Surface("inlet", p.inlet.tris * mm, "inlet", (1, 1), flow=flow)]
        surfaces += [Surface(f"outlet{o.id}", o.tris * mm, "outlet", (1, 1)) for o in p.outlets]
        u = flow / a_in
        re_ = u * dh / fl.nu
        loc = p.inlet.seed * mm
        return Setup("internal", surfaces, lo, hi, {}, _jitter(loc, size), size, [], dh, a_in, u, flow,
                     tuple(-p.inlet.normal * u), "inlet", [s.name for s in surfaces[2:]],
                     _turbulence(re_, u, fl.nu, dh, internal=True), fl.nu, fl.rho, q["iters"])
    if p.mode == "volume":
        faces_in, faces_out = set(p.faces_in), set(p.faces_out)
        tin = g.xyz[np.isin(g.tri_face, list(faces_in))] * mm
        tout = g.xyz[np.isin(g.tri_face, list(faces_out))] * mm
        twall = g.xyz[~np.isin(g.tri_face, list(faces_in | faces_out))] * mm
        a_in = _area(tin)
        dh = 4 * a_in / max(_perimeter_estimate(tin), 1e-12)
        flow = flow if flow is not None else (st.flow or st.speed * a_in)
        size = 2 * dh / q["across"]
        vol = g.prep.volume * mm ** 3
        while vol / (size / 2) ** 3 > q["budget"]:
            size *= 1.1
        lo, hi = g.lo * mm - 3 * size, g.hi * mm + 3 * size
        loc = _interior_point(g) * mm
        n_in = _mean_normal(tin)
        u = flow / a_in
        surfaces = [Surface("walls", twall, "wall", surf_lv), Surface("inlet", tin, "inlet", (1, 1), flow=flow),
                    Surface("outlet", tout, "outlet", (1, 1))]
        return Setup("volume", surfaces, lo, hi, {}, _jitter(loc, size), size, [], dh, a_in, u, flow,
                     tuple(-n_in * u), "inlet", ["outlet"], _turbulence(u * dh / fl.nu, u, fl.nu, dh, True),
                     fl.nu, fl.rho, q["iters"])
    # external and ducted: solve in the stream's frame, flow along +x and up along +z
    tri = (g.xyz - (g.lo + g.hi) / 2) @ p.R.T * mm
    blo, bhi = tri.reshape(-1, 3).min(0), tri.reshape(-1, 3).max(0)
    ext = bhi - blo
    L = float(ext.max())
    area = frontal_area(tri, np.array([1.0, 0, 0]))
    if p.mode == "external":
        u = st.speed
        up, down, side = UPSTREAM, DOWNSTREAM, SIDE
        lo = blo - np.array([up, side, side]) * L
        hi = bhi + np.array([down, side, side]) * L
        size = L / q["base"]
        level = _surface_level(size, q["surface"], ext)
        wake = Region("wake", tuple(blo - np.array([0.2, 0.3, 0.3]) * L), tuple(bhi + np.array([2.0, 0.3, 0.3]) * L),
                      max(1, level - 2))
        near = Region("near", tuple(blo - 0.15 * L), tuple(bhi + np.array([0.6, 0.15, 0.15]) * L), max(1, level - 1))
        surfaces = [Surface("walls", tri, "wall", (level, level + 1))]
        loc = np.array([lo[0] + 0.37 * (blo[0] - lo[0]), lo[1] + 0.31 * (hi[1] - lo[1]),
                        lo[2] + 0.29 * (hi[2] - lo[2])])
        re_ = u * L / fl.nu
        return Setup("external", surfaces, lo, hi, {"xmin": "inlet", "xmax": "outlet"}, loc, size, [wake, near], L,
                     area, u, 0.0, (u, 0.0, 0.0), "xmin", ["xmax"], _turbulence(re_, u, fl.nu, L, internal=False),
                     fl.nu, fl.rho, q["iters"])
    # ducted: the duct is the part's cross section, with one length of the part upstream and two downstream
    duct_area = float(ext[1] * ext[2])
    flow = flow if flow is not None else (st.flow or st.speed * duct_area)
    u = flow / duct_area
    gap = 4 * duct_area / (2 * (ext[1] + ext[2]))
    open_area = max(duct_area - area, 0.05 * duct_area)
    dh = min(gap, 4 * open_area / (2 * (ext[1] + ext[2])))
    size = L / q["base"]
    lo = blo - np.array([1.0 * ext[0], 0.0, 0.0]) - np.array([0, 1e-4, 1e-4]) * L
    hi = bhi + np.array([2.0 * ext[0], 0.0, 0.0]) + np.array([0, 1e-4, 1e-4]) * L
    level = _surface_level(size, q["surface"], ext)
    surfaces = [Surface("walls", tri, "wall", (level, level + 1))]
    loc = np.array([lo[0] + 0.41 * (blo[0] - lo[0]), (lo[1] + hi[1]) / 2 + 0.013 * ext[1],
                    (lo[2] + hi[2]) / 2 + 0.017 * ext[2]])
    re_ = u * dh / fl.nu
    return Setup("ducted", surfaces, lo, hi, {"xmin": "inlet", "xmax": "outlet", "ymin": "wall", "ymax": "wall",
                                              "zmin": "wall", "zmax": "wall"}, loc, size, [], L, duct_area, u, flow,
                 (u, 0.0, 0.0), "xmin", ["xmax"], _turbulence(re_, u, fl.nu, dh, internal=True), fl.nu, fl.rho,
                 q["iters"])


def _surface_level(size: float, level: int, ext: np.ndarray, most: int = 7) -> int:
    """The refinement level at the surface: the quality's own, raised until the part's thinnest extent spans at
    least three cells, so a long thin strut or a flat plate is not lost between background cells."""
    thin = float(np.min(ext))
    while size / 2 ** level > thin / 3 and level < most:
        level += 1
    return level


def _turbulence(re_: float, u: float, nu: float, length: float, internal: bool) -> foam.Turbulence:
    if (internal and re_ < C.LAMINAR_RE) or (not internal and re_ < 200):
        return foam.Turbulence()
    if internal:
        return foam.Turbulence.free_stream("rans", u, nu, length, intensity=min(0.1, 0.16 * re_ ** -0.125))
    return foam.Turbulence.free_stream("rans", u, nu, length, intensity=0.01, scale=0.1)


def _jitter(p: np.ndarray, size: float) -> np.ndarray:
    return np.asarray(p, float) + np.array([0.0137, 0.0191, 0.0113]) * size


def _area(tris: np.ndarray) -> float:
    return float(np.linalg.norm(np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0]), axis=1).sum() / 2)


def _mean_normal(tris: np.ndarray) -> np.ndarray:
    n = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0]).sum(0)
    return n / max(np.linalg.norm(n), 1e-300)


def _perimeter_estimate(tris: np.ndarray) -> float:
    from .shape import boundary_loops
    pts, inv = np.unique(np.round(tris.reshape(-1, 3), 12), axis=0, return_inverse=True)
    loops = boundary_loops(inv.reshape(-1, 3))
    return sum(float(np.linalg.norm(np.roll(pts[lp], -1, axis=0) - pts[lp], axis=1).sum()) for lp in loops)


def _interior_point(g: PartGeometry) -> np.ndarray:
    """A point inside the material, for meshing a fluid volume: the first of a few probes along the box diagonal."""
    for t in (0.5, 0.43, 0.57, 0.37, 0.63, 0.3, 0.7):
        pt = g.lo + t * (g.hi - g.lo)
        if g.inside(pt[None])[0]:
            return pt
    c = g.xyz.mean(1)
    n = np.cross(g.xyz[:, 1] - g.xyz[:, 0], g.xyz[:, 2] - g.xyz[:, 0])
    n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-300)
    for i in np.argsort(-np.linalg.norm(np.cross(g.xyz[:, 1] - g.xyz[:, 0], g.xyz[:, 2] - g.xyz[:, 0]), axis=1))[:20]:
        pt = c[i] - n[i] * 1e-3 * g.diag
        if g.inside(pt[None])[0]:
            return pt
    raise ValueError("could not find a point inside the fluid volume; is the part closed?")


def run_level(s: Setup, work: Path, scale: float, tag: str, fields: bool = False) -> dict:
    case = foam.Case(work / tag)
    size = s.size * scale
    box(case, s.lo, s.hi, size, s.kinds)
    surfaces = [Surface(x.name, x.tris, x.kind, x.level, x.flow, x.scalar) for x in s.surfaces]
    snappy(case, surfaces, s.location, s.regions, max_cells=4_000_000)
    for pt in case.patches:
        if pt.name == s.inlet and s.mode in ("external", "ducted"):
            pt.kind, pt.flow = "inlet", s.flow if s.mode == "ducted" else 0.0
    foam.write_physics(case, s.nu, s.turb)
    foam.write_steady(case, s.turb, s.iters, "", scalar=s.heat)      # meshing reads the schemes and controls
    t0 = time.time()
    grid = mesh(case)
    if grid.patches.get(s.inlet, 0) == 0:
        raise RuntimeError("the grid lost the inlet; the lid may not seal the passage")
    if grid.patches.get("walls", 0) == 0:
        raise RuntimeError("the grid has no wall faces, so the part fell between its cells; use a finer quality")
    foam.write_fields(case, s.u_in, s.turb, scalar={"inlet": 0.0, "wall": 1.0} if s.heat else None)
    if s.unsteady:
        from .transient import run_transient
        out = {"size": size, "cells": grid.cells, "grid": grid, "t_mesh": time.time() - t0}
        out.update(run_transient(case, s, grid, tag, fields))
        out["iterations"] = len(out.get("history", []))
        out["swing"] = abs(out["halves"][0] - out["halves"][1]) / max(abs(out["cd"]), 1e-300) \
            if "halves" in out else 0.0
        if fields:
            out["fields"] = _fields(case, s)
        case.clean()
        return out
    fo = _functions(s, grid)
    foam.write_steady(case, s.turb, s.iters, fo, scalar=s.heat, tol=1e-5)
    _non_orthogonal(case, grid)
    cd = case.root / "system" / "controlDict"
    cd.write_text(cd.read_text().replace("runTimeModifiable false;", "runTimeModifiable true;"))
    t_mesh = time.time() - t0
    log = case.run("simpleFoam", timeout=7200, watch=lambda c: settled(c, s, grid))
    out = {"tag": tag, "size": size, "cells": grid.cells, "grid": grid, "log": log, "t_mesh": t_mesh,
           "t_solve": time.time() - t0 - t_mesh}
    out.update(_results(case, s, grid))
    if fields:
        out["fields"] = _fields(case, s)
    case.clean()
    return out


def settled(case_dir: Path, s: Setup, grid, least: int = 150) -> bool:
    """True once the answer has stopped moving: the mean of the last window and the window before it agree to 0.2%
    and the scalar, when carried, to 0.5%."""
    case = foam.Case.__new__(foam.Case)
    case.root = Path(case_dir)
    if s.mode == "external":
        h = foam.column(case, "forces", "Cd")
    else:
        outs = [o for o in s.outlets if grid.patches.get(o, 0)]
        h = foam.column(case, f"p_{s.inlet}", "weightedAverage(total(p))") - \
            foam.column(case, f"p_{outs[0]}", "weightedAverage(total(p))")
    n = len(h)
    w = max(40, n // 10)
    if n < max(least, 2 * w):
        return False
    a, b = np.mean(h[-w:]), np.mean(h[-2 * w:-w])
    return abs(a - b) <= 2e-3 * max(abs(a), 1e-300)


def _non_orthogonal(case: foam.Case, grid) -> None:
    """Snapped grids need non orthogonal correction; one corrector holds up to about 70 degrees."""
    p = case.root / "system" / "fvSolution"
    n = 1 if grid.max_non_ortho < 70 else 2
    p.write_text(p.read_text().replace("nNonOrthogonalCorrectors 0;", f"nNonOrthogonalCorrectors {n};"))


def _functions(s: Setup, grid) -> str:
    fo = foam.fo_yplus() if s.turb.model != "laminar" else ""
    if s.mode == "external":
        fo += foam.fo_forces("walls", s.speed, s.ref_length, s.ref_area, lift=(0, 0, 1), drag=(1, 0, 0))
    ends = [s.inlet] + [o for o in s.outlets if grid.patches.get(o, 0)]
    fo += ("    ptot\n    {\n        type pressure; libs (fieldFunctionObjects); mode total; p p; U U; rho rhoInf;\n"
           "        rhoInf 1; pRef 0; executeControl timeStep; writeControl writeTime; log false;\n    }\n")
    for e in ends:
        fo += foam.fo_patch_sum(f"q_{e}", e)
        fo += foam.fo_patch_sum(f"p_{e}", e, "total(p)", "weightedAverage", "phi")
        fo += foam.fo_patch_sum(f"ps_{e}", e, "p", "areaAverage")
    if s.heat:
        fo += foam.fo_scalar(s.pr)
    return fo + foam.fo_cell_centres()


def _results(case: foam.Case, s: Setup, grid) -> dict:
    out = {}
    ends = [s.inlet] + [o for o in s.outlets if grid.patches.get(o, 0)]
    q = {e: float(foam.column(case, f"q_{e}", "sum(phi)")[-1]) for e in ends}
    pt = {e: foam.column(case, f"p_{e}", "weightedAverage(total(p))") for e in ends}
    ps = {e: float(foam.column(case, f"ps_{e}", "areaAverage(p)")[-1]) for e in ends}
    q_in = -q[s.inlet]
    out["flows"] = {e: v for e, v in q.items() if e != s.inlet}
    out["q_in"], out["q_out"] = q_in, sum(out["flows"].values())
    outs = [e for e in ends if e != s.inlet]
    drops = {e: (pt[s.inlet][-1] - pt[e][-1]) * s.rho for e in outs}
    hist = (pt[s.inlet] - sum(pt[e] * q[e] for e in outs) / max(sum(q[e] for e in outs), 1e-300)) * s.rho
    out["dp_total"] = float(sum(drops[e] * q[e] for e in outs) / max(sum(q[e] for e in outs), 1e-300))
    out["dp_static"] = float((ps[s.inlet] - sum(ps[e] * q[e] for e in outs) / max(sum(q[e] for e in outs), 1e-300))
                             * s.rho)
    out["drops"] = drops
    out["history"] = hist
    if s.mode == "external":
        head, arr = foam.table(case, "forces")
        qd = 0.5 * s.rho * s.speed ** 2 * s.ref_area
        cols = {c: arr[:, head.index(c)] for c in ("Cd", "Cl", "Cs") if c in head}
        out["cd"], out["cl"] = float(cols["Cd"][-1]), float(cols["Cl"][-1])
        out["cs"] = float(cols.get("Cs", np.zeros(1))[-1])
        out["drag"], out["lift"], out["side"] = out["cd"] * qd, out["cl"] * qd, out["cs"] * qd
        out["history"] = cols["Cd"] * qd
    tail = out["history"][int(0.8 * len(out["history"])):]
    out["value"] = float(np.mean(tail)) if s.mode == "external" else float(out["history"][-1])
    out["swing"] = float(np.ptp(tail) / max(abs(np.mean(tail)), 1e-300)) if len(tail) else 0.0
    if s.turb.model != "laminar":
        head, arr = foam.table(case, "yPlus")
        if len(arr):
            out["yplus"] = {"walls": tuple(float(x) for x in arr[-1, -3:])}
    out["iterations"] = len(out["history"])
    return out


def _fields(case: foam.Case, s: Setup) -> dict:
    last = foam.last_time(case)
    f = {"C": foam.read_field(last / "C"), "U": foam.read_field(last / "U"), "p": foam.read_field(last / "p")}
    b = read_boundary_faces(case, ["walls"]).get("walls")
    if b is not None:      # zero gradient walls carry the pressure of the cell beside them
        f["wall_centres"], f["wall_areas"], f["wall_owner"] = b["centres"], b["areas"], b["owner"]
        f["wall_p"] = f["p"][b["owner"]] if f["p"].ndim == 1 and len(f["p"]) > 1 else None
        if s.heat:
            f["T"] = foam.read_field(last / "T")
            nut = last / "nut"
            f["wall_nut"] = read_patch_values(nut, "walls") if nut.exists() else None
    return f
