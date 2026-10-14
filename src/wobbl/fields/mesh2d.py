"""Planar and axisymmetric models: named shapes in a bounded air domain, meshed with gmsh for GetDP.

A model is a list of shapes (rectangles, disks and polygons, in meters) inside
an outer disk that can be clipped to a half or quarter plane. Shapes later in
the list win where they overlap earlier ones. A shape marked as a hole is cut
out, and its edges become a boundary named after it (a conductor at a fixed
voltage, a wire surface). An optional shell outside the disk is mapped to
infinity by GetDP, which stands in for open space.

Meshing runs in a child process (see mesh.isolate), and the result carries the
node coordinates, triangles and region of every triangle for plotting.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass
class Shape:
    name: str
    kind: str                       # rect, disk, poly
    data: tuple                     # rect (x0, y0, x1, y1), disk (cx, cy, r), poly ((x, y), ...)
    hole: bool = False
    size: float | None = None       # element size inside, or along the edges of a hole


@dataclass
class Model2D:
    kind: str                       # planar or axi (x is the radius, y the axis)
    shapes: list[Shape]
    radius: float                   # outer disk radius
    center: tuple[float, float] = (0.0, 0.0)
    clip: str = ""                  # half planes the domain keeps, e.g. "x>=0" or "y>=0,y<=0.003"
    shell: float = 0.0              # outer radius of the shell mapped to infinity, 0 for none
    size: float = 0.0               # element size near the shapes
    far: float = 0.0                # element size at the outer boundary
    growth: float = 0.3             # how fast elements grow away from the shapes (size per distance)
    lines: dict[str, str] = field(default_factory=dict)   # boundary name: "x=0", "y<0.002", "outer", ...
    refine: list[tuple[str, float, float]] = field(default_factory=list)  # (shape, size at its edges, reach)
    spots: list[tuple[float, float, float]] = field(default_factory=list)  # (x, y, size there): sharp features

    def scaled(self, k: float) -> "Model2D":
        """The same model with every element size multiplied by k (k < 1 refines)."""
        from dataclasses import replace

        shapes = [replace(s, size=s.size * k if s.size else None) for s in self.shapes]
        return replace(self, shapes=shapes, size=self.size * k, far=self.far * k,
                       refine=[(n, h * k, d) for n, h, d in self.refine],
                       spots=[(x, y, h * k) for x, y, h in self.spots])


@dataclass
class Mesh2D:
    path: str
    X: np.ndarray                   # (N, 2) node coordinates
    tris: np.ndarray                # (M, 3) zero based corner nodes
    region: np.ndarray              # (M,) physical tag of each triangle
    tags: dict[str, int]            # region and boundary names to physical tags
    node_ids: np.ndarray            # gmsh node tag of each row of X

    @property
    def n_nodes(self) -> int:
        return len(self.X)

    def mask(self, *names: str) -> np.ndarray:
        return np.isin(self.region, [self.tags[n] for n in names if n in self.tags])


def mesh(model: Model2D, path: str | Path, timeout: float = 120.0) -> Mesh2D:
    from ..mesh.isolate import run_isolated

    d = run_isolated(_build, model, str(path), timeout=timeout)
    return Mesh2D(**d)


def _bbox_match(gm, tag: int, pred: str, center, radius, tol: float) -> bool:
    """Whether a curve meets a predicate: terms like x=0, y<0.002, r=0.01 or outer, joined by & (all) and | (any)."""
    if "|" in pred:
        return any(_bbox_match(gm, tag, p, center, radius, tol) for p in pred.split("|"))
    lo, hi = gm.getParametrizationBounds(1, tag)
    pts = np.array(gm.getValue(1, tag, list(np.linspace(lo[0], hi[0], 7)))).reshape(-1, 3)
    for term in pred.split("&"):
        term = term.strip()
        if term == "outer":
            r = np.hypot(pts[:, 0] - center[0], pts[:, 1] - center[1])
            if not np.all(r > radius * (1 - 1e-6)):
                return False
            continue
        key, op, val = term[0], term[1], float(term[2:])
        if key == "r":
            ok = np.all(np.abs(np.hypot(pts[:, 0] - center[0], pts[:, 1] - center[1]) - val) < tol)
        else:
            v = pts[:, 0] if key == "x" else pts[:, 1]
            ok = {"=": np.all(np.abs(v - val) < tol), "<": np.all(v < val + tol), ">": np.all(v > val - tol)}[op]
        if not ok:
            return False
    return True


def _build(model: Model2D, path: str) -> dict:
    import gmsh

    if gmsh.isInitialized():
        gmsh.finalize()
    gmsh.initialize(readConfigFiles=False)
    gmsh.option.setNumber("General.Terminal", 0)
    try:
        return _build_in(gmsh, model, path)
    finally:
        gmsh.finalize()


def _clipped(occ, tag: int, clip: str, big: float, center, within: float = 0.0) -> list:
    """Cut a surface down to the half planes in clip ("x>=0,y<=0.002") and, when within is set, to the disk of
    that radius, so shapes that reach past the outer boundary end at it."""
    cx, cy = center
    lim = {"x>": cx - big, "x<": cx + big, "y>": cy - big, "y<": cy + big}
    for term in filter(None, (c.strip() for c in clip.split(","))):
        lim[term[:2]] = float(term[3:])
    out = [(2, tag)]
    if clip:
        box = occ.addRectangle(lim["x>"], lim["y>"], 0, lim["x<"] - lim["x>"], lim["y<"] - lim["y>"])
        out, _ = occ.intersect(out, [(2, box)])
    if within and out:
        disk = occ.addDisk(cx, cy, 0, within, within)
        out, _ = occ.intersect(out, [(2, disk)])
    return out


def _build_in(gmsh, model: Model2D, path: str) -> dict:
    occ = gmsh.model.occ
    gm = gmsh.model
    cx, cy = model.center
    reach = max((abs(v) for s in model.shapes for v in np.ravel(s.data)), default=1.0)
    big = 4 * max(model.radius, model.shell, reach + abs(cx) + abs(cy))
    objs, names = [], []
    if model.radius > 0:                 # radius 0: the shapes are the whole domain (a duct), no air around them
        air = _clipped(occ, occ.addDisk(cx, cy, 0, model.radius, model.radius), model.clip, big, model.center)
        objs.append(air)
        names.append("air")
    if model.shell:
        outer = _clipped(occ, occ.addDisk(cx, cy, 0, model.shell, model.shell), model.clip, big, model.center)
        inner = _clipped(occ, occ.addDisk(cx, cy, 0, model.radius, model.radius), model.clip, big, model.center)
        shell, _ = occ.cut(outer, inner)
        objs.append(shell)
        names.append("shell")
    for s in model.shapes:
        if s.kind == "rect":
            x0, y0, x1, y1 = s.data
            t = occ.addRectangle(x0, y0, 0, x1 - x0, y1 - y0)
        elif s.kind == "disk":
            t = occ.addDisk(s.data[0], s.data[1], 0, s.data[2], s.data[2])
        else:
            pts = [occ.addPoint(x, y, 0) for x, y in s.data]
            ls = [occ.addLine(pts[i], pts[(i + 1) % len(pts)]) for i in range(len(pts))]
            t = occ.addPlaneSurface([occ.addCurveLoop(ls)])
        objs.append(_clipped(occ, t, model.clip if model.radius > 0 else "", big, model.center,
                             within=model.shell or model.radius))
        names.append(s.name)
    flat = [dt for o in objs for dt in o]
    if len(flat) > 1:
        _, mp = occ.fragment(flat, [])
    else:                                # a lone duct needs no fragmenting
        mp = [[flat[0]]]
    occ.synchronize()
    owner: dict[int, int] = {}          # surface tag -> index of the last shape that covers it
    k = 0
    for i, o in enumerate(objs):
        for _ in o:
            for dim, t in mp[k]:
                if dim == 2:
                    owner[t] = i
            k += 1
    holes = {i for i, s in enumerate(model.shapes, start=len(names) - len(model.shapes)) if s.hole}
    keep = {t: i for t, i in owner.items() if i not in holes}
    drop = [t for t, i in owner.items() if i in holes]
    hole_curves: dict[str, set[int]] = {}
    for t in drop:
        nm = names[owner[t]]
        for _, c in gm.getBoundary([(2, t)], oriented=False):
            hole_curves.setdefault(nm, set()).add(abs(c))
    if drop:
        occ.remove([(2, t) for t in drop], recursive=False)
        occ.synchronize()
    tags: dict[str, int] = {}
    region_names = []
    for i in sorted(set(keep.values())):
        if names[i] not in region_names:
            region_names.append(names[i])
    for j, nm in enumerate(region_names, start=1):
        surfs = [t for t, i in keep.items() if names[i] == nm]
        gm.addPhysicalGroup(2, surfs, j)
        tags[nm] = j
    # boundaries: every curve on the edge of the meshed domain or of a hole
    live = set()
    for t in keep:
        for _, c in gm.getBoundary([(2, t)], oriented=False):
            live.add(abs(c))
    tol = 1e-9 * max(model.radius, model.shell, 1e-3) * 1e3
    claimed: dict[str, list[int]] = {}
    radius = model.shell or model.radius
    for c in sorted(live):
        for nm, pred in model.lines.items():
            if _bbox_match(gm, c, pred, model.center, radius, tol):
                claimed.setdefault(nm, []).append(c)
                break
        else:
            for nm, cs in hole_curves.items():
                if c in cs:
                    claimed.setdefault(nm, []).append(c)
                    break
    for j, (nm, cs) in enumerate(claimed.items(), start=101):
        gm.addPhysicalGroup(1, cs, j)
        tags[nm] = j
    _sizes(gmsh, model, keep, names, hole_curves)
    gmsh.option.setNumber("Mesh.Algorithm", 6)
    gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)
    gm.mesh.generate(2)
    gmsh.write(path)
    ntags, xyz, _ = gm.mesh.getNodes()
    X = xyz.reshape(-1, 3)[:, :2]
    index = {int(t): i for i, t in enumerate(ntags)}
    tris, region = [], []
    for nm in region_names:
        for t in gm.getEntitiesForPhysicalGroup(2, tags[nm]):
            et, _, en = gm.mesh.getElements(2, int(t))
            for typ, conn in zip(et, en):
                if typ != 2:
                    continue
                c = np.array([index[int(n)] for n in conn], dtype=np.int64).reshape(-1, 3)
                tris.append(c)
                region.append(np.full(len(c), tags[nm], dtype=np.int32))
    return {"path": path, "X": X.astype(float), "tris": np.vstack(tris), "region": np.concatenate(region),
            "tags": tags, "node_ids": np.asarray(ntags, dtype=np.int64)}


def _sizes(gmsh, model: Model2D, keep: dict, names: list, hole_curves: dict) -> None:
    fl = gmsh.model.mesh.field
    cx, cy = model.center
    # extent of the shapes, beyond which elements grow towards the far size
    pts = []
    for s in model.shapes:
        if s.kind == "rect":
            pts += [(s.data[0], s.data[1]), (s.data[2], s.data[3])]
        elif s.kind == "disk":
            pts += [(s.data[0] - s.data[2], s.data[1] - s.data[2]), (s.data[0] + s.data[2], s.data[1] + s.data[2])]
        else:
            pts += list(s.data)
    # shapes that span the domain (a substrate) do not count, or nothing would grow
    near = [math.hypot(x - cx, y - cy) for x, y in pts]
    core = max((r for r in near if r < 0.5 * model.radius), default=0.0)
    fields = []
    f = fl.add("MathEval")
    fl.setString(f, "F", f"Min({model.far}, {model.size} + {model.growth}*Max(0, Sqrt((x-({cx}))^2+(y-({cy}))^2)"
                         f"-{core}))")
    fields.append(f)
    for i, s in enumerate(model.shapes, start=len(names) - len(model.shapes)):
        if not s.size:
            continue
        if s.hole:
            cs = sorted(hole_curves.get(s.name, ()))
            if not cs:
                continue
            d = fl.add("Distance")
            fl.setNumbers(d, "CurvesList", cs)
            fl.setNumber(d, "Sampling", 100)
            th = fl.add("MathEval")
            fl.setString(th, "F", f"{s.size} + {model.growth}*F{d}")
            fields.append(th)
        else:
            surfs = [t for t, j in keep.items() if j == i]
            if not surfs:
                continue
            m = fl.add("MathEval")
            fl.setString(m, "F", f"{s.size}")
            r = fl.add("Restrict")
            fl.setNumber(r, "InField", m)
            fl.setNumbers(r, "SurfacesList", surfs)
            fields.append(r)
    for nm, h, reach in model.refine:
        cs = sorted(hole_curves.get(nm, ()))
        if not cs:
            idx = [i for i, n in enumerate(names) if n == nm]
            surfs = [t for t, j in keep.items() if j in idx]
            cs = sorted({abs(c) for t in surfs for _, c in gmsh.model.getBoundary([(2, t)], oriented=False)})
        if not cs:
            continue
        d = fl.add("Distance")
        fl.setNumbers(d, "CurvesList", cs)
        fl.setNumber(d, "Sampling", 200)
        th = fl.add("Threshold")
        fl.setNumber(th, "InField", d)
        fl.setNumber(th, "SizeMin", h)
        fl.setNumber(th, "SizeMax", model.far)
        fl.setNumber(th, "DistMin", 0.0)
        fl.setNumber(th, "DistMax", reach)
        fields.append(th)
    for x, y, h in model.spots:
        f = fl.add("MathEval")
        fl.setString(f, "F", f"{h} + {model.growth}*Sqrt((x-({x}))^2+(y-({y}))^2)")
        fields.append(f)
    mn = fl.add("Min")
    fl.setNumbers(mn, "FieldsList", fields)
    fl.setAsBackgroundMesh(mn)
    for k in ("Mesh.MeshSizeExtendFromBoundary", "Mesh.MeshSizeFromPoints", "Mesh.MeshSizeFromCurvature"):
        gmsh.option.setNumber(k, 0)
