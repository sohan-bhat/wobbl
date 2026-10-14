"""Where a crack starts and which way it faces, from a solve of the part without the crack.

The uncracked part is meshed and solved once, then again with the mesh refined around the most stressed point,
the way a strength study homes in on its peak. A crack starts at the surface, where fatigue cracks start, at the
most stressed point away from idealized supports and loads, on the faces the crack spec names, or at the surface
point nearest a given point. Its plane is normal to the largest principal stress there, the plane a fatigue crack
opens on, unless the spec gives a normal. The same solve gives the stress the crack's faces carried before it
was there, for the weight function check, and the force across the crack's plane, for net section yield.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from ..geometry.faces import PartFeatures
from ..geometry.select import Selector, parse_point
from ..mesh.mesher import Mesh, Refinement, build_mesh
from ..structural.build import BuiltModel, build_static
from ..structural.spec import SpecError, StructuralStudy
from ..units import fmt
from . import shapes
from .crackmesh import Crack
from .spec import CrackSpec, FractureStudy


@dataclass
class Uncracked:
    mesh: Mesh
    S: np.ndarray                 # nodal stresses xx, yy, zz, xy, yz, zx, MPa
    built: BuiltModel
    excluded: np.ndarray          # nodes near idealized supports, loads and sharp corners
    principal: np.ndarray         # largest principal stress at each node, MPa
    levels: list[int] = field(default_factory=list)
    solver: str = ""


@dataclass
class Placement:
    crack: Crack
    origin: np.ndarray
    normal: np.ndarray
    toward: np.ndarray            # the direction the crack first grows, in its plane
    stress: float                 # normal stress on the crack plane at the origin before the crack, MPa
    where: str                    # in words
    wall: float = 0.0             # wall thickness along the front for through cracks, mm
    margin: float = 0.0
    notes: list[str] = field(default_factory=list)


def structural_study(st: FractureStudy) -> StructuralStudy:
    return StructuralStudy(st.part, st.material, st.loads, st.fixtures, quality="fast" if st.quality == "fast"
                           else "normal", scale=st.scale, params=st.params)


def solve_uncracked(st: FractureStudy, prep, features: PartFeatures, material, adapter, workdir, crack: CrackSpec,
                    progress=None) -> tuple[Uncracked, int]:
    """Solve the part without the crack, refine around where the crack will start and solve again. Returns the
    refined solve and the origin node."""
    from ..structural.hotspots import ZoneMap
    from ..structural.run import (QUALITY, exclusion_radius, find_sharp_corners, initial_sizing,
                                  resolve_selections)

    say = progress or (lambda a, b: None)
    sst = structural_study(st)
    loads, fixtures, _ = resolve_selections(sst, features)
    sizing = initial_sizing(prep, features, sst, QUALITY[sst.quality], loads)
    origin, unc, levels = None, None, []
    a_ref = crack.a if crack.a > 0 else HOLE_FLAW
    for lv in range(2):
        say("mesh", f"uncracked part, level {lv}: element size {sizing.h:.3g} mm")
        mesh = build_mesh(prep, sizing)
        built = build_static(mesh, loads, fixtures, material, features.faces)
        say("solve", f"uncracked part, level {lv}: {mesh.n_nodes:,} nodes")
        res = adapter.solve_static(built.model, workdir / f"uncracked{lv}")
        corners = find_sharp_corners(mesh, faces=features.faces, tangent=features.tangent_pairs)
        zmap = ZoneMap(mesh, built.zones, built.zone_kind, corners)
        measure = np.maximum(res.principal()[:, 2], 0.0)
        r_ex = exclusion_radius(features, sizing, mesh, mesh.X[measure > 0.5 * measure.max()])
        unc = Uncracked(mesh, res.S, built, zmap.excluded_mask(mesh.X, r_ex), measure, solver=res.solver)
        levels.append(mesh.n_nodes)
        origin = find_origin(crack, unc, features, prep)
        if lv == 0:
            h = float(np.nanmin(mesh.node_sizes()[[origin]]))
            size = max(min(sizing.h / 5.0, 0.5 * a_ref, h / 2), 0.02 * sizing.h)
            sizing.refinements.append(Refinement(size, points=[tuple(mesh.X[origin])], radius=max(3 * a_ref, 4 * size),
                                                 grow=3 * sizing.h, label="crack origin"))
    unc.levels = levels
    return unc, origin


def boundary_nodes(mesh: Mesh) -> np.ndarray:
    return np.unique(mesh.tris[:, :6] if mesh.tris.shape[1] == 6 else mesh.tris)


def find_origin(crack: CrackSpec, unc: Uncracked, features: PartFeatures, prep) -> int:
    """The node a crack starts from."""
    mesh = unc.mesh
    surf = boundary_nodes(mesh)
    where = crack.where.strip()
    if where == "hottest":
        cand = surf[~unc.excluded[surf]]
        if not len(cand):
            cand = surf
        return int(cand[np.argmax(unc.principal[cand])])
    try:
        p = parse_point(where)
        return int(surf[np.argmin(np.linalg.norm(mesh.X[surf] - p, axis=1))])
    except Exception:  # noqa: BLE001  not a point: a face selector
        pass
    sel = Selector(features.faces, features.bbox).select(where)
    nodes = np.unique(mesh.tris[sel.tri_mask(mesh)].ravel())
    if not len(nodes):
        raise SpecError(f"the crack's faces '{where}' cover no part of the mesh")
    return int(nodes[np.argmax(unc.principal[nodes])])


def principal_direction(S: np.ndarray) -> tuple[float, np.ndarray]:
    sxx, syy, szz, sxy, syz, szx = S
    T = np.array([[sxx, sxy, szx], [sxy, syy, syz], [szx, syz, szz]])
    w, v = np.linalg.eigh(T)
    return float(w[-1]), v[:, -1]


def ray_hits(p: np.ndarray, v: np.ndarray, X: np.ndarray, tris: np.ndarray) -> np.ndarray:
    """Distances along a ray to every boundary triangle it crosses, sorted."""
    V0, V1, V2 = X[tris[:, 0]], X[tris[:, 1]], X[tris[:, 2]]
    e1, e2 = V1 - V0, V2 - V0
    q = np.cross(v, e2)
    det = np.einsum("ij,ij->i", e1, q)
    ok = np.abs(det) > 1e-14
    inv = np.where(ok, 1.0 / np.where(ok, det, 1.0), 0.0)
    s = p - V0
    u = np.einsum("ij,ij->i", s, q) * inv
    r = np.cross(s, e1)
    w = (r @ v) * inv
    t = np.einsum("ij,ij->i", e2, r) * inv
    hit = ok & (u >= -1e-9) & (w >= -1e-9) & (u + w <= 1 + 1e-9) & (t > 1e-9)
    return np.sort(t[hit])


class PartShape:
    """The uncracked part's surface, for asking whether points are inside it and how far a ray runs before leaving."""

    def __init__(self, mesh: Mesh):
        self.X, self.tris = mesh.X, mesh.tris[:, :3]
        v = np.array([0.5773, 0.5779, 0.5769])
        self._v = v / np.linalg.norm(v)

    def inside(self, points) -> np.ndarray:
        """Which points are inside, by the parity of boundary crossings along a ray."""
        return np.array([len(ray_hits(np.asarray(p, float), self._v, self.X, self.tris)) % 2 == 1
                         for p in np.atleast_2d(points)], bool)

    def ahead(self, p, v, skip: float = 0.0) -> float:
        """Distance from p along v to the first boundary crossing past ``skip``. A ray lying in a flat face does
        not cross it, and a ray just grazing a curved face crosses it within ``skip``."""
        t = ray_hits(np.asarray(p, float), np.asarray(v, float), self.X, self.tris)
        t = t[t > skip]
        return float(t[0]) if len(t) else math.inf


def _first(p, v, mesh, skip: float = 0.0) -> float:
    """Distance to the first boundary crossing along a ray, past ``skip``. A point on a curved face sits a hair
    outside the flat triangles that facet it, so rays from the surface skip that hair."""
    t = ray_hits(p, v, mesh.X, mesh.tris[:, :3])
    t = t[t > skip]
    return float(t[0]) if len(t) else math.inf


def _face_normals(mesh: Mesh, o: np.ndarray, radius: float) -> list[np.ndarray]:
    """Outward normals of the faces meeting at a surface point: one for a flat or curved face, two at an edge."""
    n, a, c = mesh.tri_geometry()
    near = np.linalg.norm(c - o, axis=1) <= radius
    if not near.any():
        near = np.argsort(np.linalg.norm(c - o, axis=1))[:6]
    groups: list[list[int]] = []
    for i in np.nonzero(near)[0] if near.dtype == bool else near:
        for g in groups:
            if n[g[0]] @ n[i] > math.cos(math.radians(35)):
                g.append(i)
                break
        else:
            groups.append([i])
    groups.sort(key=lambda g: -a[g].sum())
    out = []
    for g in groups[:2]:
        v = (n[g] * a[g][:, None]).sum(0)
        out.append(v / np.linalg.norm(v))
    return out


def place(spec: CrackSpec, unc: Uncracked, origin: int, features: PartFeatures, h_front: float) -> Placement:
    """Build the starting crack at the origin node."""
    mesh = unc.mesh
    o = mesh.X[origin].copy()
    sigma, ndir = principal_direction(unc.S[origin])
    normal = np.asarray(spec.normal, float) if spec.normal is not None else ndir
    normal = normal / np.linalg.norm(normal)
    size = float(np.nanmin(mesh.node_sizes()[[origin]]))
    faces = _face_normals(mesh, o, max(2.0 * size, 0.3 * spec.a))
    notes = []
    a, c = spec.a, spec.c or spec.a
    hole = _hole_at(spec, features, o)
    if spec.shape == "corner":
        if hole is not None:
            axis, ap, r, half = hole
            ax = np.asarray(axis, float)
            z = float((o - ap) @ ax)
            end = half if z >= 0 else -half
            o = o + (end - z) * ax
            radial = o - ap - ((o - ap) @ ax) * ax
            radial /= np.linalg.norm(radial)
            d1, d2 = radial, -np.sign(end) * ax
        elif len(faces) == 2:
            d1, d2 = -faces[1], -faces[0]
        else:
            raise SpecError("a corner crack needs an edge where two faces meet; pick a point on one, or a hole")
        normal = np.cross(d1, d2)
        normal /= np.linalg.norm(normal)
        margin = _margin(mesh, o, -(d1 + d2) / np.linalg.norm(d1 + d2), a, h_front)
        crack = shapes.corner(o, normal, d1, d2, a, c, margin)
        toward = d1
    else:
        inward = -faces[0]
        if spec.toward is not None:
            inward = np.asarray(spec.toward, float)
        d = inward - (inward @ normal) * normal
        if np.linalg.norm(d) < 1e-6:
            raise SpecError("the crack plane is parallel to the surface it starts from; give a normal or a toward "
                            "direction")
        d /= np.linalg.norm(d)
        w = np.cross(normal, d)
        if spec.shape == "surface":
            margin = _margin(mesh, o, -d, a, h_front)
            crack = shapes.surface(o, normal, d, a, c, margin)
        else:
            # a point inside the wall: half the crack along its way in, or just under the face it starts on
            probe = o + (0.5 * a * d if spec.shape == "through" else 0.25 * size * -faces[0])
            lo, hi = _first(probe, -w, mesh), _first(probe, w, mesh)
            if not (np.isfinite(lo) and np.isfinite(hi)):
                raise SpecError("could not find the wall the crack runs through; give its normal and toward "
                                "directions")
            o = o + float((probe - o) @ w) * w + 0.5 * (hi - lo) * w
            half = 0.5 * (hi + lo)
            if spec.shape == "center":
                margin = max(2.0 * h_front, 0.1 * half)
                crack = shapes.center(o, normal, d, a, (half, half), margin)
            else:
                margin = _margin(mesh, o, -d, a, h_front)
                crack = shapes.through(o, normal, d, a, (half, half), margin)
            notes.append(f"The wall is {2 * half:.3g} mm thick where the crack runs through it.")
        toward = d
    s_n = float(normal @ _tensor(unc.S[origin]) @ normal)
    where = _describe(features, mesh, o, hole)
    return Placement(crack, o, normal, toward, s_n, where, 2 * half if spec.shape in ("through", "center") else 0.0,
                     margin, notes)


# Starting flaws of damage tolerance practice (MIL-A-83444, carried into JSSG-2006): a 0.05 in corner crack at a
# hole, or a 0.05 in through crack where the wall there is no thicker; elsewhere a semicircular surface crack 0.125
# in long, or a 0.25 in long through crack where the wall is 0.125 in or thinner.
HOLE_FLAW, SURFACE_FLAW, THROUGH_FLAW = 1.27, 3.175, 6.35
_RULE = "the starting flaws of MIL-A-83444 and JSSG-2006"


def auto_crack(spec: CrackSpec, unc: Uncracked, origin: int, features: PartFeatures) -> CrackSpec:
    """The starting crack damage tolerance practice assumes at the origin, sized down to a fifth of the wall ahead
    when the part is too small for it, so the crack has room to grow."""
    from dataclasses import replace

    mesh = unc.mesh
    o = mesh.X[origin]
    size = float(np.nanmin(mesh.node_sizes()[[origin]]))
    inward = -_face_normals(mesh, o, 2.0 * size)[0]
    normal = np.asarray(spec.normal, float) if spec.normal is not None else principal_direction(unc.S[origin])[1]
    normal = normal / np.linalg.norm(normal)
    hole = _hole_at(spec, features, o)
    toward = spec.toward
    if hole is not None:
        wall = 2.0 * hole[3]
        ahead = _first(o, inward, mesh, skip=0.25 * size)
        a = min(HOLE_FLAW, 0.2 * ahead)
        if wall > HOLE_FLAW:
            shape, c = "corner", min(a, 0.4 * wall)
            why = f"a {fmt(a)} mm corner crack where the {fmt(2 * hole[2])} mm hole meets a face"
        else:
            shape, c = "through", None
            why = f"a {fmt(a)} mm crack through the {fmt(wall)} mm wall at the {fmt(2 * hole[2])} mm hole"
    else:
        wall = _first(o, inward, mesh, skip=0.25 * size)
        d = inward - (inward @ normal) * normal
        d = d / max(np.linalg.norm(d), 1e-12)
        w = np.cross(normal, d)
        if wall <= SURFACE_FLAW:
            # a thin wall loaded across its face: a crack through it, centered on the origin, along the wall
            along = np.cross(normal, inward)
            along = along / max(np.linalg.norm(along), 1e-12)
            mid = o + 0.5 * wall * inward
            room = min(_first(mid, along, mesh), _first(mid, -along, mesh))
            shape, a, c = "center", min(0.5 * THROUGH_FLAW, 0.2 * room), None
            toward = along.tolist()
            why = f"a {fmt(2 * a)} mm crack through the {fmt(wall)} mm wall"
        else:
            probe = o + 0.25 * SURFACE_FLAW * d
            width = _first(probe, w, mesh) + _first(probe, -w, mesh)
            if width >= 2.0 * SURFACE_FLAW:
                a = min(0.5 * SURFACE_FLAW, 0.2 * wall)
                shape, c = "surface", a
                why = f"a semicircular surface crack {fmt(a)} mm deep"
            else:
                shape, a, c = "through", min(0.5 * SURFACE_FLAW, 0.2 * wall), None
                why = f"a {fmt(a)} mm crack through the {fmt(width)} mm edge"
    rule = f"{why}, after {_RULE}"
    if a < (HOLE_FLAW if hole is not None else 0.5 * SURFACE_FLAW) - 1e-9:
        rule += ", made smaller to leave the crack room in this small part"
    return replace(spec, shape=shape, a=float(a), c=None if c is None else float(c), toward=toward,
                   text=f"{shape} {fmt(a)} mm (automatic)", reason=rule[0].upper() + rule[1:] + ".")


def _tensor(S) -> np.ndarray:
    sxx, syy, szz, sxy, syz, szx = S
    return np.array([[sxx, sxy, szx], [sxy, syy, syz], [szx, syz, szz]])


def _margin(mesh: Mesh, o: np.ndarray, out: np.ndarray, a: float, h_front: float) -> float:
    """How far a crack's mouth sits outside the part: a quarter of its size, at least two front elements, and
    under half the gap to the next wall that way, so it never reaches across a hole."""
    size = float(np.nanmin(mesh.node_sizes()[[int(np.argmin(np.linalg.norm(mesh.X - o, axis=1)))]]))
    gap = _first(o, out, mesh, skip=0.25 * size)
    return float(min(max(0.25 * a, 2.0 * h_front), 0.45 * gap if np.isfinite(gap) else math.inf))


def _hole_at(spec: CrackSpec, features: PartFeatures, o: np.ndarray):
    """The hole a crack starts from, as (axis, center, radius, half length), when it starts on one."""
    best = None
    for f in features.faces.values():
        if f.role != "hole" or f.axis is None or f.axis_point is None:
            continue
        ax = np.asarray(f.axis, float)
        ap = np.asarray(f.axis_point, float)
        rel = o - ap
        dist = abs(np.linalg.norm(rel - (rel @ ax) * ax) - f.radius)
        if dist < 0.05 * f.radius + 1e-6 and abs(rel @ ax) <= 0.5 * (f.length or 0) + 1e-6:
            if best is None or dist < best[0]:
                best = (dist, (ax, ap, f.radius, 0.5 * (f.length or 0.0)))
    return None if best is None else best[1]


def _describe(features: PartFeatures, mesh: Mesh, o: np.ndarray, hole) -> str:
    from ..structural.hotspots import nearest_face

    at = f"({', '.join(f'{v:.3g}' for v in o)})"
    if hole is not None:
        return f"at the edge of the {2 * hole[2]:.3g} mm hole at {at}"
    fid = nearest_face(mesh, o)
    f = features.faces.get(fid) if fid is not None else None
    if f is not None and f.role == "fillet":
        return f"on the {f.radius:.3g} mm fillet (face {f.id}) at {at}"
    return f"on face {fid} at {at}" if fid is not None else f"at {at}"


def stress_profile(unc: Uncracked, p: Placement, depth: float, k: int = 60) -> tuple[np.ndarray, np.ndarray]:
    """Stress normal to the crack plane before the crack, along its first growth direction from the origin."""
    from ..mesh.probe import sample

    x = np.linspace(0.0, depth, k)
    pts = p.origin + np.outer(np.maximum(x, 1e-4 * depth), p.toward)
    S, valid = sample(unc.mesh.X, unc.mesh.tets, unc.S, pts)
    n = p.normal
    sig = np.array([n @ _tensor(s) @ n for s in S])
    if not valid.all() and valid.any():
        sig = np.interp(x, x[valid], sig[valid])
    return x, sig


def section(unc: Uncracked, p: Placement) -> tuple[float, float]:
    """Area of the part's section in the crack plane around the origin, mm2, and the force across it, N."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    from ..verify.beam import cut_tets

    sn = np.array([p.normal @ _tensor(s) @ p.normal for s in unc.S])
    pts, faces, a, b, t = cut_tets(unc.mesh.X, unc.mesh.tets, p.normal, p.origin, quadratic=True)
    if len(faces) == 0:
        return 0.0, 0.0
    # cut points on the same mesh edge, or at the same node, are one point; keep the piece of the cut that holds
    # the point nearest the origin
    key = np.where((t <= 0)[:, None], np.column_stack([a, a]), np.where((t >= 1)[:, None], np.column_stack([b, b]),
                                                                          np.sort(np.column_stack([a, b]), axis=1)))
    _, pid = np.unique(key, axis=0, return_inverse=True)
    pid = pid.ravel()
    f = pid[faces]
    n = int(pid.max()) + 1
    links = coo_matrix((np.ones(2 * len(f)), (np.r_[f[:, 0], f[:, 1]], np.r_[f[:, 1], f[:, 2]])), shape=(n, n))
    _, part = connected_components(links, directed=False)
    keep = part[f[:, 0]] == part[pid[int(np.argmin(np.linalg.norm(pts - p.origin, axis=1)))]]
    tri = pts[faces[keep]]
    area = 0.5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)
    sv = sn[a] + t * (sn[b] - sn[a])
    return float(area.sum()), float((area * sv[faces[keep]].mean(1)).sum())
