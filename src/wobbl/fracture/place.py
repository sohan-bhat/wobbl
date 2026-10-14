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
            size = max(min(sizing.h / 5.0, 0.5 * crack.a, h / 2), 0.02 * sizing.h)
            sizing.refinements.append(Refinement(size, points=[tuple(mesh.X[origin])], radius=max(3 * crack.a, 4 * size),
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


def inside_test(mesh: Mesh):
    """A function telling which points lie inside the part, by the parity of boundary crossings along a ray."""
    X, tris = mesh.X, mesh.tris[:, :3]
    v = np.array([0.5773, 0.5779, 0.5769])
    v /= np.linalg.norm(v)

    def inside(points) -> np.ndarray:
        return np.array([len(ray_hits(np.asarray(p, float), v, X, tris)) % 2 == 1 for p in np.atleast_2d(points)],
                        bool)
    return inside


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
            probe = o + (0.5 * a if spec.shape == "through" else 0.0) * d
            lo, hi = _first(probe, -w, mesh), _first(probe, w, mesh)
            if not (np.isfinite(lo) and np.isfinite(hi)):
                raise SpecError("could not find the wall the crack runs through; give its normal and toward "
                                "directions")
            o = o + 0.5 * (hi - lo) * w
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
    import pyvista as pv

    from ..report.render import _grid

    x = np.linspace(0.0, depth, k)
    pts = p.origin + np.outer(np.maximum(x, 1e-4 * depth), p.toward)
    g = _grid(unc.mesh.X, unc.mesh.tets)
    g.point_data["S"] = unc.S
    got = pv.PolyData(pts).sample(g)
    S = np.asarray(got.point_data["S"])
    valid = np.asarray(got.point_data.get("vtkValidPointMask", np.ones(k)), bool)
    n = p.normal
    sig = np.array([n @ _tensor(s) @ n for s in S])
    if not valid.all() and valid.any():
        sig = np.interp(x, x[valid], sig[valid])
    return x, sig


def section(unc: Uncracked, p: Placement) -> tuple[float, float]:
    """Area of the part's section in the crack plane around the origin, mm2, and the force across it, N."""
    from ..report.render import _grid

    g = _grid(unc.mesh.X, unc.mesh.tets)
    g.point_data["sn"] = np.array([p.normal @ _tensor(s) @ p.normal for s in unc.S])
    try:
        sl = g.slice(normal=p.normal, origin=p.origin, generate_triangles=True)
        if sl.n_cells == 0:
            return 0.0, 0.0
        sl = sl.connectivity(extraction_mode="closest", closest_point=p.origin)
        sizes = sl.compute_cell_sizes(length=False, area=True, volume=False)
        area = np.asarray(sizes.cell_data["Area"])
        sn = np.asarray(sl.point_data_to_cell_data().cell_data["sn"])
    except Exception:  # noqa: BLE001  without a section, net section yield is not checked
        return 0.0, 0.0
    return float(area.sum()), float((area * sn).sum())
