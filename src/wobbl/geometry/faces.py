"""Describe a part the way an engineer would: faces, holes, fillets, sharp inside corners, wall thickness.

Faces and corners come from the boundary of a mesh, so STEP and STL parts get
the same treatment and normals are always outward. Wall thickness is sampled on
the CAD faces themselves when the part has any, so it does not move with the mesh.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np

from ..mesh.mesher import Mesh, SurfaceMesh


@dataclass
class FaceInfo:
    id: int
    kind: str                     # plane, cylinder, cone, sphere, torus, freeform
    cad_type: str
    area: float
    centroid: list[float]
    normal: list[float] | None    # outward unit normal for planes
    bbox: list[list[float]]
    radius: float | None = None   # cylinders, fillets and holes
    axis: list[float] | None = None
    axis_point: list[float] | None = None
    concave: bool = False         # curves into the material: holes, inside fillets
    length: float | None = None   # extent along the axis for cylinders
    role: str = ""                # hole, boss, fillet, flat, curved
    sweep: float = 0.0            # angle covered around the axis, radians
    group: int | None = None      # faces split from one hole or boss (common in STL) share a group id

    def to_dict(self) -> dict:
        d = asdict(self)
        for k in ("area", "radius", "length"):
            if d[k] is not None:
                d[k] = round(d[k], 4)
        for k in ("centroid", "normal", "axis", "axis_point"):
            if d[k] is not None:
                d[k] = [round(v, 4) for v in d[k]]
        d["bbox"] = [[round(v, 4) for v in row] for row in d["bbox"]]
        return d

    def describe(self) -> str:
        c = ", ".join(f"{v:.3g}" for v in self.centroid)
        if self.role == "hole":
            return f"face {self.id}: hole, diameter {2 * self.radius:.3g} mm, at ({c})"
        if self.role == "boss":
            return f"face {self.id}: outside cylinder, diameter {2 * self.radius:.3g} mm, at ({c})"
        if self.role == "fillet":
            return f"face {self.id}: inside fillet, radius {self.radius:.3g} mm, at ({c})"
        if self.kind == "plane":
            n = ", ".join(f"{v:+.2f}" for v in self.normal)
            return f"face {self.id}: flat, {self.area:.4g} mm2, normal ({n}), at ({c})"
        return f"face {self.id}: {self.kind}, {self.area:.4g} mm2, at ({c})"


@dataclass
class Corner:
    """A sharp inside corner (re-entrant edge): linear elastic stress there is singular."""
    faces: tuple[int, int]
    angle_deg: float              # angle between the face normals
    length: float
    midpoint: list[float]
    points: np.ndarray = field(repr=False, default=None)   # edge midpoints along the corner
    curves: list[int] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"faces": list(self.faces), "angle_deg": round(self.angle_deg, 1), "length": round(self.length, 3),
                "midpoint": [round(v, 3) for v in self.midpoint], "curves": self.curves}


@dataclass
class PartFeatures:
    faces: dict[int, FaceInfo]
    corners: list[Corner]
    thickness_typical: float
    thickness_min: float
    thickness_by_face: dict[int, float]
    surface_area: float
    volume: float
    bbox: list[list[float]]
    tangent_pairs: set = field(default_factory=set)

    def holes(self) -> list[FaceInfo]:
        """One entry per physical hole (split faces are represented by their first face)."""
        out, seen = [], set()
        for f in self.faces.values():
            if f.role == "hole" and (f.group is None or f.group not in seen):
                out.append(f)
                if f.group is not None:
                    seen.add(f.group)
        return out

    def fillets(self) -> list[FaceInfo]:
        return [f for f in self.faces.values() if f.role == "fillet"]

    def to_dict(self, max_faces: int = 200) -> dict:
        faces = sorted(self.faces.values(), key=lambda f: -f.area)[:max_faces]
        return {
            "bbox": self.bbox, "volume_mm3": round(self.volume, 3), "surface_area_mm2": round(self.surface_area, 3),
            "thickness_typical_mm": round(self.thickness_typical, 3), "thickness_min_mm": round(self.thickness_min, 3),
            "n_faces": len(self.faces),
            "holes": [{"face": h.id, "diameter": round(2 * h.radius, 3), "center": [round(v, 3) for v in h.axis_point],
                       "axis": [round(v, 3) for v in h.axis], "length": round(h.length or 0, 3)} for h in self.holes()],
            "fillets": [{"face": f.id, "radius": round(f.radius, 3)} for f in self.fillets()],
            "sharp_inside_corners": [c.to_dict() for c in self.corners],
            "faces": [f.to_dict() for f in faces],
        }


def _fit_cylinder(pts: np.ndarray, nrm: np.ndarray, w: np.ndarray, verts: np.ndarray):
    """Axis from the normals (they are all perpendicular to it), then a circle fit in the normal plane."""
    M = (nrm * w[:, None]).T @ nrm
    vals, vecs = np.linalg.eigh(M)
    axis = vecs[:, 0]
    if vals[0] > 0.02 * vals.sum():
        return None
    # project points to the plane through the centroid perpendicular to the axis
    c0 = np.average(pts, axis=0, weights=w)
    u = np.cross(axis, [1, 0, 0] if abs(axis[0]) < 0.9 else [0, 1, 0])
    u /= np.linalg.norm(u)
    v = np.cross(axis, u)
    qv = verts - c0  # mesh vertices lie exactly on the surface, so fit the circle to them
    x, y = qv @ u, qv @ v
    A = np.column_stack([x, y, np.ones_like(x)])
    b = x ** 2 + y ** 2
    sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    cx, cy = sol[0] / 2, sol[1] / 2
    r2 = sol[2] + cx ** 2 + cy ** 2
    if r2 <= 0:
        return None
    r = float(np.sqrt(r2))
    resid = np.abs(np.hypot(x - cx, y - cy) - r)
    if resid.mean() > 0.03 * r + 1e-6:
        return None
    q = pts - c0
    center = c0 + cx * u + cy * v
    radial = pts - center
    radial -= np.outer(radial @ axis, axis)
    radial /= np.maximum(np.linalg.norm(radial, axis=1), 1e-12)[:, None]
    concave = float(np.average(np.einsum("ij,ij->i", radial, nrm), weights=w)) < 0
    t = qv @ axis  # extent along the axis from the vertices, which sit on the true edges
    center = center + axis * (0.5 * (t.max() + t.min()))
    sweep = _sweep_angle(radial, axis)
    return axis, center, r, concave, float(t.max() - t.min()), sweep


def _sweep_angle(radial: np.ndarray, axis: np.ndarray) -> float:
    u = radial[0]
    v = np.cross(axis, u)
    ang = np.sort(np.arctan2(radial @ v, radial @ u))
    gaps = np.diff(np.concatenate([ang, [ang[0] + 2 * np.pi]]))
    return float(2 * np.pi - gaps.max())


def analyze_faces(mesh: Mesh) -> dict[int, FaceInfo]:
    n, a, c = mesh.tri_geometry()
    out: dict[int, FaceInfo] = {}
    for s in mesh.surfaces():
        if s < 0:
            continue
        m = mesh.tri_surf == s
        w, nn, cc = a[m], n[m], c[m]
        area = float(w.sum())
        cen = (cc * w[:, None]).sum(0) / area
        nmean = (nn * w[:, None]).sum(0)
        nlen = np.linalg.norm(nmean) / area
        pts = mesh.X[np.unique(mesh.tris[m].ravel())]
        bbox = [pts.min(0).tolist(), pts.max(0).tolist()]
        cad = mesh.surface_types.get(int(s), "Discrete")
        info = FaceInfo(int(s), "freeform", cad, area, cen.tolist(), None, bbox)
        if nlen > 0.9995 or cad == "Plane":
            info.kind = "plane"
            info.normal = (nmean / np.linalg.norm(nmean)).tolist()
            info.role = "flat"
        else:
            fit = _fit_cylinder(cc, nn, w, pts) if len(cc) >= 6 else None
            if fit is not None:
                axis, center, r, concave, length, sweep = fit
                info.kind = "cylinder" if cad in ("Cylinder", "Discrete", "BSplineSurface") else cad.lower()
                info.radius, info.axis, info.axis_point = r, axis.tolist(), center.tolist()
                info.concave, info.length, info.sweep = concave, length, sweep
                if concave and sweep > 1.5 * np.pi:
                    info.role = "hole"
                elif concave:
                    info.role = "fillet"
                elif sweep > 1.5 * np.pi:
                    info.role = "boss"
                else:
                    info.role = "curved"
            else:
                info.kind = {"Cone": "cone", "Sphere": "sphere", "Torus": "torus"}.get(cad, "freeform")
                info.role = "curved"
        out[int(s)] = info
    _group_split_cylinders(out)
    return out


def _group_split_cylinders(faces: dict[int, FaceInfo]):
    """STL patches and some CAD exports split one hole into several partial cylinders; join them back."""
    cyl = [f for f in faces.values() if f.radius is not None and f.role in ("fillet", "curved")]
    used = set()
    for f in cyl:
        if f.id in used:
            continue
        a0, p0 = np.asarray(f.axis), np.asarray(f.axis_point)
        same = [f]
        for g in cyl:
            if g.id == f.id or g.id in used or g.concave != f.concave:
                continue
            a1, p1 = np.asarray(g.axis), np.asarray(g.axis_point)
            d = p1 - p0
            off = np.linalg.norm(d - np.dot(d, a0) * a0)
            if abs(np.dot(a0, a1)) > 0.995 and abs(g.radius - f.radius) < 0.03 * f.radius and off < 0.05 * f.radius:
                same.append(g)
        if len(same) > 1 and sum(x.sweep for x in same) > 1.5 * np.pi:
            for x in same:
                used.add(x.id)
                x.role = "hole" if f.concave else "boss"
                x.group = f.id


def _exact_normals(face_ids: np.ndarray, pts: np.ndarray, tri_n: np.ndarray, faces) -> np.ndarray:
    """CAD normals where the geometry is known: planes have one normal, cylinders point along the radius.

    Coarse meshes facet small fillets so badly that the triangles next to a tangent edge look like a sharp
    corner; the exact normals of a fillet and its neighbour agree at the tangent line.
    """
    out = tri_n.copy()
    if not faces:
        return out
    for k, (fid, p) in enumerate(zip(face_ids, pts)):
        f = faces.get(int(fid))
        if f is None:
            continue
        if f.kind == "plane" and f.normal is not None:
            out[k] = f.normal
        elif f.axis is not None and f.axis_point is not None:
            a = np.asarray(f.axis)
            r = p - np.asarray(f.axis_point)
            r -= (r @ a) * a
            nr = np.linalg.norm(r)
            if nr > 0:
                r /= nr
                out[k] = -r if f.concave else r
    return out


def find_sharp_corners(mesh: Mesh, min_angle_deg: float = 30.0, faces: dict | None = None,
                       tangent: set | None = None) -> list[Corner]:
    """Edges where two faces meet at an inside (re-entrant) angle without a fillet.

    ``tangent`` lists face pairs known from the CAD to meet smoothly; they are never sharp corners.
    """
    tri = mesh.tris[:, :3]
    k = len(tri)
    e = np.vstack([tri[:, [0, 1]], tri[:, [1, 2]], tri[:, [2, 0]]])
    owner = np.tile(np.arange(k), 3)
    third = np.concatenate([tri[:, 2], tri[:, 0], tri[:, 1]])
    es = np.sort(e, axis=1)
    key = es[:, 0].astype(np.int64) * len(mesh.X) + es[:, 1]
    order = np.argsort(key, kind="stable")
    ks = key[order]
    pair = np.nonzero(ks[1:] == ks[:-1])[0]
    i1, i2 = order[pair], order[pair + 1]
    t1, t2 = owner[i1], owner[i2]
    diff = mesh.tri_surf[t1] != mesh.tri_surf[t2]
    if tangent:
        a, b = np.minimum(mesh.tri_surf[t1], mesh.tri_surf[t2]), np.maximum(mesh.tri_surf[t1], mesh.tri_surf[t2])
        smooth = np.array([(int(x), int(y)) in tangent for x, y in zip(a, b)], bool)
        diff &= ~smooth
    i1, i2, t1, t2 = i1[diff], i2[diff], t1[diff], t2[diff]
    if len(i1) == 0:
        return []
    n, a, c = mesh.tri_geometry()
    n1, n2 = n[t1], n[t2]
    p0, p1 = mesh.X[es[i1, 0]], mesh.X[es[i1, 1]]
    mid = 0.5 * (p0 + p1)
    edir = (p1 - p0) / np.maximum(np.linalg.norm(p1 - p0, axis=1), 1e-12)[:, None]
    v1 = mesh.X[third[i1]] - mid
    v1 -= np.einsum("ij,ij->i", v1, edir)[:, None] * edir
    v1 /= np.maximum(np.linalg.norm(v1, axis=1), 1e-12)[:, None]
    if faces:
        n1 = _exact_normals(mesh.tri_surf[t1], mid, n1, faces)
        n2 = _exact_normals(mesh.tri_surf[t2], mid, n2, faces)
    concave = np.einsum("ij,ij->i", v1, n2) > 0.15
    ang = np.degrees(np.arccos(np.clip(np.einsum("ij,ij->i", n1, n2), -1, 1)))
    sharp = concave & (ang > min_angle_deg)
    corners: list[Corner] = []
    if not sharp.any():
        return corners
    sp = np.sort(np.column_stack([mesh.tri_surf[t1[sharp]], mesh.tri_surf[t2[sharp]]]), axis=1)
    lens = np.linalg.norm(p1 - p0, axis=1)[sharp]
    mids = mid[sharp]
    angs = ang[sharp]
    for pr in np.unique(sp, axis=0):
        m = (sp == pr).all(1)
        L = float(lens[m].sum())
        pts = mids[m]
        cm = pts[np.argmin(np.linalg.norm(pts - pts.mean(0), axis=1))]
        shared = sorted(set(mesh.surface_curves.get(int(pr[0]), [])) & set(mesh.surface_curves.get(int(pr[1]), [])))
        corners.append(Corner((int(pr[0]), int(pr[1])), float(np.median(angs[m])), L, cm.tolist(), pts,
                              [abs(x) for x in shared]))
    corners.sort(key=lambda c: -c.length)
    return corners


def ray_hits(orig: np.ndarray, dirs: np.ndarray, V0: np.ndarray, V1: np.ndarray, V2: np.ndarray,
             skip: np.ndarray | None = None, chunk: int = 4_000_000,
             exit_normals: np.ndarray | None = None) -> np.ndarray:
    """Distance to the first triangle hit along each ray (Moller-Trumbore, vectorized). inf when nothing is hit.

    With ``exit_normals`` (outward triangle normals) only hits where the ray leaves the solid count, so a ray
    starting a hair outside a curved face skips the facet it starts behind and finds the far wall.
    """
    R, T = len(orig), len(V0)
    e1, e2 = V1 - V0, V2 - V0
    best = np.full(R, np.inf)
    step = max(1, chunk // max(T, 1))
    for r0 in range(0, R, step):
        o = orig[r0:r0 + step, None, :]
        d = dirs[r0:r0 + step, None, :]
        p = np.cross(d, e2[None])
        det = np.einsum("rtk,tk->rt", p, e1)
        ok = np.abs(det) > 1e-12
        inv = np.where(ok, 1.0 / np.where(ok, det, 1.0), 0.0)
        s = o - V0[None]
        u = np.einsum("rtk,rtk->rt", s, p) * inv
        q = np.cross(s, e1[None])
        v = np.einsum("rk,rtk->rt", dirs[r0:r0 + step], q) * inv
        t = np.einsum("tk,rtk->rt", e2, q) * inv
        hit = ok & (u >= -1e-9) & (v >= -1e-9) & (u + v <= 1 + 1e-9) & (t > 1e-9)
        if exit_normals is not None:
            hit &= np.einsum("rk,tk->rt", dirs[r0:r0 + step], exit_normals) > 0
        if skip is not None:
            hit[np.arange(hit.shape[0]), skip[r0:r0 + step]] = False
        t = np.where(hit, t, np.inf)
        best[r0:r0 + step] = t.min(axis=1)
    return best


def triangle_samples(mesh, n: int = 800) -> dict:
    """Thickness sample points from the triangles themselves, for parts with no CAD faces (STL): triangle
    centroids picked in proportion to area, evenly and the same way every time."""
    from ..mesh.mesher import systematic_pick

    nrm, a, c = mesh.tri_geometry()
    pick, count = systematic_pick(a, n)
    return {"pts": c[pick], "normals": nrm[pick], "weights": count * a.sum() / n, "faces": mesh.tri_surf[pick]}


def wall_thickness(mesh: Mesh, samples: int = 800) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Cast rays inward from points spread over the surface; return (face ids, thickness, area weights).

    The points come from the CAD faces when the surface mesh carries them, so the answer depends on the
    geometry and not on how the triangles happened to fall.
    """
    smp = getattr(mesh, "samples", None) or triangle_samples(mesh, samples)
    n, _, _ = mesh.tri_geometry()
    tri = mesh.tris[:, :3]
    t = exit_distances(smp["pts"], -smp["normals"], mesh.X[tri[:, 0]], mesh.X[tri[:, 1]], mesh.X[tri[:, 2]], n)
    ok = np.isfinite(t)
    return smp["faces"][ok], t[ok], smp["weights"][ok]


def exit_distances(orig, dirs, V0, V1, V2, nrm) -> np.ndarray:
    """Distance along each ray to where it leaves the solid, the same as ``ray_hits`` with exit normals.

    Most walls are thin next to the part, so each ray first tests only the triangles that could lie within a
    short reach of its start (a hit at distance t lies within t of the start, so nothing closer is missed).
    Rays that find nothing that close try a longer reach and then every triangle.
    """
    from scipy.spatial import cKDTree

    cen = (V0 + V1 + V2) / 3.0
    rad = float(max(np.linalg.norm(V - cen, axis=1).max() for V in (V0, V1, V2)))
    diag = float(np.linalg.norm(np.ptp(np.vstack([V0, V1, V2]), axis=0)))
    tree = cKDTree(cen)
    out = np.full(len(orig), np.inf)
    todo = np.arange(len(orig))
    for reach in (0.08 * diag, 0.25 * diag):
        for i, cand in zip(todo, tree.query_ball_point(orig[todo], reach + rad)):
            if cand:
                c = np.asarray(cand)
                t = ray_hits(orig[i:i + 1], dirs[i:i + 1], V0[c], V1[c], V2[c], exit_normals=nrm[c])[0]
                if t <= reach:
                    out[i] = t
        todo = todo[~np.isfinite(out[todo])]
        if not len(todo):
            return out
    out[todo] = ray_hits(orig[todo], dirs[todo], V0, V1, V2, exit_normals=nrm)
    return out


def thickness_stats(th: np.ndarray, w: np.ndarray, thin: float = 0.05) -> tuple[float, float]:
    """(typical, thinnest) wall from weighted thickness samples.

    Typical is the area weighted median. Thinnest is the area weighted mean of the thinnest ``thin`` share of
    the surface rather than a quantile: a quantile jumps between two wall sizes when the thin share sits near
    the cut (3 mm or 5.5 mm on the cast motor mount), while the mean of the tail moves smoothly.
    """
    order = np.argsort(th, kind="stable")
    th, w = th[order], w[order]
    cw = np.cumsum(w) / w.sum()
    typical = float(th[min(np.searchsorted(cw, 0.5), len(th) - 1)])
    share = np.minimum(w, np.maximum(thin * w.sum() - (np.cumsum(w) - w), 0.0))
    return typical, float((share * th).sum() / share.sum())


def analyze(mesh: Mesh) -> PartFeatures:
    faces = analyze_faces(mesh)
    tangent = getattr(mesh, "tangent_pairs", set())
    corners = find_sharp_corners(mesh, faces=faces, tangent=tangent)
    fs, th, w = wall_thickness(mesh)
    by_face = {}
    for s in np.unique(fs):
        by_face[int(s)] = float(np.median(th[fs == s]))
    if len(th):
        typical, tmin = thickness_stats(th, w)
    else:
        typical = tmin = float(np.min(mesh.X.max(0) - mesh.X.min(0)))
    _, a, _ = mesh.tri_geometry()
    return PartFeatures(faces, corners, typical, tmin, by_face, float(a.sum()), mesh.volume(),
                        [mesh.X.min(0).round(4).tolist(), mesh.X.max(0).round(4).tolist()], tangent)
