"""Tetrahedral meshing with gmsh: second order tets, curvature aware sizing and local refinement.

The result is a solver neutral ``Mesh``: node coordinates, 10 node tets in
CalculiX order, and the outer boundary as 6 node triangles that remember which
CAD surface, element and element face they came from. Loads, fixtures and
heat boundary conditions are all built from that boundary.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field, replace

import numpy as np

from ..geometry.cad import GeometryError, PreparedGeometry, gmsh_session, load_into_session

# gmsh tet10 node order -> CalculiX C3D10 order
GMSH_TO_CCX_TET10 = np.array([0, 1, 2, 3, 4, 5, 6, 7, 9, 8])
# CalculiX tet faces: corner indices, midside indices, opposite corner
TET_FACES = [((0, 1, 2), (4, 5, 6), 3), ((0, 3, 1), (7, 8, 4), 2),
             ((1, 3, 2), (8, 9, 5), 0), ((2, 3, 0), (9, 7, 6), 1)]


class MeshError(RuntimeError):
    pass


@dataclass
class Refinement:
    """Make elements no larger than ``size`` near some geometry, growing back to the global size over ``grow``."""
    size: float
    surfaces: list[int] = field(default_factory=list)
    curves: list[int] = field(default_factory=list)
    points: list[tuple[float, float, float]] = field(default_factory=list)
    radius: float = 0.0          # for points: radius of the fine ball
    grow: float = 0.0            # transition distance (default 6 x size)
    label: str = ""
    corners_only: bool = False   # size the CAD vertices of the surfaces instead of a distance field (cheap)


@dataclass
class Sizing:
    h: float                                  # global element size, mm
    refinements: list[Refinement] = field(default_factory=list)
    order: int = 2
    curvature: int = 16                        # elements per full circle on curved geometry (0 = off)
    h_min: float = 0.0


@dataclass
class Mesh:
    X: np.ndarray                  # (n,3) node coordinates, mm
    tets: np.ndarray               # (m,10) or (m,4), CalculiX order, 0 based
    tris: np.ndarray               # (k,6) or (k,3) boundary faces, outward ordered
    tri_surf: np.ndarray           # (k,) CAD surface tag of each boundary face
    tri_elem: np.ndarray           # (k,) element index
    tri_face: np.ndarray           # (k,) CalculiX face number 1..4
    order: int
    sizing: Sizing | None = None
    t_mesh: float = 0.0
    notes: list[str] = field(default_factory=list)
    surface_types: dict[int, str] = field(default_factory=dict)
    surface_curves: dict[int, list[int]] = field(default_factory=dict)

    @property
    def n_nodes(self) -> int:
        return len(self.X)

    @property
    def n_elems(self) -> int:
        return len(self.tets)

    @property
    def tri_corners(self) -> np.ndarray:
        return self.tris[:, :3]

    def tri_geometry(self, mask=None):
        """Outward unit normals, areas and centroids of boundary faces (flat corner triangles)."""
        t = self.tris[:, :3] if mask is None else self.tris[mask, :3]
        p0, p1, p2 = self.X[t[:, 0]], self.X[t[:, 1]], self.X[t[:, 2]]
        c = np.cross(p1 - p0, p2 - p0)
        a = 0.5 * np.linalg.norm(c, axis=1)
        n = c / np.maximum(2 * a, 1e-300)[:, None]
        return n, a, (p0 + p1 + p2) / 3.0

    def surface_mask(self, surfaces) -> np.ndarray:
        return np.isin(self.tri_surf, np.asarray(list(surfaces), dtype=int))

    def surface_nodes(self, surfaces) -> np.ndarray:
        m = self.surface_mask(surfaces)
        return np.unique(self.tris[m].ravel())

    def surfaces(self) -> np.ndarray:
        return np.unique(self.tri_surf)

    def element_sizes(self) -> np.ndarray:
        """Characteristic size (mean corner edge length) of each element."""
        c = self.X[self.tets[:, :4]]
        e = [np.linalg.norm(c[:, i] - c[:, j], axis=1) for i, j in ((0, 1), (1, 2), (2, 0), (0, 3), (1, 3), (2, 3))]
        return np.mean(e, axis=0)

    def node_sizes(self) -> np.ndarray:
        """Local element size at each node (smallest adjacent element)."""
        hs = self.element_sizes()
        out = np.full(len(self.X), np.inf)
        for k in range(self.tets.shape[1]):
            np.minimum.at(out, self.tets[:, k], hs)
        out[~np.isfinite(out)] = np.nan
        return out

    def volume(self) -> float:
        c = self.X[self.tets[:, :4]]
        v = np.einsum("ij,ij->i", c[:, 1] - c[:, 0], np.cross(c[:, 2] - c[:, 0], c[:, 3] - c[:, 0])) / 6.0
        return float(np.abs(v).sum())


def _boundary(tets: np.ndarray, X: np.ndarray, order: int):
    m = len(tets)
    n = len(X)
    corners = []
    mids = []
    opp = []
    fid = []
    for f, (c, md, o) in enumerate(TET_FACES):
        corners.append(tets[:, list(c)])
        if order == 2:
            mids.append(tets[:, list(md)])
        opp.append(tets[:, o])
        fid.append(np.full(m, f + 1))
    C = np.concatenate(corners)
    O = np.concatenate(opp)
    F = np.concatenate(fid)
    E = np.tile(np.arange(m), 4)
    M = np.concatenate(mids) if order == 2 else None
    s = np.sort(C, axis=1).astype(np.int64)
    key = (s[:, 0] * n + s[:, 1]) * n + s[:, 2]
    uniq, inv, cnt = np.unique(key, return_inverse=True, return_counts=True)
    once = cnt[inv] == 1
    C, O, F, E, key = C[once], O[once], F[once], E[once], key[once]
    if M is not None:
        M = M[once]
    p0, p1, p2 = X[C[:, 0]], X[C[:, 1]], X[C[:, 2]]
    nrm = np.cross(p1 - p0, p2 - p0)
    inward = np.einsum("ij,ij->i", nrm, X[O] - p0) > 0
    C[inward] = C[inward][:, [0, 2, 1]]
    if M is not None:
        M[inward] = M[inward][:, [2, 1, 0]]
        tris = np.hstack([C, M])
    else:
        tris = C
    return tris, E, F, key


def estimate_nodes(prep: PreparedGeometry, sizing: Sizing) -> float:
    vol = max(prep.volume, 1e-9)
    est = 1.45 * vol / (sizing.h ** 3 / 6.5)
    for r in sizing.refinements:
        rad = (r.radius or 0) + (r.grow or 6 * r.size)
        local = (4 / 3) * np.pi * rad ** 3 * max(1, len(r.points)) if r.points else 0.25 * vol
        est += 1.45 * min(local, vol) / (r.size ** 3 / 6.5) * 0.3
    return est


def build_mesh(prep: PreparedGeometry, sizing: Sizing, verbose: bool = False, timeout: float | None = None) -> Mesh:
    """Mesh the prepared geometry with the given sizing.

    Each attempt runs in a child process; if gmsh hangs or crashes, the next, more conservative
    strategy is tried before giving up.
    """
    from .isolate import IsolatedCrash, IsolatedTimeout, run_isolated

    if timeout is None:
        # generous on purpose: a timeout moves on to the next strategy, which gives a different mesh, so it must
        # only catch a hung mesher and never a busy machine (serial meshing runs about 15k nodes a second)
        timeout = 90.0 + estimate_nodes(prep, sizing) / 750.0
    attempts = [dict(alg2d=6, alg3d=10, curved=True), dict(alg2d=5, alg3d=1, curved=True),
                dict(alg2d=1, alg3d=1, curved=False)]
    errors = []
    # a size that is coarse next to a thin curved wall can make the surface mesh fold through itself; when
    # every strategy fails, a finer size usually meshes cleanly (the refinements too, since a coarse feature
    # refinement can be the size that folds)
    for scale in (1.0, 0.75, 0.55):
        trial = sizing if scale == 1.0 else Sizing(
            h=sizing.h * scale, refinements=[replace(r, size=r.size * scale) for r in sizing.refinements],
            order=sizing.order, curvature=sizing.curvature, h_min=sizing.h_min)
        for i, att in enumerate(attempts if scale == 1.0 else attempts[1:]):
            try:
                mesh = run_isolated(_build, prep, trial, verbose, timeout=timeout, **att)
                if i > 0 or scale != 1.0:
                    mesh.notes.append("Meshing needed a fallback strategy" +
                                      ("" if att["curved"] else " (straight sided second order elements)") +
                                      (f" and a {scale:g}x finer size" if scale != 1.0 else "") + ".")
                return mesh
            except (MeshError, IsolatedCrash, IsolatedTimeout) as e:
                errors.append(str(e))
    raise MeshError(f"gmsh could not mesh this part ({'; '.join(errors[:3])}). Check the CAD for self "
                    "intersections and gaps, or pass --mesh-size.")


def flat_elements(X: np.ndarray, tets: np.ndarray, rel: float = 1e-9) -> int:
    """Count tetrahedra whose four corners lie in one plane, judged against their longest edge."""
    c = X[tets[:, :4]]
    a, b, d = c[:, 1] - c[:, 0], c[:, 2] - c[:, 0], c[:, 3] - c[:, 0]
    vol = np.abs(np.einsum("ij,ij->i", a, np.cross(b, d))) / 6.0
    edge = np.max([np.linalg.norm(c[:, i] - c[:, j], axis=1)
                   for i, j in ((0, 1), (1, 2), (2, 0), (0, 3), (1, 3), (2, 3))], axis=0)
    return int((vol <= rel * edge ** 3).sum())


def _sampling(gmsh, r: Refinement) -> int:
    """Distance fields sample their entities on a grid; a long fillet refined to a fine size needs samples
    closer than the target size, or the fine band breaks up between the samples."""
    span = 0.0
    for dim, tags in ((2, r.surfaces), (1, r.curves)):
        for t in tags:
            try:
                bb = gmsh.model.getBoundingBox(dim, int(t))
                span = max(span, float(np.linalg.norm(np.array(bb[3:]) - np.array(bb[:3]))))
            except Exception:  # noqa: BLE001
                pass
    return int(np.clip(span / max(r.size, 1e-9), 20, 400))


def _add_fields(gmsh, sizing: Sizing):
    fields = []
    point_size: dict[int, float] = {}
    for r in sizing.refinements:
        grow = r.grow or 6.0 * r.size
        if r.corners_only:
            # measure distance to the edges of the faces instead of the faces themselves: gmsh evaluates
            # the field thousands of times per curve, and edges are far cheaper to sample than surfaces
            curves = set()
            for s in r.surfaces:
                try:
                    for dim, tag in gmsh.model.getBoundary([(2, int(s))], combined=False, oriented=False):
                        if dim == 1:
                            curves.add(abs(int(tag)))
                except Exception:  # noqa: BLE001
                    continue
            if not curves:
                continue
            d = gmsh.model.mesh.field.add("Distance")
            gmsh.model.mesh.field.setNumbers(d, "CurvesList", sorted(curves))
            gmsh.model.mesh.field.setNumber(d, "Sampling", 20)
            t = gmsh.model.mesh.field.add("Threshold")
            gmsh.model.mesh.field.setNumber(t, "InField", d)
            gmsh.model.mesh.field.setNumber(t, "SizeMin", r.size)
            gmsh.model.mesh.field.setNumber(t, "SizeMax", sizing.h)
            gmsh.model.mesh.field.setNumber(t, "DistMin", 0.5 * r.size)
            gmsh.model.mesh.field.setNumber(t, "DistMax", 0.5 * r.size + grow)
            fields.append(t)
            continue
        if r.surfaces or r.curves:
            d = gmsh.model.mesh.field.add("Distance")
            if r.surfaces:
                gmsh.model.mesh.field.setNumbers(d, "SurfacesList", list(map(int, r.surfaces)))
            if r.curves:
                gmsh.model.mesh.field.setNumbers(d, "CurvesList", list(map(int, r.curves)))
            gmsh.model.mesh.field.setNumber(d, "Sampling", _sampling(gmsh, r))
            t = gmsh.model.mesh.field.add("Threshold")
            gmsh.model.mesh.field.setNumber(t, "InField", d)
            gmsh.model.mesh.field.setNumber(t, "SizeMin", r.size)
            gmsh.model.mesh.field.setNumber(t, "SizeMax", sizing.h)
            gmsh.model.mesh.field.setNumber(t, "DistMin", r.radius)
            gmsh.model.mesh.field.setNumber(t, "DistMax", r.radius + grow)
            fields.append(t)
        for p in r.points:
            b = gmsh.model.mesh.field.add("Ball")
            gmsh.model.mesh.field.setNumber(b, "VIn", r.size)
            gmsh.model.mesh.field.setNumber(b, "VOut", sizing.h)
            gmsh.model.mesh.field.setNumber(b, "Radius", max(r.radius, 2 * r.size))
            gmsh.model.mesh.field.setNumber(b, "Thickness", grow)
            gmsh.model.mesh.field.setNumber(b, "XCenter", float(p[0]))
            gmsh.model.mesh.field.setNumber(b, "YCenter", float(p[1]))
            gmsh.model.mesh.field.setNumber(b, "ZCenter", float(p[2]))
            fields.append(b)
    if point_size:
        gmsh.model.mesh.setSize([(0, t) for _, t in gmsh.model.getEntities(0)], sizing.h)
        for tag, size in point_size.items():
            gmsh.model.mesh.setSize([(0, tag)], size)
        gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 1)
    if fields:
        fm = gmsh.model.mesh.field.add("Min")
        gmsh.model.mesh.field.setNumbers(fm, "FieldsList", fields)
        gmsh.model.mesh.field.setAsBackgroundMesh(fm)


def _build(prep: PreparedGeometry, sizing: Sizing, verbose: bool, alg2d: int, alg3d: int, curved: bool) -> Mesh:
    t0 = time.time()
    with gmsh_session(verbose) as gmsh:
        load_into_session(gmsh, prep)
        opt = gmsh.option.setNumber
        opt("Mesh.MeshSizeMax", sizing.h)
        opt("Mesh.MeshSizeMin", sizing.h_min or 0.0)
        opt("Mesh.MeshSizeFromPoints", 0)
        opt("Mesh.MeshSizeFromCurvature", sizing.curvature)
        opt("Mesh.MeshSizeExtendFromBoundary", 1)
        opt("Mesh.Algorithm", alg2d)
        opt("Mesh.Algorithm3D", alg3d)
        opt("Mesh.Optimize", 1)
        opt("Mesh.ElementOrder", 1)
        if prep.kind == "stl":
            opt("Mesh.MeshSizeFromCurvature", 0)
        _add_fields(gmsh, sizing)
        try:
            gmsh.model.mesh.generate(3)
        except Exception as e:  # noqa: BLE001
            raise MeshError(str(e)) from e
        if sizing.order == 2:
            opt("Mesh.SecondOrderLinear", 0 if (curved and prep.kind != "stl") else 1)
            try:
                gmsh.model.mesh.setOrder(2)
            except Exception as e:  # noqa: BLE001
                raise MeshError(f"second order conversion failed: {e}") from e
        etype = 11 if sizing.order == 2 else 4
        types, etags, enodes = gmsh.model.mesh.getElements(3)
        if etype not in list(types):
            raise MeshError("no tetrahedra were generated")
        k = list(types).index(etype)
        straightened = 0
        if sizing.order == 2 and curved:
            q = np.asarray(gmsh.model.mesh.getElementQualities(etags[k], "minDetJac"))
            if (q <= 0).any():
                # curving midside nodes onto the CAD folded some thin elements. High order optimization can
                # take minutes on thin walls; straight sided quadratic elements are always valid and cost nothing
                straightened = int((q <= 0).sum())
                gmsh.model.mesh.setOrder(1)
                opt("Mesh.SecondOrderLinear", 1)
                gmsh.model.mesh.setOrder(2)
                types, etags, enodes = gmsh.model.mesh.getElements(3)
                k = list(types).index(etype)
        ntags, coords, _ = gmsh.model.mesh.getNodes()
        nper = 10 if sizing.order == 2 else 4
        raw = np.asarray(enodes[k], dtype=np.int64).reshape(-1, nper)
        surf_keys = []
        stypes, scurves = {}, {}
        for _, s in gmsh.model.getEntities(2):
            stypes[s] = gmsh.model.getType(2, s)
            scurves[s] = [int(c) for c in gmsh.model.getAdjacencies(2, s)[1]]
            st, _, sn = gmsh.model.mesh.getElements(2, s)
            for tt, nn in zip(st, sn):
                if tt in (2, 9):
                    w = 3 if tt == 2 else 6
                    surf_keys.append((s, np.asarray(nn, dtype=np.int64).reshape(-1, w)[:, :3]))
    # compact numbering: only nodes used by tets
    tagmax = int(max(ntags.max(), raw.max())) + 1
    lut = np.full(tagmax, -1, dtype=np.int64)
    used = np.unique(raw)
    lut[used] = np.arange(len(used))
    pos = np.full(tagmax, -1, dtype=np.int64)
    pos[np.asarray(ntags, dtype=np.int64)] = np.arange(len(ntags))
    Xall = np.asarray(coords).reshape(-1, 3)
    X = Xall[pos[used]]
    tets = lut[raw]
    if sizing.order == 2:
        tets = tets[:, GMSH_TO_CCX_TET10]
    flat = flat_elements(X, tets)
    if flat:
        # straightening cannot fix these: they have no volume however the midside nodes sit, and the solver
        # rejects them, so let the next meshing strategy try
        raise MeshError(f"the mesher left {flat} flat elements with all four corners in one plane")
    tris, E, F, key = _boundary(tets, X, sizing.order)
    # map boundary faces to CAD surfaces
    n = len(X)
    tri_surf = np.full(len(tris), -1, dtype=np.int64)
    if surf_keys:
        ks, ss = [], []
        for s, corners in surf_keys:
            c = np.sort(lut[corners], axis=1)
            ks.append((c[:, 0] * n + c[:, 1]) * n + c[:, 2])
            ss.append(np.full(len(c), s))
        ks = np.concatenate(ks)
        ss = np.concatenate(ss)
        order = np.argsort(ks)
        ks, ss = ks[order], ss[order]
        idx = np.clip(np.searchsorted(ks, key), 0, len(ks) - 1)
        hit = ks[idx] == key
        tri_surf[hit] = ss[idx[hit]]
    mesh = Mesh(X, tets, tris, tri_surf, E, F, sizing.order, sizing, time.time() - t0,
                surface_types=stypes, surface_curves=scurves)
    if straightened:
        mesh.notes.append(f"{straightened} curved elements folded on thin walls, so this mesh uses straight sided "
                          "second order elements.")
    if (tri_surf < 0).any():
        mesh.notes.append(f"{int((tri_surf < 0).sum())} boundary faces could not be matched to a CAD surface.")
    if len(tets) == 0:
        raise GeometryError("the mesh is empty")
    return mesh


@dataclass
class SurfaceMesh:
    """Boundary only mesh with outward oriented triangles. Cheap to build, used to describe a part."""
    X: np.ndarray
    tris: np.ndarray
    tri_surf: np.ndarray
    surface_types: dict[int, str] = field(default_factory=dict)
    surface_curves: dict[int, list[int]] = field(default_factory=dict)
    t_mesh: float = 0.0
    tangent_pairs: set = field(default_factory=set)   # surface pairs that meet smoothly (a fillet and its wall)
    samples: dict | None = None   # points on the CAD faces for wall thickness: pts, normals, weights, faces

    tri_geometry = Mesh.tri_geometry
    surface_mask = Mesh.surface_mask
    surface_nodes = Mesh.surface_nodes
    surfaces = Mesh.surfaces

    def volume(self) -> float:
        p0, p1, p2 = (self.X[self.tris[:, i]] for i in range(3))
        return float(abs(np.einsum("ij,ij->i", p0, np.cross(p1, p2)).sum() / 6.0))


def _count_hits(o, d, V0, e1, e2):
    p = np.cross(d, e2)
    det = np.einsum("tk,tk->t", np.broadcast_to(p, e1.shape), e1)
    ok = np.abs(det) > 1e-14
    inv = np.where(ok, 1.0 / np.where(ok, det, 1.0), 0.0)
    sv = o - V0
    u = np.einsum("tk,tk->t", sv, np.broadcast_to(p, e1.shape)) * inv
    q = np.cross(sv, e1)
    v = (q @ d) * inv
    t = np.einsum("tk,tk->t", e2, q) * inv
    return int((ok & (u >= 0) & (v >= 0) & (u + v <= 1) & (t > 1e-9)).sum())


def build_surface_mesh(prep: PreparedGeometry, h: float | None = None, curvature: int = 8) -> SurfaceMesh:
    """Triangulate the boundary only and orient every CAD face outward by ray parity."""
    from .isolate import IsolatedCrash, IsolatedTimeout, run_isolated

    errors = []
    for alg in (6, 5, 1):
        try:
            return run_isolated(_surface, prep, h, curvature, alg, timeout=90.0)
        except (MeshError, IsolatedCrash, IsolatedTimeout) as e:
            errors.append(str(e))
    raise MeshError(f"could not triangulate the part's surface ({'; '.join(errors)})")


def _surface(prep: PreparedGeometry, h: float | None, curvature: int, alg: int) -> SurfaceMesh:
    t0 = time.time()
    h = h or prep.diag / 30.0
    with gmsh_session() as gmsh:
        load_into_session(gmsh, prep)
        sm = _surface_from_session(gmsh, h, curvature, alg, prep.diag, prep.kind == "stl")
        if prep.kind != "stl":
            sm.samples = cad_samples(gmsh, sm)
    sm.t_mesh = time.time() - t0
    return sm


# Roberts' R2 sequence: a quasi random walk that covers a square evenly, with no random numbers involved
_R2 = np.array([0.7548776662466927, 0.5698402909980532])


def cad_samples(gmsh, sm: SurfaceMesh, target: int = 800) -> dict | None:
    """Points spread over every CAD face in proportion to its area, with exact outward normals.

    Wall thickness is measured by casting rays inward from these points. They come from each face's own
    parametrization rather than from the triangles, so the estimate depends on the geometry and stays put when
    the surface mesh changes. Each point carries the area it stands for.
    """
    faces = [int(t) for _, t in gmsh.model.getEntities(2)]
    try:
        areas = np.array([gmsh.model.occ.getMass(2, t) for t in faces], float)
    except Exception:  # noqa: BLE001  not an OpenCASCADE model
        return None
    total = float(areas.sum())
    if total <= 0:
        return None
    from scipy.spatial import cKDTree

    n_tri, a_tri, c_tri = sm.tri_geometry()
    out = {"pts": [], "normals": [], "weights": [], "faces": []}
    for tag, area in zip(faces, areas):
        own = np.nonzero(sm.tri_surf == tag)[0]
        if area <= 0 or not len(own):
            continue
        want = max(1, int(round(target * area / total)))
        got = _face_samples(gmsh, tag, want)
        if got is None:  # a face gmsh cannot evaluate: its triangles stand in, picked the same way every time
            pick, count = systematic_pick(a_tri[own], want)
            pts, nrm, jac = c_tri[own[pick]], n_tri[own[pick]].copy(), count.astype(float)
        else:
            pts, nrm, jac = got
            # the parametrization fixes the sign of a CAD normal; the oriented surface mesh knows which side is out
            _, j = cKDTree(c_tri[own]).query(pts)
            flip = np.einsum("ij,ij->i", nrm, n_tri[own[j]]) < 0
            nrm[flip] *= -1.0
        out["pts"].append(pts)
        out["normals"].append(nrm)
        out["weights"].append(area * jac / jac.sum())
        out["faces"].append(np.full(len(pts), tag))
    if not out["pts"]:
        return None
    return {k: np.concatenate(v) for k, v in out.items()}


def systematic_pick(weights: np.ndarray, n: int) -> tuple[np.ndarray, np.ndarray]:
    """Pick about ``n`` items in proportion to their weights, evenly and without randomness.

    Returns the picked indices and how many of the ``n`` evenly spaced marks fell in each.
    """
    cw = np.cumsum(weights)
    marks = (np.arange(n) + 0.5) * cw[-1] / n
    idx = np.minimum(np.searchsorted(cw, marks), len(weights) - 1)
    return np.unique(idx, return_counts=True)


def _face_samples(gmsh, tag: int, want: int):
    """``want`` points inside one trimmed CAD face, their unit normals and area scale (|dX/du x dX/dv|)."""
    try:
        lo, hi = (np.asarray(b, float) for b in gmsh.model.getParametrizationBounds(2, tag))
        uv, k = [], 0
        while len(uv) < want and k < 20 * want + 50:
            k += 1
            q = lo + (hi - lo) * np.mod(0.5 + _R2 * k, 1.0)
            if gmsh.model.isInside(2, tag, q.tolist(), parametric=True):
                uv.append(q)
        if not uv:
            return None
        flat = np.asarray(uv).ravel().tolist()
        pts = np.asarray(gmsh.model.getValue(2, tag, flat), float).reshape(-1, 3)
        nrm = np.asarray(gmsh.model.getNormal(tag, flat), float).reshape(-1, 3)
        der = np.asarray(gmsh.model.getDerivative(2, tag, flat), float).reshape(-1, 6)
    except Exception:  # noqa: BLE001  an entity gmsh cannot evaluate: the triangles stand in for it
        return None
    jac = np.linalg.norm(np.cross(der[:, :3], der[:, 3:]), axis=1)
    nlen = np.linalg.norm(nrm, axis=1)
    ok = (nlen > 0) & np.isfinite(jac) & (jac > 0)
    if not ok.any():
        return None
    return pts[ok], nrm[ok] / nlen[ok, None], jac[ok]


def _surface_from_session(gmsh, h: float, curvature: int = 8, alg: int = 6, diag: float | None = None,
                          is_stl: bool = False) -> SurfaceMesh:
    """Surface mesh of whatever geometry is loaded in an open gmsh session."""
    if diag is None:
        bb = gmsh.model.getBoundingBox(-1, -1)
        diag = float(np.linalg.norm(np.array(bb[3:]) - np.array(bb[:3])))
    t0 = time.time()
    opt = gmsh.option.setNumber
    opt("Mesh.MeshSizeMax", min(h, diag / 30.0))
    opt("Mesh.MeshSizeMin", diag / 250.0)
    opt("Mesh.MeshSizeFromPoints", 0)
    opt("Mesh.MeshSizeFromCurvature", 0 if is_stl else curvature)
    opt("Mesh.Algorithm", alg)
    if not is_stl:
        try:
            gmsh.model.mesh.generate(2)
        except Exception as e:  # noqa: BLE001
            raise MeshError(f"surface meshing failed: {e}") from e
    ntags, coords, _ = gmsh.model.mesh.getNodes()
    stypes, scurves, chunks = {}, {}, []
    for _, s in gmsh.model.getEntities(2):
        stypes[s] = gmsh.model.getType(2, s)
        scurves[s] = [int(c) for c in gmsh.model.getAdjacencies(2, s)[1]]
        st, _, sn = gmsh.model.mesh.getElements(2, s)
        for tt, nn in zip(st, sn):
            if tt == 2:
                chunks.append((s, np.asarray(nn, dtype=np.int64).reshape(-1, 3)))
    sm = _orient_surface(ntags, coords, chunks, stypes, scurves, diag, t0)
    if not is_stl:
        sm.tangent_pairs = _tangent_pairs(gmsh)
    return sm


def _tangent_pairs(gmsh, cos_limit: float = 0.966) -> set:
    """Pairs of CAD faces that meet tangentially, judged from the exact CAD normals along their shared edge.

    A fillet, a torus blend or a B-spline round meets its walls without a kink. A coarse mesh facets it so
    the triangles there look like a sharp corner; the CAD normals do not.
    """
    pairs = set()
    for _, c in gmsh.model.getEntities(1):
        try:
            up, _ = gmsh.model.getAdjacencies(1, c)
        except Exception:  # noqa: BLE001
            continue
        if len(up) != 2:
            continue
        s1, s2 = int(up[0]), int(up[1])
        try:
            lo, hi = gmsh.model.getParametrizationBounds(1, c)
            dots = []
            for f in (0.25, 0.5, 0.75):
                xyz = gmsh.model.getValue(1, c, [lo[0] + f * (hi[0] - lo[0])])
                n = []
                for s in (s1, s2):
                    _, uv = gmsh.model.getClosestPoint(2, s, xyz)
                    v = np.asarray(gmsh.model.getNormal(s, uv), float)
                    n.append(v / max(np.linalg.norm(v), 1e-300))
                dots.append(abs(float(n[0] @ n[1])))
        except Exception:  # noqa: BLE001
            continue
        if min(dots) >= cos_limit:
            pairs.add((min(s1, s2), max(s1, s2)))
    return pairs


def _orient_surface(ntags, coords, chunks, stypes, scurves, diag, t0) -> SurfaceMesh:
    if not chunks:
        raise MeshError("no surface triangles were generated")
    ntags = np.asarray(ntags, dtype=np.int64)
    lut = np.full(int(ntags.max()) + 1, -1, dtype=np.int64)
    lut[ntags] = np.arange(len(ntags))
    X = np.asarray(coords).reshape(-1, 3)
    tris = np.vstack([lut[c] for _, c in chunks])
    tri_surf = np.concatenate([np.full(len(c), s) for s, c in chunks])
    used = np.unique(tris)
    remap = np.full(len(X), -1, dtype=np.int64)
    remap[used] = np.arange(len(used))
    X, tris = X[used], remap[tris]
    # orient each surface outward: a ray leaving outward crosses the boundary an even number of times
    V0, V1, V2 = X[tris[:, 0]], X[tris[:, 1]], X[tris[:, 2]]
    e1, e2 = V1 - V0, V2 - V0
    nrm = np.cross(e1, e2)
    area = 0.5 * np.linalg.norm(nrm, axis=1)
    rng = np.random.default_rng(1)
    for s in np.unique(tri_surf):
        idx = np.nonzero(tri_surf == s)[0]
        pick = idx[np.argsort(-area[idx])[:3]]
        votes = 0
        for i in pick:
            n = nrm[i] / max(np.linalg.norm(nrm[i]), 1e-300)
            d = n + 0.05 * rng.standard_normal(3)
            d /= np.linalg.norm(d)
            c = (V0[i] + V1[i] + V2[i]) / 3.0 + d * 1e-7 * diag
            votes += 1 if _count_hits(c, d, V0, e1, e2) % 2 == 1 else -1
        if votes > 0:
            tris[idx] = tris[idx][:, [0, 2, 1]]
    return SurfaceMesh(X, tris, tri_surf, stypes, scurves, time.time() - t0)


def auto_sizing(prep: PreparedGeometry, thickness: float | None = None, fineness: float = 1.0,
                max_nodes: int = 60000) -> Sizing:
    """A first global size: fine enough for two second order elements across typical walls, capped by node budget."""
    diag = prep.diag
    h = diag / 22.0
    if thickness and thickness > 0:
        h = min(h, 0.9 * thickness)
    # budget: a 10 node tet mesh has roughly 1.4 nodes per (h^3/6.5) of volume
    vol = max(prep.volume, 1e-9)
    nodes_est = 1.45 * vol / (h ** 3 / 6.5)
    if nodes_est > max_nodes:
        h *= (nodes_est / max_nodes) ** (1 / 3)
    h /= fineness
    return Sizing(h=float(h))


def requested_size(sizing: Sizing, mesh: Mesh, pts: np.ndarray) -> np.ndarray:
    """The element size a sizing asks for at each point, evaluated the way gmsh evaluates the fields.

    The global size, lowered by each refinement: a threshold on the distance to its faces or edges, or a ball
    around its points. Distances to faces and edges are measured to the mesh nodes on them. This is a property
    of the geometry and the sizing rules, unlike the size of whichever element happens to sit at a point.
    """
    from scipy.spatial import cKDTree

    pts = np.atleast_2d(np.asarray(pts, float))
    h = float(sizing.h)
    out = np.full(len(pts), h)
    tris = mesh.tris[:, :3]
    pairs = np.unique(np.column_stack([tris.ravel(), np.repeat(mesh.tri_surf, 3)]), axis=0)
    shared = np.bincount(pairs[:, 0], minlength=len(mesh.X)) > 1      # nodes on an edge between faces
    on = {int(s): np.unique(tris[mesh.tri_surf == s]) for s in np.unique(mesh.tri_surf)}

    def edge_nodes(curve: int) -> np.ndarray:
        faces = [s for s, cs in mesh.surface_curves.items() if curve in {abs(int(c)) for c in cs} and s in on]
        if not faces:
            return np.zeros(0, np.int64)
        nodes = on[faces[0]]
        for s in faces[1:]:
            nodes = np.intersect1d(nodes, on[s])
        return nodes[shared[nodes]]

    def ramp(d, d0, d1, inner):
        return np.where(d <= d0, inner, np.where(d >= d1, h, inner + (h - inner) * (d - d0) / max(d1 - d0, 1e-12)))

    for r in sizing.refinements:
        grow = r.grow or 6.0 * r.size
        if r.corners_only:
            nodes = [on[s][shared[on[s]]] for s in r.surfaces if s in on]
            d0, d1 = 0.5 * r.size, 0.5 * r.size + grow
        else:
            nodes = [on[s] for s in r.surfaces if s in on] + [edge_nodes(int(c)) for c in r.curves]
            d0, d1 = r.radius, r.radius + grow
        nodes = np.concatenate(nodes) if nodes else np.zeros(0, np.int64)
        if len(nodes):
            d, _ = cKDTree(mesh.X[np.unique(nodes)]).query(pts)
            out = np.minimum(out, ramp(d, d0, d1, r.size))
        for p in r.points:
            d = np.linalg.norm(pts - np.asarray(p, float), axis=1)
            rad = max(r.radius, 2.0 * r.size)
            out = np.minimum(out, ramp(d, rad, rad + grow, r.size))
    return out
