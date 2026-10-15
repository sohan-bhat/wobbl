"""Assemblies: bodies that touch without being glued, meshed so each keeps its own nodes where it meets another.

The strength study bonds touching bodies into one part. Here every body keeps its own surface at an interface, so
two bodies can press together, slide and open apart. Faces of different bodies that lie on each other become
contact pairs, each with the gap or overlap the CAD leaves between them, and pairs the user glues are fused into one
mesh the way the strength study does it.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..geometry.cad import GeometryError, PreparedGeometry, gmsh_session, load_into_session
from ..geometry.faces import PartFeatures, analyze
from ..mesh.mesher import (GMSH_TO_CCX_TET10, Mesh, MeshError, Sizing, SurfaceMesh, _add_fields, _boundary,
                           _count_hits, _tangent_pairs, cad_samples, flat_elements)


@dataclass
class FacePair:
    """Two faces of different bodies that lie on each other. ``slave`` is the smaller one."""
    slave: int
    master: int
    body_slave: int
    body_master: int
    area: float                   # mm2 of the slave face that sits on the master
    gap: float                    # median gap, mm; negative where the CAD has them overlap
    gap_min: float
    gap_max: float

    def describe(self) -> str:
        return f"face {self.slave} (body {self.body_slave}) on face {self.master} (body {self.body_master})"


@dataclass
class Assembly:
    prep: PreparedGeometry
    features: PartFeatures
    body_faces: dict[int, list[int]]          # body number, from 1 -> its face ids
    pairs: list[FacePair] = field(default_factory=list)
    bonded: list[tuple[int, int]] = field(default_factory=list)
    surface: SurfaceMesh | None = None

    @property
    def n_bodies(self) -> int:
        return len(self.body_faces)

    def body_of_face(self) -> dict[int, int]:
        return {f: b for b, fs in self.body_faces.items() for f in fs}


# --------------------------------------------------------------------------------------------------- preparing
def prepare_assembly(source: str | Path, workdir: str | Path, bonded=(), params: dict | None = None,
                     scale: float = 1.0) -> Assembly:
    """Read a multi body STEP, BREP or parametric .py part and keep its bodies apart, except pairs of bodies listed
    in ``bonded`` (1 based), which are glued into one. Finds the faces that touch between bodies."""
    from ..geometry.parametric import build_step, is_parametric
    from ..mesh.isolate import IsolatedCrash, IsolatedTimeout, run_isolated

    work = Path(workdir)
    work.mkdir(parents=True, exist_ok=True)
    notes = []
    src = str(source)
    if is_parametric(src):
        src, used = build_step(src, params, work / "geometry")
        notes.append("Built from " + Path(str(source)).name + " with " +
                     ", ".join(f"{k}={v:g}" if isinstance(v, (int, float)) else f"{k}={v}" for k, v in used.items())
                     + ".")
    try:
        prep, faces_of, sm = run_isolated(_prepare, src, str(work / "geometry"), list(bonded), scale, timeout=240.0)
    except (IsolatedTimeout, IsolatedCrash) as e:
        raise GeometryError(f"could not read the assembly {Path(str(source)).name} ({e})") from None
    prep.source = Path(str(source)).resolve()
    prep.notes = notes + prep.notes
    feats = analyze(sm)
    asm = Assembly(prep, feats, faces_of, [], list(bonded), sm)
    asm.pairs = find_pairs(sm, feats, faces_of, prep.diag)
    return asm


def _prepare(src: str, workdir: str, bonded: list, scale: float):
    """In a child process: import, glue the bonded bodies, write a .brep and mesh the surface with every body's
    faces turned outward on their own."""
    work = Path(workdir)
    work.mkdir(parents=True, exist_ok=True)
    notes = []
    with gmsh_session() as gmsh:
        gmsh.option.setString("Geometry.OCCTargetUnit", "MM")
        gmsh.model.occ.importShapes(src, highestDimOnly=True)
        gmsh.model.occ.synchronize()
        vols = [t for _, t in gmsh.model.getEntities(3)]
        if not vols or any(gmsh.model.occ.getMass(3, v) <= 0 for v in vols):
            raise GeometryError(f"{Path(src).name} has no valid closed solids")
        if scale and scale != 1.0:
            gmsh.model.occ.dilate([(3, v) for v in vols], 0, 0, 0, scale, scale, scale)
            gmsh.model.occ.synchronize()
            notes.append(f"Scaled by {scale:g}.")
        groups = [[v] for v in vols]
        for a, b in bonded:
            if not (1 <= a <= len(vols) and 1 <= b <= len(vols)):
                raise GeometryError(f"cannot bond bodies {a} and {b}: the file has {len(vols)} bodies")
            ga = next(g for g in groups if vols[a - 1] in g)
            gb = next(g for g in groups if vols[b - 1] in g)
            if ga is not gb:
                ga += gb
                groups.remove(gb)
        if bonded:
            # glue each group with a fragment of its own, so faces between groups stay apart
            for g in [g for g in groups if len(g) > 1]:
                gmsh.model.occ.fragment([(3, g[0])], [(3, v) for v in g[1:]])
            gmsh.model.occ.synchronize()
            notes.append("Bodies " + ", ".join(f"{a} and {b}" for a, b in bonded) + " are glued together.")
        vols = [t for _, t in gmsh.model.getEntities(3)]
        out = work / "assembly.brep"
        gmsh.write(str(out))
        bb = gmsh.model.getBoundingBox(-1, -1)
        vol = float(sum(gmsh.model.occ.getMass(3, v) for v in vols))
        prep = PreparedGeometry(Path(src), out, "cad", (tuple(bb[:3]), tuple(bb[3:])), vol, len(vols),
                                len(gmsh.model.getEntities(2)), notes)
    # reload the written file, so every later step sees the same tags
    with gmsh_session() as gmsh:
        load_into_session(gmsh, prep)
        vols = [t for _, t in gmsh.model.getEntities(3)]
        faces_of = {}
        for k, v in enumerate(vols):
            faces_of[k + 1] = sorted({abs(int(s)) for s in gmsh.model.getAdjacencies(3, v)[1]})
        sm = _surface(gmsh, prep.diag, faces_of)
    return prep, faces_of, sm


def _surface(gmsh, diag: float, faces_of: dict[int, list[int]]) -> SurfaceMesh:
    """Surface triangles with each body's faces turned outward by ray parity against that body alone. Against the
    whole assembly a face pressed on another body would test as inside it and flip."""
    t0 = time.time()
    opt = gmsh.option.setNumber
    opt("Mesh.MeshSizeMax", diag / 30.0)
    opt("Mesh.MeshSizeMin", diag / 250.0)
    opt("Mesh.MeshSizeFromPoints", 0)
    opt("Mesh.MeshSizeFromCurvature", 8)
    opt("Mesh.Algorithm", 6)
    try:
        gmsh.model.mesh.generate(2)
    except Exception as e:  # noqa: BLE001
        raise MeshError(f"surface meshing failed: {e}") from e
    ntags, coords, _ = gmsh.model.mesh.getNodes()
    ntags = np.asarray(ntags, np.int64)
    lut = np.full(int(ntags.max()) + 1, -1, np.int64)
    lut[ntags] = np.arange(len(ntags))
    X = np.asarray(coords).reshape(-1, 3)
    stypes, scurves, chunks = {}, {}, []
    for _, s in gmsh.model.getEntities(2):
        stypes[s] = gmsh.model.getType(2, s)
        scurves[s] = [int(c) for c in gmsh.model.getAdjacencies(2, s)[1]]
        st, _, sn = gmsh.model.mesh.getElements(2, s)
        for tt, nn in zip(st, sn):
            if tt == 2:
                chunks.append((s, lut[np.asarray(nn, np.int64).reshape(-1, 3)]))
    tris = np.vstack([c for _, c in chunks])
    tri_surf = np.concatenate([np.full(len(c), s) for s, c in chunks])
    used = np.unique(tris)
    remap = np.full(len(X), -1, np.int64)
    remap[used] = np.arange(len(used))
    X, tris = X[used], remap[tris]
    V0, V1, V2 = X[tris[:, 0]], X[tris[:, 1]], X[tris[:, 2]]
    e1, e2 = V1 - V0, V2 - V0
    nrm = np.cross(e1, e2)
    area = 0.5 * np.linalg.norm(nrm, axis=1)
    rng = np.random.default_rng(1)
    for faces in faces_of.values():
        own = np.isin(tri_surf, faces)
        o0, oe1, oe2 = V0[own], e1[own], e2[own]
        for s in faces:
            idx = np.nonzero(tri_surf == s)[0]
            if not len(idx):
                continue
            votes = 0
            for i in idx[np.argsort(-area[idx])[:3]]:
                n = nrm[i] / max(np.linalg.norm(nrm[i]), 1e-300)
                d = n + 0.05 * rng.standard_normal(3)
                d /= np.linalg.norm(d)
                c = (V0[i] + V1[i] + V2[i]) / 3.0 + d * 1e-7 * diag
                votes += 1 if _count_hits(c, d, o0, oe1, oe2) % 2 == 1 else -1
            if votes > 0:
                tris[idx] = tris[idx][:, [0, 2, 1]]
    sm = SurfaceMesh(X, tris, tri_surf, stypes, scurves, time.time() - t0)
    sm.tangent_pairs = _tangent_pairs(gmsh)
    sm.samples = cad_samples(gmsh, sm)
    return sm


# ------------------------------------------------------------------------------------------------------ pairs
def find_pairs(sm: SurfaceMesh, feats: PartFeatures, faces_of: dict[int, list[int]], diag: float,
               reach: float | None = None) -> list[FacePair]:
    """Faces of different bodies that face each other within ``reach`` (2% of the assembly by default): the
    centroid of each triangle of one face is measured along the normal of the nearest triangle of the other, and
    the pair stands when a good share of the face sits that close and the faces point at each other."""
    from scipy.spatial import cKDTree

    reach = reach if reach is not None else 0.02 * diag
    body_of = {f: b for b, fs in faces_of.items() for f in fs}
    n, a, c = sm.tri_geometry()
    pairs = []
    ids = sorted(feats.faces)
    for i, fa in enumerate(ids):
        for fb in ids[i + 1:]:
            ba, bb = body_of.get(fa), body_of.get(fb)
            if ba is None or bb is None or ba == bb:
                continue
            A, B = feats.faces[fa], feats.faces[fb]
            lo = np.maximum(np.asarray(A.bbox[0]), np.asarray(B.bbox[0])) - reach
            hi = np.minimum(np.asarray(A.bbox[1]), np.asarray(B.bbox[1])) + reach
            if (lo > hi).any():
                continue
            # the smaller face is measured against the larger one
            s, m = (fa, fb) if A.area <= B.area else (fb, fa)
            ms, mm = sm.tri_surf == s, sm.tri_surf == m
            tree = cKDTree(c[mm])
            dist, j = tree.query(c[ms])
            nm, cm = n[mm][j], c[mm][j]
            d = np.einsum("ij,ij->i", c[ms] - cm, nm)                  # + outside the master body
            lateral = np.linalg.norm(c[ms] - cm - d[:, None] * nm, axis=1)
            size = np.sqrt(2.0 * a[mm][j])
            facing = np.einsum("ij,ij->i", n[ms], nm) < -0.9
            close = facing & (np.abs(d) <= reach) & (lateral <= 1.5 * size)
            share = float(a[ms][close].sum())
            if share < 0.05 * float(a[ms].sum()) or share <= 0:
                continue
            w = a[ms][close]
            dc = d[close]
            order = np.argsort(dc)
            med = float(dc[order][np.searchsorted(np.cumsum(w[order]), 0.5 * w.sum())])
            lo_gap, hi_gap = float(dc.min()), float(dc.max())
            S, M = feats.faces[s], feats.faces[m]
            if S.radius and M.radius and S.axis is not None and M.axis is not None and S.concave != M.concave:
                # a pin in a hole: the facets of the two circles make the measured gap noisy, the radii do not
                hole, pin = (S, M) if S.concave else (M, S)
                ax = np.asarray(hole.axis)
                off = np.asarray(pin.axis_point) - np.asarray(hole.axis_point)
                off = float(np.linalg.norm(off - (off @ ax) * ax))
                if abs(abs(ax @ np.asarray(pin.axis)) - 1) < 1e-3 and off < 0.05 * hole.radius:
                    med = lo_gap = hi_gap = float(hole.radius - pin.radius)
            pairs.append(FacePair(s, m, body_of[s], body_of[m], share, med, lo_gap, hi_gap))
    return pairs


# ---------------------------------------------------------------------------------------------------- meshing
def build_assembly_mesh(asm: Assembly, sizing: Sizing, timeout: float | None = None) -> Mesh:
    """Second order tetrahedra of every body, with ``mesh.body`` naming the body of each element. Bodies meet
    without sharing nodes unless they were glued. A failed attempt falls back to straight sided elements, then to a
    finer size, as the strength mesher does."""
    from dataclasses import replace

    from ..mesh.isolate import IsolatedCrash, IsolatedTimeout, run_isolated
    from ..mesh.mesher import estimate_nodes

    if timeout is None:
        timeout = 90.0 + estimate_nodes(asm.prep, sizing) / 750.0
    errors = []
    for scale in (1.0, 0.75):
        trial = sizing if scale == 1.0 else Sizing(h=sizing.h * scale, order=sizing.order,
                                                   curvature=sizing.curvature,
                                                   refinements=[replace(r, size=r.size * scale)
                                                                for r in sizing.refinements])
        for curved in (True, False):
            try:
                mesh = run_isolated(_mesh, asm.prep, trial, curved, len(asm.body_faces), timeout=timeout)
                if not curved or scale != 1.0:
                    mesh.notes.append("Meshing needed a fallback strategy.")
                return mesh
            except (MeshError, IsolatedCrash, IsolatedTimeout) as e:
                errors.append(str(e))
    raise MeshError(f"gmsh could not mesh the assembly ({'; '.join(errors[:3])})")


# the ten nodes of a CalculiX second order tet in (L2, L3, L4) space: four corners, then the middles of 01, 12, 20, 03, 13, 23
_TET10_POINTS = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [.5, 0, 0], [.5, .5, 0], [0, .5, 0], [0, 0, .5],
                          [.5, 0, .5], [0, .5, .5]], float)


def inverted_elements(X: np.ndarray, tets: np.ndarray) -> int:
    """Count second order tets whose Jacobian is not positive at one of their ten nodes. Curving the sides of a coarse
    element on a tight curve can turn it inside out, which the solver refuses."""
    u = _TET10_POINTS
    L = np.column_stack([1 - u.sum(1), u])                       # barycentric coordinates at each node
    # dN/d(L2, L3, L4) of the ten shape functions at each node, with L1 = 1 - L2 - L3 - L4
    d = np.zeros((10, 10, 3))
    for k in range(3):
        dL = np.zeros((10, 4))
        dL[:, k + 1] = 1.0
        dL[:, 0] = -1.0
        for c in range(4):
            d[:, c, k] = (4 * L[:, c] - 1) * dL[:, c]
        for j, (a, b) in enumerate(((0, 1), (1, 2), (2, 0), (0, 3), (1, 3), (2, 3))):
            d[:, 4 + j, k] = 4 * (dL[:, a] * L[:, b] + L[:, a] * dL[:, b])
    bad = 0
    for lo in range(0, len(tets), 20000):
        P = X[tets[lo:lo + 20000]]                                # (n, 10, 3)
        J = np.einsum("pck,nci->npik", d, P)                       # (n, nodes, xyz, local)
        bad += int((np.linalg.det(J) <= 0).any(axis=1).sum())
    return bad


def _mesh(prep: PreparedGeometry, sizing: Sizing, curved: bool, n_bodies: int) -> Mesh:
    t0 = time.time()
    with gmsh_session() as gmsh:
        load_into_session(gmsh, prep)
        opt = gmsh.option.setNumber
        opt("Mesh.MeshSizeMax", sizing.h)
        opt("Mesh.MeshSizeMin", sizing.h_min or 0.0)
        opt("Mesh.MeshSizeFromPoints", 0)
        opt("Mesh.MeshSizeFromCurvature", sizing.curvature)
        opt("Mesh.MeshSizeExtendFromBoundary", 1)
        opt("Mesh.Algorithm", 6)
        opt("Mesh.Algorithm3D", 10)
        opt("Mesh.Optimize", 1)
        opt("Mesh.ElementOrder", 1)
        _add_fields(gmsh, sizing)
        try:
            gmsh.model.mesh.generate(3)
            opt("Mesh.SecondOrderLinear", 0 if curved else 1)
            gmsh.model.mesh.setOrder(2)
        except Exception as e:  # noqa: BLE001
            raise MeshError(str(e)) from e
        vols = [t for _, t in gmsh.model.getEntities(3)]
        raw, body = [], []
        for k, v in enumerate(vols):
            types, _, enodes = gmsh.model.mesh.getElements(3, v)
            if 11 not in list(types):
                raise MeshError(f"no tetrahedra in body {k + 1}")
            r = np.asarray(enodes[list(types).index(11)], np.int64).reshape(-1, 10)
            raw.append(r)
            body.append(np.full(len(r), k + 1))
        ntags, coords, _ = gmsh.model.mesh.getNodes()
        surf_keys, stypes, scurves = [], {}, {}
        for _, s in gmsh.model.getEntities(2):
            stypes[s] = gmsh.model.getType(2, s)
            scurves[s] = [int(c) for c in gmsh.model.getAdjacencies(2, s)[1]]
            st, _, sn = gmsh.model.mesh.getElements(2, s)
            for tt, nn in zip(st, sn):
                if tt in (2, 9):
                    w = 3 if tt == 2 else 6
                    surf_keys.append((s, np.asarray(nn, np.int64).reshape(-1, w)[:, :3]))
    raw = np.vstack(raw)
    tagmax = int(max(np.max(ntags), raw.max())) + 1
    lut = np.full(tagmax, -1, np.int64)
    used = np.unique(raw)
    lut[used] = np.arange(len(used))
    pos = np.full(tagmax, -1, np.int64)
    pos[np.asarray(ntags, np.int64)] = np.arange(len(ntags))
    X = np.asarray(coords).reshape(-1, 3)[pos[used]]
    tets = lut[raw][:, GMSH_TO_CCX_TET10]
    if flat_elements(X, tets):
        raise MeshError("the mesher left flat elements")
    if tets.shape[1] == 10 and (bad := inverted_elements(X, tets)):
        raise MeshError(f"curving the sides left {bad} second order elements inside out")
    tris, E, F, key = _boundary(tets, X, 2)
    nn = len(X)
    tri_surf = np.full(len(tris), -1, np.int64)
    ks, ss = [], []
    for s, corners in surf_keys:
        cc = np.sort(lut[corners], axis=1)
        ks.append((cc[:, 0] * nn + cc[:, 1]) * nn + cc[:, 2])
        ss.append(np.full(len(cc), s))
    ks, ss = np.concatenate(ks), np.concatenate(ss)
    order = np.argsort(ks)
    ks, ss = ks[order], ss[order]
    idx = np.clip(np.searchsorted(ks, key), 0, len(ks) - 1)
    hit = ks[idx] == key
    tri_surf[hit] = ss[idx[hit]]
    mesh = Mesh(X, tets, tris, tri_surf, E, F, 2, sizing, time.time() - t0, surface_types=stypes,
                surface_curves=scurves)
    mesh.body = np.concatenate(body)
    return mesh
