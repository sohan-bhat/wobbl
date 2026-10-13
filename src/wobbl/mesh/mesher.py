"""Tetrahedral meshing with gmsh: second order tets, curvature aware sizing and local refinement.

The result is a solver neutral ``Mesh``: node coordinates, 10 node tets in
CalculiX order, and the outer boundary as 6 node triangles that remember which
CAD surface, element and element face they came from. Loads, fixtures and
heat boundary conditions are all built from that boundary.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

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


def build_mesh(prep: PreparedGeometry, sizing: Sizing, verbose: bool = False) -> Mesh:
    """Mesh the prepared geometry with the given sizing. Retries with safer settings before giving up."""
    attempts = [dict(alg3d=10, curved=True), dict(alg3d=1, curved=True), dict(alg3d=1, curved=False)]
    last_err = None
    for i, att in enumerate(attempts):
        try:
            mesh = _build(prep, sizing, verbose=verbose, **att)
            if i > 0:
                mesh.notes.append("Meshing needed a fallback strategy" +
                                  ("" if att["curved"] else " (straight sided second order elements)") + ".")
            return mesh
        except MeshError as e:
            last_err = e
    raise MeshError(f"gmsh could not mesh this part: {last_err}. Try a larger --mesh-size, or check the CAD "
                    "for self intersections and gaps.")


def _add_fields(gmsh, sizing: Sizing):
    fields = []
    for r in sizing.refinements:
        grow = r.grow or 6.0 * r.size
        if r.surfaces or r.curves:
            d = gmsh.model.mesh.field.add("Distance")
            if r.surfaces:
                gmsh.model.mesh.field.setNumbers(d, "SurfacesList", list(map(int, r.surfaces)))
            if r.curves:
                gmsh.model.mesh.field.setNumbers(d, "CurvesList", list(map(int, r.curves)))
            gmsh.model.mesh.field.setNumber(d, "Sampling", 60)
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
    if fields:
        fm = gmsh.model.mesh.field.add("Min")
        gmsh.model.mesh.field.setNumbers(fm, "FieldsList", fields)
        gmsh.model.mesh.field.setAsBackgroundMesh(fm)


def _build(prep: PreparedGeometry, sizing: Sizing, verbose: bool, alg3d: int, curved: bool) -> Mesh:
    t0 = time.time()
    with gmsh_session(verbose) as gmsh:
        load_into_session(gmsh, prep)
        opt = gmsh.option.setNumber
        opt("Mesh.MeshSizeMax", sizing.h)
        opt("Mesh.MeshSizeMin", sizing.h_min or 0.0)
        opt("Mesh.MeshSizeFromPoints", 0)
        opt("Mesh.MeshSizeFromCurvature", sizing.curvature)
        opt("Mesh.MeshSizeExtendFromBoundary", 1)
        opt("Mesh.Algorithm", 6)
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
        if sizing.order == 2 and curved:
            q = np.asarray(gmsh.model.mesh.getElementQualities(etags[k], "minDetJac"))
            if (q <= 0).any():
                gmsh.model.mesh.optimize("HighOrder")
                types, etags, enodes = gmsh.model.mesh.getElements(3)
                k = list(types).index(etype)
                q = np.asarray(gmsh.model.mesh.getElementQualities(etags[k], "minDetJac"))
                if (q <= 0).any():
                    raise MeshError(f"{int((q <= 0).sum())} curved elements are inverted")
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
    if (tri_surf < 0).any():
        mesh.notes.append(f"{int((tri_surf < 0).sum())} boundary faces could not be matched to a CAD surface.")
    if len(tets) == 0:
        raise GeometryError("the mesh is empty")
    return mesh


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
