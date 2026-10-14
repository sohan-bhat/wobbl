"""The midsurface of a thin walled solid, meshed with 6 node triangles that carry the wall thickness.

A thin wall is two faces of the solid facing each other a wall thickness apart. Rays cast inward from the
skin find each face's partner and the thickness. One face of every pair is kept as the skin, picked so the
kept faces join up into one sheet (the outside of a box, not its outside top and inside walls), and every
skin node moves inward by half the local thickness. Where kept faces meet at an angle, the node moves so it
sits half a thickness inside each of them, which puts the corner of a box's midsurface where it belongs.

Rims, edges and chunky features such as screw bosses are not walls. They are left out of the sheet, and the
shell keeps track of how much of the part's material it still holds, so the solid cross check and the
assumptions can say what the shell model leaves out.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..geometry.cad import PreparedGeometry
from ..geometry.faces import PartFeatures
from ..mesh.mesher import SurfaceMesh

THIN_RATIO = 0.1        # a part is thin walled when its typical wall is under a tenth of its middle dimension
WALL_SHARE = 0.75       # and walls that face a partner cover at least this share of its surface
OPPOSITE_COS = -0.8     # a ray's exit face counts as the partner when the normals are this close to opposite
THICK_LIMIT = 3.0       # a wall more than this many times the typical one is a chunky feature, not a wall
CURVATURE = 16          # elements per full circle on curved walls
# corner nodes, then midside nodes 0-1, 1-2 and 2-0, the order of a CalculiX S6 and a gmsh 6 node triangle
TRI_EDGES = ((0, 1), (1, 2), (2, 0))


@dataclass
class ShellMesh:
    X: np.ndarray                  # (n,3) midsurface nodes, mm
    tris: np.ndarray               # (m,6) S6 connectivity, 0 based
    tri_surf: np.ndarray           # (m,) CAD face of the solid's skin each element came from
    thickness: np.ndarray          # (n,) wall thickness at each node, mm
    skin: np.ndarray               # (n,3) where each node sat on the solid's skin before moving inward
    normals: np.ndarray            # (n,3) unit normals pointing from the midsurface out to the kept skin
    solid_volume: float = 0.0
    kept_faces: list[int] = field(default_factory=list)
    left_out: list[int] = field(default_factory=list)   # faces of the solid that are neither skin nor its partner
    notes: list[str] = field(default_factory=list)
    h: float = 0.0
    solid: SurfaceMesh | None = field(default=None, repr=False)   # the solid's skin, where loads are picked

    @property
    def n_nodes(self) -> int:
        return len(self.X)

    @property
    def n_elems(self) -> int:
        return len(self.tris)

    def element_geometry(self, X: np.ndarray | None = None):
        """Unit normals, areas and centroids of the elements (flat corner triangles)."""
        X = self.X if X is None else X
        p0, p1, p2 = (X[self.tris[:, k]] for k in range(3))
        c = np.cross(p1 - p0, p2 - p0)
        a = 0.5 * np.linalg.norm(c, axis=1)
        return c / np.maximum(2 * a, 1e-300)[:, None], a, (p0 + p1 + p2) / 3.0

    def element_thickness(self) -> np.ndarray:
        return self.thickness[self.tris[:, :3]].mean(1)

    def volume(self) -> float:
        _, a, _ = self.element_geometry()
        return float((a * self.element_thickness()).sum())

    def node_weights(self) -> np.ndarray:
        """Area each node stands for, from the HRZ lumping of a 6 node triangle (a twentieth of the element on
        each corner, the rest on the midsides), so every node of a uniform sheet carries a positive share."""
        _, a, _ = self.element_geometry()
        w = np.zeros(self.n_nodes)
        for k in range(3):
            np.add.at(w, self.tris[:, k], a * CORNER_SHARE)
            np.add.at(w, self.tris[:, 3 + k], a * MIDSIDE_SHARE)
        return w


# HRZ lumping of the 6 node triangle's consistent mass matrix: diagonal terms 6 (corners) and 32 (midsides)
CORNER_SHARE = 6.0 / (3 * 6.0 + 3 * 32.0)
MIDSIDE_SHARE = 32.0 / (3 * 6.0 + 3 * 32.0)


def middle_dimension(features: PartFeatures) -> float:
    lo, hi = (np.asarray(b, float) for b in features.bbox)
    return float(np.sort(hi - lo)[1])


def thin_walled(features: PartFeatures) -> tuple[bool, str]:
    """Whether a part's walls are thin next to its size, judged from its typical wall and bounding box. A cheap
    first test: the midsurface itself also checks how much of the surface pairs up into walls."""
    mid = middle_dimension(features)
    t = features.thickness_typical
    ratio = t / max(mid, 1e-9)
    if ratio <= THIN_RATIO:
        return True, f"walls about {t:.3g} mm on a part {mid:.3g} mm across ({100 * ratio:.2g}%)"
    return False, (f"walls about {t:.3g} mm are {100 * ratio:.2g}% of the part's {mid:.3g} mm middle dimension, "
                   f"over the {100 * THIN_RATIO:.0f}% a shell model needs")


def first_exits(orig, dirs, V0, V1, V2, nrm, reach: float, k: int = 128) -> tuple[np.ndarray, np.ndarray]:
    """Distance along each ray to where it leaves the solid and the triangle it leaves through (-1 for none).

    Each ray tests the k triangles whose centers lie nearest its start within ``reach``, all rays at once. A
    wall's far face lies a thickness away, so it is among them; a ray that finds nothing comes back inf, which
    is what a chunky feature should give.
    """
    from scipy.spatial import cKDTree

    cen = (V0 + V1 + V2) / 3.0
    rad = float(max(np.linalg.norm(V - cen, axis=1).max() for V in (V0, V1, V2)))
    k = min(k, len(cen))
    _, cand = cKDTree(cen).query(orig, k=k, distance_upper_bound=reach + rad)
    cand = np.asarray(cand).reshape(len(orig), k)
    t = np.full(len(orig), np.inf)
    hit = np.full(len(orig), -1, dtype=np.int64)
    step = max(1, 200_000 // k)
    for r0 in range(0, len(orig), step):
        c = cand[r0:r0 + step]
        valid = c < len(cen)
        c = np.where(valid, c, 0)
        d = dirs[r0:r0 + step]
        e1, e2 = V1[c] - V0[c], V2[c] - V0[c]
        p = np.cross(d[:, None, :], e2)
        det = np.einsum("rtk,rtk->rt", p, e1)
        ok = np.abs(det) > 1e-12
        inv = np.where(ok, 1.0 / np.where(ok, det, 1.0), 0.0)
        s = orig[r0:r0 + step, None, :] - V0[c]
        u = np.einsum("rtk,rtk->rt", s, p) * inv
        q = np.cross(s, e1)
        v = np.einsum("rk,rtk->rt", d, q) * inv
        dist = np.einsum("rtk,rtk->rt", e2, q) * inv
        good = (valid & ok & (u >= -1e-9) & (v >= -1e-9) & (u + v <= 1 + 1e-9) & (dist > 1e-9) & (dist <= reach)
                & (np.einsum("rtk,rk->rt", nrm[c], d) > 0))
        dist = np.where(good, dist, np.inf)
        j = np.argmin(dist, axis=1)
        rows = np.arange(len(j))
        best = dist[rows, j]
        found = np.flatnonzero(np.isfinite(best))
        t[r0 + found] = best[found]
        hit[r0 + found] = c[rows, j][found]
    return t, hit


def _face_adjacency(sm: SurfaceMesh) -> dict[int, set[int]]:
    """CAD faces that share mesh nodes, which on a conforming surface mesh means they share an edge."""
    pairs = np.unique(np.column_stack([sm.tris[:, :3].ravel(), np.repeat(sm.tri_surf, 3)]), axis=0)
    adj: dict[int, set[int]] = {int(s): set() for s in np.unique(sm.tri_surf)}
    order = np.argsort(pairs[:, 0], kind="stable")
    pairs = pairs[order]
    cut = np.flatnonzero(np.diff(pairs[:, 0])) + 1
    for grp in np.split(pairs[:, 1], cut):
        if len(grp) > 1:
            for s in grp:
                adj[int(s)].update(int(x) for x in grp if x != s)
    return adj


def _pick_skin(sm: SurfaceMesh, area: np.ndarray, thin: np.ndarray, hit: np.ndarray) -> tuple[list[int], list[int]]:
    """Keep one face of every wall pair so the kept faces join into one sheet.

    Faces are visited breadth first from the largest wall face across shared edges. A visited face is kept and
    the faces its rays land on are dropped, so the walk stays on one side of the walls: from the outside of a
    box's top it reaches the outside of its sides, never their insides. Wall faces the walk cannot reach start
    walks of their own, largest first.
    """
    faces = [int(s) for s in np.unique(sm.tri_surf)]
    total = {f: float(area[sm.tri_surf == f].sum()) for f in faces}
    wall = {f: float(area[(sm.tri_surf == f) & thin].sum()) for f in faces}
    walls = [f for f in faces if wall[f] >= 0.5 * total[f]]
    partners: dict[int, dict[int, float]] = {f: {} for f in faces}
    for i in np.flatnonzero(thin):
        f, g = int(sm.tri_surf[i]), int(sm.tri_surf[hit[i]])
        partners[f][g] = partners[f].get(g, 0.0) + float(area[i])
    adj = _face_adjacency(sm)
    kept, dropped = [], set()
    for seed in sorted(walls, key=lambda f: -total[f]):
        if seed in dropped or seed in kept:
            continue
        queue = [seed]
        while queue:
            f = queue.pop(0)
            if f in dropped or f in kept:
                continue
            kept.append(f)
            for g, w in partners[f].items():
                if g != f and w >= 0.1 * wall[f] and g not in kept:
                    dropped.add(g)
            queue += sorted((g for g in adj[f] if g in walls and g not in dropped and g not in kept),
                            key=lambda g: -total[g])
    return kept, sorted(dropped)


def midsurface(prep: PreparedGeometry, features: PartFeatures, h: float) -> ShellMesh:
    """Mesh the midsurface of a thin walled part with 6 node triangles of about size h."""
    sm = skin_mesh(prep, h, CURVATURE)
    sm.tangent_pairs = set(features.tangent_pairs)
    nrm, area, cen = sm.tri_geometry()
    V0, V1, V2 = (sm.X[sm.tris[:, k]] for k in range(3))
    t_typ = features.thickness_typical
    reach = THICK_LIMIT * t_typ * 1.5
    lift = 1e-6 * prep.diag   # rays start a hair inside the skin, so they never hit the face they leave
    t, hit = first_exits(cen - nrm * lift, -nrm, V0, V1, V2, nrm, reach)
    t = t + lift
    paired = (hit >= 0) & (np.einsum("ij,ij->i", nrm, nrm[np.maximum(hit, 0)]) < OPPOSITE_COS)
    thin = paired & (t <= THICK_LIMIT * t_typ)
    kept, dropped = _pick_skin(sm, area, thin, hit)
    keep = np.isin(sm.tri_surf, kept)
    if not keep.any():
        raise ValueError("no faces of this part pair up into walls, so it has no midsurface")
    keep &= _largest_component(sm.tris[:, :3], keep)
    tris3 = sm.tris[keep, :3]
    used = np.unique(tris3)
    remap = np.full(len(sm.X), -1, dtype=np.int64)
    remap[used] = np.arange(len(used))
    skin = sm.X[used]
    tris3 = remap[tris3]
    k_nrm, k_area, k_t, k_thin = nrm[keep], area[keep], t[keep], thin[keep]
    face_t = {f: float(np.median(t[(sm.tri_surf == f) & thin])) if ((sm.tri_surf == f) & thin).any() else t_typ
              for f in kept}
    # a triangle whose ray ran into a chunky feature (the top of a boss under a lid) takes its face's wall
    k_t = np.where(k_thin, k_t, [face_t.get(int(f), t_typ) for f in sm.tri_surf[keep]])
    k_surf = sm.tri_surf[keep]
    th = _node_mean(tris3, k_area, k_t, len(skin))
    tris3, alive = _collapse_folds(skin, tris3, k_surf, th, sm.tangent_pairs)
    k_nrm, k_area, k_surf, k_t = k_nrm[alive], k_area[alive], k_surf[alive], k_t[alive]
    tris3, skin = _compact(tris3, skin)
    th = _node_mean(tris3, k_area, k_t, len(skin))
    offset, normals = _offsets(tris3, k_nrm, k_area, th, len(skin))
    Xc = skin + offset
    # an element the offset turned over has no midsurface to stand for; the ones left must still join up
    p0, p1, p2 = (Xc[tris3[:, k]] for k in range(3))
    upright = np.einsum("ij,ij->i", np.cross(p1 - p0, p2 - p0), k_nrm) > 1e-9 * k_area.max()
    upright &= _largest_component(tris3, upright)
    tris3, k_surf = tris3[upright], k_surf[upright]
    tris3, skin, Xc, th, normals = _compact(tris3, skin, Xc, th, normals)
    X, tris, skin6, th6, nrm6 = _quadratic(Xc, skin, tris3, th, normals)
    left = sorted(set(int(s) for s in np.unique(sm.tri_surf)) - set(kept) - set(dropped))
    out = ShellMesh(X, tris, k_surf, th6, skin6, nrm6, prep.volume, kept, left, h=h, solid=sm)
    share = out.volume() / max(prep.volume, 1e-9)
    out.notes.append(f"The shell model holds {100 * share:.0f}% of the part's material in {len(kept)} wall faces.")
    lost = int((~np.isin(sm.tri_surf, kept + dropped)).sum())
    if lost:
        out.notes.append(f"{len(left)} faces that are rims, edges or chunky features such as bosses are not walls "
                         "and are left out of the shell.")
    return out


def _node_mean(tris3, area, vals, n) -> np.ndarray:
    """Area weighted mean of a per element value at each node."""
    w, s = np.zeros(n), np.zeros(n)
    for k in range(3):
        np.add.at(w, tris3[:, k], area)
        np.add.at(s, tris3[:, k], area * vals)
    return s / np.maximum(w, 1e-300)


def _compact(tris3, *node_arrays):
    """Drop nodes no element uses and renumber; returns the triangles and each node array cut to match."""
    used = np.unique(tris3)
    remap = np.full(len(node_arrays[0]), -1, dtype=np.int64)
    remap[used] = np.arange(len(used))
    return (remap[tris3], *(a[used] for a in node_arrays))


def _collapse_folds(skin, tris3, surf, th, tangent: set) -> tuple[np.ndarray, np.ndarray]:
    """Where two kept faces meet at a fold, merge the skin nodes that lie within half a wall of the other face
    into the nearest node on the fold between them.

    Moved inward, those nodes would land past the other wall's midsurface and turn their elements over: a box's
    side wall would poke up through its lid. Merged into the fold, the strip of elements between them and the
    fold closes up and is dropped, and the walls meet along nodes they share. Faces that meet tangentially (a
    wall and its fillet) make no fold. Returns the new triangles and which old ones survive.
    """
    from scipy.spatial import cKDTree

    from .mapping import point_triangle_distance

    n = len(skin)
    faces_at: list[set] = [set() for _ in range(n)]
    for k in range(3):
        for i, f in zip(tris3[:, k], surf):
            faces_at[i].add(int(f))

    def folded(f, g) -> bool:
        return f != g and (min(f, g), max(f, g)) not in tangent

    node_face = np.array([min(fs) for fs in faces_at])
    # a node on the line where two faces fold against each other anchors the fold
    anchors = np.array([i for i in range(n) if any(folded(f, g) for f in faces_at[i] for g in faces_at[i])],
                       dtype=np.int64)
    if not len(anchors):
        return tris3, np.ones(len(tris3), bool)
    is_anchor = np.zeros(n, bool)
    is_anchor[anchors] = True
    cen = skin[tris3].mean(1)
    rad = float(np.linalg.norm(skin[tris3] - cen[:, None], axis=2).max())
    # a node just past the strip would leave a sliver along the fold, so the strip reaches a third of an element
    # further; hn is the mean length of the edges at each node
    hn, cnt = np.zeros(n), np.zeros(n)
    for i, j in TRI_EDGES:
        L = np.linalg.norm(skin[tris3[:, i]] - skin[tris3[:, j]], axis=1)
        for c in (i, j):
            np.add.at(hn, tris3[:, c], L)
            np.add.at(cnt, tris3[:, c], 1.0)
    reach = 0.5 * th + 0.35 * hn / np.maximum(cnt, 1)
    target = np.arange(n)
    d_fold, _ = cKDTree(skin[anchors]).query(skin)
    cand = np.flatnonzero(~is_anchor & (d_fold <= reach + rad))
    near = cKDTree(cen).query_ball_point(skin[cand], reach[cand] + rad)
    for i, tri_ids in zip(cand, near):
        if not tri_ids:
            continue
        f = int(node_face[i])
        t = np.asarray(tri_ids)
        t = t[np.array([folded(f, int(g)) for g in surf[t]])]
        if not len(t):
            continue
        A, B, C = (skin[tris3[t, k]] for k in range(3))
        d, _, _ = point_triangle_distance(np.repeat(skin[i][None], len(t), 0), A, B, C)
        if d.min() >= reach[i]:
            continue
        g = int(surf[t[int(np.argmin(d))]])
        on_fold = anchors[[f in faces_at[a] and g in faces_at[a] for a in anchors]]
        if len(on_fold):
            target[i] = on_fold[int(np.argmin(np.linalg.norm(skin[on_fold] - skin[i], axis=1)))]
    new = target[tris3]
    alive = (new[:, 0] != new[:, 1]) & (new[:, 1] != new[:, 2]) & (new[:, 0] != new[:, 2])
    new, alive2 = _collapse_slivers(skin, new[alive])
    keep = np.flatnonzero(alive)[alive2]
    out = np.zeros(len(tris3), bool)
    out[keep] = True
    return new, out


def _collapse_slivers(skin, tris3, rounds: int = 3) -> tuple[np.ndarray, np.ndarray]:
    """Merge the shortest edge of elements far thinner than they are long (under a tenth), which a fold merge
    can leave behind and which CalculiX's expansion through a thick wall would turn inside out."""
    alive = np.ones(len(tris3), bool)
    tris3 = tris3.copy()
    for _ in range(rounds):
        P = skin[tris3]
        e = np.stack([np.linalg.norm(P[:, j] - P[:, i], axis=1) for i, j in TRI_EDGES], 1)
        area = 0.5 * np.linalg.norm(np.cross(P[:, 1] - P[:, 0], P[:, 2] - P[:, 0]), axis=1)
        height = 2 * area / np.maximum(e.max(1), 1e-300)
        bad = np.flatnonzero(alive & (height < 0.1 * e.max(1)))
        if not len(bad):
            break
        target = np.arange(len(skin))
        for k in bad:
            i, j = TRI_EDGES[int(np.argmin(e[k]))]
            a, b = sorted((tris3[k, i], tris3[k, j]))
            if target[a] == a and target[b] == b:
                target[b] = a
        tris3 = target[tris3]
        alive &= (tris3[:, 0] != tris3[:, 1]) & (tris3[:, 1] != tris3[:, 2]) & (tris3[:, 0] != tris3[:, 2])
    return tris3[alive], alive


def skin_mesh(prep: PreparedGeometry, h: float, curvature: int) -> SurfaceMesh:
    """The solid's boundary triangles at size h, oriented outward, in a child process like every mesh. Leaner
    than the surface mesh that describes a part: the face analysis already knows which faces meet tangentially
    and how thick the walls are, so neither is worked out again."""
    from ..mesh.isolate import IsolatedCrash, IsolatedTimeout, run_isolated
    from ..mesh.mesher import MeshError

    errors = []
    for alg in (6, 5, 1):
        try:
            return run_isolated(_skin, prep, h, curvature, alg, timeout=120.0)
        except (MeshError, IsolatedCrash, IsolatedTimeout) as e:
            errors.append(str(e))
    raise MeshError(f"could not mesh the part's surface for its midsurface ({'; '.join(errors)})")


def _skin(prep: PreparedGeometry, h: float, curvature: int, alg: int) -> SurfaceMesh:
    import time

    from ..geometry.cad import gmsh_session, load_into_session
    from ..mesh.mesher import MeshError, _orient_surface

    t0 = time.time()
    with gmsh_session() as gmsh:
        load_into_session(gmsh, prep)
        opt = gmsh.option.setNumber
        opt("Mesh.MeshSizeMax", h)
        opt("Mesh.MeshSizeMin", 0.25 * h)
        opt("Mesh.MeshSizeFromPoints", 0)
        opt("Mesh.MeshSizeFromCurvature", 0 if prep.kind == "stl" else curvature)
        opt("Mesh.Algorithm", alg)
        if prep.kind != "stl":
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
    return _orient_surface(ntags, coords, chunks, stypes, scurves, prep.diag, t0)


def _largest_component(tris: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Mask of the triangles in the largest node connected group among the masked ones."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    idx = np.flatnonzero(mask)
    t = tris[idx]
    n = int(tris.max()) + 1
    rows = np.concatenate([t[:, 0], t[:, 1], t[:, 2]])
    cols = np.concatenate([t[:, 1], t[:, 2], t[:, 0]])
    g = coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n))
    _, lab = connected_components(g, directed=False)
    tl = lab[t[:, 0]]
    vals, counts = np.unique(tl, return_counts=True)
    out = np.zeros(len(tris), bool)
    out[idx[tl == vals[np.argmax(counts)]]] = True
    return out


def _offsets(tris3, nrm, area, th, n) -> tuple[np.ndarray, np.ndarray]:
    """Move each skin node so it sits half a thickness inside every kept face around it (least squares), and
    return the unit normal pointing back out to the skin."""
    M = np.zeros((n, 3, 3))
    b = np.zeros((n, 3))
    w = np.zeros(n)
    nn = np.einsum("ti,tj->tij", nrm, nrm) * area[:, None, None]
    for k in range(3):
        np.add.at(M, tris3[:, k], nn)
        np.add.at(b, tris3[:, k], nrm * area[:, None])
        np.add.at(w, tris3[:, k], area)
    # directions no face constrains (along an edge, or in the plane of a flat wall) get no motion at all: the
    # least squares solve keeps only the directions with real weight
    val, vec = np.linalg.eigh(M)
    rhs = -0.5 * th[:, None] * b
    proj = np.einsum("nji,nj->ni", vec, rhs)
    keep = val > 1e-3 * w[:, None]
    proj = np.where(keep, proj / np.where(keep, val, 1.0), 0.0)
    d = np.einsum("nij,nj->ni", vec, proj)
    out = -d / np.maximum(np.linalg.norm(d, axis=1), 1e-300)[:, None]
    return d, out


def _quadratic(Xc, skin, tris3, th, normals):
    """Add a midside node on every edge: straight sided 6 node triangles on the moved corners."""
    edges = np.sort(np.concatenate([tris3[:, list(e)] for e in TRI_EDGES]), axis=1)
    uniq, inv = np.unique(edges, axis=0, return_inverse=True)
    n = len(Xc)
    mid = n + inv.reshape(3, -1).T
    X = np.vstack([Xc, 0.5 * (Xc[uniq[:, 0]] + Xc[uniq[:, 1]])])
    sk = np.vstack([skin, 0.5 * (skin[uniq[:, 0]] + skin[uniq[:, 1]])])
    t = np.concatenate([th, 0.5 * (th[uniq[:, 0]] + th[uniq[:, 1]])])
    nm = normals[uniq[:, 0]] + normals[uniq[:, 1]]
    nm = np.vstack([normals, nm / np.maximum(np.linalg.norm(nm, axis=1), 1e-300)[:, None]])
    return X, np.hstack([tris3, mid]), sk, t, nm


def shape_s6(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Quadratic shape functions of a 6 node triangle at area coordinates (u, v), shape (k, 6)."""
    w = 1.0 - u - v
    return np.column_stack([w * (2 * w - 1), u * (2 * u - 1), v * (2 * v - 1), 4 * w * u, 4 * u * v, 4 * v * w])


__all__ = ["ShellMesh", "midsurface", "thin_walled", "shape_s6"]
