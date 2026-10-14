"""Put a crack into a part and mesh it, with a fine tube of elements along the crack front.

A crack is a ruled surface through ordered section curves: the crack mouth just outside the part (or, for a
crack with two tips, the far front) and then each front the crack has grown through. OpenCASCADE keeps only the
piece inside the part and embeds it in the solid, so gmsh meshes both together and the tetrahedra on either side
of the crack share its nodes. Wobbl then gives the tetrahedra on one side their own copies of those nodes, except
along the front, which opens the crack. The midside nodes of the elements touching the front move to the quarter
points, which builds the square root singularity of a crack tip into those elements (Barsoum, 1976).
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..geometry.cad import PreparedGeometry, gmsh_session, load_into_session
from ..mesh.mesher import GMSH_TO_CCX_TET10, Mesh, MeshError, _boundary, _face_samples, flat_elements

CRACK_FACE = -2          # tri_surf of the crack faces
_TET_EDGES = [(0, 1), (1, 2), (2, 0), (0, 3), (1, 3), (2, 3)]   # midside nodes 4 to 9 of a CalculiX C3D10


@dataclass
class Crack:
    """A crack as ordered section curves across it, each a (k, 3) array of points running the same way. The
    ruled surface through them is the crack; it may reach outside the part, which is trimmed off."""
    kind: str                     # through, center, surface, corner
    sections: list[np.ndarray]
    live: tuple[bool, bool]       # whether the first and the last section are fronts that grow
    normal: np.ndarray            # the starting crack plane's normal
    origin: np.ndarray
    size: float                   # starting size, mm
    notes: list[str] = field(default_factory=list)

    def front_sections(self) -> list[int]:
        return [i for i, ok in ((0, self.live[0]), (len(self.sections) - 1, self.live[1])) if ok]


@dataclass
class Sizing:
    h: float                      # far from the crack, mm
    h_front: float                # along the front, mm
    h_crack: float                # over the crack faces, mm
    curvature: int = 16


@dataclass
class CrackedMesh:
    mesh: Mesh                    # crack faces carry tri_surf CRACK_FACE
    crack_tris: np.ndarray        # (k, 6) crack faces on the lower side, ordered so their normals point up
    pairs: np.ndarray             # (m, 2) a crack face node below and its copy above
    fronts: list[np.ndarray]      # corner node indices along each front, in order
    sizing: Sizing
    area: float                   # crack face area, mm2
    notes: list[str] = field(default_factory=list)
    quarter_points: int = 0


class CrackError(RuntimeError):
    def __init__(self, message: str, hint: str = ""):
        super().__init__(message)
        self.hint = hint


class PartSeparated(CrackError):
    """The crack has cut the part in two."""


def _section_curve(occ, pts: np.ndarray) -> int:
    tags = [occ.addPoint(*map(float, p)) for p in pts]
    return occ.addSpline(tags) if len(tags) > 2 else occ.addLine(tags[0], tags[1])


def _crack_surfaces(gmsh, crack: Crack) -> list[tuple[int, int]]:
    occ = gmsh.model.occ
    wires = [occ.addWire([_section_curve(occ, s)]) for s in crack.sections]
    return [d for d in occ.addThruSections(wires, makeSolid=False, makeRuled=True) if d[0] == 2]


def _nearest(gmsh, dim: int, tags: list[int], pts: np.ndarray) -> list[float]:
    """For each entity, the largest distance from the points to it."""
    out = []
    for t in tags:
        d = 0.0
        for p in pts:
            q, _ = gmsh.model.getClosestPoint(dim, t, list(map(float, p)))
            d = max(d, float(np.linalg.norm(np.asarray(q) - p)))
        out.append(d)
    return out


def _probe_points(gmsh, dim: int, tag: int, want: int = 4) -> np.ndarray:
    if dim == 2:
        got = _face_samples(gmsh, tag, want)
        if got is not None:
            return got[0]
    lo, hi = gmsh.model.getParametrizationBounds(dim, tag)
    u = np.asarray(lo) + (np.asarray(hi) - np.asarray(lo)) * np.linspace(0.2, 0.8, want)[:, None]
    return np.asarray(gmsh.model.getValue(dim, tag, u.ravel().tolist()), float).reshape(-1, 3)


def _build(prep: PreparedGeometry, crack: Crack, sizing: Sizing, alg3d: int, curved: bool) -> dict:
    """In a child process: cut the crack into the part, mesh it, and return raw arrays in gmsh numbering."""
    t0 = time.time()
    with gmsh_session() as gmsh:
        occ = gmsh.model.occ
        load_into_session(gmsh, prep)
        vols = gmsh.model.getEntities(3)
        orig = [t for _, t in gmsh.model.getEntities(2)]
        boxes = {t: np.asarray(gmsh.model.getBoundingBox(2, t)) for t in orig}
        copies = dict(zip(orig, [t for _, t in occ.copy([(2, t) for t in orig])]))
        surfs = _crack_surfaces(gmsh, crack)
        inside, _ = occ.intersect(surfs, vols, removeObject=True, removeTool=False)
        occ.synchronize()
        inside = [d for d in inside if d[0] == 2]
        if not inside:
            raise CrackError("the crack does not cut into the part", "Check where the crack starts and its normal")
        # a crack plane can also cross other walls far from the crack; keep only the pieces the crack is made of
        ref = np.asarray(crack.sections[0 if crack.live[0] else -1], float)
        ref = ref[len(ref) // 2]
        if len(inside) > 1:
            gap = _nearest(gmsh, 2, [t for _, t in inside], [ref])
            near = min(gap)
            drop = [d for d, g in zip(inside, gap) if g > near + 0.05 * crack.size + 1e-9]
            if drop:
                occ.remove(drop, recursive=True)
                inside = [d for d in inside if d not in drop]
        _, omap = occ.fragment(vols, inside)
        occ.synchronize()
        vols_after = gmsh.model.getEntities(3)
        if len(vols_after) > len(vols):
            raise PartSeparated("the crack has cut through the part")
        cracks = sorted({t for m in omap[len(vols):] for d, t in m if d == 2})
        bound = sorted({int(s) for _, v in vols_after for s in gmsh.model.getAdjacencies(3, v)[1]} - set(cracks))
        # every boundary face comes back with a new tag: find which original face each piece of it lies on
        diag = prep.diag
        tol = 1e-5 * diag + 1e-7
        to_orig = {}
        for s in bound:
            pts = _probe_points(gmsh, 2, s)
            cands = [o for o in orig if (pts >= boxes[o][:3] - 10 * tol).all() and (pts <= boxes[o][3:] + 10 * tol).all()]
            if not cands:
                to_orig[s] = -1
                continue
            gaps = _nearest(gmsh, 2, [copies[o] for o in cands], pts)
            k = int(np.argmin(gaps))
            to_orig[s] = cands[k] if gaps[k] < 10 * tol else -1
        occ.remove([(2, c) for c in copies.values()], recursive=True)
        occ.synchronize()
        # the crack's edges: mouths lie on the part's faces and go into their meshes, the rest is the front
        front, mouths = [], []
        crack_curves = sorted({int(c) for s in cracks for c in gmsh.model.getAdjacencies(2, s)[1]})
        for c in crack_curves:
            up = [int(s) for s in gmsh.model.getAdjacencies(1, c)[0]]
            if any(s in bound for s in up):
                continue
            pts = _probe_points(gmsh, 1, c, 3)
            gaps = _nearest(gmsh, 2, bound, pts)
            k = int(np.argmin(gaps))
            if gaps[k] < 10 * tol:
                gmsh.model.mesh.embed(1, [c], 2, bound[k])
                mouths.append(c)
            elif sum(1 for s in up if s in cracks) == 1:
                front.append(c)
        if not front:
            raise CrackError("the crack has no front inside the part", "The crack may already cross the part")
        _fields(gmsh, sizing, front, cracks)
        opt = gmsh.option.setNumber
        opt("Mesh.MeshSizeMax", sizing.h)
        opt("Mesh.MeshSizeMin", 0.0)
        opt("Mesh.MeshSizeFromPoints", 0)
        opt("Mesh.MeshSizeFromCurvature", sizing.curvature)
        opt("Mesh.MeshSizeExtendFromBoundary", 0)
        opt("Mesh.Algorithm", 6)
        opt("Mesh.Algorithm3D", alg3d)
        opt("Mesh.Optimize", 1)
        opt("Mesh.ElementOrder", 1)
        try:
            gmsh.model.mesh.generate(3)
            opt("Mesh.SecondOrderLinear", 0 if curved else 1)
            gmsh.model.mesh.setOrder(2)
        except Exception as e:  # noqa: BLE001
            raise MeshError(str(e)) from e
        types, etags, enodes = gmsh.model.mesh.getElements(3)
        if 11 not in list(types):
            raise MeshError("no tetrahedra were generated")
        k = list(types).index(11)
        straightened = 0
        if curved:
            q = np.asarray(gmsh.model.mesh.getElementQualities(etags[k], "minDetJac"))
            if (q <= 0).any():
                straightened = int((q <= 0).sum())
                gmsh.model.mesh.setOrder(1)
                opt("Mesh.SecondOrderLinear", 1)
                gmsh.model.mesh.setOrder(2)
                types, etags, enodes = gmsh.model.mesh.getElements(3)
                k = list(types).index(11)
        ntags, coords, _ = gmsh.model.mesh.getNodes()
        tri = {}
        for s in bound + cracks:
            st, _, sn = gmsh.model.mesh.getElements(2, s)
            for tt, nn in zip(st, sn):
                if tt == 9:
                    tri[s] = np.asarray(nn, np.int64).reshape(-1, 6)
        lines = []
        for c in front:
            st, _, sn = gmsh.model.mesh.getElements(1, c)
            for tt, nn in zip(st, sn):
                if tt == 8:
                    lines.append(np.asarray(nn, np.int64).reshape(-1, 3))
    return {"ntags": np.asarray(ntags, np.int64), "coords": np.asarray(coords).reshape(-1, 3),
            "tets": np.asarray(enodes[k], np.int64).reshape(-1, 10), "tri": tri, "cracks": cracks,
            "to_orig": to_orig, "lines": lines, "straightened": straightened, "t": time.time() - t0}


def _fields(gmsh, sizing: Sizing, front: list[int], cracks: list[int]):
    """A tube of front sized elements around the front, growing back to the global size, and the crack faces at
    their own size."""
    fld = gmsh.model.mesh.field
    length = sum(gmsh.model.occ.getMass(1, c) for c in front)
    d = fld.add("Distance")
    fld.setNumbers(d, "CurvesList", front)
    fld.setNumber(d, "Sampling", int(np.clip(2 * length / sizing.h_front, 20, 4000)))
    t = fld.add("Threshold")
    core = 3.0 * sizing.h_front
    fld.setNumber(t, "InField", d)
    fld.setNumber(t, "SizeMin", sizing.h_front)
    fld.setNumber(t, "SizeMax", sizing.h)
    fld.setNumber(t, "DistMin", core)
    fld.setNumber(t, "DistMax", core + (sizing.h - sizing.h_front) / 0.4)
    d2 = fld.add("Distance")
    fld.setNumbers(d2, "SurfacesList", cracks)
    fld.setNumber(d2, "Sampling", 60)
    t2 = fld.add("Threshold")
    fld.setNumber(t2, "InField", d2)
    fld.setNumber(t2, "SizeMin", sizing.h_crack)
    fld.setNumber(t2, "SizeMax", sizing.h)
    fld.setNumber(t2, "DistMin", 0.0)
    fld.setNumber(t2, "DistMax", (sizing.h - sizing.h_crack) / 0.4 + 1e-9)
    m = fld.add("Min")
    fld.setNumbers(m, "FieldsList", [t, t2])
    fld.setAsBackgroundMesh(m)


def build_cracked_mesh(prep: PreparedGeometry, crack: Crack, sizing: Sizing, quarter: bool = True,
                       timeout: float = 180.0, dump: Path | None = None) -> CrackedMesh:
    """Mesh the part with the crack in it, open the crack and move the front's midside nodes to quarter points.

    A cracked part meshes in seconds, so an attempt that runs for minutes has stalled in gmsh's boundary recovery,
    and the next algorithm takes over. ``dump`` keeps the inputs in a file for a crack that will not mesh.
    """
    from ..mesh.isolate import IsolatedCrash, IsolatedTimeout, run_isolated

    if dump is not None:
        import pickle
        dump.parent.mkdir(parents=True, exist_ok=True)
        dump.write_bytes(pickle.dumps((prep, crack, sizing)))
    errors = []
    for alg3d, curved in ((1, True), (10, True), (4, False)):
        try:
            raw = run_isolated(_build, prep, crack, sizing, alg3d, curved, timeout=timeout)
            break
        except (MeshError, IsolatedCrash, IsolatedTimeout) as e:
            errors.append(str(e))
    else:
        raise CrackError(f"gmsh could not mesh the part with the crack in it ({'; '.join(errors[:2])})",
                         "A smaller growth step or a coarser --mesh-size may help")
    return _assemble(raw, crack, sizing, quarter)


def _assemble(raw: dict, crack: Crack, sizing: Sizing, quarter: bool) -> CrackedMesh:
    """Compact the numbering, open the crack and build the boundary the loads and supports select from."""
    ntags, coords, tets_raw = raw["ntags"], raw["coords"], raw["tets"]
    used = np.unique(tets_raw)
    lut = np.full(int(max(ntags.max(), used.max())) + 1, -1, np.int64)
    lut[used] = np.arange(len(used))
    pos = np.full(lut.size, -1, np.int64)
    pos[ntags] = np.arange(len(ntags))
    X = coords[pos[used]]
    tets = lut[tets_raw][:, GMSH_TO_CCX_TET10]
    if flat_elements(X, tets):
        raise MeshError("the mesher left flat elements around the crack")
    n0 = len(X)
    crack_tris, ent = [], []
    for s in raw["cracks"]:
        if s in raw["tri"]:
            crack_tris.append(lut[raw["tri"][s]])
            ent.append(np.full(len(raw["tri"][s]), s))
    crack_tris, ent = np.vstack(crack_tris), np.concatenate(ent)
    crack_tris = _orient(X, crack_tris, ent, crack.normal)
    fronts, front_nodes = _chains(raw["lines"], lut)
    dup = np.setdiff1d(np.unique(crack_tris), front_nodes)
    upper = _upper_tets(X, tets, crack_tris, dup)
    copy_of = np.full(n0, -1, np.int64)
    copy_of[dup] = n0 + np.arange(len(dup))
    sub = tets[upper]
    hit = np.isin(sub, dup)
    sub[hit] = copy_of[sub[hit]]
    tets[upper] = sub
    X = np.vstack([X, X[dup]])
    orig_of = np.concatenate([np.arange(n0), dup])
    moved = _quarter_points(X, tets, front_nodes) if quarter else 0
    tris, E, F, _ = _boundary(tets, X, 2)
    keys = {}
    for s, t in raw["tri"].items():
        tag = CRACK_FACE if s in raw["cracks"] else raw["to_orig"].get(s, -1)
        for row in np.sort(lut[t[:, :3]], axis=1):
            keys[(int(row[0]), int(row[1]), int(row[2]))] = tag
    corners = np.sort(orig_of[tris[:, :3]], axis=1)
    tri_surf = np.array([keys.get((int(a), int(b), int(c)), -1) for a, b, c in corners], np.int64)
    notes = []
    if raw["straightened"]:
        notes.append(f"{raw['straightened']} curved elements folded, so the cracked mesh uses straight sided "
                     "second order elements.")
    mesh = Mesh(X, tets, tris, tri_surf, E, F, 2, None, raw["t"], notes)
    _, area, _ = mesh.tri_geometry()
    pairs = np.column_stack([dup, copy_of[dup]])
    a, b, c = (X[crack_tris[:, i]] for i in range(3))
    crack_area = float(0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1).sum())
    return CrackedMesh(mesh, crack_tris, pairs, fronts, sizing, crack_area, notes, moved)


def _orient(X: np.ndarray, tris: np.ndarray, ent: np.ndarray, normal: np.ndarray) -> np.ndarray:
    """Turn every crack face so the normals agree across the faces the crack is made of, pointing along the
    starting crack normal. gmsh orders each face's triangles consistently; neighbors are matched across the
    edges they share."""
    nrm = np.cross(X[tris[:, 1]] - X[tris[:, 0]], X[tris[:, 2]] - X[tris[:, 0]])
    ids = list(np.unique(ent))
    links: dict[int, dict[int, bool]] = {e: {} for e in ids}   # neighbor face -> whether it already agrees
    edge_owner: dict[tuple[int, int], int] = {}
    for i, t in enumerate(tris[:, :3]):
        for a, b in ((t[0], t[1]), (t[1], t[2]), (t[2], t[0])):
            key = (min(a, b), max(a, b))
            j = edge_owner.get(key)
            if j is None:
                edge_owner[key] = i
            elif ent[j] != ent[i] and ent[j] not in links[ent[i]]:
                same = float(nrm[i] @ nrm[j]) > 0
                links[ent[i]][ent[j]] = links[ent[j]][ent[i]] = same
    flip = {}
    for root in ids:   # walk out from each face not reached yet, flipping neighbors to agree
        if root in flip:
            continue
        flip[root] = False
        todo = [root]
        while todo:
            e = todo.pop()
            for nb, same in links[e].items():
                if nb not in flip:
                    flip[nb] = flip[e] ^ (not same)
                    todo.append(nb)
    out = tris.copy()
    turn = np.array([flip[e] for e in ent])
    out[turn] = out[turn][:, [0, 2, 1, 5, 4, 3]]
    nrm[turn] *= -1
    if float(nrm.sum(0) @ np.asarray(normal, float)) < 0:
        out = out[:, [0, 2, 1, 5, 4, 3]]
    return out


def _chains(lines: list[np.ndarray], lut: np.ndarray) -> tuple[list[np.ndarray], np.ndarray]:
    """Front line elements (corner, corner, midside) into ordered chains of corner nodes, and every front node."""
    if not lines:
        return [], np.zeros(0, np.int64)
    seg = lut[np.vstack(lines)]
    nxt: dict[int, list[int]] = {}
    for a, b, _ in seg:
        nxt.setdefault(int(a), []).append(int(b))
        nxt.setdefault(int(b), []).append(int(a))
    seen, chains = set(), []
    ends = [n for n, v in nxt.items() if len(v) == 1] or [next(iter(nxt))]
    for start in ends + list(nxt):
        if start in seen:
            continue
        chain, prev, cur = [start], None, start
        seen.add(start)
        while True:
            step = [v for v in nxt[cur] if v != prev and v not in seen]
            if not step:
                break
            prev, cur = cur, step[0]
            chain.append(cur)
            seen.add(cur)
        chains.append(np.asarray(chain, np.int64))
    return chains, np.unique(seg)


def _upper_tets(X: np.ndarray, tets: np.ndarray, crack_tris: np.ndarray, dup: np.ndarray) -> np.ndarray:
    """Tetrahedra on the upper side of the crack, among those touching a crack node that gets a copy.

    Tetrahedra that share a face holding such a node, and that is not a crack face, are on the same side. Each
    connected group touches a crack face somewhere, and the side of that crack face it sits on decides the group.
    """
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    is_dup = np.zeros(len(X), bool)
    is_dup[dup] = True
    touch = np.nonzero(is_dup[tets[:, :4]].any(axis=1))[0]
    faces_local = [(0, 1, 2), (0, 1, 3), (1, 2, 3), (0, 2, 3)]
    rows, keys, opp = [], [], []
    for f in faces_local:
        c = tets[touch][:, list(f)]
        rows.append(np.arange(len(touch)))
        keys.append(np.sort(c, axis=1))
        opp.append(tets[touch][:, [k for k in range(4) if k not in f][0]])
    rows, keys, opp = np.concatenate(rows), np.vstack(keys), np.concatenate(opp)
    n = len(X)
    code = (keys[:, 0] * n + keys[:, 1]) * n + keys[:, 2]
    ck = np.sort(crack_tris[:, :3], axis=1)
    crack_code = (ck[:, 0] * n + ck[:, 1]) * n + ck[:, 2]
    on_crack = np.isin(code, crack_code)
    order = np.argsort(code, kind="stable")
    sc = code[order]
    same = np.nonzero(sc[1:] == sc[:-1])[0]
    i, j = order[same], order[same + 1]
    keep = ~on_crack[i] & is_dup[keys[i]].any(axis=1)
    g = coo_matrix((np.ones(int(keep.sum())), (rows[i][keep], rows[j][keep])), shape=(len(touch), len(touch)))
    _, label = connected_components(g, directed=False)
    # the side of each crack face its tetrahedra sit on
    nrm = np.cross(X[crack_tris[:, 1]] - X[crack_tris[:, 0]], X[crack_tris[:, 2]] - X[crack_tris[:, 0]])
    where = {int(c): k for k, c in enumerate(crack_code)}
    up_labels = set()
    for r, c, o in zip(rows[on_crack], code[on_crack], opp[on_crack]):
        k = where[int(c)]
        if float((X[o] - X[crack_tris[k, 0]]) @ nrm[k]) > 0:
            up_labels.add(int(label[r]))
    up = np.isin(label, list(up_labels))
    return touch[up]


def _quarter_points(X: np.ndarray, tets: np.ndarray, front_nodes: np.ndarray) -> int:
    """Move the midside node of every edge that runs from a front node to the quarter point next to the front."""
    on = np.zeros(len(X), bool)
    on[front_nodes] = True
    done = np.zeros(len(X), bool)
    count = 0
    for k, (a, b) in enumerate(_TET_EDGES):
        ia, ib, im = tets[:, a], tets[:, b], tets[:, 4 + k]
        for tip, far in ((ia, ib), (ib, ia)):
            sel = on[tip] & ~on[far] & ~done[im]
            if sel.any():
                m = im[sel]
                m, first = np.unique(m, return_index=True)
                X[m] = X[tip[sel][first]] + 0.25 * (X[far[sel][first]] - X[tip[sel][first]])
                done[m] = True
                count += len(m)
    return count
