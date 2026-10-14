"""What a CAD part means for flow: its walls, the openings of any passage through it, and which openings join.

An opening is a hole in a flat face whose far side is empty, like the end of a pipe or a port on a manifold: the
loop of edges inside the face is capped with a flat lid, and a point just behind the lid that ray casting finds
outside the material seeds the passage. Voxels then tell which openings share a passage: the part's surface and
every lid are drawn into a grid, the empty voxels are grouped by six way connection, and openings whose seeds land
in the same enclosed group join through the part. A group that reaches the edge of the grid is the outside air,
so a lid that fails to seal its passage is caught rather than meshed.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass
class Opening:
    id: int
    face: int                     # the CAD face the hole is in
    centre: np.ndarray            # mm
    normal: np.ndarray            # unit, pointing out of the passage
    area: float                   # mm2
    perimeter: float              # mm
    tris: np.ndarray              # the lid, (n, 3, 3) mm
    seed: np.ndarray              # a point in the passage just behind the lid, mm
    passage: int = -1

    @property
    def dh(self) -> float:
        return 4 * self.area / self.perimeter

    def describe(self) -> str:
        c = ", ".join(f"{v:.4g}" for v in self.centre)
        n = ", ".join(f"{v:+.2f}" for v in self.normal)
        return f"opening {self.id}: {self.area:.4g} mm2 (hydraulic diameter {self.dh:.3g} mm) at ({c}), facing ({n})"


@dataclass
class Passage:
    id: int
    openings: list[int]
    volume: float                 # mm3
    lo: np.ndarray
    hi: np.ndarray


@dataclass
class PartGeometry:
    source: str
    prep: object
    X: np.ndarray                 # surface vertices, mm
    tris: np.ndarray              # (n, 3) vertex indices, outward
    tri_face: np.ndarray          # CAD face of each triangle
    faces: dict
    openings: list[Opening] = field(default_factory=list)
    passages: list[Passage] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def xyz(self) -> np.ndarray:
        return self.X[self.tris]

    @property
    def lo(self) -> np.ndarray:
        return self.X.min(0)

    @property
    def hi(self) -> np.ndarray:
        return self.X.max(0)

    @property
    def diag(self) -> float:
        return float(np.linalg.norm(self.hi - self.lo))

    @property
    def area(self) -> float:
        t = self.xyz
        return float(np.linalg.norm(np.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0]), axis=1).sum() / 2)

    def inside(self, pts: np.ndarray) -> np.ndarray:
        """True where points lie inside the material, by the parity of ray crossings along three directions."""
        return inside_surface(self.xyz, np.atleast_2d(pts))

    def passage_of(self, opening: int) -> Passage | None:
        pid = self.openings[opening].passage
        return next((p for p in self.passages if p.id == pid), None)


def load(path: str, work: str | Path, params: dict | None = None, fineness: float = 1.0) -> PartGeometry:
    """Read a STEP, IGES, BREP, STL or parametric .py part, triangulate its surface finely enough for snapping a
    flow grid onto, and find its openings and passages."""
    from ..geometry.cad import prepare
    from ..geometry.faces import analyze_faces
    from ..geometry.parametric import build_step, is_parametric
    from ..mesh.mesher import build_surface_mesh

    work = Path(work)
    notes = []
    if is_parametric(path):
        step, used = build_step(path, params, work)
        prep = prepare(step, work)
        notes.append(f"Built from {Path(path).name} with " + ", ".join(
            f"{k}={v:g}" if isinstance(v, (int, float)) else f"{k}={v}" for k, v in used.items()) + ".")
    else:
        prep = prepare(path, work)
    sm = build_surface_mesh(prep, h=prep.diag / (60 * fineness), curvature=16)
    g = PartGeometry(str(path), prep, np.asarray(sm.X, float), np.asarray(sm.tris[:, :3], int),
                     np.asarray(sm.tri_surf, int), analyze_faces(sm), notes=notes + list(prep.notes))
    g.openings = find_openings(g)
    g.passages = group_passages(g)
    return g


# --- openings ----------------------------------------------------------------------------------------------------
def boundary_loops(tris: np.ndarray) -> list[list[int]]:
    """The closed loops of edges that border a patch of triangles, as vertex index lists."""
    e = np.sort(np.concatenate([tris[:, [0, 1]], tris[:, [1, 2]], tris[:, [2, 0]]]), axis=1)
    uniq, count = np.unique(e, axis=0, return_counts=True)
    edges = uniq[count == 1]
    adj: dict[int, list[int]] = {}
    for a, b in edges:
        adj.setdefault(int(a), []).append(int(b))
        adj.setdefault(int(b), []).append(int(a))
    loops, seen = [], set()
    for start in adj:
        if start in seen:
            continue
        loop, prev, cur = [start], None, start
        seen.add(start)
        while True:
            nxt = [v for v in adj[cur] if v != prev]
            if not nxt:
                break
            prev, cur = cur, nxt[0]
            if cur == start:
                break
            if cur in seen:
                break
            loop.append(cur)
            seen.add(cur)
        if len(loop) >= 3:
            loops.append(loop)
    return loops


def _newell(p: np.ndarray) -> np.ndarray:
    return 0.5 * np.cross(p, np.roll(p, -1, axis=0)).sum(0)


def ear_clip(p2: np.ndarray) -> list[tuple[int, int, int]]:
    """Triangles filling a simple polygon given as 2D points in order."""
    n = len(p2)
    idx = list(range(n))
    if _signed_area(p2) < 0:
        idx.reverse()
    out = []
    guard = 0
    while len(idx) > 3 and guard < 10 * n:
        guard += 1
        m = len(idx)
        for k in range(m):
            a, b, c = idx[k - 1], idx[k], idx[(k + 1) % m]
            if _cross2(p2[a], p2[b], p2[c]) <= 1e-14:
                continue
            others = [j for j in idx if j not in (a, b, c)]
            if others and _in_triangle(p2[others], p2[a], p2[b], p2[c]).any():
                continue
            out.append((a, b, c))
            idx.pop(k)
            break
        else:      # no ear found (a degenerate loop), fan the rest
            break
    for k in range(1, len(idx) - 1):
        out.append((idx[0], idx[k], idx[k + 1]))
    return out


def _signed_area(p: np.ndarray) -> float:
    return 0.5 * float(np.sum(p[:, 0] * np.roll(p[:, 1], -1) - np.roll(p[:, 0], -1) * p[:, 1]))


def _cross2(a, b, c) -> float:
    return float((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]))


def _in_triangle(p, a, b, c) -> np.ndarray:
    def s(p1, p2, q):
        return (p2[0] - p1[0]) * (q[:, 1] - p1[1]) - (p2[1] - p1[1]) * (q[:, 0] - p1[0])
    d1, d2, d3 = s(a, b, p), s(b, c, p), s(c, a, p)
    return (d1 >= -1e-12) & (d2 >= -1e-12) & (d3 >= -1e-12)


def find_openings(g: PartGeometry) -> list[Opening]:
    out = []
    for fid, info in sorted(g.faces.items()):
        if info.kind != "plane" or info.normal is None:
            continue
        loops = boundary_loops(g.tris[g.tri_face == fid])
        if len(loops) < 2:
            continue
        n = np.asarray(info.normal, float)
        areas = [abs(float(_newell(g.X[lp]) @ n)) for lp in loops]
        outer = int(np.argmax(areas))
        u = np.cross(n, [1.0, 0, 0] if abs(n[0]) < 0.9 else [0, 1.0, 0])
        u /= np.linalg.norm(u)
        v = np.cross(n, u)
        for k, lp in enumerate(loops):
            if k == outer or areas[k] <= 0:
                continue
            p = g.X[lp]
            tri_idx = ear_clip(np.c_[p @ u, p @ v])
            tris = p[np.array(tri_idx)]
            ta = np.linalg.norm(np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0]), axis=1) / 2
            perim = float(np.linalg.norm(np.roll(p, -1, axis=0) - p, axis=1).sum())
            dh = 4 * areas[k] / perim
            c = tris[int(np.argmax(ta))].mean(0)
            seed = c - n * max(0.05 * dh, 1e-4 * g.diag)
            if g.inside(seed[None])[0]:
                continue        # the loop rims something solid, like a boss standing on the face
            centre = (tris.mean(1) * ta[:, None]).sum(0) / ta.sum()
            out.append(Opening(len(out), fid, centre, n, areas[k], perim, tris, seed))
    return out


# --- passages ----------------------------------------------------------------------------------------------------
def sample_triangles(tris: np.ndarray, spacing: float) -> np.ndarray:
    """Points covering each triangle no further apart than spacing."""
    e = np.stack([tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 1], tris[:, 0] - tris[:, 2]], 1)
    nsub = np.maximum(1, np.ceil(np.linalg.norm(e, axis=2).max(1) / spacing)).astype(int)
    out = [tris.reshape(-1, 3)]
    for n in np.unique(nsub):
        t = tris[nsub == n]
        i, j = np.meshgrid(np.arange(n + 1), np.arange(n + 1), indexing="ij")
        keep = i + j <= n
        a, b = i[keep] / n, j[keep] / n
        w = np.c_[1 - a - b, a, b]
        out.append(np.einsum("pk,tkd->tpd", w, t).reshape(-1, 3))
    return np.concatenate(out)


def group_passages(g: PartGeometry, max_voxels: float = 6e6) -> list[Passage]:
    from scipy import ndimage

    if not g.openings:
        return []
    size = g.hi - g.lo
    h = min(min(o.dh for o in g.openings) / 4, g.diag / 60)
    h = max(h, (np.prod(size + 4 * h) / max_voxels) ** (1 / 3))
    lo = g.lo - 2 * h
    shape = np.ceil((size + 4 * h) / h).astype(int) + 1
    wall = np.zeros(shape, bool)
    lids = np.concatenate([o.tris for o in g.openings])
    pts = sample_triangles(np.concatenate([g.xyz, lids]), 0.5 * h)
    ijk = np.clip(np.floor((pts - lo) / h).astype(int), 0, shape - 1)
    wall[ijk[:, 0], ijk[:, 1], ijk[:, 2]] = True
    labels, _ = ndimage.label(~wall)
    outside = labels[0, 0, 0]
    groups: dict[int, list[int]] = {}
    for o in g.openings:
        found = 0
        for step in range(0, 4):      # walk inward off any voxel the wall drawing covered
            q = np.floor((o.seed - o.normal * step * h - lo) / h).astype(int)
            if np.all(q >= 0) and np.all(q < shape) and labels[tuple(q)] and labels[tuple(q)] != outside:
                found = int(labels[tuple(q)])
                break
        if found:
            groups.setdefault(found, []).append(o.id)
    out = []
    for lab, ids in groups.items():
        inside = labels == lab
        cells = np.argwhere(inside)
        grown = int(ndimage.binary_dilation(inside).sum())     # the drawn wall eats about half a voxel all round
        for i in ids:
            g.openings[i].passage = len(out)
        out.append(Passage(len(out), ids, float(0.5 * (len(cells) + grown) * h ** 3), lo + cells.min(0) * h,
                           lo + (cells.max(0) + 1) * h))
    return out


# --- shared geometry helpers -------------------------------------------------------------------------------------
def inside_surface(tri: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Points inside a closed triangulated surface, by majority over three ray directions."""
    dirs = np.array([[0.5773, 0.5774, 0.5773], [-0.2673, 0.5345, -0.8018], [0.8165, -0.4082, -0.4083]])
    v0, e1, e2 = tri[:, 0], tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]
    votes = np.zeros(len(pts), int)
    for d in dirs:
        d = d / np.linalg.norm(d)
        p = np.cross(d, e2)
        det = np.einsum("tk,tk->t", p, e1)
        ok = np.abs(det) > 1e-18
        inv = np.where(ok, 1 / np.where(ok, det, 1), 0)
        for i, o in enumerate(pts):
            s = o - v0
            uu = np.einsum("tk,tk->t", s, p) * inv
            q = np.cross(s, e1)
            vv = (q @ d) * inv
            t = np.einsum("tk,tk->t", e2, q) * inv
            hits = ok & (uu >= 0) & (vv >= 0) & (uu + vv <= 1) & (t > 0)
            votes[i] += int(hits.sum()) % 2
    return votes >= 2


def frontal_area(tri: np.ndarray, d: np.ndarray, px: int = 600) -> float:
    """The area of the part's silhouette seen along d, in the units of tri squared."""
    from PIL import Image, ImageDraw

    d = np.asarray(d, float) / np.linalg.norm(d)
    u = np.cross(d, [0, 0, 1.0] if abs(d[2]) < 0.9 else [1.0, 0, 0])
    u /= np.linalg.norm(u)
    v = np.cross(d, u)
    p = np.stack([tri @ u, tri @ v], -1)
    lo, hi = p.reshape(-1, 2).min(0), p.reshape(-1, 2).max(0)
    span = float((hi - lo).max()) * 1.02 or 1.0
    scale = px / span
    img = Image.new("1", (px + 2, px + 2), 0)
    draw = ImageDraw.Draw(img)
    for t in (p - lo) * scale + 1:
        draw.polygon([tuple(x) for x in t], fill=1)
    return float(np.asarray(img).sum()) / scale ** 2


def frame(direction, up=None) -> np.ndarray:
    """A rotation whose rows are the flow direction, the side direction and up, so R @ p puts the flow along +x and
    up along +z."""
    d = np.asarray(direction, float)
    d /= np.linalg.norm(d)
    if up is None:
        up = [0, 0, 1.0] if abs(d[2]) < 0.9 else [0, 1.0, 0]
    z = np.asarray(up, float)
    z = z - (z @ d) * d
    if np.linalg.norm(z) < 1e-9:
        raise ValueError("the up direction is along the flow; give another one")
    z /= np.linalg.norm(z)
    y = np.cross(z, d)
    return np.vstack([d, y, z])


def describe(g: PartGeometry) -> list[str]:
    size = g.hi - g.lo
    lines = [f"{size[0]:.4g} x {size[1]:.4g} x {size[2]:.4g} mm, {len(g.faces)} faces, "
             f"{len(g.openings)} openings in {len(g.passages)} passages"]
    for o in g.openings:
        lines.append("  " + o.describe() + (f", passage {o.passage}" if o.passage >= 0 else ", leads nowhere"))
    for p in g.passages:
        lines.append(f"  passage {p.id}: openings {', '.join(map(str, p.openings))}, "
                     f"{p.volume / 1e3:.4g} cm3")
    return lines


def angle_between(a, b) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    return math.degrees(math.acos(np.clip(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)), -1, 1)))
