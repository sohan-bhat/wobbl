"""Heatmap images: stress, safety factor, deflection and a zoom on the governing spot.

Wobbl draws these itself with numpy and matplotlib. The outer surface of the
mesh, each quadratic face split into small flat triangles that follow its
curve, is rasterized with a depth buffer at twice the final size, smooth shaded
under three lights and averaged down; matplotlib then adds the color bar and
the labels. No OpenGL or VTK is involved, so a laptop, a headless server and CI
all draw the same picture.
"""
from __future__ import annotations

import os
import threading
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Callable

import numpy as np

BG = "#0e0f11"
FG = "#f2f2ee"
MARK = "#ffd400"      # crash test yellow
ARTIFACT = "#9aa0a6"
EDGE = "#6b7078"

SS = 2                    # supersampling: each final pixel averages SS x SS rendered ones
SUBDIVIDE = 600_000       # most small triangles from splitting quadratic faces 4 x 4; past that they split 2 x 2
CHUNK = 1 << 20           # candidate pixels the rasterizer tests in one vectorized pass
FEATURE_ANGLE = 35.0      # degrees between faces where shading stays sharp and the deflected view draws an edge
# the lights of VTK's three light kit, in camera space (x right, y up, z toward the viewer): a key light up and to
# the right and two weaker fills from below
LIGHTS = ((1.0, (0.5, 0.7071, 0.5)), (0.6, (-0.75, -0.5, 0.433)), (0.5, (0.75, -0.5, 0.433)))
OVERVIEW = (0.03, 0.05, 0.83, 0.95)   # where a whole part is framed: left, top, right, bottom, as image fractions

ISO_DIRS = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (0.8, -0.8)], float)


def _view_dir(X: np.ndarray, focus: np.ndarray | None, normal: np.ndarray | None) -> np.ndarray:
    """An isometric style direction that faces the point of interest."""
    dirs = ISO_DIRS / np.linalg.norm(ISO_DIRS, axis=1)[:, None]
    if normal is None:
        if focus is None:
            return dirs[[d[2] > 0 and d[0] > 0 and d[1] < 0 for d in dirs]][0]
        c = X.mean(0)
        normal = focus - c
        if np.linalg.norm(normal) < 1e-9:
            normal = np.array([1.0, -1.0, 1.0])
    normal = normal / np.linalg.norm(normal)
    score = dirs @ normal + 0.15 * (dirs[:, 2] > 0)
    return dirs[int(np.argmax(score))]


def _surface_normal_at(X: np.ndarray, tris: np.ndarray, p: np.ndarray) -> np.ndarray | None:
    if tris is None or len(tris) == 0:
        return None
    t = tris[:, :3]
    c = X[t].mean(1)
    i = np.argsort(np.linalg.norm(c - p, axis=1))[:6]
    n = np.cross(X[t[i, 1]] - X[t[i, 0]], X[t[i, 2]] - X[t[i, 0]])
    n = n.sum(0)
    return n / np.linalg.norm(n) if np.linalg.norm(n) > 0 else None


# ------------------------------------------------------------------ surface
def _tri_grid(n: int) -> tuple[np.ndarray, np.ndarray]:
    """Barycentric coordinates of a triangle split n times along each edge, and its n * n small triangles, each
    ordered like the big one."""
    pts, idx = [], {}
    for j in range(n + 1):
        for i in range(n + 1 - j):
            idx[i, j] = len(pts)
            pts.append((1.0 - (i + j) / n, i / n, j / n))
    tris = []
    for j in range(n):
        for i in range(n - j):
            tris.append((idx[i, j], idx[i + 1, j], idx[i, j + 1]))
            if i + j < n - 1:
                tris.append((idx[i + 1, j], idx[i + 1, j + 1], idx[i, j + 1]))
    return np.array(pts), np.array(tris)


def _shape(l: np.ndarray) -> np.ndarray:
    """6 node triangle shape functions at barycentric points; midside nodes 3, 4, 5 sit on edges 0-1, 1-2, 2-0."""
    l0, l1, l2 = l.T
    return np.column_stack([l0 * (2 * l0 - 1), l1 * (2 * l1 - 1), l2 * (2 * l2 - 1), 4 * l0 * l1, 4 * l1 * l2,
                            4 * l2 * l0])


def _shape_grad(l: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Derivatives of the 6 node shape functions along edge 0-1 and edge 0-2."""
    l0, l1, l2 = l.T
    z = np.zeros_like(l0)
    du = np.column_stack([1 - 4 * l0, 4 * l1 - 1, z, 4 * (l0 - l1), 4 * l2, -4 * l2])
    dv = np.column_stack([1 - 4 * l0, z, 4 * l2 - 1, -4 * l1, 4 * l1, 4 * (l0 - l2)])
    return du, dv


_NODES = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [.5, .5, 0], [0, .5, .5], [.5, 0, .5]], float)


def _length(v: np.ndarray) -> np.ndarray:
    """Lengths of vectors along the last axis, written out for three components."""
    return np.sqrt(v[..., 0] * v[..., 0] + v[..., 1] * v[..., 1] + v[..., 2] * v[..., 2])


def _apply(Wt: np.ndarray, F: np.ndarray) -> np.ndarray:
    """Weights (p, j) applied to per face values (k, j) or (k, j, c): (k, p) or (k, p, c), as one matrix product."""
    if F.ndim == 2:
        return F @ Wt.T
    k, j, c = F.shape
    return (Wt @ F.transpose(1, 0, 2).reshape(j, k * c)).reshape(len(Wt), k, c).transpose(1, 0, 2)


def _node_normals(X: np.ndarray, tris: np.ndarray) -> np.ndarray:
    """Outward unit normal at each node of each boundary face, (faces, nodes, 3).

    A quadratic face takes its normal from its shape functions, so a curved face shades as a curve. The faces
    around a node that meet at less than FEATURE_ANGLE share their area weighted mean and shade smoothly across
    element edges. At a sharp edge the faces on each side form their own group, found by narrowing the mean to the
    faces near it, so neither side's shading leans toward the other and the edge stays crisp.
    """
    P = X[tris]
    flat = np.cross(P[:, 1] - P[:, 0], P[:, 2] - P[:, 0])
    area = _length(flat)
    if tris.shape[1] == 6:
        du, dv = _shape_grad(_NODES)
        n = np.cross(_apply(du, P), _apply(dv, P))
    else:
        n = np.repeat(flat[:, None], tris.shape[1], axis=1)
    ln = _length(n)[..., None]
    own = flat[:, None] / np.maximum(area, 1e-300)[:, None, None]
    n = np.where(ln > 1e-9 * area[:, None, None], n / np.maximum(ln, 1e-300), own).reshape(-1, 3)
    ids, wa = tris.ravel(), np.repeat(area, tris.shape[1])
    cos = np.cos(np.radians(FEATURE_ANGLE))

    def mean(sel):
        w = wa * sel
        S = np.column_stack([np.bincount(ids, weights=n[:, c] * w, minlength=len(X)) for c in range(3)])
        return (S / np.maximum(_length(S), 1e-300)[:, None])[ids]

    def dot(a, b):
        return a[:, 0] * b[:, 0] + a[:, 1] * b[:, 1] + a[:, 2] * b[:, 2]

    m = mean(np.ones(len(ids)))
    for _ in range(2):
        m = mean(dot(n, m) > cos)
    first = dot(n, m) > cos
    m2 = mean(~first)
    second = ~first & (dot(n, m2) > cos)
    return np.where(first[:, None], m, np.where(second[:, None], m2, n)).reshape(len(tris), -1, 3)


@dataclass
class Surface:
    """The boundary as flat triangles: points, triangles, unit normals and nodal fields carried to the points."""
    P: np.ndarray
    T: np.ndarray
    N: np.ndarray
    fields: dict[str, np.ndarray] = field(default_factory=dict)


def surface(X, tris, fields: dict | None = None, n: int | None = None, normals=None) -> Surface:
    """Split each boundary face n times along each edge, quadratic faces along their curve, and carry nodal fields,
    scalar or vector, to the new points with each face's shape functions. normals, from _node_normals, can be
    passed in when several surfaces share the same faces."""
    X, tris = np.asarray(X, float), np.asarray(tris)
    quad = tris.shape[1] == 6
    if n is None:
        n = (4 if 16 * len(tris) <= SUBDIVIDE else 2) if quad else 1
    l, sub = _tri_grid(n)
    W = _shape(l) if quad else l
    P = _apply(W, X[tris]).reshape(-1, 3)
    N = _apply(W, _node_normals(X, tris) if normals is None else normals).reshape(-1, 3)
    N /= np.maximum(_length(N), 1e-300)[:, None]
    out = {}
    for name, f in (fields or {}).items():
        f = np.asarray(f, float)
        out[name] = _apply(W, f[tris]).ravel() if f.ndim == 1 else _apply(W, f[tris]).reshape(-1, f.shape[1])
    T = (np.arange(len(tris))[:, None, None] * len(l) + sub[None]).reshape(-1, 3)
    return Surface(P, T, N, out)


def subdivision(cam: "Camera", size: tuple[int, int], X, tris) -> int:
    """How finely to split quadratic faces for a view: in two along each edge while the faces on screen are under
    20 pixels across, else in four, so curves and fields stay smooth without drawing triangles smaller than a
    pixel. Splitting in two keeps the midside nodes, where a peak often sits; flat faces are not split."""
    tris = np.asarray(tris)
    if tris.shape[1] != 6:
        return 1
    xa, ya, _ = cam.project(X[tris[:, 0]])
    xb, yb, _ = cam.project(X[tris[:, 1]])
    on = (xa >= 0) & (xa < size[0]) & (ya >= 0) & (ya < size[1])
    px = np.hypot(xb - xa, yb - ya)[on] if on.any() else np.hypot(xb - xa, yb - ya)
    return 4 if float(np.median(px)) > 20.0 else 2


def fitted(cam: "Camera", size: tuple[int, int], X, tris, fields: dict | None = None, normals=None) -> Surface:
    """surface() split as finely as this view needs."""
    return surface(X, tris, fields, subdivision(cam, size, np.asarray(X, float), tris), normals)


def feature_edges(X, tris, angle: float = FEATURE_ANGLE) -> np.ndarray:
    """Segments (s, 2, 3) along the edges where boundary faces meet at more than angle degrees, through the midside
    node of quadratic faces."""
    X, tris = np.asarray(X, float), np.asarray(tris)
    t, k = tris[:, :3], len(tris)
    e = np.concatenate([t[:, [0, 1]], t[:, [1, 2]], t[:, [2, 0]]])
    key = np.sort(e, axis=1)
    order = np.lexsort((key[:, 1], key[:, 0]))
    same = np.all(key[order][1:] == key[order][:-1], axis=1)
    a, b = order[:-1][same], order[1:][same]
    nf = np.cross(X[t[:, 1]] - X[t[:, 0]], X[t[:, 2]] - X[t[:, 0]])
    nf /= np.maximum(np.linalg.norm(nf, axis=1, keepdims=True), 1e-300)
    sharp = a[np.einsum("ij,ij->i", nf[a % k], nf[b % k]) < np.cos(np.radians(angle))]
    p, q = X[e[sharp, 0]], X[e[sharp, 1]]
    if tris.shape[1] != 6:
        return np.stack([p, q], axis=1)
    m = X[np.concatenate([tris[:, 3], tris[:, 4], tris[:, 5]])[sharp]]
    return np.concatenate([np.stack([p, m], axis=1), np.stack([m, q], axis=1)])


# ------------------------------------------------------------------- camera
@dataclass
class Camera:
    """A perspective view and how it maps points to pixels of the final image."""
    eye: np.ndarray
    right: np.ndarray
    up: np.ndarray
    fwd: np.ndarray
    f: float
    cx: float
    cy: float

    def project(self, P, scale: float = 1.0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Pixel x and y and inverse depth (larger is nearer), on an image scale times the final size."""
        d = np.atleast_2d(np.asarray(P, float)) - self.eye
        iz = 1.0 / np.maximum(d @ self.fwd, 1e-12)
        return (self.cx + self.f * (d @ self.right) * iz) * scale, (self.cy - self.f * (d @ self.up) * iz) * scale, iz


def camera(view, pts, size: tuple[int, int], focus=None, zoom: float = 3.2, box=OVERVIEW,
           span: float | None = None) -> Camera:
    """A camera looking back along view, with z (or y for a view from straight above) pointing up.

    Without a focus point it stands where VTK's reset would put it, far enough for a 30 degree lens to take in the
    bounding sphere of pts, and the lens is set so pts fill box. With one it looks straight at the point from 2.2
    part sizes away through a lens zoom times narrower, the close up of a hot spot.
    """
    W, H = size
    view = np.asarray(view, float) / np.linalg.norm(view)
    pts = np.asarray(pts, float)
    lo, hi = pts.min(0), pts.max(0)
    span = span or float(np.linalg.norm(hi - lo)) or 1.0
    fwd = -view
    right = np.cross(fwd, [0.0, 0.0, 1.0] if abs(view[2]) < 0.95 else [0.0, 1.0, 0.0])
    right /= np.linalg.norm(right)
    up = np.cross(right, fwd)
    if focus is not None:
        eye = np.asarray(focus, float) + view * 2.2 * span
        return Camera(eye, right, up, fwd, 0.5 * H / np.tan(np.radians(15.0 / zoom)), 0.5 * W, 0.5 * H)
    eye = 0.5 * (lo + hi) + view * 0.5 * float(np.linalg.norm(hi - lo)) / np.sin(np.radians(15.0))
    d = pts - eye
    z = d @ fwd
    u, v = (d @ right) / z, (d @ up) / z
    x0, y0, x1, y1 = box[0] * W, box[1] * H, box[2] * W, box[3] * H
    f = min((x1 - x0) / max(np.ptp(u), 1e-12), (y1 - y0) / max(np.ptp(v), 1e-12))
    return Camera(eye, right, up, fwd, f, 0.5 * (x0 + x1) - f * 0.5 * (u.max() + u.min()),
                  0.5 * (y0 + y1) + f * 0.5 * (v.max() + v.min()))


# --------------------------------------------------------------- rasterizer
def _rasterize(x, y, iz, T, W: int, H: int):
    """Depth buffered rasterization of triangles from the pixel coordinates and inverse depths of their corners.

    Returns per pixel the nearest triangle (-1 for none), the screen space weights of its corners 1 and 2 and the
    inverse depth there. Each triangle is cut into the rows of pixel centers it covers and each row into the span
    between its edges, all vectorized, so only pixels inside a triangle are ever tested, big or small.
    """
    zbuf = np.zeros(W * H, np.float32)
    tri = np.full(W * H, -1, np.int32)
    w1buf, w2buf = np.zeros(W * H, np.float32), np.zeros(W * H, np.float32)
    x0, x1, x2 = (x[T[:, c]] for c in range(3))
    y0, y1, y2 = (y[T[:, c]] for c in range(3))
    z0, z1, z2 = (iz[T[:, c]] for c in range(3))
    area = (x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0)
    j0 = np.maximum(np.ceil(np.minimum(np.minimum(y0, y1), y2) - 0.5), 0).astype(np.int64)
    j1 = np.minimum(np.floor(np.maximum(np.maximum(y0, y1), y2) - 0.5), H - 1).astype(np.int64)
    xlo = np.maximum(np.minimum(np.minimum(x0, x1), x2), 0.0)
    xhi = np.minimum(np.maximum(np.maximum(x0, x1), x2), float(W))
    ok = np.nonzero((np.abs(area) > 1e-12) & (j1 >= j0) & (xhi > xlo))[0]
    # per triangle, each corner weight is a straight function of the pixel, w = a x + b y + c, and so is the
    # inverse depth; worked out once here, a row only evaluates them
    ok_area = area[ok]
    a1, b1 = ((y2 - y0)[ok] / ok_area), (-(x2 - x0)[ok] / ok_area)
    a2, b2 = (-(y1 - y0)[ok] / ok_area), ((x1 - x0)[ok] / ok_area)
    c1 = -(a1 * x0[ok] + b1 * y0[ok])
    c2 = -(a2 * x0[ok] + b2 * y0[ok])
    dz1, dz2 = (z1 - z0)[ok], (z2 - z0)[ok]
    za, zb, zc = dz1 * a1 + dz2 * a2, dz1 * b1 + dz2 * b2, z0[ok] + dz1 * c1 + dz2 * c2
    j0, rows, xlo, xhi = j0[ok], (j1 - j0)[ok] + 1, xlo[ok], xhi[ok]
    work = np.cumsum(rows * np.maximum(xhi - xlo, 1.0))   # about how many pixels each prefix covers
    start = 0
    while start < len(ok):
        stop = max(int(np.searchsorted(work, work[start] - rows[start] * max(xhi[start] - xlo[start], 1.0) + CHUNK,
                                       side="right")), start + 1)
        sel = np.arange(start, stop)
        t = np.repeat(sel, rows[sel])
        py = j0[t] + (np.arange(len(t)) - np.repeat(np.cumsum(rows[sel]) - rows[sel], rows[sel]))
        cy = py + 0.5
        # along the row each weight is s x + c; keep the x where all three are >= 0
        s1, k1 = a1[t], b1[t] * cy + c1[t]
        s2, k2 = a2[t], b2[t] * cy + c2[t]
        lo, hi = xlo[t], xhi[t]
        for sl, k in ((s1, k1), (s2, k2), (-s1 - s2, 1.0 - k1 - k2)):
            with np.errstate(divide="ignore", invalid="ignore"):
                edge = (-1e-7 - k) / sl
            lo = np.where(sl > 1e-12, np.maximum(lo, edge), lo)
            hi = np.where(sl < -1e-12, np.minimum(hi, edge), np.where((np.abs(sl) <= 1e-12) & (k < -1e-7), -1.0, hi))
        i0 = np.maximum(np.ceil(lo - 0.5), 0).astype(np.int64)
        n = np.maximum(np.minimum(np.floor(hi - 0.5), W - 1).astype(np.int64) - i0 + 1, 0)
        r = np.repeat(np.arange(len(t)), n)
        px = i0[r] + (np.arange(len(r)) - np.repeat(np.cumsum(n) - n, n))
        tr = t[r]
        cx = (px + 0.5).astype(np.float32)
        z = za[tr].astype(np.float32) * cx + (zb[t] * cy + zc[t]).astype(np.float32)[r]
        pix = py[r] * W + px
        np.maximum.at(zbuf, pix, z)
        win = np.nonzero(z >= zbuf[pix])[0]
        pix, rw, cw = pix[win], r[win], cx[win]
        tri[pix] = ok[tr[win]]
        w1buf[pix] = s1[rw] * cw + k1[rw]
        w2buf[pix] = s2[rw] * cw + k2[rw]
        start = stop
    return tri, w1buf, w2buf, zbuf


def _rgb(color: str) -> np.ndarray:
    from matplotlib.colors import to_rgb

    return np.array(to_rgb(color))


@dataclass
class Look:
    """How a surface is colored by its field and lit."""
    cmap: object = "turbo"                # a matplotlib colormap or its name
    clim: tuple[float, float] = (0.0, 1.0)
    above: str | None = None              # color for values over the top of the scale, else the top color
    nan: str = "#4a4d52"                  # color where the field has no value
    ambient: float = 0.3
    diffuse: float = 0.8
    specular: float = 0.0
    power: float = 30.0
    transform: Callable[[np.ndarray], np.ndarray] | None = None   # applied to the field at each pixel


@dataclass
class Lines:
    segments: np.ndarray                  # (s, 2, 3)
    color: str = EDGE
    alpha: float = 0.6
    width: float = 1.5                    # final pixels
    depth_test: bool = True               # False draws the lines through the surface, for a see through part


def _lighting(cam: Camera, P: np.ndarray, N: np.ndarray, power: float = 30.0) -> tuple[np.ndarray, np.ndarray]:
    """Diffuse light and the specular highlight at each point, before a look scales them. The small triangles span
    a few pixels, so lighting their corners and blending across them looks the same as lighting every pixel."""
    ex, ey, ez = (cam.eye[i] - P[:, i] for i in range(3))
    inv = 1.0 / np.sqrt(ex * ex + ey * ey + ez * ez)
    ex, ey, ez = ex * inv, ey * inv, ez * inv
    nx, ny, nz = N[:, 0], N[:, 1], N[:, 2]
    diff, spec = np.zeros(len(P)), np.zeros(len(P))
    for k, (lx, ly, lz) in LIGHTS:
        L = lx * cam.right + ly * cam.up - lz * cam.fwd
        L /= np.linalg.norm(L)
        diff += k * np.maximum(nx * L[0] + ny * L[1] + nz * L[2], 0.0)
        hx, hy, hz = ex + L[0], ey + L[1], ez + L[2]
        hn = np.maximum(np.sqrt(hx * hx + hy * hy + hz * hz), 1e-300)
        spec += k * np.maximum((nx * hx + ny * hy + nz * hz) / hn, 0.0) ** power
    return diff, spec


@dataclass
class Frame:
    """A view rasterized once, ready to be colored by any look. Pixels a single triangle fills are kept as one
    sample at their center; pixels at an edge of the surface or under a line keep all SS x SS samples."""
    size: tuple[int, int]
    one: np.ndarray            # final pixels a single triangle fills
    one_attr: np.ndarray       # field, diffuse and specular light at their centers, (n, 3)
    many: np.ndarray           # final pixels kept as samples
    hit: np.ndarray            # which of their samples, (m, SS * SS), the surface covers
    many_attr: np.ndarray      # field and light at the covered samples, (hit.sum(), 3)
    on_line: np.ndarray        # which samples a line covers, (m, SS * SS)
    lines: Lines | None


def _interpolate(attr, iz, T, tri, w1, w2):
    """Perspective correct interpolation of per point attributes, (points, 3), at samples of known triangle and
    screen space weights, written out plane by plane."""
    c0, c1, c2 = T[tri, 0], T[tri, 1], T[tri, 2]
    q1, q2 = w1 * iz[c1], w2 * iz[c2]
    q0 = (1.0 - w1 - w2) * iz[c0]
    inv = 1.0 / (q0 + q1 + q2)
    q0, q1, q2 = q0 * inv, q1 * inv, q2 * inv
    out = np.empty((len(tri), attr.shape[1]), np.float32)
    for j in range(attr.shape[1]):
        a = attr[:, j]
        out[:, j] = q0 * a[c0] + q1 * a[c1] + q2 * a[c2]
    return out


# how far apart two neighboring pixels may be before their edge is antialiased: in the field, as a share of its
# spread in the view, and in the diffuse and specular light
EDGE_FIELD, EDGE_LIGHT = 0.015, 0.04


def _edges(h: int, w: int, covered: np.ndarray, feat: list[np.ndarray]) -> np.ndarray:
    """Pixels on either side of a jump to a horizontal or vertical neighbor: surface against background, or any
    feature plane (scaled so 1 is a visible step) changing by more than 1."""
    cov = covered.reshape(h, w)
    f = [a.reshape(h, w) for a in feat]
    edge = np.zeros((h, w), bool)
    for dy, dx in ((0, 1), (1, 0)):
        ca, cb = cov[:h - dy, :w - dx], cov[dy:, dx:]
        jump = ca != cb
        for a in f:
            jump |= np.abs(a[:h - dy, :w - dx] - a[dy:, dx:]) > 1.0
        edge[:h - dy, :w - dx] |= jump
        edge[dy:, dx:] |= jump
    return edge.ravel()


def _touching(x, y, T, mask: np.ndarray) -> np.ndarray:
    """Indices of the triangles whose bounding box meets a marked pixel of the (h, w) mask. Not only the triangles
    that win a pixel center are wanted: where the mesh is finer than the pixels, the triangles that fill the samples
    between centers win none, and leaving them out punches holes in the picture."""
    h, w = mask.shape
    S = np.zeros((h + 1, w + 1), np.int32)
    S[1:, 1:] = mask.cumsum(0).cumsum(1)
    xs, ys = x[T], y[T]
    x0 = np.clip(np.floor(xs.min(1)), 0, w).astype(np.int64)
    x1 = np.clip(np.floor(xs.max(1)) + 1, 0, w).astype(np.int64)
    y0 = np.clip(np.floor(ys.min(1)), 0, h).astype(np.int64)
    y1 = np.clip(np.floor(ys.max(1)) + 1, 0, h).astype(np.int64)
    return np.nonzero(S[y1, x1] - S[y0, x1] - S[y1, x0] + S[y0, x0] > 0)[0]


def frame(size: tuple[int, int], cam: Camera, surf: Surface, values, lines: Lines | None = None,
          two_sided: bool = False, power: float = 30.0) -> Frame:
    """Rasterize a surface and interpolate its field and lighting, once for any number of looks.

    The surface is rasterized once at the final size, one sample at each pixel center. Where the picture can step
    sharply from one pixel to the next, at its outline, at a crease in the light, at a jump in the field or under
    a line, the pixels are sampled again SS x SS times from the triangles around them, so edges come out smooth
    while the smooth inside, most of the picture, costs one sample a pixel.

    A closed surface hides its back faces behind its front ones, so only faces toward the camera are drawn. A
    two sided surface, a shell's midsurface, draws both sides and lights each from the side the camera sees.
    """
    w, h = size
    P, T, N = surf.P, surf.T, surf.N
    x, y, iz = cam.project(P)
    if two_sided:
        N = N * np.where(np.einsum("ij,ij->i", N, cam.eye - P) < 0, -1.0, 1.0)[:, None]
    elif len(T):
        toward = np.einsum("ij,ij->i", np.cross(P[T[:, 1]] - P[T[:, 0]], P[T[:, 2]] - P[T[:, 0]]), cam.eye - P[T[:, 0]])
        T = T[toward > 0]
    tri, w1, w2, zbuf = _rasterize(x, y, iz, T, w, h)
    diff, spec = _lighting(cam, P, N, power)
    attr = np.column_stack([np.asarray(values, float), diff, spec]).astype(np.float32)
    iz32 = iz.astype(np.float32)
    covered = tri >= 0
    pix = np.nonzero(covered)[0]
    at = _interpolate(attr, iz32, T, tri[pix], w1[pix], w2[pix])
    # what decides whether neighbors look alike: the field against its spread, and the light
    val = at[:, 0]
    finite = np.isfinite(val)
    lo, hi = np.percentile(val[finite], [1, 99]) if finite.any() else (0.0, 1.0)
    feat = [np.zeros(h * w, np.float32) for _ in range(3)]
    feat[0][pix] = np.where(finite, val / (max(hi - lo, 1e-30) * EDGE_FIELD), 1e9)
    feat[1][pix] = at[:, 1] / EDGE_LIGHT
    feat[2][pix] = at[:, 2] / EDGE_LIGHT
    edge = _edges(h, w, covered, feat)
    W, H = w * SS, h * SS
    line = None
    if lines is not None and len(lines.segments):
        hits = _line_samples(zbuf, w, cam, lines, W, H)
        line = np.zeros(W * H, bool)
        line[hits] = True
        py, px = np.divmod(hits, W)
        edge[(py // SS) * w + px // SS] = True
    many = np.nonzero(edge)[0]
    keep = ~edge[pix]
    one, one_attr = pix[keep], at[keep]
    # the samples of the edge pixels, from the triangles seen in and around them
    k = np.arange(SS * SS)
    sub = ((many // w) * SS * W + (many % w) * SS)[:, None] + ((k // SS) * W + k % SS)[None, :]
    near = edge.reshape(h, w).copy()
    near[1:] |= near[:-1].copy()
    near[:-1] |= near[1:].copy()
    near[:, 1:] |= near[:, :-1].copy()
    near[:, :-1] |= near[:, 1:].copy()
    hit = np.zeros(sub.shape, bool)
    many_attr = np.zeros((0, 3), np.float32)
    if near.any():
        cand = _touching(x, y, T, near)
        t2, v1, v2, _ = _rasterize(x * SS, y * SS, iz, T[cand], W, H)
        got = t2[sub]
        hit = got >= 0
        smp = sub[hit]
        many_attr = _interpolate(attr, iz32, T, cand[got[hit]], v1[smp], v2[smp])
    on_line = line[sub] if line is not None else np.zeros(sub.shape, bool)
    return Frame(size, one, one_attr, many, hit, many_attr, on_line, lines)


def paint(fr: Frame, look: Look, coverage: bool = False):
    """Color a frame: an (H, W, 3) array of floats in 0..1, and with coverage=True also the share of each pixel
    the surface or lines cover, for laying pictures over each other."""
    from matplotlib import colormaps

    w, h = fr.size
    lo, hi = look.clim
    hi = hi if hi > lo else lo + max(abs(lo), 1.0) * 1e-9
    cmap = colormaps[look.cmap] if isinstance(look.cmap, str) else look.cmap
    lut = cmap(np.linspace(0, 1, 256))[:, :3].astype(np.float32)

    def color(a):
        val = a[:, 0] if look.transform is None else look.transform(a[:, 0])
        missing = ~np.isfinite(val)
        rgb = lut[np.clip((np.where(missing, lo, val) - lo) * (255.0 / (hi - lo)) + 0.5, 0, 255).astype(np.int64)]
        if look.above:
            rgb[val > hi + 1e-9 * abs(hi)] = _rgb(look.above)
        rgb[missing] = _rgb(look.nan)
        light = look.ambient + look.diffuse * a[:, 1]
        return np.clip(rgb * light[:, None] + (look.specular * a[:, 2])[:, None], 0, 1)

    bg = _rgb(BG).astype(np.float32)
    out = np.empty((h * w, 3), np.float32)
    out[:] = bg
    out[fr.one] = color(fr.one_attr)
    samples = np.empty(fr.hit.shape + (3,), np.float32)
    samples[:] = bg
    samples[fr.hit] = color(fr.many_attr)
    if fr.lines is not None:
        a = fr.lines.alpha
        samples[fr.on_line] = samples[fr.on_line] * (1 - a) + _rgb(fr.lines.color).astype(np.float32) * a
    out[fr.many] = samples.mean(1)
    out = out.reshape(h, w, 3)
    if not coverage:
        return out
    cover = np.zeros(h * w, np.float32)
    cover[fr.one] = 1.0
    cover[fr.many] = (fr.hit | fr.on_line).mean(1)
    return out, cover.reshape(h, w)


def draw(size: tuple[int, int], cam: Camera, surf: Surface, values, look: Look, lines: Lines | None = None,
         two_sided: bool = False, coverage: bool = False):
    """The picture of a surface colored by a field at its points: frame() then paint()."""
    return paint(frame(size, cam, surf, values, lines, two_sided, look.power), look, coverage)


def _line_samples(zbuf, zw: int, cam: Camera, lines: Lines, W: int, H: int) -> np.ndarray:
    """The samples line segments cover, tested against the depth of the surface at the pixel holding each sample
    (zbuf, zw pixels wide) unless lines.depth_test is off."""
    xa, ya, za = cam.project(lines.segments[:, 0], SS)
    xb, yb, zb = cam.project(lines.segments[:, 1], SS)
    n = np.ceil(np.hypot(xb - xa, yb - ya)).astype(np.int64) + 2
    seg = np.repeat(np.arange(len(n)), n)
    t = (np.arange(n.sum()) - np.repeat(np.cumsum(n) - n, n)) / np.repeat(n - 1, n)
    x, y, z = (a[seg] + t * (b - a)[seg] for a, b in ((xa, xb), (ya, yb), (za, zb)))
    r = 0.5 * lines.width * SS
    hits = []
    for dx in np.arange(-r + 0.5, r, 1.0):
        for dy in np.arange(-r + 0.5, r, 1.0):
            px, py = np.floor(x + dx).astype(np.int64), np.floor(y + dy).astype(np.int64)
            ok = (px >= 0) & (px < W) & (py >= 0) & (py < H)
            px, py, zk = px[ok], py[ok], z[ok]
            if lines.depth_test:
                front = zk >= zbuf[(py // SS) * zw + px // SS] * (1 - 4e-3)
                px, py = px[front], py[front]
            hits.append(py * W + px)
    return np.unique(np.concatenate(hits))


# ------------------------------------------------------------- annotation
@dataclass
class Mark:
    xyz: np.ndarray
    label: str = ""
    color: str = MARK
    small: bool = False


@dataclass
class Panel:
    """A drawn view with what matplotlib lays over it: a color bar (cmap, clim, title), marks and a caption."""
    rgb: np.ndarray
    cam: Camera
    bar: tuple | None = None
    marks: list[Mark] = field(default_factory=list)
    text: str = ""
    text_size: float = 15


FONTS = Path(__file__).with_name("fonts")
# the glyphs the report's subset fonts carry (see scripts/subset_report_fonts.py); other text falls back
SUBSET = set(chr(c) for c in range(0x20, 0x7F)) | set("\u00a0°±²³µ·×\u2019\u201c\u201d\u2026")
PT = 100 / 72                         # pixels per point, the scale the pictures' type was set at


@lru_cache(maxsize=None)
def _font(px: int, bold: bool, plain: bool = False):
    """IBM Plex Mono at a pixel size, or DejaVu Sans Mono from matplotlib when this Pillow cannot read woff2 or
    the text needs glyphs outside the subset."""
    from PIL import ImageFont

    if not plain:
        try:
            return ImageFont.truetype(str(FONTS / ("plex-mono-600.woff2" if bold else "plex-mono-400.woff2")), px)
        except OSError:
            pass
    import matplotlib

    name = "DejaVuSansMono-Bold.ttf" if bold else "DejaVuSansMono.ttf"
    return ImageFont.truetype(str(Path(matplotlib.get_data_path()) / "fonts" / "ttf" / name), px)


def _text_font(text: str, pt: float, bold: bool = False):
    return _font(round(pt * PT), bold, not set(text) <= SUBSET)


def _smooth(size: tuple[int, int], paint_fn, scale: int = 4):
    """A small RGBA patch drawn scale times larger and scaled down, so its curves come out smooth."""
    from PIL import Image, ImageDraw

    big = Image.new("RGBA", (size[0] * scale, size[1] * scale), (0, 0, 0, 0))
    paint_fn(ImageDraw.Draw(big), scale)
    return big.resize(size, Image.LANCZOS)


def _rgba(color: str, alpha: float = 1.0) -> tuple[int, int, int, int]:
    r, g, b = (_rgb(color) * 255 + 0.5).astype(int)
    return int(r), int(g), int(b), int(alpha * 255 + 0.5)


def _mark(img, x: float, y: float, m: Mark, W: int, x0: int):
    """A marked point, a disc ringed in near black, with its label in a rounded box above and to one side."""
    from PIL import ImageDraw

    r, ring = (5.4 if m.small else 7.3), 3.0
    R_ = r + ring / 2
    pad = int(R_ + 2)
    disc = _smooth((2 * pad, 2 * pad), lambda d, k: d.ellipse(
        [(pad - R_) * k, (pad - R_) * k, (pad + R_) * k, (pad + R_) * k], fill=_rgba(m.color),
        outline=_rgba("#111111"), width=round(ring * k)))
    img.alpha_composite(disc, (round(x0 + x) - pad, round(y) - pad))
    if not m.label:
        return
    font = _text_font(m.label, 10 if m.small else 13, bold=True)
    l, t, rr, bb = font.getbbox(m.label)
    padx = round(0.3 * font.size)
    bw, bh = rr - l + 2 * padx, font.size + 2 * padx
    off = round(9 * PT)
    left = x > 0.72 * W
    bx = round(x0 + x - off - bw) if left else round(x0 + x + off)
    by = round(y - off - bh)
    box = _smooth((bw, bh), lambda d, k: d.rounded_rectangle([0, 0, bw * k - 1, bh * k - 1], radius=padx * k,
                                                            fill=_rgba(m.color, 0.95)))
    img.alpha_composite(box, (bx, by))
    ImageDraw.Draw(img).text((bx + padx - l, by + bh / 2), m.label, font=font, fill=_rgba("#111111"), anchor="lm")


def _bar(img, bar, W: int, H: int, x0: int):
    """The color bar: the scale with six values beside it, the swatch for values over it, and its title."""
    from matplotlib import colormaps
    from PIL import Image, ImageDraw

    cmap, (lo, hi), title = bar[:3]
    above = bar[3] if len(bar) > 3 else None
    hi = hi if hi > lo else lo + max(abs(lo), 1.0) * 1e-9
    left, top, width, height = x0 + round(0.885 * W), round(0.2 * H), round(0.016 * W), round(0.675 * H)
    cm = colormaps[cmap] if isinstance(cmap, str) else cmap
    ramp = (cm(np.linspace(1, 0, height))[:, :3] * 255 + 0.5).astype(np.uint8)
    img.paste(Image.fromarray(np.repeat(ramp[:, None, :], width, axis=1), "RGB"), (left, top))
    d = ImageDraw.Draw(img)
    for t in np.linspace(lo, hi, 6):
        label = f"{t:.3g}"
        y = top + (hi - t) / (hi - lo) * (height - 1)
        d.text((left + width + round(5 * PT), y), label, font=_text_font(label, 10), fill=_rgba(FG), anchor="lm")
    head = top
    if above:   # the color of values over the scale, as VTK's bar showed it
        sh = round(0.022 * H)
        sy = top - round(0.008 * H) - sh
        d.rectangle([left, sy, left + width - 1, sy + sh - 1], fill=_rgba(above))
        d.text((left - round(0.004 * W), sy + sh / 2), "above", font=_text_font("above", 10), fill=_rgba(FG),
               anchor="rm")
        head -= round(0.03 * H)
    rows = title.split("\n")
    step = round(11.5 * PT * 1.1)
    for i, row in enumerate(reversed(rows)):
        d.text((x0 + 0.893 * W, head - round(0.015 * H) - i * step), row, font=_text_font(row, 11.5),
               fill=_rgba(FG), anchor="md")


_FONT_LOCK = threading.Lock()     # a FreeType face is not safe to draw with from two threads at once


def compose(path: str | Path, panels: list[Panel]) -> str:
    """Lay the panels side by side, add their bars, marks and captions, and save a PNG."""
    from PIL import Image, ImageDraw

    Wt, Ht = sum(p.rgb.shape[1] for p in panels), max(p.rgb.shape[0] for p in panels)
    base = np.empty((Ht, Wt, 3), np.uint8)
    base[:] = (_rgb(BG) * 255 + 0.5).astype(np.uint8)
    x0 = 0
    for p in panels:
        H, W = p.rgb.shape[:2]
        base[:H, x0:x0 + W] = (np.clip(p.rgb, 0, 1) * 255 + 0.5).astype(np.uint8)
        x0 += W
    img = Image.fromarray(base, "RGB").convert("RGBA")
    with _FONT_LOCK:
        _overlays(img, panels)
    img.convert("RGB").save(path, compress_level=6)
    return str(path)


def _overlays(img, panels: list[Panel]):
    from PIL import ImageDraw

    x0 = 0
    for p in panels:
        H, W = p.rgb.shape[:2]
        for m in p.marks:
            px, py, _ = p.cam.project(m.xyz)
            if np.isfinite(px[0]) and np.isfinite(py[0]):
                _mark(img, float(px[0]), float(py[0]), m, W, x0)
        if p.text:
            ImageDraw.Draw(img).text((x0 + 8, 8), p.text, font=_text_font(p.text, p.text_size), fill=_rgba(FG),
                                     anchor="la")
        if p.bar:
            _bar(img, p.bar, W, H, x0)
        x0 += W


# ----------------------------------------------------------------- pictures
def render_structural(out_dir: str | Path, X, tets, tris, measure, U, governing_xyz, governing_value,
                      measure_label: str, allowable: float, target_sf: float, spots: list[dict],
                      unit: str = "MPa", scale_note: str = "") -> dict[str, str]:
    """Write stress.png, zoom.png, safety.png and deformed.png; returns name -> path."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    try:
        return _render(out, np.asarray(X, float), np.asarray(tris), np.asarray(measure, float),
                       np.asarray(U, float), np.asarray(governing_xyz, float), float(governing_value), measure_label,
                       allowable, target_sf, spots, unit)
    except Exception as e:  # noqa: BLE001  a picture is never a reason to lose a finished study
        if os.environ.get("WOBBL_DEBUG"):
            raise
        return {"error": f"rendering failed: {str(e)[:200]}"}


def _render(out, X, tris, measure, U, g, gval, mlabel, allowable, target_sf, spots, unit):
    view = _view_dir(X, g, _surface_normal_at(X, tris, g))
    span = float(np.linalg.norm(X.max(0) - X.min(0)))
    rim = X[np.unique(tris)]
    normals = _node_normals(X, tris)
    artifacts = [s for s in spots if s.get("excluded") and s.get("values") and s["values"][-1] > gval]
    stress = Look("turbo", (0.0, gval if gval > 0 else float(measure.max())), "#ffffff" if artifacts else None,
                  specular=0.15)
    bar = (stress.cmap, stress.clim, f"{mlabel}\n{unit}", stress.above)

    def marks(mspan):
        out_ = [Mark(g, f"{gval:.3g} {unit}")]
        labeled = [g]
        for s in artifacts[:3]:
            xyz = np.asarray(s["xyz"], float)
            # a label close to one already placed would cover it, so only the first of a cluster is named
            near = min(np.linalg.norm(xyz - q) for q in labeled) < 0.06 * mspan
            out_.append(Mark(xyz, "" if near else "artifact", ARTIFACT, small=True))
            if not near:
                labeled.append(xyz)
        return out_

    big, small = (1600, 1000), (1200, 900)
    top = max(2.0 * target_sf, 3.0)
    safety = Look("RdYlGn", (0.0, top),
                  transform=lambda s: np.where(s > 0, allowable / np.maximum(s, 1e-30), 99.0))

    def overview():
        cam = camera(view, rim, big, span=span)
        surf = fitted(cam, big, X, tris, {"stress": measure}, normals)
        # the stress and safety pictures share the view, so the part is rasterized once and colored twice
        fr = frame(big, cam, surf, surf.fields["stress"], power=stress.power)
        sf = [Mark(g, f"SF {allowable / gval:.2f}" if gval > 0 else "")]
        return {"stress": compose(out / "stress.png", [Panel(paint(fr, stress), cam, bar, marks(span))]),
                "safety": compose(out / "safety.png", [Panel(paint(fr, safety), cam, (safety.cmap, safety.clim,
                                                                                      "safety factor"), sf)])}

    def zoom():
        cam = camera(view, rim, small, focus=g, zoom=3.2, span=span)
        surf = fitted(cam, small, X, tris, {"stress": measure}, normals)
        rgb = draw(small, cam, surf, surf.fields["stress"], stress)
        return {"zoom": compose(out / "zoom.png", [Panel(rgb, cam, bar, marks(span / (3.2 / 1.15)))])}

    def deformed():
        umag = np.linalg.norm(U, axis=1)
        factor = 0.08 * span / umag.max() if umag.max() > 0 else 1.0
        Xw = X + factor * U
        cam = camera(view, np.vstack([Xw[np.unique(tris)], rim]), big,
                     box=(OVERVIEW[0], 0.08, OVERVIEW[2], OVERVIEW[3]), span=span)
        warped = fitted(cam, big, Xw, tris, {"umag": umag})
        look = Look("viridis", (float(umag.min()), float(umag.max())))
        rgb = draw(big, cam, warped, warped.fields["umag"], look, Lines(feature_edges(X, tris)))
        return {"deformed": compose(out / "deformed.png", [Panel(rgb, cam, (look.cmap, look.clim, "deflection\nmm"),
                                                                 text=f"shape exaggerated {factor:.3g}x")])}

    done = _in_parallel([overview, zoom, deformed])
    return {k: done[k] for k in ("stress", "zoom", "safety", "deformed")}


def _in_parallel(jobs: list) -> dict:
    """Run picture jobs side by side in threads, numpy and Pillow letting go of the interpreter lock while they
    work, and merge what they return. WOBBL_RENDER_THREADS=1 runs them one after another."""
    threads = int(os.environ.get("WOBBL_RENDER_THREADS", "0") or 0) or min(len(jobs), os.cpu_count() or 1)
    out: dict = {}
    if threads <= 1:
        for job in jobs:
            out.update(job())
        return out
    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(threads) as pool:
        for got in pool.map(lambda job: job(), jobs):
            out.update(got)
    return out


def render_scalar(path: str | Path, X, tets, tris, values, label: str, unit: str, cmap: str = "inferno",
                  mark_xyz=None, mark_label: str = "", clim=None) -> str:
    """One heatmap of any nodal field (temperature, current density) with an optional marked point."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    X, values, tris = np.asarray(X, float), np.asarray(values, float), np.asarray(tris)
    g = np.asarray(mark_xyz if mark_xyz is not None else X[int(np.argmax(values))], float)
    view = _view_dir(X, g, _surface_normal_at(X, tris, g))
    size = (1600, 1000)
    cam = camera(view, X[np.unique(tris)], size)
    surf = fitted(cam, size, X, tris, {"v": values})
    look = Look(cmap, tuple(clim) if clim is not None else (float(values.min()), float(values.max())))
    return compose(path, [Panel(draw(size, cam, surf, surf.fields["v"], look), cam, (cmap, look.clim,
                                                                                   f"{label}\n{unit}"),
                                [Mark(g, mark_label)] if mark_label else [])])


def line_plot(path: str | Path, x, ys: dict, xlabel: str, ylabel: str, hline: float | None = None,
              hlabel: str = "") -> str:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=150, facecolor=BG)
    ax.set_facecolor(BG)
    colors = [MARK, "#4a90d9", "#3ccf7a", "#ff6b6b"]
    for (name, y), col in zip(ys.items(), colors):
        ax.plot(x, y, color=col, lw=2, label=name)
    if hline is not None:
        ax.axhline(hline, color="#ff4d4f", ls="--", lw=1.2, label=hlabel)
    ax.set_xlabel(xlabel, color=FG)
    ax.set_ylabel(ylabel, color=FG)
    ax.tick_params(colors=FG)
    for sp in ax.spines.values():
        sp.set_color("#3a3d42")
    ax.grid(alpha=0.15)
    ax.legend(facecolor=BG, edgecolor="#3a3d42", labelcolor=FG)
    fig.tight_layout()
    fig.savefig(path, facecolor=BG)
    plt.close(fig)
    return str(path)




def convergence_plot(path: str | Path, levels: list[dict], spots: list[dict], unit: str = "MPa",
                     label: str = "governing stress") -> str:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=150, facecolor=BG)
    ax.set_facecolor(BG)
    n = [L["n_nodes"] for L in levels]
    g = [L["governing"] for L in levels]
    for s in spots:
        vals = s.get("values", [])
        if len(vals) == len(n):
            style = dict(color=ARTIFACT, lw=1, ls="--") if s.get("excluded") else dict(color="#4a90d9", lw=1)
            ax.plot(n, vals, marker="o", ms=3, alpha=0.7, **style)
    ax.plot(n, g, marker="o", color=MARK, lw=2.5, label=label)
    ax.set_xscale("log")
    ax.set_xlabel("mesh nodes", color=FG)
    ax.set_ylabel(unit, color=FG)
    ax.tick_params(colors=FG)
    for sp in ax.spines.values():
        sp.set_color("#3a3d42")
    ax.grid(alpha=0.15)
    ax.legend(facecolor=BG, edgecolor="#3a3d42", labelcolor=FG)
    ax.set_title("mesh convergence: yellow is the reported value, dashed are excluded artifacts", color=FG,
                 fontsize=9)
    fig.tight_layout()
    fig.savefig(path, facecolor=BG)
    plt.close(fig)
    return str(path)
