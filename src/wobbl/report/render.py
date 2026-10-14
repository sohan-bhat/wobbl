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
from dataclasses import dataclass, field
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


def _node_normals(X: np.ndarray, tris: np.ndarray) -> np.ndarray:
    """Outward unit normal at each node of each boundary face, (faces, nodes, 3).

    A quadratic face takes its normal from its shape functions, so a curved face shades as a curve. The faces
    around a node that meet at less than FEATURE_ANGLE share their area weighted mean and shade smoothly across
    element edges. At a sharp edge the faces on each side form their own group, found by narrowing the mean to the
    faces near it, so neither side's shading leans toward the other and the edge stays crisp.
    """
    P = X[tris]
    flat = np.cross(P[:, 1] - P[:, 0], P[:, 2] - P[:, 0])
    area = np.linalg.norm(flat, axis=1)
    if tris.shape[1] == 6:
        du, dv = _shape_grad(_NODES)
        n = np.cross(np.einsum("pj,kjc->kpc", du, P), np.einsum("pj,kjc->kpc", dv, P))
    else:
        n = np.repeat(flat[:, None], tris.shape[1], axis=1)
    ln = np.linalg.norm(n, axis=2, keepdims=True)
    own = flat[:, None] / np.maximum(area, 1e-300)[:, None, None]
    n = np.where(ln > 1e-9 * area[:, None, None], n / np.maximum(ln, 1e-300), own).reshape(-1, 3)
    ids, wa = tris.ravel(), np.repeat(area, tris.shape[1])
    cos = np.cos(np.radians(FEATURE_ANGLE))

    def mean(sel):
        w = n * (wa * sel)[:, None]
        S = np.column_stack([np.bincount(ids, weights=w[:, c], minlength=len(X)) for c in range(3)])
        return (S / np.maximum(np.linalg.norm(S, axis=1, keepdims=True), 1e-300))[ids]

    m = mean(np.ones(len(ids)))
    for _ in range(2):
        m = mean(np.einsum("ij,ij->i", n, m) > cos)
    first = np.einsum("ij,ij->i", n, m) > cos
    m2 = mean(~first)
    second = ~first & (np.einsum("ij,ij->i", n, m2) > cos)
    return np.where(first[:, None], m, np.where(second[:, None], m2, n)).reshape(len(tris), -1, 3)


@dataclass
class Surface:
    """The boundary as flat triangles: points, triangles, unit normals and nodal fields carried to the points."""
    P: np.ndarray
    T: np.ndarray
    N: np.ndarray
    fields: dict[str, np.ndarray] = field(default_factory=dict)


def surface(X, tris, fields: dict | None = None, n: int | None = None) -> Surface:
    """Split each boundary face n times along each edge, quadratic faces along their curve, and carry nodal fields,
    scalar or vector, to the new points with each face's shape functions."""
    X, tris = np.asarray(X, float), np.asarray(tris)
    quad = tris.shape[1] == 6
    if n is None:
        n = (4 if 16 * len(tris) <= SUBDIVIDE else 2) if quad else 1
    l, sub = _tri_grid(n)
    W = _shape(l) if quad else l
    P = np.einsum("pj,kjc->kpc", W, X[tris]).reshape(-1, 3)
    N = np.einsum("pj,kjc->kpc", W, _node_normals(X, tris)).reshape(-1, 3)
    N /= np.maximum(np.linalg.norm(N, axis=1, keepdims=True), 1e-300)
    out = {}
    for name, f in (fields or {}).items():
        f = np.asarray(f, float)
        out[name] = (np.einsum("pj,kj->kp", W, f[tris]).ravel() if f.ndim == 1 else
                     np.einsum("pj,kjc->kpc", W, f[tris]).reshape(-1, f.shape[1]))
    T = (np.arange(len(tris))[:, None, None] * len(l) + sub[None]).reshape(-1, 3)
    return Surface(P, T, N, out)


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
    tri = np.full(W * H, -1, np.int64)
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
    rows = j1[ok] - j0[ok] + 1
    work = np.cumsum(rows * np.maximum(xhi[ok] - xlo[ok], 1.0))   # about how many pixels each prefix covers
    start = 0
    while start < len(ok):
        stop = max(int(np.searchsorted(work, work[start] - (rows[start] * max(xhi[ok[start]] - xlo[ok[start]], 1.0))
                                       + CHUNK, side="right")), start + 1)
        t = np.repeat(ok[start:stop], rows[start:stop])
        py = j0[t] + (np.arange(len(t)) - np.repeat(np.cumsum(rows[start:stop]) - rows[start:stop],
                                                      rows[start:stop]))
        cy = py + 0.5 - y0[t]
        a = area[t]
        # each corner weight is a straight function of x along the row, w = s x + c; keep the x where all are >= 0
        s1, c1 = (y2 - y0)[t] / a, (-x0[t] * (y2 - y0)[t] - cy * (x2 - x0)[t]) / a
        s2, c2 = -(y1 - y0)[t] / a, (cy * (x1 - x0)[t] + x0[t] * (y1 - y0)[t]) / a
        lo, hi = xlo[t].copy(), xhi[t].copy()
        for s, c in ((s1, c1), (s2, c2), (-s1 - s2, 1.0 - c1 - c2)):
            with np.errstate(divide="ignore", invalid="ignore"):
                edge = (-1e-7 - c) / s
            lo = np.where(s > 1e-12, np.maximum(lo, edge), lo)
            hi = np.where(s < -1e-12, np.minimum(hi, edge), hi)
            hi = np.where((np.abs(s) <= 1e-12) & (c < -1e-7), -1.0, hi)
        i0 = np.maximum(np.ceil(lo - 0.5), 0).astype(np.int64)
        i1 = np.minimum(np.floor(hi - 0.5), W - 1).astype(np.int64)
        n = np.maximum(i1 - i0 + 1, 0)
        r = np.repeat(np.arange(len(t)), n)
        px = i0[r] + (np.arange(len(r)) - np.repeat(np.cumsum(n) - n, n))
        # inverse depth is a straight function of x along the row too; the weights are worked out for winners only
        dz1, dz2 = (z1 - z0)[t], (z2 - z0)[t]
        sz, cz = (dz1 * s1 + dz2 * s2).astype(np.float32), (z0[t] + dz1 * c1 + dz2 * c2).astype(np.float32)
        cx = (px + 0.5).astype(np.float32)
        z = sz[r] * cx + cz[r]
        pix = py[r] * W + px
        np.maximum.at(zbuf, pix, z)
        win = np.nonzero(z >= zbuf[pix])[0]
        pix, rw, cw = pix[win], r[win], px[win] + 0.5
        tri[pix], w1buf[pix], w2buf[pix] = t[rw], s1[rw] * cw + c1[rw], s2[rw] * cw + c2[rw]
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


def _lighting(cam: Camera, P: np.ndarray, N: np.ndarray, look: Look) -> tuple[np.ndarray, np.ndarray]:
    """Diffuse and specular light at each point. The small triangles span a few pixels, so lighting their corners
    and blending across them looks the same as lighting every pixel, at a fraction of the work."""
    eye = cam.eye - P
    eye /= np.linalg.norm(eye, axis=1, keepdims=True)
    diff, spec = np.zeros(len(P)), np.zeros(len(P))
    for k, (lx, ly, lz) in LIGHTS:
        L = lx * cam.right + ly * cam.up - lz * cam.fwd
        L /= np.linalg.norm(L)
        diff += k * np.maximum(N @ L, 0.0)
        if look.specular:
            h = eye + L
            h /= np.maximum(np.linalg.norm(h, axis=1, keepdims=True), 1e-300)
            spec += k * np.maximum(np.einsum("ij,ij->i", N, h), 0.0) ** look.power
    return look.ambient + look.diffuse * diff, look.specular * spec


def draw(size: tuple[int, int], cam: Camera, surf: Surface, values, look: Look, lines: Lines | None = None,
         two_sided: bool = False, coverage: bool = False):
    """The picture of a surface colored by a field at its points, as an (H, W, 3) array of floats in 0..1, and with
    coverage=True also the share of each pixel the surface or lines cover, for laying pictures over each other.

    A closed surface hides its back faces behind its front ones, so only faces toward the camera are drawn. A
    two sided surface, a shell's midsurface, draws both sides and lights each from the side the camera sees.
    """
    from matplotlib import colormaps

    W, H = size[0] * SS, size[1] * SS
    x, y, iz = cam.project(surf.P, SS)
    P, T, N = surf.P, surf.T, surf.N
    if two_sided:
        N = N * np.where(np.einsum("ij,ij->i", N, cam.eye - P) < 0, -1.0, 1.0)[:, None]
    else:
        toward = np.einsum("ij,ij->i", np.cross(P[T[:, 1]] - P[T[:, 0]], P[T[:, 2]] - P[T[:, 0]]), cam.eye - P[T[:, 0]])
        T = T[toward > 0]
    tri, w1, w2, zbuf = _rasterize(x, y, iz, T, W, H)
    pix = np.nonzero(tri >= 0)[0]
    lit, shine = _lighting(cam, P, N, look)
    a = np.column_stack([iz, values, lit, shine]).astype(np.float32)[T][tri[pix]]   # (pixels, corner, 4)
    w1, w2 = w1[pix], w2[pix]
    q = np.column_stack([1.0 - w1 - w2, w1, w2]) * a[:, :, 0]
    q /= q.sum(1, keepdims=True)          # perspective correct weights of the corners
    val, light, spec = np.einsum("ij,ijk->ki", q, a[:, :, 1:])
    if look.transform is not None:
        val = look.transform(val)
    lo, hi = look.clim
    hi = hi if hi > lo else lo + max(abs(lo), 1.0) * 1e-9
    cmap = colormaps[look.cmap] if isinstance(look.cmap, str) else look.cmap
    lut = cmap(np.linspace(0, 1, 256))[:, :3].astype(np.float32)
    missing = ~np.isfinite(val)
    rgb = lut[np.clip((np.where(missing, lo, val) - lo) * (255.0 / (hi - lo)) + 0.5, 0, 255).astype(np.int64)]
    if look.above:
        rgb[val > hi + 1e-9 * abs(hi)] = _rgb(look.above)
    rgb[missing] = _rgb(look.nan)
    rgb = np.clip(rgb * light[:, None] + spec[:, None], 0, 1)
    if lines is not None and len(lines.segments):
        pix, rgb = _draw_lines(pix, rgb, zbuf, cam, lines, W, H)
    # average each SS x SS block down to one pixel, the background filling what the surface leaves
    py, px = np.divmod(pix, W)
    block = (py // SS) * size[0] + px // SS
    n = size[0] * size[1]
    cover = np.bincount(block, minlength=n)
    bg = _rgb(BG)
    out = np.column_stack([np.bincount(block, weights=rgb[:, c], minlength=n) + bg[c] * (SS * SS - cover)
                           for c in range(3)]) / (SS * SS)
    out = out.reshape(size[1], size[0], 3)
    return (out, (cover / (SS * SS)).reshape(size[1], size[0])) if coverage else out


def _draw_lines(pix, rgb, zbuf, cam: Camera, lines: Lines, W: int, H: int):
    """Blend depth tested line segments into the drawn subpixels (sorted indices and colors), each pixel once."""
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
            p = py[ok] * W + px[ok]
            hits.append(p[z[ok] >= zbuf[p] * (1 - 1e-4)] if lines.depth_test else p)
    line = np.unique(np.concatenate(hits))
    color, alpha = _rgb(lines.color), lines.alpha
    at = np.minimum(np.searchsorted(pix, line), max(len(pix) - 1, 0))
    on = (len(pix) > 0) & (pix[at] == line) if len(pix) else np.zeros(len(line), bool)
    rgb[at[on]] = rgb[at[on]] * (1 - alpha) + color * alpha
    over = np.tile(_rgb(BG) * (1 - alpha) + color * alpha, (int((~on).sum()), 1))
    return np.concatenate([pix, line[~on]]), np.concatenate([rgb, over.astype(rgb.dtype)])


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


def compose(path: str | Path, panels: list[Panel]) -> str:
    """Lay the panels side by side, add their bars, marks and captions, and save a PNG."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize
    from PIL import Image

    Wt, Ht = sum(p.rgb.shape[1] for p in panels), max(p.rgb.shape[0] for p in panels)
    fig = plt.figure(figsize=(Wt / 100, Ht / 100), dpi=100, facecolor=BG)
    x0 = 0
    for p in panels:
        H, W = p.rgb.shape[:2]
        fx, fw, fh = x0 / Wt, W / Wt, H / Ht
        fig.figimage((np.clip(p.rgb, 0, 1) * 255 + 0.5).astype(np.uint8), xo=x0, yo=Ht - H, origin="upper",
                     zorder=-1)   # pixel for pixel
        ax = fig.add_axes([fx, 1 - fh, fw, fh])
        ax.set_xlim(0, W)
        ax.set_ylim(H, 0)
        ax.axis("off")
        for m in p.marks:
            px, py, _ = p.cam.project(m.xyz)
            px, py = float(px[0]), float(py[0])
            ax.scatter([px], [py], s=60 if m.small else 110, c=m.color, edgecolors="#111111", linewidths=2.2,
                       zorder=5)
            if m.label:
                left = px > 0.72 * W
                ax.annotate(m.label, (px, py), xytext=(-9 if left else 9, 9), textcoords="offset points",
                            ha="right" if left else "left", va="bottom", color="#111111", family="monospace",
                            fontweight="bold", fontsize=10 if m.small else 13, zorder=6,
                            bbox=dict(boxstyle="round,pad=0.3", fc=m.color, ec="none", alpha=0.95))
        if p.text:
            ax.text(8, 8, p.text, ha="left", va="top", color=FG, family="monospace", fontsize=p.text_size)
        if p.bar:
            cmap, (lo, hi), title = p.bar[:3]
            above = p.bar[3] if len(p.bar) > 3 else None
            hi = hi if hi > lo else lo + max(abs(lo), 1.0) * 1e-9
            left, bottom, width, height = fx + 0.885 * fw, 1 - fh + 0.125 * fh, 0.016 * fw, 0.675 * fh
            cax = fig.add_axes([left, bottom, width, height])
            cb = fig.colorbar(ScalarMappable(Normalize(lo, hi), cmap), cax=cax)
            cb.outline.set_visible(False)
            ticks = np.linspace(lo, hi, 6)
            cb.set_ticks(ticks, labels=[f"{t:.3g}" for t in ticks])
            cax.tick_params(colors=FG, labelsize=10, length=0, pad=5)
            for lb in cax.get_yticklabels():
                lb.set_family("monospace")
            top = bottom + height
            if above:   # the color of values over the scale, as VTK's bar showed it
                sw = fig.add_axes([left, top + 0.008 * fh, width, 0.022 * fh], facecolor=above)
                sw.set_xticks([])
                sw.set_yticks([])
                for sp in sw.spines.values():
                    sp.set_visible(False)
                fig.text(left - 0.004 * fw, top + 0.019 * fh, "above", ha="right", va="center", color=FG,
                         family="monospace", fontsize=10)
                top += 0.03 * fh
            fig.text(fx + 0.893 * fw, top + 0.015 * fh, title, ha="center", va="bottom", color=FG,
                     family="monospace", fontsize=11.5, linespacing=1.1)
        x0 += W
    fig.canvas.draw()
    rgb = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
    plt.close(fig)
    Image.fromarray(rgb).save(path, compress_level=6)
    return str(path)


def _shot(size, view, surf: Surface, values, look: Look, focus=None, zoom: float = 3.2, span=None, box=OVERVIEW,
          lines: Lines | None = None, frame_pts=None):
    cam = camera(view, surf.P if frame_pts is None else frame_pts, size, focus, zoom, box, span)
    return draw(size, cam, surf, values, look, lines), cam


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
    surf = surface(X, tris, {"stress": measure})
    artifacts = [s for s in spots if s.get("excluded") and s.get("values") and s["values"][-1] > gval]
    stress = Look("turbo", (0.0, gval if gval > 0 else float(measure.max())), "#ffffff" if artifacts else None,
                  specular=0.15)
    files = {}
    for name, size, focus in (("stress", (1600, 1000), None), ("zoom", (1200, 900), g)):
        rgb, cam = _shot(size, view, surf, surf.fields["stress"], stress, focus, span=span)
        mspan = span if focus is None else span / (3.2 / 1.15)
        marks = [Mark(g, f"{gval:.3g} {unit}")]
        for s in artifacts[:3]:
            xyz = np.asarray(s["xyz"], float)
            near = np.linalg.norm(xyz - g) < 0.06 * mspan  # keep the governing label readable
            marks.append(Mark(xyz, "" if near else "artifact", ARTIFACT, small=True))
        files[name] = compose(out / f"{name}.png", [Panel(rgb, cam, (stress.cmap, stress.clim, f"{mlabel}\n{unit}",
                                                                     stress.above), marks)])

    top = max(2.0 * target_sf, 3.0)
    safety = Look("RdYlGn", (0.0, top),
                  transform=lambda s: np.where(s > 0, allowable / np.maximum(s, 1e-30), 99.0))
    rgb, cam = _shot((1600, 1000), view, surf, surf.fields["stress"], safety, span=span)
    files["safety"] = compose(out / "safety.png", [Panel(rgb, cam, (safety.cmap, safety.clim, "safety factor"),
                                                         [Mark(g, f"SF {allowable / gval:.2f}" if gval > 0 else "")])])

    umag = np.linalg.norm(U, axis=1)
    factor = 0.08 * span / umag.max() if umag.max() > 0 else 1.0
    warped = surface(X + factor * U, tris, {"umag": umag})
    deflection = Look("viridis", (float(umag.min()), float(umag.max())))
    rgb, cam = _shot((1600, 1000), view, warped, warped.fields["umag"], deflection, span=span,
                     box=(OVERVIEW[0], 0.08, OVERVIEW[2], OVERVIEW[3]), lines=Lines(feature_edges(X, tris)),
                     frame_pts=np.vstack([warped.P, X]))
    files["deformed"] = compose(out / "deformed.png", [Panel(rgb, cam, (deflection.cmap, deflection.clim,
                                                                        "deflection\nmm"),
                                                             text=f"shape exaggerated {factor:.3g}x")])
    return files


def render_scalar(path: str | Path, X, tets, tris, values, label: str, unit: str, cmap: str = "inferno",
                  mark_xyz=None, mark_label: str = "", clim=None) -> str:
    """One heatmap of any nodal field (temperature, current density) with an optional marked point."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    X, values = np.asarray(X, float), np.asarray(values, float)
    g = np.asarray(mark_xyz if mark_xyz is not None else X[int(np.argmax(values))], float)
    view = _view_dir(X, g, _surface_normal_at(X, tris, g))
    surf = surface(X, tris, {"v": values})
    look = Look(cmap, tuple(clim) if clim is not None else (float(values.min()), float(values.max())))
    rgb, cam = _shot((1600, 1000), view, surf, surf.fields["v"], look)
    return compose(path, [Panel(rgb, cam, (cmap, look.clim, f"{label}\n{unit}"),
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
