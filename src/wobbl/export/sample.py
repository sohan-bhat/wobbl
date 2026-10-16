"""A run's scene.wobbl read back and posed at any playback position, as plain triangles, lines and dots.

This follows what the browser viewer does with the same file (wobbl.viz.viewer): every object at a fractional
frame position, deformed, posed, colored by its field and lit by its glow, so a still or a clip of the run shows
what the 3D view shows. Nothing here draws; render.py turns a posed frame into pixels.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from ..viz.scene import array, load

OVER = np.array([242, 242, 238]) / 255.0
NONE = np.array([85, 88, 94]) / 255.0
GLOW = np.array([1.0, 0.85, 0.2])
STOPS = {"viridis": "440154 482878 3e4989 31688e 26828e 1f9e89 35b779 6ece58 b5de2b fde725",
         "inferno": "000004 1b0c41 4a0c6b 781c6d a52c60 cf4446 ed6925 fb9b06 f7d13d fcffa4",
         "coolwarm": "3b4cc0 6889ee 9abbff c9d8ef edd1c2 f7a889 e26952 b40426",
         "blues": "0b1a33 0f3b6e 1663a6 2f8fd1 6bb8e8 b6e0f7", "gray": "303338 e6e6e6", "heat": "9a9488 c9a05a e8742a f2401f",
         "RdBu": "053061 2166ac 4393c3 92c5de d1e5f0 f7f7f7 fddbc7 f4a582 d6604d b2182b 67001f"}
CPK = {"H": "f2f2f2", "C": "555a60", "N": "3b6fe0", "O": "e8423b", "F": "7fd17f", "P": "ff8c1a", "S": "f2d230",
       "Cl": "33d14a", "Na": "a35cf2", "K": "8f40d4", "Li": "cc80ff", "Mg": "8aff00", "Al": "bfa6a6", "Si": "d9b380",
       "Fe": "e06633", "Cu": "c88033", "Zn": "7d80b0", "Ti": "bfc2c7", "Ni": "50d050", "Ca": "3dff00", "B": "ffb5b5",
       "Ar": "80d1e3", "He": "d9ffff"}
RAD = {"H": 0.31, "C": 0.76, "N": 0.71, "O": 0.66, "F": 0.57, "S": 1.05, "P": 1.07, "Cl": 1.02, "Na": 1.66, "K": 2.03,
       "Li": 1.28, "Fe": 1.32, "Al": 1.21, "Si": 1.11, "Cu": 1.32, "Ti": 1.6}


def rgb(h: str) -> np.ndarray:
    h = h.lstrip("#")
    return np.array([int(h[i:i + 2], 16) for i in (0, 2, 4)], float) / 255.0


def lut(name: str) -> np.ndarray:
    """The viewer's 256 color table for a colormap name."""
    x = np.linspace(0, 1, 256)
    if name in STOPS:
        stops = np.array([rgb(s) for s in STOPS[name].split()])
        at = np.linspace(0, 1, len(stops))
        return np.stack([np.interp(x, at, stops[:, k]) for k in range(3)], 1)
    r = 0.13572138 + x * (4.6153926 + x * (-42.66032258 + x * (132.13108234 + x * (-152.94239396 + x * 59.28637943))))
    g = 0.09140261 + x * (2.19418839 + x * (4.84296658 + x * (-14.18503333 + x * (4.27729857 + x * 2.82956604))))
    b = 0.1066733 + x * (12.64194608 + x * (-60.58204836 + x * (110.36276771 + x * (-89.90310912 + x * 27.34824973))))
    return np.clip(np.stack([r, g, b], 1), 0, 1)


@dataclass
class Surface:
    """Triangles in world space with a color per vertex. flat surfaces own their vertices, one triangle each."""
    P: np.ndarray
    T: np.ndarray
    C: np.ndarray
    alpha: float = 1.0
    glow: float = 0.0
    gcol: np.ndarray = field(default_factory=lambda: GLOW.copy())
    ground: bool = False
    grid: float = 0.0
    flat: bool = False
    plane: tuple = ()            # a ground: (up axis, level, center along the other two, half size)
    prop: bool = False


@dataclass
class Strokes:
    """Polylines and dots in world space, drawn over the surfaces where nothing hides them."""
    paths: list[np.ndarray]
    colors: list[np.ndarray]           # one (points, 3) array or one color per path
    width: float = 2.0
    alpha: float = 1.0
    dots: bool = False


@dataclass
class Posed:
    surfaces: list[Surface]
    strokes: list[Strokes]
    t: float


class Field:
    def __init__(self, f: dict):
        self.lo, self.hi = (f.get("range") or [0.0, 1.0])[:2]
        self.limit = f.get("limit")
        self.lut = lut(f.get("colormap") or "turbo")

    def colors(self, v) -> np.ndarray:
        v = np.asarray(v, float)
        i = np.clip(np.rint((np.nan_to_num(v, nan=self.lo) - self.lo) / ((self.hi - self.lo) or 1.0) * 255), 0,
                    255).astype(int)
        c = self.lut[i]
        if self.limit is not None:
            c[v > self.limit] = OVER
        c[~np.isfinite(v)] = NONE
        return c


def quat_matrix(q) -> np.ndarray:
    w, x, y, z = np.asarray(q, float) / (np.linalg.norm(q) or 1.0)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def z_to(d) -> np.ndarray:
    """The rotation taking +z onto direction d."""
    d = np.asarray(d, float)
    n = np.linalg.norm(d)
    if n < 1e-12:
        return np.eye(3)
    d = d / n
    v = np.cross([0.0, 0.0, 1.0], d)
    c = d[2]
    if c < -0.999999:
        return np.diag([1.0, -1.0, -1.0])
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx / (1 + c)


# ---------------------------------------------------------------- primitive shapes in their own frame
def _lathe(profile, n: int, closed: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """A body of revolution about z from (radius, height) points, n segments around. Long profile edges are cut so
    no triangle is much longer than it is wide, which keeps the rasterizer's edge passes local."""
    prof = np.asarray(profile, float)
    step = 2 * np.pi * max(float(np.abs(prof[:, 0]).max()), 1e-12) / n
    pts = [prof[0]]
    for p, q in zip(prof[:-1], prof[1:]):
        k = min(24, max(1, int(np.ceil(np.linalg.norm(q - p) / step))))
        pts += [p + (q - p) * (i / k) for i in range(1, k + 1)]
    prof = np.asarray(pts)
    a = np.linspace(0, 2 * np.pi, n + 1)[:-1]
    r, h = prof[:, :1], prof[:, 1:]
    P = np.stack([r * np.cos(a), r * np.sin(a), np.broadcast_to(h, (len(prof), n))], -1).reshape(-1, 3)
    i, j = np.meshgrid(np.arange(len(prof) - 1), np.arange(n), indexing="ij")
    p, q = (i * n + j).ravel(), (i * n + (j + 1) % n).ravel()
    T = np.stack([np.stack([p, q, q + n], 1), np.stack([p, q + n, p + n], 1)], 1).reshape(-1, 3)
    return P, T


def _grid_face(c, du, dv, m: int) -> tuple[np.ndarray, np.ndarray]:
    """A flat face from corner c along du and dv, cut m x m."""
    g = np.linspace(0, 1, m + 1)
    P = np.array([c + du * a + dv * b for b in g for a in g])
    T = []
    for j in range(m):
        for i in range(m):
            p = j * (m + 1) + i
            T += [(p, p + 1, p + m + 2), (p, p + m + 2, p + m + 1)]
    return P, np.asarray(T)


def primitive(shape: str, s: list[float]) -> tuple[np.ndarray, np.ndarray]:
    H = np.pi / 2
    if shape in ("sphere", "ellipsoid"):
        k = [s[0]] * 3 if shape == "sphere" else s[:3]
        ang = np.linspace(-H, H, 15)
        P, T = _lathe([(np.cos(t), np.sin(t)) for t in ang], 28)
        return P * np.asarray(k, float), T
    if shape == "capsule":
        r, h = s[0], s[1]
        ang = np.linspace(-H, H, 15)
        prof = [(r * np.cos(t), (-h if t < 0 else h) + r * np.sin(t)) for t in ang]
        return _lathe(prof, 24)
    if shape in ("cylinder", "hex"):
        r, h = s[0], s[1]
        return _lathe([(0, -h), (r, -h), (r, -h), (r, h), (r, h), (0, h)], 32 if shape == "cylinder" else 6)
    if shape == "cone":
        r, h = s[0], s[1]
        return _lathe([(0, -h), (r, -h), (r, -h), (0, h)], 28)
    if shape == "torus":
        R, r = s[0], s[1]
        ang = np.linspace(0, 2 * np.pi, 17)
        return _lathe([(R + r * np.cos(t), r * np.sin(t)) for t in ang], 32)
    if shape == "plane":
        a, b = s[0], s[1]
        return _grid_face(np.array([-a, -b, 0.0]), np.array([2 * a, 0, 0.0]), np.array([0, 2 * b, 0.0]), 8)
    hx, hy, hz = (list(s) + [s[-1]] * 3)[:3]
    e = np.array([hx, hy, hz], float)
    Ps, Ts, n = [], [], 0
    for ax in range(3):
        for sg in (-1.0, 1.0):
            u, v = (ax + 1) % 3, (ax + 2) % 3
            c = np.zeros(3)
            c[ax], c[u], c[v] = sg * e[ax], -e[u], -e[v]
            du, dv = np.zeros(3), np.zeros(3)
            du[u], dv[v] = 2 * e[u], 2 * e[v]
            P, T = _grid_face(c, du, dv, 6)
            Ps.append(P)
            Ts.append(T + n)
            n += len(P)
    return np.concatenate(Ps), np.concatenate(Ts)


def tube(a, b, r: float, n: int = 10) -> tuple[np.ndarray, np.ndarray]:
    d = np.asarray(b, float) - np.asarray(a, float)
    L = float(np.linalg.norm(d))
    P, T = _lathe([(r, 0), (r, L)], n)
    return P @ z_to(d).T + np.asarray(a, float), T


def arrow(o, v, r: float) -> tuple[np.ndarray, np.ndarray]:
    L = float(np.linalg.norm(v))
    if L < 1e-12:
        return np.zeros((0, 3)), np.zeros((0, 3), int)
    hl, hr = min(L * 0.45, r * 7), r * 2.6
    P, T = _lathe([(0, 0), (r, 0), (r, L - hl), (hr, L - hl), (0, L)], 12)
    return P @ z_to(v).T + np.asarray(o, float), T


def flatten(P, T) -> tuple[np.ndarray, np.ndarray]:
    """Every triangle with its own three vertices, so it shades flat."""
    T = np.asarray(T, int)
    return np.asarray(P, float)[T.ravel()], np.arange(T.size).reshape(-1, 3)


# ---------------------------------------------------------------- the scene
class Scene:
    """A scene file read once and posed at any frame position u in 0 .. frames - 1."""

    def __init__(self, path):
        self.h, self.blob = load(path)
        self.tl = self.h.get("timeline", {})
        self.times = np.asarray(self.tl.get("t") or [0.0], float)
        self.n = int(self.tl.get("frames") or len(self.times))
        self.fields = [Field(f) for f in self.h.get("fields", [])]
        self._cache: dict = {}
        self.up = np.array([0.0, 1.0, 0.0]) if self.h.get("up") == "y" else np.array([0.0, 0.0, 1.0])

    def arr(self, d):
        if isinstance(d, dict) and "b" in d:
            key = id(d)
            if key not in self._cache:
                self._cache[key] = array(self.h, self.blob, d)
            return self._cache[key]
        return np.asarray(d, float)

    @property
    def step(self) -> bool:
        return self.tl.get("between") == "step"

    def time(self, u: float) -> float:
        a, b, k = self._ab(u)
        return float(self.times[min(a, len(self.times) - 1)] * (1 - k) + self.times[min(b, len(self.times) - 1)] * k)

    def _ab(self, u: float) -> tuple[int, int, float]:
        u = min(max(float(u), 0.0), self.n - 1.0)
        a = int(math.floor(u))
        b = min(a + 1, self.n - 1)
        k = 0.0 if self.step else u - a
        return a, b, k

    def series(self, s, a, b, k) -> float:
        s = list(s)
        if len(s) == 1:
            return float(s[0])
        return float(s[a] * (1 - k) + s[b] * k)

    def at(self, arr, a, b, k):
        """A per frame array (frames x ...) blended between two frames."""
        arr = np.asarray(arr)
        return arr[a] * (1 - k) + arr[b] * k if k else arr[a]

    def pose(self, o, a, b, k) -> tuple[np.ndarray, np.ndarray]:
        p = o.get("pose") or ({"pos": o.get("pos"), "quat": o.get("quat")} if o.get("kind") == "body" else None)
        if not p or p.get("pos") is None:
            return np.eye(3), np.zeros(3)
        pos, q = self.arr(p["pos"]), self.arr(p["quat"])
        pos = self.at(pos, a, b, k) if pos.ndim == 2 else pos
        q = self.at(q, a, b, k) if q.ndim == 2 else q
        return quat_matrix(q), np.asarray(pos, float)

    def readouts(self, u: float) -> list[tuple[str, float, str, int]]:
        a, b, k = self._ab(u)
        return [(r["name"], self.series(r["values"], a, b, k), r.get("unit", ""), int(r.get("digits", 3)))
                for r in self.h.get("readouts", [])]

    def posed(self, u: float) -> Posed:
        a, b, k = self._ab(u)
        t = self.time(u)
        surfaces, strokes = [], []
        grow = {i: (g.get("factor") or 1.0, g.get("anchor")) for g in (self.h.get("magnify") or {}).get("groups", [])
                for i in g.get("objects", [])}
        for i, o in enumerate(self.h.get("objects", [])):
            w = o.get("when")
            if o.get("guide") or (w and ((w[0] is not None and t < w[0] - 1e-9) or
                                         (w[1] is not None and t > w[1] + 1e-9))):
                continue     # guides only carry labels, which a picture leaves out
            n0 = len(surfaces)
            kind = o.get("kind")
            glow = self.series(o["glow"], a, b, k) if o.get("glow") else 0.0
            if kind == "mesh":
                surfaces += self._mesh(o, a, b, k, glow)
            elif kind == "body":
                surfaces += self._body(o, a, b, k, glow)
            elif kind == "ground":
                surfaces.append(self._ground(o))
            elif kind == "arrows":
                surfaces += self._arrows(o, a, b, k)
            elif kind in ("atoms", "points"):
                s, st = self._points(o, a, b, k)
                surfaces += s
                strokes += st
            elif kind == "lines":
                strokes += self._lines(o, a, b, k, t)
            if i in grow:   # small things drawn larger about their own place, as the viewer draws them
                f, anchor = grow[i]
                c = np.asarray(anchor, float) if anchor is not None else self.pose(o, a, b, k)[1]
                for s in surfaces[n0:]:
                    s.P = c + (s.P - c) * f
        return Posed(surfaces, strokes, t)

    def _mesh(self, o, a, b, k, glow) -> list[Surface]:
        P = self.arr(o["pos"]).reshape(-1, 3)
        T = self.arr(o["tri"]).astype(int).reshape(-1, 3)
        n = len(P)
        if o.get("deform"):
            d = o["deform"]
            ex = float(d.get("exaggerate") or 1.0)
            if "scale" in d:
                P = P + self.arr(d["values"]).reshape(-1, 3) * self.series(d["scale"], a, b, k) * ex
            else:
                P = P + self.at(self.arr(d["frames"]).reshape(-1, n, 3), a, b, k) * ex
        base = rgb(o.get("color") or "#b9bdc4")
        C = np.tile(base, (n, 1))
        if o.get("field"):
            f = o["field"]
            fl = self.fields[f["f"]]
            if "frames" in f:
                v = self.at(self.arr(f["frames"]).reshape(-1, n), a, b, k)
            elif "scale" in f:
                v = self.arr(f["values"]).ravel() * self.series(f["scale"], a, b, k)
            else:
                v = self.arr(f["values"]).ravel()
            C = fl.colors(v)
        if o.get("reveal"):
            rv = self.arr(o["reveal"]["values"]).ravel()
            th = self.series(o["reveal"]["at"], a, b, k)
            T = T[(rv[T] <= th + 1e-9).all(1)]
        R, p = self.pose(o, a, b, k)
        P = P @ R.T + p
        flat = bool(o.get("flat"))
        if flat:
            idx = T.ravel()
            P, C, T = P[idx], C[idx], np.arange(idx.size).reshape(-1, 3)
        return [Surface(P, T, C, float(o.get("opacity", 1.0)), glow, GLOW.copy(), flat=flat, prop=bool(o.get("prop")))]

    def _body(self, o, a, b, k, glow) -> list[Surface]:
        P, T = primitive(o.get("shape", "box"), o.get("size", [1, 1, 1]))
        R, p = self.pose(o, a, b, k)
        base = rgb(o.get("color") or "#9aa0a6")
        if o.get("field"):
            base = self.fields[o["field"]["f"]].colors([self.series(o["field"]["values"], a, b, k)])[0]
        P, T = flatten(P @ R.T + p, T) if o.get("shape") in ("box", "hex", "plane") else (P @ R.T + p, T)
        gcol = np.minimum(1.0, rgb(o.get("color") or "#9aa0a6") * 0.5 + 0.55) if o.get("glow") else GLOW.copy()
        return [Surface(P, T, np.tile(base, (len(P), 1)), float(o.get("opacity", 1.0)), glow, gcol,
                        prop=bool(o.get("prop")))]

    def _ground(self, o) -> Surface:
        s, c, z = float(o["size"]) / 2, o.get("center", [0, 0]), float(o.get("z", 0.0))
        m = 24
        g = np.linspace(-s, s, m + 1)
        X, Y = np.meshgrid(g + c[0], g + c[1])
        P = np.column_stack([X.ravel(), Y.ravel(), np.full(X.size, z)])
        if self.h.get("up") == "y":
            P = P[:, [0, 2, 1]]
        T = []
        for i in range(m):
            for j in range(m):
                p = i * (m + 1) + j
                T += [(p, p + 1, p + m + 2), (p, p + m + 2, p + m + 1)]
        axis = 1 if self.h.get("up") == "y" else 2
        return Surface(P, np.asarray(T), np.tile(rgb(o.get("color") or "#1a1c20"), (len(P), 1)), ground=True,
                       grid=float(o.get("grid") or 0.0), plane=(axis, z, (float(c[0]), float(c[1])), s))

    def _arrows(self, o, a, b, k) -> list[Surface]:
        O, V = self.arr(o["origin"]), self.arr(o["vec"])
        O = self.at(O, a, b, k) if O.ndim == 3 else O.reshape(-1, 3)
        V = self.at(V, a, b, k) if V.ndim == 3 else V.reshape(-1, 3)
        s = self.series(o["scale"], a, b, k) if o.get("scale") else 1.0
        Ps, Ts, n = [], [], 0
        for og, v in zip(O, V * s):
            r = o.get("radius") or max(float(np.linalg.norm(v)) * 0.035, self.radius() * 0.006)
            P, T = arrow(og, v, r)
            Ps.append(P)
            Ts.append(T + n)
            n += len(P)
        if not n:
            return []
        P = np.concatenate(Ps)
        return [Surface(P, np.concatenate(Ts), np.tile(rgb(o.get("color") or "#ffd400"), (n, 1)))]

    def _points(self, o, a, b, k) -> tuple[list[Surface], list[Strokes]]:
        P = self.arr(o["pos"])
        if o["kind"] == "atoms":
            nat = len(o["el"])
            P = self.at(P.reshape(-1, nat, 3), a, b, k) if P.size > nat * 3 else P.reshape(-1, 3)
            Ps, Ts, Cs, n = [], [], [], 0
            for el, x in zip(o["el"], P):
                r = RAD.get(el, 1.2) * float(o.get("radius", 0.35))
                sp, st = primitive("sphere", [r])
                Ps.append(sp + x)
                Ts.append(st + n)
                Cs.append(np.tile(rgb(CPK.get(el, "c0c4ca")), (len(sp), 1)))
                n += len(sp)
            if o.get("bonds"):
                bonds = self.arr(o["bonds"]).astype(int).reshape(-1, 2)
                for i, j in bonds:
                    tp, tt = tube(P[i], P[j], 0.11 * float(o.get("radius", 0.35)) / 0.35, 8)
                    Ps.append(tp)
                    Ts.append(tt + n)
                    Cs.append(np.tile(np.array([150, 150, 150]) / 255.0, (len(tp), 1)))
                    n += len(tp)
            return ([Surface(np.concatenate(Ps), np.concatenate(Ts), np.concatenate(Cs))] if n else []), []
        shape = o["pos"]["shape"] if isinstance(o["pos"], dict) else np.shape(P)
        npt = shape[-2] if len(shape) > 1 else 1
        P = self.at(P.reshape(-1, npt, 3), a, b, k) if P.size > npt * 3 else P.reshape(-1, 3)
        col = rgb(o.get("color") or "#ffd400")
        C = np.tile(col, (len(P), 1))
        if o.get("field"):
            f = o["field"]
            v = self.arr(f.get("frames") or f.get("values"))
            v = self.at(v.reshape(-1, npt), a, b, k) if "frames" in f else v.ravel()
            C = self.fields[f["f"]].colors(v)
        ok = np.isfinite(P).all(1)
        return [], [Strokes([P[ok]], [C[ok]], width=float(o.get("size", 1.0)), dots=True)]

    def _lines(self, o, a, b, k, t) -> list[Strokes]:
        P = self.arr(o["pos"]).reshape(-1, 3)
        runs = list(o.get("runs") or [len(P)])
        col = rgb(o.get("color") or "#ffd400")
        vals = self.arr(o["field"]["values"]).ravel() if o.get("field") else None
        drawn = self.arr(o["drawn"]).ravel() if o.get("drawn") else None
        at = o.get("at")
        newest = max([v for v in at if v <= t + 1e-9], default=-np.inf) if at else None
        paths, colors, start = [], [], 0
        for ri, r in enumerate(runs):
            seg = slice(start, start + r)
            start += r
            if at is not None and at[ri] > t + 1e-9:
                continue
            p = P[seg]
            c = self.fields[o["field"]["f"]].colors(vals[seg]) if vals is not None else np.tile(col, (len(p), 1))
            if drawn is not None:
                keep = drawn[seg] <= t + 1e-9
                p, c = p[keep], c[keep]
            if at is not None and o.get("fade") and at[ri] < newest:
                c = c * 0.45
            if len(p) >= 2:
                paths.append(p)
                colors.append(c)
        return [Strokes(paths, colors, width=float(o.get("width", 2.0)))] if paths else []

    def bounds(self, u: float = 0.0, props: bool = True) -> tuple[np.ndarray, np.ndarray]:
        """The box around what is drawn at u, the ground left out since it only frames the rest."""
        pts = [s.P for s in self.posed(u).surfaces if not s.ground and len(s.P)]
        pts += [p for st in self.posed(u).strokes for p in st.paths if len(p)]
        if not pts:
            return np.full(3, -1.0), np.full(3, 1.0)
        X = np.concatenate(pts)
        X = X[np.isfinite(X).all(1)]
        return X.min(0), X.max(0)

    def radius(self) -> float:
        if "_r" not in self._cache:
            self._cache["_r"] = 1.0
            P = [self.arr(o["pos"]).reshape(-1, 3) for o in self.h.get("objects", []) if o.get("kind") == "mesh"]
            if P:
                X = np.concatenate(P)
                self._cache["_r"] = float(np.linalg.norm(X.max(0) - X.min(0))) / 2 or 1.0
        return self._cache["_r"]
