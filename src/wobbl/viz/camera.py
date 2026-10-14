"""Where the view starts: a direction from which the point that matters, a stress peak or a hot spot, is in plain
sight, not hidden behind a wall, a shelf or the part itself."""
from __future__ import annotations

import numpy as np


def _rot(q) -> np.ndarray:
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def boxes(scene) -> list[tuple]:
    """The fixed props as boxes (center, rotation, half sizes) a ray can hit."""
    out = []
    for o in scene.objects:
        if o.kind != "body" or not o.prop or o.pos.ndim != 1 or o.opacity < 0.9:
            continue
        s = o.size
        half = {"box": s[:3], "cylinder": [s[0], s[0], s[1]], "hex": [s[0], s[0], s[1]],
                "capsule": [s[0], s[0], s[1] + s[0]], "sphere": [s[0]] * 3}.get(o.shape)
        if half:
            out.append((o.pos, _rot(o.quat), np.asarray(half, float)))
    return out


def _hits_box(o, d, box) -> bool:
    c, R, h = box
    lo, ld = R.T @ (o - c), R.T @ d
    with np.errstate(divide="ignore", invalid="ignore"):
        t1, t2 = (-h - lo) / ld, (h - lo) / ld
    tmin = np.nanmax(np.minimum(t1, t2))
    tmax = np.nanmin(np.maximum(t1, t2))
    return bool(tmax >= max(tmin, 1e-9))


def _hits_mesh(o, d, A, E1, E2) -> bool:
    h = np.cross(d, E2)
    det = np.einsum("ij,ij->i", E1, h)
    ok = np.abs(det) > 1e-12
    inv = np.where(ok, 1 / np.where(ok, det, 1), 0)
    s = o - A
    u = np.einsum("ij,ij->i", s, h) * inv
    q = np.cross(s, E1)
    v = (q @ d) * inv
    t = np.einsum("ij,ij->i", E2, q) * inv
    return bool((ok & (u >= 0) & (v >= 0) & (u + v <= 1) & (t > 1e-6)).any())


def clear_direction(scene, point, normal=None, X=None, tris=None, size: float = 1.0) -> np.ndarray:
    """The direction to look from (pointing from point toward the eye) that sees point unblocked, facing its
    surface, from a little above where it can, and sees as much of the rest of the part as it can too."""
    p = np.asarray(point, float)
    n = None if normal is None else np.asarray(normal, float) / max(np.linalg.norm(normal), 1e-30)
    bx = boxes(scene)
    mesh, samples = None, []
    if X is not None and tris is not None:
        t = np.asarray(tris)[:, :3]
        A = X[t[:, 0]]
        mesh = (A, X[t[:, 1]] - A, X[t[:, 2]] - A)
        used = np.unique(t)
        pick = used[np.linspace(0, len(used) - 1, 12).astype(int)]
        samples = list(X[pick])
    start = p + (n if n is not None else 0) * size * 0.01
    best, score = None, -np.inf
    for el in (30, 18, 42, 6, -12):
        for az in range(0, 360, 20):
            a, e = np.radians(az), np.radians(el)
            d = np.array([np.cos(e) * np.cos(a), np.cos(e) * np.sin(a), np.sin(e)])
            blocked = any(_hits_box(start, d, b) for b in bx) or (mesh is not None and _hits_mesh(start, d, *mesh))
            hidden = sum(any(_hits_box(q + d * size * 0.01, d, b) for b in bx) for q in samples)
            sc = (0.0 if n is None else float(d @ n)) - abs(el - 26) / 60 + 0.15 * np.cos(a - np.radians(-60))
            sc -= 10 * blocked + 2.0 * hidden / max(len(samples), 1)
            if sc > score:
                best, score = d, sc
    return best
