"""Plane geometry for the board reader, with arcs, rings, point in polygon tests and pad shapes.

Everything works on numpy arrays of (x, y) points in meters. Rings are (N, 2)
arrays without a repeated closing point.
"""
from __future__ import annotations

import math

import numpy as np

# largest angle step that keeps the chord of an arc within 1% of its radius
ARC_STEP = 2.0 * math.acos(0.99)


def arc_points(center, radius: float, a0: float, sweep: float) -> np.ndarray:
    """Points on a circular arc from angle a0 over a signed sweep (radians), both ends included."""
    n = max(1, int(math.ceil(abs(sweep) / ARC_STEP - 1e-9)))
    t = a0 + sweep * np.arange(n + 1) / n
    return np.column_stack([center[0] + radius * np.cos(t), center[1] + radius * np.sin(t)])


def circle_center(p0, p1, p2):
    """Center of the circle through three points, or None when they lie on a line."""
    ax, ay = p0
    bx, by = p1
    cx, cy = p2
    d = 2.0 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by))
    scale = max(abs(ax - bx), abs(ay - by), abs(cx - bx), abs(cy - by), 1e-30)
    if abs(d) < 1e-12 * scale * scale:
        return None
    a2, b2, c2 = ax * ax + ay * ay, bx * bx + by * by, cx * cx + cy * cy
    ux = (a2 * (by - cy) + b2 * (cy - ay) + c2 * (ay - by)) / d
    uy = (a2 * (cx - bx) + b2 * (ax - cx) + c2 * (bx - ax)) / d
    return ux, uy


def arc_through(p0, pm, p1) -> np.ndarray:
    """Polyline of the circular arc that starts at p0, passes through pm and ends at p1."""
    c = circle_center(p0, pm, p1)
    if c is None:
        return np.array([p0, p1], dtype=float)
    r = math.hypot(p0[0] - c[0], p0[1] - c[1])
    a0 = math.atan2(p0[1] - c[1], p0[0] - c[0])
    am = math.atan2(pm[1] - c[1], pm[0] - c[0])
    a1 = math.atan2(p1[1] - c[1], p1[0] - c[0])
    tau = 2.0 * math.pi
    ccw = (a1 - a0) % tau
    if ccw == 0.0:
        ccw = tau
    sweep = ccw if (am - a0) % tau < ccw else ccw - tau
    pts = arc_points(c, r, a0, sweep)
    pts[0] = p0
    pts[-1] = p1
    return pts


def circle_ring(center, radius: float, n: int = 0) -> np.ndarray:
    """A closed ring approximating a circle, fine enough for the 1% rule unless n is given."""
    n = n or max(8, int(math.ceil(2.0 * math.pi / ARC_STEP)))
    t = 2.0 * math.pi * np.arange(n) / n
    return np.column_stack([center[0] + radius * np.cos(t), center[1] + radius * np.sin(t)])


def ring_area(ring: np.ndarray) -> float:
    """Unsigned area of a closed ring (shoelace formula)."""
    if len(ring) < 3:
        return 0.0
    x, y = ring[:, 0], ring[:, 1]
    return 0.5 * abs(float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def ring_centroid(ring: np.ndarray):
    """Area centroid of a closed ring, the vertex mean if it has no area."""
    x, y = ring[:, 0], ring[:, 1]
    xn, yn = np.roll(x, -1), np.roll(y, -1)
    cross = x * yn - xn * y
    a = cross.sum() / 2.0
    if abs(a) < 1e-30:
        return float(x.mean()), float(y.mean())
    return float(((x + xn) * cross).sum() / (6 * a)), float(((y + yn) * cross).sum() / (6 * a))


def _pip_direct(px, py, x0, y0, x1, y1) -> np.ndarray:
    cond = (y0[None, :] > py[:, None]) != (y1[None, :] > py[:, None])
    with np.errstate(divide="ignore", invalid="ignore"):
        xi = x0[None, :] + (py[:, None] - y0[None, :]) * (x1 - x0)[None, :] / (y1 - y0)[None, :]
    return (np.count_nonzero(cond & (px[:, None] < xi), axis=1) % 2) == 1


def points_in_ring(points, ring: np.ndarray) -> np.ndarray:
    """Even odd ray casting test of many points against one ring, banded by y for big rings."""
    pts = np.atleast_2d(np.asarray(points, dtype=float))
    m, n = len(pts), len(ring)
    out = np.zeros(m, dtype=bool)
    if m == 0 or n < 3:
        return out
    x0, y0 = ring[:, 0], ring[:, 1]
    x1, y1 = np.roll(x0, -1), np.roll(y0, -1)
    px, py = pts[:, 0], pts[:, 1]
    box = (px >= x0.min()) & (px <= x0.max()) & (py >= y0.min()) & (py <= y0.max())
    idx = np.nonzero(box)[0]
    if len(idx) == 0:
        return out
    if len(idx) * n <= 2_000_000:
        out[idx] = _pip_direct(px[idx], py[idx], x0, y0, x1, y1)
        return out
    # split the ring's height into bands and test each point only against edges crossing its band
    ylo, yhi = float(y0.min()), float(y0.max())
    nb = int(min(4096, max(16, math.sqrt(n) * 2)))
    h = (yhi - ylo) / nb or 1.0
    elo = np.clip(((np.minimum(y0, y1) - ylo) / h).astype(int), 0, nb - 1)
    ehi = np.clip(((np.maximum(y0, y1) - ylo) / h).astype(int), 0, nb - 1)
    span = ehi - elo + 1
    eband = np.repeat(elo, span) + (np.arange(span.sum()) - np.repeat(np.cumsum(span) - span, span))
    eidx = np.repeat(np.arange(n), span)
    order = np.argsort(eband, kind="stable")
    eband, eidx = eband[order], eidx[order]
    starts = np.searchsorted(eband, np.arange(nb + 1))
    pband = np.clip(((py[idx] - ylo) / h).astype(int), 0, nb - 1)
    porder = np.argsort(pband, kind="stable")
    pstarts = np.searchsorted(pband[porder], np.arange(nb + 1))
    for b in range(nb):
        if pstarts[b] == pstarts[b + 1] or starts[b] == starts[b + 1]:
            continue
        pi = idx[porder[pstarts[b]:pstarts[b + 1]]]
        ei = eidx[starts[b]:starts[b + 1]]
        chunk = max(1, 2_000_000 // max(1, len(ei)))
        for k in range(0, len(pi), chunk):
            sel = pi[k:k + chunk]
            out[sel] = _pip_direct(px[sel], py[sel], x0[ei], y0[ei], x1[ei], y1[ei])
    return out


def points_in_polygon(points, outer: np.ndarray, holes=()) -> np.ndarray:
    """Points inside the outer ring and outside every hole. Holes may overlap each other."""
    pts = np.atleast_2d(np.asarray(points, dtype=float))
    inside = points_in_ring(pts, outer)
    if not len(holes) or not inside.any():
        return inside
    sizes = np.array([len(h) for h in holes])
    allh = np.concatenate(holes)
    starts = np.concatenate([[0], np.cumsum(sizes)[:-1]])
    lo = np.minimum.reduceat(allh, starts)
    hi = np.maximum.reduceat(allh, starts)
    all_idx = np.nonzero(inside)[0]
    step = max(1, 4_000_000 // len(holes))
    for c in range(0, len(all_idx), step):
        idx = all_idx[c:c + step]
        px, py = pts[idx, 0][:, None], pts[idx, 1][:, None]
        near = (px >= lo[None, :, 0]) & (px <= hi[None, :, 0]) & (py >= lo[None, :, 1]) & (py <= hi[None, :, 1])
        for k in np.nonzero(near.any(axis=0))[0]:
            sel = idx[near[:, k]]
            sel = sel[inside[sel]]
            if len(sel):
                inside[sel] &= ~points_in_ring(pts[sel], holes[k])
    return inside


def segment_distance(p, a, b) -> np.ndarray:
    """Distance from points p (M, 2) to segments a-b (each (K, 2)), shape (M, K), plus the projection parameter."""
    p = np.atleast_2d(p)[:, None, :]
    a = np.atleast_2d(a)[None, :, :]
    b = np.atleast_2d(b)[None, :, :]
    d = b - a
    dd = (d * d).sum(-1)
    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.where(dd > 0, ((p - a) * d).sum(-1) / dd, 0.0)
    t = np.clip(t, 0.0, 1.0)
    q = a + t[..., None] * d
    return np.sqrt(((p - q) ** 2).sum(-1)), t


def shape_contains(at, size, shape: str, angle_deg: float, points, margin: float = 0.0) -> np.ndarray:
    """Whether points lie inside a pad shape (circle, oval or rotated rectangle), grown by margin."""
    pts = np.atleast_2d(np.asarray(points, dtype=float))
    d = pts - np.asarray(at, dtype=float)
    a = math.radians(angle_deg)
    c, s = math.cos(a), math.sin(a)
    lx = d[:, 0] * c + d[:, 1] * s
    ly = -d[:, 0] * s + d[:, 1] * c
    w, h = size
    if shape == "circle":
        return np.hypot(lx, ly) <= max(w, h) / 2.0 + margin
    if shape == "oval":
        r = min(w, h) / 2.0
        if w >= h:
            ex = np.clip(np.abs(lx), 0, w / 2.0 - r)
            return np.hypot(np.abs(lx) - ex, ly) <= r + margin
        ey = np.clip(np.abs(ly), 0, h / 2.0 - r)
        return np.hypot(lx, np.abs(ly) - ey) <= r + margin
    return (np.abs(lx) <= w / 2.0 + margin) & (np.abs(ly) <= h / 2.0 + margin)


def convex_hull(points) -> np.ndarray:
    """Convex hull of points in counter clockwise order (monotone chain)."""
    pts = np.unique(np.round(np.asarray(points, dtype=float), 12), axis=0)
    if len(pts) < 3:
        return pts

    def half(seq):
        out = []
        for p in seq:
            while len(out) >= 2:
                (ax, ay), (bx, by) = out[-2], out[-1]
                if (bx - ax) * (p[1] - ay) - (by - ay) * (p[0] - ax) > 0:
                    break
                out.pop()
            out.append((p[0], p[1]))
        return out

    lower = half(pts)
    upper = half(pts[::-1])
    return np.array(lower[:-1] + upper[:-1])


def min_area_rect(points):
    """Smallest rectangle around points as (center, (w, h), angle_deg) with the angle in [0, 180)."""
    hull = convex_hull(points)
    if len(hull) == 0:
        return (0.0, 0.0), (0.0, 0.0), 0.0
    if len(hull) < 3:
        lo, hi = hull.min(0), hull.max(0)
        return tuple((lo + hi) / 2.0), tuple(hi - lo), 0.0
    best = None
    edges = np.roll(hull, -1, axis=0) - hull
    for ang in np.unique(np.round(np.arctan2(edges[:, 1], edges[:, 0]) % (math.pi / 2), 9)):
        c, s = math.cos(ang), math.sin(ang)
        rx = hull[:, 0] * c + hull[:, 1] * s
        ry = -hull[:, 0] * s + hull[:, 1] * c
        area = (rx.max() - rx.min()) * (ry.max() - ry.min())
        if best is None or area < best[0] - 1e-18:
            best = (area, ang, rx, ry)
    _, ang, rx, ry = best
    w, h = rx.max() - rx.min(), ry.max() - ry.min()
    mx, my = (rx.max() + rx.min()) / 2.0, (ry.max() + ry.min()) / 2.0
    c, s = math.cos(ang), math.sin(ang)
    center = (mx * c - my * s, mx * s + my * c)
    deg = math.degrees(ang)
    if deg > 45.0:
        deg -= 90.0
        w, h = h, w
    deg = round(deg, 6) % 180.0
    return (float(center[0]), float(center[1])), (float(w), float(h)), deg


def chain_rings(polylines, tol: float):
    """Join open polylines whose ends meet within tol into closed rings.

    Returns (rings, leftovers) where leftovers are chains that never closed.
    """
    rings, pieces = [], []
    for p in polylines:
        p = np.asarray(p, dtype=float)
        if len(p) < 2:
            continue
        if len(p) > 3 and np.hypot(*(p[0] - p[-1])) <= tol:
            rings.append(p[:-1])
            continue
        pieces.append(p)
    used = np.zeros(len(pieces), dtype=bool)
    if not pieces:
        return rings, []
    starts = np.array([p[0] for p in pieces])
    ends = np.array([p[-1] for p in pieces])
    leftovers = []
    for i in range(len(pieces)):
        if used[i]:
            continue
        used[i] = True
        chain = [pieces[i]]
        head, tail = pieces[i][0], pieces[i][-1]
        while True:
            if np.hypot(*(tail - head)) <= tol and len(chain) > 1:
                break
            free = ~used
            if not free.any():
                break
            ds = np.where(free, np.hypot(*(starts - tail).T), np.inf)
            de = np.where(free, np.hypot(*(ends - tail).T), np.inf)
            js, je = int(np.argmin(ds)), int(np.argmin(de))
            if ds[js] <= tol and ds[js] <= de[je]:
                nxt = pieces[js]
                used[js] = True
            elif de[je] <= tol:
                nxt = pieces[je][::-1]
                used[je] = True
            else:
                break
            chain.append(nxt[1:])
            tail = nxt[-1]
        pts = np.vstack(chain)
        if len(pts) > 2 and np.hypot(*(pts[-1] - pts[0])) <= tol:
            rings.append(pts[:-1])
        else:
            leftovers.append(pts)
    return rings, leftovers


def bezier_points(p0, p1, p2, p3, n: int = 24) -> np.ndarray:
    """Points on a cubic Bezier curve."""
    t = np.linspace(0.0, 1.0, n + 1)[:, None]
    p0, p1, p2, p3 = (np.asarray(p, dtype=float) for p in (p0, p1, p2, p3))
    return ((1 - t) ** 3) * p0 + 3 * ((1 - t) ** 2) * t * p1 + 3 * (1 - t) * t * t * p2 + (t ** 3) * p3
