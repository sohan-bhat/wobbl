"""The air a part warms in still air, estimated from its own heat run: where it rises, how fast and how warm.

A heat run in still air gives the part a film coefficient and solves its temperatures; it does not solve the air.
This module estimates the convection plume from what the run did solve, so a view can show the air carrying the
heat away. It is an estimate, and says so: speeds and temperatures to a few tens of percent, never a solved field.

1. The heat the air takes, Q, is the run's own film on its own solved surface: the sum of h A (T - T_air) over the
   faces (the rest of the power leaves as radiation).
2. The air rises through the part's open columns: the places in its footprint where a vertical line misses the
   part (the gaps between fins), and round its sides. Below the part's top the air in them is a chimney: buoyancy
   over the height L, rho g beta dT L, drives the flow out of the top, one dynamic head 1/2 rho w^2, while the air
   carries Q away, Q = rho c_p w A dT over the open area A at the top. So w^3 = 2 g beta L Q / (rho c_p A) and
   dT = Q / (rho c_p w A), each air column warming from the air's temperature at the bottom to dT at the top.
3. Above the part the plume spreads by entrainment (Morton, Taylor and Turner 1956): its radius grows as 0.11 of
   the height above a virtual origin set so it matches the part at its top, its speed falls as the -1/3 power of
   that height and its warming as the -5/3 power, which keeps the heat it carries.

Air properties are taken at the mean of the air and the part's surface. Up is +z. The result is a list of
streamlines, each with the air's speed (mm/s) and temperature (C) at every point.
"""
from __future__ import annotations

import numpy as np

G = 9.81
ALPHA_SPREAD = 0.11                   # plume radius per unit height above its virtual origin (List 1982)


def air(T_C: float) -> tuple[float, float]:
    """Density (kg/m3) and specific heat (J/(kg K)) of dry air at T_C and one atmosphere."""
    return 101325.0 / (287.05 * (T_C + 273.15)), 1007.0


def convected(X: np.ndarray, tris: np.ndarray, T: np.ndarray, h_W_m2K: float, T_air: float) -> tuple[float, float]:
    """The heat the film takes from the part's surface (W) and the surface's area weighted mean temperature."""
    t = tris[:, :3]
    a, b, c = X[t[:, 0]], X[t[:, 1]], X[t[:, 2]]
    area = 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1) * 1e-6
    Tf = T[t].mean(1)
    return float(h_W_m2K * (area * (Tf - T_air)).sum()), float((area * Tf).sum() / max(area.sum(), 1e-30))


def _columns(X: np.ndarray, tris: np.ndarray, pts: np.ndarray, z0: float) -> tuple[np.ndarray, np.ndarray]:
    """For each xy point, whether a vertical line from z0 up meets the part, and the height of the part's top
    surface under z0 there (the floor the air comes in over, the bottom of the part where nothing is under it)."""
    t = tris[:, :3]
    A, B, C = X[t[:, 0]], X[t[:, 1]], X[t[:, 2]]
    d = (B[:, 1] - C[:, 1]) * (A[:, 0] - C[:, 0]) + (C[:, 0] - B[:, 0]) * (A[:, 1] - C[:, 1])
    ok = np.abs(d) > 1e-12
    A, B, C, d = A[ok], B[ok], C[ok], d[ok]
    zt = np.maximum(np.maximum(A[:, 2], B[:, 2]), C[:, 2])
    zc = (A[:, 2] + B[:, 2] + C[:, 2]) / 3
    hit = np.zeros(len(pts), bool)
    floor = np.full(len(pts), float(X[:, 2].min()))
    for i, (x, y) in enumerate(pts):
        l1 = ((B[:, 1] - C[:, 1]) * (x - C[:, 0]) + (C[:, 0] - B[:, 0]) * (y - C[:, 1])) / d
        l2 = ((C[:, 1] - A[:, 1]) * (x - C[:, 0]) + (A[:, 0] - C[:, 0]) * (y - C[:, 1])) / d
        cover = (l1 >= 0) & (l2 >= 0) & (l1 + l2 <= 1)
        hit[i] = bool(np.any(cover & (zt > z0)))
        under = cover & (zc <= z0)
        if under.any():
            floor[i] = float(zc[under].max())
    return hit, floor


def plume(X: np.ndarray, tris: np.ndarray, T: np.ndarray, h_W_m2K: float, T_air: float, *, lines: int = 64,
          rise: float = 2.5) -> dict | None:
    """The convection plume of a part in still air (X in mm, T in C at its nodes): streamlines from where the air
    comes in at the bottom, up through the part's open columns and on up rise times the part's width above it,
    with the air's speed in mm/s and its temperature in C at every point. None when the air takes no heat."""
    X = np.asarray(X, float)
    Q, Ts = convected(X, tris, T, h_W_m2K, T_air)
    if not (Q > 0):
        return None
    lo, hi = X.min(0), X.max(0)
    L = float(hi[2] - lo[2]) * 1e-3
    width = float(max(hi[0] - lo[0], hi[1] - lo[1]))
    rho, cp = air(0.5 * (Ts + T_air))
    beta = 1.0 / (0.5 * (Ts + T_air) + 273.15)
    # the open columns in the footprint, on a grid fine enough to find a gap between fins
    n = 96
    gx = lo[0] + (np.arange(n) + 0.5) / n * (hi[0] - lo[0])
    gy = lo[1] + (np.arange(n) + 0.5) / n * (hi[1] - lo[1])
    P = np.array([(x, y) for y in gy for x in gx])
    hit, floor = _columns(X, tris, P, lo[2] + 0.5 * (hi[2] - lo[2]))
    free = ~hit
    cell = (hi[0] - lo[0]) * (hi[1] - lo[1]) / (n * n) * 1e-6
    # the air also rises in a skin round the part's sides, a few millimetres of boundary layer
    skin = 0.004 * 2 * ((hi[0] - lo[0]) + (hi[1] - lo[1])) * 1e-3
    A = float(free.sum() * cell + skin)
    w = (2 * G * beta * max(L, 1e-4) * Q / (rho * cp * A)) ** (1 / 3)      # m/s at the top
    dT = Q / (rho * cp * w * A)                                              # K at the top
    # seeds: open columns spread over the footprint, and points just outside its sides
    rng = np.random.default_rng(7)
    inside = P[free]
    k_in = min(len(inside), int(lines * 0.7))
    sel = np.linspace(0, len(inside) - 1, k_in).astype(int)
    cells = np.flatnonzero(free)[sel] if k_in else np.zeros(0, int)
    grid = free.reshape(n, n)                     # [row = y, col = x]
    step = np.array([(hi[0] - lo[0]) / n, (hi[1] - lo[1]) / n])

    def way_out(ci):
        """The shortest straight run through open columns from cell ci out of the footprint, as a unit xy
        direction and its length in mm, or None when the column is closed all round (air rises from its floor)."""
        r, q = divmod(int(ci), n)
        best = None
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            k, rr, qq = 0, r, q
            while 0 <= rr < n and 0 <= qq < n and grid[rr, qq]:
                rr, qq, k = rr + dy, qq + dx, k + 1
            if not (0 <= rr < n and 0 <= qq < n):
                length = k * float(step[0] if dx else step[1])
                if best is None or length < best[1]:
                    best = (np.array([dx, dy], float), length)
        return best
    pick = inside[sel] if k_in else np.zeros((0, 2))
    floors = np.r_[floor[free][sel] if k_in else np.zeros(0), np.full(lines - k_in, lo[2])]
    m = lines - len(pick)
    ang = np.linspace(0, 2 * np.pi, m, endpoint=False) + rng.uniform(0, 0.3, m)
    c = np.array([(lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2])
    half = np.array([(hi[0] - lo[0]) / 2, (hi[1] - lo[1]) / 2])
    # the rectangle's edge at each angle, a few millimetres out
    d = np.c_[np.cos(ang), np.sin(ang)]
    s = 1.0 / np.max(np.abs(d) / half, axis=1)
    side = c + d * (s[:, None] + 2.5)
    seeds = np.vstack([pick, side])
    out_side = np.r_[np.zeros(len(pick), bool), np.ones(len(side), bool)]
    # the plume above: radius b = 0.11 (z - zv), matched to the part's half width at its top
    b_top = 0.5 * width
    zv = hi[2] - b_top / ALPHA_SPREAD
    z_end = hi[2] + rise * width
    paths, speeds, temps = [], [], []
    for (x, y), outer, zf in zip(seeds, out_side, floors):
        pts, sp, tc = [], [], []
        z_in = zf + 0.08 * (hi[2] - zf)
        # in from the side at the bottom: the air comes level toward its column, at the speed it rises with
        r0 = np.array([x, y]) - c
        out = None if outer else way_out(cells[len(paths)])
        if outer:
            come = np.array([x, y]) + r0 / (np.linalg.norm(r0) + 1e-9) * 0.3 * width
        elif out is not None:
            come = np.array([x, y]) + out[0] * (out[1] + 0.15 * width)
        else:
            come = np.array([x, y])
        zb = zf + 0.02 * (hi[2] - zf)
        for u in np.linspace(0, 1, 8)[:-1]:
            q = come + (np.array([x, y]) - come) * u
            pts.append([q[0], q[1], zb + (z_in - zb) * u * u])
            sp.append(w * (0.35 + 0.65 * u))
            tc.append(T_air)
        # up the column: warming steadily to dT at the top
        for z in np.linspace(z_in, hi[2], 14):
            f = (z - zf) / max(hi[2] - zf, 1e-9)
            pts.append([x, y, z])
            sp.append(w)
            tc.append(T_air + dT * f)
        # the plume: each streamline keeps its share of the plume's radius as it widens
        share = np.linalg.norm(r0) / max(b_top, 1e-9)
        share = min(share, 1.15 if outer else 1.0)
        for z in np.linspace(hi[2], z_end, 24)[1:]:
            b = ALPHA_SPREAD * (z - zv)
            g = ((hi[2] - zv) / (z - zv))
            q = c + r0 / (np.linalg.norm(r0) + 1e-9) * share * b
            pts.append([q[0], q[1], z])
            sp.append(w * g ** (1 / 3))
            tc.append(T_air + dT * g ** (5 / 3))
        paths.append(np.array(pts))
        speeds.append(np.array(sp) * 1000.0)
        temps.append(np.array(tc))
    return {"paths": paths, "speed": speeds, "temp": temps, "Q_W": Q, "w_mps": float(w), "dT_K": float(dT),
            "open_m2": A, "surface_C": Ts,
            "about": "Estimated from the run's film and solved surface: a chimney through the part's open columns "
                     "and an entraining plume above it (Morton, Taylor and Turner). Not a solved air field."}


__all__ = ["plume", "convected", "air"]
