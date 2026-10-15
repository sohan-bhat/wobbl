"""Flow runs as 3D scenes for the viewer, written through wobbl.viz.Scene from the run's own playback.json.

A steady run shows the part's surface coloured by wall pressure with streamlines traced through the velocity in the
plane through the middle, a glow running along each at its local speed. An unsteady run (DES or LES) shows the part
with particles carried through the saved frames of the swinging wake. Lengths are mm.
"""
from __future__ import annotations

import base64
import json
from pathlib import Path

import numpy as np
from scipy.ndimage import map_coordinates


def write(out: str | Path) -> Path | None:
    """Build out/scene.wobbl from out/playback.json, or None when the run has no plane flow to draw. Never raises,
    since a view that fails must not lose the result."""
    out = Path(out)
    try:
        pb = json.loads((out / "playback.json").read_text())
        if not pb.get("fields"):
            return None
        o = json.loads((out / "outcome.json").read_text()) if (out / "outcome.json").exists() else {}
        s = _build(pb, o)
        if s is None:
            return None
        verdict = (o.get("verdict") or "ok").upper()
        s.verdict(verdict, o.get("headline", "")[:300], "pass" if verdict == "OK" else "warn")
        return s.save(out)
    except Exception:  # noqa: BLE001
        return None


def decode(d: dict) -> np.ndarray:
    """An encoded array back to numbers, with NaN where a quantized value is marked missing."""
    a = np.frombuffer(base64.b64decode(d["data"]), "<" + np.dtype(d["dtype"]).str[1:]).reshape(d["shape"])
    if "scale" not in d:
        return a
    v = np.asarray(d.get("offset", 0.0)) + np.asarray(d["scale"]) * a.astype(float)
    return np.where(a == d["missing"], np.nan, v) if "missing" in d else v


class Plane:
    """One in plane velocity grid, sampled bilinearly, with the mapping from grid coordinates to the scene."""

    def __init__(self, f: dict, k: float):
        g = f["grid"]
        self.x0, self.x1, self.y0, self.y1 = (g[c] * k for c in ("x0", "x1", "y0", "y1"))
        self.nx, self.ny = g["nx"], g["ny"]
        self.u, self.v = decode(f["components"]["u"]), decode(f["components"]["v"])
        p = f.get("plane") or {"origin": [0, 0, 0], "u": [1, 0, 0], "v": [0, 1, 0]}
        self.o, self.eu, self.ev = (np.array(p[c], float) * (k if c == "origin" else 1) for c in ("origin", "u", "v"))
        self.k = k

    def cell(self, x, y):
        return ((np.asarray(y) - self.y0) / (self.y1 - self.y0) * (self.ny - 1),
                (np.asarray(x) - self.x0) / (self.x1 - self.x0) * (self.nx - 1))

    def sample(self, x, y, vel=None):
        """(u, v) at points, NaN outside the fluid. Missing cells poison the interpolation, so they do too."""
        u, v = vel if vel is not None else (self.u, self.v)
        c = self.cell(x, y)
        return (map_coordinates(u, c, order=1, mode="nearest", cval=np.nan),
                map_coordinates(v, c, order=1, mode="nearest", cval=np.nan))

    def to3d(self, x, y) -> np.ndarray:
        return self.o + np.outer(x, self.eu) + np.outer(y, self.ev)


def trace(pl: Plane, seeds: np.ndarray, steps: int = 140) -> list[tuple[np.ndarray, np.ndarray]]:
    """Streamlines by midpoint steps of a fraction of a grid cell, until they leave the fluid or the plane."""
    h = 0.8 * min(pl.x1 - pl.x0, 1e9) / pl.nx
    paths = []
    for x, y in seeds:
        pts, spd = [], []
        for _ in range(steps):
            u, v = pl.sample([x], [y])
            m = float(np.hypot(u[0], v[0]))
            if not np.isfinite(m) or m < 1e-9 or not (pl.x0 <= x <= pl.x1 and pl.y0 <= y <= pl.y1):
                break
            pts.append((x, y))
            spd.append(m)
            xm, ym = x + 0.5 * h * u[0] / m, y + 0.5 * h * v[0] / m
            um, vm = pl.sample([xm], [ym])
            mm = float(np.hypot(um[0], vm[0]))
            if not np.isfinite(mm) or mm < 1e-9:
                break
            x, y = x + h * um[0] / mm, y + h * vm[0] / mm
        if len(pts) > 4:
            p = np.array(pts)
            paths.append((pl.to3d(p[:, 0], p[:, 1]), np.array(spd)))
    return paths


def _seeds(pl: Plane, n: int = 22) -> np.ndarray:
    """Seeds on the upstream edge of the grid across the fluid there, and a few more inside on the slower side."""
    ys = np.linspace(pl.y0, pl.y1, n + 2)[1:-1]
    xs = pl.x0 + 0.01 * (pl.x1 - pl.x0)
    return np.c_[np.full(n, xs), ys]


def _scene(pb: dict, o: dict):
    from ..viz import Scene
    return Scene(pb.get("title", "Flow"), domain="flow", question=o.get("question", ""), length="mm",
                 source="Streamlines follow the velocity the solver computed in the plane through the middle.")


def _mesh(s, pb: dict):
    if not pb.get("mesh"):
        return None
    X, T = decode(pb["mesh"]["vertices"]), decode(pb["mesh"]["triangles"])
    m = s.mesh(X, T.astype(int), name="part", color="#b9bdc4")
    p = pb.get("surface", {}).get("pressure")
    if p:
        v = decode(p)
        m.color_by("wall pressure", v, unit="Pa", colormap="turbo")
    return m


def _build(pb: dict, o: dict):
    f = pb["fields"][0]
    k = 1000.0 if f["grid"].get("unit") == "m" else 1.0
    pl = Plane(f, k)
    s = _scene(pb, o)
    _mesh(s, pb)
    if pb.get("frames"):
        return _wake(s, pb, pl, k)
    lines = trace(pl, _seeds(pl))
    if not lines:
        return None
    sp = np.concatenate([w for _, w in lines])
    top = float(np.nanmax(sp))
    s.lines([p for p, _ in lines], values=[w for _, w in lines], field="speed", unit="m/s", colormap="viridis",
            range=(0.0, top), flow=np.array([float(w.mean()) for _, w in lines]) / top * (pl.x1 - pl.x0) / 3,
            spacing=(pl.x1 - pl.x0) / 14, name="streamlines", width=1.5)
    c = pl.to3d([(pl.x0 + pl.x1) / 2], [(pl.y0 + pl.y1) / 2])[0]
    s.camera(target=c, direction=(0.35, -1.0, 0.45), distance=float(pl.x1 - pl.x0) * 1.6)
    return s


def _wake(s, pb: dict, pl: Plane, k: float):
    """Particles carried through the frames of a time accurate run, then looped, re-seeded where they leave."""
    fr = pb["frames"]
    t = np.array([f["t"] for f in fr], float)
    vel = [(decode(f["fields"][0]["components"]["u"]), decode(f["fields"][0]["components"]["v"])) for f in fr]
    n = 260
    rng = np.random.default_rng(1)
    pos = np.c_[rng.uniform(pl.x0, pl.x1, n), rng.uniform(pl.y0, pl.y1, n)]
    out = []
    sub = 4
    dt = np.diff(t, append=t[-1] + (t[-1] - t[0]) / max(len(t) - 1, 1)) / sub
    for _ in range(3):                         # spin the particles through the frames once before recording
        for i in range(len(fr)):
            pos = _advect(pl, pos, vel[i], dt[i] * sub, k, rng)
    for i in range(len(fr)):
        out.append(pl.to3d(pos[:, 0], pos[:, 1]))
        pos = _advect(pl, pos, vel[i], dt[i] * sub, k, rng)
    s.timeline(t, unit="s", label="time", seconds=8.0, between="interpolated")
    sp = np.array([np.hypot(*pl.sample(pos[:, 0], pos[:, 1], vel[i])) for i in range(len(fr))])
    s.points(np.array(out), values=np.nan_to_num(sp), field="speed", unit="m/s", range=(0.0, float(np.nanmax(sp))),
             size=1.2, name="particles")
    s.camera(target=pl.to3d([(pl.x0 + pl.x1) / 2], [(pl.y0 + pl.y1) / 2])[0], direction=(0.2, -1.0, 0.5),
             distance=float(pl.x1 - pl.x0) * 1.6)
    return s


def _advect(pl: Plane, pos: np.ndarray, vel, dt: float, k: float, rng) -> np.ndarray:
    """One frame's step, in the scene's mm, with particles that leave the fluid or the plane re-seeded upstream."""
    u, v = pl.sample(pos[:, 0], pos[:, 1], vel)
    new = pos + dt * np.c_[u, v] * 1000.0
    bad = ~np.isfinite(new).all(1) | (new[:, 0] > pl.x1) | (new[:, 0] < pl.x0) | (new[:, 1] > pl.y1) | (new[:, 1] < pl.y0)
    bad |= ~np.isfinite(pl.sample(new[:, 0], new[:, 1], vel)[0])
    new[bad] = np.c_[np.full(bad.sum(), pl.x0 + 0.005 * (pl.x1 - pl.x0)), rng.uniform(pl.y0, pl.y1, bad.sum())]
    return new
