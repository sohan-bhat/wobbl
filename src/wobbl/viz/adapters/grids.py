"""Field and flow runs as scenes, from the fields they sampled on a slice: the slice colored by the field with its
contour lines over it and the parts standing on it, and for a flow, particles carried by the solved velocity.

Particles are moved through the run's velocity field here, with the time step small against a grid cell, so where
they go is the run's flow; how many there are and where they start is chosen for the eye.
"""
from __future__ import annotations

import base64
from pathlib import Path

import numpy as np

from ..scene import Scene
from .common import read, verdict

LEVELS = 250


def unpack(d: dict) -> np.ndarray:
    """A fields playback grid (integers 0 to 250 on [min, max], -1 for none), as values with NaN for none."""
    q = np.asarray(d["q"], float).reshape(d["shape"])
    return np.where(q < 0, np.nan, d["min"] + q / LEVELS * (d["max"] - d["min"]))


def unpack_flow(c: dict, shape) -> np.ndarray:
    """A flow playback component (one byte a value, 255 for no fluid), as values with NaN where there is none."""
    q = np.frombuffer(base64.b64decode(c["data"]), np.uint8).astype(float).reshape(shape)
    if "lo" in c:                      # the first encoding: lo and hi
        return np.where(q == 255, np.nan, c["lo"] + q / 254 * (c["hi"] - c["lo"]))
    return np.where(q == c.get("missing", 255), np.nan, c.get("offset", 0.0) + c.get("scale", 1.0) * q)


def plane(s, x0, x1, y0, y1, values, name, unit, cmap, z=0.0, frames=None):
    """The slice as a grid of quads at height z, colored per vertex; cells with no value are left out."""
    ny, nx = values.shape[-2:]
    gx, gy = np.meshgrid(np.linspace(x0, x1, nx), np.linspace(y0, y1, ny))
    X = np.c_[gx.ravel(), gy.ravel(), np.full(gx.size, z)]
    ok = np.isfinite(values if frames is None else values[0]).ravel()
    idx = np.arange(nx * ny).reshape(ny, nx)
    a, b, c, d = idx[:-1, :-1].ravel(), idx[:-1, 1:].ravel(), idx[1:, 1:].ravel(), idx[1:, :-1].ravel()
    keep = ok[a] & ok[b] & ok[c] & ok[d]
    T = np.concatenate([np.c_[a, b, c][keep], np.c_[a, c, d][keep]])
    if not len(T):
        return None
    m = s.mesh(X, T, name=name, max_vertices=len(X) + 1)
    v = values.reshape(len(values), -1) if frames is not None else values.ravel()
    m.color_by(name, v, unit=unit, colormap=cmap)
    return m


def contours(v: np.ndarray, x0, x1, y0, y1, n=12, z=0.3):
    """Contour lines of a sampled field, as 3D polylines on the slice."""
    try:
        import contourpy
    except ImportError:
        return []
    ny, nx = v.shape
    gen = contourpy.contour_generator(np.linspace(x0, x1, nx), np.linspace(y0, y1, ny), np.ma.masked_invalid(v))
    lv = np.linspace(np.nanmin(v), np.nanmax(v), n + 2)[1:-1]
    out = []
    for level in lv:
        for ln in gen.lines(level):
            if len(ln) > 3:
                out.append(np.c_[ln, np.full(len(ln), z)])
    return out[:400]


def parts(s, shapes, h):
    """The model's parts standing on the slice: rectangles as blocks, disks as cylinders, polygons as outlines."""
    for p in shapes or []:
        d = p["data"]
        col = "#2c2f34" if p.get("hole") else "#b9bdc4"
        if p["kind"] in ("rect", "rectangle") and len(d) == 4:
            x0, y0, x1, y1 = d
            s.body("box", [abs(x1 - x0) / 2, abs(y1 - y0) / 2, h], pos=[(x0 + x1) / 2, (y0 + y1) / 2, h], color=col,
                   name=p.get("name", "part"), opacity=0.85)
        elif p["kind"] in ("disk", "circle") and len(d) == 3:
            s.body("cylinder", [d[2], h], pos=[d[0], d[1], h], color=col, name=p.get("name", "part"), opacity=0.85)
        elif p["kind"] == "polygon" and len(d) > 2:
            ring = np.asarray(d, float)
            ring = np.vstack([ring, ring[:1]])
            s.lines([np.c_[ring, np.full(len(ring), h * 2)]], color=col, width=2.5, name=p.get("name", "part"))


def build_fields(out: Path, pb: dict) -> Path | None:
    g = pb.get("grid")
    if not g:
        return None
    o = read(out, "outcome.json") or {}
    packed = [(k, v) for k, v in pb.items() if isinstance(v, dict) and "q" in v and "shape" in v]
    if not packed:
        return None
    key, d = packed[0]
    vals = unpack(d)
    desc = pb.get("manifest", {}).get(key, key)
    name = desc.split(",")[0].strip() if isinstance(desc, str) else key
    unit = next((u for u in ("V/m", "kV/mm", "T", "mT", "A/m", "Pa", "dB", "V", "A") if isinstance(desc, str) and
                 f" {u}" in desc), "")
    x0, x1, y0, y1 = g["x0"], g["x1"], g["y0"], g["y1"]
    span = max(x1 - x0, y1 - y0)
    s = Scene(f"The {pb.get('kind', 'field').replace('-', ' ')} on its slice", domain="fields",
              question=o.get("question", ""), length="mm",
              source="The field as the run sampled it on a slice through the model, with its contour lines.")
    plane(s, x0, x1, y0, y1, vals, name, unit, "turbo")
    pot = next((unpack(v) for k, v in packed[1:] if k in ("v", "a", "az", "phi", "potential")), None)
    lines = contours(pot if pot is not None else vals, x0, x1, y0, y1, z=span * 0.002)
    if lines:
        s.lines(lines, color="#f2f2ee", width=1.2, name="field lines" if pot is not None else "contours")
    if g.get("mirror"):
        s.note("The model is axisymmetric about x = 0, and this is one half of its slice.")
    parts(s, pb.get("parts"), span * 0.01)
    nums = pb.get("numbers", {})
    for k, v in list(nums.items())[:4]:
        if isinstance(v, (int, float)):
            s.readout(k.replace("_", " "), [v], digits=4)
    verdict(s, o)
    s.camera(target=[(x0 + x1) / 2, (y0 + y1) / 2, 0], direction=(0.2, -0.8, 1.0), distance=span * 1.5)
    return s.save(out)


def particles(u, v, x0, x1, y0, y1, n=220, frames=120, rng=None):
    """Particles carried through a velocity grid by second order steps, respawning at the inflow edge."""
    rng = rng or np.random.default_rng(3)
    ny, nx = u.shape
    sp = np.nan_to_num(np.hypot(u, v))
    vmax = max(float(sp.max()), 1e-12)
    dx = (x1 - x0) / (nx - 1)
    dt = 0.6 * dx / vmax
    U, V = np.nan_to_num(u), np.nan_to_num(v)
    fluid = np.isfinite(u)

    def at(p):
        fx = np.clip((p[:, 0] - x0) / (x1 - x0) * (nx - 1), 0, nx - 1.001)
        fy = np.clip((p[:, 1] - y0) / (y1 - y0) * (ny - 1), 0, ny - 1.001)
        i, j = fx.astype(int), fy.astype(int)
        a, b = fx - i, fy - j
        w = [(1 - a) * (1 - b), a * (1 - b), (1 - a) * b, a * b]
        cells = [(j, i), (j, i + 1), (j + 1, i), (j + 1, i + 1)]
        vel = sum(wk[:, None] * np.c_[U[c], V[c]] for wk, c in zip(w, cells))
        inside = fluid[j, i] & (p[:, 0] >= x0) & (p[:, 0] <= x1) & (p[:, 1] >= y0) & (p[:, 1] <= y1)
        return vel, inside

    inflow = np.nanmean(U[:, 0]) >= 0
    xs = x0 if inflow else x1
    good = np.nonzero(fluid[:, 0 if inflow else -1])[0]
    ys = np.linspace(y0, y1, ny)[good] if len(good) else np.linspace(y0, y1, ny)

    def spawn(k):
        return np.c_[np.full(k, xs), rng.choice(ys, k) + rng.uniform(-0.5, 0.5, k) * (y1 - y0) / ny]

    p = np.c_[rng.uniform(x0, x1, n), rng.uniform(y0, y1, n)]
    _, ok = at(p)
    p[~ok] = spawn(int((~ok).sum()))
    sub = 4
    out_p, out_s = [], []
    for _ in range(frames):
        for _ in range(sub):
            v1, _ = at(p)
            v2, ok = at(p + v1 * dt / 2)
            p = p + v2 * dt
            dead = ~ok
            if dead.any():
                p[dead] = spawn(int(dead.sum()))
        vel, _ = at(p)
        out_p.append(p.copy())
        out_s.append(np.hypot(vel[:, 0], vel[:, 1]))
    return np.asarray(out_p), np.asarray(out_s), dt * sub


def build_flow(out: Path, pb: dict) -> Path | None:
    fields = pb.get("fields") or []
    if not fields:
        return None
    o = read(out, "outcome.json") or {}
    f = fields[0]
    g = f["grid"]
    shape = (g["ny"], g["nx"])
    k = 1000.0 if g.get("unit") == "m" else 1.0
    comps = f["components"]
    if "u" not in comps or "v" not in comps:
        return None
    u, v = unpack_flow(comps["u"], shape), unpack_flow(comps["v"], shape)
    x0, x1, y0, y1 = (g[a] * k for a in ("x0", "x1", "y0", "y1"))
    speed = np.hypot(u, v)
    s = Scene(pb.get("title") or "The flow", domain="flow", question=o.get("question", ""), length="mm",
              source="The slice is colored by the run's own velocity. Particles ride that velocity; how many there "
                     "are and where they enter is chosen for the eye.")
    P_, S_, dt = particles(u * k, v * k, x0, x1, y0, y1)
    s.timeline(np.arange(len(P_)) * dt, unit="s", label="time", seconds=10, between="interpolated")
    plane(s, x0, x1, y0, y1, speed, "speed", comps["u"].get("unit", "m/s"), "blues")
    solid = ~np.isfinite(u)
    if solid.any():
        cx, cy = np.meshgrid(np.linspace(x0, x1, shape[1]), np.linspace(y0, y1, shape[0]))
        h = (y1 - y0) / shape[0]
        pts = np.c_[cx[solid], cy[solid], np.zeros(int(solid.sum()))]
        s.points(pts, color="#9aa0a6", size=h * 0.6, name="solid")
    zf = (x1 - x0) * 0.004
    s.points(np.concatenate([P_, np.full(P_.shape[:2] + (1,), zf)], axis=2), values=S_ / k, field="particle speed",
             unit="m/s", colormap="inferno", size=(y1 - y0) / shape[0] * 0.5, name="particles")
    if pb.get("mesh"):
        from .life import arr
        V, F = arr(pb["mesh"]["vertices"]), arr(pb["mesh"]["triangles"]).astype(int)
        surf = s.mesh(V, F, name="part", color="#b9bdc4", opacity=0.9)
        sv = next(iter((pb.get("surface") or {}).items()), None)
        if sv:
            surf.color_by(sv[0].replace("_", " "), arr(sv[1]), unit=sv[1].get("unit", ""), colormap="inferno")
    pl = f.get("plane")
    if pl:
        o_, u_, v_ = (np.asarray(pl[k], float) for k in ("origin", "u", "v"))
        R = np.c_[u_, v_, np.cross(u_, v_)]
        for ob in s.objects:
            for attr in ("X", "P"):
                a_ = getattr(ob, attr, None)
                if isinstance(a_, np.ndarray) and ob.name in ("speed", "particles", "solid"):
                    setattr(ob, attr, a_ @ R.T + o_ * k)
    for name, sc in list(pb.get("scalars", {}).items())[:4]:
        s.readout(name.replace("_", " "), np.full(len(P_), sc["value"]), unit=sc.get("unit", ""), digits=4)
    verdict(s, o)
    if not pl:
        s.camera(target=[(x0 + x1) / 2, (y0 + y1) / 2, 0], direction=(0.1, -0.9, 1.1),
                 distance=max(x1 - x0, y1 - y0) * 1.1)
    return s.save(out)
