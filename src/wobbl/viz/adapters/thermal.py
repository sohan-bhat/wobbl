"""A heat run as a scene: the part warming over time, or coming up to its steady temperatures as the power comes on.

fields.npz, which every heat run writes, holds the mesh and the temperatures: the history of a transient run, or
the steady field. Steady conduction with convection to fixed air is linear, so the rise above the air at a share of
the power is exactly that share of the solved rise, and the view ramps the power with it. With radiation or held
temperatures that is no longer exact, so the view then shows the steady result alone.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .. import props as P
from ..scene import Scene
from .common import read, verdict

FRAMES = 40


def _faces(X, tris, selector: str):
    """Boundary triangles a plain bounding box selector names (top, bottom, x=min and so on), else None."""
    sel = (selector or "").split(" and ")[0].strip().lower()
    alias = {"top": "z=max", "bottom": "z=min"}
    sel = alias.get(sel, sel)
    if len(sel) != 5 or sel[1] != "=" or sel[0] not in "xyz" or sel[2:] not in ("min", "max"):
        return None
    ax = "xyz".index(sel[0])
    t = tris[:, :3]
    c = X[t].mean(1)
    edge = X[:, ax].min() if sel.endswith("min") else X[:, ax].max()
    tol = 0.01 * float(np.ptp(X, 0).max())
    m = np.abs(c[:, ax] - edge) < tol
    import re
    box = re.search(r"box\s+([-\d.eE,\s]+)", selector or "")
    if box:
        v = [float(x) for x in box.group(1).replace(" ", "").split(",") if x][:6]
        if len(v) == 6:
            m &= ((c >= v[:3]) & (c <= v[3:])).all(1)
    return (t[m], ax, edge, sel.endswith("min")) if m.any() else None


def build(out: Path) -> Path | None:
    p = Path(out) / "fields.npz"
    o = read(out, "outcome.json") or {}
    if not p.exists() or o.get("domain") != "thermal":
        return None
    d = np.load(p)
    st = read(out, "study.json") or {}
    X, tris, T = d["X"], d["tris"], d["T"]
    ht, hT = d["hist_t"], d["hist_T"]
    amb = float(st.get("ambient", 25.0) or 25.0)
    limit = (o.get("values") or {}).get("limit_C")
    s = Scene("The part heating up", domain="thermal", question=o.get("question", ""), length="mm")
    tmax = float(T.max())
    hot = X[int(np.argmax(T))]
    if len(ht) and len(hT) == len(ht):
        t = np.concatenate([[0.0], np.asarray(ht, float)]) / 60
        start = float(st.get("initial", amb) or amb)
        H = np.vstack([np.full((1, len(X)), start), np.asarray(hT, float)])
        s.timeline(t, unit="min", label="time", seconds=8, between="interpolated")
        cells = _cells(X, st)
        if cells:
            s.head["title"] = f"The {len(cells)} cell pack heating up"
        m = s.mesh(X, tris, name="part", keep=[int(np.argmax(T))], opacity=0.22 if cells else 1.0)
        rng = (min(H.min(), amb), max(tmax, limit or tmax))
        m.color_by("temperature", H, unit="C", colormap="inferno", range=rng, limit=limit, limit_label="limit")
        for k, (c, ax, r, half, inside) in enumerate(cells):
            s.body("cylinder", [r, half], pos=c, quat=P.z_to(np.eye(3)[ax]), color="#5b6470",
                   name=f"cell {k + 1}").color_by("temperature", H[:, inside].mean(1), unit="C", colormap="inferno",
                                                 range=rng, limit=limit, limit_label="limit")
        if cells:
            s.note(f"The block is {len(cells)} cells treated as one solid. Each cell is drawn at its place in the "
                   "block and colored by the solved temperature averaged over its volume.")
        s.readout("hottest", H.max(1), unit="C", digits=4, limit=limit)
        s.readout("average", H.mean(1), unit="C", digits=4)
        over = (o.get("values") or {}).get("time_over_limit_s")
        if over:
            s.event(over / 60, f"passes {limit:g} C", "", kind="fail")
            s.label("too hot here", hot, t0=over / 60, kind="fail")
        else:
            s.label("hottest {hottest}", hot, kind="peak")
    else:
        linear = not st.get("emissivity") and not st.get("fixed")
        m = s.mesh(X, tris, name="part", keep=[int(np.argmax(T))])
        if linear:
            ramp = np.linspace(0, 1, FRAMES)
            power = float(st.get("power", 0) or 0)
            s.timeline(ramp * (power / 1000 if power else 100), unit="W" if power else "%", label="power", seconds=6,
                       between="exact", note="The rise above the air is exactly proportional to the power in this "
                                             "linear model, so every frame of the ramp is a steady result.")
            m.color_by("temperature", np.outer(ramp, T - amb) + amb, unit="C", colormap="inferno",
                       range=(amb, max(tmax, limit or tmax)), limit=limit, limit_label="limit")
            s.readout("hottest", amb + ramp * (tmax - amb), unit="C", digits=4, limit=limit)
        else:
            s.note("Radiation or held temperatures make the result depend on the power in a nonlinear way, so this "
                   "is the steady result alone.")
            m.color_by("temperature", T, unit="C", colormap="inferno", range=(amb, max(tmax, limit or tmax)),
                       limit=limit, limit_label="limit")
            s.readout("hottest", [tmax], unit="C", digits=4, limit=limit)
        s.label("hottest {hottest}", hot, kind="fail" if limit and tmax > limit else "peak")
    heat = _faces(X, tris, st.get("heat_faces", ""))
    if heat is not None:
        t_, ax, edge, low = heat
        pts = X[np.unique(t_)]
        c = pts.mean(0)
        n = np.zeros(3)
        n[ax] = 1 if low else -1
        half = np.maximum(np.ptp(pts, 0) / 2, 1.0)
        u = [i for i in range(3) if i != ax]
        P.slab(s, c, n, half[u[0]], half[u[1]], float(np.ptp(X, 0).max()) * 0.04, color="#2c2f34", name="heat source")
        s.label(f"{float(st.get('power', 0)) / 1000:.3g} W in" if st.get("power") else "heat in",
                c - n * float(np.ptp(X, 0).max()) * 0.08, kind="note")
    verdict(s, o)
    return s.save(out)


def fan_sink(out: Path) -> Path | None:
    """A fan cooled heat sink as a scene: the sink colored by its solved temperature, with the air through every
    gap between its fins traced from the gap's own CFD, mirrored about the middle of the gap and repeated at a few
    heights, since the run solves one gap and every gap sees the same flow."""
    out = Path(out)
    p = out / "fields.npz"
    o = read(out, "outcome.json") or {}
    pb = read(out, "playback.json") or {}
    g = (o.get("values") or {}).get("sink_mm")
    if not p.exists() or not g or pb.get("kind") != "channel" or not pb.get("fields"):
        return None
    from ...flow.scene import Plane, trace
    d = np.load(p)
    X, tris, T = d["X"], d["tris"], d["T"]
    sc = pb.get("scalars") or {}
    air = float((sc.get("inlet_air") or {}).get("value", 25.0)) + float((sc.get("air_rise") or {}).get("value", 0)) / 2
    power = float((sc.get("power") or {}).get("value", 0) or 0)
    limit = (o.get("values") or {}).get("limit_C")
    tmax = float(T.max())
    s = Scene("The heat sink in its fan's air", domain="thermal", question=o.get("question", ""), length="mm",
              source="The sink's temperatures are the CalculiX solution with the films the gap's CFD gave, and the "
                     "streamlines trace the CFD's velocity through one gap, repeated in every gap.")
    ramp = np.linspace(0, 1, FRAMES)
    s.timeline(ramp * (power or 100), unit="W" if power else "%", label="power", seconds=6, between="exact",
               note="The rise above the air is exactly proportional to the power with fixed films, so every frame of "
                    "the ramp is a steady result.")
    hot = int(np.argmax(T))
    m = s.mesh(X, tris, name="heat sink", keep=[hot])
    m.color_by("temperature", np.outer(ramp, T - air) + air, unit="C", colormap="inferno",
               range=(air, max(tmax, limit or tmax)), limit=limit, limit_label="limit")
    s.readout("hottest", air + ramp * (tmax - air), unit="C", digits=4, limit=limit)
    s.readout("airflow", np.full(FRAMES, float((sc.get("airflow") or {}).get("value", 0))), unit="L/min", digits=4)
    s.label("hottest {hottest}", X[hot], kind="fail" if limit and tmax > limit else "peak")
    pl = Plane(pb["fields"][0], 1000.0 if pb["fields"][0]["grid"].get("unit") == "m" else 1.0)
    half = pl.y1 - pl.y0
    seeds = np.c_[np.full(3, pl.x0 + 0.01 * (pl.x1 - pl.x0)), pl.y0 + half * np.array([0.25, 0.55, 0.9])]
    lines = trace(pl, seeds)
    paths, speeds = [], []
    base, height, fin, pitch = g["base"], g["height"], g["fin"], g["pitch"]
    for i in range(int(g["fins"]) - 1):
        left, right = i * pitch + fin, (i + 1) * pitch
        for z in base + height * np.array([0.2, 0.5, 0.8]):
            for path, sp in lines:
                gx, gy = path[:, 0], path[:, 1]          # along the flow, and out from the fin wall
                for xw in (left + (gy - pl.y0), right - (gy - pl.y0)):
                    paths.append(np.c_[xw, gx, np.full(len(gx), z)])
                    speeds.append(sp)
    if paths:
        top = float(max(w.max() for w in speeds))
        s.lines(paths, values=speeds, field="air speed", unit="m/s", colormap="blues", range=(0.0, top),
                flow=np.array([float(w.mean()) for w in speeds]) / top * g["length"] / 2.5,
                spacing=g["length"] / 10, name="air", width=1.2)
    r = min(g["width"], height + base) * 0.45
    s.body("cylinder", [r, r * 0.12], pos=[g["width"] / 2, -r * 0.5, base + height / 2], quat=P.z_to([0, 1, 0]),
           color="#2c2f34", name="fan", prop=True)
    if power:
        s.label(f"{power:.3g} W in", [g["width"] / 2, g["length"] / 2, -0.15 * g["width"]], kind="note")
    verdict(s, o, _first(o))
    size = float(np.ptp(X, 0).max())
    s.camera(target=X.mean(0), direction=(1.0, -0.8, 0.9), distance=size * 2.4)
    return s.save(out)


def cooled(out: Path) -> Path | None:
    """A part cooled by a fluid run through it, as a scene: the part see through and colored by its solved
    temperature, with streamlines traced through the fluid's solved velocity in the plane through its middle."""
    out = Path(out)
    p = next((q for q in (out / "heat" / "fields.npz", out / "fields.npz") if q.exists()), None)
    o = read(out, "outcome.json") or {}
    pb = read(out, "playback.json") or {}
    if p is None or not pb.get("fields"):
        return None
    from ...flow.scene import Plane, _seeds, trace
    d = np.load(p)
    X, tris, T = d["X"], d["tris"], d["T"]
    limit = (o.get("values") or {}).get("limit_C")
    tmax = float(T.max())
    hot = int(np.argmax(T))
    s = Scene("The part and the fluid cooling it", domain="thermal", question=o.get("question", ""), length="mm",
              source="The part's temperatures are the CalculiX solution with the films the CFD gave, and the "
                     "streamlines trace the CFD's velocity in the plane through the middle of the flow.")
    m = s.mesh(X, tris, name="part", keep=[hot], opacity=0.55)
    m.color_by("temperature", T, unit="C", colormap="inferno", range=(float(T.min()), tmax), limit=limit,
               limit_label="limit")
    s.readout("hottest", [tmax], unit="C", digits=4, limit=limit)
    for name, sc in list((pb.get("scalars") or {}).items())[:3]:
        if name not in ("hottest",):
            s.readout(name.replace("_", " "), [sc["value"]], unit=sc.get("unit", ""), digits=4)
    s.label("hottest {hottest}", X[hot], kind="fail" if limit and tmax > limit else "peak")
    f = pb["fields"][0]
    pl = Plane(f, 1000.0 if f["grid"].get("unit") == "m" else 1.0)
    fluid = np.argwhere(np.isfinite(pl.u) & (np.hypot(pl.u, pl.v) > 0))
    pick = fluid[np.linspace(0, len(fluid) - 1, min(36, len(fluid))).astype(int)] if len(fluid) else fluid
    seeds = np.c_[pl.x0 + pick[:, 1] / (pl.nx - 1) * (pl.x1 - pl.x0), pl.y0 + pick[:, 0] / (pl.ny - 1) * (pl.y1 - pl.y0)]
    lines = trace(pl, np.vstack([_seeds(pl, 12), seeds]), steps=600)
    if lines:
        top = float(max(w.max() for _, w in lines))
        span = pl.x1 - pl.x0
        s.lines([q for q, _ in lines], values=[w for _, w in lines], field="fluid speed", unit="m/s",
                colormap="blues", range=(0.0, top), flow=np.array([float(w.mean()) for _, w in lines]) / top * span / 3,
                spacing=span / 14, name="fluid", width=1.6)
    verdict(s, o, _first(o))
    size = float(np.ptp(X, 0).max())
    s.camera(target=X.mean(0), direction=(0.4, -0.8, 1.0), distance=size * 2.2)
    return s.save(out)


def _first(o: dict) -> str:
    """The headline's first sentence as it stands, since its units carry brackets, as W/(m2 K)."""
    import re
    return re.split(r"(?<=[.!?])\s", (o.get("headline") or "").strip())[0]


CELL_D = {"18650": (18.4, 65.0), "21700": (21.2, 70.0), "26650": (26.5, 65.0)}


def _cells(X, st: dict) -> list:
    """Cylindrical cells in a block a battery study treats as one solid: the largest standard cells whose length
    matches the block's longest side, packed side by side across it. Each is (center, axis, radius, half length,
    the block's nodes inside it)."""
    if "cell" not in str(st.get("material", "")).lower():
        return []
    lo, hi = X.min(0), X.max(0)
    ext = hi - lo
    ax = int(np.argmax(ext))
    others = [i for i in range(3) if i != ax]
    fits = [(int(ext[others[0]] // d) * int(ext[others[1]] // d) * d * d, d) for d, length in CELL_D.values()
             if abs(ext[ax] - length) <= 0.15 * length and min(ext[others]) >= d]
    if not fits:
        return []
    d = max(fits)[1]                              # the cell that fills the block's section best
    n = [int(ext[i] // d) for i in others]
    out = []
    for i in range(n[0]):
        for j in range(n[1]):
            c = (lo + hi) / 2
            c[others[0]] = lo[others[0]] + ext[others[0]] * (i + 0.5) / n[0]
            c[others[1]] = lo[others[1]] + ext[others[1]] * (j + 0.5) / n[1]
            off = X - c
            off[:, ax] = 0
            inside = np.nonzero(np.linalg.norm(off, axis=1) <= d / 2)[0]
            if len(inside):
                out.append((c, ax, d / 2 * 0.97, ext[ax] / 2 * 0.97, inside))
    return out
