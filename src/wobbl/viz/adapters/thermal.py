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
        m = s.mesh(X, tris, name="part", keep=[int(np.argmax(T))])
        m.color_by("temperature", H, unit="C", colormap="inferno", range=(min(H.min(), amb), max(tmax, limit or tmax)),
                   limit=limit, limit_label="limit")
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
