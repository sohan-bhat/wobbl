"""A heat run as a scene the designer drives: drag the heat load and the part's temperatures follow.

fields.npz, which every heat run writes, holds the mesh and the temperatures: the steady field, or the history of a
transient run. Conduction with convection to fixed air is linear, so the rise above the air at any power is exactly
that share of the solved rise. With radiation or held temperatures it is not, so the heat domain solves the model at
a few powers as well and keeps them in fields.npz, and the view blends between those. A part cooled by a fluid
scales exactly about the fluid's inlet temperature, since the fluid's own warming grows with the power too.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .. import props as P
from ..scene import Scene
from .common import read, verdict



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


EXACT_POINTS = 9          # frames of an exact power sweep, linear in between, the asked power added


def _films(st: dict) -> list[dict]:
    f = st.get("films") or []
    if isinstance(f, str):                       # older study files kept the list as its text
        import ast
        try:
            f = ast.literal_eval(f)
        except (ValueError, SyntaxError):
            f = []
    return [x for x in f if isinstance(x, dict)]


def linear(st: dict) -> bool:
    """Whether the study's rise above its air is exactly proportional to its power, as the heat domain decides."""
    amb = float(st.get("ambient", 25.0) or 25.0)
    init = st.get("initial")
    return (not st.get("emissivity") and not st.get("fixed")
            and all(abs(float(f.get("t_inf", amb)) - amb) < 1e-9 for f in _films(st))
            and (not st.get("duration") or init is None or abs(float(init) - amb) < 1e-9))


def powers(p0: float, t_max: float, ref: float, limit) -> np.ndarray:
    """Multiples of the asked power for an exact sweep: none to twice the power, or on to where the hottest point
    clearly passes the limit, at most five times, with the asked power itself among them."""
    rise = t_max - ref
    top = 2.0 if limit is None or rise <= 0 else max(2.0, min(1.3 * (float(limit) - ref) / rise, 5.0))
    return np.unique(np.round(np.r_[np.linspace(0.0, top, EXACT_POINTS), 1.0], 6))


def crossing(x: np.ndarray, y: np.ndarray, level) -> float | None:
    """The first x where y rises through level, by linear interpolation, or None."""
    if level is None:
        return None
    above = np.nonzero(np.asarray(y) > level)[0]
    if not len(above):
        return None
    i = int(above[0])
    if i == 0:
        return float(x[0])
    return float(x[i - 1] + (level - y[i - 1]) / (y[i] - y[i - 1]) * (x[i] - x[i - 1]))


def build(out: Path) -> Path | None:
    """A heat run as a part the designer turns the heat load up and down on. Each power shows its steady field, or
    for a transient the field at the end of the run: scaled exactly when the model is linear in its power, else
    the run's own solves at a few powers, blended in between."""
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
    p0 = float(st.get("power", 0) or 0) / 1000
    transient = float(st.get("duration") or 0) > 0 and len(ht) > 0 and len(hT) == len(ht)
    hot0 = np.asarray(hT, float).max(1) if transient else None
    cells = _cells(X, st)
    if "sweep_W" in d.files:
        W = np.array([float(f"{w:.5g}") for w in d["sweep_W"]])
        F = np.asarray(d["sweep_T"], float)
        H = np.asarray(d["sweep_hot"], float) if transient and "sweep_hot" in d.files else None
        between = "interpolated"
        how = (f"Radiation or held temperatures make the result depend on the power in a nonlinear way, so the run "
               f"solved the model at {len(W)} powers and the view blends between them.")
    elif p0 > 0 and linear(st):
        k = powers(p0, float(T.max()), amb, limit)
        W, F = k * p0, amb + np.outer(k, T - amb)
        H = amb + np.outer(k, hot0 - amb) if transient else None
        between = "exact"
        how = "The rise above the air is exactly proportional to the power in this linear model, so every power is exact."
    else:
        W = F = H = None
    end = f" after {float(ht[-1]) / 60:.3g} min" if transient else ""
    what = f"The {len(cells)} cell pack" if cells else "The part"
    s = Scene(f"{what}{end} at each heat load" if W is not None else f"{what}{end}", domain="thermal",
              question=o.get("question", ""), length="mm")
    hot = int(np.argmax(T))
    m = s.mesh(X, tris, name="part", keep=[hot], opacity=0.22 if cells else 1.0)
    if W is None:
        s.note("The run has no power to turn up and down, so this is its result alone.")
        F, W = T[None, :], np.array([p0])
    else:
        s.timeline(W, unit="W", label="heat load", kind="input", design=p0, between=between)
        s.note(how + (f" Each power shows the part at the end of its {float(ht[-1]) / 60:.3g} min run."
                      if transient else ""))
    rng = (min(float(F.min()), amb), max(float(F.max()), amb + 1e-6))
    m.color_by("temperature", F, unit="C", colormap="inferno", range=rng, limit=limit, limit_label="limit")
    for k_, (c, ax, r, half, inside) in enumerate(cells):
        s.body("cylinder", [r, half], pos=c, quat=P.z_to(np.eye(3)[ax]), color="#5b6470",
               name=f"cell {k_ + 1}").color_by("temperature", F[:, inside].mean(1), unit="C", colormap="inferno",
                                              range=rng, limit=limit, limit_label="limit")
    if cells:
        s.note(f"The block is {len(cells)} cells treated as one solid. Each cell is drawn at its place in the "
               "block and colored by the solved temperature averaged over its volume.")
    s.readout("hottest" + end, F.max(1), unit="C", digits=4, limit=limit)
    s.readout("average" + end, F.mean(1), unit="C", digits=4)
    if not cells:
        still_air(s, st, X, tris, T, amb, F)
    if transient and H is not None and limit is not None:
        t_min = np.asarray(ht, float) / 60
        over = [t_min[-1] - c if (c := crossing(t_min, h, limit)) is not None else 0.0 for h in H]
        s.readout(f"time over {limit:g} C", over, unit="min", digits=3)
    cross = crossing(W, F.max(1), limit) if len(W) > 1 else None
    if cross is not None and cross > 0:
        s.event(cross, f"passes {limit:g} C", f"at {cross:.3g} W", kind="fail")
        s.label("too hot here {hottest" + end + "}", X[hot], t0=cross, kind="fail")
        s.label("hottest {hottest" + end + "}", X[hot], t1=cross, kind="peak")
    else:
        s.label("hottest {hottest" + end + "}", X[hot], kind="fail" if limit and F.max() > limit else "peak")
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
        s.label("heat in", c - n * float(np.ptp(X, 0).max()) * 0.08, kind="note")
    verdict(s, o)
    return s.save(out)


STILL_AIR_H = 15.0          # W/(m2 K): a film this low on every face is natural convection, still air


def still_film(st: dict) -> float | None:
    """The film coefficient (W/(m2 K)) of a part in still air: one film on every face at the ambient, low enough to
    be natural convection. None otherwise (a fan, water, several films)."""
    f = _films(st)
    amb = float(st.get("ambient", 25.0) or 25.0)
    if len(f) != 1 or str(f[0].get("faces", "")).strip() != "all" or abs(float(f[0].get("t_inf", amb)) - amb) > 1e-9:
        return None
    h = float(f[0].get("h", 0)) * 1000.0                    # the study keeps W/(mm2 K) x 1e3, as mW/(mm2 K)
    return h if 0 < h <= STILL_AIR_H else None


def still_air(s, st: dict, X, tris, T, amb: float, F) -> dict | None:
    """The air a part in still air warms, estimated from its film and solved surface (thermal/plume.py), as lines
    the air rises along with its speed and its temperature at each point. The view says it is estimated. At each
    heat load the air's warming follows the plume's law, dT as the convected heat to the 2/3, the heat being the
    film on that load's surface."""
    h = still_film(st)
    if h is None:
        return None
    from ...thermal.plume import plume
    pl = plume(X, tris, T, h, amb, lines=48)
    if pl is None:
        return None
    rise0 = float(T.mean() - amb)
    k = np.clip((F.mean(1) - amb) / rise0, 0, None) if rise0 > 0 else np.ones(len(F))
    # cool to warm on the air's own range, its rise above the room up to twice the asked load's
    ln = s.lines(pl["paths"], color="#8fb8de", width=0.6, values=[t - amb for t in pl["temp"]], field="air warming",
                 unit="C", colormap="coolwarm", range=(0.0, 2 * pl["dT_K"]), name="air")
    ln.moving(pl["speed"], scale=k ** (2 / 3), base=amb)
    s.note(f"The air is estimated, not solved. About {pl['w_mps']:.2g} m/s rising out of the top, "
           f"{pl['dT_K']:.2g} C warmer, carrying the {pl['Q_W']:.2g} W the film takes, from a chimney through "
           "the open columns and an entraining plume above (thermal/plume.py).")
    return pl


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
    inlet = float((sc.get("inlet_air") or {}).get("value", 25.0))
    power = float((sc.get("power") or {}).get("value", 0) or 0)
    limit = (o.get("values") or {}).get("limit_C")
    s = Scene("The heat sink in its fan's air at each heat load", domain="thermal", question=o.get("question", ""),
              length="mm",
              source="The sink's temperatures are the CalculiX solution with the films the gap's CFD gave, and the "
                     "streamlines trace the CFD's velocity through one gap, repeated in every gap.")
    hot = int(np.argmax(T))
    m = s.mesh(X, tris, name="heat sink", keep=[hot])
    F = _input_power(s, m, T, inlet, power, limit, "air")
    s.readout("airflow", np.full(s.n, float((sc.get("airflow") or {}).get("value", 0))), unit="L/min", digits=4)
    _hot_label(s, X[hot], F, limit)
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
        rise = float((sc.get("fluid_rise") or sc.get("air_rise") or {}).get("value", 0) or 0)
        fluid_lines(s, list(zip(paths, speeds)), X, tris, T, inlet, rise, power, "air", AIR, 1.2)
        s.readout("fastest air", np.full(s.n, top), unit="m/s", digits=3)
    r = min(g["width"], height + base) * 0.45
    s.body("cylinder", [r, r * 0.12], pos=[g["width"] / 2, -r * 0.5, base + height / 2], quat=P.z_to([0, 1, 0]),
           color="#2c2f34", name="fan", prop=True)
    s.label("heat in", [g["width"] / 2, g["length"] / 2, -0.15 * g["width"]], kind="note")
    verdict(s, o, _first(o))
    size = float(np.ptp(X, 0).max())
    s.camera(target=X.mean(0), direction=(1.0, -0.8, 0.9), distance=size * 2.4)
    return s.save(out)


def cooled(out: Path) -> Path | None:
    """A part cooled by a fluid run through it, as a scene: the part see through and colored by its solved
    temperature, with streamlines traced through the fluid's solved velocity in 3D from the inlet and through the
    fluid around the part, or in the plane through its middle for a run that kept only that."""
    out = Path(out)
    p = next((q for q in (out / "heat" / "fields.npz", out / "fields.npz") if q.exists()), None)
    o = read(out, "outcome.json") or {}
    pb = read(out, "playback.json") or {}
    if p is None or not pb.get("fields"):
        return None
    from ...flow.scene import Plane, _seeds, streamlines, trace
    d = np.load(p)
    X, tris, T = d["X"], d["tris"], d["T"]
    limit = (o.get("values") or {}).get("limit_C")
    hot = int(np.argmax(T))
    st = read(p.parent, "study.json") or {}
    sc = pb.get("scalars") or {}
    inlet = float(st.get("ambient", float(T.min())))
    power = float((sc.get("power") or {}).get("value", 0) or 0) or float(st.get("power", 0) or 0) / 1000
    s = Scene("The part and the fluid cooling it at each heat load", domain="thermal", question=o.get("question", ""),
              length="mm",
              source="The part's temperatures are the CalculiX solution with the films the CFD gave, and the "
                     "streamlines trace the CFD's velocity in the plane through the middle of the flow.")
    m = s.mesh(X, tris, name="part", keep=[hot], opacity=0.55)
    F = _input_power(s, m, T, inlet, power, limit, "fluid")
    for name, v in list(sc.items())[:3]:
        if name not in ("hottest",):
            s.readout(name.replace("_", " "), np.full(s.n, v["value"]), unit=v.get("unit", ""), digits=4)
    _hot_label(s, X[hot], F, limit)
    f = pb["fields"][0]
    pl = Plane(f, 1000.0 if f["grid"].get("unit") == "m" else 1.0)
    fluid = np.argwhere(np.isfinite(pl.u) & (np.hypot(pl.u, pl.v) > 0))
    pick = fluid[np.linspace(0, len(fluid) - 1, min(36, len(fluid))).astype(int)] if len(fluid) else fluid
    seeds = np.c_[pl.x0 + pick[:, 1] / (pl.nx - 1) * (pl.x1 - pl.x0), pl.y0 + pick[:, 0] / (pl.ny - 1) * (pl.y1 - pl.y0)]
    lines = streamlines(pb, around=(X.min(0), X.max(0)))
    if lines:
        s.head["source"] = s.head["source"].replace("in the plane through the middle of the flow",
                                                    "in 3D from the inlet and through the fluid around the part")
    else:
        lines = trace(pl, np.vstack([_seeds(pl, 12), seeds]), steps=600)
    if not lines:                        # nothing to see through to, so the part is drawn solid
        m.opacity = 1.0
        s.note("The flow holds too little fluid to trace streamlines, so the part shows alone.")
    if lines:
        rise = float((sc.get("fluid_rise") or {}).get("value", 0) or 0)
        fluid_lines(s, lines, X, tris, T, inlet, rise, power, "fluid", FLUID, 1.6)
        s.readout("fastest fluid", np.full(s.n, float(max(w.max() for _, w in lines))), unit="m/s", digits=3)
    verdict(s, o, _first(o))
    size = float(np.ptp(X, 0).max())
    s.camera(target=X.mean(0), direction=(0.4, -0.8, 1.0), distance=size * 2.6)
    return s.save(out)


AIR, FLUID = "#8fb8de", "#5aa0e0"


def fluid_lines(s, lines, X, tris, T, inlet: float, rise: float, power: float, name: str, color: str, width: float):
    """A cooling fluid's streamlines (each a path in mm and the solved speed along it in m/s) as lines it moves
    along at that speed, colored cool to warm by how much it has warmed. The run reports the fluid's warming at the
    outlet, rise, which is its energy balance (power over mass flow times specific heat). Along the way the warming
    is estimated as that share of the outlet rise the part has handed over upstream of the point: the part's faces
    weighted by area and by their solved rise above the inlet, summed along the main flow direction. The warming
    grows with the power, as everything in the run does, so each heat load scales it."""
    paths = [np.asarray(p, float) for p, _ in lines]
    speeds = [np.asarray(w, float) * 1000.0 for _, w in lines]
    s.lines(paths, color=color, name=name, width=width)
    ln = s.objects[-1]
    if rise > 0 and len(paths):
        ends = np.array([p[-1] - p[0] for p in paths if len(p) > 1])
        ax = ends.sum(0)
        ax = ax / (np.linalg.norm(ax) or 1.0)
        t = np.asarray(tris)[:, :3]
        a, b, c = X[t[:, 0]], X[t[:, 1]], X[t[:, 2]]
        w = 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1) * np.clip(T[t].mean(1) - inlet, 0, None)
        x = ((a + b + c) / 3) @ ax
        o = np.argsort(x)
        cum = np.cumsum(w[o]) / max(w.sum(), 1e-30)
        vals = [rise * np.interp(p @ ax, x[o], cum, left=0.0, right=1.0) for p in paths]
        f = s.field(f"{name} warming", "C", colormap="coolwarm", range=(0.0, 2 * rise),
                    data=np.concatenate(vals))
        ln.values, ln.f = vals, f
        k = s.t / power if power > 0 and s.n > 1 else np.ones(s.n)
        ln.moving(speeds, scale=k, base=0.0)
        s.note(f"The {name} moves at its solved speed. Its warming along the way is an estimate, the outlet's "
               f"{rise:.3g} C rise shared out by where the part hands its heat over.")
    else:
        ln.moving(speeds)
    return ln


def _input_power(s, m, T, inlet: float, power: float, limit, what: str) -> np.ndarray:
    """The heat load as the slider of a part cooled by a moving fluid, and the part colored at each power. The
    films' air or water warms in proportion to the power too, so the rise above the inlet is exactly proportional
    to the power and every power is exact. Returns the field at each frame."""
    if power > 0:
        k = powers(power, float(T.max()), inlet, limit)
        s.timeline(k * power, unit="W", label="heat load", kind="input", design=power, between="exact")
        s.note(f"The {what} keeps its speeds at any heat load, and the part's rise above the {what}'s inlet "
               "temperature is exactly proportional to the power, so every power is exact.")
        F = inlet + np.outer(k, T - inlet)
    else:
        F = T[None, :]
    m.color_by("temperature", F, unit="C", colormap="inferno", range=(min(inlet, float(F.min())), float(F.max())),
               limit=limit, limit_label="limit")
    s.readout("hottest", F.max(1), unit="C", digits=4, limit=limit)
    return F


def _hot_label(s, at, F, limit):
    W = s.t
    cross = crossing(W, F.max(1), limit) if len(W) > 1 else None
    if cross is not None and cross > 0:
        s.event(cross, f"passes {limit:g} C", f"at {cross:.3g} W", kind="fail")
        s.label("too hot here {hottest}", at, t0=cross, kind="fail")
        s.label("hottest {hottest}", at, t1=cross, kind="peak")
    else:
        s.label("hottest {hottest}", at, kind="fail" if limit and F.max() > limit else "peak")


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
