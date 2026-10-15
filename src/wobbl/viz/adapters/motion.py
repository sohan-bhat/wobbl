"""A motion run as a 3D model you turn by hand: the whole flight at real scale for each value of one input.

The input is the one a designer would change for the question, chosen by wobbl.motion.sweep: the wind for a rocket,
the drop height for a dropped body, the launch speed or angle for a thrown one. Every value on the slider is its
own flight of the run's model, so each frame is a complete result, the full path from release or the pad to the
ground with its apogee and landing spot, and nothing plays on its own. A rocket rests at its apogee, drawn from its
spec at its real size, with the drogue and main legs of its descent and the spot it lands on colored by landing
speed against the soft landing limit. A dropped body rests where it lands beside the cliff it fell from, which
rises with the slider, its path colored by speed against its terminal speed.
"""
from __future__ import annotations

import csv
import re
from pathlib import Path

import numpy as np

from .. import props as P
from .. import props_flight as F
from ..scene import Scene
from .common import quats_along, read, verdict, words

ROCK, BODY, NOSE, FIN = "#b8ab98", "#e9e9e4", "#ffd400", "#2c2f34"
UP, DROGUE, MAIN, PLUMB = "#ffd400", "#ffb020", "#ff6b3d", "#7d838c"
TALL = ("cliff", "building", "tower", "roof", "bridge", "balcony", "ledge", "skyscraper")


def rows(out: Path) -> dict | None:
    p = Path(out) / "trajectory.csv"
    if not p.exists():
        return None
    with p.open() as f:
        r = list(csv.DictReader(f))
    if not r:
        return None
    cols = {k: np.array([float(x[k]) for x in r]) for k in r[0] if k != "phase"}
    cols["phase"] = np.array([x.get("phase", "") for x in r])
    return cols


def spec_of(out: Path, o: dict):
    part = o.get("part") or ""
    if part.endswith(".toml"):
        p = (Path(out) / part).resolve()
        if p.exists():
            from ...motion.spec import read_spec
            return read_spec(p, o.get("question", ""))
    return None


def _diameter(o: dict, st) -> float:
    """The body's size: the study's, the run's saved one, or the size its checks name."""
    d = float(getattr(st, "diameter", 0) or (o.get("values") or {}).get("diameter_m") or 0)
    if d > 0:
        return d
    for text in [c.get("summary", "") for c in o.get("checks", [])] + list(o.get("notes", [])):
        m = re.search(r"(\d+(?:\.\d+)?)\s*(mm|cm|m)\s+(?:body|across)", text)
        if m:
            return float(m.group(1)) * {"mm": 1e-3, "cm": 1e-2, "m": 1.0}[m.group(2)]
    return 0.0


def build(out: Path, st=None, o: dict | None = None) -> Path | None:
    """The scene of a motion run from its folder. st, the run's MotionStudy, lets every value of the input be
    flown again; without it a rocket's spec file is read, and a run with neither shows its own one flight."""
    from ...motion import sweep

    o = o or read(out, "outcome.json")
    c = rows(out)
    if not o or c is None or o.get("domain") != "motion":
        return None
    st = st or spec_of(out, o)
    if st is not None:
        inp = sweep.choose(st)
        runs = sweep.run(st, inp)
    else:
        inp, runs = _one(c, o)
    if st is not None and getattr(st, "motor", None):
        return _rocket(out, o, st, inp, runs)
    return _body(out, o, st, c, inp, runs)


def _one(c: dict, o: dict):
    """The run's own flight as a sweep of one value, when the study is not at hand to fly any other."""
    from ...motion.sweep import Input, Point
    if len(c["t_s"]) > 2 and c["phase"][-1] == "landed":     # the table's last row is the body at rest
        c = {k: v[:-1] for k, v in c.items()}
    t = c["t_s"] + np.arange(len(c["t_s"])) * 1e-9
    h0, v0 = float(c["alt_m"][0]), float(c["speed_mps"][0])
    inp = (Input("height", "drop height", "m", 1.0, np.array([h0]), h0) if v0 == 0 else
           Input("speed", "launch speed", "m/s", 1.0, np.array([v0]), v0))
    pt = Point(inp.design, None, c["t_s"], c["x_m"], c["alt_m"], c["speed_mps"], np.gradient(c["x_m"], t),
               np.gradient(c["alt_m"], t), c["phase"])
    pt.events = {e["name"]: (e["t_s"], e["x_m"], e["alt_m"], e.get("speed_mps", 0.0)) for e in o.get("events", [])}
    v = o.get("values") or {}
    m = float(c["mass_kg"][-1]) if "mass_kg" in c else 1.0
    pt.numbers = {"flight_time_s": float(c["t_s"][-1]), "impact_speed_mps": float(v.get("impact_speed_mps", 0)),
                  "distance_m": float(c["x_m"][-1]), "max_height_m": float(c["alt_m"].max()),
                  "impact_drag_n": max(m * (9.80665 - float(c["accel_mps2"][-1])), 0.0) if "accel_mps2" in c else 0}
    return inp, [pt]


def _windows(v) -> list[tuple]:
    """The stretch of the slider each value owns, halfway to its neighbors, so exactly one flight shows."""
    v = np.asarray(v, float)
    mid = (v[1:] + v[:-1]) / 2
    return [(None if i == 0 else float(mid[i - 1]), None if i == len(v) - 1 else float(mid[i])) for i in range(len(v))]


def _timeline(s, inp, note: str):
    s.timeline(inp.values, unit=inp.unit, label=inp.label, between="step", kind="input", design=inp.design,
               note=note)


def _num(runs, k):
    return np.array([r.numbers.get(k, np.nan) for r in runs], float)


def _rocket(out, o, st, inp, runs):
    s = Scene(f"The {(st.name or 'rocket').replace('-', ' ')}'s flight against the {inp.label}", domain="motion",
              question=o.get("question", ""), length="m",
              source="Every value on the slider is its own flight of the run's model, a point mass in a vertical "
                     "plane from the pad to the ground. The rocket rests at its apogee pointing into the air it "
                     "meets there, and the spot it lands on is colored by its landing speed.")
    _timeline(s, inp, f"Each value is a full flight, flown again with the {inp.label} set to it.")
    n = len(runs)
    win = _windows(inp.values)
    af = st.airframe
    dia = float(st.diameter)
    L = float(af.length if af else dia * 20)
    cg = float(af.cg if af and af.cg else L * 0.6)
    lean = np.radians(90 - float(st.angle or 90))
    rail_dir = np.array([np.sin(lean), 0, np.cos(lean)])

    # the size of the whole picture, over every flight on the slider
    xs = np.concatenate([r.x for r in runs])
    top = float(max(r.z.max() for r in runs))
    x0, x1 = float(min(xs.min(), 0)), float(max(xs.max(), 0))
    span = max(x1 - x0, top, 1.0)
    mark = span * 0.006

    # each flight: up to apogee, under the drogue, under the main, and a plumb line from apogee to the ground
    for r, (a, b) in zip(runs, win):
        ta = r.events.get("apogee", (r.t[r.z.argmax()],))[0]
        tm = r.events.get("main deploy", (np.inf,))[0]
        P3 = np.c_[r.x, np.zeros(len(r.x)), r.z]
        for name, sel, color, w in (("climb", r.t <= ta, UP, 2.6), ("drogue descent", (r.t >= ta) & (r.t <= tm),
                                                                       DROGUE, 2.2),
                                    ("main descent", r.t >= tm, MAIN, 2.2)):
            if sel.sum() > 1:
                s.lines([P3[sel]], color=color, width=w, name=name).show(a, b)
        ax, az = r.events["apogee"][1:3] if "apogee" in r.events else (r.x[r.z.argmax()], r.z.max())
        s.lines([np.array([[ax, 0, 0], [ax, 0, az]])], color=PLUMB, width=1.0, name="under apogee",
                prop=True).show(a, b)

    # the rocket at apogee, nose into the air it meets there
    apo = np.array([[*(r.events["apogee"][1:3] if "apogee" in r.events else (r.x[r.z.argmax()], r.z.max()))]
                    for r in runs])
    pos = np.c_[apo[:, 0], np.zeros(n), apo[:, 1]]
    d = []
    for r in runs:
        k = int(np.argmin(np.abs(r.t - r.events.get("apogee", (r.t[r.z.argmax()],))[0])))
        wx = r.vx[k] + inp.study(st, r.value).wind
        d.append([wx, 0, r.vz[k]] if abs(wx) + abs(r.vz[k]) > 1e-6 else [1.0, 0, 0])
    q = quats_along(np.array(d, float))
    parts = F.airframe(L, dia, af.nose_length if af else dia * 4, af.nose_shape if af else "ogive",
                       af.fins if af else None)
    colors = {"nose": NOSE, "body tube": BODY, "fins": FIN}
    for name, (X, T) in parts.items():
        m = s.mesh(X - [0, 0, cg], T, name=name if name != "body tube" else "rocket", color=colors[name],
                   max_vertices=3000, smooth=name != "fins")
        m.move(pos, q)
    body = next(ob for ob in s.objects if ob.name == "rocket")

    # the main parachute where it opens, at its real size
    main = st.chute("main")
    if main is not None and all("main deploy" in r.events for r in runs):
        mp = np.array([[r.events["main deploy"][1], 0, r.events["main deploy"][2]] for r in runs])
        Xc, Tc = F.canopy(main.diameter, max(main.diameter * 1.3, L * 0.5))
        cm = s.mesh(Xc, Tc, name="main parachute", color=MAIN, max_vertices=900)
        cm.move(mp, (1, 0, 0, 0))
        s.label("main opens", [0, 0, main.diameter * 2], follow=cm, kind="note")

    # where it lands, colored by the landing speed against the soft landing limit
    land = np.array([[r.x[-1], 0, mark * 0.15] for r in runs])
    sink = _num(runs, "landing_speed_mps")
    spot = s.body("cylinder", [mark * 1.6, mark * 0.15], pos=land, color=MAIN, name="landing spot")
    spot.color_by("landing speed", sink, unit="m/s", colormap="turbo", range=[0, max(st.landing_limit * 2, 1)],
                  limit=st.landing_limit, limit_label="soft landing")
    s.label("lands {landing distance} out at {landing speed}", [0, 0, mark * 3], follow=spot, kind="peak")
    s.label("apogee {apogee}", [0, 0, L], follow=body, kind="peak")
    if n > 2:
        s.lines([land + [0, 0, mark * 0.05]], color="#9aa0a6", width=1.2, name="landing spots", prop=True)
        s.label(f"{inp.values[0]:g} {inp.unit}", land[0] + [0, -mark * 4, 0], kind="note")
        s.label(f"{inp.values[-1]:g} {inp.unit}", land[-1] + [0, -mark * 4, 0], kind="note")

    # the pad, the rail, the wind as an arrow that grows with the slider, a height scale and the ground
    P.pad(s, [0, 0, 0], max(dia * 6, 0.5))
    rail = float(st.rail_length or 0)
    if rail > 0:
        foot = np.array([-rail_dir[0] * (L - cg), 0, 0]) + [dia * 0.9, 0, 0]
        s.body("box", [dia * 0.12, dia * 0.12, rail / 2], pos=foot + rail_dir * rail / 2,
               quat=P.z_to(rail_dir), color=P.BOLT, name="launch rail", prop=True)
    if inp.key == "wind" and inp.values.max() > 0:
        sign = -1.0 if st.wind >= 0 else 1.0
        at = np.array([x1 + span * 0.1, 0, top * 0.55])
        s.arrows([at], [[sign * span * 0.2, 0, 0]], scale=inp.values / inp.values.max(), color="#7fb7ff",
                 name="wind", radius=span * 0.004, prop=False)
        s.label("wind {wind}", at + [0, 0, span * 0.03], kind="note")
    step = F.nice_step(top, 6)
    F.mast(s, [x0 - span * 0.06, span * 0.03, 0], top, step)
    ground = min(max(span * 1.5, 60.0), 12000.0)
    cx = (x0 + x1) / 2
    s.ground(0.0, size=ground, grid=F.nice_step(ground, 12), center=(cx, 0.0))
    s.label(f"grid {F.nice_step(ground, 12):g} m", [cx + ground * 0.3, -ground * 0.3, 0], kind="note")

    s.readout("apogee", _num(runs, "apogee_m"), unit="m", digits=4)
    s.readout("landing speed", sink, unit="m/s", digits=3, limit=st.landing_limit)
    s.readout("landing distance", np.abs(_num(runs, "drift_m")), unit="m", digits=3)
    s.readout("off the rail", _num(runs, "rail_exit_speed_mps"), unit="m/s", digits=3)
    s.readout("max Mach", _num(runs, "max_mach"), digits=3)
    s.readout("flight time", _num(runs, "flight_time_s"), unit="s", digits=3)
    verdict(s, o)
    mid = [cx, 0, top / 2]
    s.camera(target=mid, direction=(0.2, -1.0, 0.15), distance=span * 2.1, fov=34,
             presets=[{"name": "Whole flight", "target": mid, "dist": span * 2.1, "dir": (0.2, -1.0, 0.15)},
                      {"name": "Apogee", "follow": body, "dist": L * 8, "dir": (0.55, -1.0, 0.2)},
                      {"name": "Landing", "follow": spot, "dist": max(span * 0.08, L * 40),
                       "dir": (0.4, -1.0, 0.45)},
                      {"name": "Pad", "target": [0, 0, L], "dist": L * 9, "dir": (0.6, -1.0, 0.3)}])
    return s.save(out)


def _body(out, o, st, c, inp, runs):
    dropped = inp.key == "height"
    name = (getattr(st, "name", "") or o.get("part") or "body").replace("-", " ")
    name = name if len(name) < 30 else "body"
    s = Scene(f"The {name} against its {inp.label}", domain="motion", question=o.get("question", ""), length="m",
              source="Every value on the slider is its own fall or flight of the run's model, a point mass, so the "
                     "body does not spin. The path is colored by speed, and the drag arrow is the air's push at "
                     "impact.")
    _timeline(s, inp, f"Each value is a full flight, flown again with the {inp.label} set to it.")
    n = len(runs)
    win = _windows(inp.values)
    dia = _diameter(o, st)
    m0 = float(getattr(st, "mass", 0) or (c["mass_kg"][0] if "mass_kg" in c else 1.0))
    if dia <= 0:
        dia = float((6 * m0 / (np.pi * 2600)) ** (1 / 3))
        s.note("Drawn as a stone of the run's mass, as the run's own size is not saved.")
    R = dia / 2

    # each flight's path, colored by speed against the terminal speed it can never pass
    vt = float(np.nanmax(_num(runs, "terminal_speed_mps"))) if "terminal_speed_mps" in runs[0].numbers else None
    vmax = max(float(r.speed.max()) for r in runs)
    s.field("speed", "m/s", colormap="viridis", range=[0, max(vt or 0, vmax)], limit=vt,
            limit_label="terminal speed" if vt else "")
    for r, (a, b) in zip(runs, win):
        s.lines([np.c_[r.x, np.zeros(len(r.x)), r.z + R]], values=[r.speed], field="speed", unit="m/s", width=2.4,
                name="path").show(a, b)

    # the body where it lands, and a ghost of it where it started
    hit = np.array([[r.x[-1], 0, R] for r in runs])
    start = np.array([[r.x[0], 0, r.z[0] + R] for r in runs])
    X, T = F.rock(dia)
    rk = s.mesh(X, T, name=name, color=ROCK, max_vertices=1500, smooth=False)
    rk.move(hit, (1, 0, 0, 0))
    s.mesh(X, T, name="release", color=ROCK, opacity=0.4, max_vertices=600, smooth=False).move(start, (1, 0, 0, 0))

    # weight and drag at impact, the drag the air's own push on it at that speed
    g = 9.806
    drag = _num(runs, "impact_drag_n")
    wt = m0 * g
    k = R * 6 / max(wt, 1e-9)
    dirv = np.array([np.array([-r.vx[-1], 0, -r.vz[-1]]) / max(float(np.hypot(r.vx[-1], r.vz[-1])), 1e-9)
                     for r in runs])
    s.arrows(hit[:, None, :] + [0, 0, R * 1.2], np.tile([0, 0, -wt * k], (n, 1, 1)), color="#ffd400",
             labels=["weight"], name="weight", radius=R * 0.12)
    s.arrows(hit[:, None, :] + dirv[:, None, :] * R * 1.2, (dirv * (drag * k)[:, None])[:, None, :],
             color="#7fb7ff", labels=["drag {drag at impact}"], name="drag", radius=R * 0.12)

    # the cliff or building it falls from, as tall as the slider says
    hs = np.array([r.z[0] for r in runs])
    hmax = float(hs.max())
    edge = float(start[:, 0].min()) - max(dia * 2.5, 0.3)
    span = float(max(max(np.ptp(np.r_[r.x, 0]) for r in runs), hmax, 1.0))
    if hmax > 1:
        word = next((w for w in TALL if w in words(o)), "cliff" if hmax > 30 else "ledge")
        wide = max(hmax * 0.2, 6.0)
        Xb = np.array([[x, y, z] for x in (edge - wide, edge) for y in (-wide / 2, wide / 2) for z in (0, 1.0)])
        Tb = [(0, 2, 3), (0, 3, 1), (4, 5, 7), (4, 7, 6), (0, 1, 5), (0, 5, 4), (2, 6, 7), (2, 7, 3), (1, 3, 7),
              (1, 7, 5), (0, 4, 6), (0, 6, 2)]
        U = np.zeros((n, 8, 3))
        U[:, Xb[:, 2] > 0, 2] = (hs - 1.0)[:, None]
        wall = s.mesh(Xb, Tb, name=word, color="#6d655c" if word == "cliff" else P.WALL, prop=True, smooth=False)
        wall.deform(U)
        step = F.nice_step(hmax, 8)
        for z in np.arange(step, hmax + 1e-9, step):
            s.lines([np.array([[edge + 0.02, -wide / 2, z], [edge + 0.02, wide / 2, z]])], color="#c9ced4",
                    width=1.4, name="height mark", prop=True).show(z - 1e-6, None)
            s.label(f"{z:g} m", [edge + 0.05, -wide * 0.3, z], t0=z - 1e-6, kind="note")
    ground = min(max(span * 1.4, 30.0, hmax * 0.8), 6000.0)
    s.ground(0.0, size=ground, grid=F.nice_step(ground, 14), center=(float(hit[:, 0].mean()) - ground * 0.15, 0.0))
    s.label("{impact speed}", [0, 0, R * 3], follow=rk, kind="peak")

    s.readout("impact speed", _num(runs, "impact_speed_mps"), unit="m/s", digits=3)
    s.readout("fall time" if dropped else "flight time", _num(runs, "flight_time_s"), unit="s", digits=3)
    if "vacuum_speed_mps" in runs[0].numbers and dropped:
        s.readout("without air", _num(runs, "vacuum_speed_mps"), unit="m/s", digits=3)
    s.readout("drag at impact", drag, unit="N", digits=3)
    if max(abs(r.x[-1]) for r in runs) > 1:
        s.readout("distance", np.abs(_num(runs, "distance_m")), unit="m", digits=3)
    verdict(s, o)
    di = int(np.argmin(np.abs(inp.values - inp.design)))
    mid = [float(hit[:, 0].mean()), 0, float(hs[di]) / 2]
    s.camera(target=mid, direction=(0.9, -1.0, 0.15), distance=max(float(hs[di]), span * 0.4) * 2.2, fov=34,
             presets=[{"name": "Whole drop" if dropped else "Whole flight", "target": mid,
                       "dist": max(float(hs[di]), span * 0.4) * 2.2, "dir": (0.9, -1.0, 0.15)},
                      {"name": "Tallest", "target": [mid[0], 0, hmax / 2], "dist": max(hmax, span) * 2.2,
                       "dir": (0.9, -1.0, 0.15)},
                      {"name": "Impact", "target": [float(hit[di, 0]), 0, 0], "dist": max(dia * 40, 3),
                       "dir": (0.6, -1.0, 0.35)}])
    return s.save(out)
