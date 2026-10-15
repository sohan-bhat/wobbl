"""A motion run as a 3D scene: the body at its real size flying its computed path over the ground, at real scale.

Positions, speeds and times are the run's own rows from trajectory.csv. The model is a point mass in a vertical
plane, so a rocket points the way the air meets it (its velocity less the wind) once it leaves the rail, which is
how a stable rocket flies, and hangs under its parachute after one opens. The airframe, fins, rail and canopies
come from the spec at their real sizes, the plume follows the thrust curve, and the wind blows across the pad at
its own speed. A dropped or thrown body is drawn as a stone of its real size beside the cliff or building the
question names, with its drag and weight as arrows.
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

ROCK, BODY, NOSE, FIN, PLUME = "#b8ab98", "#e9e9e4", "#ffd400", "#2c2f34", "#ffa040"
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


def frames(c: dict, rocket: bool, n: int = 230) -> np.ndarray:
    """Which rows to keep as frames: for a short drop rows even in time, so it plays in real time, and for a long
    flight rows spread by path and time together, so the fast boost is not over in an instant."""
    t = c["t_s"]
    if len(t) <= n:
        return np.arange(len(t))
    if not rocket:
        return np.unique(np.searchsorted(t, np.linspace(t[0], t[-1], n)).clip(0, len(t) - 1))
    p = np.c_[c["x_m"], c["alt_m"]]
    ds = np.r_[0, np.linalg.norm(np.diff(p, axis=0), axis=1)]
    dt = np.r_[0, np.diff(t)]
    s = np.cumsum(ds / max(ds.sum(), 1e-9) + dt / max(dt.sum(), 1e-9))
    return np.unique(np.searchsorted(s, np.linspace(0, s[-1], n)).clip(0, len(t) - 1))


def _rate(c: dict, k: str) -> np.ndarray:
    """The time derivative of a column, with repeated times (rows written at one instant) nudged apart."""
    t = c["t_s"] + np.arange(len(c["t_s"])) * 1e-9
    return np.gradient(c[k], t)


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
    """The scene of a motion run from its folder. st, the run's MotionStudy, gives the body's real size and shape;
    without it a rocket's spec file is read, and a plain body is drawn from what the run saved."""
    o = o or read(out, "outcome.json")
    c = rows(out)
    if not o or c is None or o.get("domain") != "motion":
        return None
    st = st or spec_of(out, o)
    rocket = bool(st is not None and getattr(st, "motor", None)) or "boost" in set(c["phase"])
    if rocket and st is not None and getattr(st, "motor", None):
        return _rocket(out, o, c, st)
    return _body(out, o, c, st)


def _timeline(s, c, keep, rocket):
    t = c["t_s"][keep]
    dur = float(t[-1] - t[0])
    secs = float(np.clip(dur, 4, 24)) if not rocket else float(np.clip(dur / 8, 14, 30))
    s.timeline(t, unit="s", label="time", seconds=secs, between="interpolated")
    return t, dur / max(secs, 1e-9)


def _events(s, o, lift=0.0):
    for e in o.get("events", []):
        bad = e["name"] in ("impact",) and float(e.get("speed_mps", 0)) > 20
        s.event(float(e["t_s"]), e["name"], f"{e.get('speed_mps', 0):.3g} m/s" if e.get("speed_mps") else "",
                kind="fail" if bad else "info")
        at = [e["x_m"], 0, max(e["alt_m"], 0) + lift]
        if e["name"] in ("impact", "landing"):
            s.label(f"{e['name']} at {e.get('speed_mps', 0):.3g} m/s", at, t0=float(e["t_s"]),
                    kind="fail" if bad else "note")
        elif e["name"] == "apogee":
            s.label(f"apogee {e['alt_m']:.4g} m", at, t0=float(e["t_s"]), kind="peak")
        elif e["name"] in ("burnout", "main deploy", "rail exit"):
            s.label(f"{e['name']}", at, t0=float(e["t_s"]), t1=float(e["t_s"]) + 12, kind="note")


def _rocket(out, o, c, st):
    keep = frames(c, True)
    ph = c["phase"][keep]
    s = Scene(f"The {(st.name or 'rocket').replace('-', ' ')}'s flight", domain="motion", question=o.get("question", ""),
              length="m", source="Positions and times are the run's own rows, a point mass in a vertical plane. The "
                                 "rocket points into the air it meets and hangs under its parachutes once they open.")
    t, rate = _timeline(s, c, keep, True)
    n = len(t)
    af = st.airframe
    dia = float(st.diameter)
    L = float(af.length if af else dia * 20)
    cg = float(af.cg if af and af.cg else L * 0.6)
    lean = np.radians(90 - float(st.angle or 90))
    rail_dir = np.array([np.sin(lean), 0, np.cos(lean)])
    # the run's point starts on the ground; the drawn rocket stands with its aft end on the pad there
    lift = (L - cg) * np.cos(lean) + max(dia * 0.4, 0.04)
    pos = np.c_[c["x_m"][keep], np.zeros(n), c["alt_m"][keep] + lift]
    vel = np.c_[_rate(c, "x_m")[keep] + st.wind, np.zeros(n), _rate(c, "alt_m")[keep]]
    hang = np.isin(ph, ("drogue", "main", "landed"))
    on_rail = (ph == "rail") | (np.linalg.norm(vel, axis=1) < 2) & ~hang
    d = np.where(hang[:, None], [0, 0, 1.0], np.where(on_rail[:, None], rail_dir, vel))
    q = quats_along(d)
    parts = F.airframe(L, dia, af.nose_length if af else dia * 4, af.nose_shape if af else "ogive",
                       af.fins if af else None)
    colors = {"nose": NOSE, "body tube": BODY, "fins": FIN}
    for name, (X, T) in parts.items():
        m = s.mesh(X - [0, 0, cg], T, name=name if name != "body tube" else "rocket", color=colors[name],
                   max_vertices=3000, smooth=name != "fins")
        m.move(pos, q)
    body = next(ob for ob in s.objects if ob.name == "rocket")

    # the plume, as long as the thrust is strong, from the aft end
    mot = st.motor
    ct = np.array([p[0] for p in mot.curve])
    cf = np.array([p[1] for p in mot.curve])
    thrust = np.interp(t, ct, cf, left=0, right=0)
    frac = thrust / max(cf.max(), 1e-9)
    burn = float(ct[cf > 0].max()) if (cf > 0).any() else 0.0
    Xp, Tp = F.plume(dia * 0.35, L * 1.2)
    pl = s.mesh(Xp - [0, 0, cg], Tp, name="exhaust", color=PLUME, opacity=0.7, max_vertices=600)
    pl.deform(np.c_[np.zeros(len(Xp)), np.zeros(len(Xp)), Xp[:, 2]], scale=frac - 1)
    pl.move(pos, q)
    pl.glow = np.clip(frac, 0, 1)
    pl.show(0, burn)
    s.notes = [x for x in s.notes if not x.startswith("Deformation")]

    # parachutes, opening over a second and a half at their events, the canopies at their real diameters
    ev = {e["name"]: e for e in o.get("events", [])}
    chutes = [(ch, next((e for k, e in ev.items() if k.startswith(ch.name)), None)) for ch in st.chutes]
    chutes = [(ch, e) for ch, e in chutes if e is not None]
    for i, (ch, e) in enumerate(chutes):
        t0 = float(e["t_s"])
        later = [float(x["t_s"]) for _, x in chutes if float(x["t_s"]) > t0]
        lines = max(ch.diameter * 1.3, L * 0.5)
        Xc, Tc = F.canopy(ch.diameter, lines)
        open_ = np.clip((t - t0) / 1.5, 0, 1)
        cm = s.mesh(Xc, Tc, name=f"{ch.name} parachute", color="#ff6b3d" if ch.name == "main" else "#ffb020",
                    max_vertices=900)
        cm.deform(-Xc, scale=(1 - open_) * 0.92)
        cm.move(pos + [0, 0, L - cg], np.tile([1.0, 0, 0, 0], (n, 1)))
        cm.show(t0, min(later) if later else None)
    s.notes = [x for x in s.notes if not x.startswith("Deformation")]

    # the trail, the pad and rail, the wind and a height scale
    full = np.c_[c["x_m"], np.zeros(len(c["x_m"])), c["alt_m"] + lift]
    s.lines([full], values=[c["speed_mps"]], field="speed", unit="m/s", colormap="viridis", drawn=[c["t_s"]],
            width=2.5, name="path")
    pad = max(dia * 6, 0.5)
    P.pad(s, [0, 0, 0], pad)
    rail = float(st.rail_length or 0)
    if rail > 0:
        foot = np.array([-rail_dir[0] * (L - cg), 0, 0]) + [dia * 0.9, 0, 0]
        s.body("box", [dia * 0.12, dia * 0.12, rail / 2], pos=foot + rail_dir * rail / 2 + [0, dia * 0.0, 0],
               quat=P.z_to(rail_dir), color=P.BOLT, name="launch rail", prop=True)
    top = float(c["alt_m"].max())
    step = F.nice_step(top, 6)
    F.mast(s, [-max(top * 0.05, 20), max(top * 0.03, 15), 0], top, step)
    F.wind(s, [0, 0, 0], float(st.wind), [-1, 0, 0], area=max(L * 12, 30), heights=[L * 0.6, L * 1.6, L * 3],
           rate=rate)
    span = float(max(np.ptp(full[:, 0]), top, 1.0))
    ground = min(max(span * 1.4, 60.0), 6000.0)
    s.ground(0.0, size=ground, grid=F.nice_step(ground, 12), center=(float(full[:, 0].mean()), 0.0))
    s.label(f"grid {F.nice_step(ground, 12):g} m", [float(full[:, 0].mean()) + ground * 0.3, -ground * 0.3, 0],
            kind="note")
    s.label("{altitude}", [0, 0, L - cg + dia * 2], follow=body, kind="peak")

    s.readout("altitude", c["alt_m"][keep], unit="m", digits=4)
    s.readout("speed", c["speed_mps"][keep], unit="m/s", digits=3)
    if "mach" in c:
        s.readout("Mach", c["mach"][keep], digits=3)
    s.readout("thrust", thrust, unit="N", digits=4)
    if np.ptp(c["x_m"]) > 1:
        s.readout("downrange", c["x_m"][keep], unit="m", digits=3)
    _events(s, o, lift)
    verdict(s, o)
    near = L * 5.5
    s.camera(target=pos[0], direction=(0.55, -1.0, 0.18), distance=near, follow=body, fov=34,
             presets=[{"name": "Follow", "follow": body, "dist": near, "dir": (0.55, -1.0, 0.18)},
                      {"name": "Pad", "target": [0, 0, L], "dist": L * 9, "dir": (0.6, -1.0, 0.3)},
                      {"name": "Whole flight", "target": [float(full[:, 0].mean()), 0, top / 2],
                       "dist": span * 2.0, "dir": (0.15, -1.0, 0.12)}])
    return s.save(out)


def _body(out, o, c, st):
    keep = frames(c, False)
    s0 = c["alt_m"][0]
    falling = s0 > c["alt_m"][-1] + 1 and np.ptp(c["x_m"]) < s0
    name = (getattr(st, "name", "") or o.get("part") or "body").replace("-", " ")
    name = name if len(name) < 30 else "body"
    s = Scene(f"The {name} falling" if falling else f"The {name}'s flight", domain="motion",
              question=o.get("question", ""), length="m",
              source="Positions and times are the run's own rows. The model is a point mass, so the body does not "
                     "spin, and the drag arrow is the run's own deceleration below gravity.")
    t, rate = _timeline(s, c, keep, False)
    n = len(t)
    dia = _diameter(o, st)
    if dia <= 0:
        m = float(c["mass_kg"][0]) if "mass_kg" in c else 1.0
        dia = float((6 * m / (np.pi * 2600)) ** (1 / 3))
        s.note("Drawn as a stone of the run's mass, as the run's own size is not saved.")
    R = dia / 2
    pos = np.c_[c["x_m"][keep], np.zeros(n), c["alt_m"][keep] + R]
    X, T = F.rock(dia)
    rk = s.mesh(X, T, name=name, color=ROCK, max_vertices=1500, smooth=False)
    rk.move(pos, (1, 0, 0, 0))

    # weight and drag on the body: drag is what the run's deceleration leaves of gravity
    W = words(o)
    mass = c["mass_kg"][keep] if "mass_kg" in c else np.ones(n)
    g = 9.806
    vx = _rate(c, "x_m")[keep]
    vz = _rate(c, "alt_m")[keep]
    sp = np.maximum(np.hypot(vx, vz), 1e-9)
    acc = c["accel_mps2"][keep] if "accel_mps2" in c else np.full(n, g)
    # falling, drag is the weight less the net pull the run's acceleration shows; otherwise the speed's loss
    drag = np.clip(mass * (g - acc), 0, None) if falling else np.clip(mass * np.abs(np.gradient(sp, t)), 0, None)
    drag[c["phase"][keep] != c["phase"][keep][0]] = 0
    wt = float(mass[0] * g)
    k = R * 6 / max(wt, 1e-9)
    dirv = -np.c_[vx, np.zeros(n), vz] / sp[:, None]
    s.arrows(pos[:, None, :] + [0, 0, -R * 1.1], np.tile([0, 0, -wt * k], (n, 1, 1)), color="#ffd400",
             labels=["weight"], name="weight", radius=R * 0.12)
    s.arrows(pos[:, None, :] + dirv[:, None, :] * R * 1.1, (dirv * (drag * k)[:, None])[:, None, :],
             color="#7fb7ff", labels=["drag {drag}"], name="drag", radius=R * 0.12)

    s.lines([np.c_[c["x_m"], np.zeros(len(c["x_m"])), c["alt_m"] + R]], values=[c["speed_mps"]], field="speed",
            unit="m/s", colormap="viridis", drawn=[c["t_s"]], width=2.0, name="path")

    h0 = float(c["alt_m"][0])
    span = float(max(np.ptp(pos[:, 0]), np.ptp(pos[:, 2]), 1.0))
    edge = float(pos[0, 0]) - max(dia * 2.5, 0.3)
    if h0 > 1:
        word = next((w for w in TALL if w in W), "cliff" if h0 > 30 else "ledge")
        wide = max(h0 * 0.3, 6.0)
        # the face in a few rough slabs, so it reads as rock and not one flat wall
        rng = np.random.default_rng(5)
        bands = max(3, min(9, int(h0 / max(F.nice_step(h0, 6), 1e-9))))
        hb = h0 / bands
        for i in range(bands):
            jut = rng.uniform(0, wide * 0.05) if word == "cliff" else 0
            s.body("box", [wide / 2, wide / 2, hb / 2], pos=[edge - wide / 2 - jut, 0, hb * (i + 0.5)],
                   color=["#6d655c", "#615a52", "#77706a"][i % 3] if word == "cliff" else P.WALL,
                   name=word, prop=True)
        step = F.nice_step(h0, 6)
        s.lines([np.array([[edge + 0.02, -wide / 2, z], [edge + 0.02, wide / 2, z]]) for z in np.arange(step, h0, step)],
                color="#c9ced4", width=1.4, name="height marks", prop=True)
        for z in np.arange(step, h0 + 1e-9, step):
            s.label(f"{z:g} m", [edge + 0.05, -wide * 0.3, z], kind="note")
    ground = min(max(span * 1.4, 30.0, h0 * 0.8), 4000.0)
    s.ground(0.0, size=ground, grid=F.nice_step(ground, 14), center=(float(pos[:, 0].mean()) - ground * 0.15, 0.0))

    hit = next((e for e in o.get("events", []) if e["name"] in ("impact", "landing")), None)
    if hit:
        dust = s.body("torus", [max(dia * 3, 0.3), max(dia * 0.6, 0.06)], pos=[hit["x_m"], 0, 0.02],
                      color="#a39684", name="impact dust", prop=True, opacity=0.6)
        dust.show(float(hit["t_s"]))
    s.label("{speed}", [0, 0, R * 2.5], follow=rk, kind="peak")
    s.readout("height", c["alt_m"][keep], unit="m", digits=4)
    s.readout("speed", c["speed_mps"][keep], unit="m/s", digits=3)
    s.readout("drag", drag, unit="N", digits=3)
    if np.ptp(c["x_m"]) > 1:
        s.readout("distance", c["x_m"][keep], unit="m", digits=3)
    _events(s, o)
    verdict(s, o)
    near = max(dia * 22, 1.2)
    s.camera(target=pos[0], direction=(0.8, -1.0, 0.1), distance=near, follow=rk, fov=34,
             presets=[{"name": "Follow", "follow": rk, "dist": near, "dir": (0.8, -1.0, 0.1)},
                      {"name": "Whole drop" if falling else "Whole flight",
                       "target": [float(pos[:, 0].mean()), 0, float(pos[:, 2].max()) / 2],
                       "dist": span * 2.2, "dir": (0.9, -1.0, 0.15)},
                      {"name": "Ground", "target": [float(pos[-1, 0]), 0, 0], "dist": max(dia * 60, 4),
                       "dir": (0.6, -1.0, 0.35)}])
    return s.save(out)
