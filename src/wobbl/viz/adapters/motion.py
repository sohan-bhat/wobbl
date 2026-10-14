"""A motion run as a scene: the body flying its computed path over the ground, the path tracing itself behind it.

Positions, speeds and times are the run's own rows from trajectory.csv. The model is a point mass in a vertical
plane, so a rocket is drawn pointing along its path while it flies and hanging under its parachute after one
opens, and the canopies are drawn at the sizes the spec gives. The surroundings come from the question: a cliff,
a building or a launch pad with its rail.
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from .. import props as P
from ..scene import Scene
from .common import box_mesh, join, lathe, quats_along, read, verdict, words

ROCK, BODY, NOSE, FIN = "#8a8f96", "#e9e9e4", "#ffd400", "#2c2f34"


def rows(out: Path) -> dict | None:
    p = Path(out) / "trajectory.csv"
    if not p.exists():
        return None
    with p.open() as f:
        r = list(csv.DictReader(f))
    if not r:
        return None
    cols = {k: np.array([float(x[k]) for x in r]) for k in r[0] if k != "phase"}
    cols["phase"] = [x.get("phase", "") for x in r]
    return cols


def spec_of(out: Path, o: dict):
    part = o.get("part") or ""
    if part.endswith(".toml"):
        p = (Path(out) / part).resolve()
        if p.exists():
            from ...motion.spec import read_spec
            return read_spec(p, o.get("question", ""))
    return None


def frames(c: dict, rocket: bool, n: int = 220) -> np.ndarray:
    """Which rows to keep as frames: every row for a short drop, so it plays in real time, and for a long flight
    rows spread by path and time together, so the fast boost is not over in an instant."""
    t = c["t_s"]
    if len(t) <= n:
        return np.arange(len(t))
    if not rocket and t[-1] < 30:
        return np.unique(np.searchsorted(t, np.linspace(t[0], t[-1], n)).clip(0, len(t) - 1))
    p = np.c_[c["x_m"], c["alt_m"]]
    ds = np.r_[0, np.linalg.norm(np.diff(p, axis=0), axis=1)]
    dt = np.r_[0, np.diff(t)]
    s = np.cumsum(ds / max(ds.sum(), 1e-9) + dt / max(dt.sum(), 1e-9))
    return np.unique(np.searchsorted(s, np.linspace(0, s[-1], n)).clip(0, len(t) - 1))


def rocket_mesh(st) -> tuple[np.ndarray, np.ndarray]:
    """The airframe from the spec, nose up along +z with its tip at z = 0 moved to the balance point."""
    d, L = st.diameter, st.airframe.length if st.airframe else st.diameter * 20
    nl = st.airframe.nose_length if st.airframe else d * 4
    r = d / 2
    k = np.linspace(0, 1, 9)
    nose = [(r * np.sqrt(np.clip(2 * x - x * x, 0, 1)), L - nl * (1 - x)) for x in k]   # ogive like, tip at L
    tube = [(0, 0), (r, 0)] + [(a, b) for a, b in nose[1:]] + [(0, L)]
    X, T = lathe(tube, 20)
    parts = [(X, T)]
    if st.airframe and st.airframe.fins:
        f = st.airframe.fins
        root, span = f.root, f.span
        z0 = L - (f.position + root) if f.position else 0.0
        for i in range(f.count):
            a = 2 * np.pi * i / f.count
            rot = np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])
            parts.append(box_mesh(rot @ [r + span / 2, 0, z0 + root / 2], [span / 2, d * 0.03, root / 2], rot))
    X, T = join(*parts)
    cg = (L - st.airframe.cg) if st.airframe and st.airframe.cg else L * 0.4
    return X - [0, 0, cg], T


def build(out: Path, st=None, o: dict | None = None) -> Path | None:
    """The scene of a motion run from its folder. st, the run's MotionStudy, gives the body's real size and shape;
    without it a rocket's spec file is read, and a plain body is drawn from its mass."""
    o = o or read(out, "outcome.json")
    c = rows(out)
    if not o or c is None or o.get("domain") != "motion":
        return None
    st = st or spec_of(out, o)
    rocket = bool(st is not None and getattr(st, "motor", None)) or "burn" in c["phase"]
    keep = frames(c, rocket)
    t = c["t_s"][keep]
    pos = np.c_[c["x_m"][keep], np.zeros(len(keep)), c["alt_m"][keep]]
    W = words(o)
    name = (getattr(st, "name", "") or "body").replace("-", " ")
    s = Scene(f"The {name}'s flight" if rocket else f"The {name} falling" if c["alt_m"][0] > c["alt_m"][-1] + 1
              else f"The {name}'s flight", domain="motion", question=o.get("question", ""), length="m",
              source="Positions and times are the run's own rows. The model is a point mass in a vertical plane.")
    dur = float(t[-1] - t[0])
    s.timeline(t, unit="s", label="time", seconds=float(np.clip(dur, 4, 24)) if not rocket else 22.0,
               between="interpolated")
    span = float(max(np.ptp(pos[:, 0]), np.ptp(pos[:, 2]), 1.0))
    full = np.c_[c["x_m"], np.zeros(len(c["x_m"])), c["alt_m"]]
    s.lines([full], values=[c["speed_mps"]], field="speed", unit="m/s", colormap="viridis", drawn=[c["t_s"]],
            width=2.5, name="path")
    dia = float(getattr(st, "diameter", 0) or 0)
    if rocket and st is not None:
        X, T = rocket_mesh(st)
        vel = np.gradient(pos, t, axis=0) if len(t) > 1 else np.tile([0, 0, 1.0], (len(t), 1))
        up = np.tile([0.0, 0.0, 1.0], (len(t), 1))
        hang = np.array([p in ("drogue", "main") for p in np.asarray(c["phase"])[keep]])
        on_rail = np.array([p == "rail" for p in np.asarray(c["phase"])[keep]])
        lean = np.radians(90 - float(getattr(st, "angle", 90) or 90))
        rail_dir = np.array([np.sin(lean), 0, np.cos(lean)])
        d = np.where(hang[:, None], up, np.where(on_rail[:, None] | (np.linalg.norm(vel, axis=1) < 1)[:, None],
                                                 rail_dir, vel))
        s.mesh(X, T, name=name, color=BODY, max_vertices=4000).move(pos, quats_along(d))
        L = st.airframe.length if st.airframe else dia * 20
        for ch in getattr(st, "chutes", []):
            ev = next((e for e in o.get("events", []) if e["name"].startswith(ch.name)), None)
            if ev is None:
                continue
            canopy = s.body("cone", [ch.diameter / 2, ch.diameter * 0.18], pos=pos + [0, 0, L * 0.9 + ch.diameter],
                            color="#ff6b3d" if ch.name == "main" else "#ffb020", name=f"{ch.name} parachute")
            canopy.show(ev["t_s"])
        P.pad(s, [0, 0, 0], max(dia * 6, 0.4), rail=float(getattr(st, "rail_length", 0) or 0))
    else:
        if dia <= 0:
            m = float(c["mass_kg"][0]) if "mass_kg" in c else 1.0
            dia = float((6 * m / (np.pi * 2600)) ** (1 / 3))
            s.note("Drawn as a ball of the run's mass at the density of stone, as the run's own size is not saved.")
        s.body("sphere", [dia / 2], pos=pos, color=ROCK, name=name)
    h0 = float(c["alt_m"][0])
    if h0 > 1 and W & {"cliff", "building", "tower", "roof", "bridge", "balcony", "ledge"}:
        wide = max(h0 * 0.35, 4.0)
        word = next(w for w in ("cliff", "building", "tower", "roof", "bridge", "balcony", "ledge") if w in W)
        edge = float(pos[0, 0]) - max(dia * 3, 0.15)
        s.body("box", [wide / 2, wide / 2, h0 / 2], pos=[edge - wide / 2, 0, h0 / 2], color=P.WALL, name=word, prop=True)
        # marks up the face every few metres, so the fall reads as it passes them
        step = 10.0 if h0 <= 150 else 25.0
        s.lines([np.array([[edge + 0.01, -wide / 2, z], [edge + 0.01, wide / 2, z]]) for z in np.arange(step, h0, step)],
                color="#9aa0a6", width=1.2, name="height marks", prop=True)
    ground = max(span * 1.3, 40.0)
    s.ground(0.0, size=min(ground, 4000.0), grid=float(10 ** np.floor(np.log10(max(dia * 40, 0.5)))) if span < 600
             else 50.0, center=(float(pos[:, 0].mean()), 0.0))
    s.readout("altitude", c["alt_m"][keep], unit="m", digits=4)
    s.readout("speed", c["speed_mps"][keep], unit="m/s", digits=3)
    if rocket and "mach" in c:
        s.readout("Mach", c["mach"][keep], digits=3)
    if "x_m" in c and np.ptp(c["x_m"]) > 1:
        s.readout("downrange", c["x_m"][keep], unit="m", digits=3)
    for e in o.get("events", []):
        bad = e["name"] in ("impact",) and float(e.get("speed_mps", 0)) > 20
        s.event(float(e["t_s"]), e["name"], f"{e.get('speed_mps', 0):.3g} m/s" if e.get("speed_mps") else "",
                kind="fail" if bad else "info")
        if e["name"] in ("impact", "landing"):
            s.label(f"{e['name']} at {e.get('speed_mps', 0):.3g} m/s", [e["x_m"], 0, max(e["alt_m"], 0)],
                    t0=float(e["t_s"]), kind="fail" if bad else "note")
        if e["name"] == "apogee":
            s.label(f"apogee {e['alt_m']:.4g} m", [e["x_m"], 0, e["alt_m"]], t0=float(e["t_s"]), kind="peak")
    verdict(s, o)
    body = s.objects[1]
    near = (st.airframe.length if st is not None and st.airframe else 2.0) * 6 if rocket else max(dia * 14, 0.8)
    s.camera(target=pos[0], direction=(0.35, -1.0, 0.25), distance=near, follow=body,
             presets=[{"name": "Follow", "follow": body, "dist": near, "dir": (0.35, -1.0, 0.25)},
                      {"name": "Whole path", "target": [float(pos[:, 0].mean()), 0, float(np.ptp(pos[:, 2]) / 2)],
                       "dist": span * 1.9, "dir": (0.0, -1.0, 0.18)}])
    return s.save(out)
