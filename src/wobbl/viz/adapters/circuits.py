"""A circuit run as a scene: a bench with a breadboard, the real parts standing in the holes board.plan gives
them, round wires between, and a bench supply with its leads. Current runs through every wire and lead as moving
glows whose speed follows the solved current frame by frame. Each part warms from tan toward red by its power
against its rating, an LED glows by its solved current, and a part that goes past its rating is flagged on itself
at the moment it does.

board.json, written by the circuit run next to its outcome, holds the plan (circuits/board.py), the key reading
and the run's time span. A transient plays its own time. An operating point is swept, the supply turned up from
zero in solved steps, so the timeline shows where along the way a rating is crossed (circuit_frames.py).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .. import props_bench as B
from ..scene import Scene
from .circuit_frames import frames, sweep
from .common import read, verdict

PITCH = 2.54                      # mm between breadboard holes
LED = {"red": "#ff3b30", "green": "#34c759", "blue": "#2f7bff", "white": "#f2f2ee", "yellow": "#ffd60a",
       "amber": "#ff9f0a", "orange": "#ff9f0a", "ir": "#7a1f1f", "uv": "#8e5cff"}
SPEED = 16.0                      # mm a second of playback for the largest current
HEAT = "share of rating"


def xy(x, y, z=0.0) -> np.ndarray:
    return np.array([x * PITCH, -y * PITCH, z])


def _time_unit(t: np.ndarray) -> tuple[float, str]:
    top = float(np.max(np.abs(t))) if len(t) else 0.0
    for f, u in ((1, "s"), (1e3, "ms"), (1e6, "us"), (1e9, "ns")):
        if top * f >= 1:
            return f, u
    return 1.0, "s"


def build(out: Path) -> Path | None:
    b = read(out, "board.json")
    if not b or "plan" not in b:
        return None
    o = read(out, "outcome.json") or {}
    pb = read(out, "playback.json")
    pl = (b.get("sweep") or {}).get("plan") or b["plan"]
    parts, wires, ins = pl.get("parts", []), pl.get("wires", []), pl.get("instruments", [])
    F = frames(b, pb)
    n = len(F["t"])
    s = Scene("The circuit on the bench", domain="circuit", question=o.get("question", ""),
              source="Parts sit in the holes the board plan gives them. The glows in each wire move at the current "
                     "the run solved there, and each part warms by its solved power against its rating.")
    if n > 1:
        if F["unit"] == "s":
            f, u = _time_unit(F["t"])
            s.timeline(F["t"] * f, unit=u, label="time", seconds=8, between="interpolated")
        else:
            s.timeline(F["t"], unit="V", label=F["label"], seconds=8, between="interpolated",
                       note="The supply is turned up from zero to its value in solved steps.")
    T = s.t
    bd = pl["board"]
    W, H = bd["width"] * PITCH, bd["height"] * PITCH
    B.bench(s, [W / 2 - 30, -H / 2 - 10, -8.5], W / 2 + 70, H / 2 + 45)
    rows = bd["rows"]
    B.breadboard(s, W, H, np.array([xy(2.2 + c - 1, y, 0.03) for c in range(1, bd["cols"] + 1)
                                    for y in rows.values()]))
    ch = (rows.get("e", 7.6) + rows.get("f", 10.6)) / 2
    s.body("box", [W / 2 - 2, 1.0, 0.3], pos=[W / 2, -ch * PITCH, 0.0], color="#d8d4c8", name="channel", prop=True)
    for r, col in (("t+", "#d23b30"), ("t-", "#2f6fd6"), ("b-", "#2f6fd6"), ("b+", "#d23b30")):
        if r in rows:
            y = rows[r] + (-0.5 if r in ("t+", "b-") else 0.5)
            s.body("box", [W / 2 - 3, 0.18, 0.05], pos=xy(bd["width"] / 2, y, 0.02), color=col, name="rail stripe",
                   prop=True)

    allcur = [np.abs(x) for x in F["parts"] + F["wires"] + [y for i in F["leads"] for y in i]]
    peak = max([float(x.max()) for x in allcur if len(x)] + [1e-12])
    paths, speeds = [], []

    for j, w in enumerate(wires):
        pts = [xy(x, y, 0.6) for x, y in w["points"]]
        if len(pts) == 2:
            a, c = pts
            L = float(np.linalg.norm(c - a))
            pts = B.arch(a, c, min(1.5 + L * 0.15, 5.0)) if w.get("kind") == "jumper" else \
                np.array([a, a + [0, 0, 1.2], c + [0, 0, 1.2], c])
        pts = np.asarray(pts, float)
        B.wire(s, pts, 0.38, w.get("color") or "#2b2e33", f"wire {w.get('net', '')}")
        paths.append(pts + [0, 0, 0.4])
        speeds.append(SPEED * F["wires"][j] / peak)

    hot, labels = [], []
    for j, p in enumerate(parts):
        pins = [xy(q["x"], q["y"]) for q in p.get("pins", [])]
        if not pins:
            continue
        cur, heat = F["parts"][j], F["heat"][j]
        body, top = place(s, p, pins, cur, heat, o.get("question", ""))
        a, c = pins[0], pins[-1] if len(pins) > 1 else pins[0]
        for q in (a, c):
            paths.append(np.array([q + [0, 0, 0.2], q + [0, 0, top]]))
        speeds += [SPEED * cur / peak, -SPEED * cur / peak]
        mid = np.mean(pins, axis=0)
        labels.append((p, mid + [0, 0, top + 6]))
        if heat is not None and float(np.max(heat)) > 1:
            hot.append((p, mid, top, heat))

    for i, inst in enumerate(ins):
        instrument(s, inst, i, F["leads"][i] if i < len(F["leads"]) else [], peak, paths, speeds)

    live = [k for k, v in enumerate(speeds) if float(np.max(np.abs(v))) > 1e-9]
    if live:
        flow = np.stack([np.broadcast_to(speeds[k], (n,)) for k in live], axis=1)
        s.lines([paths[k] for k in live], color="#ffd400", width=1.4, name="current", flow=flow, spacing=6.0)

    loaded = sorted((j for j, h in enumerate(F["heat"]) if h is not None), key=lambda j: -float(np.max(F["heat"][j])))
    named = {parts[j]["name"] for j in loaded[:2]} | {p["name"] for p in parts if p.get("kind") == "led" or
                                                       len(p.get("pins", [])) > 2}
    for p, at in labels:
        if p not in [h[0] for h in hot] and (p["name"] in named or len(parts) <= 6):
            s.label(f"{p['name']} {p.get('label') or ''}".strip(), at, kind="note")
    for p, mid, top, heat in hot[:3]:
        k = int(np.argmax(heat > 1))
        t0 = float(T[k]) if n > 1 else None
        ring = s.body("torus", [5.5, 0.45], pos=mid + [0, 0, 0.6], color="#ff3b30", name=f"{p['name']} over rating")
        ring.light(np.ones(n))
        if t0 is not None:
            ring.show(t0, None)
            s.label(f"{p['name']} {p.get('label') or ''}".strip(), mid + [0, 0, top + 6], t1=t0, kind="note")
        s.label(f"{p['name']} past its rating, {{{p['name']} load}}", mid + [0, 0, top + 6], t0=t0, kind="fail")
        s.event(t0 if t0 is not None else 0.0, f"{p['name']} past its rating",
                f"{100 * float(heat[k]):.0f}% of its rating", kind="fail")

    readouts(s, b, F, parts, ins, hot)
    verdict(s, o)
    used = np.array([xy(q["x"], q["y"]) for p in parts for q in p.get("pins", [])] or [[W / 2, -H / 2, 0]])
    lo, hi = used.min(0), used.max(0)
    s.camera(target=(lo + hi) / 2 + [-10, 0, 4], direction=(-0.2, -0.8, 0.85),
             distance=max(float(np.linalg.norm(hi - lo)) * 1.6, 120.0),
             presets=[{"name": "top", "target": [W / 2, -H / 2, 0], "dir": [0, -0.08, 1], "dist": W * 1.3},
                      {"name": "close", "target": [W / 2, -H / 2, 4], "dir": [0.3, -0.7, 0.6], "dist": W * 0.6}])
    return s.save(out)


def place(s, p, pins, cur, heat, question: str = "") -> tuple:
    """The part on the board. Returns its body and how high its leads rise."""
    body, name = p.get("body") or "", p["name"]
    a, c = pins[0], pins[-1]
    top = 3.0
    if body == "axial-resistor" and len(pins) == 2:
        bd = B.resistor(s, a, c, p.get("value"), name)
    elif body == "axial-diode" and len(pins) == 2:
        bd = B.diode(s, a, c, name)
    elif body == "led-5mm":
        said = [w for w in LED if w in question.lower().split()]
        named = any(w in f"{p.get('model', '')} {p.get('label', '')}".lower() for w in LED)
        word = said[0] if len(said) == 1 and not named else str(p.get("color") or "")
        col = LED.get(word.lower(), "#ff3b30")
        for g in B.led(s, pins, col, name):
            g.light(np.clip(np.abs(cur) / 0.02, 0, 1) ** 0.6)
        return None, top
    elif body == "radial-capacitor":
        bd, top = B.capacitor(s, pins, p.get("value"), name), 1.0
    elif body == "inductor":
        bd, top = B.inductor(s, pins, name), 1.5
    elif body == "to220":
        bd = B.to220(s, pins, name)
    elif body == "to92":
        bd = B.to92(s, pins, name)
    elif body in ("dip", "ic") or len(pins) > 3:
        bd, top = B.dip(s, pins, name), 2.2
    else:
        bd, top = B.module(s, pins, name), 2.0
    if heat is not None and bd is not None:
        bd.color_by(HEAT, np.asarray(heat, float), unit="", colormap="heat", range=(0, 1), limit=1,
                    limit_label="its rating")
    return bd, top


def instrument(s, inst, i, leads, peak, paths, speeds):
    """A bench supply, generator or meter in front of the board, its leads arching to their holes."""
    at = np.array([-26.0 - 42.0 * i, -14.0, 0.0])
    tips = B.supply(s, at, inst["name"], posts=2,
                    display={"generator": "#ffb347", "meter": "#9ad0ff", "current": "#9ad0ff"}.get(inst["kind"], "#7cf29a"))
    read_ = "  ".join(f"{v} {u}" for v, u in inst.get("readings", []))
    s.label(f"{inst['name']} {read_}".strip(), at + [0, 0, 28], kind="note")
    for j, ld in enumerate(inst.get("leads", [])):
        post = tips[0 if ld.get("post") == "+" else 1]
        hole = xy(ld["x"], ld["y"], 0.4)
        L = float(np.linalg.norm(hole - post))
        pts = B.arch(post, hole, max(8.0, L * 0.25), 9)
        B.wire(s, pts, 0.6, "#c8352b" if ld.get("post") == "+" else "#1c1d20", "test lead")
        paths.append(pts + [0, 0, 0.6])
        speeds.append(SPEED * (leads[j] if j < len(leads) else (ld.get("current") or 0)) / peak)


def readouts(s, b, F, parts, ins, hot):
    n = s.n
    if F["kind"] == "sweep":
        s.readout("supply", F["t"], unit="V", digits=3)
    key = b.get("key")
    if key:
        ser = key.get("series")
        if n > 1 and ser and len(ser) == n:
            v = np.asarray(ser, float)
        elif n > 1:
            v = _key_frames(key, F, parts)
        else:
            v = np.full(n, key["value"])
        if v is not None:
            s.readout(key["label"].lower(), v, unit=key.get("unit", ""), digits=3, limit=key.get("limit"))
    sup = next((k for k, x in enumerate(ins) if x.get("kind") == "supply"), None)
    if sup is not None and F["leads"][sup]:
        s.readout("supply current", np.abs(F["leads"][sup][0]) * 1000, unit="mA", digits=3)
    leds = [j for j, p in enumerate(parts) if p.get("kind") == "led"]
    for j in leds[:1]:
        s.readout(f"{parts[j]['name']} current", np.abs(F["parts"][j]) * 1000, unit="mA", digits=3)
    order = sorted((j for j, h in enumerate(F["heat"]) if h is not None), key=lambda j: -float(np.max(F["heat"][j])))
    for j in order[:2]:
        s.readout(f"{parts[j]['name']} load", 100 * np.asarray(F["heat"][j], float), unit="% of rating", digits=3,
                  limit=100)
    for p, *_ in hot:
        j = parts.index(p)
        if j not in order[:2]:
            s.readout(f"{p['name']} load", 100 * np.asarray(F["heat"][j], float), unit="% of rating", digits=3,
                      limit=100)


def _key_frames(key, F, parts):
    """The key reading along a sweep, scaled from its part's own current or power, which it is a reading of."""
    ref = key["label"].split()[0]
    j = next((k for k, p in enumerate(parts) if p["name"].lower() in (ref.lower(), "x" + ref.lower())), None)
    if j is None:
        return None
    unit = str(key.get("unit", ""))
    p = parts[j]
    if unit.endswith("A") and p.get("current"):
        return key["value"] * np.abs(F["parts"][j]) / abs(p["current"])
    if unit.endswith("W") and F["heat"][j] is not None and p.get("heat"):
        return key["value"] * np.asarray(F["heat"][j]) / p["heat"]
    return None


def write_board(out, c, w, stress) -> Path | None:
    """board.json for a finished circuit run: its board plan and key reading, which the 3D view is built from,
    and for an operating point the supply sweep that gives the view its timeline."""
    from ...circuits.board import plan
    from ...circuits.pictures import key_reading, wave_data
    pl = plan(c, wave_data(c, w))
    doc = {"plan": pl, "key": key_reading(c, w, stress),
           "span": [float(w.t[0]), float(w.t[-1])] if getattr(w, "kind", "") == "tran" else None}
    if getattr(w, "kind", "") != "tran":
        try:
            doc["sweep"] = sweep(c, pl, Path(out) / "work")
        except Exception:  # noqa: BLE001  the sweep only adds a timeline to the view
            doc["sweep"] = None
    p = Path(out) / "board.json"
    p.write_text(json.dumps(doc, separators=(",", ":"), default=_num))
    return p


def _num(x):
    if isinstance(x, (np.floating, np.integer)):
        return x.item()
    if isinstance(x, np.ndarray):
        return x.tolist()
    return str(x)
