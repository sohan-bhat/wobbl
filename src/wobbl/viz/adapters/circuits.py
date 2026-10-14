"""A circuit run as a scene: its parts on a breadboard, wired as board.plan lays them out, with current moving
along every wire and lead at a speed set by the run's own current, and each part colored by how close it runs to
its rating.

board.json, written by the circuit run next to its outcome, holds the plan (circuits/board.py), the key reading
and the run's time span. A part past its rating is shown failing in a second step, held after the first, because
the run solves the current and the rating says what that current does to the part. The view says so.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .. import props as P
from ..scene import Scene
from .common import read, verdict

PITCH = 2.54                      # mm between breadboard holes
LED = {"red": "#ff3b30", "green": "#34c759", "blue": "#2f7bff", "white": "#f2f2ee", "yellow": "#ffd60a",
       "amber": "#ff9f0a", "orange": "#ff9f0a", "ir": "#7a1f1f", "uv": "#8e5cff"}
LEAD = "#b8bcc4"
SPEED = 14.0                      # mm a second of playback for the largest current


def xy(x, y, z=0.0) -> np.ndarray:
    return np.array([x * PITCH, -y * PITCH, z])


def build(out: Path) -> Path | None:
    b = read(out, "board.json")
    if not b or "plan" not in b:
        return None
    o = read(out, "outcome.json") or {}
    pl = b["plan"]
    parts, wires, ins = pl.get("parts", []), pl.get("wires", []), pl.get("instruments", [])
    burnt = [p for p in parts if (p.get("heat") or 0) > 1]
    steps = 2 if burnt else 1
    s = Scene("The circuit on its breadboard", domain="circuit", question=o.get("question", ""),
              source="Parts and wires sit where the board plan puts them, and every glow moves at the current the "
                     "run measured in that wire.")
    if burnt:
        s.timeline([0.0, 1.0], unit="", label="step", seconds=7, between="step",
                   note="Step 1 runs at the solved current. Step 2 shows what running past its rating does to a "
                        "real part, drawn and not simulated.")
    peak = max([abs(w.get("current") or 0) for w in wires] + [abs(p.get("current") or 0) for p in parts] + [1e-12])
    board(s, pl["board"])
    flow_paths, flow_speeds = [], []
    for w in wires:
        pts = [xy(x, y, 0.7) for x, y in w["points"]]
        if w.get("kind") == "jumper" and len(pts) == 2:
            a, c = pts
            mid = (a + c) / 2 + [0, 0, min(np.linalg.norm(c - a) * 0.35, 8)]
            pts = [a, (a + mid) / 2 + [0, 0, 1.5], mid, (mid + c) / 2 + [0, 0, 1.5], c]
        s.lines([np.array(pts)], color=w.get("color") or "#2b2e33", width=3.5, name=f"wire {w.get('net', '')}",
                prop=True)
        flow_paths.append(np.array(pts))
        flow_speeds.append(SPEED * (w.get("current") or 0) / peak)
    labels = []
    for p in parts:
        lead_paths, lead_speed = place(s, p, steps, burnt)
        flow_paths += lead_paths
        flow_speeds += [SPEED * (p.get("current") or 0) / peak * sgn for sgn in lead_speed]
        if p.get("pins"):
            c = np.mean([xy(q["x"], q["y"]) for q in p["pins"]], axis=0)
            labels.append((p, c))
    for i, inst in enumerate(ins):
        instrument(s, inst, i, pl["board"], flow_paths, flow_speeds, peak)
    live = [k for k, v in enumerate(flow_speeds) if abs(v) > 1e-9]
    if live:
        s.lines([flow_paths[k] + [0, 0, 0.25] for k in live], color="#ffd400", width=1.2, name="current",
                flow=np.array([flow_speeds[k] for k in live]), spacing=7.0)
    for p, c in labels:
        hot = (p.get("heat") or 0) > 1
        s.label(f"{p['name']} {p.get('label') or ''}".strip() + (f", {100 * p['heat']:.0f}% of its rating"
                                                                 if p.get("heat") else ""),
                c + [0, 0, 9], kind="fail" if hot else "note")
    key = b.get("key")
    vals = np.ones(steps)
    if key:
        s.readout(key["label"].lower(), key["value"] * vals, unit=key.get("unit", ""), digits=3, limit=key.get("limit"))
    sup = next((x for x in ins if x.get("kind") == "supply"), None)
    if sup and sup.get("current") is not None:
        s.readout("supply current", abs(sup["current"]) * 1000 * vals, unit="mA", digits=3)
    for p in burnt[:2]:
        s.event(1.0, f"{p['name']} past its rating", "it burns out", kind="fail")
    verdict(s, o)
    W, H = pl["board"]["width"] * PITCH, pl["board"]["height"] * PITCH
    s.camera(target=[W / 2, -H / 2, 0], direction=(0.1, -0.75, 1.0), distance=max(W, H) * 1.35)
    return s.save(out)


def board(s, bd):
    W, H = bd["width"] * PITCH, bd["height"] * PITCH
    s.body("box", [W / 2, H / 2, 4.25], pos=[W / 2, -H / 2, -4.25], color="#e9e6dc", name="breadboard", prop=True)
    rows = bd["rows"]
    holes = [xy(2.2 + c - 1, y, 0.02) for c in range(1, bd["cols"] + 1) for r, y in rows.items()]
    s.points(np.array(holes), color="#4a4a48", size=0.32, name="holes")
    ch = (rows.get("e", 7.6) + rows.get("f", 10.6)) / 2
    s.body("box", [W / 2 - 2, 1.0, 0.3], pos=[W / 2, -ch * PITCH, 0.0], color="#cfcbbf", name="channel", prop=True)
    for r, col in (("t+", "#d23b30"), ("t-", "#2f6fd6"), ("b-", "#2f6fd6"), ("b+", "#d23b30")):
        if r in rows:
            y = rows[r] + (-0.5 if r in ("t+", "b-") else 0.5)
            s.lines([np.array([xy(1.2, y, 0.05), xy(bd["width"] - 1.2, y, 0.05)])], color=col, width=2, prop=True,
                    name="rail line")


def _heat_frames(p, steps):
    h = float(p.get("heat") or 0)
    return np.full(steps, h)


def place(s, p, steps, burnt) -> tuple[list, list]:
    """The part's body and leads. Returns its leads as paths and the direction its current runs along each."""
    pins = [xy(q["x"], q["y"]) for q in p.get("pins", [])]
    if not pins:
        return [], []
    body = p.get("body") or ""
    dead = p in burnt
    paths, signs = [], []
    a = pins[0]
    c = pins[-1] if len(pins) > 1 else pins[0]
    mid = (a + c) / 2
    heat = _heat_frames(p, steps)
    if body in ("axial-resistor", "axial-diode") and len(pins) == 2:
        h = 3.2
        d = c - a
        L = float(np.linalg.norm(d)) or PITCH
        bl = min(L * 0.55, 6.5)
        u = d / L
        ea, eb = mid - u * bl / 2 + [0, 0, h], mid + u * bl / 2 + [0, 0, h]
        paths += [np.array([a, a + [0, 0, h], ea]), np.array([eb, c + [0, 0, h], c])]
        signs += [1, 1]
        for path in paths:
            s.lines([path], color=LEAD, width=2.2, name=f"{p['name']} lead", prop=True)
        col = "#d9c49a" if body == "axial-resistor" else "#2b2e33"
        bd = s.body("cylinder", [1.25, bl / 2], pos=mid + [0, 0, h], quat=P.z_to(d), color=col, name=p["name"])
        if p.get("heat") is not None:
            bd.color_by("heat", np.where(np.arange(steps) == 1, 1.6, heat) if dead else heat,
                        unit="share of rating", colormap="heat", range=(0, 1), limit=1, limit_label="its rating")
        return paths, signs
    top = 2.0
    for q in pins:
        paths.append(np.array([q, q + [0, 0, top]]))
        signs.append(1)
        s.lines([paths[-1]], color=LEAD, width=2.2, name=f"{p['name']} lead", prop=True)
    if body == "led-5mm":
        col = LED.get(str(p.get("color") or "").lower(), "#ff3b30")
        glow = np.full(steps, min(abs(p.get("current") or 0) / 0.02, 1.0))
        for c_, when in ((col, (None, 0.0) if dead else None), ("#3b3430", (1.0, None) if dead else False)):
            if when is False:
                continue
            cyl = s.body("cylinder", [2.5, 2.4], pos=mid + [0, 0, top + 2.4], color=c_, name=p["name"])
            dome = s.body("sphere", [2.5], pos=mid + [0, 0, top + 4.8], color=c_, name=f"{p['name']} dome")
            for b_ in (cyl, dome):
                if c_ == col:
                    b_.light(glow)
                if when:
                    b_.show(*when)
        if dead:
            s.label("burnt out", mid + [0, 0, top + 11], t0=1.0, kind="fail")
    elif body == "radial-capacitor":
        s.body("cylinder", [2.6, 3.2], pos=mid + [0, 0, top + 3.2], color="#1d3a63", name=p["name"])
    elif body == "inductor":
        s.body("torus", [3.2, 1.3], pos=mid + [0, 0, top + 1.4], color="#3a4a3a", name=p["name"])
    elif body == "to220":
        s.body("box", [5.0, 2.2, 4.5], pos=mid + [0, 0, top + 4.5], color="#1c1d20", name=p["name"])
        s.body("box", [5.0, 0.6, 3.0], pos=mid + [0, 1.6, top + 11.5], color="#c9ccd1", name=f"{p['name']} tab")
    else:
        lo, hi = np.min(pins, 0), np.max(pins, 0)
        half = np.maximum((hi - lo) / 2 + 1.5, 2.0)
        s.body("box", [half[0], half[1], 2.2], pos=mid + [0, 0, top + 2.2], color="#1c1d20", name=p["name"])
    return paths, signs


def instrument(s, inst, i, bd, paths, speeds, peak):
    """A bench supply, generator or meter beside the board, its leads running to their holes."""
    at = xy(-9.0, 3.0 + 9.0 * i, 0.0)
    col = {"supply": "#2b2e33", "generator": "#26323d", "meter": "#2d2b26", "current": "#2d2b26"}.get(inst["kind"],
                                                                                                    "#2b2e33")
    s.body("box", [9.0, 7.0, 6.0], pos=at + [0, 0, 6.0], color=col, name=inst["name"], prop=True)
    read_ = "  ".join(f"{v} {u}" for v, u in inst.get("readings", []))
    s.label(f"{inst['name']} {read_}".strip(), at + [0, 0, 14], kind="note")
    for j, ld in enumerate(inst.get("leads", [])):
        post = at + [9.0, (j - 0.5) * 5, 8.0]
        hole = xy(ld["x"], ld["y"], 0.5)
        mid = (post + hole) / 2 + [0, 0, 10]
        path = np.array([post, (post + mid) / 2 + [0, 0, 2], mid, (mid + hole) / 2 + [0, 0, 2], hole])
        s.lines([path], color="#c8352b" if ld.get("post") == "+" else "#2b2e33", width=3, name="lead", prop=True)
        paths.append(path)
        speeds.append(SPEED * (ld.get("current") or 0) / peak)      # out of the post along the lead


def write_board(out, c, w, stress) -> Path | None:
    """board.json for a finished circuit run: its board plan and key reading, which the 3D view is built from."""
    from ...circuits.board import plan
    from ...circuits.pictures import key_reading, wave_data
    pl = plan(c, wave_data(c, w))
    doc = {"plan": pl, "key": key_reading(c, w, stress),
           "span": [float(w.t[0]), float(w.t[-1])] if getattr(w, "kind", "") == "tran" else None}
    p = Path(out) / "board.json"
    p.write_text(json.dumps(doc, separators=(",", ":"), default=_num))
    return p


def _num(x):
    if isinstance(x, (np.floating, np.integer)):
        return x.item()
    if isinstance(x, np.ndarray):
        return x.tolist()
    return str(x)
