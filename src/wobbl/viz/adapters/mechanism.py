"""A mechanism run as a scene: every MuJoCo geom moving with its recorded pose, and the joints' torques ticking.

The poses are the run's own, recorded at evenly spaced steps; the viewer blends between them and says so. Driven
joints show the torque they need beside what their motor can give at that speed, and a joint that asks for more
than its motor has turns its readout red.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from ..scene import Scene
from .common import plain, read, verdict

FAIL_WORDS = ("limit", "stall", "slip", "tip", "bind", "collide", "hits", "lift off", "separat")


def build(out: Path, pb: dict | None = None) -> Path | None:
    pb = pb or read(out, "playback.json")
    if not pb or pb.get("manifest", {}).get("kind") != "mechanism":
        return None
    m = pb["manifest"]
    t = np.asarray(pb["t"], float)
    o = read(out, "outcome.json") or {}
    s = Scene(m.get("title", "Mechanism").capitalize(), domain="mechanism", question=m.get("question", ""),
              length="m", source=m.get("source", ""))
    s.timeline(t, unit="s", label="time", seconds=float(m.get("suggested_seconds") or max(t[-1], 3)),
               between="interpolated")
    moving = [np.asarray(b["pos"], float) for b in pb["bodies"] if not b.get("static")]
    allp = np.concatenate([p.reshape(-1, 3) for p in moving] or [np.zeros((1, 3))])
    reach = float(np.ptp(allp, 0).max()) or 0.5
    centre = allp.mean(0)
    for b in pb["bodies"]:
        shape, size = b["shape"], [float(x) for x in b["size"]]
        if shape == "plane":
            hx = size[0] or reach * 2.5
            hy = size[1] or reach * 2.5
            s.ground(float(np.asarray(b["pos"])[2]), size=2 * max(hx, hy), grid=_grid(reach), color="#1d1f23",
                     center=np.asarray(b["pos"], float)[:2])
            continue
        if shape not in ("sphere", "box", "cylinder", "capsule", "ellipsoid"):
            s.note(f"The {b['name']} is a {shape} geom, which this view does not draw.")
            continue
        if shape == "sphere":
            size = size[:1]
        s.body(shape, size, pos=b["pos"], quat=b["quat"], color=b.get("color", "#9aa0a6"), name=b["name"])
    shown = 0
    for name, j in pb.get("joints", {}).items():
        if "needed" in j and shown < 3:
            need = np.asarray(j["needed"]["values"], float)
            lim = np.abs(np.asarray(j.get("limit", {}).get("values", need), float))
            unit = j["needed"].get("unit", "N m")
            s.readout(f"{name} needs", np.abs(need), unit=unit, digits=3, limit=lim)
            s.readout(f"{name} motor gives", lim, unit=unit, digits=3)
            shown += 1
    for name, j in list(pb.get("joints", {}).items())[:4]:
        if "needed" not in j and shown < 4 and np.ptp(j["angle"]["values"]) > 1e-3:
            v = np.asarray(j["angle"]["values"], float)
            if j["angle"].get("unit") == "m":
                s.readout(f"{name} position", v * 1000, unit="mm", digits=3)
            else:
                s.readout(f"{name} angle", np.degrees(v), unit="deg", digits=3)
            shown += 1
    for name, ser in pb.get("series", {}).items():
        if len(s.readouts) < 7:
            s.readout(name, ser["values"], unit=ser.get("unit", ""), digits=3)
    for e in pb.get("events", []):
        bad = any(w in e["name"].lower() for w in FAIL_WORDS)
        s.event(float(e["t_s"]), e["name"], e.get("detail", ""), kind="fail" if bad else "info")
    verdict(s, {**o, "verdict": o.get("verdict") or m.get("verdict"), "headline": o.get("headline") or m.get("headline")})
    s.camera(target=centre, direction=(0.55, -1.0, 0.55), distance=reach * 2.6 + 0.2)
    return s.save(out)


def _grid(reach: float) -> float:
    return float(10 ** np.floor(np.log10(max(reach, 1e-3) / 3)))


__all__ = ["build", "plain"]
