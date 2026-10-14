"""Manufacturing runs as scenes: a mold filling from its gates, a casting freezing in from the mold walls, a print
rising layer by layer.

The surface carries the times the run solved at each vertex (when the melt arrived, when the metal started and
finished freezing), so each frame shows exactly which places are full, freezing or solid at that time.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .. import props as P
from ..scene import Scene
from .common import read, verdict


def build(out: Path, pb: dict) -> Path | None:
    m = pb.get("manifest", {})
    kind = m.get("kind")
    if kind not in ("molding", "casting", "printing", "print") or "mesh" not in pb:
        return None
    o = read(out, "outcome.json") or {}
    V = np.asarray(pb["mesh"]["vertices"], float).reshape(-1, 3)
    F = np.asarray(pb["mesh"]["triangles"], int).reshape(-1, 3)
    f = {k: np.asarray([np.nan if x is None else x for x in v], float) for k, v in pb["fields"].items()}
    fr = pb.get("frames", {})
    t = np.asarray(fr.get("time_s", [0.0]), float)
    s = Scene(m.get("title", "Process"), domain="process", question=o.get("question", ""),
              source="Each vertex carries the times the run solved there, so every frame is exact at its time.")
    s.timeline(t, unit="s", label="time", seconds=8, between="exact",
               note="Between frames the view reads the same solved times, so in between is exact too.")
    if kind == "molding":
        tf = np.where(f["t_fill"] < 0, np.inf, f["t_fill"])
        s.mesh(V, F, name="mold cavity", color="#9aa0a6", opacity=0.18, max_vertices=len(V) + 1)
        melt = s.mesh(V, F, name="melt", max_vertices=len(V) + 1)
        melt.color_by("melt arrives", np.where(np.isfinite(tf), tf, np.nan), unit="s", colormap="viridis")
        melt.reveal(np.where(np.isfinite(tf), tf, 1e30), t)
        for g in m.get("gates", []):
            P.arrow(s, g, [0, 0, -1], float(np.ptp(V, 0).max()) * 0.2, label="gate")
        if fr.get("gate_pressure_MPa"):
            s.readout("gate pressure", fr["gate_pressure_MPa"], unit="MPa", digits=3, limit=fr.get("machine_limit_MPa"))
        if fr.get("share_full"):
            s.readout("full", np.asarray(fr["share_full"], float) * 100, unit="%", digits=3)
        ev = pb.get("events", {})
        for p_ in ev.get("weld_lines", [])[:3]:
            s.label("weld line", p_, t0=float(t[-1]) * 0.6, kind="note")
        for p_ in ev.get("air_traps", [])[:3]:
            s.label("air trap", p_, t0=float(t[-1]) * 0.6, kind="fail")
    elif kind == "casting":
        tl, ts = f["t_liquidus"], f["t_solidus"]
        liquid = np.clip((ts[None, :] - t[:, None]) / np.maximum(ts - np.maximum(tl, 0), 1e-9)[None, :], 0, 1)
        metal = s.mesh(V, F, name="metal", max_vertices=len(V) + 1)
        metal.color_by("still liquid", liquid, colormap="inferno", range=(0, 1))
        lo, hi = V.min(0), V.max(0)
        pad = float(np.ptp(V, 0).max()) * 0.15
        P.wire_box(s, lo - pad, hi + pad, color="#8b6b47", name="sand mold")
        s.readout("still liquid", liquid.mean(1) * 100, unit="%", digits=3)
        last = pb.get("events", {}).get("last_to_freeze")
        if last:
            s.event(last["t_s"], "last spot freezes", "shrinkage porosity risk", kind="fail" if pb["events"].get(
                "pockets") else "info")
            s.label("freezes last", last["xyz"], t0=last["t_s"] * 0.8, kind="fail")
    else:
        h = next((v for k, v in f.items() if "height" in k or "layer" in k), None)
        if h is None:
            return None
        part = s.mesh(V, F, name="print", color="#ffd400", max_vertices=len(V) + 1)
        part.reveal(h, np.asarray(fr.get("height_mm", np.linspace(h.min(), h.max(), len(t))), float))
        lo, hi = V.min(0), V.max(0)
        P.slab(s, [(lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, lo[2]], [0, 0, 1], np.ptp(V, 0)[0], np.ptp(V, 0)[1],
               2.0, color=P.STEEL, name="print bed")
    verdict(s, o)
    return s.save(out)
