"""A cell runaway run as a scene the designer drives: drag the oven's temperature and the cell shows the worst it
gets there, settling a little above the oven or running away past the onset.

The run's playback.json keeps the cell followed to its peak in each of a range of ovens (runaway.oven_sweep). The
cell is colored by how fast it heats itself at worst, against the 10 C/min that marks runaway, and glows with its
peak temperature. An older run without the sweep has no oven to drag, so the chemistry adapter draws it instead.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .. import props as P
from ..scene import Scene
from .common import read, verdict

TRIGGER = 10.0            # C/min of self heating that counts as runaway, as the run defines it


def build(out: Path, pb: dict | None = None) -> Path | None:
    out = Path(out)
    pb = pb if pb is not None else (read(out, "playback.json") or {})
    sw, c = pb.get("sweep"), pb.get("cell") or {}
    if not sw or len(sw.get("oven_C") or []) < 2:
        from .chem import build as chem
        return chem(out, pb)
    o = read(out, "outcome.json") or {}
    oven = np.asarray(sw["oven_C"], float)
    peak = np.asarray(sw["peak_C"], float)
    rate = np.asarray(sw["self_heating_C_min"], float)
    D, L = float(c.get("diameter_mm", 18.0)), float(c.get("length_mm", 65.0))
    onset, here = c.get("critical_C"), c.get("surroundings_C")
    s = Scene(f"An {c.get('label', 'cell')} cell in an oven at any temperature", domain="battery",
              question=o.get("question", ""), length="mm",
              source="Each oven is its own transient of the lumped cell with its reactants used up as they react, "
                     "followed until the cell peaks.")
    s.timeline(oven, unit="C", label="oven", kind="input", between="interpolated",
               design=here if here is not None else float(oven[len(oven) // 2]))
    s.note("Each oven shows the worst the cell gets in it, at its peak. The cell counts as running away once it "
           f"heats itself faster than {TRIGGER:g} C/min. Past that trigger the kinetics no longer forecast it.")
    z = L / 2 + 6
    b = s.body("cylinder", [D / 2, L / 2], pos=[0, 0, z], color="#3c6e9e", name="cell")
    b.color_by("self heating", rate, unit="C/min", colormap="turbo", range=(0.0, 2 * TRIGGER), limit=TRIGGER,
               limit_label="runaway")
    b.light(np.clip((peak - 150) / 250, 0, 1))
    P.wire_box(s, [-L * 0.6, -L * 0.6, 0], [L * 0.6, L * 0.6, L * 1.4], name="oven")
    s.readout("peak self heating", rate, unit="C/min", digits=3, limit=TRIGGER)
    s.readout("peak cell temperature", peak, unit="C", digits=4)
    s.readout("above the oven", peak - oven, unit="K", digits=3)
    s.readout("time to peak", sw["peak_min"], unit="min", digits=3)
    s.readout("SEI left at the peak", np.asarray(sw["sei_left"], float) * 100, unit="%", digits=3)
    s.readout("electrolyte left at the peak", np.asarray(sw["electrolyte_left"], float) * 100, unit="%", digits=3)
    s.label("oven", [0, -L * 0.6, L * 1.5], kind="note")
    if onset is not None and oven[0] <= onset <= oven[-1]:
        s.event(onset, "runaway starts", c.get("trigger", ""), kind="fail")
        s.label("runs away, {peak self heating}", [0, 0, L + 14], t0=onset, kind="fail")
        s.label("settles, {peak cell temperature}", [0, 0, L + 14], t1=onset - 1e-3, kind="peak")
    if here is not None:
        s.event(here, "where the cell sits", "its surroundings in the question", kind="info")
    s.ground(0, size=L * 4, grid=10, color="#1d1f23")
    verdict(s, o)
    s.camera(target=[0, 0, L * 0.6], direction=(0.3, -1, 0.45), distance=L * 4.0)
    return s.save(out)
