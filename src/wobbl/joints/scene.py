"""Contact and bolted joint runs as 3D scenes: the parts pressing together, the bolts tightening, then the load.

Every frame is a solved increment. A bolted run first tightens its bolts (stage 0 to 1), then applies the load
(stage 1 to 2); a contact run applies its loads (0 to 1) and then any prescribed movement (1 to 2). The parts are
colored by von Mises stress, the bolts are drawn on their real axes, and the contact force and bolt forces tick
along.
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np


def write_scene(out: Path, kind: str, study, o, r, base, selections, lines=None) -> Path | None:
    from ..solvers.calculix_nl import von_mises
    from ..viz import props as P
    from ..viz.adapters.strength import load_props, supports
    from ..viz.scene import Scene

    mesh, res = r["mesh"], r["res"]
    frames = [fr for fr in res.frames if "U" in fr and "S" in fr]
    if not frames:
        return None
    X = mesh.X
    lo, hi = X.min(0), X.max(0)
    span, size = float(np.linalg.norm(hi - lo)), float(np.ptp(X, 0).max())
    t = np.array([0.0] + [float(fr["time"]) for fr in frames])
    U = np.stack([np.zeros_like(frames[0]["U"])] + [np.asarray(fr["U"], float) for fr in frames])
    vm = np.stack([np.zeros(mesh.n_nodes)] + [von_mises(fr["S"]) for fr in frames])
    name = (Path(study.part).stem or "assembly").replace("_", " ").replace("-", " ")
    bolted = kind == "bolts"
    s = Scene(f"{name.capitalize()}, {'bolted' if bolted else 'in contact'}", domain=kind, question=study.question,
              source="Every frame is a solved increment of the contact solve with friction"
                     + (", the bolts tightened first and the load applied after." if bolted else "."))
    s.timeline(t, unit="", label="stage", seconds=8, between="interpolated",
               note=("0 to 1 the bolts are tightened, 1 to 2 the load comes on" if bolted else
                     "0 to 1 the loads come on" + (", 1 to 2 the faces given a movement move" if t[-1] > 1.01
                                                  else "")))
    top = float(np.percentile(vm, 99.5)) or 1.0
    part = s.mesh(X, mesh.tris, name="assembly", max_vertices=14000)
    part.color_by("von Mises", vm, unit="MPa", range=(0.0, top))
    umax = float(np.abs(U).max())
    part.deform(U, exaggerate=float(f"{max(0.05 * span / umax, 1.0):.2g}") if umax > 0 else 1.0)
    load_scale = np.clip(t - 1.0, 0, 1) if bolted else np.clip(t, 0, 1)
    run = SimpleNamespace(mesh=mesh, study=base, selections=selections)
    words = set((study.question or "").lower().split())
    try:
        supports(s, run, X, size, words)
        load_props(s, run, X, size, load_scale, words)
    except Exception:  # noqa: BLE001  the parts and their fields are the view; props are a courtesy
        pass
    totals = res.contact_totals or {}

    def at_times(rows, key):
        out = np.zeros(len(t))
        for row in rows:
            k = int(np.argmin(np.abs(t - row["time"])))
            if abs(t[k] - row["time"]) < 1e-6:
                out[k] += row[key]
        return out

    contact = sum((at_times(rows, "normal_force") for rows in totals.values()), np.zeros(len(t)))
    if bolted and lines:
        for ln, b in zip(lines, r["bolts"]):
            mid = 0.5 * (ln.head + ln.nut)
            P.bolt(s, mid, -ln.axis, ln.bolt.d / 2, ln.clamp + 1.2 * ln.bolt.d, head=1.0, name=ln.bolt.name)
        force = np.vstack([np.concatenate([[0.0], res.bolt_force[n]]) for n in res.bolt_force])
        s.readout("most loaded bolt", force.max(0) / 1000, unit="kN", digits=3,
                  limit=max(ln.bolt.proof * ln.bolt.stress_area for ln in lines) / 1000)
        s.readout("clamp through the contact", contact / 1000, unit="kN", digits=3)
        s.event(1.0, "tightened", f"each bolt to {r['bolts'][0]['preload'] / 1000:.3g} kN", kind="info")
    else:
        s.readout("contact force", contact, unit="N", digits=3)
    s.readout("peak stress", vm.max(axis=1), unit="MPa", digits=3)
    if o.verdict in ("separates", "slips", "fails"):
        s.event(float(t[-1]), {"separates": "joint opens", "slips": "parts slide", "fails": "overloaded"}[o.verdict],
                "", kind="fail")
    tone = {"holds": "pass", "marginal": "warn"}.get(o.verdict, "fail")
    s.verdict(o.verdict.upper(), o.headline.split(" Confidence")[0], tone)
    s.camera(target=(lo + hi) / 2, direction=P.unit([1.0, -1.4, 0.9]),
             distance=0.5 * span / np.sin(np.radians(15)) * 1.15, fov=30)
    return s.save(out)
