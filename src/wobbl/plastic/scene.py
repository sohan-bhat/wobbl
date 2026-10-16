"""A plasticity run as a 3D scene under a load the viewer sets: the part at that load with its plastic zone and its
deflection, from no load up to the load that collapses it, and the permanent set the run found at the asked load.

Every frame is a solved increment, nothing scaled. Up to the asked load the frames are the loading half of the
service run, with the true stress strain curve and large deflection theory. Past it they are the elastic perfectly
plastic copy ramped until it gives way. Supports and loads are drawn as the strength view draws them, on their real
faces.
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np

SET = 0.2       # percent plastic strain that marks yield, the offset that defines a metal's yield stress


def write_scene(out: Path, study, o, mesh, base, selections, lim, service) -> Path | None:
    from ..viz import props as P
    from ..viz.adapters.strength import load_props, supports
    from ..viz.scene import Scene

    X = mesh.X
    lo, hi = X.min(0), X.max(0)
    span, size = float(np.linalg.norm(hi - lo)), float(np.ptp(X, 0).max())
    frames, load = [], []
    if service and "error" not in service:
        # only the loading half: taking the load off again is not something the viewer sets
        for fr in service.get("frames") or [service["on"]]:
            if float(fr["time"]) <= 1 + 1e-9:
                frames.append(fr)
                load.append(float(fr["time"]))
    reached = max(load, default=0.0)
    ramp = [fr for fr in lim.get("frames", []) if "PEEQ" in fr and "U" in fr]
    if len(ramp) > 1:
        # the last increment of a mechanism runs away under small deflection theory; it is shown, but no further
        # than four times the increment before it, which already reads as the part giving way
        prev = float(np.linalg.norm(ramp[-2]["U"], axis=1).max())
        last = dict(ramp[-1])
        top = float(np.linalg.norm(last["U"], axis=1).max())
        if prev > 0 and top > 4 * prev:
            last["U"] = last["U"] * (4 * prev / top)
        ramp[-1] = last
    past = [fr for fr in ramp if float(fr["time"]) * lim["ramp"] > reached + 1e-9]
    for fr in past:
        frames.append(fr)
        load.append(float(fr["time"]) * lim["ramp"])
    if len(frames) < 2:
        return None
    lam = np.array([0.0] + load)
    U = np.stack([np.zeros_like(frames[0]["U"])] + [np.asarray(f["U"], float) for f in frames])
    pe = 100 * np.stack([np.zeros(mesh.n_nodes)] + [np.asarray(f["PEEQ"], float) for f in frames])
    total = sum(float(ld.magnitude) for ld in base.loads if ld.kind in ("force", "remote", "bolt", "pressure"))
    kn = total >= 1000
    unit, per = ("kN", total / 1000) if kn else (("N", total) if total > 0 else ("x the load", 1.0))
    name = (Path(study.part).stem or "part").replace("_", " ").replace("-", " ")
    s = Scene(f"{name.capitalize()} past yield", domain="plastic", question=study.question,
              source="Every frame is a solved increment. Up to the asked load it is the loading run with the true "
                     "stress strain curve and large deflection theory, past it an elastic perfectly plastic copy "
                     "ramped to collapse.")
    s.timeline(lam * per, unit=unit, label="load", between="interpolated", kind="input", design=per,
               note="each frame is solved, the loads between them are interpolated")
    part = s.mesh(X, mesh.tris, name="part", max_vertices=12000, keep=[int(np.argmax(pe[-1]))])
    part.color_by("plastic strain", pe, unit="%", colormap="magma", range=(0.0, SET), limit=SET,
                  limit_label="yield, 0.2% set")
    umax = float(np.abs(U[: len(lam) - len(past)]).max()) if len(lam) > len(past) + 1 else float(np.abs(U).max())
    ex = float(f"{max(0.06 * span / umax, 1.0):.2g}") if umax > 0 else 1.0
    part.deform(U, exaggerate=ex)
    run = SimpleNamespace(mesh=mesh, study=base, selections=selections)
    words = set((study.question or "").lower().split())
    try:
        supports(s, run, X, size, words)
        load_props(s, run, X, size, lam / max(float(lam.max()), 1e-12), words)
    except Exception:  # noqa: BLE001  the part and its fields are the view; props are a courtesy
        pass
    s.readout("peak plastic strain", pe.max(axis=1), unit="%", digits=3, limit=SET)
    s.readout("deflection", np.linalg.norm(U, axis=2).max(axis=1), unit="mm", digits=3)
    v = o.values
    if v.get("collapsed"):
        s.readout("reserve to collapse", np.where(lam > 0, v["collapse_factor"] / np.maximum(lam, 1e-12), 99),
                  unit="x", digits=3, limit=None)
    s.event(v["first_yield_factor"] * per, "first yield", f"at {v['first_yield_factor']:.2f}x the load", kind="warn")
    if "permanent_set_mm" in v:
        s.note(f"Unloaded from the asked load the run keeps a permanent set of {v['permanent_set_mm']:.3g} mm.")
    if v.get("collapsed"):
        s.event(float(lam[-1] * per), "collapse", f"at {v['collapse_factor']:.3g}x the load", kind="fail")
    hinge = np.asarray(v.get("hinge_xyz", X[int(np.argmax(pe[-1]))]), float)
    s.label("hinge {peak plastic strain}", hinge, kind="peak", t0=v["first_yield_factor"] * per)
    word = o.verdict.upper()
    tone = {"holds": "pass", "yields": "warn", "marginal": "warn", "collapses": "fail"}.get(o.verdict, "info")
    s.verdict(word, o.headline.split(". Confidence")[0], tone)
    s.camera(target=(lo + hi) / 2, direction=P.unit([1.0, -1.4, 0.8]),
             distance=0.5 * span / np.sin(np.radians(15)) * 1.15, fov=30)
    return s.save(out)

