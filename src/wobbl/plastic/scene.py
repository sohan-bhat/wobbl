"""A plasticity run as a 3D scene: the load going on and coming off with the plastic zone spreading, the part
springing back to its permanent set, then the ramp to collapse.

Every frame is a solved increment, nothing scaled: the load and unload run with the true stress strain curve and
large deflection theory first, then the elastic perfectly plastic copy ramped until it gives way. Supports and loads
are drawn as the strength view draws them, on their real faces.
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np


def write_scene(out: Path, study, o, mesh, base, selections, lim, service) -> Path | None:
    from ..viz import props as P
    from ..viz.adapters.strength import load_props, supports
    from ..viz.scene import Scene

    X = mesh.X
    lo, hi = X.min(0), X.max(0)
    span, size = float(np.linalg.norm(hi - lo)), float(np.ptp(X, 0).max())
    frames, t, load, label = [], [], [], []
    if service and "error" not in service:
        for fr in service.get("frames") or [service["on"], service["off"]]:
            tt = float(fr["time"])
            frames.append(fr)
            t.append(tt)
            load.append(tt if tt <= 1 else 2 - tt)
            label.append("loading" if tt <= 1 + 1e-9 else "unloading")
    start = t[-1] if t else 0.0
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
    for fr in ramp:
        frames.append(fr)
        t.append(start + float(fr["time"]))
        load.append(float(fr["time"]) * lim["ramp"])
        label.append("collapse ramp")
    if len(frames) < 2:
        return None
    t = np.array([0.0] + t)
    U = np.stack([np.zeros_like(frames[0]["U"])] + [np.asarray(f["U"], float) for f in frames])
    pe = 100 * np.stack([np.zeros(mesh.n_nodes)] + [np.asarray(f["PEEQ"], float) for f in frames])
    load = np.array([0.0] + load)
    name = (Path(study.part).stem or "part").replace("_", " ").replace("-", " ")
    s = Scene(f"{name.capitalize()} past yield", domain="plastic", question=study.question,
              source="Every frame is a solved increment: the load on and off with the true stress strain curve and "
                     "large deflection theory, then an elastic perfectly plastic copy ramped to collapse.")
    s.timeline(t, unit="", label="stage", seconds=10, between="interpolated",
               note="0 to 1 the load goes on, 1 to 2 it comes off, past 2 the load is ramped until the part collapses")
    part = s.mesh(X, mesh.tris, name="part", max_vertices=12000,
                  keep=[int(np.argmax(pe[-1]))])
    part.color_by("plastic strain", pe, unit="%", colormap="magma", range=(0.0, max(float(pe.max()), 1e-6)))
    umax = float(np.abs(U[: len(t) - len(ramp)]).max()) if len(t) > len(ramp) + 1 else float(np.abs(U).max())
    ex = float(f"{max(0.06 * span / umax, 1.0):.2g}") if umax > 0 else 1.0
    part.deform(U, exaggerate=ex)
    run = SimpleNamespace(mesh=mesh, study=base, selections=selections)
    words = set((study.question or "").lower().split())
    try:
        supports(s, run, X, size, words)
        load_props(s, run, X, size, np.clip(load / max(float(load.max()), 1e-12), 0, None), words)
    except Exception:  # noqa: BLE001  the part and its fields are the view; props are a courtesy
        pass
    s.readout("load", load, unit="x the load", digits=3)
    s.readout("peak plastic strain", pe.max(axis=1), unit="%", digits=3)
    v = o.values
    up = t <= 1.0 + 1e-9                    # the loading frames, where the load only rises
    lam_y = v["first_yield_factor"]
    if lam_y < 1 and up.sum() > 1:
        when = float(np.interp(lam_y, load[up], t[up]))
    elif ramp:                              # it yields only past the load, on the ramp to collapse
        when = float(np.interp(lam_y, load[-len(ramp):], t[-len(ramp):]))
    else:
        when = start
    s.event(when, "first yield", f"at {lam_y:.2f}x the load", kind="warn")
    if "permanent_set_mm" in v:
        s.event(start, "load off", f"a permanent set of {v['permanent_set_mm']:.3g} mm", kind="info")
    if v.get("collapsed"):
        s.event(float(t[-1]), "collapse", f"at {v['collapse_factor']:.3g}x the load", kind="fail")
    hinge = np.asarray(v.get("hinge_xyz", X[int(np.argmax(pe[-1]))]), float)
    s.label("hinge {peak plastic strain}", hinge, kind="peak", t0=start)
    word = o.verdict.upper()
    tone = {"holds": "pass", "yields": "warn", "marginal": "warn", "collapses": "fail"}.get(o.verdict, "info")
    s.verdict(word, o.headline.split(". Confidence")[0], tone)
    s.camera(target=(lo + hi) / 2, direction=P.unit([1.0, -1.4, 0.8]),
             distance=0.5 * span / np.sin(np.radians(15)) * 1.15, fov=30)
    return s.save(out)
