"""Contact and bolted joint runs as 3D scenes under a load the viewer sets.

Every frame is a solved increment of the contact solve. A bolted run tightens its bolts first and then applies the
load, so its frames start from the clamped joint and sweep the external load up to the asked one, with the bolt
force, the clamp left at the interface and any opening reading off at each load. A contact run sweeps its loads the
same way, then any prescribed movement. The parts are colored by von Mises stress and the bolts are drawn on their
real axes.
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np


def write_scene(out: Path, kind: str, study, o, r, base, selections, lines=None) -> Path | None:
    from ..materials import resolve_material
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
    bolted = kind == "bolts"
    t_all = np.array([0.0] + [float(fr["time"]) for fr in frames])
    # a bolted joint's tightening is not something the viewer sets: its sweep starts from the clamped joint
    first = int(np.flatnonzero(t_all <= 1.0 + 1e-9)[-1]) if bolted else 0
    t = t_all[first:]
    zero = [np.zeros_like(frames[0]["U"])]
    U = np.stack(zero + [np.asarray(fr["U"], float) for fr in frames])[first:]
    vm = np.stack([np.zeros(mesh.n_nodes)] + [von_mises(fr["S"]) for fr in frames])[first:]
    opening = np.array([0.0] + [float(np.max(fr["open"])) if len(fr.get("open", [])) else 0.0
                                for fr in frames])[first:]
    total = sum(float(ld.magnitude) for ld in base.loads if ld.kind in ("force", "remote", "bolt", "pressure"))
    share = np.clip(t - 1.0, 0, 1) if bolted else np.clip(t, 0, 1)      # the fraction of the load on
    moved = not bolted and t[-1] > 1.01
    name = (Path(study.part).stem or "assembly").replace("_", " ").replace("-", " ")
    s = Scene(f"{name.capitalize()}, {'bolted' if bolted else 'in contact'}", domain=kind, question=study.question,
              source="Every frame is a solved increment of the contact solve with friction"
                     + (", from the joint as the bolts clamp it to the full load." if bolted else "."))
    if total > 0 and not moved:
        s.timeline(total * share, unit="N", label="load", between="interpolated", kind="input", design=total,
                   note="each frame is solved, the loads between them are interpolated")
    else:
        s.timeline(t, unit="", label="load, then movement" if moved else "load", between="interpolated",
                   kind="input", design=float(t[-1]),
                   note="0 to 1 the loads come on" + (", 1 to 2 the faces given a movement move" if moved else ""))
    tt = s.t
    # the bolt heads and nuts are tied to rings of nodes, an idealization that sets the stress right there; those
    # spots sit under the drawn heads and are left without a value, as the strength view leaves its support spots
    tied = np.zeros(mesh.n_nodes, bool)
    for ln in lines or []:
        for end in (ln.head, ln.nut):
            tied |= np.linalg.norm(X - np.asarray(end, float), axis=1) <= 0.75 * ln.bolt.dw
    if tied.any() and not tied.all():
        vm[:, tied] = np.nan
        s.note("The stress right under each bolt head and nut is left out, where the ring the bolt is tied to "
               "sets it.")
    allow = min(resolve_material(m).allowable for m in {study.material, *getattr(study, "body_materials", {}).values()})
    part = s.mesh(X, mesh.tris, name="assembly", max_vertices=14000)
    part.color_by("von Mises", vm, unit="MPa", range=(0.0, allow), limit=allow, limit_label="yield")
    umax = float(np.abs(U).max())
    part.deform(U, exaggerate=float(f"{max(0.05 * span / umax, 1.0):.2g}") if umax > 0 else 1.0)
    run = SimpleNamespace(mesh=mesh, study=base, selections=selections)
    words = set((study.question or "").lower().split())
    try:
        supports(s, run, X, size, words)
        load_props(s, run, X, size, share, words)
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
        force = np.vstack([np.concatenate([[0.0], res.bolt_force[n]]) for n in res.bolt_force])[:, first:]
        s.readout("most loaded bolt", force.max(0) / 1000, unit="kN", digits=3,
                  limit=max(ln.bolt.proof * ln.bolt.stress_area for ln in lines) / 1000)
        s.readout("clamp through the contact", contact / 1000, unit="kN", digits=3)
        s.readout("joint opening", opening, unit="mm", digits=3)
    else:
        s.readout("contact force", contact, unit="N", digits=3)
    s.readout("peak stress", np.nanmax(vm, axis=1), unit="MPa", digits=3, limit=allow)
    if o.verdict in ("separates", "slips", "fails"):
        s.event(float(tt[-1]), {"separates": "joint opens", "slips": "parts slide", "fails": "overloaded"}[o.verdict],
                "", kind="fail")
    tone = {"holds": "pass", "marginal": "warn"}.get(o.verdict, "fail")
    s.verdict(o.verdict.upper(), o.headline.split(" Confidence")[0], tone)
    s.camera(target=(lo + hi) / 2, direction=P.unit([1.0, -1.4, 0.9]),
             distance=0.5 * span / np.sin(np.radians(15)) * 1.15, fov=30)
    return s.save(out)
